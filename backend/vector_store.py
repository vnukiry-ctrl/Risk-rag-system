import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchAny
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# EMBEDDINGS PROVIDER - EASY SWAP HERE
# ============================================================
# Current: Ollama (Free, Open Source), called directly via its batched
# /api/embed endpoint. LangChain's OllamaEmbeddings.embed_documents()
# sends one HTTP request per text (~2s each); batching cuts that ~15x.

# DECISION (UNIVERSAL, but re-evaluate every time): embedding model choice
# is the single biggest lever on retrieval quality in this whole pipeline --
# bigger than chunk size or top_k. nomic-embed-text is general-purpose and
# has no notion of domain concepts (e.g. "coverage limit" vs "premium"); a
# domain-tuned or larger model is the first thing to try if retrieval
# quality is the bottleneck on a new project, before touching chunk sizes.
EMBEDDINGS_PROVIDER = "ollama"  # Options: "ollama" or "anthropic"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
VECTOR_SIZE = 768  # nomic-embed-text output dimension


class OllamaEmbedder:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": OLLAMA_EMBED_MODEL, "input": texts},
                timeout=120,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not reach Ollama at {OLLAMA_BASE_URL} — is it running? "
                f"(`ollama serve`, and `ollama pull {OLLAMA_EMBED_MODEL}` if not already pulled)"
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama at {OLLAMA_BASE_URL} timed out embedding {len(texts)} chunks")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama embedding request failed: {e}")
        return resp.json()["embeddings"]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


if EMBEDDINGS_PROVIDER == "ollama":
    embeddings = OllamaEmbedder()

elif EMBEDDINGS_PROVIDER == "anthropic":
    # TODO: TO USE ANTHROPIC INSTEAD:
    # 1. pip install langchain-anthropic
    # 2. Set ANTHROPIC_API_KEY in .env
    # 3. Change EMBEDDINGS_PROVIDER = "anthropic"
    from langchain_anthropic import AnthropicEmbeddings
    embeddings = AnthropicEmbeddings(model="claude-3-5-sonnet-20241022")
    VECTOR_SIZE = 1024

COLLECTION_NAME = "insurance_documents"

_client = None


QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")


def get_client() -> QdrantClient:
    """Return the process-wide Qdrant client, creating it on first use.

    Uses on-disk local mode so the index survives restarts. For deployment,
    swap this for QdrantClient(url=...) pointing at a real Qdrant server —
    index_documents()/semantic_search() don't need to change.
    """
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
    return _client


def setup_vector_store():
    """Kept for backwards compatibility with earlier Step 5 script."""
    return get_client(), embeddings


# DECISION (UNIVERSAL): parent/child sizes are tuned to how big a coherent
# unit of meaning is in *this* document type (a policy clause), not a fixed
# rule. Re-derive for a new project from its own documents -- code comments,
# support tickets, and contracts all have different natural unit sizes.
def index_documents(
    parent_chunks: List[Dict],
    parent_chunk_size: int = 2000,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 50,
) -> int:
    """Split each document into parent/child chunks and (re)build the Qdrant collection.

    Parent-child (small-to-big) retrieval: small child chunks are embedded and
    searched, since small chunks give more precise similarity matches; each
    child's payload carries its parent chunk's full text, which is what gets
    returned to the caller. That way a precise match still comes back with
    the surrounding passage instead of an isolated fragment. Splitting is
    purely size-based (no reliance on document structure/headers), since
    real-world insurance documents don't format section markup consistently.

    parent_chunks: list of {"content": full_text, "metadata": {...}, "source_file": str}
    as produced by insurance_loader.load_insurance_documents()["parent_chunks"].
    Returns the number of child chunks indexed.
    """
    client = get_client()
    # DECISION (UNIVERSAL): cosine is the standard default for normalized
    # text embeddings and is rarely worth revisiting -- leave as-is.
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    texts = []
    payloads = []
    for doc in parent_chunks:
        content = doc.get("content") or ""
        if not content.strip():
            continue
        meta = doc.get("metadata") or {}
        source_file = doc.get("source_file", meta.get("source_file", "unknown"))

        for parent_text in parent_splitter.split_text(content):
            for child_text in child_splitter.split_text(parent_text):
                texts.append(child_text)
                payloads.append({
                    "text": child_text,
                    "parent_text": parent_text,
                    "source_file": source_file,
                    "policy_number": meta.get("policy_number"),
                    "insurance_company": meta.get("insurance_company"),
                    "insurance_type": meta.get("insurance_type"),
                })

    if not texts:
        logger.warning("No content to index across %d documents", len(parent_chunks))
        return 0

    logger.info("Embedding %d child chunks from %d documents", len(texts), len(parent_chunks))

    batch_size = 50
    points = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        batch_vectors = embeddings.embed_documents(batch_texts)
        for offset, vector in enumerate(batch_vectors):
            idx = start + offset
            points.append(PointStruct(id=idx, vector=vector, payload=payloads[idx]))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Indexed %d chunks into Qdrant collection %r", len(points), COLLECTION_NAME)
    return len(points)


def semantic_search(query: str, top_k: int = 2, source_files: List[str] = None) -> List[Dict]:
    """Embed the query, match against child chunks, and return each match's parent text.

    Matching happens on the small child chunks (precise embeddings), but the
    text returned is the larger parent chunk each match belongs to, so callers
    get the surrounding passage rather than an isolated fragment.

    source_files, when given, restricts the search to those documents --
    child chunks embed generic insurance vocabulary ("coverage limit",
    "deductible"), so an unscoped search for a question naming one policy
    can still pull in similarly-worded chunks from unrelated policies.
    """
    client = get_client()
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        return []

    query_vector = embeddings.embed_query(query)
    query_filter = None
    if source_files:
        query_filter = Filter(
            must=[FieldCondition(key="source_file", match=MatchAny(any=source_files))]
        )
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
    ).points

    # Multiple child chunks can belong to the same parent; Qdrant returns
    # results ordered by score, so keeping the first occurrence per parent
    # dedupes without losing the best-scoring match for that passage.
    seen_parents = set()
    matches = []
    for r in results:
        parent_text = r.payload.get("parent_text")
        if parent_text in seen_parents:
            continue
        seen_parents.add(parent_text)
        matches.append({
            "text": parent_text,
            "source_file": r.payload.get("source_file"),
            "policy_number": r.payload.get("policy_number"),
            "insurance_company": r.payload.get("insurance_company"),
            "score": r.score,
        })

    return matches


if __name__ == "__main__":
    print("Setting up vector store...")
    client, _ = setup_vector_store()
    print("Vector store initialized successfully!")
    print(f"Embeddings provider: {EMBEDDINGS_PROVIDER}")
