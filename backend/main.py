import logging
import os
import time
import uuid
from collections import deque
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from typing import Dict, List, Optional
from insurance_loader import load_insurance_documents
from vector_store import index_documents, semantic_search
from llm_client import llm_complete
from feedback_store import record_feedback
from documents_store import save_documents, load_documents
from experiments import get_variant, log_experiment_result
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Insurance RAG System",
    description="RAG system for insurance document analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

documents_db = load_documents()

# DECISION (UNIVERSAL, Milestone 5.2 -- history only, no query condensation
# yet): in-memory dict keyed by session_id, deque(maxlen=...) caps stored
# turns per session. Same durability tradeoff already made for documents_db
# -- lost on restart, which is fine for a conversation but wouldn't be for
# feedback_store.py's data. History is the raw (question, answer) text only,
# not the retrieved context, so replaying it into the prompt stays cheap
# regardless of how large top_k's chunks are.
MAX_HISTORY_TURNS = 5
sessions_db = {}

# DECISION (UNIVERSAL, Milestone 5.4, ADR-0009): 0.5 is an untuned starting
# default, not a measured value -- same honesty as MAX_HISTORY_TURNS and
# top_k above. ADR-0004 measured genuinely-topical-but-wrong-document
# matches clustering at 0.71-0.76 (cosine, Qdrant), so this floor sits below
# that range deliberately: it's meant to catch retrieval that found nothing
# even loosely on-topic (the oversized-PDF case), not to second-guess a
# borderline-but-real match. Recalibrate against real queries once a larger
# golden eval set exists (see ADR-0005's same deferred-tuning stance).
MIN_RETRIEVAL_SCORE = 0.5

# DECISION (UNIVERSAL, readiness for real data): 24000 chars (~6k tokens at
# the common ~4-chars/token English-text approximation -- no tokenizer
# dependency pulled in just for an approximate budget) is an untuned
# starting default, same honesty as the other constants above. Safe today
# only because top_k defaults small and chunks are capped in size, which is
# exactly why this exists ahead of need: a larger real document set, a
# higher top_k, or bigger parent chunks could otherwise grow the prompt
# past the model's context window (or just past what's useful) with no
# warning. Recalibrate once real documents reveal a typical chunk-size/
# top_k combination worth budgeting precisely for.
CONTEXT_CHAR_BUDGET = 24000


class QueryRequest(BaseModel):
    question: str
    # DECISION (UNIVERSAL): 5 is an untuned default carried over from Milestone
    # 2, not a measured value -- tune it against a golden eval set (known
    # question -> correct-answer pairs) for whatever documents the new
    # project actually has, rather than reusing this number.
    top_k: int = 5
    # DECISION (UNIVERSAL, mechanism built ahead of use -- see ADR-0006/0007):
    # None preserves today's exact behavior (top_k above, temperature=0
    # below). Naming a variant from experiments.VARIANTS overrides both for
    # this request only, so A/B comparisons are opt-in per call, not a
    # global toggle -- see experiments.py for why this isn't "live" yet.
    variant: Optional[str] = None
    # DECISION (UNIVERSAL, Milestone 5.2): omit on the first call of a
    # conversation -- the server mints one and returns it. Pass it back on
    # every follow-up to carry that conversation's history forward. An
    # unrecognized id (expired process restart, typo) is treated as the
    # start of a new session rather than an error -- see sessions_db above.
    session_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str  # "up" or "down" -- no third option; forcing a choice is what makes this actionable
    comment: Optional[str] = None
    sources: Optional[List[dict]] = None
    # DECISION (UNIVERSAL, readiness for real data): pass through the same
    # /query response fields verbatim so this feedback entry can be joined
    # back to the variant/confidence state that produced the answer -- see
    # feedback_store.record_feedback.
    session_id: Optional[str] = None
    variant: Optional[str] = None
    low_confidence: Optional[bool] = None


def find_relevant_source_files(question: str) -> Optional[List[str]]:
    """Scope retrieval to specific documents when the question names them.

    Matches the question text against known policy numbers, insured names,
    and insurers (substring, case-insensitive). Length-gated to avoid short
    values matching incidentally. Returns None -- meaning search unscoped --
    when nothing matches, so a generic question still searches everything.
    """
    q = question.lower()
    matches = set()
    insured_name_matches: Dict[str, List[dict]] = {}

    for doc in documents_db.values():
        if "error" in doc:
            continue
        source_file = doc.get("source_file")
        if not source_file:
            continue

        policy_number = doc.get("policy_number")
        if policy_number and len(policy_number) >= 4 and policy_number.lower() in q:
            matches.add(source_file)
            continue

        # DECISION (UNIVERSAL, fixes the Milestone 3 known gap / ADR-0004
        # follow-up -- see tests/golden_set.py's Mount Royal case): checked
        # as an independent, first-class signal (like policy_number/insurer
        # below), not just a tiebreaker nested under insured_name. Live data
        # showed why: AVP406486's extracted insured_name is a long formal
        # phrase ("User Group of the Board of Governors of Mount Royal
        # University as on file") that's never a substring of how a real
        # question names it, so it never entered the insured_name match path
        # at all -- only the wrong document (BW240599, insured_name "MOUNT
        # ROYAL UNIVERSITY") matched by name and won by default. A question
        # naming the specific insurance_type ("Commercial General Liability")
        # now scopes to the right document directly, regardless of whether
        # its insured_name phrasing matches.
        insurance_type = doc.get("insurance_type")
        if insurance_type and len(insurance_type) >= 6 and insurance_type.lower() in q:
            matches.add(source_file)
            continue

        insured_name = doc.get("insured_name")
        if insured_name and len(insured_name) >= 6 and insured_name.lower() in q:
            insured_name_matches.setdefault(insured_name, []).append(doc)
            continue

        insurance_company = doc.get("insurance_company")
        if insurance_company and len(insurance_company) >= 6 and insurance_company.lower() in q:
            matches.add(source_file)

    # DECISION (UNIVERSAL): a genuine insured_name collision -- two documents
    # sharing the identical extracted name string -- is a separate case from
    # the one above: matching on name alone can't tell them apart. When that
    # happens, narrow using insurance_type; if it still doesn't resolve to
    # exactly one document, leave these candidates out of scoping entirely
    # rather than guessing -- an unscoped search that finds the right chunks
    # on merit beats a confidently wrong scope (same principle as ADR-0009's
    # confidence gate).
    for insured_name, docs in insured_name_matches.items():
        if len(docs) == 1:
            matches.add(docs[0]["source_file"])
            continue
        narrowed = [
            doc for doc in docs
            if doc.get("insurance_type") and len(doc["insurance_type"]) >= 6
            and doc["insurance_type"].lower() in q
        ]
        if len(narrowed) == 1:
            matches.add(narrowed[0]["source_file"])

    return sorted(matches) if matches else None


def condense_query(question: str, history: deque) -> str:
    """Rewrite a follow-up question into a standalone one, using prior turns.

    DECISION (UNIVERSAL, Milestone 5.3, ADR-0008): only called when history is
    non-empty -- the first turn in a session has nothing to condense against,
    so retrieval uses the raw question directly and this LLM call never
    fires. That's a deliberate cost guard: query rewriting adds a full extra
    LLM round-trip before retrieval even starts, and most turns in a session
    are the first one or are already standalone.

    Falls back to the raw question on any failure -- same graceful-degradation
    discipline as the 5.1 format-dispatch layer (a broken rewrite should
    degrade to today's behavior, not break the query).
    """
    history_text = "\n".join(
        f"User: {turn['question']}\nAssistant: {turn['answer']}" for turn in history
    )

    try:
        rewritten = llm_complete(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the user's latest question into a standalone question that "
                        "makes sense without the conversation history, by resolving pronouns "
                        "and implied references (e.g. \"its deductible\" -> \"<policy name> "
                        "deductible\"). Preserve the original meaning and intent exactly -- do "
                        "not answer the question, add information, or change what is being "
                        "asked. If the question is already standalone, return it unchanged. "
                        "Reply with only the rewritten question, no preamble or quotes."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Conversation history:\n{history_text}\n\nLatest question: {question}",
                },
            ],
            temperature=0,
            max_tokens=100,
        )
        rewritten = (rewritten or "").strip().strip('"')
        return rewritten or question
    except Exception:
        logger.exception("Query condensation failed, falling back to raw question")
        return question


@app.get("/")
async def root():
    return {"status": "alive", "service": "Insurance RAG System"}


@app.post("/extract")
async def extract_documents():
    # DECISION (UNIVERSAL): load_insurance_documents() is fully synchronous,
    # blocking I/O (disk reads, LLM calls per document). Calling it directly
    # inside this async def would block FastAPI's single-threaded event loop
    # for the entire request -- not just this one, every other in-flight
    # request (including /health and /query) queues behind it with no
    # response until it returns. run_in_threadpool offloads it to a worker
    # thread so the event loop stays free to serve everything else while
    # this runs. Same reasoning applies to index_documents() below.
    logger.info("Extraction requested")
    try:
        results = await run_in_threadpool(load_insurance_documents)
    except Exception as e:
        logger.exception("Document loading failed")
        raise HTTPException(status_code=500, detail=f"Document loading failed: {str(e)}")

    successful = []
    failed = []
    for meta in results["metadata"]:
        doc_id = meta.get("policy_number") or meta.get("source_file", "unknown")
        documents_db[doc_id] = meta
        (failed if "error" in meta else successful).append(meta)
    save_documents(documents_db)

    # Indexing calls an external embedding service (Ollama) and can fail
    # independently of extraction; don't let that discard the extraction
    # work that already succeeded.
    chunks_indexed = 0
    indexing_error = None
    try:
        chunks_indexed = await run_in_threadpool(index_documents, results["parent_chunks"])
    except Exception as e:
        logger.exception("Indexing failed")
        indexing_error = str(e)

    logger.info(
        "Extraction complete: %d successful, %d failed, %d chunks indexed",
        len(successful), len(failed), chunks_indexed,
    )

    return {
        "total": len(successful) + len(failed),
        "successful": successful,
        "failed": failed,
        "chunks_indexed": chunks_indexed,
        "indexing_error": indexing_error,
    }


@app.get("/documents")
async def list_documents():
    return {"total": len(documents_db), "documents": list(documents_db.values())}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "documents_loaded": len(documents_db)}


@app.post("/search/metadata")
async def search_documents_metadata(query: str, top_k: int = 5):
    """Keyword substring match over extracted structured fields (policy #, insurer, etc.).

    Not semantic search -- for vector-based retrieval over document content, use /query.
    """
    if not query:
        raise HTTPException(status_code=400, detail="query required")

    q = query.lower().strip()
    matches = []

    for key, doc in documents_db.items():
        text_blob = " ".join([
            str(doc.get("policy_number") or ""),
            str(doc.get("insurance_company") or ""),
            str(doc.get("insurance_type") or ""),
            str(doc.get("insured_name") or ""),
            str(doc.get("broker") or ""),
            str(doc.get("coverholder") or ""),
            str(doc.get("notes") or ""),
            " ".join(doc.get("key_coverages") or []),
            " ".join(doc.get("exclusions") or []),
        ]).lower()

        if q in text_blob:
            matches.append({
                "policy_number": doc.get("policy_number"),
                "insurance_company": doc.get("insurance_company"),
                "source": key,
                "preview": text_blob[:200],
            })

    return {"query": query, "total": len(matches[:top_k]), "results": matches[:top_k]}


@app.post("/query")
async def query_documents(request: QueryRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="question required")

    variant_config = get_variant(request.variant)
    effective_top_k = variant_config["top_k"] if variant_config else request.top_k
    effective_temperature = variant_config["temperature"] if variant_config else 0

    session_id = request.session_id or str(uuid.uuid4())
    history = sessions_db.get(session_id, deque(maxlen=MAX_HISTORY_TURNS))

    logger.info(
        "Query received: %r (top_k=%d, variant=%s, session=%s, history_turns=%d)",
        request.question, effective_top_k, request.variant, session_id, len(history),
    )
    start_time = time.time()

    try:
        # DECISION (UNIVERSAL, Milestone 5.3, ADR-0008): retrieval runs against
        # search_query (condensed when history exists), not request.question,
        # so a follow-up like "what about its deductible?" resolves the
        # pronoun before embedding/entity-matching. The answer prompt below
        # still uses request.question -- the user's original phrasing -- so
        # the reply reads naturally rather than echoing the rewrite.
        search_query = request.question
        if history:
            search_query = await run_in_threadpool(condense_query, request.question, history)
            if search_query != request.question:
                logger.info("Condensed query: %r -> %r", request.question, search_query)

        source_files = find_relevant_source_files(search_query)
        if source_files:
            logger.info("Scoping search to %d matched document(s): %s", len(source_files), source_files)
        # Same run_in_threadpool reasoning as /extract above: semantic_search
        # (Qdrant + embedding call) and llm_complete (Groq call) below are
        # both blocking network I/O -- without offloading them, one slow
        # query freezes the whole server for every other concurrent request.
        chunks = await run_in_threadpool(
            semantic_search, search_query, top_k=effective_top_k, source_files=source_files
        )

        # DECISION (UNIVERSAL, Milestone 5.4, ADR-0009): retrieval-confidence
        # gating. An empty `chunks` list already produces an honest "no
        # excerpts found" message via the `else` branch below -- this covers
        # the other half of the oversized-PDF bug (Milestone 3 known gaps):
        # chunks DO come back, just from an unrelated policy, and the LLM
        # answered confidently anyway instead of refusing. If nothing
        # retrieved clears MIN_RETRIEVAL_SCORE, refuse before the LLM ever
        # sees the weak context -- cheaper and more reliable than asking the
        # model to police its own grounding on top of a bad retrieval.
        if chunks and max(c["score"] for c in chunks) < MIN_RETRIEVAL_SCORE:
            best_score = max(c["score"] for c in chunks)
            logger.warning(
                "Low-confidence retrieval for %r: best score %.3f < %.2f threshold, refusing to answer",
                request.question, best_score, MIN_RETRIEVAL_SCORE,
            )
            answer = (
                "I don't have enough relevant information in the indexed documents to "
                "answer this confidently. The closest matches found weren't similar enough "
                "to the question to be trustworthy -- try rephrasing, naming the policy "
                "directly, or confirm the right document has been extracted and indexed."
            )
            sources = [
                {"source_file": c["source_file"], "policy_number": c.get("policy_number"), "score": c["score"]}
                for c in chunks
            ]
            history.append({"question": request.question, "answer": answer})
            sessions_db[session_id] = history
            log_experiment_result(
                variant=request.variant,
                question=request.question,
                top_k=effective_top_k,
                temperature=effective_temperature,
                sources=sources,
                latency=time.time() - start_time,
                session_id=session_id,
                low_confidence=True,
            )
            return {
                "question": request.question,
                "answer": answer,
                "sources": sources,
                "chunks_searched": len(chunks),
                "session_id": session_id,
                "low_confidence": True,
            }

        if chunks:
            # Structured fields (policy #, limits, dates) are extracted once per whole
            # document and are more reliable for simple facts than a handful of raw
            # excerpts, which can surface an unrelated dollar figure (e.g. a deductible)
            # instead of the actual coverage limit. Give the LLM both.
            by_source = {
                doc.get("source_file"): doc
                for doc in documents_db.values()
                if "error" not in doc
            }
            summarized_sources = set()
            context = ""

            summary_lines = []
            for c in chunks:
                doc = by_source.get(c["source_file"])
                if doc and c["source_file"] not in summarized_sources:
                    summarized_sources.add(c["source_file"])
                    summary_lines.append(
                        f"- {c['source_file']}: policy {doc.get('policy_number', 'N/A')}, "
                        f"insurer {doc.get('insurance_company', 'N/A')}, "
                        f"type {doc.get('insurance_type', 'N/A')}, "
                        f"coverage limit {doc.get('coverage_limit', 'N/A')}, "
                        f"deductible {doc.get('deductible', 'N/A')}, "
                        f"premium {doc.get('premium_amount', 'N/A')}, "
                        f"period {doc.get('period_from', 'N/A')} to {doc.get('period_to', 'N/A')}"
                    )
            if summary_lines:
                context += "Structured policy summary (authoritative for these fields):\n"
                context += "\n".join(summary_lines) + "\n\n"

            # DECISION (UNIVERSAL, readiness for real data): chunks arrive
            # score-ordered (vector_store.semantic_search), so the budget is
            # spent on the best matches first -- lower-ranked chunks get
            # truncated or dropped before higher-ranked ones lose anything.
            # A chunk that doesn't fully fit is truncated to the remaining
            # budget rather than dropped whole, so partial context beats none.
            excerpt_pieces = []
            excerpt_chars = 0
            chunks_included = 0
            chunks_truncated = 0
            for c in chunks:
                header = f"=== {c['source_file']} (policy {c.get('policy_number') or 'N/A'}) ===\n"
                remaining = CONTEXT_CHAR_BUDGET - excerpt_chars
                if remaining <= len(header):
                    break
                text = c["text"]
                piece = header + text + "\n\n"
                if len(piece) > remaining:
                    allowed_text_len = remaining - len(header) - len("\n[...truncated...]\n\n")
                    if allowed_text_len <= 0:
                        break
                    piece = header + text[:allowed_text_len] + "\n[...truncated...]\n\n"
                    chunks_truncated += 1
                excerpt_pieces.append(piece)
                excerpt_chars += len(piece)
                chunks_included += 1

            if chunks_included < len(chunks) or chunks_truncated:
                logger.warning(
                    "Context budget applied: included %d/%d chunks (%d truncated) within %d-char budget",
                    chunks_included, len(chunks), chunks_truncated, CONTEXT_CHAR_BUDGET,
                )

            context += "Relevant excerpts from insurance documents:\n\n"
            context += "".join(excerpt_pieces)
        else:
            context = "No relevant document excerpts found. Please run /extract first if you haven't."

        prompt = f"{context}\n\nUser Question: {request.question}\n\nProvide a clear, accurate answer using the summary and excerpts above. Prefer the structured policy summary for simple facts like limits, premiums, and dates. If the answer is not covered, say so."

        # DECISION (UNIVERSAL, Milestone 5.2): prior turns are replayed as
        # real user/assistant messages, not flattened into the prompt string
        # -- this is "full history in prompt" from the 5.2 options table,
        # scoped to raw Q&A text only (no re-attached context per turn).
        # It does NOT fix retrieval for follow-ups like "what about its
        # deductible" -- semantic_search above only ever sees the current
        # question. That's query condensation (5.3), deliberately deferred.
        history_messages = []
        for turn in history:
            history_messages.append({"role": "user", "content": turn["question"]})
            history_messages.append({"role": "assistant", "content": turn["answer"]})

        answer = await run_in_threadpool(
            llm_complete,
            messages=[
                # DECISION (DOMAIN-SPECIFIC): system prompt's persona and
                # grounding instruction are insurance wording -- rewrite for
                # a new domain, but keep the shape: role + "state clearly
                # when info isn't in the excerpts" is what curbs hallucination.
                {"role": "system", "content": "You are an expert insurance policy analyst. Answer questions about insurance documents accurately and helpfully. If information is not in the provided excerpts, clearly state that."},
                *history_messages,
                {"role": "user", "content": prompt}
            ],
            # DECISION (UNIVERSAL, but the value is domain-dependent): 0 was
            # chosen because this task is literal fact retrieval (policy
            # numbers, limits, dates) where answer consistency matters more
            # than variety. A generation/brainstorming task would want higher.
            # Overridable per-request via `variant` (experiments.py) -- 0
            # remains the default the moment no variant is named.
            temperature=effective_temperature,
            # DECISION (UNIVERSAL): 500 is untested against this project's
            # longest realistic answer (e.g. a multi-policy comparison) --
            # verify against real usage rather than assuming it's enough.
            max_tokens=500,
        )

        sources = [{"source_file": c["source_file"], "policy_number": c.get("policy_number"), "score": c["score"]} for c in chunks]

        history.append({"question": request.question, "answer": answer})
        sessions_db[session_id] = history

        logger.info("Query answered using %d chunks", len(chunks))

        log_experiment_result(
            variant=request.variant,
            question=request.question,
            top_k=effective_top_k,
            temperature=effective_temperature,
            sources=sources,
            latency=time.time() - start_time,
            session_id=session_id,
            low_confidence=False,
        )

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "chunks_searched": len(chunks),
            "session_id": session_id,
            "low_confidence": False,
        }
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Record a thumbs up/down (plus optional comment) on a previously
    returned answer -- see feedback_store.py for why this is a durable file,
    not in-memory state like documents_db."""
    if request.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    entry = record_feedback(
        question=request.question,
        answer=request.answer,
        rating=request.rating,
        comment=request.comment,
        sources=request.sources,
        session_id=request.session_id,
        variant=request.variant,
        low_confidence=request.low_confidence,
    )
    return {"status": "recorded", "entry": entry}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)