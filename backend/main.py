import logging
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from insurance_loader import load_insurance_documents
from vector_store import index_documents, semantic_search
from llm_client import llm_complete
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

documents_db = {}


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


@app.get("/")
async def root():
    return {"status": "alive", "service": "Insurance RAG System"}


@app.post("/extract")
async def extract_documents():
    logger.info("Extraction requested")
    try:
        results = load_insurance_documents()
    except Exception as e:
        logger.exception("Document loading failed")
        raise HTTPException(status_code=500, detail=f"Document loading failed: {str(e)}")

    successful = []
    failed = []
    for meta in results["metadata"]:
        doc_id = meta.get("policy_number") or meta.get("source_file", "unknown")
        documents_db[doc_id] = meta
        (failed if "error" in meta else successful).append(meta)

    # Indexing calls an external embedding service (Ollama) and can fail
    # independently of extraction; don't let that discard the extraction
    # work that already succeeded.
    chunks_indexed = 0
    indexing_error = None
    try:
        chunks_indexed = index_documents(results["parent_chunks"])
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

    logger.info("Query received: %r (top_k=%d)", request.question, request.top_k)

    try:
        chunks = semantic_search(request.question, top_k=request.top_k)

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

            context += "Relevant excerpts from insurance documents:\n\n"
            for c in chunks:
                context += f"=== {c['source_file']} (policy {c.get('policy_number') or 'N/A'}) ===\n"
                context += f"{c['text']}\n\n"
        else:
            context = "No relevant document excerpts found. Please run /extract first if you haven't."

        prompt = f"{context}\n\nUser Question: {request.question}\n\nProvide a clear, accurate answer using the summary and excerpts above. Prefer the structured policy summary for simple facts like limits, premiums, and dates. If the answer is not covered, say so."

        answer = llm_complete(
            messages=[
                {"role": "system", "content": "You are an expert insurance policy analyst. Answer questions about insurance documents accurately and helpfully. If information is not in the provided excerpts, clearly state that."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )

        sources = [{"source_file": c["source_file"], "policy_number": c.get("policy_number"), "score": c["score"]} for c in chunks]

        logger.info("Query answered using %d chunks", len(chunks))

        return {
            "question": request.question,
            "answer": answer,
            "sources": sources,
            "chunks_searched": len(chunks)
        }
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)