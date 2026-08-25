# Risk-RAG-System

A Retrieval Augmented Generation (RAG) system for insurance document analysis, built with:
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector Database**: Qdrant
- **Embeddings**: Ollama (`nomic-embed-text`)
- **LLM**: Groq (`openai/gpt-oss-120b`)

## Project Overview

Ingests insurance policy PDFs, extracts structured metadata (policy number, insurer, coverage limits, dates, etc.) via LLM, indexes them for semantic search, and answers natural-language questions with source-attributed responses.

## Features

- LLM-based metadata extraction from insurance PDFs (Groq)
- Text normalization to strip PDF layout noise before extraction/indexing
- Parent-child (size-based) chunking for retrieval: small chunks embedded for precision, larger parent passages returned for context
- Semantic search over indexed documents (Qdrant + Ollama embeddings)
- LLM-based Q&A with source attribution, blending structured metadata with retrieved excerpts
- Structured logging and per-document error handling/reporting
- Swappable LLM/embeddings providers (Groq ↔ Anthropic, Ollama ↔ Anthropic) via one-line config changes

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend API | FastAPI |
| Frontend UI | Streamlit |
| Document Processing | LangChain text splitters |
| Vector Storage | Qdrant (persistent, on-disk local mode) |
| Embeddings | Ollama (`nomic-embed-text`), swappable to Anthropic |
| LLM | Groq (`openai/gpt-oss-120b`), swappable to Anthropic |

## Setup

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install fastapi uvicorn python-dotenv pypdf PyMuPDF qdrant-client langchain-text-splitters requests openai
```
Also requires [Ollama](https://ollama.com) running locally with the `nomic-embed-text` model pulled (`ollama pull nomic-embed-text`), and a `GROQ_API_KEY` in `backend/.env`.

### Frontend
```bash
cd frontend
python -m venv venv
venv\Scripts\activate
pip install streamlit requests
```

## Project Status

### Milestone 1: Core Data Pipeline — ✅ Complete
- [x] Document extraction (LLM-based metadata via Groq)
- [x] Text cleaning & normalization
- [x] Chunking strategy (size-based parent-child)
- [x] Error handling for problematic PDFs
- [x] Logging & monitoring

### Milestone 2: Vector Search Foundation — ✅ Complete
- [x] Embeddings generation (Ollama, swappable)
- [x] Vector store integration (Qdrant, persistent on-disk)
- [x] Semantic search endpoint (verified end-to-end against real documents)
- [x] Top-K retrieval (fixed default; full tuning deferred until a production-scale golden eval set exists)
- [x] Performance testing (retrieval smoke test — 5/5 hit rate; see `backend/perf_smoke_test.py`)

Known gaps carried forward: one oversized PDF fails LLM metadata extraction (Groq per-request token limit); extracted document metadata (`documents_db`) is in-memory only and must be rebuilt via `/extract` after each backend restart — see `DOCUMENTATION2.md` Step 10 for details.

### Milestone 3: RAG Chain Implementation — functional, basic
- [x] LLM integration (Groq, swappable)
- [x] Prompt engineering with structured + retrieved context
- [x] Source attribution
- [ ] Response quality tuning / evaluation

### Milestones 4–10 — not started
Quality & evaluation, advanced features, production deployment, monitoring/ops, security & compliance, documentation, and continuous improvement have not been started.

## Author
Victoria Nukiry
