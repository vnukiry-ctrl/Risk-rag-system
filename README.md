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

_Last updated: 2026-08-25 (Milestone 2 complete — see `DOCUMENTATION2.md` Step 10)_

The roadmap below separates the **build phases** (the actual pipeline/system work, done in sequence) from **documentation** and **continuous improvement**, which aren't phases with an end state — they run alongside the build phases on an ongoing basis rather than being "reached" in turn.

### Build Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | Core Data Pipeline | ✅ Complete |
| 2 | Vector Search Foundation | ✅ Complete |
| 3 | RAG Chain Implementation | 🟡 Functional, basic |
| 4 | Quality & Evaluation | ⬜ Not started |
| 5 | Advanced Features | ⬜ Not started |
| 6 | Production Deployment | ⬜ Not started |
| 7 | Monitoring & Ops | ⬜ Not started |
| 8 | Security & Compliance | ⬜ Not started |

#### Milestone 1: Core Data Pipeline — ✅ Complete
- [x] Document extraction (LLM-based metadata via Groq)
- [x] Text cleaning & normalization
- [x] Chunking strategy (size-based parent-child)
- [x] Error handling for problematic PDFs
- [x] Logging & monitoring

#### Milestone 2: Vector Search Foundation — ✅ Complete
- [x] Embeddings generation (Ollama, swappable)
- [x] Vector store integration (Qdrant, persistent on-disk)
- [x] Semantic search endpoint (verified end-to-end against real documents)
- [x] Top-K retrieval (fixed default; full tuning deferred until a production-scale golden eval set exists)
- [x] Performance testing (retrieval smoke test — 5/5 hit rate; see `backend/perf_smoke_test.py`)

**Known gaps carried forward:**
- One oversized PDF (`25-26 Group Accident Policy 100013386.pdf`) fails LLM metadata extraction (Groq per-request token limit) — its content is still indexed and searchable via `/query`, just without structured metadata.
- Extracted document metadata (`documents_db`) is in-memory only and must be rebuilt via `/extract` after each backend restart — the Qdrant vector index persists, but this dict doesn't. See `DOCUMENTATION2.md` Step 10 for details.
- Top-k is a fixed default (5), not tuned — deliberately deferred until a real production-scale golden Q&A eval set exists.

#### Milestone 3: RAG Chain Implementation — 🟡 Functional, basic
- [x] LLM integration (Groq, swappable)
- [x] Prompt engineering with structured + retrieved context
- [x] Source attribution
- [ ] Response quality tuning / evaluation

#### Milestones 4–8 — ⬜ Not started
Quality & evaluation, advanced features, production deployment, monitoring/ops, and security & compliance have not been started.

### Ongoing (not build phases)

These don't have a "done" milestone number — they run continuously alongside whatever build phase is active.

**Documentation & Training — 🟡 In progress**
- [x] Build log / decision history (`DOCUMENTATION.md`, `DOCUMENTATION2.md`)
- [x] API documentation (FastAPI auto-generated Swagger UI at `/docs`)
- [x] This README (setup, status, architecture)
- [ ] Standalone architecture diagram / design-decisions doc
- [ ] Deployment guide
- [ ] User manual / developer onboarding guide

**Continuous Improvement — ⬜ Not started**
Feedback analysis, fine-tuning, performance optimization, and cost reduction have no dedicated process yet — there isn't enough real usage to act on. Will pick up once Milestone 3+ produces real user feedback to iterate against.

## Author
Victoria Nukiry
