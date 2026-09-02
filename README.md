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
- Multi-turn conversations: per-session chat history, plus history-aware query condensation so follow-up questions ("what about its deductible?") retrieve correctly
- Retrieval-confidence gating: refuses to answer (instead of guessing) when nothing retrieved is similar enough to the question to trust
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
pip install fastapi uvicorn python-dotenv pypdf PyMuPDF qdrant-client langchain-text-splitters requests openai pytest
```
Also requires [Ollama](https://ollama.com) running locally with the `nomic-embed-text` model pulled (`ollama pull nomic-embed-text`), and a `GROQ_API_KEY` in `backend/.env`.

### Frontend
```bash
cd frontend
python -m venv venv
venv\Scripts\activate
pip install streamlit requests
```

## Learning Guide

[`docs/learning-guide.md`](docs/learning-guide.md) — the field of options behind each
build phase (chunking methods, embedding/vector-DB choices, evaluation metrics, etc.),
when to use each, and what "good" looks like where it's measurable. Distinct from
`docs/adr/`: the ADRs record *what this project chose and why*; the learning guide
teaches *the menu it chose from*. Currently covers Milestones 1–5, growing with
each milestone as it's built.

## Testing & Benchmarking

Quality-evaluation suite (Milestone 4), run against the live API:
```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload   # in one terminal
pytest tests/ -v -s         # in another
```
Retrieval hit rate/MRR/precision@k, structured-fact answer correctness, and
latency percentiles are all scored against `tests/golden_set.py` — a small,
hand-verified set of question -> known-correct-fact pairs (see that file's
docstring for what "golden set" means at this scale and why the numbers
aren't yet statistically trustworthy). Two cases are marked `xfail` for
known, understood bugs (see Milestone 3 known gaps below) rather than
skipped, so the suite still surfaces the moment either one gets fixed.

**What "good" looks like for each metric:**

| Metric | Ideal (good) | Not good |
|---|---|---|
| Hit Rate@k | ≥ 90% (small curated set) | < 70% |
| MRR | ≥ 0.8 (near 1.0 = correct chunk always ranks first) | < 0.5 |
| Precision@k | ~1.0 *only* on entity-scoped questions (ADR-0004); 0.2–0.5 is normal/expected on unscoped ones | a drop specifically on a scoped question |
| Answer correctness (structured facts) | 100% — no partial credit on a dollar figure or policy number | any miss at all |
| Latency p50 / p95 | aspirational UX target: < 3s / < 6s | this system's measured real floor: < 18s / < 28s (Groq round-trip dominates; see `test_latency_percentiles`) |

Every number above is either a general UX/IR convention (latency, hit rate,
MRR) or was pulled from this project's own measured runs (precision's
scoped/unscoped split, the latency floor) — not guessed and left unchecked.
See `backend/tests/test_quality.py` for where each one is scored, and the
top-k experiment (`backend/tests/topk_experiment.py` /
`backend/tests/topk_results.json`) for how precision and latency actually
move as `top_k` changes.

Indexing/embedding throughput benchmark (separate from query-time
performance): `python backend/benchmark_indexing.py` — **stop the backend
first** (Qdrant's on-disk mode locks its storage folder to one process at a
time) and see the script's docstring for what it measures and why.

## Project Status

_Last updated: 2026-08-27 (Milestone 5 in progress — multi-format support, chat history, history-aware query condensation, retrieval-confidence gating for hallucination detection)_

The roadmap below separates the **build phases** (the actual pipeline/system work, done in sequence) from **documentation** and **continuous improvement**, which aren't phases with an end state — they run alongside the build phases on an ongoing basis rather than being "reached" in turn.

### Build Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | Core Data Pipeline | ✅ Complete |
| 2 | Vector Search Foundation | ✅ Complete |
| 3 | RAG Chain Implementation | 🟡 Functional, basic |
| 4 | Quality & Evaluation | 🟡 In progress |
| 5 | Advanced Features | 🟡 In progress |
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

**Known gaps carried forward (found by Milestone 4's eval suite):**
- Entity-scoped filtering (`find_relevant_source_files`, `main.py`, ADR-0004) matches on insured-name substrings, which breaks when one insured has multiple policies — "Mount Royal University Commercial General Liability policy" wrongly scopes to a different Mount Royal policy (BW240599). See `backend/tests/golden_set.py`.
- The oversized-PDF document (`25-26 Group Accident Policy 100013386.pdf`, already noted as a metadata-extraction failure in Milestone 2) doesn't surface in top-5 semantic search for an on-topic question about its own content, and the LLM hallucinates a confident wrong attribution instead of stating it doesn't know. Worse than previously documented — tracked as `xfail` in the same test file.

#### Milestone 4: Quality & Evaluation — 🟡 In progress
- [x] Evaluation metrics (relevance, accuracy, latency) — `backend/tests/test_quality.py`: hit rate, MRR, precision@k, structured-fact answer correctness, latency percentiles
- [x] Automated test suite — `pytest backend/tests/`, replacing `perf_smoke_test.py`'s manual print-and-eyeball pattern with real assertions
- [ ] A/B testing framework — mechanism built (`backend/experiments.py`, `variant` field on `/query`), **not yet verified end-to-end** and not activated; see ADR-0007 for the open verification item and ADR-0006 for why real comparisons stay deferred. `experiments_log.jsonl` now records each result's full retrieval scores plus `session_id`/`low_confidence` (previously flattened away) so the comparison can actually be drawn from real traffic once it exists — see ADR-0007's 2026-09-02 update.
- [x] User feedback loop — `POST /feedback` (`backend/feedback_store.py`), durable JSONL log, verified working. Now also records `session_id`/`variant`/`low_confidence` per entry, and the frontend (`frontend/app.py`) has a 👍/👎 UI wired to it — previously there was no way for real feedback to reach this log at all.
- [x] Performance benchmarks — `backend/benchmark_indexing.py` (indexing/embedding throughput, separate from query-time latency)

Deliverable: quality dashboard with KPIs.

#### Milestone 5: Advanced Features — 🟡 In progress
- [x] Multi-format support — DOCX (`extract_text_from_docx`) and image OCR (`extract_text_from_image`, pytesseract) added to `insurance_loader.py`'s format-dispatch layer. Unsupported/unrecognized file formats are now logged as an error record instead of silently excluded from the file scan (§5.6). **Known gap:** the `tesseract` OCR binary isn't installed on this dev machine (pip only installs the `pytesseract` wrapper) — this fails loudly with an actionable error record rather than silently, and `tests/test_multi_format.py`'s real-OCR test `skipif`s until the binary is present. Install it before relying on OCR in any environment that needs it.
- [x] Chat history & context carryover — `session_id`-keyed `sessions_db` (`main.py`), an in-memory `deque(maxlen=5)` of raw (question, answer) pairs replayed as real messages ahead of each new turn. Answers can now reference prior turns; lost on backend restart, same trade-off as `documents_db`. The Streamlit frontend didn't actually send `session_id` back on follow-ups until 2026-09-02 (each question hit the API as a fresh session, so this never fired in practice) — it now persists across turns in `st.session_state`, with a "New conversation" reset.
- [x] Query rewriting (multi-hop questions) — `condense_query()` (`main.py`, [ADR-0008](docs/adr/0008-history-aware-query-condensation.md)): when a session has history, the latest question is rewritten into a standalone one (pronouns/implied references resolved) *before* retrieval, so a follow-up like "what about its deductible?" retrieves correctly. Skipped on a session's first turn (no history to condense against); falls back to the raw question on any LLM failure. Decomposition/HyDE/step-back deliberately left out — no observed failure case for them yet.
- [x] Hallucination detection — retrieval-confidence gating (`main.py`, [ADR-0009](docs/adr/0009-retrieval-confidence-gating.md)): if nothing retrieved clears `MIN_RETRIEVAL_SCORE`, `/query` skips the LLM and returns an explicit "not enough relevant information" response (`low_confidence: true`) instead of answering from weak context. **Live-verified 2026-09-02 — does not fix the oversized-PDF misattribution bug it targeted.** The wrong-document match in that case scores ~0.72, inside the same band ADR-0004 measured for genuinely-correct matches, so a flat similarity floor can't tell them apart without also rejecting legitimate answers. `test_hallucination_gate_group_accident` is now `xfail`, documenting this as a known gap (see ADR-0009 Consequences) rather than something more threshold-tuning will fix. The gate does still work for its other target: retrieval that comes back weak across the board.
- [ ] Confidence scoring

Deliverable: feature-rich RAG system.

#### Milestone 6: Production Deployment — ⬜ Not started
- [ ] API hardening (rate limiting, auth)
- [ ] Database persistence (PostgreSQL)
- [ ] Caching layer (Redis)
- [ ] Load testing & scaling
- [ ] Docker containerization

Deliverable: production-ready deployment.

#### Milestone 7: Monitoring & Ops — ⬜ Not started
- [ ] Logging (structured logs)
- [ ] Monitoring (Prometheus/Grafana)
- [ ] Alerting (failures, latency)
- [ ] Analytics (usage, costs)
- [ ] Auto-recovery

**Reminder carried forward from Milestone 5:** Milestone 5's cheap-path fallbacks (multi-format extraction failures, low-confidence retrieval, hallucination-gate triggers) are logged as structured records only (layer 1 of the graceful-degradation pattern — see `docs/learning-guide.md` §5.6). Still owed here:
- [ ] A queryable failure-rate metric/counter per fallback type (layer 2) — turns "it failed once" into "it fails N% of the time," the actual trigger for deciding whether to build the expensive path
- [ ] Real monitoring/alerting on those rates (layer 3), folded into this milestone's Prometheus/Grafana + Alerting items above

Deliverable: 24/7 monitoring dashboard.

#### Milestone 8: Security & Compliance — ⬜ Not started
- [ ] User authentication (OAuth2/JWT)
- [ ] Authorization (role-based access)
- [ ] Data encryption (in transit, at rest)
- [ ] Audit logging
- [ ] GDPR/compliance

Deliverable: security audit passed.

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
