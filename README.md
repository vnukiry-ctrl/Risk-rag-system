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
pip install -r requirements.txt
```
Also requires [Ollama](https://ollama.com) running locally with the `nomic-embed-text` model pulled (`ollama pull nomic-embed-text`), [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on PATH (image-format documents only — `pip install pytesseract` installs only the Python wrapper, never the binary itself), and a `GROQ_API_KEY` plus `API_KEYS` in `backend/.env` (see `backend/.env.example`; see [ADR-0010](docs/adr/0010-api-key-auth-and-rate-limiting.md) for what `API_KEYS` gates).

### Frontend
```bash
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Needs its own `API_KEY` in `frontend/.env` (see `frontend/.env.example`), matching one of the backend's `API_KEYS` entries.

### Docker (optional, Milestone 6.3)
```bash
echo {} > backend/documents_db.json   # first run only -- see docker-compose.yml's comment for why
docker compose up --build
```
Runs backend (`:8000`) and frontend (`:8501`) as containers; Qdrant stays embedded inside the backend process (its own deliberate design, not containerized separately) and Ollama stays a host-level dependency (`OLLAMA_BASE_URL` in `docker-compose.yml` points at `host.docker.internal`). `backend/data/`, `backend/qdrant_data/`, and the runtime JSON/JSONL files are bind-mounted so indexed documents and logs persist across container restarts. **Written but not build-verified** — this machine doesn't have Docker installed, so the Dockerfiles/compose config haven't been run end-to-end yet; if a build fails, that's the first thing to check.

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
latency percentiles are all scored against `tests/golden_set.py` — 14 cases
(12 structured-fact + 2 known-limitation) hand-verified directly against
the 16 real Mount Royal University insurance documents in `backend/data/`
(rebuilt 2026-09-08/09-11, replacing an earlier version tested against 9
placeholder documents — see that file's docstring). Two real, understood
gaps stay in the set as `known_limitation` cases rather than being dropped:
a `documents_db` key-collision bug (two document pairs share a policy
number; whichever loads later in `os.listdir()` order silently overwrites
the other's structured metadata) and one document whose real declarations
sit behind enough boilerplate that LLM metadata extraction never reaches
them (though retrieval and hallucination-avoidance both still work
correctly for it — see the case notes in `golden_set.py`).

**What "good" looks like for each metric:**

| Metric | Ideal (good) | Not good |
|---|---|---|
| Hit Rate@k | ≥ 90% (small curated set) | < 70% |
| MRR | ≥ 0.8 general convention; **this suite's measured floor is 0.75** — measured result has landed at 0.79 consistently (hit rate 83%, 10/12), with *which* 1-2 cases miss the top-k window varying between runs (ordinary embedding-ranking noise, not a persistent per-document bug — see `test_mean_reciprocal_rank`) | < 0.5 |
| Precision@k | ~1.0 *only* on entity-scoped questions (ADR-0004, matches on policy number/insured name/insurer/insurance_type); measured 0.72-0.79 across `top_k` 2-5 since most real questions name one of those four things; 0.2–0.5 is normal/expected on the few that don't | a broad drop across many cases |
| Answer correctness (structured facts) | 100% — no partial credit on a dollar figure or policy number | any miss at all |
| Latency p50 / p95 | aspirational UX target: < 3s / < 6s; measured floor at the tuned `top_k=2` default: < 15s / < 20s (provisional — see `test_latency_percentiles`) | this system's *previous* measured floor at the old `top_k=5` default was < 18s / < 28s (Groq round-trip dominates) |

Every number above is either a general UX/IR convention (latency, hit rate,
MRR) or was pulled from this project's own measured runs (precision's
scoped/unscoped split, the latency floor) — not guessed and left unchecked.
See `backend/tests/test_quality.py` for where each one is scored, and the
top-k experiment (`backend/tests/topk_experiment.py` /
`backend/tests/topk_results.json`, [ADR-0011](docs/adr/0011-topk-tuned-from-real-data.md))
for how hit rate/MRR/precision/latency actually move as `top_k` changes —
that sweep is what picked the current `top_k=2` default.

Indexing/embedding throughput benchmark (separate from query-time
performance): `python backend/benchmark_indexing.py` — **stop the backend
first** (Qdrant's on-disk mode locks its storage folder to one process at a
time) and see the script's docstring for what it measures and why.

## Project Status

_Last updated: 2026-09-11 (real documents loaded — 16 Mount Royal University insurance policies replace the placeholder set; `golden_set.py` rebuilt from scratch and hand-verified against them; top-k tuned from k=5 to k=2 with real measurements, closing Milestone 3's top-k half — see ADR-0011; Milestone 6 started — API key auth and per-endpoint rate limiting)_

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
- [x] Top-K retrieval — **tuned 2026-09-11** from the placeholder default (5) to a measured value (2), via `topk_experiment.py`'s sweep against the real document set; see [ADR-0011](docs/adr/0011-topk-tuned-from-real-data.md)
- [x] Performance testing (retrieval smoke test — 5/5 hit rate; see `backend/perf_smoke_test.py`)

**Known gaps carried forward:**
- Extracted document metadata (`documents_db`) is in-memory only and must be rebuilt via `/extract` after each backend restart — the Qdrant vector index persists, but this dict doesn't. See `DOCUMENTATION2.md` Step 10 for details.
- `documents_db`'s dict-keyed-by-policy_number design silently drops one document's metadata whenever two real files share a policy number (a policy plus its own adjustment endorsement or invoice, in the real data) — see `golden_set.py`'s `known_limitation` cases for the two instances this actually hits.
- One real document (`26-27 Excess Side A D&O Policy 01-142-91-44 - Insured Copy.pdf`) has its actual declarations buried behind ~16 pages of insurer privacy-policy boilerplate, which LLM metadata extraction never reads past — every structured field comes back empty for it. Retrieval and answer-grounding both still work correctly despite this (verified live), so it's a structured-metadata gap only, not a Q&A-quality one.

#### Milestone 3: RAG Chain Implementation — 🟡 Functional, basic
- [x] LLM integration (Groq, swappable)
- [x] Prompt engineering with structured + retrieved context
- [x] Source attribution
- [x] Response quality tuning / evaluation — **top-k half closed 2026-09-11** ([ADR-0011](docs/adr/0011-topk-tuned-from-real-data.md)): measured against the real 16-document golden set, not guessed. Context-budget tuning (`CONTEXT_CHAR_BUDGET`, ADR-0005) and the confidence-gate threshold (`MIN_RETRIEVAL_SCORE`, ADR-0009) are still open — lower-risk now that less text gets retrieved per query by default, but neither was itself re-measured by this pass.

**Known gaps carried forward (found by Milestone 4's eval suite):**
- Entity-scoped filtering (`find_relevant_source_files`, `main.py`, ADR-0004) matches on policy number, insured name, insurer, or insurance_type substrings — most real questions name one of those four things and scope correctly (measured precision 0.72-0.79), but a question that only describes a policy some other way still searches unscoped, same as any RAG query without metadata filtering.
- Retrieval occasionally misses the expected document within the tuned `top_k=2` window — measured hit rate 10/12 (83%) consistently, but *which* 1-2 of the 12 structured-fact cases miss varies run to run (seen so far: the Property/Umbrella pair, and separately the Crime/Medical-Malpractice pair), consistent with ordinary embedding-ranking noise at a tight window rather than one specific document having a persistent problem. Re-sweep `top_k` (ADR-0011) if this pattern starts consistently naming the same document.
- The old Milestone 2/3 known gap about an oversized placeholder PDF (`25-26 Group Accident Policy 100013386.pdf`) hallucinating a wrong attribution no longer applies — that document isn't part of the real data set. The real set's closest analog (the Excess Side A D&O extraction gap, Milestone 2 known gaps above) was live-verified 2026-09-08 to **not** reproduce that failure: it retrieves correctly and the LLM declines rather than guessing.

#### Milestone 4: Quality & Evaluation — 🟡 In progress
- [x] Evaluation metrics (relevance, accuracy, latency) — `backend/tests/test_quality.py`: hit rate, MRR, precision@k, structured-fact answer correctness, latency percentiles
- [x] Automated test suite — `pytest backend/tests/`, replacing `perf_smoke_test.py`'s manual print-and-eyeball pattern with real assertions
- [ ] A/B testing framework — mechanism built (`backend/experiments.py`, `variant` field on `/query`), **not yet verified end-to-end** and not activated; see ADR-0007 for the open verification item and ADR-0006 for why real comparisons stay deferred. `experiments_log.jsonl` now records each result's full retrieval scores plus `session_id`/`low_confidence` (previously flattened away) so the comparison can actually be drawn from real traffic once it exists — see ADR-0007's 2026-09-02 update.
- [x] User feedback loop — `POST /feedback` (`backend/feedback_store.py`), durable JSONL log, verified working. Now also records `session_id`/`variant`/`low_confidence` per entry, and the frontend (`frontend/app.py`) has a 👍/👎 UI wired to it — previously there was no way for real feedback to reach this log at all.
- [x] Performance benchmarks — `backend/benchmark_indexing.py` (indexing/embedding throughput, separate from query-time latency)

Deliverable: quality dashboard with KPIs.

#### Milestone 5: Advanced Features — 🟡 In progress
- [x] Multi-format support — DOCX (`extract_text_from_docx`) and image OCR (`extract_text_from_image`, pytesseract) added to `insurance_loader.py`'s format-dispatch layer. Unsupported/unrecognized file formats are now logged as an error record instead of silently excluded from the file scan (§5.6). ~~Known gap: the `tesseract` OCR binary isn't installed~~ — **installed 2026-09-02** (UB-Mannheim build, via winget) and added to PATH. `tests/test_multi_format.py`'s real-OCR test now runs instead of `skipif`-ing, and passes (a font-size bug in the test's own synthetic image was fixed along the way — unrelated to the install itself). Any environment this deploys to will still need its own install of the binary; `pip install pytesseract` only ever installs the wrapper.
- [x] Chat history & context carryover — `session_id`-keyed `sessions_db` (`main.py`), an in-memory `deque(maxlen=5)` of raw (question, answer) pairs replayed as real messages ahead of each new turn. Answers can now reference prior turns; lost on backend restart, same trade-off as `documents_db`. The Streamlit frontend didn't actually send `session_id` back on follow-ups until 2026-09-02 (each question hit the API as a fresh session, so this never fired in practice) — it now persists across turns in `st.session_state`, with a "New conversation" reset.
- [x] Query rewriting (multi-hop questions) — `condense_query()` (`main.py`, [ADR-0008](docs/adr/0008-history-aware-query-condensation.md)): when a session has history, the latest question is rewritten into a standalone one (pronouns/implied references resolved) *before* retrieval, so a follow-up like "what about its deductible?" retrieves correctly. Skipped on a session's first turn (no history to condense against); falls back to the raw question on any LLM failure. Decomposition/HyDE/step-back deliberately left out — no observed failure case for them yet.
- [x] Hallucination detection — retrieval-confidence gating (`main.py`, [ADR-0009](docs/adr/0009-retrieval-confidence-gating.md)): if nothing retrieved clears `MIN_RETRIEVAL_SCORE`, `/query` skips the LLM and returns an explicit "not enough relevant information" response (`low_confidence: true`) instead of answering from weak context. **Live-verified 2026-09-02 — does not fix the oversized-PDF misattribution bug it targeted.** The wrong-document match in that case scores ~0.72, inside the same band ADR-0004 measured for genuinely-correct matches, so a flat similarity floor can't tell them apart without also rejecting legitimate answers. `test_hallucination_gate_group_accident` is now `xfail`, documenting this as a known gap (see ADR-0009 Consequences) rather than something more threshold-tuning will fix. The gate does still work for its other target: retrieval that comes back weak across the board.
- [ ] Confidence scoring
- [x] Context token budget — `CONTEXT_CHAR_BUDGET` (`main.py`, readiness for real data, 2026-09-02): retrieved excerpts were previously concatenated into the prompt with no limit, safe only because `top_k` and chunk size were both small. Excerpts (already score-ordered) now fill a character budget best-match-first, truncating or dropping lower-ranked chunks before the model's context window or usefulness is at risk. A char-count approximation (~4 chars/token), not a real tokenizer — untuned starting default, same as `top_k`/`MIN_RETRIEVAL_SCORE`.

Deliverable: feature-rich RAG system.

#### Milestone 6: Production Deployment — 🟡 In progress
- [x] API hardening (rate limiting, auth) — shared-secret `X-API-Key` header (`API_KEYS` env var, comma-separated) gates every endpoint except `/` and `/health`; unset disables auth with a startup warning, so local dev without a `.env` still works but nothing non-local can silently ship unauthenticated. Per-key rate limiting via `slowapi` (`backend/main.py`), keyed by the API key (falls back to remote address when auth is disabled): 20/min on `/query`, 5/min on `/extract`, 30/min on `/feedback` and `/search/metadata`, 100/min default elsewhere. Limits are untuned starting defaults, same honesty as `top_k`/`MIN_RETRIEVAL_SCORE`. The Streamlit frontend (`frontend/app.py`) sends the same key via `API_KEY` in its own `.env`. Not yet done: TLS termination and locking down CORS (`allow_origins=["*"]`) — left for actual deployment, since there's no real origin to restrict to yet.
- [ ] Database persistence (PostgreSQL)
- [ ] Caching layer (Redis)
- [ ] Load testing & scaling
- [x] Docker containerization — `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`. Backend + frontend are containerized; Qdrant runs embedded inside the backend process rather than as its own service (`vector_store.py`'s existing design, not changed for this), and Ollama stays a host-level dependency reached via `host.docker.internal`, matching the project's existing local-dev posture rather than adding a new service. `backend/requirements.txt` was populated for the first time as part of this (previously empty — the README's manual `pip install` line, now replaced by `pip install -r requirements.txt`, had drifted from what's actually needed); `frontend/requirements.txt` was corrected too — it was missing `pandas` entirely (imported by `app.py`, would have failed a fresh install) and pinned to versions well behind what's actually installed and working. **Not build-verified** — this dev machine doesn't have Docker installed, so nothing here has actually been built and run yet.

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
