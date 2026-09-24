import logging
import os
import re
import time
import uuid
from collections import deque
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.concurrency import run_in_threadpool
from typing import Dict, List, Optional, Tuple
from insurance_loader import load_insurance_documents
from vector_store import index_documents, semantic_search, count_chunks_by_source
from llm_client import llm_complete
from feedback_store import record_feedback
from documents_store import save_documents, load_documents, PROFESSIONAL_STORE
from db import init_db
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

# DECISION (UNIVERSAL, Milestone 6.1): shared-secret API key(s) via env var,
# not per-user accounts -- this is a single-deployment internal tool, not a
# multi-tenant product, so a lightweight gate is proportionate to the risk.
# Comma-separate API_KEYS to hand out more than one (e.g. frontend + a
# script) without them sharing a secret or needing a rotation that breaks
# the other caller. An unset API_KEYS is treated as "auth disabled" rather
# than refused at startup, so local dev without a .env still works -- but
# that means any non-local deployment MUST set it, which the warning below
# exists to make impossible to miss.
API_KEYS = {k.strip() for k in os.getenv("API_KEYS", "").split(",") if k.strip()}
if not API_KEYS:
    logger.warning(
        "No API_KEYS configured -- all endpoints are running WITHOUT authentication. "
        "Set API_KEYS (comma-separated) in .env before deploying anywhere but localhost."
    )

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(_api_key_header)) -> None:
    if not API_KEYS:
        return
    if key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# DECISION (UNIVERSAL, Milestone 6.1): rate limits are keyed by API key when
# one is present, falling back to remote address only for the unauthenticated
# (no API_KEYS configured) case -- so quota is per caller, not per NAT/proxy
# IP shared by many callers behind it. Limits below are untuned starting
# defaults (same honesty as top_k/MIN_RETRIEVAL_SCORE elsewhere in this
# file): /query and /extract both fan out to external services (Groq, the
# embedding backend) and are the ones actually worth protecting; the default
# covers everything else so no route is ever fully unbounded.
def _rate_limit_key(request: Request) -> str:
    return request.headers.get("X-API-Key") or get_remote_address(request)


limiter = Limiter(key_func=_rate_limit_key, default_limits=["100/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

init_db()
documents_db = load_documents()

# Classify-then-target pipeline results, kept in a separate store from
# documents_db above so the two extraction strategies can sit side by side
# for comparison instead of one overwriting the other.
documents_db_professional = load_documents(store=PROFESSIONAL_STORE)

# DECISION (UNIVERSAL, Milestone 5.2 -- history only, no query condensation
# yet): in-memory dict keyed by session_id, deque(maxlen=...) caps stored
# turns per session. Same durability tradeoff already made for documents_db
# -- lost on restart, which is fine for a conversation but wouldn't be for
# feedback_store.py's data. History is the raw (question, answer) text only,
# not the retrieved context, so replaying it into the prompt stays cheap
# regardless of how large top_k's chunks are.
MAX_HISTORY_TURNS = 5
sessions_db = {}

# DECISION (UNIVERSAL, Milestone 5.4/3, ADR-0009 update 2026-09-14): measured
# against real queries, not guessed -- 95 real golden-set questions (logged
# in experiments_log.jsonl) never scored below 0.686, while five deliberately
# off-topic probes ("what's the capital of France?", "how do I bake a
# chocolate cake?", a laptop-warranty question) scored 0.448-0.626, all
# comfortably below that real floor. 0.6 sits with real margin (0.086) under
# the measured legitimate minimum while catching those clearly-unrelated
# cases the old 0.5 let straight through (capital-of-France scored 0.563 --
# above the old floor). NOT fixed by this or any score-only floor: two
# insurance-domain-adjacent-but-uncovered probes (flood insurance on a
# personal home, marine cargo) scored 0.699 and 0.73 -- inside the real
# legitimate range, because they share genuine insurance vocabulary with
# real matches. No single cosine-similarity floor can separate "covered
# topic" from "insurance-shaped but not in these documents" when the
# wording genuinely overlaps; the LLM's own "not stated in these documents"
# instruction is already doing that job correctly (verified live for all
# three off-topic probes, not just the two edge cases) and isn't something
# this gate needs to duplicate.
MIN_RETRIEVAL_SCORE = 0.6

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
    # DECISION (UNIVERSAL, Milestone 2 tuning closed out 2026-09-11, ADR-0005):
    # measured, not guessed -- topk_experiment.py swept k=1,2,3,5 against the
    # real 16-document golden set. Recall (hit rate 0.83) and MRR (0.79) both
    # plateau starting at k=2 and don't improve at k=3 or k=5; precision is
    # actually highest at k=2 (0.79, vs 0.75 at k=3 and 0.72 at k=5, since
    # larger k just adds more non-matching padding chunks); latency roughly
    # triples from k=2 to k=3 (~9s -> ~13s) since the LLM prompt grows with
    # every extra retrieved chunk. k=1 loses real recall (0.75). k=2 is the
    # elbow on every axis at once -- not a compromise between them.
    top_k: int = 2
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


_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')


def _normalize_policy_number(policy_number: str) -> str:
    return re.sub(r"\s+", " ", policy_number.strip()).lower()


def _group_by_policy_number() -> Dict[str, List[dict]]:
    """All non-error documents_db entries, grouped by normalized policy
    number -- a group with more than one entry is the same policy appearing
    as multiple periods/renewals/endorsements, not multiple policies."""
    families: Dict[str, List[dict]] = {}
    for doc in documents_db.values():
        if "error" in doc:
            continue
        policy_number = doc.get("policy_number")
        if not policy_number:
            continue
        families.setdefault(_normalize_policy_number(policy_number), []).append(doc)
    return families


def _doc_years(doc: dict) -> set:
    """Every 4-digit year this document's period touches, from the parsed
    ISO date when available and from the raw string either way -- a raw
    value like '01 July 2026' still yields '2026' even if parsing failed."""
    years = set()
    for field in ("period_from_iso", "period_to_iso", "period_from", "period_to"):
        raw = doc.get(field)
        if raw:
            years |= {m.group(0) for m in _YEAR_RE.finditer(str(raw))}
    return years


def _sort_key(doc: dict) -> Tuple[int, str]:
    """Higher is more recent. Prefers a parsed period end/start date;
    falls back to upload time when dates couldn't be parsed at all, rather
    than guessing an order from unparseable strings."""
    iso = doc.get("period_to_iso") or doc.get("period_from_iso")
    if iso:
        return (2, iso)
    extracted = doc.get("extracted_date")
    if extracted:
        return (1, extracted)
    return (0, "")


def _pick_latest(docs: List[dict]) -> Tuple[dict, List[dict]]:
    ordered = sorted(docs, key=_sort_key, reverse=True)
    return ordered[0], ordered[1:]


def _describe_period(doc: dict) -> str:
    return f"{doc.get('period_from') or '?'} to {doc.get('period_to') or '?'}"


def find_relevant_source_files(question: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """Scope retrieval to specific documents when the question names them.

    Matches the question text against known policy numbers, insured names,
    and insurers (substring, case-insensitive). Length-gated to avoid short
    values matching incidentally. Returns (None, None) -- meaning search
    unscoped -- when nothing matches, so a generic question still searches
    everything.

    DECISION (fixes a real gap: same policy #, different period): a matched
    policy number can belong to more than one indexed document -- a renewal,
    endorsement, or extension of the same policy, each its own file/metadata
    record. Blending all of them into one retrieval risks mixing one year's
    premium/limit with another's. When the question names a period (a year
    that matches exactly one document in the family), that one is used.
    Otherwise the most recent by period_to/period_from (falling back to
    upload time) is used, and the second return value carries a note --
    built here in code, not left to the LLM to remember -- disclosing that
    other periods exist so the caller can surface it in the answer.
    """
    q = question.lower()
    matches = set()
    insured_name_matches: Dict[str, List[dict]] = {}
    policy_number_hits: Dict[str, dict] = {}

    for doc in documents_db.values():
        if "error" in doc:
            continue
        source_file = doc.get("source_file")
        if not source_file:
            continue

        policy_number = doc.get("policy_number")
        if policy_number and len(policy_number) >= 4 and policy_number.lower() in q:
            policy_number_hits[_normalize_policy_number(policy_number)] = doc
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

    scope_note = None
    if policy_number_hits:
        families = _group_by_policy_number()
        question_years = {m.group(0) for m in _YEAR_RE.finditer(q)}
        for norm_pn, sample_doc in policy_number_hits.items():
            family = families.get(norm_pn) or [sample_doc]
            if len(family) == 1:
                matches.add(family[0]["source_file"])
                continue

            year_matched = [d for d in family if question_years & _doc_years(d)] if question_years else []
            if len(year_matched) == 1:
                matches.add(year_matched[0]["source_file"])
                continue

            latest, others = _pick_latest(family)
            matches.add(latest["source_file"])
            other_periods = "; ".join(_describe_period(o) for o in others)
            scope_note = (
                f"Note: policy {sample_doc.get('policy_number')} has {len(family)} versions on file. "
                f"This answer uses the most recent period on file ({_describe_period(latest)}). "
                f"Other periods on file: {other_periods}. Ask by year to use one of those instead."
            )

    return (sorted(matches) if matches else None), scope_note


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


@app.post("/extract", dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def extract_documents(request: Request):
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
        # DECISION (bug fix): keyed by source_file, not policy_number -- two
        # files sharing a policy number (a renewal, an endorsement) used to
        # collide on this key and silently overwrite each other, so only the
        # last-extracted one ever made it into documents_db. source_file is
        # unique per upload; policy-number grouping for "same policy,
        # multiple periods" is done separately in find_relevant_source_files.
        doc_id = meta.get("source_file", "unknown")
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


@app.get("/documents", dependencies=[Depends(require_api_key)])
async def list_documents():
    chunk_counts = await run_in_threadpool(count_chunks_by_source)
    documents = [
        {**doc, "chunks_indexed": chunk_counts.get(doc.get("source_file"), 0)}
        for doc in documents_db.values()
    ]
    return {"total": len(documents), "documents": documents}


@app.post("/extract/professional", dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def extract_documents_professional(request: Request):
    """Same extraction pipeline as /extract, but using the classify-then-target
    declarations-window selection (insurance_loader.select_declarations_window)
    instead of a blind text[:8000] truncation. Writes to documents_db_professional
    / the PROFESSIONAL_STORE table row -- entirely separate from /extract's
    documents_db -- so the two extraction strategies can be compared side by
    side on the same document set instead of one overwriting the other.

    Deliberately does NOT re-run index_documents(): chunking/embedding for
    /query is identical either way (Milestone 1 steps 1-2), so this only
    needs to re-run the metadata extraction step to produce a comparable table.
    """
    logger.info("Professional-pipeline extraction requested")
    try:
        results = await run_in_threadpool(load_insurance_documents, None, "classify_then_target")
    except Exception as e:
        logger.exception("Document loading failed (professional pipeline)")
        raise HTTPException(status_code=500, detail=f"Document loading failed: {str(e)}")

    successful = []
    failed = []
    for meta in results["metadata"]:
        doc_id = meta.get("source_file", "unknown")
        documents_db_professional[doc_id] = meta
        (failed if "error" in meta else successful).append(meta)
    save_documents(documents_db_professional, store=PROFESSIONAL_STORE)

    logger.info(
        "Professional-pipeline extraction complete: %d successful, %d failed",
        len(successful), len(failed),
    )

    return {
        "total": len(successful) + len(failed),
        "successful": successful,
        "failed": failed,
    }


@app.get("/documents/professional", dependencies=[Depends(require_api_key)])
async def list_documents_professional():
    chunk_counts = await run_in_threadpool(count_chunks_by_source)
    documents = [
        {**doc, "chunks_indexed": chunk_counts.get(doc.get("source_file"), 0)}
        for doc in documents_db_professional.values()
    ]
    return {"total": len(documents), "documents": documents}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "documents_loaded": len(documents_db)}


@app.post("/search/metadata", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def search_documents_metadata(request: Request, query: str, top_k: int = 5):
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


@app.post("/query", dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
async def query_documents(request: Request, body: QueryRequest):
    if not body.question:
        raise HTTPException(status_code=400, detail="question required")

    variant_config = get_variant(body.variant)
    effective_top_k = variant_config["top_k"] if variant_config else body.top_k
    effective_temperature = variant_config["temperature"] if variant_config else 0

    session_id = body.session_id or str(uuid.uuid4())
    history = sessions_db.get(session_id, deque(maxlen=MAX_HISTORY_TURNS))

    logger.info(
        "Query received: %r (top_k=%d, variant=%s, session=%s, history_turns=%d)",
        body.question, effective_top_k, body.variant, session_id, len(history),
    )
    start_time = time.time()

    try:
        # DECISION (UNIVERSAL, Milestone 5.3, ADR-0008): retrieval runs against
        # search_query (condensed when history exists), not body.question,
        # so a follow-up like "what about its deductible?" resolves the
        # pronoun before embedding/entity-matching. The answer prompt below
        # still uses body.question -- the user's original phrasing -- so
        # the reply reads naturally rather than echoing the rewrite.
        search_query = body.question
        if history:
            search_query = await run_in_threadpool(condense_query, body.question, history)
            if search_query != body.question:
                logger.info("Condensed query: %r -> %r", body.question, search_query)

        source_files, policy_scope_note = find_relevant_source_files(search_query)
        if source_files:
            logger.info("Scoping search to %d matched document(s): %s", len(source_files), source_files)
        if policy_scope_note:
            logger.info(policy_scope_note)
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
                body.question, best_score, MIN_RETRIEVAL_SCORE,
            )
            answer = (
                "I don't have enough relevant information in the indexed documents to "
                "answer this confidently. The closest matches found weren't similar enough "
                "to the question to be trustworthy -- try rephrasing, naming the policy "
                "directly, or confirm the right document has been extracted and indexed."
            )
            if policy_scope_note:
                answer += f"\n\n{policy_scope_note}"
            sources = [
                {"source_file": c["source_file"], "policy_number": c.get("policy_number"), "score": c["score"]}
                for c in chunks
            ]
            history.append({"question": body.question, "answer": answer})
            sessions_db[session_id] = history
            log_experiment_result(
                variant=body.variant,
                question=body.question,
                top_k=effective_top_k,
                temperature=effective_temperature,
                sources=sources,
                latency=time.time() - start_time,
                session_id=session_id,
                low_confidence=True,
            )
            return {
                "question": body.question,
                "answer": answer,
                "sources": sources,
                "chunks_searched": len(chunks),
                "session_id": session_id,
                "low_confidence": True,
                "policy_scope_note": policy_scope_note,
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

        prompt = f"{context}\n\nUser Question: {body.question}\n\nProvide a clear, accurate answer using the summary and excerpts above. Prefer the structured policy summary for simple facts like limits, premiums, and dates. If the answer is not covered, say so."

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

        if policy_scope_note:
            answer += f"\n\n{policy_scope_note}"

        sources = [{"source_file": c["source_file"], "policy_number": c.get("policy_number"), "score": c["score"]} for c in chunks]

        history.append({"question": body.question, "answer": answer})
        sessions_db[session_id] = history

        logger.info("Query answered using %d chunks", len(chunks))

        log_experiment_result(
            variant=body.variant,
            question=body.question,
            top_k=effective_top_k,
            temperature=effective_temperature,
            sources=sources,
            latency=time.time() - start_time,
            session_id=session_id,
            low_confidence=False,
        )

        return {
            "question": body.question,
            "answer": answer,
            "sources": sources,
            "chunks_searched": len(chunks),
            "session_id": session_id,
            "low_confidence": False,
            "policy_scope_note": policy_scope_note,
        }
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/feedback", dependencies=[Depends(require_api_key)])
@limiter.limit("30/minute")
async def submit_feedback(request: Request, body: FeedbackRequest):
    """Record a thumbs up/down (plus optional comment) on a previously
    returned answer -- see feedback_store.py for why this is a durable file,
    not in-memory state like documents_db."""
    if body.rating not in ("up", "down"):
        raise HTTPException(status_code=400, detail="rating must be 'up' or 'down'")

    entry = record_feedback(
        question=body.question,
        answer=body.answer,
        rating=body.rating,
        comment=body.comment,
        sources=body.sources,
        session_id=body.session_id,
        variant=body.variant,
        low_confidence=body.low_confidence,
    )
    return {"status": "recorded", "entry": entry}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)