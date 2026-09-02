"""
Milestone 4 quality-evaluation suite.

Covers three of the five no-blocker Milestone 4 items in one file since they
all read off the same set of live API calls: retrieval metrics (hit rate,
MRR, precision@k), answer-correctness scoring, and latency percentiles.
Automated test suite (the fourth item) is this file itself, replacing
perf_smoke_test.py's manual "read the printed HIT/MISS lines" pattern with
real assertions that fail a run.

GOOD automated test suite: real assert()s that fail CI/a local run on
regression, deterministic given the same live data, and covers known edge
cases (the known_limitation cases below) instead of only the happy path.
NOT GOOD: print-and-eyeball output (the old pattern), tests that pass
regardless of what the API actually returned, or a suite that only ever
tests the cases that are known to work.

Run with: pytest tests/ -v   (from backend/, with the API already running --
see conftest.py for why this suite talks to HTTP instead of importing
vector_store directly).
"""
import statistics
import time

import pytest
import requests

from golden_set import GOLDEN_SET

# Opts this file (and only this file) into conftest.py's require_running_backend
# fixture -- this suite genuinely needs the live API, unlike test_multi_format.py.
pytestmark = pytest.mark.usefixtures("require_running_backend")

BASE_URL = "http://127.0.0.1:8000"


def reciprocal_rank(sources, expected_policy):
    """1/rank of the first source matching expected_policy; 0 if absent.

    MEANING: how far down the ranked results the correct chunk sits, not
    just whether it's present at all -- a hit at rank 1 and a hit at rank 5
    both count as a "hit" for hit-rate, but only MRR tells them apart.
    IDEAL: 1.0 (always rank 1). Below 0.5 means the correct chunk is
    typically buried past rank 2, which becomes a real problem the moment
    top_k is lowered from its current default of 5.
    """
    for rank, s in enumerate(sources, start=1):
        if s.get("policy_number") == expected_policy:
            return 1.0 / rank
    return 0.0


def precision_at_k(sources, expected_policy):
    """Fraction of returned sources that belong to the expected policy.

    MEANING: in most RAG setups this metric is naturally noisy (padding
    chunks retrieved for context aren't "wrong," just not the answer chunk).
    Entity-scoped filtering (ADR-0004, main.py's find_relevant_source_files)
    restricts search to the matched document and pushes precision toward
    1.0 -- but ONLY when the question contains a literal policy number,
    insured name, or insurer name substring. CORRECTED AFTER FIRST REAL RUN:
    the initial version of this comment claimed every golden-set question
    triggers scoping; measured precision (0.2 on the "medical professional
    liability" case) disproved that -- that question names the
    insurance_type, which find_relevant_source_files() does NOT match on, so
    it searches unscoped and comes back noisy like any unscoped RAG query.
    IDEAL: ~1.0 on questions that name an entity literally; ordinary/noisier
    (0.2-0.5 is fine) on questions that only describe the insurance_type,
    since those are never scoped by design today. A drop on an
    entity-naming question specifically is the real regression signal.
    """
    if not sources:
        return 0.0
    matching = sum(1 for s in sources if s.get("policy_number") == expected_policy)
    return matching / len(sources)


@pytest.fixture(scope="session")
def query_results():
    """Run every golden-set question against the live API exactly once.

    Shared across all tests in this file so hit-rate/MRR/precision/latency/
    correctness scoring don't each re-issue the same LLM calls -- each query
    costs a real Groq API call plus local embedding time, so re-running the
    same 7 questions four times over would be a wasteful, avoidable choice.
    """
    results = []
    for case in GOLDEN_SET:
        start = time.time()
        resp = requests.post(
            f"{BASE_URL}/query",
            json={"question": case["question"], "top_k": 5},
            timeout=60,
        )
        elapsed = time.time() - start
        resp.raise_for_status()
        results.append({"case": case, "elapsed": elapsed, "response": resp.json()})
    return results


def _structured_fact_results(query_results):
    return [r for r in query_results if r["case"]["type"] == "structured_fact"]


def test_retrieval_hit_rate(query_results):
    """Is the expected document present anywhere in the top-k results?

    MEANING: the coarsest retrieval signal -- did semantic search find the
    right document at all, regardless of rank or precision.
    IDEAL: 1.0 for a small curated set like this one, where every question
    names a single, unambiguous document (the known_limitation cases, which
    deliberately test the ambiguous/failing side of retrieval, are scored
    separately below and excluded here).
    NOT GOOD: below ~0.7 -- at that point retrieval is failing on cases
    designed to be easy, which points at chunking/embedding/scoping, not
    at the LLM.
    """
    cases = _structured_fact_results(query_results)
    hits = [
        r["case"]["expected_policy"] in {s.get("policy_number") for s in r["response"].get("sources", [])}
        for r in cases
    ]
    hit_rate = sum(hits) / len(cases)
    print(f"\nRetrieval hit rate: {sum(hits)}/{len(cases)} ({hit_rate:.0%})")
    assert hit_rate >= 0.8, f"hit rate {hit_rate:.0%} is below the 80% floor -- see printed per-case detail above"


def test_mean_reciprocal_rank(query_results):
    cases = _structured_fact_results(query_results)
    rrs = [reciprocal_rank(r["response"].get("sources", []), r["case"]["expected_policy"]) for r in cases]
    mrr = statistics.mean(rrs)
    print(f"\nMRR: {mrr:.3f} (per-case: {[round(x, 2) for x in rrs]})")
    assert mrr >= 0.8, f"MRR {mrr:.3f} is below the 0.8 floor -- correct chunks are ranking lower than expected"


def test_precision_at_k(query_results):
    cases = _structured_fact_results(query_results)
    precisions = [precision_at_k(r["response"].get("sources", []), r["case"]["expected_policy"]) for r in cases]
    avg_precision = statistics.mean(precisions)
    print(f"\nPrecision@k: {avg_precision:.3f} (per-case: {[round(x, 2) for x in precisions]})")
    assert avg_precision >= 0.6, (
        f"precision@k {avg_precision:.3f} is below the 0.6 floor -- since these questions are "
        f"entity-scoped, a drop here likely means find_relevant_source_files() (main.py) regressed"
    )


def test_structured_fact_answer_correctness(query_results):
    """Exact-match-by-containment on the generated answer text.

    MEANING: retrieval can be perfect while the LLM still writes the wrong
    number into the answer -- this test checks what the user actually reads,
    not just what was retrieved. Scored as substring containment (does the
    known-correct value appear in the answer text) rather than a strict
    equality on the whole answer, since free-text answers legitimately vary
    in wording around the number.
    IDEAL (insurance-domain decision, not a general one): 100% -- a wrong
    dollar figure, deductible, or policy number is a liability/compliance
    problem, not an approximation error, so there is no "partial credit"
    threshold the way there might be for a fuzzy relevance score.
    NOT GOOD: any failure here at all. Unlike the retrieval metrics above,
    this one has no tolerance band -- every failure is reported individually
    below so it's obvious which fact the LLM got wrong.
    """
    cases = _structured_fact_results(query_results)
    failures = []
    for r in cases:
        answer = r["response"].get("answer", "").lower()
        expected_values = [v.lower() for v in r["case"]["expected_answer_contains"]]
        if not any(v in answer for v in expected_values):
            failures.append((r["case"]["question"], expected_values, r["response"].get("answer", "")[:200]))

    if failures:
        detail = "\n".join(f"  Q: {q}\n    expected one of {vals} in answer, got: {ans!r}" for q, vals, ans in failures)
        pytest.fail(f"{len(failures)}/{len(cases)} structured-fact answers missing expected value:\n{detail}")


def test_latency_percentiles(query_results):
    """p50/p95 end-to-end /query latency (embedding + retrieval + LLM generation).

    MEANING: percentiles over avg/min/max because a single slow outlier gets
    averaged away in a small sample but would show up in a real p95 -- the
    metric this replaces (perf_smoke_test.py's avg/min/max) can't tell "one
    query" from "consistently a bit slow."
    ASPIRATIONAL UX IDEAL vs. MEASURED REALITY: general UX guidance would
    target p50 < 3s, p95 < 6s for a synchronous Q&A endpoint. The first real
    run of this suite measured p50 ~13s / p95 ~21s instead -- dominated by
    the Groq LLM call (cloud round-trip, max_tokens=500) plus local Ollama
    embedding. The floor below is set from THAT measurement, not the
    aspirational target, on purpose: an assert should catch a regression
    from today's real baseline, not fail every run against a number nobody
    has hit yet. Tightening it is future work (shorter max_tokens, streaming,
    a faster model) -- a separate decision from "does this suite work."
    NOT GOOD: any new failure here means a real regression from the measured
    baseline, not proof the system is slow in general -- it already was.
    CAVEAT: with only 7 samples, this p95 is illustrative, not a trustworthy
    tail estimate -- a real p95 needs hundreds of samples (production
    traffic), which is exactly the kind of number this suite can't produce
    yet at 9 documents and no live users.
    """
    latencies = sorted(r["elapsed"] for r in query_results)
    p50 = statistics.median(latencies)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 2 else latencies[0]
    print(f"\nLatency  p50: {p50:.2f}s  p95: {p95:.2f}s  min: {latencies[0]:.2f}s  max: {latencies[-1]:.2f}s")
    assert p50 < 18, f"p50 latency {p50:.2f}s exceeds the 18s regression floor (measured baseline: ~13s)"
    assert p95 < 28, f"p95 latency {p95:.2f}s exceeds the 28s regression floor (measured baseline: ~21s)"


@pytest.mark.xfail(
    strict=True,
    reason="ambiguous insured_name ('Mount Royal University') breaks entity-scoped "
           "filtering in find_relevant_source_files() -- see golden_set.py case note",
)
def test_known_limitation_ambiguous_entity_scoping(query_results):
    """If this starts passing, the scoping bug was fixed -- promote the case
    in golden_set.py to 'structured_fact' rather than leaving it here."""
    case = next(r for r in query_results if r["case"]["question"].startswith("What is the coverage limit for the Mount Royal"))
    sources = case["response"].get("sources", [])
    hit = case["case"]["expected_policy"] in {s.get("policy_number") for s in sources}
    assert hit, "expected miss: ambiguous entity name currently scopes to the wrong document"


@pytest.mark.xfail(
    strict=True,
    reason="MIN_RETRIEVAL_SCORE=0.5 can't tell this case apart from a real match -- "
           "the wrong-document (BW240599) hit scores ~0.72, inside the same 0.71-0.76 "
           "band ADR-0004 measured for genuinely-correct matches, so a single similarity "
           "floor can't gate on this without also rejecting legitimate answers. See "
           "ADR-0009 Consequences (verified 2026-09-02) for the live-run result.",
)
def test_hallucination_gate_group_accident(query_results):
    """Milestone 5.4 (ADR-0009): the low-confidence case must not silently
    misattribute coverage to the wrong policy.

    Unlike the xfail test below (which checks whether the real document
    becomes retrievable), this checks the other half of the same bug: even
    while the document stays unretrievable, the system must not confidently
    answer from the wrong one. Two outcomes are acceptable -- retrieval
    already finds the real document, or the confidence gate (main.py,
    MIN_RETRIEVAL_SCORE) refuses to answer -- everything else means a
    confidently wrong answer slipped through.

    If this starts passing, either the document became retrievable (see the
    xfail below) or a smarter gate (reranking, per-document score margin,
    LLM self-consistency) replaced the flat score floor -- remove this xfail
    and note which one in ADR-0009.
    """
    case = next(r for r in query_results if "Group Accident" in r["case"]["question"])
    response = case["response"]
    retrieved_files = {s.get("source_file") for s in response.get("sources", [])}
    found_real_doc = "25-26 Group Accident Policy 100013386.pdf" in retrieved_files
    refused = response.get("low_confidence") is True
    assert found_real_doc or refused, (
        "neither retrieved the real document nor refused on low confidence -- got a "
        f"confident answer sourced from the wrong policy: {response.get('answer', '')[:200]!r}"
    )


@pytest.mark.xfail(
    strict=True,
    reason="Group Accident policy (100013386.pdf) doesn't surface in top-5 for an "
           "on-topic query, and the LLM hallucinates a wrong attribution instead of "
           "saying not found -- see golden_set.py case note",
)
def test_known_limitation_group_accident_retrievability(query_results):
    """If this starts passing, the doc has become retrievable for this
    question -- promote the case in golden_set.py and separately verify the
    LLM no longer misattributes coverage to the wrong policy when it is."""
    case = next(r for r in query_results if "Group Accident" in r["case"]["question"])
    retrieved_files = {s.get("source_file") for s in case["response"].get("sources", [])}
    assert "25-26 Group Accident Policy 100013386.pdf" in retrieved_files, (
        "expected miss: this document isn't being retrieved for its own topic"
    )
