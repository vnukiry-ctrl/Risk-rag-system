# ADR-0009: Retrieval-confidence gating before answering

**Status:** Accepted (mechanism built; does not catch the reproduction case it targeted, see Consequences)
**Date:** 2026-08-27

## Context
Milestone 3's known gaps (and `backend/tests/golden_set.py`'s `known_limitation` cases)
document a specific, reproduced failure: asking "What does the Group Accident insurance
policy cover?" doesn't surface that document's own chunks in the top-5 semantic search
results at all. Instead of stating it doesn't know, the LLM confidently answers using an
unrelated policy's chunks (BW240599) — a wrong attribution stated with full confidence.
The explicit "say so if it's not in the excerpts" grounding instruction (`docs/learning-guide.md`
§3.2) does not hold in this case: chunks *are* present in the context, they're just from the
wrong document, so the LLM has no signal telling it the excerpts don't actually answer the
question.

## Decision
Before building the answer prompt, check the retrieved chunks' similarity scores (already
returned by Qdrant, no extra call needed). If chunks came back but none clear
`MIN_RETRIEVAL_SCORE`, skip the LLM call entirely and return an explicit "not enough
relevant information" response instead — same shape response (`question`, `answer`,
`sources`, `chunks_searched`, `session_id`), plus a `low_confidence: true` flag callers can
check. An empty `chunks` list already produced an honest "no excerpts found" message before
this change; this covers the other half of the same bug — chunks that exist but are
irrelevant. Implemented inline in `/query` (`main.py`), right after `semantic_search`.

## Alternatives considered
- **LLM-as-judge / faithfulness scoring** (post-hoc, reusing the 4.2 groundedness metric at
  runtime) — rejected for now: it requires the potentially-hallucinated answer to be
  generated first, then a second LLM call to check it — real added latency and cost per
  query, on top of not fixing anything before the bad answer already exists.
- **Source-attribution verification** (string/entity-match the answer's claims back to
  retrieved excerpts) — rejected: cheaper than LLM-as-judge, but still post-hoc, and
  wouldn't catch this specific failure any more directly than the gate does — the answer
  correctly cites the wrong policy's real source file, it's just the wrong file.
- **Leave the grounding instruction as the only defense** — this is what's failing; rejected
  as the status quo that created the bug in the first place.

## Consequences
- Directly addresses the reproduced failure at its root: weak retrieval no longer reaches
  the LLM framed as if it were adequate context.
- Adds zero extra LLM calls — the score was already computed by Qdrant; this is pure
  post-retrieval logic.
- **Verified live 2026-09-02, and it does not catch the case it was built for.** Running
  `test_hallucination_gate_group_accident` against the live API: the Group Accident question
  retrieves BW240599 (the wrong policy) at score ~0.72 and answers confidently — the gate
  never fires because 0.72 clears the 0.5 floor. This confirms the risk flagged when
  `MIN_RETRIEVAL_SCORE = 0.5` was chosen: [ADR-0004](0004-entity-scoped-retrieval-filtering.md)'s
  0.71–0.76 "genuinely-topical-but-wrong-document" cluster is exactly where this wrong match
  landed, and it's indistinguishable by score alone from a real, correct match in that same
  band. **A single similarity-score floor cannot fix this specific failure** — raising the
  threshold to exclude 0.72 would also reject legitimate borderline-but-real answers, not just
  this wrong one. The test is now `xfail` (`backend/tests/test_quality.py`), documenting this
  as a known gap rather than a bug to chase with more threshold-tuning.
- What the gate still does: catch the *other* half of Milestone 3's known gaps — retrieval that
  comes back weak/scattered across the board (nothing loosely on-topic), which is a different
  and genuinely score-distinguishable failure mode from "one wrong document happens to score
  respectably."
- Fixing the Group Accident case for real needs something score-independent: making the
  document retrievable in the first place (the existing `xfail` retrievability test), a
  reranker, a per-query score-margin check (best-vs-second-best gap, not an absolute floor),
  or an LLM self-consistency/faithfulness check post-answer. None implemented yet — deferred
  until there's enough real traffic to justify the added latency/cost of any of them.
- Same deferred-tuning stance as [ADR-0005](0005-defer-top-k-and-context-budget-tuning.md):
  a real threshold needs real score distributions across more documents and questions than
  currently exist. Watch for false refusals (a genuinely answerable question scoring just
  under 0.5) as more documents are added — that's the signal to revisit the number, not a
  guess made in advance of evidence.
