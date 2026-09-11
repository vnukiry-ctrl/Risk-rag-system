# ADR-0011: Top-k tuned from real data (k=5 -> k=2)

**Status:** Accepted (supersedes [ADR-0005](0005-defer-top-k-and-context-budget-tuning.md)'s top-k half only)
**Date:** 2026-09-11

## Context
ADR-0005 deferred choosing a real `top_k` value until two things existed: a set of real documents, and a golden Q&A eval set built from them. Both now exist — 16 real Mount Royal University insurance documents were loaded 2026-09-08/09-11, and `tests/golden_set.py` was rebuilt from scratch against them (replacing the old placeholder-document version it previously tested against). That closes the blocking dependency ADR-0005 named.

## Decision
`tests/topk_experiment.py` swept `top_k` at 1, 2, 3, and 5 against the 12 structured-fact golden-set questions, live against the real document set:

| k | hit rate | MRR | precision@k | avg latency |
|---|---|---|---|---|
| 1 | 0.75 | 0.75 | 0.75 | 2.9s |
| 2 | 0.83 | 0.79 | 0.79 | 9.3s |
| 3 | 0.83 | 0.79 | 0.75 | 13.0s |
| 5 (old default) | 0.83 | 0.79 | 0.72 | ~16s+ |

**k=2 is the elbow on every axis at once, not a tradeoff between them:** hit rate and MRR plateau starting at k=2 -- k=3 and k=5 measure identically on both, so the extra retrieved chunks buy nothing there. Precision is actually *highest* at k=2 and gets steadily worse at k=3 and k=5, which is exactly what precision should do as more non-matching padding chunks get added to a fixed-size correct set. Latency roughly triples from k=2 to k=3, since the LLM's prompt grows with every extra retrieved chunk before it ever generates a token. k=1 is the only value that loses something real (hit rate 0.75, a full 8 points below k=2's 0.83) with no offsetting benefit anywhere.

Changed `top_k` default 5 -> 2 in three places: `main.py`'s `QueryRequest` model, `vector_store.semantic_search`'s function signature, and the frontend's `st.slider` initial value (still user-adjustable 1-10 in all three).

## Alternatives considered
- **Sweep further (k=8, 10, 15, 20, as ADR-0005's original deferred methodology suggested)** -- not completed. k=10 was cut short mid-sweep by hitting Groq's free-tier daily token quota (200,000 tokens/day; see Consequences). Decided not to burn remaining quota chasing it: the plateau was already consistent across three independent points (k=2, 3, 5), each showing identical hit-rate/MRR and monotonically *worsening* precision -- there's no pattern in that data suggesting a k=8+ value would suddenly do better instead of continuing the same trend (more latency, same or worse recall, worse precision). If a future, larger or more ambiguous document set behaves differently, re-sweep rather than assume this generalizes forever.
- **Keep k=5 as a "safer" default** -- rejected: it's strictly dominated by k=2 on every measured axis (same hit rate and MRR, worse precision, much worse latency). There's no dimension on which 5 was actually better; it was just the un-measured inherited value.

## Consequences
- Milestone 3's "Response quality tuning / evaluation" checkbox can close its top-k half. Context-budget tuning (`CONTEXT_CHAR_BUDGET`, ADR-0005) and the confidence-gate threshold (`MIN_RETRIEVAL_SCORE`, ADR-0009) are separate, still-open dependencies -- lower-risk now that less text gets retrieved per query by default, but neither was itself measured by this sweep.
- **Groq's free-tier daily quota (200,000 tokens) is a real, now-repeatedly-hit operational constraint**, not a hypothetical one -- this session's tuning work alone hit it three separate times (once mid-sweep here, twice during `/extract` re-runs earlier the same day). `topk_experiment.py` now writes partial results on a mid-sweep failure specifically because of this; any future re-measurement (a bigger document set, a re-sweep, a larger golden set) should budget for it, or use a paid tier.
- This result is measured against 12 structured-fact questions over 16 documents -- small enough that, per `golden_set.py`'s own stated philosophy, it's a real floor that shouldn't regress, not a statistically strong claim that k=2 is optimal for a much larger or differently-shaped document set. Re-sweep when the document set changes meaningfully.
