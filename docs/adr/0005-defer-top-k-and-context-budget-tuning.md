# ADR-0005: Defer top-k and context-budget tuning

**Status:** Top-k superseded by [ADR-0011](0011-topk-tuned-from-real-data.md); context-budget still deferred
**Date:** 2026-08-25 (top-k deferral originating in Milestone 2)

## Context
Two related knobs remain untuned by measurement rather than by decision:

- **top-k** (`main.py`, default 5, user-adjustable to 10 in the frontend): the number of chunks retrieved per query. No value has been tested against known-correct answers — 5 is a carried-over default, not a measured optimum.
- **Context assembly** (`main.py`, `/query` context-building block): retrieved chunks are concatenated into the prompt with no token budget or truncation. This is currently safe only because chunk sizes are capped (~2000 chars) and top-k defaults low — but the model has already hit a per-request token limit once during document extraction (a separate but related symptom), and nothing currently prevents the same failure mode in `/query` if a user raises top-k or a future document set has larger chunks.

Both require the same missing prerequisite to tune correctly: a golden evaluation set — a set of real questions with known-correct answers, run against real documents, so that raising or lowering top-k (or truncating context) can be measured against actual answer accuracy instead of guessed at.

## Decision
Explicitly defer tuning both until a golden eval set exists, rather than picking values without a way to measure their effect. This is a decision to not decide yet, not an oversight — recorded so it isn't mistaken for "already handled."

## Alternatives considered
- **Guess a "reasonable" top-k and a token budget now** — rejected: without a way to measure the effect, any chosen number is exactly as arbitrary as the current default, just with more apparent confidence attached.
- **Build the golden eval set immediately** — not rejected, just not yet in scope; blocked on having a larger/more representative set of real documents and question types than currently available.

## Consequences
- Both `/query`'s retrieval breadth and its context-window safety remain provisional. The context-budget gap in particular is a latent risk, not a hypothetical one — the token-limit failure has already happened once elsewhere in this pipeline.
- Milestone 3's "Response quality tuning / evaluation" checkbox cannot honestly close while this is deferred — they share the same blocking dependency.
- Revisit trigger: once a golden eval set exists, both should be tuned together, since top-k and context-budget interact (raising top-k without a corresponding truncation strategy is what would trigger the token-limit failure mode described above).
- **2026-09-11 update:** the golden eval set now exists (16 real documents, `tests/golden_set.py`). Top-k was measured and resolved — see [ADR-0011](0011-topk-tuned-from-real-data.md), which supersedes this ADR for that half only. Context-budget tuning is still deferred; it wasn't addressed by that measurement.
