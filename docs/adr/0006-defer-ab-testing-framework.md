# ADR-0006: Defer A/B testing framework

**Status:** Deferred
**Date:** 2026-08-25

## Context
Milestone 4's checklist includes an A/B testing framework — infrastructure to route requests to two variants (e.g. different `top_k`, temperature, or prompt wording) and compare their metrics. Building the routing/logging mechanism itself is straightforward and not insurance-specific. The problem is what it would compare against: this project currently has 9 real documents and no live users, so any two variants would be compared on the same 5-7 golden-set questions (`backend/tests/golden_set.py`) — nowhere near enough samples to distinguish a real effect from noise. A/B testing on a sample that small doesn't produce a meaningful result; it produces a coin flip dressed up as one.

## Decision
Defer building the A/B testing framework until there's either real user traffic or a golden set large enough for statistically meaningful comparisons. This is the same shape of decision as ADR-0005 (defer top-k/context-budget tuning): a decision to not decide yet, recorded so it isn't mistaken for an oversight.

## Alternatives considered
- **Build the routing/logging mechanism now, populate it with real comparisons later** — rejected for now: unused infrastructure that can't be exercised meaningfully yet is speculative scope, not readiness. Revisit once there's something real to compare.
- **Run informal comparisons on the current golden set anyway** — not rejected as a technique, just not "A/B testing" — this is what golden-set regression testing (`tests/test_quality.py`) already does for single-variant changes (did this change make the metrics worse), which doesn't need dedicated A/B infrastructure.

## Consequences
- Milestone 4's "A/B testing framework" checkbox stays unchecked until the blocking dependency (traffic or a larger golden set) is resolved.
- Any variant decision made in the meantime (e.g. tuning `top_k`, per ADR-0005) has to rely on the golden-set eval suite's aggregate metrics, not a true A/B comparison — a real but accepted limitation at this project's current scale.
- Revisit trigger: once real user queries are flowing (unblocks the user-feedback-loop item too) or the golden set grows past roughly a few dozen cases per variant, whichever comes first.
