# ADR-0007: A/B testing mechanism built, not activated

**Status:** Accepted (mechanism only) — activation remains Deferred per ADR-0006
**Date:** 2026-08-25

## Context
ADR-0006 deferred *building* an A/B testing framework at all, reasoning that infrastructure nobody can exercise meaningfully yet is speculative scope. Since then, the ask changed to something narrower: build the routing/logging *mechanism* now, without turning on real variant comparisons, so that the moment there's real traffic or a larger golden set, activating it is a config change rather than a rebuild. This is not a reversal of ADR-0006 — the blocker it identified (no real traffic, no large enough golden set) still applies to actually *using* this for a decision. It's a narrower decision: build the plumbing, leave the tap off.

## Decision
Added `backend/experiments.py`: named variant configs (`control`, `wider_retrieval`, `higher_temperature`), a `get_variant()` lookup, and a `log_experiment_result()` that appends to `experiments_log.jsonl`. `/query`'s new `variant` field (`main.py`) defaults to `None`, which preserves the exact current behavior (`top_k` from the request, `temperature=0`) — naming a variant is opt-in per request, not a global switch.

## Alternatives considered
- **Do nothing until real traffic exists** (ADR-0006's original stance) — superseded for the mechanism specifically, at the user's request, on the reasoning that the mechanism itself carries near-zero risk (default-off, additive-only) while removing later rebuild work.
- **Turn variants on by default / randomly assign traffic to them now** — rejected: with 9 documents and no live users, a random split has nothing meaningful to measure. Explicit opt-in (`variant` param) keeps this a deliberate act, not silent A/B'ing of unwitting test traffic.

## Consequences
- `/query`'s behavior is unchanged for every existing caller (frontend, `perf_smoke_test.py`, `tests/`) — none of them pass `variant`.
- **Open verification item, not yet closed:** the `variant` code path itself has not been exercised end-to-end against a live Qdrant-backed query. Verification was blocked by an unrelated environment issue — a stale/orphaned process holding port 8000 and the Qdrant on-disk lock that Windows' own tools (`tasklist`, `Get-CimInstance`) couldn't identify (see conversation 2026-08-25). **Revisit trigger:** once that port is clear and the backend restarts cleanly, run one `/query` call with `"variant": "wider_retrieval"` and confirm `chunks_searched` reflects `top_k=10` and the entry lands in `experiments_log.jsonl` — only then is this ADR's mechanism considered verified, not just written.
- Actually *activating* comparisons (reading `experiments_log.jsonl` to draw a conclusion) remains blocked on ADR-0006's original condition: real traffic or a golden set large enough per variant to be statistically meaningful.

## Update (2026-09-02): logging schema fixed ahead of real data, still not verified end-to-end
While preparing for real usage, found that `log_experiment_result()` was flattening each
result's `sources` down to bare `source_files` — dropping the retrieval score, and never
recording `session_id` or `low_confidence` at all. That would have made a later variant
comparison (or ADR-0009's confidence-threshold recalibration) impossible to do from this log
alone once real traffic exists — exactly the situation this ADR's "plumbing, not a rebuild"
reasoning was meant to avoid. Fixed: `sources` (with scores) plus `session_id` and
`low_confidence` are now recorded on every `/query` call. `feedback_store.record_feedback()`
got the same fix — a rating now carries `session_id`/`variant`/`low_confidence` so a "down"
vote can be joined back to what produced it. The frontend (`frontend/app.py`) previously never
sent `session_id` back on follow-up questions and had no feedback UI at all, so under real
usage `feedback_log.jsonl` would have stayed empty and no session ever accumulated multi-turn
history — both fixed (session persists in `st.session_state` across turns; 👍/👎 buttons call
`/feedback`). **The open verification item above is unchanged and still outstanding** — this
only ensures that once it's run, the data needed to interpret the result is actually captured.
