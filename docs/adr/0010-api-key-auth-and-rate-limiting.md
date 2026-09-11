# ADR-0010: Shared API-key auth and per-key rate limiting

**Status:** Accepted
**Date:** 2026-09-02

## Context
Milestone 6 (Production Deployment) starts from every endpoint in `backend/main.py` being
open to any caller with no rate limit — fine for a `localhost`-only dev loop, not fine the
moment this deploys anywhere reachable by anyone else. Two gaps needed closing: nothing
verifies who's calling, and nothing stops a caller (malicious or just a retry loop) from
hammering `/query`/`/extract`, both of which fan out to paid/rate-limited external services
(Groq, the embedding backend).

## Decision
**Auth:** a shared-secret API key, not per-user accounts. `API_KEYS` (comma-separated, so
more than one caller — the frontend, a script — can hold an independently rotatable key)
is read once at startup. `require_api_key` (a FastAPI dependency using `APIKeyHeader`,
header name `X-API-Key`) gates every endpoint except `/` and `/health`, which stay open
for load-balancer/monitoring checks that shouldn't need a secret. An **unset `API_KEYS` is
treated as auth disabled**, not refused at startup — local dev without a `.env` keeps
working — but it logs a `WARNING` on every startup, so a deployment that forgot to set it
can't miss the signal in its own logs.

**Rate limiting:** `slowapi` (a `limits`-backed FastAPI/Starlette middleware), keyed by the
caller's API key — falling back to remote address only in the no-auth-configured case, so
quota is per caller rather than per shared NAT/proxy IP. Limits are route-specific:
20/minute on `/query` and 5/minute on `/extract` (the two that call out to Groq/the
embedding backend), 30/minute on `/feedback` and `/search/metadata`, and a 100/minute
default for anything else (`/documents`). These are untuned starting numbers — same honesty
as `top_k`/`MIN_RETRIEVAL_SCORE` elsewhere in this codebase — picked to be generous for a
single legitimate UI session while still bounding worst case, not measured against real
concurrent usage.

The frontend (`frontend/app.py`) reads its own `API_KEY` from `.env` and sends it as
`X-API-Key` on every backend call it makes.

## Alternatives considered
- **Per-user accounts (JWT/OAuth)** — rejected for now: this is a single-deployment
  internal tool with one real caller (the Streamlit frontend) plus maybe a script, not a
  multi-tenant product. A user table, login flow, and token refresh would be pure overhead
  for a system nobody's building a login page for.
- **API-gateway-level rate limiting (nginx, Cloudflare, etc.)** — real option once this
  actually sits behind a gateway, but Milestone 6's other items (Docker, load testing) aren't
  done yet, so there's no gateway to configure. In-app `slowapi` works today, standalone, and
  isn't wasted if a gateway is added later — defense in depth, not a dead end.
- **In-memory-only rate limiter with no storage backend** — this is `slowapi`'s default and
  what's actually configured: fine for a single-process deployment (matches `sessions_db`/
  `documents_db`'s existing in-memory, lost-on-restart posture), but won't share state across
  multiple worker processes/replicas. Revisit with `limits`' Redis backend if this ever runs
  as more than one process — see Consequences.

## Consequences
- Every endpoint except `/` and `/health` now requires a valid key when `API_KEYS` is set;
  `frontend/app.py` was updated to send one, so the existing UI keeps working once both
  `.env` files carry the same key.
- CORS is still `allow_origins=["*"]` — deliberately untouched here. Locking it down needs a
  real deployment origin to restrict to, which doesn't exist yet; tracked as a Milestone 6
  follow-up, not silently forgotten.
- The in-memory rate-limit store means quota resets on restart and isn't shared across
  processes — consistent with every other piece of in-memory state in this codebase
  (`documents_db`, `sessions_db`), but will need `limits`' Redis backend the moment this runs
  as more than one uvicorn worker or behind a multi-replica deployment.
- All five rate limits are guesses, not measurements. Watch for legitimate multi-turn
  sessions tripping `/query`'s 20/minute (a fast back-and-forth conversation could plausibly
  approach that) once there's real usage to observe — same "recalibrate once real traffic
  exists" stance as [ADR-0005](0005-defer-top-k-and-context-budget-tuning.md).
