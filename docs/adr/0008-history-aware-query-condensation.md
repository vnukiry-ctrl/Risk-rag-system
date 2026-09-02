# ADR-0008: History-aware query condensation before retrieval

**Status:** Accepted
**Date:** 2026-08-26

## Context
Milestone 5.2 added chat history to `/query`: prior (question, answer) turns are replayed
to the LLM as real messages, so the *answer* can reference earlier context. But
`semantic_search` only ever sees `request.question` verbatim. A natural follow-up like
"what about its deductible?" embeds and entity-matches on its own words alone — "its" isn't
a policy number or insured name, so [ADR-0004](0004-entity-scoped-retrieval-filtering.md)'s
entity-scoped filter can't narrow the search, and the embedding carries none of the prior
turn's meaning. Retrieval for genuine multi-hop follow-ups was still broken even after 5.2,
exactly as flagged in `docs/learning-guide.md` §5.2's "dependency to note."

## Decision
Before retrieval, if the session has any prior turns, send the latest question plus the
history to the LLM and ask it to rewrite the question as a standalone one (resolving
pronouns/implied references, preserving intent, no added information). The rewritten query
is used for both `find_relevant_source_files()` and `semantic_search()`. The *original*
question is still what goes into the final answer prompt and gets stored back into history,
so the reply reads naturally rather than echoing the rewrite. Implemented as
`condense_query()` in [main.py](../../backend/main.py).

Skipped entirely on the first turn of a session (empty history) — there is nothing to
condense against, and it avoids paying an extra LLM round-trip on the common case of a
standalone question.

## Alternatives considered
- **Query decomposition** (split a compound question into sub-queries, retrieve each,
  combine) — rejected for now: no observed real question in this document set actually
  needs cross-policy comparison; adding it speculatively would be the same "cost not yet
  earned" mistake the guide's own §5.3 table warns against.
- **HyDE** (embed a hypothetical answer instead of the question) — rejected: this project's
  questions are short factual lookups (policy numbers, limits, dates) whose phrasing already
  resembles the source text closely enough that embedding mismatch hasn't been observed as a
  failure mode.
- **Step-back prompting** — rejected: the documents are single-policy PDFs with no broader
  background context to step back into; nothing to retrieve at a "more general" level that
  isn't already in the structured summary.
- **Flatten condensation into the existing history-replay prompt** (let the answering LLM
  call implicitly resolve the reference) — rejected: that only fixes the *answer*, not
  retrieval; `semantic_search` runs before that LLM call ever sees the history, so this
  doesn't touch the actual bug.

## Consequences
- Follow-up questions ("its deductible", "what about the auto policy" after a CGL question)
  now retrieve against a query that names the resolved entity, which also lets ADR-0004's
  entity-scoped filter engage on turns two and later where it couldn't before.
- Adds one extra LLM call (small prompt, `max_tokens=100`, `temperature=0`) per follow-up
  turn — paid only when history is non-empty, not on every query.
- The rewrite quality is bounded by the LLM's ability to resolve references correctly; a bad
  rewrite silently changes what gets retrieved with no explicit failure signal beyond a
  logged before/after pair. Not yet covered by an automated case in
  `backend/tests/golden_set.py` — worth adding once a real multi-turn failure is observed
  live, same discipline ADR-0004's entity-ambiguity case followed.
- If this call fails (LLM error, timeout), retrieval silently falls back to the raw
  question — today's pre-5.3 behavior, not a hard failure.
