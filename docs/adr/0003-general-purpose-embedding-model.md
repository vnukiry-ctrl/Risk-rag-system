# ADR-0003: Ollama `nomic-embed-text`, general-purpose embeddings

**Status:** Superseded by [ADR-0013](0013-voyage-law-embeddings.md)
**Date:** 2026-08-25 (originating in Milestone 2)

## Context
Retrieval quality depends more on the embedding model than on any other single knob in this pipeline (chunk size, top-k, or prompt wording) — it determines whether semantically related text actually lands close together in vector space at all. This project needed a free, locally-run, swappable embedding provider during development, before committing to a paid/hosted option.

A live test run during this project surfaced a symptom consistent with a general-purpose embedding's limits: retrieval scores across clearly different, unrelated policy documents clustered tightly (0.71-0.76) for the same query, rather than showing a sharp separation between "this document is what you're asking about" and "this document just shares generic insurance vocabulary." A domain-tuned embedding would be expected to separate these more cleanly, since it would encode e.g. "coverage limit" and "premium" as distinct concepts rather than similarly-weighted general text.

## Decision
Use Ollama's `nomic-embed-text` (768-dim, general-purpose) as the embedding provider, via a batched direct call to Ollama's `/api/embed` endpoint ([vector_store.py](../../backend/vector_store.py)), with the provider swap already stubbed for Anthropic embeddings.

## Alternatives considered
- **A domain-tuned or larger embedding model** — not adopted yet; deferred until retrieval quality is actually measured against a golden eval set (there's no baseline number to know if a swap would help, only the score-clustering symptom above as a hint).
- **Anthropic embeddings** — stubbed as the swap path but not adopted, to keep development free/local while the system was unproven.

## Consequences
- Zero cost and no external dependency during development, at the cost of embedding quality that's a plausible bottleneck on retrieval accuracy — the score-clustering symptom is suggestive, not conclusive, without a measured baseline.
- This is the first thing to try if retrieval quality is the bottleneck on a *new* project, before spending time on chunk-size or top-k tuning — those are second-order compared to whether the embedding model understands the domain's vocabulary at all.
- Revisit trigger: once a golden eval set exists (see [ADR-0005](0005-defer-top-k-and-context-budget-tuning.md)), measure retrieval accuracy with the current model as a baseline before deciding whether a swap is worth the cost/latency trade-off.
