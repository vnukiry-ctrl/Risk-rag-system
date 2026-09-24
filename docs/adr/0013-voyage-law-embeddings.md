# ADR-0013: Voyage AI `voyage-law-2` embeddings, replacing Ollama `nomic-embed-text`

**Status:** Accepted, pending golden-set re-verification
**Date:** 2026-09-24

## Context
[ADR-0003](0003-general-purpose-embedding-model.md) chose Ollama's `nomic-embed-text` — free, local, general-purpose — with an explicit revisit trigger: once a golden eval set existed, measure retrieval accuracy with it as a baseline before deciding whether a domain-tuned or larger model was worth the cost/latency trade-off. That golden set now exists (`tests/golden_set.py`, 14 hand-verified cases against the real Mount Royal University documents).

Separately, this project moved from "runs on my machine" toward "other people need to reach it." Ollama is a host-level dependency (no local Ollama process, no embeddings, no matter how the rest of the stack is deployed) — a Docker Compose service, a PaaS deploy, and a cloud VM all either can't run it or require deliberately standing up Ollama somewhere reachable. Removing that dependency was going to be necessary regardless of embedding quality.

Given both were on the table at once, the general-purpose-vs-domain-tuned question was worth resolving as part of the same change rather than swapping providers twice: `nomic-embed-text` has no notion of "exclusion" vs "endorsement" vs "coverage limit" as legal-contract concepts, which is exactly the vocabulary this project's documents are made of.

## Decision
Switch to Voyage AI's `voyage-law-2` — a 1024-dimension model trained specifically on legal/contract text, which insurance policies are — called via the `voyageai` SDK ([vector_store.py](../../backend/vector_store.py)). Both `embed_documents()` and `embed_query()` pass Voyage's `input_type` parameter (`"document"` / `"query"` respectively), using its asymmetric-embedding support rather than embedding both sides of a retrieval the same way.

Because `index_documents()` already calls `client.recreate_collection()` on every reindex ([vector_store.py](../../backend/vector_store.py)), the 768→1024 dimension change needed no migration step — just a full re-extraction against the real document set.

## Alternatives considered
- **OpenAI `text-embedding-3-small`** — cheaper ($0.02/1M vs. free-tier-then-usage) and the largest ecosystem/community, but general-purpose — same vocabulary blindness `nomic-embed-text` had, just a better general model.
- **Cohere `embed-v4`** — multimodal, but that capability isn't needed here (documents are already OCR'd to text); priciest of the three considered, and the legacy `embed-v3` generation caps input at 512 tokens/chunk, a real constraint against this project's 2000-char parent chunks.
- **Stay on Ollama, defer the swap further** — rejected: the hosting blocker made "defer" no longer free the way it was during early development; every hosting option evaluated needed this resolved first.
- **Keep the stubbed `"anthropic"` provider option** — dropped as a swap path here. It pointed `AnthropicEmbeddings` at `claude-3-5-sonnet`, a chat model with no embeddings endpoint; that stub never actually worked and wasn't fixed as part of this change since Voyage covers the domain-tuned need directly.

## Consequences
- Removes the host-level Ollama dependency entirely — Docker Compose no longer needs `OLLAMA_BASE_URL`/`host.docker.internal` wiring, and every hosting option considered (VM, PaaS, Tailscale) becomes equally viable on the embeddings front.
- Trades free-and-local for a hosted API: at this project's document volume, Voyage's 50M-token free tier on `voyage-law-2` is expected to cover it indefinitely, but this is no longer a zero-external-dependency system.
- The 768→1024 dimension change means the existing Qdrant index is stale until re-extraction runs; nothing reads the old vectors as valid in the meantime since `recreate_collection()` replaces the whole collection on the next `/extract`.
- **Revisit trigger inherited from ADR-0003, now actually actionable**: re-run `tests/golden_set.py` against the rebuilt index and record hit rate/MRR/precision@k/latency next to the `nomic-embed-text` baseline. If the domain-tuned model doesn't measurably beat general-purpose on this document set, that's worth knowing — the reasoning above is a strong prior, not a substitute for the measurement ADR-0003 asked for.
- Supersedes [ADR-0003](0003-general-purpose-embedding-model.md).
