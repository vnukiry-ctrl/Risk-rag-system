# ADR-0002: Parent-child chunking, size-based splitting

**Status:** Accepted
**Date:** 2026-08-25 (originating in Milestone 1/2 work)

## Context
Insurance PDFs need to be split into pieces small enough to embed precisely (a query about "deductible" shouldn't match on similarity to an entire 10-page document) but large enough that a match still carries surrounding context (a deductible figure alone, with no clause around it, is easy to misattribute).

Real-world insurance documents in this project's `data/` folder don't use consistent section markup — no reliable headers, inconsistent formatting between insurers/brokers — so structure-aware splitting (by heading, by section number) wasn't viable without per-document custom parsing.

## Decision
Use size-based recursive character splitting at two levels ([vector_store.py](../../backend/vector_store.py)):
- Parent chunks: 2000 chars, 200 overlap — the unit returned to the LLM as context.
- Child chunks: 400 chars, 50 overlap — the unit actually embedded and searched.

Each child's payload carries a reference to its parent's full text; search matches on the child (precise), returns the parent (contextual) — "small-to-big" retrieval.

## Alternatives considered
- **Structure-aware splitting** (by section/heading) — rejected: this document set doesn't format section markup consistently enough to parse reliably; would need per-insurer/per-broker templates.
- **Single-level chunking** (no parent/child split) — rejected: forces a size trade-off with no good answer — small chunks embed precisely but return isolated fragments; large chunks return context but embed imprecisely (a big chunk's embedding is diluted across many topics).

## Consequences
- Works without any per-document format assumptions, at the cost of chunk boundaries sometimes falling mid-clause.
- The specific sizes (2000/400) are tuned to this document type's natural unit size (a policy clause), not derived from a general rule — they have not been validated against a golden eval set (see [ADR-0005](0005-defer-top-k-and-context-budget-tuning.md)).
- **Not portable as-is to a new project.** A different document type (code files, support tickets, contracts with reliable headers) has a different natural unit size and may support structure-aware splitting where this document set couldn't. Re-derive both the splitting strategy and the sizes from the new project's actual documents rather than reusing these numbers.
