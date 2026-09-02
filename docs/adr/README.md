# Architecture Decision Records

This directory records *why* non-obvious choices in this project were made — not what the code does (the code shows that), but the reasoning, the alternative considered, and the trade-off accepted. Inline code comments tagged `DECISION (UNIVERSAL...)` / `DECISION (DOMAIN-SPECIFIC...)` point back here for the ones worth a fuller record.

Each ADR is a single choice, one file, numbered in decision order. Once written, an ADR is not edited to reflect new information — if a decision is reversed, write a new ADR that supersedes the old one and mark the old one's status accordingly. The record is a history, not a living doc.

## Template

```markdown
# ADR-000X: <short title of the decision>

**Status:** Proposed | Accepted | Deferred | Superseded by ADR-000Y
**Date:** YYYY-MM-DD

## Context
What situation forced this decision? What constraint, symptom, or requirement made the default/obvious choice not good enough?

## Decision
What was actually chosen. State it as a decision, not a description of the code.

## Alternatives considered
What else was on the table, and why it lost.

## Consequences
What this makes easier, what it makes harder, and what would need to be true to revisit it.
```

## Index

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-temperature-0-for-factual-qa.md) | LLM temperature = 0 for policy Q&A | Accepted |
| [0002](0002-parent-child-chunking-strategy.md) | Parent-child chunking, size-based splitting | Accepted |
| [0003](0003-general-purpose-embedding-model.md) | Ollama `nomic-embed-text`, general-purpose embeddings | Accepted (provisional) |
| [0004](0004-entity-scoped-retrieval-filtering.md) | Filter retrieval to named policy/insurer when the question names one | Accepted |
| [0005](0005-defer-top-k-and-context-budget-tuning.md) | Defer top-k and context-budget tuning | Deferred |
| [0006](0006-defer-ab-testing-framework.md) | Defer A/B testing framework | Deferred |
| [0007](0007-ab-mechanism-built-not-activated.md) | A/B testing mechanism built, not activated | Accepted (mechanism only) |
| [0008](0008-history-aware-query-condensation.md) | History-aware query condensation before retrieval | Accepted |
| [0009](0009-retrieval-confidence-gating.md) | Retrieval-confidence gating before answering | Accepted (threshold untuned) |
