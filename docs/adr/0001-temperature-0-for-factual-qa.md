# ADR-0001: LLM temperature = 0 for policy Q&A

**Status:** Accepted
**Date:** 2026-08-25

## Context
`/query` answers factual questions about insurance policies — coverage limits, deductibles, dates, whether something is covered — by grounding an LLM call in retrieved document excerpts plus a structured metadata summary. The endpoint was initially built with `temperature=0.7`, Groq's convenience default rather than a value chosen for this task.

Temperature controls how sharply the model's next-token distribution is sampled: at 0 it always picks the top-scoring token (deterministic); as it rises, lower-probability tokens get a real chance of being picked, so identical prompts can produce differently-worded (or differently-scoped) answers run to run.

For this endpoint specifically, testing the same question multiple times at 0.7 showed two effects: superficial rewording (harmless) and volunteered detail not asked for — e.g. a run appending the deductible unprompted when only the coverage limit was asked — which is scope creep, not error, but inconsistent scope creep across identical questions.

## Decision
Set `temperature=0` for the `/query` answer-generation call ([main.py](../../backend/main.py)).

## Alternatives considered
- **0.3** — a common "light variation" middle ground for chat-style tasks. Rejected: there's no case for varied phrasing here (no user reads the same answer twice expecting a fresh take), and 0.3 still showed measurable answer-to-answer drift in testing.
- **0.7 (status quo)** — rejected as an unexamined default carried over from a generic API example, not chosen for this domain.

## Consequences
- Repeated identical questions now return identical answers, which matters for trust in a domain where the "same" answer worded two different ways looks like the system doesn't actually know the fact.
- Loses the (unneeded) natural-language variety a nonzero temperature gives; not a cost here since answers are meant to be read once, not as a conversational back-and-forth.
- Risk noted but not yet observed: strict greedy decoding (temp 0) can occasionally degenerate into repetitive phrasing on long generations. `max_tokens=500` keeps answers short enough that this hasn't shown up; revisit if answer length grows (e.g. multi-policy comparison responses).
- This value is domain-dependent, not universal — a generation or brainstorming task in a future project should not inherit 0 by default; re-derive from that task's actual failure mode the way this ADR did.
