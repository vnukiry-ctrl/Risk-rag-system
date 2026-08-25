# ADR-0004: Filter retrieval to named policy/insurer when the question names one

**Status:** Accepted
**Date:** 2026-08-25

## Context
A live test query — "What is the coverage limit and deductible on the Mount Royal University policy?" — returned 5 retrieved chunks: 1 from the Mount Royal University policy and 4 from unrelated policies (a garage automobile policy, two different CGL policies). Similarity scores were tightly clustered (0.71-0.76), meaning the embedding model was matching on generic insurance vocabulary ("coverage limit", "deductible") shared across all policies, not on the named entity in the question.

The LLM answered correctly in this instance because the structured metadata summary (see `/query` context assembly) still surfaced the right policy's numbers alongside the noise. But the retrieval itself was not scoped — a harder question, or a query where the structured summary doesn't disambiguate, would hand the LLM a context window mixing facts from multiple unrelated policies with no signal about which belongs to which.

## Decision
Before running semantic search, scan the question for a case-insensitive substring match against each known document's `policy_number`, `insured_name`, or `insurance_company` (length-gated at 4-6+ chars to avoid short-value false positives). If any documents match, restrict the vector search to just those documents via a Qdrant payload filter (`MatchAny` on `source_file`); otherwise search unscoped. Implemented as `find_relevant_source_files()` in [main.py](../../backend/main.py), filter support added to `semantic_search()` in [vector_store.py](../../backend/vector_store.py).

## Alternatives considered
- **Do nothing, rely on the structured summary to disambiguate** — this is what the system was doing; rejected because it only worked by luck in the test case, and provides no protection when the structured summary itself is ambiguous or missing (e.g. an unextracted document).
- **A dedicated NER/entity-linking step** — rejected as disproportionate for a document set this size; substring matching against already-known metadata fields is simpler and has no additional dependency.
- **Always filter to top-1 best-matching document** — rejected: an entity (e.g. a university) can legitimately have multiple relevant policies (CGL, accident, auto); collapsing to one document would under-retrieve for genuinely multi-policy questions.

## Consequences
- Questions naming a specific policy/insurer/insured now search only relevant documents, removing the cross-contamination risk observed in testing.
- Generic questions ("what policies do we have with a $5,000,000 limit") still search unscoped, since no entity name would match.
- False-positive risk: a short or generic company/insured name could substring-match incidentally inside an unrelated question. Not yet observed, but worth watching if the document set grows — the length gate (4-6 chars) is a coarse guard, not a precise one.
- This pattern (detect named entity → filter retrieval) generalizes to any RAG system where documents have identifying metadata and users often name the specific record they mean (a customer ID, a case number, a project name) — worth reapplying, with the entity fields re-derived per project.
