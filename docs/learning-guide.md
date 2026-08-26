# RAG System — Learning Guide

A reference for the *field of options* behind each build phase — not just what this
project chose (that's what `docs/adr/` records), but the landscape of alternatives,
when each applies, and what "good" looks like for anything measurable. Grows one
milestone at a time as the project builds them; currently covers Milestones 1–5.

How to read this doc alongside the ADRs: **this file teaches the menu, the ADRs
record the order placed.** When a section below matches something this project
actually did, it says so and links the ADR — but most rows here are options this
project did *not* pick, included so the choice is a real one next time, not a default.

---

## Milestone 1 — Core Data Pipeline

### 1.1 Document text extraction

| Method | Best for | Trade-off |
|---|---|---|
| **pypdf** | Simple, well-formed PDFs; quick prototyping | Lightweight and dependency-free, but struggles with complex layouts, embedded/subset fonts, and multi-column text |
| **PyMuPDF (fitz)** | Complex real-world PDFs, custom/CID fonts | Faster and more robust than pypdf; still text-layer-only — can't read scanned images. **This project's choice** (`insurance_loader.py`), chosen specifically for CID-font handling |
| **pdfplumber** | Documents where table/layout structure matters | Best positional accuracy (tables, columns) but noticeably slower on large batches |
| **OCR** (pytesseract + pdf2image) | Scanned documents with no embedded text layer at all | Only option when there's no text layer to extract; adds real latency and error rate — never reach for it if a text layer exists |

**When to use which:** try the fast text-layer extractor first (PyMuPDF or pypdf); fall back to pdfplumber only if table structure is getting mangled; reach for OCR only when extraction returns empty text (a scanned image, not a real PDF).

### 1.2 Text normalization

What it is: cleaning extraction artifacts (layout whitespace, encoding glitches, boilerplate) *before* the text reaches chunking or an LLM prompt — noise here dilutes both embedding quality and the LLM's effective context window.

| Technique | Fixes |
|---|---|
| Whitespace collapsing | PDF layout produces lines that are empty except for spaces (column/table artifacts) |
| Replacement-character stripping (`U+FFFD`) | Fonts the extractor can't map (seen in this project with subset fonts encoding smart quotes/dashes) |
| Unicode normalization (NFC/NFKC) | Visually-identical characters with different byte representations (smart quotes, ligatures) |
| Boilerplate removal | Repeated headers/footers/page numbers across every page |

**This project's choice:** whitespace collapsing + `U+FFFD` stripping (`normalize_text()`, `insurance_loader.py`) — no boilerplate removal yet, since insurance policy PDFs don't carry the kind of repeated header/footer noise that would need it.

### 1.3 Metadata extraction approach

| Approach | When to use | Trade-off |
|---|---|---|
| **Rule-based / regex** | Documents follow a small number of fixed templates | Fast, free, deterministic — but breaks the moment a new document format appears |
| **LLM-based** | Documents vary in layout/vocabulary (real-world scanned/varied insurance paperwork) | Flexible across formats without per-template code, but costs a real API call per document and needs prompt engineering to stay reliable. **This project's choice** (`insurance_loader.py`, `extract_metadata_with_llm`) |
| **Hybrid** | High document volume where most follow known templates, a minority don't | Regex/template pass first (cheap, fast) with LLM fallback only for what doesn't match — the usual answer once volume makes LLM-per-document too slow/costly |

### 1.4 Chunking strategies

This is the decision with the widest range of real options — the "right" one depends
entirely on what a coherent unit of meaning looks like in *your* documents.

| Strategy | What it does | Best for | Trade-off |
|---|---|---|---|
| **Fixed-size (character/token count)** | Splits every N characters/tokens, no awareness of content | Quick baseline, uniform document types | Can split mid-sentence or mid-clause, breaking meaning at the boundary |
| **Recursive character splitting** | Tries paragraph → sentence → word boundaries in order, falling back only when a chunk is still too big | General-purpose default across most text | Still purely structural — doesn't know where an *idea* actually ends |
| **Sentence/paragraph-based** | Splits on natural language boundaries | Prose-heavy documents (articles, reports) | Chunk sizes become uneven, which can skew embedding quality across a collection |
| **Semantic chunking** | Uses embedding similarity between adjacent sentences to detect topic-shift boundaries | Long documents covering multiple distinct topics | Meaningfully better boundaries, but costs embedding calls just to *decide* chunk boundaries, before indexing even starts |
| **Document-structure-aware** | Splits on headers/sections (Markdown headers, HTML tags, detected PDF section titles) | Documents with reliable, consistent structural markup | Falls apart the moment source documents don't format structure consistently — true of most real-world scanned/varied PDFs |
| **Parent-child (small-to-big)** | Small chunks are embedded/searched for precision; each match returns its larger parent chunk for context | Precise retrieval matching + needing surrounding context in the answer | More moving parts (two chunk sizes to tune, not one) — but avoids the "matched a precise phrase, returned an isolated fragment with no context" failure. **This project's choice** (`vector_store.py`, ADR-0002), picked specifically because insurance documents don't format section structure consistently, ruling out the structure-aware option above |

**When to use which:** start with recursive character splitting as the default. Move to
parent-child specifically when precise phrase matching keeps returning fragments too
small to answer from. Reach for semantic chunking only once retrieval quality is the
proven bottleneck and simpler chunking has been ruled out — it's the most expensive
option on this list to build and run.

### 1.5 Error handling patterns

| Pattern | When to use |
|---|---|
| **Fail-fast** (one bad document stops the whole batch) | Small trusted batches where a bad document usually signals a real bug worth stopping for |
| **Per-document isolation** (one failure is recorded, the rest of the batch continues) | Real-world document sets of any size — one malformed PDF shouldn't block extraction for the other 8. **This project's choice** (`_error_record()`, `insurance_loader.py`) |

---

## Milestone 2 — Vector Search Foundation

### 2.1 Embedding models

| Type | Examples | When to use |
|---|---|---|
| **General-purpose open models** | `nomic-embed-text`, `all-MiniLM-L6-v2` | Free, self-hostable, no API cost per query — good default when there's no domain-specific vocabulary the model needs to understand. **This project's choice** (Ollama, ADR-0003), explicitly flagged there as "provisional" since it has no notion of insurance concepts |
| **Commercial API embeddings** | OpenAI `text-embedding-3`, Cohere Embed | Stronger general semantic quality, but a per-call cost and a network dependency for every embed |
| **Domain-tuned / fine-tuned** | A model fine-tuned on legal, medical, or financial text | Once general-purpose embeddings are the proven retrieval bottleneck — the single biggest lever on retrieval quality in the whole pipeline, bigger than chunk size or top_k |
| **Multilingual models** | `multilingual-e5`, LaBSE | Document sets spanning more than one language |

### 2.2 Vector databases

| Database | Deployment | Best for |
|---|---|---|
| **Qdrant** (on-disk local mode) | Embedded, single-process | Local development, small-to-medium collections, zero infra setup. **This project's choice** (`vector_store.py`) — with a real gotcha: on-disk mode locks its storage folder to **one process at a time**, confirmed directly this session when a second client and a live server collided |
| **Qdrant / Weaviate / Milvus** (server mode) | Self-hosted server or managed cloud | Production traffic, concurrent access from multiple processes — the direct fix for the single-process lock above |
| **Pinecone** | Fully managed cloud | Production without wanting to operate vector infra at all; usage-based cost |
| **Chroma** | Embedded or client-server | Similar niche to Qdrant's local mode — simple embedded option for small projects |
| **pgvector** | Postgres extension | Teams already running Postgres who want vectors alongside relational data in one system, avoiding a second database entirely |
| **FAISS** | In-process library, no server | Pure similarity search at large scale with no need for metadata filtering/CRUD — a library, not a database |

**When to move off on-disk local mode:** the moment more than one process needs the
index at the same time — a live API server plus a benchmark script, or any real
concurrent traffic, not just "the collection got big."

### 2.3 Distance metrics

| Metric | When it applies |
|---|---|
| **Cosine similarity** | Standard default for normalized text embeddings — measures direction, not magnitude. **This project's choice**, flagged in ADR-context as "rarely worth revisiting" |
| **Euclidean (L2)** | Embeddings where absolute magnitude carries meaning (less common for text) |
| **Dot product** | Equivalent to cosine when vectors are pre-normalized to unit length; a minor performance optimization in that case |

### 2.4 Retrieval strategies

| Strategy | What it does | When to use |
|---|---|---|
| **Plain top-k semantic search** | Embed the query, return the k nearest vectors | Default starting point for any RAG system |
| **Hybrid search** (BM25 + vector, reciprocal rank fusion) | Combines keyword matching with semantic similarity | Queries that name exact terms/codes a pure embedding model might blur (policy numbers, product SKUs, part numbers) |
| **Metadata filtering / entity-scoped retrieval** | Restrict the vector search to documents matching a known field (an ID, a name) | Query names a specific entity and unscoped search would dilute results with same-vocabulary unrelated documents. **This project's choice** (`find_relevant_source_files`, ADR-0004) — with a real discovered failure mode: it breaks when one entity (e.g. one insured party) has multiple documents, since the filter can't tell them apart on name alone |
| **Re-ranking** (cross-encoder) | Retrieve a wider candidate set with a fast method, then re-score the top N with a slower, more accurate model | Precision matters more than latency, and the retriever's fast similarity score is known to be an imperfect proxy for true relevance |

### 2.5 top_k tuning

**What it controls:** how many chunks come back per query. Not a free dial — it trades against three things at once:

- **Recall vs precision** — raising k almost never hurts recall (rarely makes the right answer *less* likely to be present), but this project's own measured data shows precision falling steadily as k rises (1.00 → 0.90 → 0.80 → 0.74 from k=1 to k=5) — extra slots fill with padding, not new relevant content.
- **Latency** — every extra chunk is more context the LLM has to read before answering; this project measured latency roughly tracking k, though not perfectly monotonically at small sample sizes.
- **Context budget** — enough chunks at large enough size can exceed a model's context window or a provider's per-request token limit. Concatenating chunks with no truncation step (this project's current state, an open decision — see the learning guide's Milestone 3 section) makes this risk grow silently with k.

**Optimal value:** there isn't a universal one — it must be tuned against a golden
evaluation set (real questions with known-correct answers), which is exactly what
Milestone 4 exists to build. Never pick a top_k from intuition and call it done.

---

## Milestone 3 — RAG Chain Implementation

### 3.1 LLM provider choices

| Option | When to use | Trade-off |
|---|---|---|
| **Cloud API** (Groq, OpenAI, Anthropic, Gemini) | Need strong quality without hosting infrastructure | Per-token cost, network latency, and — learned directly this session — a shared API key can hit a **daily token quota** under moderate testing load, not just heavy production traffic. **This project's choice** (Groq, `llm_client.py`), swappable to Anthropic in one place |
| **Self-hosted** (Ollama, vLLM) | Data privacy requirements, no per-token cost at scale, no external dependency | Requires real compute (GPU for anything beyond small models) and ongoing hosting/ops effort |

### 3.2 Prompt engineering patterns

| Pattern | When to use |
|---|---|
| **Zero-shot** | Simple, well-specified tasks the model already handles reliably |
| **Few-shot** (examples in the prompt) | Task has a specific expected format/style the model doesn't default to on its own — this project's metadata-extraction prompt uses this (policy-number format hints) |
| **System/user role separation** | Always — keeps persona/instructions separate from the actual content being processed |
| **Explicit grounding instruction** ("say so if it's not in the excerpts") | Any task where hallucination is worse than an honest "I don't know" — true of essentially all factual RAG. **This project's choice**, in both the metadata-extraction and the `/query` system prompt |
| **Structured-field + raw-excerpt hybrid context** | Some facts are more reliably captured once, structured (a policy number, a limit) than re-derived from raw text every query | This project's `/query` context assembly: a structured summary block plus raw excerpts, specifically because raw excerpts alone can surface an unrelated number (e.g. a deductible when the question asked about a coverage limit) |

### 3.3 Temperature & sampling parameters

| Parameter | Low value | High value |
|---|---|---|
| **Temperature** | Deterministic, consistent output — right for literal fact retrieval (policy numbers, dates, dollar amounts) where getting the *same* correct answer every time matters more than variety. **This project's choice: 0** (ADR-0001) | More varied/creative phrasing — right for brainstorming, drafting, open-ended generation, never for a task graded on exact correctness |
| **max_tokens** | Caps runaway generation, reduces cost | Too low silently truncates a genuinely long correct answer (this project flags its 500-token cap as untested against a realistic multi-policy comparison — worth verifying before trusting it) |

### 3.4 Context assembly strategies

| Strategy | When to use | Trade-off |
|---|---|---|
| **Simple concatenation** | Small top_k, small chunk sizes, low risk of exceeding context limits | Simplest to implement, but has no safety net — this project's current approach, with an explicitly open, unresolved risk (no token budget or truncation step) if `top_k` or chunk size ever grows |
| **Token-budgeted truncation** | Chunk count or size could plausibly grow past the model's context window | Adds real complexity (what gets cut when over budget?) but is the direct fix for the risk above |
| **Map-reduce / refine** | Context far too large for one prompt (many long documents) | Summarize/answer per-chunk, then combine — more LLM calls, more latency and cost, but the only option once one prompt can't hold everything |
| **Summarization-based compression** | Context relevant but verbose | Trades some fidelity for fitting more distinct sources into a fixed budget |

---

## Milestone 4 — Quality & Evaluation

### 4.1 Retrieval metrics

| Metric | Meaning | When to use | Ideal value |
|---|---|---|---|
| **Hit Rate@k** | Is the correct document present anywhere in the top-k results? | Coarsest retrieval signal — always compute this first | ≥ 90% on a small curated set; < 70% means retrieval is failing on cases designed to be easy |
| **MRR** (Mean Reciprocal Rank) | How high does the correct result rank, not just whether it's present | Whenever top_k might be lowered later — a hit buried at rank 5 is fragile in a way a rank-1 hit isn't | ≥ 0.8 (near 1.0 = always first); < 0.5 means correct results are typically buried |
| **Precision@k** | What fraction of the k returned results are actually relevant | Naturally noisy in most RAG setups (padding chunks aren't "wrong," just not the answer) — meaningful mainly when there's a scoping/filtering mechanism that *should* push it toward 1.0 | ~1.0 on entity-scoped questions; 0.2–0.5 is normal and expected on unscoped ones — this project's own measured run showed exactly that split |
| **Recall@k** *(not yet built in this project)* | Of all truly relevant documents, how many made it into the top-k | Multi-relevant-document scenarios (recommendation-style retrieval, not "there's exactly one right document") | Depends on how many relevant documents typically exist per query — not meaningful for this project's one-correct-document-per-question shape |

### 4.2 Answer-quality metrics

| Method | When to use | Trade-off |
|---|---|---|
| **Exact-match / containment** | Structured facts with one objectively correct value — a policy number, a dollar figure, a date | No partial credit, by design: a wrong dollar figure is a liability problem, not an approximation error, in a domain like insurance. **This project's choice** for structured-fact cases (`test_structured_fact_answer_correctness`) |
| **LLM-as-judge (faithfulness/groundedness)**, e.g. RAGAS-style | Open-ended questions with no single correct string ("what does this policy cover") | Scores whether every claim traces back to retrieved context — catches hallucination that exact-match can't even express a question about |
| **Traditional NLG metrics** (BLEU, ROUGE) | Rarely the right choice for RAG Q&A | Measure lexical/n-gram overlap with a reference answer, not factual correctness — a factually correct answer phrased differently scores low; a fluent, confidently wrong answer can score fine. Better suited to translation/summarization than fact-based Q&A |

### 4.3 Latency benchmarking

**Percentiles (p50/p95/p99), not average/min/max** — an average hides a bad tail: if 9
of 10 queries take 3s and one takes 40s, the average still looks fine while 1 in 10
users had a broken experience. p95/p99 are what actually describe "the slow case a
real user will eventually hit."

**Ideal vs. measured, and why both matter:** general UX guidance targets p50 < 3s,
p95 < 6s for a synchronous Q&A endpoint. This project's own first real measurement
came in far higher (p50 ≈ 13s, p95 ≈ 19–21s, dominated by the LLM call) — the lesson
isn't "the target was wrong," it's **set the regression-catching threshold from what
you actually measured, not from an aspirational number nobody has hit yet.** Tightening
the real number toward the aspirational one (shorter `max_tokens`, streaming, a faster
model) is separate follow-up work from "does the test suite work."

### 4.4 Test suite design

| Approach | Trade-off |
|---|---|
| **Manual script, human reads output** | Fast to write, but regressions require a human to notice — this project's original `perf_smoke_test.py` pattern |
| **Automated asserts (pytest)** | Pass/fail is unambiguous and repeatable — the direct upgrade path from the row above. **This project's choice** (`backend/tests/test_quality.py`) |
| **`xfail` for known, understood bugs** | A bug is real and tracked, but not yet fixed | Keeps the bug in the suite's active memory — if it ever silently starts passing (`XPASS`), that's the signal someone should promote it to a real case, rather than the fix going unnoticed. **This project's choice** for its two discovered bugs (ambiguous entity scoping; the unretrievable oversized-PDF document) |

### 4.5 A/B testing & feedback loops

| Component | When it's premature | When to build it | When to activate it |
|---|---|---|---|
| **A/B testing framework** | No real traffic and too small a golden set to compare variants meaningfully — a comparison on 5–9 documents is noise dressed up as a result | The *mechanism* (named variant configs, routing, logging) can be built ahead of use at near-zero risk if it defaults to today's exact behavior | Once there's real traffic or a golden set large enough per variant for the comparison to mean something. **This project's state:** mechanism built (`experiments.py`), activation explicitly deferred (ADR-0006/0007) |
| **User feedback loop** | Never premature to build the *capture* mechanism — it doesn't need real documents, just an endpoint and durable storage | Build it before real users arrive, not after — the whole point is catching real-world feedback from day one | The moment real users start asking questions. **This project's state:** built and verified (`/feedback`, `feedback_store.py`) |

---

## Milestone 5 — Advanced Features

Not started yet — this section surveys the field of options for each planned item
so the choice, once made, is deliberate rather than a default. No rows below say
"this project's choice" yet.

**Guiding principle for this milestone:** no real documents exist yet to drive these
decisions from measured data, and the system needs to handle a range of document
types and cases it hasn't seen. Default to the cheapest, most generic option for each
item (the leftmost/top row in each table below) and escalate to a more expensive
option only once a real, observed case proves the cheap one insufficient — the same
ladder this project already used for extraction (1.1) and chunking (1.4). The one
condition that makes this safe: **the cheap path must fail loudly, not silently.**
The oversized-PDF hallucination bug (Milestone 3 known gaps) is exactly what happens
when a cheap path fails quietly — the document silently didn't surface, and the LLM
filled the gap with a confident wrong answer instead of "I don't know." Every item
below needs a detectable-failure signal before it's trusted, not just a fallback plan.
See §5.6 for how that detection is actually built, layer by layer.

### 5.1 Multi-format document support

| Format | Extraction approach | Trade-off |
|---|---|---|
| **DOCX** | `python-docx`, or convert to PDF first and reuse the existing PyMuPDF path | Structured XML under the hood — generally more reliable than PDF text extraction, but tables/embedded objects need their own handling. **This project's choice** (`extract_text_from_docx`, `insurance_loader.py`) — paragraphs only for now, since no real DOCX document has surfaced table content yet to justify walking `doc.tables` too |
| **Images (scanned pages, photos of documents)** | OCR (pytesseract, or a cloud OCR API) | Same territory as the OCR fallback noted in 1.1 — real latency and error rate, and accuracy depends heavily on scan quality. **This project's choice** (`extract_text_from_image`, `insurance_loader.py`) — with a live example of the §5.6 discipline in practice: the `pytesseract` **pip package** is just a wrapper, the `tesseract` **binary** is a separate OS-level install this dev machine doesn't have. That gap now fails loudly (a distinct, actionable error record naming the missing binary) instead of silently returning "no text found" — see `tests/test_multi_format.py::test_missing_tesseract_binary_fails_loudly_not_silently`, and the real-OCR test that `skipif`s until the binary is actually present |
| **Plain text / Markdown / HTML** | Direct read, no extraction step needed | Easiest format to add — mainly a normalization/chunking question, not an extraction one. `.txt` already supported from Milestone 1 |
| **Format-dispatch layer** | Route by file extension/MIME type to the right extractor, normalize all outputs to one common text representation before it reaches chunking | The actual integration work — every format above still has to land in the same shape `insurance_loader.py` already produces, or chunking/metadata extraction need format-specific branches. **This project's choice**: an extension → extractor-function dict (`_TEXT_EXTRACTORS`, `insurance_loader.py`), checked via `_is_supported_extension()` before any I/O happens |

**When to use which:** DOCX first if real documents in that format exist — it's the
cheapest addition on top of the current pipeline. Reach for OCR only when a format
has no text layer at all, same rule as 1.1.

**Built alongside the dispatch layer, not after:** the file-scan step used to silently
filter to just `.txt`/`.pdf` — anything else vanished from `files` with no trace. That
was exactly the kind of quiet failure the Milestone 5 guiding principle warns against.
The dispatch layer now logs every unrecognized extension as a structured error record
(`_error_record`, same shape as every other extraction failure) instead of dropping it
— layer 1 of the graceful-degradation pattern in §5.6, built at the same time as the
feature itself rather than bolted on afterward.

### 5.2 Chat history & context carryover

| Approach | What it does | Trade-off |
|---|---|---|
| **Stateless (current state)** | Every `/query` is independent, no memory of prior turns | Simplest possible implementation, but can't answer "what about its deductible?" after a prior turn already named the policy |
| **Full history in prompt** | Prepend the raw conversation transcript to each new query | Simple to build, but grows the prompt (and cost/latency) with every turn, and eventually exceeds context budget |
| **Summarized history** | Periodically compress older turns into a running summary, keep only recent turns verbatim | Bounds prompt growth, at the cost of losing exact wording from summarized turns |
| **Query condensation (history-aware retrieval)** | Use the LLM to rewrite the latest user turn into a standalone query *before* retrieval, using history only for that rewrite, not the answer | Directly fixes retrieval for follow-up questions ("its deductible" → "Mount Royal University policy deductible") — this is the piece query rewriting (5.3) and chat history actually share |

**Dependency to note:** true multi-hop query rewriting (5.3) needs *some* form of
history to rewrite against — building history carryover first is the natural order,
matching the mentor take from the milestone kickoff discussion.

### 5.3 Query rewriting (multi-hop questions)

| Technique | What it does | When to use |
|---|---|---|
| **History-aware condensation** | Fold prior conversation turns into one standalone query before retrieval (see 5.2) | Follow-up questions that only make sense given earlier turns |
| **Query decomposition** | Split one complex question into multiple sub-queries, retrieve for each, combine results | "Compare the liability limits on policy A and policy B" — two separate retrievals, one synthesis step |
| **HyDE (Hypothetical Document Embeddings)** | Ask the LLM to draft a hypothetical answer first, embed *that* instead of the raw question, then retrieve | Short/vague queries whose embedding doesn't closely resemble the phrasing of the actual answer text |
| **Step-back prompting** | Ask a more general question first to retrieve broader context, then answer the specific question against it | Questions that need background context the specific phrasing wouldn't retrieve directly |

**Cost to weigh:** every technique above adds at least one extra LLM call before
retrieval even starts — worth it only once plain single-shot retrieval is a proven
bottleneck for real multi-hop questions, not a hypothetical one.

### 5.4 Hallucination detection

| Approach | What it catches | Trade-off |
|---|---|---|
| **Explicit grounding instruction only (current state)** | Relies entirely on the LLM following the "say so if it's not in the excerpts" instruction from 3.2 | Cheapest option, but this project already has a documented failure case where it doesn't hold — the oversized-PDF document (Milestone 3 known gaps) produces a confident wrong attribution instead of an honest "I don't know" |
| **LLM-as-judge / faithfulness scoring** (RAGAS-style, same tool noted in 4.2) | Post-hoc: scores whether each claim in the answer traces back to retrieved context | Reuses the 4.2 groundedness metric as a *runtime* check, not just an offline eval metric — an extra LLM call per query, so it adds real latency and cost |
| **Source-attribution verification** | Checks that every cited source/fact in the answer actually appears in the retrieved excerpts, via string/entity matching rather than another LLM call | Cheaper than LLM-as-judge, catches unsupported attributions specifically — the exact shape of the oversized-PDF bug — but won't catch subtler paraphrased hallucination |
| **Retrieval-confidence gating** | If retrieval similarity scores are all low (nothing genuinely relevant came back), refuse to answer instead of answering from weak context | Directly addresses the oversized-PDF case at its root cause — the document didn't surface in top-5 at all, so the LLM was answering from irrelevant context it should have flagged as insufficient |

**Where to start:** the project already has the reproduction case ready (`xfail` in
`backend/tests/golden_set.py`) — building the detector against a known failure and
watching that specific test flip from `xfail` to passing is a much stronger signal
than building blind against a hypothetical.

### 5.5 Confidence scoring

| Signal | What it measures | Trade-off |
|---|---|---|
| **Retrieval similarity score** | How close the top retrieved chunk(s) are to the query embedding | Already computed for free by the vector search (Qdrant returns it) — cheapest signal available, but a proxy for relevance, not for whether the *answer* is correct |
| **LLM self-reported confidence** | Ask the model to state a confidence level alongside its answer | No extra retrieval cost, but LLM-reported confidence is known to be poorly calibrated — models are often equally "confident" when right and wrong |
| **Agreement across multiple samples** (self-consistency) | Run the same query N times (or at nonzero temperature) and check how often answers agree | More reliable signal than self-reported confidence, but costs N× the LLM calls per query — expensive for a synchronous endpoint already measured at p50 ≈ 13s (4.3) |
| **Combined score (retrieval + groundedness)** | Blend the retrieval-similarity signal with the hallucination-detection groundedness score from 5.4 | The two signals catch different failure modes — retrieval confidence catches "nothing relevant was found," groundedness catches "something relevant was found but the answer strayed from it" — combining them covers more ground than either alone |

**Why this pairs with 5.4:** confidence scoring and hallucination detection are the
same underlying problem viewed from two angles — one answers "should this response
be trusted," the other answers "is this specific claim actually wrong." Building them
together, rather than in sequence, avoids computing overlapping signals twice.

### 5.6 How to build the "fails loudly" discipline — the professional pattern

The industry term for this is **graceful degradation with observability**: a cheap
path is allowed to fail, but the fact that it failed must be visible, queryable, and
eventually actionable. It has three layers, and a project this size shouldn't build
all three at once — over-building observability before there's real traffic to
observe is its own kind of premature work, the same trap avoided in 4.5's A/B
framework (mechanism built, activation deferred).

| Layer | What it is | When to build it | This project's state |
|---|---|---|---|
| **1. Structured failure logging** | Every fallback trigger writes a structured record: what failed, why, on which document/query, what path was taken instead | Now, alongside each Milestone 5 item — no new pattern needed, extend `_error_record()` (`insurance_loader.py`, §1.5) to the new fallback points | To build, in Milestone 5 |
| **2. A failure-rate metric/counter** | Aggregates the logs from layer 1 into a queryable rate per fallback type ("cheap-path X failed N% of queries this week") | Once queries/extractions happen regularly enough for a rate to mean anything — a JSONL log you can count is enough at this scale, no metrics server needed yet, same spirit as `feedback_store.py` | Deferred to Milestone 7 (see reminder in README roadmap) |
| **3. Full monitoring/alerting/dashboards** | Prometheus/Grafana-style live dashboards, paging on breached thresholds | Once there's real production traffic worth paging someone over | Deferred to Milestone 7 — it's that milestone's actual deliverable |

**Escalation stays manual, not automatic**, at every layer above: a detected failure
gets logged and (once layer 2 exists) counted, but nothing here should silently swap
in the expensive path on its own. A human (or the eval suite) decides whether the
failure rate justifies building the escalation — same discipline as ADR-0006/0007's
deliberately unactivated A/B framework. Revisit "should this auto-escalate" only
after layer 2 produces a real number to decide from, never from intuition.

---

*Next: Milestones 6–8 will be added here as they're built.*
