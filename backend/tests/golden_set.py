"""
Golden evaluation set for retrieval + answer-quality testing (Milestone 4).

WHAT THIS IS: question -> known-correct-fact pairs, verified by hand against
the 9 real documents in backend/data/ via a live /extract + /query run
against this project's own API (not invented -- an invented "correct"
answer would be worse than no golden set at all, since it would silently
teach the suite to expect something false).

IDEAL SITUATION: this file grows over time -- every new document type added,
every user-reported wrong answer, becomes one more case here. At 9
documents and a handful of cases, this set can tell you "did something
obviously break" (a regression check), not "is retrieval good" in any
statistically trustworthy sense -- that needs dozens-to-hundreds of cases
across a representative document set, which ADR-0005 explicitly defers
until more real documents exist. Don't read strong conclusions into these
numbers yet; read them as a floor that shouldn't get worse.

Case types (control how test_quality.py scores each one):
- "structured_fact": the answer must contain a specific, exact value
  (policy number, dollar figure, company name). No partial credit --
  matches the insurance-domain rule that a wrong dollar figure is a
  liability problem, not a rounding error.
- "known_limitation": a case that currently FAILS for a real, understood
  reason (see each case's "note"). Marked xfail in the test suite rather
  than skipped or deleted, so the suite still runs it every time: if it
  ever starts passing on its own, that's a signal the fix landed and this
  case needs to be promoted to "structured_fact" and re-verified, not proof
  the bug quietly disappeared.
"""

GOLDEN_SET = [
    {
        "question": "What is the coverage limit for policy AVP406486?",
        "expected_policy": "AVP406486",
        "type": "structured_fact",
        "expected_answer_contains": ["5,000,000", "5000000"],
    },
    {
        "question": "What is the deductible on the medical professional liability policy?",
        "expected_policy": "25/00008257/00",
        "type": "structured_fact",
        "expected_answer_contains": ["50,000", "50000"],
    },
    {
        "question": "What insurance company underwrites the garage automobile policy?",
        "expected_policy": "ALCOG108",
        "type": "structured_fact",
        "expected_answer_contains": ["aviva"],
    },
    {
        "question": "What is the premium for the garage automobile policy?",
        "expected_policy": "ALCOG108",
        "type": "structured_fact",
        "expected_answer_contains": ["335"],
    },
    {
        "question": "What is the coverage limit for the Kidnap and Ransom policy?",
        "expected_policy": "XLKR10271",
        "type": "structured_fact",
        "expected_answer_contains": ["3,000,000", "3000000"],
    },
    {
        "question": "What is the coverage limit for the Mount Royal University Commercial General Liability policy?",
        "expected_policy": "AVP406486",
        "type": "known_limitation",
        "note": (
            "Entity-scoped filtering (ADR-0004) matches the question against "
            "insured_name substrings. 'Mount Royal University' is the insured "
            "on more than one policy (this CGL policy AND the BW240599 "
            "Personal Accident policy), so the scope filter picks the wrong "
            "single document instead of narrowing correctly -- confirmed live: "
            "this exact question currently returns only BW240599 chunks and an "
            "empty answer. Rephrasing to name the policy number directly "
            "('policy AVP406486') works today; the ambiguous-entity case does "
            "not. Fix belongs in find_relevant_source_files() (main.py) -- "
            "disambiguate by insurance_type too, not insured_name alone."
        ),
    },
    {
        "question": "What does the Group Accident insurance policy cover?",
        "expected_policy": None,
        "type": "known_limitation",
        "note": (
            "This is the oversized-PDF document already flagged in the README "
            "known gaps (25-26 Group Accident Policy 100013386.pdf) as failing "
            "LLM metadata extraction. Confirmed live that the gap is worse than "
            "documented: its chunks don't surface in the top-5 semantic search "
            "results for an on-topic question about its own content, AND the "
            "LLM answers confidently anyway by misattributing the coverage to "
            "an unrelated policy (BW240599) instead of stating it doesn't know. "
            "That second part -- confident wrong answer instead of a stated "
            "'not found' -- is the actual quality risk an insurance Q&A system "
            "cannot have; worth its own investigation, not just a retrieval fix."
        ),
    },
]
