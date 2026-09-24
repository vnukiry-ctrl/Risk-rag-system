"""
Golden evaluation set for retrieval + answer-quality testing (Milestone 4).

WHAT THIS IS: question -> known-correct-fact pairs, verified by hand against
the 16 real documents in backend/data/ (Mount Royal University's actual
insurance program) -- each fact below was cross-checked against the source
PDF's declarations page directly, not copied from the LLM's own extraction
output, since trusting the extractor's own answer as "ground truth" would be
circular and wouldn't catch extraction bugs.

REPLACES the previous version of this file, which was built against 9
placeholder/test documents (AVP406486, BW240599, ALCOG108, XLKR10271, the
oversized Group Accident PDF, etc.) that predated the real document set and
no longer exist in backend/data/ as of 2026-09-08. Two of those old policy
numbers (AVP406486, ALCOG108) happen to collide with real documents' own
policy numbers below -- coincidental, not a carried-over case.

IDEAL SITUATION: this file grows over time -- every new document type added,
every user-reported wrong answer, becomes one more case here. At 16
documents and 14 cases, this set can tell you "did something obviously
break" (a regression check), not "is retrieval good" in any statistically
trustworthy sense -- that needs dozens-to-hundreds of cases across a larger
document set, which ADR-0005 explicitly defers. Don't read strong
conclusions into these numbers yet; read them as a floor that shouldn't get
worse.

Case types (control how test_quality.py scores each one):
- "structured_fact": the answer must contain a specific, exact value
  (policy number, dollar figure, company name). No partial credit --
  matches the insurance-domain rule that a wrong dollar figure is a
  liability problem, not a rounding error.
- "known_limitation": a real, understood gap, documented here rather than
  silently dropped. Excluded from the scored tests (hit rate/MRR/precision/
  correctness all filter on type == "structured_fact") -- included so the
  gap stays visible in the file instead of being forgotten.
"""

GOLDEN_SET = [
    {
        "question": "What is the deductible for Employee Dishonesty coverage under "
        "crime protection policy SAA E738388 06?",
        "expected_policy": "SAA E738388 06",
        "type": "structured_fact",
        "expected_answer_contains": ["75,000", "75000"],
        "note": (
            "Worded with the literal policy number since 2026-09-14, not just "
            "'the Mount Royal University crime protection policy' -- see the "
            "medical-malpractice case below for why: insurance_type phrasing is "
            "not deterministic across LLM extraction runs, so a question that "
            "only matches on that phrase can silently stop scoping correctly the "
            "next time /extract runs, with no code change involved."
        ),
    },
    {
        "question": "What is the Third Party Liability limit under the garage "
        "automobile policy ALCOG108?",
        "expected_policy": "ALCOG108",
        "type": "structured_fact",
        "expected_answer_contains": ["2,000,000", "2000000"],
    },
    {
        "question": "What is the 2026-2027 renewal premium for automobile policy ALCOA108?",
        "expected_policy": "ALCOA108",
        "type": "structured_fact",
        "expected_answer_contains": ["12,940", "12940"],
        "note": (
            "ALCOA108 also covers a second real document -- the 2025-2026 21B "
            "adjustment endorsement (premium $1,152) -- which documents_db used to "
            "drop entirely (see the B0621FMOUN000426 known_limitation case below "
            "for why: dict-by-policy_number keying kept only the later-loaded "
            "document's metadata). This question was worded to land on the "
            "renewal, the doc that won the collision, so it read as a clean "
            "structured_fact rather than a second collision case. FIXED "
            "2026-09-17 (ADR-0012): documents_db is now keyed by source_file, so "
            "both documents keep their own metadata; find_relevant_source_files "
            "now resolves an unscoped question about ALCOA108 to whichever period "
            "is most recent, disclosing the other. Not yet re-verified against a "
            "fresh /extract of the real documents -- this question and its "
            "expected answer haven't been re-checked against that fix."
        ),
    },
    {
        "question": "What is the shared limit of insurance under the Definity "
        "D&O/EPL package policy 1000013292?",
        "expected_policy": "1000013292",
        "type": "structured_fact",
        "expected_answer_contains": ["2,000,000", "2000000"],
    },
    {
        "question": "What is the deductible per loss under the errors and omissions "
        "liability policy SRD677452?",
        "expected_policy": "SRD677452",
        "type": "structured_fact",
        "expected_answer_contains": ["5,000", "5000"],
    },
    {
        "question": "What is the limit of insurance per occurrence under the Fine "
        "Arts policy CAA0000558SP26A?",
        "expected_policy": "CAA0000558SP26A",
        "type": "structured_fact",
        "expected_answer_contains": ["10,340,731", "10340731"],
    },
    {
        "question": "What is the general aggregate limit under Commercial General "
        "Liability policy AVP406486 for the Mount Royal University User Group program?",
        "expected_policy": "AVP406486",
        "type": "structured_fact",
        "expected_answer_contains": ["5,000,000", "5000000"],
        "note": (
            "Same style of question as this file's original flagship "
            "entity-scoping case (ADR-0004/ADR-0004 follow-up) -- kept for "
            "continuity, now against the real AVP406486 document instead of the "
            "old placeholder pair (AVP406486 vs BW240599). Not yet re-verified "
            "whether the old near-miss ranking behavior recurs here; if MRR looks "
            "off on this case specifically, check for a semantically-close "
            "unscoped competitor the way the old note described. Policy number "
            "added to the question text 2026-09-14 for the same "
            "insurance_type-phrasing-isn't-deterministic reason as the crime "
            "policy case above -- this one hadn't broken yet, but relied on the "
            "same fragile match path."
        ),
    },
    {
        "question": "What is the limit of liability for the excess liability policy SPRGL2602187?",
        "expected_policy": "SPRGL2602187",
        "type": "structured_fact",
        "expected_answer_contains": ["40,000,000", "40000000"],
    },
    {
        "question": "What is the Public Liability limit per occurrence under the "
        "general liability policy SPRGL2602185?",
        "expected_policy": "SPRGL2602185",
        "type": "structured_fact",
        "expected_answer_contains": ["10,000,000", "10000000"],
    },
    {
        "question": "What is the annual premium for medical malpractice policy "
        "26/00008257/00 covering Mount Royal University?",
        "expected_policy": "26/00008257/00",
        "type": "structured_fact",
        "expected_answer_contains": ["11,812.50", "11812.50", "11,812", "11812"],
        "note": (
            "Worded with the literal policy number since 2026-09-14 -- a real, "
            "live-observed regression: this question originally read 'the medical "
            "malpractice policy covering Mount Royal University' and relied on "
            "find_relevant_source_files() matching insurance_type='Medical "
            "Malpractice' as a substring. A later /extract run's own re-extraction "
            "(non-deterministic LLM output, same document, no code change) "
            "produced insurance_type='Medical Professional Liability' instead -- "
            "a semantically equivalent but literally different string, which no "
            "longer matched. Scoping silently stopped triggering, and unscoped "
            "search returned the wrong document (Excess Side A D&O) as the top "
            "match, with the LLM then confidently answering from it instead of "
            "declining -- worse than a clean miss. Entity-scoped filtering "
            "(ADR-0004) is thus not just vulnerable to the already-documented "
            "policy-number key collisions, but to insurance_type phrasing drift "
            "across separate extraction runs of the *same* document. Naming the "
            "policy number directly sidesteps it for this question, but doesn't "
            "fix the underlying fragility -- see docs/adr/0004-entity-scoped-"
            "retrieval-filtering.md's Consequences for the open follow-up."
        ),
    },
    {
        "question": "What is the annual premium for the property policy underwritten "
        "by Factory Mutual Insurance Company?",
        "expected_policy": "1168007",
        "type": "structured_fact",
        "expected_answer_contains": ["341,802", "341802"],
    },
    {
        "question": "What is the limit of liability for the umbrella liability policy SPRGL2602186?",
        "expected_policy": "SPRGL2602186",
        "type": "structured_fact",
        "expected_answer_contains": ["10,000,000", "10000000"],
    },
    {
        "question": "What is the limit of liability for the Excess Side A D&O policy 01-142-91-44?",
        "expected_policy": None,
        "type": "known_limitation",
        "note": (
            "LLM metadata extraction returns every field as None for this "
            "document (verified 2026-09-08) -- its actual declarations (insurer "
            "AIG Insurance Company of Canada, limit $1,000,000 excess of a "
            "$2,000,000 underlying limit, premium $8,006) sit on page 17 of a "
            "44-page PDF, behind ~16 pages of AIG privacy-policy boilerplate that "
            "front the document; whatever the extractor reads never reaches the "
            "real content. UNLIKE the old Group Accident known-limitation case "
            "this replaces: retrieval is NOT broken here (this document ranks 1st "
            "for its own question, score ~0.82) and the LLM does NOT hallucinate "
            "-- live-verified, it correctly declines to state a primary limit of "
            "liability rather than confidently answering wrong. The only real gap "
            "is documents_db having no usable structured facts for this document "
            "(policy_number, coverage_limit, etc. all null), which blocks "
            "entity-scoped filtering (ADR-0004) from ever scoping onto it by its "
            "own policy number. Fix would be extracting from a later text window "
            "instead of always starting at the document's first page."
        ),
    },
    {
        "question": "What is the total premium for the Contingent Protective policy "
        "B0621FMOUN000426 across its full 3-year term?",
        "expected_policy": "B0621FMOUN000426",
        "type": "structured_fact",
        "expected_answer_contains": ["12,987", "12987"],
        "note": (
            "Was a known_limitation until 2026-09-21 -- documents_db used to be a "
            "dict keyed by policy_number (main.py's /extract handler); this "
            "policy number is shared by two real files -- the actual 3-year "
            "policy (2026-07-28 to 2029-07-01, total premium CAD $12,987 across "
            "three $4,329 installments) and its Year-1 installment invoice (MRU "
            "26-27 Contingent Protective (Year 1 of 3) Invoice 1011216.pdf). "
            "Whichever loaded later in os.listdir() order won the key and "
            "silently overwrote the other's metadata -- previously the invoice, "
            "leaving only its single $4,329 installment and a wrong period_to of "
            "2027 instead of 2029. FIXED 2026-09-17 (ADR-0012): documents_db is "
            "now keyed by source_file (both documents keep their own metadata), "
            "and find_relevant_source_files resolves an unscoped question about "
            "this policy number to the most recent period on file, disclosing "
            "the other. VERIFIED 2026-09-21 against a live /extract + /query run "
            "on the real documents: both files now keep separate metadata, and "
            "this question (naming no specific year) correctly resolves to the "
            "3-year policy and answers CAD 12,987 with a disclosure note about "
            "the Year-1-invoice version also on file -- not the invoice's "
            "$4,329. That same verification pass found and fixed a second real "
            "gap: parse_flexible_date() (insurance_loader.py) didn't handle "
            "ordinal day suffixes ('28th July 2026', '1st July 2029', the actual "
            "format this policy's dates use), which silently failed to parse "
            "the 3-year policy's period and would have made the *shorter* "
            "invoice period rank as 'latest' by falling back to extracted_date. "
            "A question-worded structured_fact case couldn't have caught that on "
            "its own, since expected_policy matches either family member "
            "(they share the same policy_number) -- expected_answer_contains' "
            "dollar-figure check is what actually distinguishes the two."
        ),
    },
]
