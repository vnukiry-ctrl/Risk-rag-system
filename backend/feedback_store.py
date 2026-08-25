import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DECISION (UNIVERSAL): append-only JSONL file, not the in-memory documents_db
# pattern used elsewhere in this project. Feedback is the entire point of
# this feature -- losing it on every restart the way documents_db does would
# defeat it. A real database (Milestone 6) is future work; a durable local
# file is the right amount of durability for a project with no live traffic
# yet, without pulling in a DB dependency early.
FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", os.path.join(BASE_DIR, "feedback_log.jsonl"))


def record_feedback(
    question: str,
    answer: str,
    rating: str,
    comment: Optional[str] = None,
    sources: Optional[List[Dict]] = None,
) -> Dict:
    """Append one feedback entry. Returns the stored record (with timestamp).

    Sources are stored alongside the rating so a "down" vote can be traced
    back to which document/chunk produced the bad answer -- a rating with no
    way to trace the retrieval behind it isn't actionable, just a number.
    """
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "rating": rating,
        "comment": comment,
        "sources": sources or [],
    }
    with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Feedback recorded: rating=%s question=%r", rating, question)
    return entry


def read_feedback() -> List[Dict]:
    """Read all recorded feedback. Empty list if nothing's been recorded yet.

    IDEAL SITUATION this exists for: once there's enough real feedback, the
    "down"-rated entries here become new golden_set.py cases (a real wrong
    answer someone reported), and the "up"-rated ones are a sanity check that
    the golden set's questions still reflect what users actually ask.
    """
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return []
    with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
