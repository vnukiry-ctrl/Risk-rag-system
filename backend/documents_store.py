import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# DECISION (UNIVERSAL): whole-dict JSON snapshot, not the append-only JSONL
# pattern feedback_store.py uses -- documents_db is a full replace on every
# /extract (not a stream of discrete events), so saving it as one JSON object
# and overwriting it each time matches how it's actually produced. Same
# durability tier as feedback_store.py: a local file is enough for a project
# with no live traffic yet, without pulling in a DB dependency early.
DOCUMENTS_STORE_PATH = os.getenv("DOCUMENTS_STORE_PATH", os.path.join(BASE_DIR, "documents_db.json"))


def save_documents(documents_db: Dict) -> None:
    """Persist the full documents_db snapshot, overwriting any previous save."""
    with open(DOCUMENTS_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(documents_db, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d document(s) to %s", len(documents_db), DOCUMENTS_STORE_PATH)


def load_documents() -> Dict:
    """Load the last saved documents_db snapshot. Empty dict if none exists yet
    -- a fresh checkout or a wiped file just means running /extract again,
    same as it would before this file existed."""
    if not os.path.exists(DOCUMENTS_STORE_PATH):
        return {}
    with open(DOCUMENTS_STORE_PATH, "r", encoding="utf-8") as f:
        documents_db = json.load(f)
    logger.info("Loaded %d document(s) from %s", len(documents_db), DOCUMENTS_STORE_PATH)
    return documents_db
