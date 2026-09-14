import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_LOG_PATH = os.getenv("EXPERIMENTS_LOG_PATH", os.path.join(BASE_DIR, "experiments_log.jsonl"))

# DECISION (UNIVERSAL, deliberately not activated yet -- see ADR-0006 and its
# follow-up ADR-0007): this is the MECHANISM for A/B testing -- named variant
# configs, a way to route a request to one, and a log to compare them from --
# built now so turning on a real comparison later is a config change, not a
# rebuild. It is NOT turned on by default: /query's `variant` field defaults
# to None, which preserves today's exact behavior (top_k from the request,
# temperature=0). Don't read the presence of this file as "A/B testing is
# happening" -- it isn't, until there's real traffic or a golden set large
# enough to make a comparison meaningful (ADR-0006's blocker still applies).
#
# VERIFIED end-to-end 2026-09-14 (ADR-0007's "Update" section): POST /query with
# {"variant": "wider_retrieval"} confirmed chunks_searched reflected top_k=10
# and the call landed correctly in experiments_log.jsonl with variant/
# top_k/temperature all recorded. The mechanism itself works; this does NOT
# mean a real A/B comparison has been run -- that still needs real traffic
# or a golden set large enough to draw a conclusion from (ADR-0006's
# original blocker), which is a separate, still-open thing.
#
# "control"'s top_k updated 5 -> 2 (2026-09-14) to track main.py's
# QueryRequest default after ADR-0011's tuning -- this variant exists so a
# request can explicitly pin today's baseline behavior even if the default
# changes again later, so it needs to track the default, not fossilize the
# old one.
VARIANTS: Dict[str, Dict] = {
    "control": {"top_k": 2, "temperature": 0},
    "wider_retrieval": {"top_k": 10, "temperature": 0},
    "higher_temperature": {"top_k": 2, "temperature": 0.3},
}


def get_variant(name: Optional[str]) -> Optional[Dict]:
    """Look up a named variant's config. None (no variant requested) or an
    unknown name both return None, meaning: caller keeps its own defaults."""
    if not name:
        return None
    variant = VARIANTS.get(name)
    if variant is None:
        logger.warning("Unknown experiment variant requested: %r", name)
    return variant


def log_experiment_result(
    variant: Optional[str],
    question: str,
    top_k: int,
    temperature: float,
    sources: List[Dict],
    latency: float,
    session_id: Optional[str] = None,
    low_confidence: bool = False,
) -> None:
    """Record which variant served a request and what it returned.

    Logged even when variant is None (the "control"/default path), so a
    later analysis can compare "requests that opted into a variant" against
    the ordinary baseline traffic from the same time window.

    DECISION (UNIVERSAL, readiness for real data, no A/B or confidence-gate
    verification happening yet): `sources` used to be flattened down to just
    `source_files` here, silently dropping each chunk's retrieval score --
    the exact number ADR-0009's confidence gate and any future A/B precision
    comparison need. Kept as full `sources` dicts now, plus `low_confidence`
    and `session_id`, so once real traffic exists, a threshold or variant
    comparison can be done straight from this log instead of needing new
    instrumentation first.
    """
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "variant": variant,
        "session_id": session_id,
        "question": question,
        "top_k": top_k,
        "temperature": temperature,
        "sources": sources,
        "low_confidence": low_confidence,
        "latency": latency,
    }
    with open(EXPERIMENTS_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_experiment_results() -> List[Dict]:
    """Read all logged experiment results. Empty list if none logged yet."""
    if not os.path.exists(EXPERIMENTS_LOG_PATH):
        return []
    with open(EXPERIMENTS_LOG_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
