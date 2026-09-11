"""
Ad-hoc top-k experiment (not part of the pytest suite) -- reruns the
structured-fact golden-set questions at several top_k values against the
live API and records hit rate / MRR / precision / latency at each, so the
effect of top_k can be seen instead of guessed at (ADR-0005 explicitly
deferred picking a top_k value until this kind of measurement existed).

Usage: with the backend running, `python tests/topk_experiment.py` from backend/.
Writes tests/topk_results.json for the chart step to read.

NOTE: each (question, k) pair is a real Groq LLM call. Running this back to
back with the pytest suite and manual testing exhausted Groq's free-tier
daily token quota (200,000/day) mid-run once already -- a 429 here means
quota, not a retrieval bug. Check the error body before assuming otherwise.
"""
import json
import os
import statistics
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from golden_set import GOLDEN_SET
from test_quality import BASE_URL, _AUTH_HEADERS, precision_at_k, reciprocal_rank

K_VALUES = [1, 2, 3, 5, 10]
STRUCTURED_CASES = [c for c in GOLDEN_SET if c["type"] == "structured_fact"]


def run():
    results = {}
    out_path = os.path.join(os.path.dirname(__file__), "topk_results.json")
    try:
        for k in K_VALUES:
            per_case = []
            for case in STRUCTURED_CASES:
                start = time.time()
                resp = requests.post(
                    f"{BASE_URL}/query",
                    json={"question": case["question"], "top_k": k},
                    headers=_AUTH_HEADERS,
                    timeout=120,
                )
                elapsed = time.time() - start
                resp.raise_for_status()
                sources = resp.json().get("sources", [])
                hit = case["expected_policy"] in {s.get("policy_number") for s in sources}
                per_case.append({
                    "question": case["question"],
                    "hit": hit,
                    "rr": reciprocal_rank(sources, case["expected_policy"]),
                    "precision": precision_at_k(sources, case["expected_policy"]),
                    "latency": elapsed,
                })

            results[k] = {
                "hit_rate": sum(c["hit"] for c in per_case) / len(per_case),
                "mrr": statistics.mean(c["rr"] for c in per_case),
                "precision": statistics.mean(c["precision"] for c in per_case),
                "avg_latency": statistics.mean(c["latency"] for c in per_case),
                "per_case": per_case,
            }
            r = results[k]
            print(f"k={k:>2}: hit_rate={r['hit_rate']:.2f}  mrr={r['mrr']:.2f}  "
                  f"precision={r['precision']:.2f}  avg_latency={r['avg_latency']:.1f}s")
    finally:
        # Write whatever k-values completed even on a mid-run failure (e.g.
        # Groq daily quota, per this file's own docstring) -- a partial sweep
        # is still useful data; losing it entirely on the last k value's
        # failure would be a worse outcome than saving what's there.
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {out_path} ({len(results)}/{len(K_VALUES)} k-values completed)")


if __name__ == "__main__":
    run()
