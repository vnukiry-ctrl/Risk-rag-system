"""
Milestone 2 performance smoke test.

Runs a handful of known-answer questions against the running API,
measures /query latency, and checks the expected policy number shows
up in the returned sources. Not a substitute for the full k-tuning
eval (deferred until real golden dataset exists) -- just confirms
semantic search is fast and roughly accurate before calling retrieval
"done".

Usage: start the backend (uvicorn main:app), then run this script.
"""
import sys
import time
import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"

TEST_CASES = [
    {
        "question": "What is the coverage limit for the Mount Royal University Commercial General Liability policy?",
        "expected_policy": "AVP406486",
    },
    {
        "question": "What is the deductible on the medical professional liability policy?",
        "expected_policy": "25.00008257.00",
    },
    {
        "question": "What insurance company underwrites the garage automobile policy?",
        "expected_policy": "ALCOG108",
    },
    {
        "question": "What is covered under the Kidnap and Ransom policy?",
        "expected_policy": "XLKR10271",
    },
    {
        "question": "What is the premium for the Personal Accident policy at Mount Royal University?",
        "expected_policy": "BW240599",
    },
]


def run():
    latencies = []
    hits = 0

    for case in TEST_CASES:
        start = time.time()
        resp = requests.post(
            f"{BASE_URL}/query",
            json={"question": case["question"], "top_k": 5},
            timeout=60,
        )
        elapsed = time.time() - start
        latencies.append(elapsed)

        resp.raise_for_status()
        data = resp.json()
        source_policies = {s.get("policy_number") for s in data.get("sources", [])}
        hit = case["expected_policy"] in source_policies
        hits += hit

        print(f"[{'HIT' if hit else 'MISS'}] ({elapsed:.2f}s) {case['question']}")
        print(f"   expected policy: {case['expected_policy']} | retrieved: {source_policies}")
        print(f"   answer: {data['answer'][:150]}...")
        print()

    print("=" * 60)
    print(f"Retrieval hit rate: {hits}/{len(TEST_CASES)}")
    print(f"Latency  avg: {sum(latencies)/len(latencies):.2f}s  "
          f"min: {min(latencies):.2f}s  max: {max(latencies):.2f}s")


if __name__ == "__main__":
    run()
