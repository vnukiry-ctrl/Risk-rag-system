import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session", autouse=True)
def require_running_backend():
    """Skip the whole suite with a clear reason if the API isn't reachable.

    Why this suite talks to the live HTTP API instead of importing
    vector_store functions directly: Qdrant's on-disk local mode locks its
    storage folder to a single process (confirmed directly while building
    this suite -- opening a second QdrantClient against the same path while
    the dev server was running raised a portalocker "AlreadyLocked" error,
    not a graceful "in use, try again"). Going through HTTP, like
    perf_smoke_test.py already does, sidesteps that entirely and exercises
    the real request path (retrieval + entity scoping + LLM generation)
    instead of just the retrieval layer in isolation.

    GOOD: a one-line skip telling the developer to start the backend first.
    NOT GOOD (what this fixture prevents): every test in the file failing
    with a raw ConnectionError traceback that reads like a broken test
    rather than an unmet precondition.
    """
    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        pytest.skip(
            f"Backend not reachable at {BASE_URL} -- start it first "
            f"(uvicorn main:app --reload, from backend/) before running this suite."
        )
