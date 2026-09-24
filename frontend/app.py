import os
import streamlit as st
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Insurance RAG System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Overridable for Docker Compose, where "localhost" from inside the frontend
# container would mean the frontend container itself, not the backend one --
# compose sets this to the backend service's name (see docker-compose.yml).
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
# DECISION (UNIVERSAL, Milestone 6.1): matches backend/main.py's API_KEYS --
# the frontend is just another caller and authenticates the same way. Empty
# when unset, same as the backend's "no API_KEYS configured" dev-mode default,
# so a local backend running without auth still works with no extra setup.
API_KEY = os.getenv("API_KEY", "")
API_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.sidebar.title("🏢 Insurance RAG System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📄 Documents", "🔍 Search", "❓ Ask Questions"]
)

st.sidebar.markdown("---")
st.sidebar.info("RAG system for analyzing insurance documents")


if page == "🏠 Home":
    st.title("📋 Insurance Document RAG System")

    # DECISION (bug fix): these used to be hardcoded "0" placeholders that
    # never reflected real state. Documents/Policies now come from a live
    # /documents call; API Status reflects whether that call actually
    # succeeded rather than assuming the backend is up.
    try:
        docs_response = requests.get(f"{API_BASE_URL}/documents", headers=API_HEADERS, timeout=5)
        api_up = docs_response.status_code == 200
        documents = docs_response.json().get("documents", []) if api_up else []
    except Exception:
        api_up = False
        documents = []

    doc_count = len(documents)
    policy_count = len({d.get("policy_number") for d in documents if d.get("policy_number")})

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", doc_count)
    with col2:
        st.metric("Policies", policy_count)
    with col3:
        st.metric("API Status", "🟢 Running" if api_up else "🔴 Unreachable")

    st.markdown("---")
    st.subheader("Quick Start")

    if st.button("📥 Extract Documents", key="extract_home"):
        with st.spinner("Extracting..."):
            try:
                response = requests.post(f"{API_BASE_URL}/extract", headers=API_HEADERS, timeout=300)
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Extracted {data.get('total', 0)} documents "
                               f"({len(data.get('successful', []))} succeeded, {len(data.get('failed', []))} failed)")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")


elif page == "📄 Documents":
    st.title("📄 Extracted Documents")

    # DECISION: two tabs, two independent backend stores (documents_db vs
    # documents_db_professional) -- rendering both from the same helper below
    # so the "old way" (head_truncate) and "professional way"
    # (classify_then_target) results can be compared side by side on the
    # same document set instead of one overwriting the other.
    def render_documents_tab(list_endpoint: str, extract_endpoint: str, key_prefix: str):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            run_extract = st.button("📥 Run extraction", key=f"{key_prefix}_extract")
        with col3:
            if st.button("🔄 Refresh", key=f"{key_prefix}_refresh"):
                st.rerun()

        if run_extract:
            with st.spinner("Extracting... this calls the LLM once per document"):
                try:
                    resp = requests.post(f"{API_BASE_URL}{extract_endpoint}", headers=API_HEADERS, timeout=300)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Done: {len(data.get('successful', []))} succeeded, {len(data.get('failed', []))} failed")
                    else:
                        st.error(f"Error: {resp.status_code} - {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {str(e)}")

        try:
            response = requests.get(f"{API_BASE_URL}{list_endpoint}", headers=API_HEADERS)
            if response.status_code != 200:
                st.error(f"Error: {response.status_code}")
                return

            docs_data = response.json()
            total = docs_data.get("total", 0)
            documents = docs_data.get("documents", [])

            st.metric("Total Documents", total)

            if not documents:
                st.info("No documents yet. Click 'Run extraction' above.")
                return

            st.markdown("---")

            # DECISION (bug fix): every value here goes through str() -- the
            # extraction LLM has no enforced output type for numeric-looking
            # fields (premium_amount etc.), so one document can come back
            # with an int and another with a string for the same field. A
            # mixed-type column crashes st.dataframe()'s Arrow conversion for
            # the *entire* table (observed live: "Expected bytes, got a
            # 'int' object" on Premium), not just that cell, so this must be
            # normalized before pd.DataFrame() ever sees it.
            def _cell(value, default="N/A"):
                return default if value is None else str(value)

            table_rows = [
                {
                    "Document": _cell(doc.get("source_file"), "Unknown"),
                    "Policy #": _cell(doc.get("policy_number")),
                    "Type": _cell(doc.get("insurance_type")),
                    "Insurer": _cell(doc.get("insurance_company")),
                    "Broker": _cell(doc.get("broker")),
                    "Insured": _cell(doc.get("insured_name")),
                    "Period From": _cell(doc.get("period_from")),
                    "Period To": _cell(doc.get("period_to")),
                    "Premium": _cell(doc.get("premium_amount")),
                    "Coverage Limit": _cell(doc.get("coverage_limit")),
                    "Deductible": _cell(doc.get("deductible")),
                    "Chunks": doc.get("chunks_indexed", 0),
                    "Status": "⚠️ Error" if "error" in doc else "✅ Extracted",
                }
                for doc in documents
            ]
            st.dataframe(
                pd.DataFrame(table_rows),
                width="stretch",
                hide_index=True,
            )

            st.markdown("---")
            st.subheader("Document Details")
            for i, doc in enumerate(documents):
                with st.expander(f"📋 {doc.get('policy_number', 'Unknown')} - {doc.get('source_file', 'Unknown')}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Policy Information**")
                        st.write(f"Policy #: {doc.get('policy_number', 'N/A')}")
                        st.write(f"Type: {doc.get('insurance_type', 'N/A')}")
                        st.write(f"Company: {doc.get('insurance_company', 'N/A')}")
                        st.write(f"Broker: {doc.get('broker', 'N/A')}")
                    with col2:
                        st.write("**Coverage & Dates**")
                        st.write(f"From: {doc.get('period_from', 'N/A')}")
                        st.write(f"To: {doc.get('period_to', 'N/A')}")
                        st.write(f"Premium: ${doc.get('premium_amount', 'N/A')}")
                        st.write(f"Coverage Limit: {doc.get('coverage_limit', 'N/A')}")

                    st.write("**Insured**")
                    st.write(f"{doc.get('insured_name', 'N/A')}")
                    st.write(f"{doc.get('insured_address', 'N/A')}")

                    if doc.get('key_coverages'):
                        st.write("**Coverages**")
                        for coverage in doc.get('key_coverages', []):
                            st.write(f"• {coverage}")
        except Exception as e:
            st.error(f"Connection error: {str(e)}")

    tab_old, tab_pro = st.tabs(["🕰️ Old Way (head truncation)", "🧭 Professional Way (classify-then-target)"])
    with tab_old:
        render_documents_tab("/documents", "/extract", "old")
    with tab_pro:
        render_documents_tab("/documents/professional", "/extract/professional", "pro")


elif page == "🔍 Search":
    st.title("🔍 Search Documents")
    
    search_query = st.text_input("Search by keyword:")
    top_k = st.slider("Number of results", 1, 10, 5)
    
    if search_query and st.button("Search"):
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/search/metadata",
                    params={"query": search_query, "top_k": top_k},
                    headers=API_HEADERS,
                )
                if response.status_code == 200:
                    results = response.json().get("results", [])
                    st.success(f"Found {len(results)} results")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"Result {i}: {result.get('policy_number', 'Unknown')}"):
                            st.write(f"Source: {result.get('source')}")
                            st.write(f"Company: {result.get('insurance_company')}")
                            st.write(f"Preview: {result.get('preview')}")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


elif page == "❓ Ask Questions":
    st.title("❓ Ask Questions")

    # DECISION (UNIVERSAL, readiness for real data): session_id and the turn
    # history live in st.session_state so a conversation actually persists
    # across reruns/turns in the browser -- without this, every question hit
    # /query as a fresh session, condense_query() (ADR-0008) never had
    # history to fire on, and no real multi-turn usage data could ever be
    # collected. "New conversation" is the only way to reset it, matching
    # backend session semantics (an unrecognized/absent session_id just
    # starts a new one).
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "turns" not in st.session_state:
        st.session_state.turns = []  # list of dicts: question, answer, sources, session_id, variant, low_confidence, rated

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Session: {st.session_state.session_id or '(new conversation)'}")
    with col2:
        if st.button("🔄 New conversation"):
            st.session_state.session_id = None
            st.session_state.turns = []
            st.rerun()

    question = st.text_area("Ask about insurance documents:")
    # Default 2, not 5 -- measured via topk_experiment.py against the real
    # document set (see ADR-0005): recall/MRR plateau at k=2 and precision
    # is actually highest there, so a higher default would only add latency.
    top_k = st.slider("Documents to consider", 1, 10, 2)

    if st.button("Get Answer"):
        if not question:
            st.warning("Enter a question")
        else:
            with st.spinner("Searching and answering..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={
                            "question": question,
                            "top_k": top_k,
                            "session_id": st.session_state.session_id,
                        },
                        headers=API_HEADERS,
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state.session_id = result.get("session_id")
                        st.session_state.turns.append({
                            "question": question,
                            "answer": result.get("answer"),
                            "sources": result.get("sources") or [],
                            "session_id": result.get("session_id"),
                            "variant": None,
                            "low_confidence": result.get("low_confidence", False),
                            "rated": None,
                        })
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    st.markdown("---")

    # Most recent turn first, so the answer just given is immediately visible.
    for i, turn in enumerate(reversed(st.session_state.turns)):
        idx = len(st.session_state.turns) - 1 - i
        st.subheader(f"Q: {turn['question']}")
        if turn["low_confidence"]:
            st.warning("⚠️ Low confidence -- the system didn't find a close enough match to answer reliably.")
        st.info(turn["answer"])

        if turn["sources"]:
            st.markdown("**Sources**")
            for source in turn["sources"]:
                score = source.get("score")
                score_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
                st.write(
                    f"📄 {source.get('source_file', 'Unknown')} "
                    f"(policy {source.get('policy_number', 'N/A')}, score {score_str})"
                )

        # DECISION (UNIVERSAL, readiness for real data): this UI didn't exist
        # before -- feedback_log.jsonl would stay empty under real usage with
        # no way to record a rating. Buttons are disabled after one vote per
        # turn (feedback is meant to be a single up/down per answer, not a
        # counter) and pass session_id/variant/low_confidence through so a
        # later analysis can join a "down" vote back to what produced it.
        if turn["rated"]:
            st.caption(f"Feedback recorded: {turn['rated']}")
        else:
            fb_col1, fb_col2 = st.columns([1, 1])
            with fb_col1:
                if st.button("👍 Helpful", key=f"up_{idx}"):
                    try:
                        requests.post(
                            f"{API_BASE_URL}/feedback",
                            json={
                                "question": turn["question"],
                                "answer": turn["answer"],
                                "rating": "up",
                                "sources": turn["sources"],
                                "session_id": turn["session_id"],
                                "variant": turn["variant"],
                                "low_confidence": turn["low_confidence"],
                            },
                            headers=API_HEADERS,
                        )
                        st.session_state.turns[idx]["rated"] = "up"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            with fb_col2:
                if st.button("👎 Not helpful", key=f"down_{idx}"):
                    try:
                        requests.post(
                            f"{API_BASE_URL}/feedback",
                            json={
                                "question": turn["question"],
                                "answer": turn["answer"],
                                "rating": "down",
                                "sources": turn["sources"],
                                "session_id": turn["session_id"],
                                "variant": turn["variant"],
                                "low_confidence": turn["low_confidence"],
                            },
                            headers=API_HEADERS,
                        )
                        st.session_state.turns[idx]["rated"] = "down"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        st.markdown("---")


st.markdown("---")
st.markdown(f"Insurance RAG System | Backend: {API_BASE_URL}")