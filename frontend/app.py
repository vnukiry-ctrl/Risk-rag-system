import streamlit as st
import requests
import pandas as pd
from datetime import datetime

st.set_page_config(
    page_title="Insurance RAG System",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE_URL = "http://localhost:8000"

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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents", "0")
    with col2:
        st.metric("Policies", "0")
    with col3:
        st.metric("API Status", "🟢 Running")
    
    st.markdown("---")
    st.subheader("Quick Start")
    
    if st.button("📥 Extract Documents", key="extract_home"):
        with st.spinner("Extracting..."):
            try:
                response = requests.post(f"{API_BASE_URL}/extract")
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ Extracted {len(data)} documents!")
                else:
                    st.error(f"Error: {response.status_code}")
            except Exception as e:
                st.error(f"Connection error: {str(e)}")


elif page == "📄 Documents":
    st.title("📄 Extracted Documents")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Document List")
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        if response.status_code == 200:
            docs_data = response.json()
            total = docs_data.get("total", 0)
            documents = docs_data.get("documents", [])
            
            st.metric("Total Documents", total)
            
            if documents:
                st.markdown("---")
                for doc in documents:
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
            else:
                st.info("No documents yet. Extract documents on Home page.")
        else:
            st.error(f"Error: {response.status_code}")
    except Exception as e:
        st.error(f"Connection error: {str(e)}")


elif page == "🔍 Search":
    st.title("🔍 Search Documents")
    
    search_query = st.text_input("Search by keyword:")
    top_k = st.slider("Number of results", 1, 10, 5)
    
    if search_query and st.button("Search"):
        with st.spinner("Searching..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/search",
                    params={"query": search_query, "top_k": top_k}
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
    
    question = st.text_area("Ask about insurance documents:")
    top_k = st.slider("Documents to consider", 1, 10, 5)
    
    if st.button("Get Answer"):
        if not question:
            st.warning("Enter a question")
        else:
            with st.spinner("Searching and answering..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query",
                        json={"question": question, "top_k": top_k}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.subheader("Answer")
                        st.info(result.get("answer"))
                        if result.get("sources"):
                            st.subheader("Sources")
                            for source in result.get("sources"):
                                st.write(f"📄 {source}")
                    else:
                        st.error(f"Error: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


st.markdown("---")
st.markdown("Insurance RAG System | Backend: http://localhost:8000")