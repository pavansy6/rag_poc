"""Streamlit application for the CISO RAG Assistant.

This module defines the UI, pipeline loading, and chat interaction flow for a
Streamlit-based assistant that answers cybersecurity and compliance questions.
The core pipeline building logic is shared via rag/pipeline_builder.py.
"""

import os
import pickle
import streamlit as st

from config import MITRE_INDEX_PATH
from rag.pipeline_builder import build_rag_pipeline
from retrieval.hybrid_retreiver import HybridRetriever


def _mitre_chunk_count():
    """Count saved MITRE chunks from the persisted FAISS store metadata."""
    pkl = os.path.join(MITRE_INDEX_PATH, "chunks.pkl")
    if not os.path.exists(pkl):
        return 0
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return len(data["texts"] if isinstance(data, dict) else data)


@st.cache_resource
def load_pipeline():
    """Load or build the retrieval pipeline used by the Streamlit application.

    Returns:
        tuple[RAGEngine, int, str]: The pipeline engine, total chunk count, and
            MITRE index status string.
    """
    rag = build_rag_pipeline()
    
    # Get chunk counts from the retriever
    doc_count = len(rag.retriever.bm25.chunks) if hasattr(rag.retriever.bm25, 'chunks') else 0
    
    # Determine MITRE status
    mitre_store = rag.retriever.mitre_vector
    if mitre_store is None:
        status = "enterprise-attack.json not found — MITRE disabled"
    elif os.path.exists(os.path.join(MITRE_INDEX_PATH, "index.faiss")):
        status = f"loaded ({_mitre_chunk_count()} chunks)"
    else:
        status = f"built ({len(mitre_store.texts)} chunks)"
    
    return rag, doc_count, status


# ── page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CISO RAG Assistant", layout="wide")
st.title("CISO RAG Assistant")
st.write("Ask cybersecurity and compliance-related questions.")

with st.spinner("Loading..."):
    rag, doc_count, mitre_status = load_pipeline()

mitre_ok = "not found" not in mitre_status and "disabled" not in mitre_status

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Index status")
    st.metric("Document Chunks", doc_count)
    st.metric("MITRE ATT&CK", "✓ Active" if mitre_ok else "✗ Disabled", delta=mitre_status)
    st.divider()
    st.caption("MITRE queries are detected automatically from keywords in your question.")
    
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── chat ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source_tag"):
            st.caption(msg["source_tag"])

if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Pass conversation history to the RAG engine for context awareness
            answer, route_info = rag.ask(query, return_route=True, conversation_history=st.session_state.messages[:-1])
            st.caption(f"Model used: {route_info['model']}")
        st.markdown(answer)

        is_mitre = any(s in query.lower() for s in HybridRetriever.MITRE_SIGNALS)
        source_tag = "Context includes MITRE ATT&CK data" if is_mitre and mitre_ok else "Context from internal documents"
        st.caption(source_tag)

    st.session_state.messages.append({"role": "assistant", "content": answer, "source_tag": source_tag})