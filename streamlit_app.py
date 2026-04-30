"""Streamlit application for the CISO RAG Assistant.

This module defines the UI, pipeline loading, and chat interaction flow for a
Streamlit-based assistant that answers cybersecurity and compliance questions.
"""

import os
import pickle
import streamlit as st

from config import DOCS_PATH, MITRE_INDEX_PATH, MITRE_JSON_PATH
from models.llm import LLM
from models.embeddings import Embedder
from ingestion.loader import Loader
from ingestion.chunker import Chunker
from vectordb.faiss_store import FAISSStore
from retrieval.retreiver import Retriever
from retrieval.bm25_retreiver import BM25Retriever
from retrieval.hybrid_retreiver import HybridRetriever
from rag.engine import RAGEngine
from rag.router import Router
from mitre_chunker import save_mitre_documents


def _mitre_exists():
    """Return True if the MITRE FAISS index already exists locally."""
    return os.path.exists(os.path.join(MITRE_INDEX_PATH, "index.faiss"))


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
    loader, chunker, embedder = Loader(), Chunker(), Embedder()

    # doc store
    chunks = []
    for doc in loader.load_documents(DOCS_PATH):
        chunks.extend(chunker.chunk(doc))
    
    doc_store = FAISSStore(len(embedder.embed(chunks[:1])[0]))
    doc_store.add(embedder.embed(chunks), chunks)

    # mitre store
    mitre_store, status = None, "enterprise-attack.json not found — MITRE disabled"

    if _mitre_exists():
        mitre_store = FAISSStore.load(MITRE_INDEX_PATH, doc_store.index.d)
        status = f"loaded ({_mitre_chunk_count()} chunks)"

    elif os.path.exists(MITRE_JSON_PATH):
        texts, meta = loader.load_mitre(MITRE_JSON_PATH)
        mitre_store = FAISSStore(doc_store.index.d)
        bar = st.sidebar.progress(0, text="Building MITRE index...")

        for i in range(0, len(texts), 64):
            mitre_store.add(embedder.embed(texts[i:i+64]), texts[i:i+64], meta[i:i+64])
            bar.progress(min((i + 64) / len(texts), 1.0), text=f"MITRE: {min(i+64, len(texts))}/{len(texts)}")

        mitre_store.save(MITRE_INDEX_PATH)
        save_mitre_documents(texts, meta, "mitre_chunks.json")
        status = f"built ({len(texts)} chunks)"

    embed_fn = embedder.embed

    retriever = HybridRetriever(bm25=BM25Retriever(chunks), vector=Retriever(embed_fn, doc_store), mitre_vector=mitre_store)

    llm = LLM()
    router = Router()

    return RAGEngine(retriever=retriever, llm=llm, router=router, embed_fn=embed_fn), len(chunks), status


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