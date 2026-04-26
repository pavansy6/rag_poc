"""
Streamlit Web Interface for RAG Assistant

Changelog from old version:
- Wired in MITRE ATT&CK vector store alongside existing doc store
- HybridRetriever now includes MITRE as a 3rd source for attack-related queries
- RAGEngine receives embed_fn so MITRE search works at query time
- Added sidebar showing index stats (doc chunks, MITRE chunks)
- Added source badge on responses (MITRE hit vs docs only)
- Fixed SYSTEM_PROMPT bug: now uses .format() instead of broken f-string
- All caching unchanged — heavy models still load once
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import streamlit as st

from config import DOCS_PATH, MITRE_INDEX_PATH, MITRE_JSON_PATH
from models.llm import LLM
from models.embeddings import Embedder
from ingestion.loader import Loader
from ingestion.chunker import Chunker
from vectordb.faiss_store import FAISSStore
from retrieval.retreiver import Retreiver
from retrieval.bm25_retreiver import BM25Retriever
from retrieval.hybrid_retreiver import HybridRetriever
from rag.engine import RAGEngine


# ── helpers ──────────────────────────────────────────────────────────────────

def _mitre_index_exists() -> bool:
    return os.path.exists(os.path.join(MITRE_INDEX_PATH, "index.faiss"))


def _count_mitre_chunks() -> int:
    pkl = os.path.join(MITRE_INDEX_PATH, "chunks.pkl")
    if not os.path.exists(pkl):
        return 0
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return len(data["texts"]) if isinstance(data, dict) else len(data)


# ── cached pipeline ───────────────────────────────────────────────────────────

@st.cache_resource
def load_rag():
    loader   = Loader()
    chunker  = Chunker()
    embedder = Embedder()

    # ── doc store (unchanged from your original) ─────────────────────────
    raw_docs = loader.load_documents(DOCS_PATH)
    chunks   = []
    for doc in raw_docs:
        chunks.extend(chunker.chunk(doc))

    embeddings = embedder.embed(chunks)
    doc_store  = FAISSStore(dim=len(embeddings[0]))
    doc_store.add(embeddings, chunks)
    doc_store.save()

    # ── MITRE store ──────────────────────────────────────────────────────
    mitre_store = None
    mitre_status = "not loaded"

    if _mitre_index_exists():
        mitre_store  = FAISSStore.load(MITRE_INDEX_PATH, dim=len(embeddings[0]))
        mitre_status = f"loaded ({_count_mitre_chunks()} chunks)"
    elif os.path.exists(MITRE_JSON_PATH):
        mitre_status = "building…"
        texts, metadata = loader.load_mitre(MITRE_JSON_PATH)
        mitre_store = FAISSStore(dim=len(embeddings[0]))

        batch_size = 64
        progress = st.sidebar.progress(0, text="Building MITRE index…")
        for i in range(0, len(texts), batch_size):
            bt = texts[i: i + batch_size]
            bm = metadata[i: i + batch_size]
            be = embedder.embed(bt)
            mitre_store.add(be, bt, bm)
            progress.progress(
                min((i + batch_size) / len(texts), 1.0),
                text=f"MITRE: {min(i+batch_size, len(texts))}/{len(texts)}"
            )

        os.makedirs(MITRE_INDEX_PATH, exist_ok=True)
        import faiss
        faiss.write_index(
            mitre_store.index,
            os.path.join(MITRE_INDEX_PATH, "index.faiss")
        )
        with open(os.path.join(MITRE_INDEX_PATH, "chunks.pkl"), "wb") as f:
            pickle.dump(
                {"texts": mitre_store.texts, "metadata": mitre_store.metadata}, f
            )
        mitre_status = f"built ({len(texts)} chunks)"
    else:
        mitre_status = "enterprise-attack.json not found — MITRE disabled"

    # ── retrievers ───────────────────────────────────────────────────────
    bm25_retriever   = BM25Retriever(chunks)
    vector_retriever = Retreiver(embedder, doc_store)

    hybrid = HybridRetriever(
        bm25         = bm25_retriever,
        vector       = vector_retriever,
        mitre_vector = mitre_store,         # None if not available — hybrid handles it
    )

    # embed_fn adapter so RAGEngine / HybridRetriever can embed at query time
    def embed_fn(texts: list[str]) -> list[list[float]]:
        return embedder.embed(texts)

    llm = LLM()
    rag = RAGEngine(hybrid, llm, embed_fn=embed_fn)

    return rag, len(chunks), mitre_status


# ── page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="CISO RAG Assistant", layout="wide")
st.title("CISO RAG Assistant")
st.write("Ask cybersecurity and compliance-related questions.")

# ── load pipeline ─────────────────────────────────────────────────────────────

with st.spinner("Loading models and indexes…"):
    rag, doc_chunk_count, mitre_status = load_rag()

# ── sidebar — index stats ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("Index status")

    st.metric("Doc chunks", doc_chunk_count)

    mitre_ok = "not found" not in mitre_status and "disabled" not in mitre_status
    st.metric(
        "MITRE ATT&CK",
        "✓ active" if mitre_ok else "✗ disabled",
        delta=mitre_status,
        delta_color="normal" if mitre_ok else "off",
    )

    st.divider()
    st.caption("MITRE queries are detected automatically from keywords in your question.")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ── session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── chat history ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source_tag"):
            st.caption(msg["source_tag"])

# ── user input ────────────────────────────────────────────────────────────────

MITRE_SIGNALS = HybridRetriever.MITRE_SIGNALS   # reuse the same list

query = st.chat_input("Ask a question…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            answer = rag.ask(query)

        st.markdown(answer)

        # show a small source badge so the user knows where context came from
        is_mitre_query = any(s in query.lower() for s in MITRE_SIGNALS)
        source_tag = (
            "Context includes MITRE ATT&CK data"
            if is_mitre_query and mitre_ok
            else "Context from internal documents"
        )
        st.caption(source_tag)

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    answer,
        "source_tag": source_tag,
    })