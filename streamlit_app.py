import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import DOCS_PATH
from models.llm import LLM
from models.embeddings import Embedder
from ingestion.loader import Loader
from ingestion.chunker import Chunker
from vectordb.faiss_store import FAISSStore
from retrieval.retreiver import Retreiver
from retrieval.bm25_retreiver import BM25Retriever
from retrieval.hybrid_retreiver import HybridRetriever
from rag.engine import RAGEngine


@st.cache_resource
def load_rag():
    # Load and process docs
    loader = Loader()
    chunker = Chunker()
    embedder = Embedder()

    raw_docs = loader.load_documents(DOCS_PATH)

    chunks = []
    for doc in raw_docs:
        chunks.extend(chunker.chunk(doc))

    embeddings = embedder.embed(chunks)

    store = FAISSStore(dim=len(embeddings[0]))
    store.add(embeddings, chunks)
    store.save()

    # retriever = Retreiver(embedder, store)
    retriever = BM25Retriever(chunks)

    # vector_retriever = Retreiver(embedder, store)
    # bm25_retriever = BM25Retriever(chunks)

    # retriever = HybridRetriever(bm25_retriever, vector_retriever)

    llm = LLM()
    rag = RAGEngine(retriever, llm)

    return rag


# UI
st.set_page_config(page_title="CISO RAG Assistant", layout="wide")

st.title("CISO RAG Assistant")
st.write("Ask cybersecurity/compliance-related questions.")

rag = load_rag()

# Session memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
query = st.chat_input("Ask a question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag.ask(query)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})