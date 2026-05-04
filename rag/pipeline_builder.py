"""Shared pipeline building utilities for document and MITRE retrieval.

This module consolidates the construction logic for both the document FAISS store
and MITRE vector store to avoid duplication across app.py and streamlit_app.py.
"""

import os
from typing import Tuple, Optional
from config import DOCS_PATH, MITRE_INDEX_PATH, MITRE_JSON_PATH, FAISS_INDEX_PATH
from ingestion.loader import Loader
from ingestion.chunker import Chunker
from vectordb.faiss_store import FAISSStore
from models.embeddings import Embedder, get_embed_function
from models.llm import LLM
from retrieval.bm25_retreiver import BM25Retriever
from retrieval.retreiver import Retriever
from retrieval.hybrid_retreiver import HybridRetriever
from rag.engine import RAGEngine


def get_embedding_dimension() -> int:
    """Get the embedding dimension dynamically.
    
    Returns:
        int: Dimensionality of embeddings.
    """
    embedder = Embedder()
    return embedder.get_dimension()


def build_document_store() -> Tuple[FAISSStore, list]:
    """Build or load the document FAISS store.

    Checks if a saved document store exists. If it does, loads from disk.
    Otherwise, loads documents from DOCS_PATH, chunks them, embeds them,
    and saves the FAISS index.

    Returns:
        Tuple[FAISSStore, list]: The FAISS store and list of chunk texts.
    """
    embedding_dim = get_embedding_dimension()
    
    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        store = FAISSStore.load(FAISS_INDEX_PATH, embedding_dim)
        return (store, store.texts)

    # Build from scratch
    loader = Loader()
    chunker = Chunker()
    documents = loader.load_documents(DOCS_PATH)
    chunks = [chunk for doc in documents for chunk in chunker.chunk(doc)]
    
    store = FAISSStore(embedding_dim)
    embed_fn = get_embed_function()
    embeddings = embed_fn(chunks)
    store.add(embeddings, chunks)
    store.save(FAISS_INDEX_PATH)
    return (store, chunks)


def build_mitre_store(embedding_dim: Optional[int] = None) -> FAISSStore:
    """Build or load the MITRE FAISS store.

    If a prebuilt index exists, it is loaded. Otherwise, loads MITRE JSON,
    embeds it in batches, saves the index, and returns the store.

    Args:
        embedding_dim (Optional[int]): Embedding dimension. If None, computed dynamically.

    Returns:
        FAISSStore: The initialized MITRE FAISS store.
    """
    if embedding_dim is None:
        embedding_dim = get_embedding_dimension()
    
    index_file = os.path.join(MITRE_INDEX_PATH, 'index.faiss')
    if os.path.exists(index_file):
        return FAISSStore.load(MITRE_INDEX_PATH, embedding_dim)
    
    # Build from scratch
    loader = Loader()
    texts, metadata = loader.load_mitre(MITRE_JSON_PATH)
    store = FAISSStore(embedding_dim)
    
    embed_fn = get_embed_function()
    for i in range(0, len(texts), 64):
        batch_texts, batch_metadata = texts[i:i + 64], metadata[i:i + 64]
        store.add(embed_fn(batch_texts), batch_texts, batch_metadata)
    
    os.makedirs(MITRE_INDEX_PATH, exist_ok=True)
    store.save(MITRE_INDEX_PATH)
    return store


def build_rag_pipeline() -> RAGEngine:
    """Construct the full RAG engine with document and MITRE retrieval.

    Returns:
        RAGEngine: The assembled retrieval-augmented generation engine.
    """
    document_store, document_chunks = build_document_store()
    mitre_store = build_mitre_store()
    llm = LLM()
    embed_fn = get_embed_function()
    
    bm25_retriever = BM25Retriever(document_chunks)
    vector_retriever = Retriever(embed_fn, document_store)
    hybrid_retriever = HybridRetriever(bm25=bm25_retriever, vector=vector_retriever, mitre_vector=mitre_store)
    engine = RAGEngine(retriever=hybrid_retriever, llm=llm, embed_fn=embed_fn)
    return engine
