"""
app.py — terminal entry point
Shows how to wire the MITRE store into your existing pipeline.
Your existing doc ingestion flow is completely unchanged.
"""

import os
from config import (
    DOCS_PATH, FAISS_INDEX_PATH, MITRE_INDEX_PATH,
    MITRE_JSON_PATH, CHUNK_SIZE, CHUNK_OVERLAP
)

from ingestion.loader  import Loader
from ingestion.chunker import Chunker

from vectordb.faiss_store import FAISSStore

from models.embeddings import get_embedding_model   # your existing embedding fn
from models.llm        import get_llm               # your existing LLM

from retrieval.bm25_retreiver  import BM25Retriever
from retrieval.retreiver       import Retreiver
from retrieval.hybrid_retreiver import HybridRetriever

from rag.engine import RAGEngine


# ── embedding dimension for your model ──────────────────────────────────────
# nomic-embed-text → 768. Change if you switch models.
EMBED_DIM = 768


def embed_fn(texts: list[str]) -> list[list[float]]:
    """
    Thin wrapper so any part of the code can call embed_fn(list[str]).
    Swap the internals for OpenAI / HuggingFace / Ollama as needed.
    """
    model = get_embedding_model()
    return [model.encode(t).tolist() for t in texts]


# ── 1. build or load the DOCS vector store (your existing flow) ──────────────
def get_doc_store() -> FAISSStore:
    loader  = Loader()
    chunker = Chunker()

    docs   = loader.load_documents(DOCS_PATH)
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc))

    store = FAISSStore(dim=EMBED_DIM)
    embeddings = embed_fn(chunks)
    store.add(embeddings, chunks)           # no metadata — plain doc chunks
    return store


# ── 2. build or load the MITRE vector store ──────────────────────────────────
def get_mitre_store() -> FAISSStore:
    # check if we already have a saved index so we don't re-embed every run
    index_file = os.path.join(MITRE_INDEX_PATH, "index.faiss")

    if os.path.exists(index_file):
        print("[app] Loading existing MITRE index...")
        return FAISSStore.load(MITRE_INDEX_PATH, dim=EMBED_DIM)

    print("[app] Building MITRE index from JSON (one-time, takes a few minutes)...")
    loader = Loader()
    texts, metadata = loader.load_mitre(MITRE_JSON_PATH)

    store = FAISSStore(dim=EMBED_DIM)

    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        batch_meta  = metadata[i: i + batch_size]
        batch_embs  = embed_fn(batch_texts)
        store.add(batch_embs, batch_texts, batch_meta)
        print(f"  embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    # save so next run is instant
    os.makedirs(MITRE_INDEX_PATH, exist_ok=True)

    import faiss, pickle
    faiss.write_index(store.index, os.path.join(MITRE_INDEX_PATH, "index.faiss"))
    with open(os.path.join(MITRE_INDEX_PATH, "chunks.pkl"), "wb") as f:
        import pickle
        pickle.dump({"texts": store.texts, "metadata": store.metadata}, f)

    print(f"[app] MITRE index saved → ./{MITRE_INDEX_PATH}/")
    return store


# ── 3. wire everything together ──────────────────────────────────────────────
def build_pipeline():
    doc_store   = get_doc_store()
    mitre_store = get_mitre_store()

    llm = get_llm()

    bm25_retriever   = BM25Retriever()          # your existing class
    vector_retriever = Retreiver(doc_store)      # your existing class

    hybrid = HybridRetriever(
        bm25   = bm25_retriever,
        vector = vector_retriever,
        mitre_vector = mitre_store,              # NEW — pass None to disable
    )

    engine = RAGEngine(
        retriever = hybrid,
        llm       = llm,
        embed_fn  = embed_fn,                   # needed for MITRE vector search
    )
    return engine


# ── 4. terminal loop ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building pipeline...")
    engine = build_pipeline()
    print("Ready. Type 'exit' to quit.\n")

    while True:
        query = input("Ask: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        answer = engine.ask(query)
        print(f"\n{answer}\n")