"""Build and run the RAG pipeline for document and MITRE retrieval.

This module constructs the retrieval and generation pipeline, including
loading documents, chunking them, building FAISS stores, and running a
simple interactive command-line loop for user queries.
"""

import os
from config import DOCS_PATH, MITRE_INDEX_PATH, MITRE_JSON_PATH
from ingestion.loader import Loader
from ingestion.chunker import Chunker
from vectordb.faiss_store import FAISSStore
from models.embeddings import Embedder
from models.llm import LLM
from retrieval.bm25_retreiver import BM25Retriever
from retrieval.retreiver import Retriever
from retrieval.hybrid_retreiver import HybridRetriever
from rag.engine import RAGEngine
EMBEDDING_DIMENSION = 768

def get_embedding_function():
    """Return a closure that computes embeddings for a list of texts.

    Returns:
        Callable[[List[str]], List[List[float]]]: A function that accepts a list of
            strings and returns their embeddings using :class:`Embedder`.
    """
    embedder = Embedder()

    def embed_texts(texts):
        return embedder.embed(texts)
    return embed_texts

def build_document_store():
    """Build a FAISS store for document chunks from the docs folder.

    This function loads files from ``DOCS_PATH``, chunks each document, embeds
    the resulting text chunks, and indexes them in a FAISS store.

    Returns:
        tuple[FAISSStore, List[str]]: The FAISS store and the list of chunk texts.
    """
    loader = Loader()
    chunker = Chunker()
    documents = loader.load_documents(DOCS_PATH)
    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk(doc))
    store = FAISSStore(EMBEDDING_DIMENSION)
    embeddings = get_embedding_function()(chunks)
    store.add(embeddings, chunks)
    return (store, chunks)


def build_mitre_store():
    """Create or load a FAISS store for MITRE ATT&CK content.

    If a prebuilt index exists under ``MITRE_INDEX_PATH``, it is loaded.
    Otherwise, the function loads MITRE JSON data, embeds it in batches,
    saves the index to disk, and returns the new store.

    Returns:
        FAISSStore: The initialized MITRE FAISS store.
    """
    index_file = os.path.join(MITRE_INDEX_PATH, 'index.faiss')
    if os.path.exists(index_file):
        return FAISSStore.load(MITRE_INDEX_PATH, EMBEDDING_DIMENSION)
    loader = Loader()
    texts, metadata = loader.load_mitre(MITRE_JSON_PATH)
    store = FAISSStore(EMBEDDING_DIMENSION)
    batch_size = 64
    embed_fn = get_embedding_function()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadata = metadata[i:i + batch_size]
        batch_embeddings = embed_fn(batch_texts)
        store.add(batch_embeddings, batch_texts, batch_metadata)
    os.makedirs(MITRE_INDEX_PATH, exist_ok=True)
    store.save(MITRE_INDEX_PATH)
    return store


def build_rag_pipeline():
    """Construct the full RAG engine with document and MITRE retrieval.

    Returns:
        RAGEngine: The assembled retrieval-augmented generation engine.
    """
    document_store, document_chunks = build_document_store()
    mitre_store = build_mitre_store()
    llm = LLM()
    embed_fn = get_embedding_function()
    bm25_retriever = BM25Retriever(document_chunks)
    vector_retriever = Retriever(embed_fn, document_store)
    hybrid_retriever = HybridRetriever(bm25=bm25_retriever, vector=vector_retriever, mitre_vector=mitre_store)
    engine = RAGEngine(retriever=hybrid_retriever, llm=llm, embed_fn=embed_fn)
    return engine

def run_interactive_loop(engine):
    """Run a terminal-based interactive query loop.

    Args:
        engine: A RAG engine exposing an ``ask`` method for generating answers.
    """
    while True:
        try:
            query = input('Ask: ').strip()
            if query.lower() in ('exit', 'quit'):
                break
            if not query:
                continue
            answer = engine.ask(query)
            print(f'\n{answer}\n')
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f'Error: {e}')


if __name__ == '__main__':
    engine = build_rag_pipeline()
    run_interactive_loop(engine)