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
    embedder = Embedder()

    def embed_texts(texts):
        return embedder.embed(texts)
    return embed_texts

def build_document_store():
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