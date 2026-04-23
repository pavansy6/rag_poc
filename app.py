import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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

# Query loop
while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break
    print(rag.ask(q))