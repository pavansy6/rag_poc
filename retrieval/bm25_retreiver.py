from typing import List
from rank_bm25 import BM25Okapi

class BM25Retriever:

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query: str, k: int=3) -> List[str]:
        tokenized_query = query.lower().split()
        top_results = self.bm25.get_top_n(tokenized_query, self.chunks, n=k)
        return top_results