from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunks):
        self.chunks = chunks
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, k=3):
        tokenized_query = query.lower().split()
        return self.bm25.get_top_n(tokenized_query, self.chunks, n=k)