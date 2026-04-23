class HybridRetriever:
    def __init__(self, bm25, vector):
        self.bm25 = bm25
        self.vector = vector

    def retrieve(self, query):
        bm25_results = self.bm25.retrieve(query, k=2)
        vector_results = self.vector.retrieve(query)

        combined = bm25_results + vector_results

        return list(dict.fromkeys(combined))