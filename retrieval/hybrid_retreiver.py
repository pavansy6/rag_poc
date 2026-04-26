class HybridRetriever:
    """
    A hybrid retrieval strategy that combines results from both lexical (BM25) 
    and semantic (Vector) search algorithms to improve recall.
    """
    def __init__(self, bm25, vector):
        """
        Initializes the hybrid retriever.
        
        Args:
            bm25 (BM25Retriever): Instance of the BM25 lexical retriever.
            vector (Retreiver): Instance of the semantic vector retriever.
        """
        self.bm25 = bm25
        self.vector = vector

    def retrieve(self, query):
        """
        Fetches combined results from BM25 and Vector search, prioritizing varied 
        sources and deduplicating matches.
        
        Args:
            query (str): The search input.
            
        Returns:
            list[str]: A combined, deduplicated list of context chunks.
        """
        # Fetch top 2 results lexicographically
        bm25_results = self.bm25.retrieve(query, k=2)
        # Fetch the top 1 result semantically
        vector_results = self.vector.retrieve(query)[:1]

        # Combine results while retaining ordering and wiping out duplicates
        combined = list(dict.fromkeys(bm25_results + vector_results))
        return combined[:2]