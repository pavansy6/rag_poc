from rank_bm25 import BM25Okapi

class BM25Retriever:
    """
    A lexical, keyword-based retriever utilizing the BM25 algorithm for exact match scoring.
    """
    def __init__(self, chunks):
        """
        Initializes the BM25 engine by tokenizing the provided text chunks.
        
        Args:
            chunks (list[str]): The plain text chunks to be tokenized and indexed.
        """
        self.chunks = chunks
        tokenized_chunks = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def retrieve(self, query, k=3):
        """
        Retrieves the top k most relevant text chunks based on keyword matching.
        
        Args:
            query (str): The user's search query.
            k (int, optional): The maximum number of results to fetch. Defaults to 3.
            
        Returns:
            list[str]: The top lexical matches.
        """
        tokenized_query = query.lower().split()
        return self.bm25.get_top_n(tokenized_query, self.chunks, n=k)