class Retreiver:
    """
    A semantic, vector-based retriever utilizing an embeddings model and a FAISS vector store.
    Note: Class name has a typo, intended to be "Retriever".
    """
    def __init__(self, embedder, store):
        """
        Initializes the standard vector retriever.
        
        Args:
            embedder: Initialization of the embedding model instance.
            store: Initialization of the FAISSStore instance.
        """
        self.embedder = embedder
        self.store = store

    def retrieve(self, query):
        """
        Retrieves relevant documents by converting the user's query into an embedding 
        and performing a similarity search in FAISS.
        
        Args:
            query (str): The plain text search string.
            
        Returns:
            list[str]: Semantic matches from the vector index.
        """
        embedding = self.embedder.embed([query])[0]
        return self.store.search(embedding)