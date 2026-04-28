"""Simple retriever wrapper that converts queries to embeddings and queries a vector store.

This module provides :class:`Retriever` which accepts an embedding function
and a vector store. It exposes a single :meth:`retrieve` method that returns
the top-k search results from the vector store for a given text query.
"""


class Retriever:
    """Wraps an embedding function and vector store to retrieve relevant chunks.

    Args:
        embed_fn (Callable[[List[str]], List[List[float]]]): A function that
            converts a list of strings into embeddings. The retriever expects
            the first (and only) element's embedding when called with a single
            query string in a list.
        vector_store: Object exposing a `search(embedding, k=...)` method which
            returns the top-k matching items for an embedding.
    """

    def __init__(self, embed_fn, vector_store):
        self.embed_fn = embed_fn
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 5):
        """Return the top-k results for a textual query.

        Args:
            query (str): The text query to embed and search for.
            k (int): Number of top results to return (default: 5).

        Returns:
            Iterable: The result of ``vector_store.search`` for the query
            embedding (typically a list of text chunks or scored hits).
        """
        query_embedding = self.embed_fn([query])[0]
        return self.vector_store.search(query_embedding, k=k)