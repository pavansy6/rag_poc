"""BM25-based lexical retriever for short text chunks.

This module provides a thin wrapper around ``rank_bm25.BM25Okapi`` that
accepts a list of text chunks and returns the top-n matching chunks for a
given query string.
"""

from typing import List
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Lexical retriever using BM25 scoring.

    Args:
        chunks (List[str]): Pre-indexed text chunks to search over.
    """

    def __init__(self, chunks: List[str]):
        self.chunks = chunks
        # Simple whitespace tokenization on lowercased text for BM25 indexing.
        self.bm25 = BM25Okapi([chunk.lower().split() for chunk in chunks])

    def retrieve(self, query: str, k: int=3) -> List[str]:
        """Return the top-k chunks for a textual query using BM25.

        Args:
            query (str): The search query.
            k (int): Number of top chunks to return (default: 3).

        Returns:
            List[str]: The top matching text chunks.
        """
        return self.bm25.get_top_n(query.lower().split(), self.chunks, n=k)