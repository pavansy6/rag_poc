"""Embedding utilities backed by Ollama embeddings.

This module provides embedding functions for converting text documents into numeric
vectors suitable for semantic search and FAISS indexing.
"""

from typing import List, Callable
from langchain_ollama import OllamaEmbeddings


class Embedder:
    """Wraps the Ollama embedding model for text embedding generation."""
    
    _instance = None

    def __new__(cls, model_name: str = 'nomic-embed-text'):
        """Implement singleton pattern to cache embedder instance."""
        if cls._instance is None:
            instance = super().__new__(cls)
            instance.model = OllamaEmbeddings(model=model_name)
            instance._embedding_dim = None
            cls._instance = instance
        return cls._instance

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings.

        Args:
            texts (List[str]): Texts to convert into embeddings.

        Returns:
            List[List[float]]: The computed embeddings for each input text.
        """
        return self.model.embed_documents(texts)
    
    def get_dimension(self) -> int:
        """Get embedding dimension by embedding a test string.
        
        Returns:
            int: Dimensionality of the embeddings.
        """
        if self._embedding_dim is None:
            self._embedding_dim = len(self.embed(["test"])[0])
        return self._embedding_dim


def get_embed_function() -> Callable[[List[str]], List[List[float]]]:
    """Get a cached embedding function.
    
    Returns:
        Callable: Function that embeds text using the cached Embedder singleton.
    """
    return Embedder().embed