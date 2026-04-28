"""Embedding utilities backed by Ollama embeddings.

This module provides :class:`Embedder` for converting text documents into numeric
vectors suitable for semantic search and FAISS indexing.
"""

from typing import List
from langchain_ollama import OllamaEmbeddings


class Embedder:
    """Wraps the Ollama embedding model for text embedding generation."""

    def __init__(self, model_name: str = 'nomic-embed-text'):
        """Initialize the embedder with a chosen model name."""
        self.model = OllamaEmbeddings(model=model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of text strings.

        Args:
            texts (List[str]): Texts to convert into embeddings.

        Returns:
            List[List[float]]: The computed embeddings for each input text.
        """
        return self.model.embed_documents(texts)