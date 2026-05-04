"""FAISS-backed vector store utilities for embedding search.

This module wraps a FAISS inner product index with persistent storage of
associated texts and optional metadata filtering.
"""

import faiss
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class FAISSStore:
    """Store embeddings, texts, and metadata in a FAISS index."""

    def __init__(self, dim: int):
        """Initialize a FAISS index for inner product similarity search.

        Args:
            dim (int): Dimensionality of the embeddings being indexed.
        """
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add embeddings, texts, and optional metadata to the FAISS store.

        Args:
            embeddings: Iterable of embedding vectors.
            texts (List[str]): Corresponding text chunks.
            metadata (Optional[List[Dict[str, Any]]]): Optional metadata for each chunk.
        """
        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)

        self.index.add(vecs)
        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

    def search(self, embedding, k: int = 5, filter_by: Optional[Dict[str, Any]] = None):
        """Search for the top-k nearest chunks to a query embedding.

        Args:
            embedding: A single query embedding vector.
            k (int): Number of results to return.
            filter_by (Optional[Dict[str, Any]]): Metadata filter for result pruning.

        Returns:
            List[Dict[str, Any]]: Result dictionaries containing text, metadata, and score.
        """
        q = np.array([embedding], dtype="float32")
        faiss.normalize_L2(q)

        fetch_k = k * 4 if filter_by else k
        distances, indices = self.index.search(q, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue

            meta = self.metadata[idx]
            if filter_by and not _matches(meta, filter_by):
                continue

            results.append({"text": self.texts[idx], "metadata": meta, "score": float(dist)})
            if len(results) == k:
                break

        return results

    def save(self, folder: Optional[str] = None):
        """Persist the FAISS index and chunk metadata to disk.

        Args:
            folder (Optional[str]): Destination folder for the index and metadata.
        """
        folder = folder or f"vectorstore_{datetime.now().strftime('%Y-%m-%d')}"
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))

        with open(os.path.join(folder, "chunks.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)
        print(f"[FAISSStore] Saved {len(self.texts)} chunks → ./{folder}/")

    @classmethod
    def load(cls, folder: str, dim: int):
        """Load a saved FAISS index and associated metadata from disk.

        Args:
            folder (str): Folder containing ``index.faiss`` and ``chunks.pkl``.
            dim (int): Embedding dimension used to construct the index.

        Returns:
            FAISSStore: The loaded store instance.
        """
        store = cls(dim)
        store.index = faiss.read_index(os.path.join(folder, "index.faiss"))

        with open(os.path.join(folder, "chunks.pkl"), "rb") as f:
            data = pickle.load(f)

        if isinstance(data, list):
            store.texts = data
            store.metadata = [{} for _ in data]
        else:
            store.texts = data["texts"]
            store.metadata = data["metadata"]

        print(f"[FAISSStore] Loaded {len(store.texts)} chunks from ./{folder}/")
        return store


def _matches(meta: Dict[str, Any], filter_by: Dict[str, Any]) -> bool:
    """Determine whether metadata matches all filter criteria."""
    for key, wanted in filter_by.items():
        val = meta.get(key)
        if val is None:
            return False
        if isinstance(val, list):
            if wanted not in val:
                return False
        elif val != wanted:
            return False
    return True