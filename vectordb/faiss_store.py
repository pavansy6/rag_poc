import faiss
import numpy as np
import pickle
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class FAISSStore:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)
        self.texts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def add(self, embeddings, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        vecs = np.array(embeddings, dtype="float32")
        faiss.normalize_L2(vecs)

        self.index.add(vecs)
        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

    def search(self, embedding, k: int = 5, filter_by: Optional[Dict[str, Any]] = None):
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

    def search_texts(self, embedding, k: int = 5):
        return [r["text"] for r in self.search(embedding, k=k)]

    def save(self, folder: Optional[str] = None):
        folder = folder or f"vectorstore_{datetime.now().strftime('%Y-%m-%d')}"
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, "index.faiss"))

        with open(os.path.join(folder, "chunks.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)
        print(f"[FAISSStore] Saved {len(self.texts)} chunks → ./{folder}/")

    @classmethod
    def load(cls, folder: str, dim: int):
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