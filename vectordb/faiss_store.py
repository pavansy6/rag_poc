import faiss
import numpy as np
import pickle
import os
from datetime import datetime


class FAISSStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []
        self.metadata = []          # parallel to self.texts — added for MITRE support

    def add(self, embeddings, texts, metadata=None):
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.texts.extend(texts)
        # if caller passes no metadata (e.g. your existing doc pipeline), fill with empty dicts
        if metadata is None:
            metadata = [{} for _ in texts]
        self.metadata.extend(metadata)

    def search(self, embedding, k=5, filter_by=None):
        """
        filter_by: optional dict for post-retrieval filtering, e.g.
            {"chunk_type": "technique", "platforms": "Windows"}
        Works for both string and list metadata values.
        """
        embedding = np.array([embedding]).astype("float32")
        faiss.normalize_L2(embedding)

        # over-fetch so we still get k results after filtering
        fetch_k = k * 4 if filter_by else k
        distances, indices = self.index.search(embedding, fetch_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            if filter_by and not _matches_filter(meta, filter_by):
                continue
            results.append({
                "text":     self.texts[idx],
                "metadata": meta,
                "score":    float(dist),
            })
            if len(results) == k:
                break

        return results

    def search_texts(self, embedding, k=5):
        """
        Convenience wrapper — returns plain list[str] like the old search().
        Keeps your existing retriever code working without changes.
        """
        return [r["text"] for r in self.search(embedding, k=k)]

    def save(self):
        timestamp = datetime.now().strftime("%Y-%m-%d")
        folder_name = f"vectorstore_{timestamp}"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        faiss.write_index(self.index, os.path.join(folder_name, "index.faiss"))

        with open(os.path.join(folder_name, "chunks.pkl"), "wb") as f:
            pickle.dump({"texts": self.texts, "metadata": self.metadata}, f)

        print(f"[FAISSStore] Saved {len(self.texts)} chunks → ./{folder_name}/")

    @classmethod
    def load(cls, folder_name, dim):
        store = cls(dim)
        store.index = faiss.read_index(os.path.join(folder_name, "index.faiss"))

        with open(os.path.join(folder_name, "chunks.pkl"), "rb") as f:
            data = pickle.load(f)

        # backwards compat: old saves stored texts as a plain list
        if isinstance(data, list):
            store.texts    = data
            store.metadata = [{} for _ in data]
        else:
            store.texts    = data["texts"]
            store.metadata = data["metadata"]

        print(f"[FAISSStore] Loaded {len(store.texts)} chunks from ./{folder_name}/")
        return store


def _matches_filter(meta: dict, filter_by: dict) -> bool:
    for key, value in filter_by.items():
        meta_val = meta.get(key)
        if meta_val is None:
            return False
        if isinstance(meta_val, list):
            if value not in meta_val:
                return False
        else:
            if meta_val != value:
                return False
    return True