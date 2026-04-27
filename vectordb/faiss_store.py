import faiss
import numpy as np
import pickle
import os
from datetime import datetime

class FAISSStore:

    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []
        self.metadata = []

    def add(self, embeddings, texts, metadata=None):
        vecs = np.array(embeddings).astype('float32')
        faiss.normalize_L2(vecs)
        
        self.index.add(vecs)
        self.texts.extend(texts)
        self.metadata.extend(metadata or [{} for _ in texts])

    def search(self, embedding, k=5, filter_by=None):
        
        q = np.array([embedding]).astype('float32')
        faiss.normalize_L2(q)
        
        distances, indices = self.index.search(q, k * 4 if filter_by else k)
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
        
            if idx == -1:
                continue
            meta = self.metadata[idx]
        
            if filter_by and (not _matches(meta, filter_by)):
                continue
            results.append({'text': self.texts[idx], 'metadata': meta, 'score': float(dist)})
        
            if len(results) == k:
                break
        return results

    def search_texts(self, embedding, k=5):
        
        return [r['text'] for r in self.search(embedding, k=k)]

    def save(self, folder=None):
        folder = folder or f"vectorstore_{datetime.now().strftime('%Y-%m-%d')}"
        os.makedirs(folder, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder, 'index.faiss'))
        
        with open(os.path.join(folder, 'chunks.pkl'), 'wb') as f:
            pickle.dump({'texts': self.texts, 'metadata': self.metadata}, f)
        print(f'[FAISSStore] Saved {len(self.texts)} chunks → ./{folder}/')

    @classmethod
    def load(cls, folder, dim):
        store = cls(dim)
        store.index = faiss.read_index(os.path.join(folder, 'index.faiss'))
        
        with open(os.path.join(folder, 'chunks.pkl'), 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            store.texts, store.metadata = (data, [{} for _ in data])
        else:
            store.texts, store.metadata = (data['texts'], data['metadata'])
        
        print(f'[FAISSStore] Loaded {len(store.texts)} chunks from ./{folder}/')
        return store

def _matches(meta, filter_by):
    
    for k, v in filter_by.items():
        val = meta.get(k)
        
        if val is None:
            return False
        if isinstance(val, list):
            if v not in val:
                return False
        elif val != v:
            return False
    
    return True