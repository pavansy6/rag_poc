import faiss
import numpy as np
import pickle
import os
from datetime import datetime

class FAISSStore:
    def __init__(self, dim):
        # Switch from IndexFlatL2 to IndexFlatIP
        self.index = faiss.IndexFlatIP(dim)
        self.texts = []

    def add(self, embeddings, texts):
        # Convert to float32
        embeddings = np.array(embeddings).astype("float32")
        
        # CRITICAL: Normalize vectors for Inner Product to act as Cosine Similarity
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.texts.extend(texts)

    def search(self, embedding, k=5):
        # Convert query embedding to float32 and reshape
        embedding = np.array([embedding]).astype("float32")
        
        # CRITICAL: Normalize the query vector as well
        faiss.normalize_L2(embedding)
        
        distances, indices = self.index.search(embedding, k)
        return [self.texts[i] for i in indices[0]]

    def save(self):
        """
        Creates a timestamped directory (vectorstore_YYYY-MM-DD) and saves 
        the FAISS index and chunks inside it.
        """
        # Format: vectorstore_2026-04-23
        timestamp = datetime.now().strftime("%Y-%m-%d")
        folder_name = f"vectorstore_{timestamp}"
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        index_file_path = os.path.join(folder_name, "index.faiss")
        pkl_file_path = os.path.join(folder_name, "chunks.pkl")

        faiss.write_index(self.index, index_file_path)
        
        with open(pkl_file_path, "wb") as f:
            pickle.dump(self.texts, f)
