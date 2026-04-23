import faiss
import numpy as np
import pickle
from config import FAISS_INDEX_PATH

class FAISSStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)

    def search(self, embedding, k=3):
        distances, indices = self.index.search(
            np.array([embedding]).astype("float32"), k
        )
        return [self.texts[i] for i in indices[0]]

    def save(self):
        faiss.write_index(self.index, f"{FAISS_INDEX_PATH}.index")
        with open(f"{FAISS_INDEX_PATH}.pkl", "wb") as f:
            pickle.dump(self.texts, f)