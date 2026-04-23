import faiss
import numpy as np
import pickle
import os
from datetime import datetime
from config import FAISS_INDEX_PATH

class FAISSStore:
    """
    A wrapper class for managing a local FAISS (Facebook AI Similarity Search) 
    vector database, used to store and retrieve dense vector embeddings efficiently.
    """
    def __init__(self, dim):
        """
        Initializes an empty FAISS L2 (Euclidean distance) index.
        
        Args:
            dim (int): The dimensionality of the vector embeddings to be added.
        """
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        """
        Adds vectors and their corresponding text chunks to the FAISS index and local store.
        
        Args:
            embeddings (numpy.ndarray): Dense vector embeddings generated from the text.
            texts (list[str]): The plain text chunks that align with the embeddings.
        """
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(texts)

    def search(self, embedding, k=3):
        """
        Performs a similarity search against the FAISS index using an input embedding.
        
        Args:
            embedding (numpy.ndarray): The vectorized query string.
            k (int, optional): The number of top-matching documents to return. Defaults to 3.
            
        Returns:
            list[str]: A list containing the top text chunks that match the query semantically.
        """
        distances, indices = self.index.search(
            np.array([embedding]).astype("float32"), k
        )
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
