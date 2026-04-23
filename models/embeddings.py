from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

class Embedder:
    """
    A utility class to transform text chunks into numerical vector embeddings using 
    HuggingFace's SentenceTransformer models.
    """
    def __init__(self):
        """Initializes the SentenceTransformer with the configured embedding model."""
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts):
        """
        Converts a list of text strings into a list of dense vector embeddings.
        
        Args:
            texts (list[str]): A list of text strings to embed.
        
        Returns:
            numpy.ndarray: An array of dense numerical vectors representing the text semantics.
        """
        return self.model.encode(texts)