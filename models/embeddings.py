from typing import List
from langchain_ollama import OllamaEmbeddings

class Embedder:

    def __init__(self, model_name: str='nomic-embed-text'):
        self.model = OllamaEmbeddings(model=model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.embed_documents(texts)
        return embeddings