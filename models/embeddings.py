from langchain_ollama import OllamaEmbeddings

class Embedder:
    def __init__(self):
        # This matches the model you just pulled in Ollama
        self.model = OllamaEmbeddings(model="nomic-embed-text")

    def embed(self, texts):
        # OllamaEmbeddings uses embed_documents instead of encode
        return self.model.embed_documents(texts)