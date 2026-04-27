class Retriever:

    def __init__(self, embed_fn, vector_store):
        self.embed_fn = embed_fn
        self.vector_store = vector_store

    def retrieve(self, query, k=5):
        query_embedding = self.embed_fn([query])[0]
        results = self.vector_store.search(query_embedding, k=k)
        return results