class Retreiver:
    def __init__(self, embedder, store):
        self.embedder = embedder
        self.store = store

    def retrieve(self, query):
        embedding = self.embedder.embed([query])[0]
        return self.store.search(embedding)