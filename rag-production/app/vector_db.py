import faiss
import numpy as np

class VectorDB:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors):
        self.index.add(vectors.astype("float32"))

    def search(self, query, k: int):
        _, indices = self.index.search(query.astype("float32"), k)
        return indices[0]
