import threading
import faiss
from sentence_transformers import SentenceTransformer
from app.observability import logger


class Retriever:
    def __init__(self, model_name: str, top_k: int):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.index = None
        self.documents = []
        self._lock = threading.Lock()

    def build(self, documents: list[str]):
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        with self._lock:
            self.index = index
            self.documents = documents

        logger.info(
            "index_built",
            extra={"documents_indexed": len(documents)},
        )

    def load_index(self, index: faiss.Index, documents: list[str]):
        with self._lock:
            self.index = index
            self.documents = documents

    def retrieve(self, query: str) -> list[str]:
        with self._lock:
            if self.index is None:
                return []

            q = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

            _, ids = self.index.search(q, self.top_k)
            return [self.documents[i] for i in ids[0] if i < len(self.documents)]
