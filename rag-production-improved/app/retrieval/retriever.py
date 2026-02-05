import threading
import faiss
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from app.retrieval.faiss_factory import build_faiss_index
from app.observability.logger import logger

class Retriever:
    def __init__(self, model_name: str, top_k: int, enable_metadata: bool = True):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k
        self.enable_metadata = enable_metadata
        self.index = None
        self.documents = []
        self.metadata = []
        self._lock = threading.Lock()

    def build(self, documents: list[str], metadata: list[dict] = None):
        embeddings = self.model.encode(
            documents,
            batch_size=64,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        index = build_faiss_index(embeddings)

        with self._lock:
            self.index = index
            self.documents = documents
            self.metadata = metadata if metadata else []

        logger.info(
            "index_built",
            extra={
                "documents_indexed": len(documents),
                "index_type": type(index).__name__,
                "metadata_enabled": self.enable_metadata,
            },
        )

    def load_index(self, index: faiss.Index, documents: list[str], metadata: list[dict] = None):
        with self._lock:
            self.index = index
            self.documents = documents
            self.metadata = metadata if metadata else []

    def retrieve(self, query: str, filters: dict = None) -> list[str]:
        with self._lock:
            if self.index is None:
                return []

            q = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

            # Get more results if filtering is needed
            k = self.top_k * 3 if filters and self.metadata else self.top_k
            _, ids = self.index.search(q, k)
            
            results = []
            for i in ids[0]:
                if i >= len(self.documents):
                    continue
                
                # Apply metadata filters if provided
                if filters and self.metadata and i < len(self.metadata):
                    if not self._matches_filters(self.metadata[i], filters):
                        continue
                
                results.append(self.documents[i])
                
                # Stop once we have enough results
                if len(results) >= self.top_k:
                    break
            
            return results

    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        """Check if metadata matches all filter criteria."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            meta_value = metadata[key]
            
            # Handle different filter types
            if isinstance(value, dict):
                # Range filters: {"$gte": 100, "$lte": 500}
                if "$gte" in value and meta_value < value["$gte"]:
                    return False
                if "$lte" in value and meta_value > value["$lte"]:
                    return False
                if "$gt" in value and meta_value <= value["$gt"]:
                    return False
                if "$lt" in value and meta_value < value["$lt"]:
                    return False
                if "$eq" in value and meta_value != value["$eq"]:
                    return False
                if "$ne" in value and meta_value == value["$ne"]:
                    return False
                if "$in" in value and meta_value not in value["$in"]:
                    return False
            else:
                # Direct equality
                if meta_value != value:
                    return False
        
        return True


