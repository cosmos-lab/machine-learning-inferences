# Used for thread-safe access to FAISS index & in-memory docs
import threading

# FAISS vector similarity search library (local in-memory vector DB)
import faiss

# Type hints for readability
from typing import List, Dict, Optional

# HuggingFace sentence embedding model
# Converts text → dense vector embeddings
from sentence_transformers import SentenceTransformer

# Factory that builds FAISS index from embeddings
from app.retrieval.faiss_factory import build_faiss_index

# Structured logger for observability
from app.observability.logger import logger


class Retriever:

    def __init__(self, model_name: str, top_k: int, enable_metadata: bool = True):

        # Load embedding model (e.g. all-MiniLM-L6-v2)
        # Used for converting documents & query → vectors
        self.model = SentenceTransformer(model_name)

        # Number of top similar results to return
        self.top_k = top_k

        # Enable metadata filtering during retrieval
        self.enable_metadata = enable_metadata

        # FAISS index (stores vectors only)
        self.index = None

        # In-memory document storage (FAISS does NOT store text)
        self.documents = []

        # In-memory metadata storage (FAISS does NOT store metadata)
        self.metadata = []

        # Lock for safe concurrent reads/writes in FastAPI
        self._lock = threading.Lock()

        # ------------------------------------------------------
        # 🚀 FUTURE VECTOR DB INSERTION POINT
        # Instead of FAISS you can initialize:
        #
        # self.vector_db = QdrantClient(...)
        # self.vector_db = MilvusClient(...)
        #
        # Then FAISS will not be needed.
        # ------------------------------------------------------


    def build(self, documents: list[str], metadata: list[dict] = None):

        # Convert documents → embeddings (vector representation)
        embeddings = self.model.encode(
            documents,
            batch_size=64,
            normalize_embeddings=True,   # cosine similarity support
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        # Build FAISS index from embeddings
        # FAISS stores ONLY vectors, not text or metadata
        index = build_faiss_index(embeddings)

        # Thread-safe write to retriever state
        with self._lock:

            # Store FAISS index in memory
            self.index = index

            # Store original documents separately
            # Needed because FAISS only returns vector IDs
            self.documents = documents

            # Store metadata separately
            self.metadata = metadata if metadata else []

        # ------------------------------------------------------
        # 🚀 VECTOR DB SHOULD BE ADDED HERE (INSTEAD OF ABOVE)
        #
        # Example:
        #
        # self.vector_db.upsert(
        #     ids=[str(i) for i in range(len(documents))],
        #     vectors=embeddings,
        #     payloads=metadata,
        #     documents=documents
        # )
        #
        # Then REMOVE:
        #   self.index
        #   self.documents
        #   self.metadata
        #
        # because Vector DB will store all of them internally.
        # ------------------------------------------------------

        # Log indexing information
        logger.info(
            "index_built",
            extra={
                "documents_indexed": len(documents),
                "index_type": type(index).__name__,
                "metadata_enabled": self.enable_metadata,
            },
        )


    def load_index(self, index: faiss.Index, documents: list[str], metadata: list[dict] = None):

        # Used when loading FAISS index from disk
        with self._lock:
            self.index = index
            self.documents = documents
            self.metadata = metadata if metadata else []


    def retrieve(self, query: str, filters: dict = None) -> list[str]:

        with self._lock:

            # If index not built yet
            if self.index is None:
                return []

            # Convert query → embedding vector
            q = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype("float32")

            # If metadata filters are used,
            # search more results initially then filter later
            k = self.top_k * 3 if filters and self.metadata else self.top_k

            # FAISS similarity search
            # returns distances + vector IDs
            _, ids = self.index.search(q, k)

            results = []

            # Loop over matched vector IDs
            for i in ids[0]:

                # Ignore invalid index
                if i >= len(self.documents):
                    continue

                # Apply metadata filtering if needed
                if filters and self.metadata and i < len(self.metadata):
                    if not self._matches_filters(self.metadata[i], filters):
                        continue

                # Map FAISS vector ID → original document
                results.append(self.documents[i])

                # Stop when enough results collected
                if len(results) >= self.top_k:
                    break

            return results

        # ------------------------------------------------------
        # 🚀 VECTOR DB VERSION SHOULD BE HERE
        #
        # results = self.vector_db.search(
        #     vector=q[0],
        #     top_k=self.top_k,
        #     filters=filters
        # )
        #
        # return [r.document for r in results]
        #
        # No need for:
        #   self.documents
        #   self.metadata
        #   manual filtering
        # ------------------------------------------------------


    def _matches_filters(self, metadata: dict, filters: dict) -> bool:
        """Check if metadata matches all filter criteria."""

        for key, value in filters.items():

            # Metadata key must exist
            if key not in metadata:
                return False

            meta_value = metadata[key]

            # Handle range / logical filters
            if isinstance(value, dict):

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
                # Direct equality filter
                if meta_value != value:
                    return False

        return True


