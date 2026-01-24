# app/index.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL, DATA_PATH

embed_model = SentenceTransformer(EMBED_MODEL)

def load_documents():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def build_index(documents):
    embeddings = embed_model.encode(
        documents,
        normalize_embeddings=True
    )

    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine similarity
    index.add(embeddings)

    return index, documents
