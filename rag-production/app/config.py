import os

# Models
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
GEN_MODEL = os.getenv(
    "GEN_MODEL",
    "google/flan-t5-base",
)

# Retrieval
TOP_K = int(os.getenv("TOP_K", 5))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))

# Data
DATA_PATH = os.getenv("DATA_PATH", "data/doc1.txt")

# Artifacts (MLOps foundation)
ARTIFACT_DIR = "artifacts"
INDEX_DIR = f"{ARTIFACT_DIR}/index"
META_DIR = f"{ARTIFACT_DIR}/meta"

INDEX_PATH = f"{INDEX_DIR}/faiss.index"
META_PATH = f"{META_DIR}/index.json"
