import os

# Embedding model
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Generative model (for user-friendly answers)
GEN_MODEL = os.getenv("GEN_MODEL", "google/flan-t5-base")

# Path to document data
DATA_PATH = os.getenv("DATA_PATH", "data/doc1.txt")

# Number of top results to retrieve from FAISS
TOP_K = int(os.getenv("TOP_K", 10))

# Maximum tokens to generate for answer
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
