import os

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL = os.getenv("GEN_MODEL", "google/flan-t5-base")

TOP_K = int(os.getenv("TOP_K", 3))

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))
MIN_NEW_TOKENS = int(os.getenv("MIN_NEW_TOKENS", 20))

REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.1))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", 3))

DO_SAMPLE = False
NUM_BEAMS = 1
TEMPERATURE = 0.0
EARLY_STOPPING = False
LENGTH_PENALTY = 1.2

DATA_PATH = os.getenv("DATA_PATH", "data/doc1.txt")

# Chunking settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 128))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "semantic")  # "semantic", "recursive", "sentence"

# Metadata settings
ENABLE_METADATA = os.getenv("ENABLE_METADATA", "true").lower() == "true"

ARTIFACT_DIR = "artifacts"
INDEX_DIR = f"{ARTIFACT_DIR}/index"
META_DIR = f"{ARTIFACT_DIR}/meta"

INDEX_PATH = f"{INDEX_DIR}/faiss.index"
META_PATH = f"{META_DIR}/index.json"
CHUNKS_PATH = f"{META_DIR}/chunks.json"

# Langfuse settings
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


