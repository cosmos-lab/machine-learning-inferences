import os

# Model Configuration
EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

GEN_MODEL = os.getenv(
    "GEN_MODEL",
    "google/flan-t5-base",
)

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", 3))

# Generation Configuration
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))
MIN_NEW_TOKENS = int(os.getenv("MIN_NEW_TOKENS", 20))

REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.1))
NO_REPEAT_NGRAM_SIZE = int(os.getenv("NO_REPEAT_NGRAM_SIZE", 3))

# Deterministic decoding for factual RAG
DO_SAMPLE = False
NUM_BEAMS = 1
TEMPERATURE = 0.0
EARLY_STOPPING = False
LENGTH_PENALTY = 1.2

# Data
DATA_PATH = os.getenv("DATA_PATH", "data/doc1.txt")

# Artifacts (MLOps)
ARTIFACT_DIR = "artifacts"
INDEX_DIR = f"{ARTIFACT_DIR}/index"
META_DIR = f"{ARTIFACT_DIR}/meta"

INDEX_PATH = f"{INDEX_DIR}/faiss.index"
META_PATH = f"{META_DIR}/index.json"
