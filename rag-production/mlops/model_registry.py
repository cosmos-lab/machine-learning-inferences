import json
import time
import os

REGISTRY_PATH = "models/manifest.json"

def register(embedding_model, generator_model, index_version):
    os.makedirs("models", exist_ok=True)
    record = {
        "embedding_model": embedding_model,
        "generator_model": generator_model,
        "faiss_index_version": index_version,
        "timestamp": time.time()
    }
    with open(REGISTRY_PATH, "w") as f:
        json.dump(record, f, indent=2)

def load():
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)
