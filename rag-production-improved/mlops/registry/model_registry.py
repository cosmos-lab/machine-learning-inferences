import os
import json
import hashlib
from datetime import datetime
from app.config.settings import META_PATH, DATA_PATH
from app.observability.logger import logger

REGISTRY_FILE = "mlops/model_registry.json"

def hash_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

if __name__ == "__main__":
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found: {META_PATH}")

    with open(META_PATH, "r") as f:
        meta = json.load(f)

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "document_path": DATA_PATH,
        "embed_model": meta.get("embed_model"),
        "generator_model": meta.get("generator_model"),
        "artifact_hash": hash_file(META_PATH),
    }

    registry = []
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE, "r") as f:
            registry = json.load(f)

    registry.append(entry)

    os.makedirs(os.path.dirname(REGISTRY_FILE), exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

    logger.info("model_registry_updated", extra={"document_path": DATA_PATH})
    print(f"Model registry updated, entry added for {DATA_PATH}")


