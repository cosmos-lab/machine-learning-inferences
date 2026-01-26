"""
Purpose:
Creates a snapshot of the RAG system state for observability and auditing.
Captures index size, embedding dimensions, top-K, and model versions for monitoring.
"""

import json
import os
from datetime import datetime
from app.pipeline import RAGPipeline
from app.config import DATA_PATH, META_PATH
from app.observability import logger

SNAPSHOT_DIR = "mlops/snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load_from_file(DATA_PATH)

    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH, "r") as f:
            meta = json.load(f)

    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "document_path": DATA_PATH,
        "retriever_model": meta.get("embed_model"),
        "generator_model": meta.get("generator_model"),
        "top_k": meta.get("top_k"),
        "indexed_chunks": len(pipeline.retriever.documents),
    }

    snapshot_file = os.path.join(SNAPSHOT_DIR, "snapshot.json")
    with open(snapshot_file, "w") as f:
        json.dump(snapshot, f, indent=2)

    logger.info("monitoring_snapshot_saved", extra={"snapshot_file": snapshot_file})
    print(f"Monitoring snapshot saved at {snapshot_file}")
