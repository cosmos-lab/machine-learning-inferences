import json
import os
from datetime import datetime
from app.core.pipeline import RAGPipeline
from app.config.settings import DATA_PATH, META_PATH

SNAPSHOT_DIR = "mlops/snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load_from_file(DATA_PATH)

    meta = {}
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            meta = json.load(f)

    snapshot = {
        "timestamp": datetime.utcnow().isoformat(),
        "document_path": DATA_PATH,
        "retriever_model": meta.get("embed_model"),
        "generator_model": meta.get("generator_model"),
        "top_k": meta.get("top_k"),
        "indexed_chunks": len(pipeline.retriever.documents),
    }

    with open(os.path.join(SNAPSHOT_DIR, "snapshot.json"), "w") as f:
        json.dump(snapshot, f, indent=2)


