# Lightweight Extractive RAG System

## **Objective**

Build a **production-ready Retrieval-Augmented Generation (RAG) system** in Python, with **FAISS-based retrieval**, **transformer-based generation**, and **enterprise MLOps best practices** for artifact management, monitoring, and evaluation.

---

## **Key Features**

* **FastAPI API**: Query the RAG system in real-time with `/ask`, `/reload`, and `/health` endpoints.
* **FAISS Vector Search**: Efficient semantic search with `SentenceTransformers` embeddings.
* **Transformer Generator**: Generate answers using context with `Flan-T5` or custom seq2seq models.
* **MLOps Integration**:

  * Build and persist artifacts (`build_artifacts.py`)
  * Evaluate RAG performance (`evaluate_rag.py`)
  * Maintain model registry (`model_registry.py`)
  * Monitoring snapshots (`monitoring_snapshot.py`)
* **Production-ready**: Docker/Podman containers, async API, CORS support.
* **Unit Tests**: Verify retriever, generator, and pipeline functionality.

---

## **Architecture**

```
        ┌───────────────────────────┐
        │     FastAPI API           │
        │  (/ask, /reload, /health) │
        └─────────┬─────────────────┘
                  │
           ┌──────▼───────────┐
           │ RAG Pipeline     │
           │──────────────────│
           │ Retriever        │
           │  └─ FAISS        │
           │ Generator        │
           │  └─ Transformers │
           └──────┬───────────┘
                  │
        ┌─────────▼─────────┐
        │   Document Data   │
        │   artifacts/      │
        └───────────────────┘
```

* **Retriever**: Embeds documents using SentenceTransformers and indexes them in FAISS.
* **Generator**: Uses seq2seq models to generate answers based on retrieved context.
* **Artifacts & Metadata**: Stored in `artifacts/` for reproducibility and MLOps traceability.

---


## Hardware Notes

Got it! Here’s a **concise, professional way to phrase it** for your README, emphasizing that your **default configuration is chosen for modest hardware**:

---

Got it! Here’s a **clear, professional phrasing** with an example for the README:

---

## **Hardware Requirements & Recommendations**

* **Default configuration**: Optimized for **modest hardware** (16 GB RAM, CPU-based inference, or limited GPU).

* **Scaling up for better performance or accuracy**:

  * If you have a **dedicated GPU** or **more RAM** (32 GB+), you can choose **larger models** for higher accuracy.
  * Example:

    * **Embedding model**: `sentence-transformers/all-mpnet-base-v2` (larger, more accurate than MiniLM)
    * **Generator model**: `google/flan-t5-large` (more detailed answers than `flan-t5-base`)

* **Recommendation**: Adjust models based on your **hardware availability** vs **accuracy requirements**. For high-throughput production, consider offloading the generator to a **GPU server or API**.

---


## **Installation**

### Build API container

```bash
podman build -f Dockerfile -t rag_production:latest .
```

### Build MLOps container

```bash
podman build -f Dockerfile.mlops -t rag_production_mlops:latest .
```

---

## **Usage**

### Run API

```bash
podman run --rm -p 8000:8000 \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag_production:latest
```

### MLOps Scripts (Podman)

```bash
# Build Artifacts
podman run --rm -w /app -e PYTHONPATH=/app \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/mlops:/app/mlops:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag_production_mlops:latest \
  python mlops/build_artifacts.py

# Evaluate RAG
podman run --rm -w /app -e PYTHONPATH=/app \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/mlops:/app/mlops:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag_production_mlops:latest \
  python mlops/evaluate_rag.py

# Update Model Registry
podman run --rm -w /app -e PYTHONPATH=/app \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/mlops:/app/mlops:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag_production_mlops:latest \
  python mlops/model_registry.py

# Create Monitoring Snapshot
podman run --rm -w /app -e PYTHONPATH=/app \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/mlops:/app/mlops:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag_production_mlops:latest \
  python mlops/monitoring_snapshot.py
```

### Run Tests

```bash
podman run --rm -w /app -e PYTHONPATH=/app \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/mlops:/app/mlops:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag_production_mlops:latest \
  pytest /app/mlops/tests
```

---

## **Configuration**

* `app/config.py` handles model selection, retrieval settings, and data paths:

  * `EMBED_MODEL`, `GEN_MODEL`
  * `TOP_K`, `MAX_NEW_TOKENS`
  * `DATA_PATH`, `INDEX_PATH`, `META_PATH`

---

## **Directory Structure**

```
.
├─ app/               # API & pipeline
├─ mlops/             # MLOps scripts
├─ data/              # Input Documents
├─ artifacts/         # FAISS index & metadata
├─ models/            # Model manifest
├─ Dockerfile         # API container
├─ Dockerfile.mlops   # MLOps container
└─ README.md
```

---

## **License**

MIT License

