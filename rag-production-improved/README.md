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

### Run API

```bash
podman build -f Dockerfile.dev -t rag-dev:latest .

podman run --rm -p 8000:8000 \
  -v $(pwd)/app:/app/app:Z \
  -v $(pwd)/data:/app/data:Z \
  -v $(pwd)/artifacts:/app/artifacts:Z \
  rag-dev:latest
```

### MLOps Scripts (Podman)

```bash
podman build -f Dockerfile.mlops -t rag-production-improved-mlops:latest .

# Build Artifacts
podman run --rm \
  -v $(pwd)/data:/mlops/data:Z \
  -v $(pwd)/artifacts:/mlops/artifacts:Z \
  rag-production-improved-mlops:latest \
  python mlops/build/build_artifacts.py

# Evaluate RAG
podman run --rm \
  -v $(pwd)/data:/mlops/data:Z \
  -v $(pwd)/artifacts:/mlops/artifacts:Z \
  rag-production-improved-mlops:latest \
  python mlops/evaluation/evaluate_rag.py

# Update Model Registry
podman run --rm \
  -v $(pwd)/data:/mlops/data:Z \
  -v $(pwd)/mlops:/mlops/mlops:Z \
  -v $(pwd)/artifacts:/mlops/artifacts:Z \
  rag-production-improved-mlops:latest \
  python mlops/registry/model_registry.py

# Monitoring Snapshot
podman run --rm \
  -v $(pwd)/data:/mlops/data:Z \
  -v $(pwd)/mlops:/mlops/mlops:Z \
  -v $(pwd)/artifacts:/mlops/artifacts:Z \
  rag-production-improved-mlops:latest \
  python mlops/monitoring/monitoring_snapshot.py

# Run Tests
podman run --rm \
  -v $(pwd)/data:/mlops/data:Z \
  -v $(pwd)/artifacts:/mlops/artifacts:Z \
  rag-production-improved-mlops:latest \
  pytest mlops/tests
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

## **API Reference**

The RAG system exposes a simple HTTP API for querying documents, managing artifacts, and health monitoring.

**Base URL (local):**

```
http://localhost:8000
```

### Ask a Question

Retrieve an answer generated using retrieved document context.

```
GET /ask?q=<question>
```

**Example**

```
http://localhost:8000/ask?q=What+is+this+system+used+for?
```

---

### Reload / Rebuild Documents

Reloads the source documents and rebuilds the FAISS index if required.

```
GET /reload
GET /reload?doc=data/doc1.txt
```

---

### Health Check

Lightweight endpoint for liveness and readiness checks.

```
GET /health
```

**Response**

```json
{ "status": "ok" }
```

---

### Notes

* Designed for **private and internal deployments**
* Stateless API suitable for containerized environments
* Authentication, authorization, and audit logging are planned as part of the **Enterprise Production Roadmap**

---

## Enterprise Adoption & Support

If you are an organization looking to implement a **secure, private, and cost-effective RAG system**, feel free to reach out.

I help enterprises build **on-prem and private RAG solutions** that:

* Answer questions from **internal documents and SOPs**
* Ensure **data privacy** (no data leaves your environment)
* Operate at **low infrastructure cost**
* Run **on-premises or in private cloud environments**
* Scale from **laptop prototypes to production systems**

📧 **Contact**: `cosmos.lab.contact@gmail.com`





