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
podman run --rm -p 8001:8000 \
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


## Enterprise Production Roadmap

This repository provides a production-grade foundation for Retrieval-Augmented Generation (RAG).
The following roadmap defines the key capability areas required to harden the system for enterprise production use:

* **Security, authentication, and access control**
  Ensure secure usage through authentication, RBAC, and auditability.

* **Scalable document and knowledge management**
  Support large, evolving document corpora with proper lifecycle handling.

* **Observability, monitoring, and analytics**
  Provide visibility into system performance, usage patterns, and reliability.

* **Model, index, and artifact lifecycle management**
  Enable safe upgrades, versioning, evaluation, and rollback.

* **Deployment scalability and infrastructure flexibility**
  Support CPU, GPU, and cloud-native deployment patterns.

* **Governance, compliance, and operational readiness**
  Meet enterprise expectations around data privacy, reliability, and maintainability.

---

## TODO / Enterprise Readiness Checklist

The following items outline the concrete implementation tasks required to achieve the enterprise production roadmap.

### Security & Access Control

* API authentication (API keys or JWT-based auth)
* Role-based access control (RBAC)
* Action-level permissions (read, reload, rebuild)
* Audit logging for sensitive operations

### Multi-Tenancy & Isolation

* Tenant or workspace-level isolation
* Per-tenant FAISS index and artifacts
* Namespace separation for documents and metadata

### Document Lifecycle Management

* Add / update / delete documents dynamically
* Incremental index updates
* Document versioning and change tracking

### Observability & Monitoring

* Request-level metrics (latency, throughput)
* Retrieval hit/miss and empty-result tracking
* Error rate and failure monitoring
* Optional Prometheus-compatible metrics

### Evaluation & Quality Assurance

* Golden Q&A dataset support
* Offline accuracy and relevance scoring
* Regression detection across model or index changes
* Evaluation gates before production rollout

### Model & Artifact Management

* Explicit model and index versioning
* Immutable artifact storage
* Rollback support for models and indices

### Configuration & Environment Management

* Environment separation (dev / staging / prod)
* Environment-based model and resource selection
* Feature flags for high-risk operations

### Reliability & Failure Handling

* Graceful degradation on generator failures
* Timeouts for retrieval and generation steps
* User-friendly error responses

### Deployment & Scalability

* GPU-backed inference support
* External LLM API integration option
* Horizontal scaling patterns and guidance

### Governance & Documentation

* Data privacy and retention policy
* Model usage scope and limitations
* Upgrade and backward-compatibility policy
* Security and compliance notes

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





