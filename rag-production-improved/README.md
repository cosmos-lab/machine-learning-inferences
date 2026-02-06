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


## Enterprise Adoption & Support

If you are an organization looking to implement a **secure, private, and cost-effective RAG system**, feel free to reach out.

I help enterprises build **on-prem and private RAG solutions** that:

* Answer questions from **internal documents and SOPs**
* Ensure **data privacy** (no data leaves your environment)
* Operate at **low infrastructure cost**
* Run **on-premises or in private cloud environments**
* Scale from **laptop prototypes to production systems**

📧 **Contact**: `cosmos.lab.contact@gmail.com`


---

# RAG Maturity Model (L1 → L5)


# 🟢 L1 — Foundational RAG (Proof of Concept)

> Goal: “System can answer questions using retrieved context”

This is portfolio / demo / hackathon level.

---

## Retrieval & Ingestion Basics

* Basic document ingestion
* Basic chunking (even if naive)
* Embedding generation
* Vector index creation
* Similarity search retrieval

---

## Generation Basics

* Prompt template with retrieved context
* Single-shot generation
* Basic response formatting

---

## Basic Engineering

* Simple API endpoint
* Model loading pipeline
* Logging
* Environment configuration
* Unit testing for core components

---

## Basic MLOps

* Artifact versioning
* Model registry basics
* Basic evaluation experimentation

---

### 🔥 Outcome

✔ Working RAG
✔ Demonstrates core concept
❌ Low reliability
❌ No trust guarantees
❌ No performance optimization

---

# 🟡 L2 — Reliable RAG System (Production MVP)

> Goal: “Answers are consistently relevant and stable”

This is where real product usability begins.

---

## Retrieval Intelligence

* Token-based chunking
* Configurable chunk size & overlap
* Paragraph-aware splitting
* Multi-format ingestion
* OCR ingestion
* Metadata storage
* Metadata-based filtering

---

## Context Management

* Tokenizer-based truncation
* Max token guards
* Dynamic chunk pruning
* Context overflow protection

---

## Multi-Tenant Retrieval Isolation

* Tenant index isolation
* Per-tenant embedding namespace
* Data leakage validation

---

## Retrieval Failure Handling

* Retry with relaxed filters
* Broader fallback retrieval
* LLM-only fallback response

---

## Trust & Output Safety (Baseline)

* Citation generation
* Answer-with-sources formatting
* Confidence threshold fallback
* Basic hallucination detection
* Prompt injection detection
* Malicious document filtering
* Output moderation guardrails

---

### 🔥 Outcome

✔ Good answer quality
✔ Safer responses
✔ Usable for real apps
❌ Performance still limited
❌ Evaluation still immature

---

# 🟠 L3 — Advanced Retrieval Intelligence (Research-Grade RAG)

> Goal: “System actively reasons about retrieval”

This is where RAG becomes *intelligent*, not just retrieval + generation.

---

## Hybrid Retrieval

* BM25 + dense retrieval
* Retrieval weighting control
* Multi-stage retrieval pipelines
* Hierarchical document → section → chunk retrieval

---

## Reranking

* Cross-encoder reranking
* Optional reranking configuration

---

## Query Understanding

* Query rewriting
* Multi-query expansion
* Self-ask / decomposition retrieval
* Query intent classification
* Dynamic retrieval routing

---

## Advanced Trust & Validation

* Retrieval trust scoring
* Context sanitization
* Structured output validation
* Citation validation
* Response normalization

---

## Early Multi-Hop Reasoning

* Multi-hop retrieval pipelines
* Complex query decomposition

---

### 🔥 Outcome

✔ High retrieval precision
✔ Strong reasoning capability
✔ Competitive with research systems
❌ Expensive
❌ Hard to tune

---

# 🔵 L4 — Enterprise RAG Platform (Production Scale & Governed)

> Goal: “System is observable, scalable, compliant, and safe”

This is where most serious SaaS AI products must reach.

---

## Performance & Scalability

* Embedding caching
* Retrieval result caching
* Cache TTL controls
* Batch embedding & generation
* Transformer batching optimization
* Model warm pooling
* Multi-worker inference
* Worker health tracking

---

## Index Infrastructure

* Incremental index updates
* Streaming ingestion
* Background embedding workers
* Temporal retrieval weighting
* Offline index build pipelines
* Index validation pipelines
* Blue/green index deployment
* Distributed vector database support

---

## Platform Engineering

* Stateless API design
* External artifact storage
* Horizontal autoscaling
* Rolling index updates
* Async inference services
* Token streaming responses
* CLI ingestion tools
* SDK integrations
* API versioning

---

## Security & Privacy

* Authentication & authorization
* Encryption at rest
* PII redaction
* Safe structured logging
* Tenant rate limiting
* Query size enforcement
* Abuse throttling

---

## Observability & Reliability

* Structured metrics collection
* Distributed tracing
* OpenTelemetry integration
* Jaeger / Grafana tracing export
* Timeout / empty retrieval alerting
* Data drift alerting
* SLO & error budget tracking
* Availability monitoring
* Graceful shutdown handling
* Model load error handling
* Index compatibility validation

---

### 🔥 Outcome

✔ Enterprise-ready platform
✔ Scales to production load
✔ Meets reliability standards
❌ Still reactive rather than self-improving

---

# 🔴 L5 — Autonomous & Self-Improving RAG (Frontier / FAANG-Level)

> Goal: “System continuously measures and improves itself”

Very few organizations reach this maturity.

---

## Evaluation Intelligence

* Recall@K tracking
* MRR tracking
* Benchmark dataset pipelines
* Evaluation report storage

---

## Generation Quality Measurement

* Faithfulness scoring
* Answer relevance scoring
* Offline hallucination metrics
* Reference-answer evaluation pipelines

---

## Regression Safety & Dataset Governance

* Golden Q&A datasets
* Answer regression testing
* Drift detection across versions
* Synthetic dataset generation
* Automatic evaluation coverage expansion

---

## Prompt Lifecycle Management

* Versioned prompt templates
* Prompt registry
* Prompt history
* A/B prompt testing
* Prompt benchmarking

---

## Continuous Learning Systems

* User feedback ingestion
* Retrieval failure analytics
* Feedback-driven dataset improvement
* Real-time learning pipelines
* Autonomous retrieval improvement

---

## ML Governance

* Model version pinning
* Index version pinning
* Rollback systems
* Dataset lineage tracking
* Dataset drift monitoring
* Embedding drift detection
* Offline vs online pipeline separation
* CI/CD artifact validation

---

## Compliance & Governance

* Document deletion propagation
* GDPR workflows
* Retention policy enforcement

---

## Cost Intelligence

* Token usage monitoring
* Embedding cost tracking
* Reranker cost tracking
* Tenant cost attribution
* Cost anomaly detection
* Budget enforcement

---

### 🔥 Outcome

✔ Self-monitoring AI system
✔ Automatically improving accuracy
✔ Regulated & auditable
✔ Research + production convergence

---

# ⭐ Visual Evolution Summary

```
L1 → Works
L2 → Reliable
L3 → Intelligent
L4 → Scalable
L5 → Self-Improving
```

---

# ⭐ Industry Mapping (Very Useful for Resume / Architecture Talks)

| Level | Industry Equivalent                                                  |
| ----- | -------------------------------------------------------------------- |
| L1    | Hackathon / demo RAG                                                 |
| L2    | Early production chatbot                                             |
| L3    | Advanced retrieval research systems                                  |
| L4    | Enterprise AI SaaS platform                                          |
| L5    | Frontier GenAI platform (OpenAI / Anthropic / Google DeepMind style) |










