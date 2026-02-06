
# RAG

## Retrieval Quality & Context Intelligence

* Token-based chunking
* Configurable chunk size
* Configurable chunk overlap
* Paragraph-aware splitting
* Multi-format ingestion
* OCR ingestion
* Metadata storage
* Metadata-aware retrieval
* Metadata filtering
* Tenant isolation retrieval
* Multi-tenant embedding namespace
* Data leakage testing
* Context window truncation
* Max token guards
* Dynamic chunk pruning
* Hybrid retrieval (BM25 + Dense)
* Retrieval weighting configuration
* Cross-encoder reranking
* Optional reranking configuration
* Multi-stage retrieval pipelines
* Hierarchical retrieval pipelines

---

## Query Intelligence & Reasoning

* Query rewriting
* Multi-query expansion
* Self-ask / decomposition retrieval
* Query intent classification
* Retrieval strategy routing
* Multi-hop reasoning support

---

## LLM Trust & Output Intelligence

* Hallucination detection
* Citation enforcement
* Answer-with-sources formatting
* Confidence scoring thresholds
* “I don’t know” fallback
* Output moderation guardrails
* Prompt injection detection
* Malicious document filtering
* Context sanitization
* Retrieval trust scoring
* Structured output validation
* Response normalization
* Citation validation

---

## Failure Handling Intelligence

* Retrieval retry with relaxed filters
* Broader fallback retrieval
* LLM-only fallback response

---

---

# RAG Plumbing (Infrastructure, Performance, Reliability)

## Performance & Latency Optimization

* Embedding caching
* Retrieval result caching
* Configurable cache TTL
* Redis / LRU cache support
* Batch embedding requests
* Batch generation processing
* Transformer batching optimization
* Model warm pooling
* Multi-worker inference
* Worker health tracking

---

## Indexing & Data Infrastructure

* Incremental index updates
* Streaming ingestion
* Background embedding workers
* Temporal retrieval weighting
* Offline index build pipelines
* Index validation before promotion
* Blue/green index deployment
* Distributed FAISS / Vector DB migration

---

## Platform Scalability & Runtime Operations

* Stateless API design
* External artifact storage
* Concurrency limits
* Backpressure handling
* Hot reload support
* Rolling index updates
* Horizontal autoscaling
* Async batch inference services
* GPU inference optimization

---

## API Platform Engineering

* Versioned APIs
* Token streaming responses
* CLI ingestion tooling
* SDK integrations

---

## Security & Runtime Controls

* Authentication & authorization
* Encryption at rest
* PII redaction
* PII-safe logging
* Tenant rate limiting
* Query size limits
* Prompt abuse throttling

---

## Reliability & Observability

* Metrics collection (Latency / Error Rate / Retrieval Metrics)
* Distributed tracing
* OpenTelemetry integration
* Jaeger / Grafana export
* Timeout alerts
* Empty retrieval alerts
* Data drift alerting
* Latency and accuracy SLOs
* Error budget tracking
* Availability monitoring
* Graceful shutdown handling
* Model loading error handling
* Index compatibility validation

---

# RAG Governance, Evaluation & Continuous Learning

## Retrieval Evaluation & Benchmarking

* Recall@K metrics
* MRR metrics
* Benchmark dataset support
* Evaluation report storage

---

## Generation Quality Evaluation

* Faithfulness scoring
* Answer relevance scoring
* Offline hallucination evaluation
* Reference-answer evaluation pipeline

---

## Dataset & Regression Safety

* Golden Q&A dataset
* Answer regression testing
* Answer drift tracking
* Synthetic dataset generation
* Evaluation coverage expansion

---

## Prompt Lifecycle Governance

* Versioned prompt templates
* Prompt registry
* Prompt history tracking
* A/B prompt testing
* Prompt benchmarking

---

## Learning & Feedback Systems

* User feedback collection
* Retrieval failure tracking
* Feedback-driven dataset improvement
* Real-time learning pipelines
* Autonomous self-improving retrieval

---

## ML Governance & Version Control

* Model version pinning
* Index version pinning
* Rollback support
* Dataset lineage tracking
* Dataset drift detection
* Embedding drift monitoring
* Offline vs online pipeline separation
* CI/CD artifact validation

---

## Compliance & Data Governance

* Document deletion propagation
* GDPR workflows
* Retention enforcement

---

## Cost Governance

* Token usage tracking
* Embedding cost tracking
* Reranker cost tracking
* Per-tenant cost attribution
* Cost anomaly detection
* Budget enforcement policies


---



