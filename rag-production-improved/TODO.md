
# RAG System Improvement TODO

This document defines the roadmap to evolve this project from a strong RAG portfolio system into an enterprise-grade, production-ready Retrieval Augmented Generation platform.

---

# HIGH PRIORITY - Core RAG Quality & Reliability

## Document Chunking & Ingestion

### Why
Current ingestion splits documents by line which degrades retrieval quality and scalability.

### Tasks
- [ ] Implement token-based chunking
- [ ] Add configurable chunk size
- [ ] Add configurable chunk overlap
- [ ] Support paragraph-aware splitting
- [ ] Support multi-format ingestion
  - PDF
  - DOCX
  - HTML
  - Plain text
- [ ] Add OCR pipeline for scanned documents
- [ ] Store chunk metadata
  - document_id
  - source_path
  - chunk_index
  - page / section mapping
  - ingestion timestamp

---

## Metadata-Aware Retrieval & Governance

### Tasks
- [ ] Return metadata alongside retrieved chunks
- [ ] Add metadata-based filtering
  - Tenant isolation
  - Document type filtering
  - Access control enforcement

---

## Multi-Tenant Isolation Architecture

### Tasks
- [ ] Tenant-level index isolation strategy
- [ ] Per-tenant embedding namespace
- [ ] Data leakage testing between tenants

---

## Context Window Management

### Tasks
- [ ] Implement tokenizer-based truncation
- [ ] Add max input token guard
- [ ] Dynamically reduce retrieved chunks if overflow detected
- [ ] Log context truncation events
- [ ] Implement dynamic chunk pruning

---

## Hybrid Retrieval & Reranking

### Tasks
- [ ] Implement hybrid retrieval
  - BM25 sparse retrieval
  - Dense embedding retrieval
- [ ] Add retrieval weighting configuration
- [ ] Add cross-encoder reranking
- [ ] Make reranking optional via configuration
- [ ] Implement multi-stage retrieval pipelines
  - Coarse retrieval → fine reranking pipeline
  - Hierarchical document → section → chunk retrieval

### Suggested Models
- cross-encoder/ms-marco-MiniLM-L-6-v2

---

## LLM Reliability & Trust

### Tasks
- [ ] Hallucination detection (runtime guardrails)
- [ ] Citation enforcement
- [ ] Answer-with-sources response format
- [ ] Confidence scoring thresholds
- [ ] Explicit "I don’t know" fallback responses
- [ ] Output guardrails and moderation
- [ ] Prompt injection detection
- [ ] Malicious document filtering
- [ ] Context sanitization before LLM call
- [ ] Retrieval content trust scoring

---

## Query Understanding & Routing

### Tasks
- [ ] Query rewriting using LLM
- [ ] Multi-query retrieval expansion
- [ ] Self-ask / decomposition retrieval pipelines
- [ ] Query intent classification
  - Detect factual vs generative vs conversational queries
  - Dynamically route retrieval strategy

---

## Retrieval Failure Recovery Strategy

### Tasks
- [ ] Retry retrieval with relaxed filters
- [ ] Fallback to broader search
- [ ] Fallback to LLM-only answer with warning

---

## Answer Post-Processing & Normalization

### Tasks
- [ ] Response formatting normalization
- [ ] Citation formatting validation
- [ ] Structured output schema validation

---

# MEDIUM PRIORITY - Evaluation, Prompting & Regression Safety

## Retrieval Evaluation

### Tasks
- [ ] Implement Recall@K
- [ ] Implement MRR
- [ ] Support benchmark datasets
- [ ] Store evaluation reports

---

## Generation Evaluation

### Tasks
- [ ] Faithfulness scoring
- [ ] Answer relevance scoring
- [ ] Hallucination detection heuristics (offline evaluation metrics)
- [ ] Reference-answer evaluation pipeline

### Suggested Libraries
- ragas
- deepeval

---

## Golden Dataset & Regression Testing

### Tasks
- [ ] Create curated Q&A dataset
- [ ] Implement regression testing for answers
- [ ] Track answer drift across versions
- [ ] LLM-based synthetic Q&A dataset generation
- [ ] Auto expansion of evaluation coverage

---

## Prompt Lifecycle Management

### Tasks
- [ ] Move prompts into versioned templates
- [ ] Build prompt registry
- [ ] Store prompt history
- [ ] Support A/B prompt testing
- [ ] Add prompt evaluation benchmarking

---

## Evaluation & Learning Systems

### Tasks
- [ ] Collect user feedback
- [ ] Track retrieval failure cases
- [ ] Automatically improve dataset using feedback
- [ ] Real-time learning pipeline

---

# MEDIUM PRIORITY - Performance & Scalability

## Caching & Latency Optimization

### Tasks
- [ ] Implement embedding caching
- [ ] Implement retrieval result caching
- [ ] Add configurable cache TTL
- [ ] Support Redis or in-memory LRU cache

---

## Batch Processing & Inference Optimization

### Tasks
- [ ] Support batch embedding requests
- [ ] Support batch generation processing
- [ ] Optimize transformer inference batching

---

## Model Serving Optimization

### Tasks
- [ ] Model warm pooling
- [ ] Multi-worker inference support
- [ ] Worker health tracking

---

## Index Freshness & Streaming Knowledge Updates

### Tasks
- [ ] Incremental index updates
- [ ] Streaming ingestion support
- [ ] Background embedding workers
- [ ] Temporal retrieval weighting

---

## Index Build & Deployment Pipelines

### Tasks
- [ ] Offline index build pipeline
- [ ] Index validation before promotion
- [ ] Blue/green index deployment

---

# ENTERPRISE PLATFORM READINESS

## Security & Privacy

### Tasks
- [ ] Authentication & authorization
  - API keys
  - JWT
  - mTLS
- [ ] Encryption at rest for indexes and artifacts
- [ ] Request / response PII redaction
- [ ] PII-safe structured logging
- [ ] Rate limiting per tenant / user
- [ ] Query size limits
- [ ] Prompt injection abuse throttling

---

## Cost Observability & Budget Controls

### Tasks
- [ ] Track token usage per request
- [ ] Track embedding cost
- [ ] Track reranker inference cost
- [ ] Add per-tenant cost attribution
- [ ] Add cost anomaly alerts
- [ ] Budget enforcement policies

---

## Observability & Monitoring

### Tasks
- [ ] Structured metrics collection
  - Latency p95 / p99
  - Retrieval quality metrics
  - Error rate tracking
- [ ] End-to-end distributed tracing
  - Retrieval
  - Reranking
  - Generation
- [ ] Integrate OpenTelemetry
- [ ] Export traces to Jaeger / Grafana
- [ ] Alerting for:
  - Timeouts
  - Empty retrieval
  - Data drift signals
- [ ] Define SLOs for latency and accuracy
- [ ] Error budget tracking
- [ ] Availability monitoring

---

## Scalability & Operations

### Tasks
- [ ] Stateless API design
- [ ] Externalized index / artifact storage
- [ ] Concurrency limits
- [ ] Backpressure handling
- [ ] Hot reload support
- [ ] Rolling index updates without downtime
- [ ] Horizontal autoscaling support
- [ ] Async batch inference service

---

## MLOps Governance & Versioning

### Tasks
- [ ] Model and index version pinning
- [ ] Rollback support
- [ ] Dataset lineage tracking
- [ ] Dataset drift detection
- [ ] Embedding drift monitoring
- [ ] Strict offline vs online pipeline separation
- [ ] Artifact validation during CI/CD

---

## Compliance & Data Governance

### Tasks
- [ ] Document deletion propagation to index
- [ ] GDPR / Right-to-be-forgotten workflows
- [ ] Retention policy enforcement

---

## Developer & API Experience

### Tasks
- [ ] Versioned APIs (e.g. /v1/ask)
- [ ] Token-level streaming responses
- [ ] CLI ingestion tools
- [ ] SDK support for integrations

---

# TECHNICAL DEBT FIXES

- [ ] Fix Generator max_new_tokens shadowing bug
- [ ] Add model loading error handling
- [ ] Add graceful shutdown hooks
- [ ] Add index compatibility validation

---

# FUTURE SCALING TARGETS

- [ ] Distributed FAISS or vector database migration
- [ ] GPU inference optimization
- [ ] Multi-hop reasoning support
- [ ] Autonomous self-improving retrieval

---

# IMPLEMENTATION MILESTONES

## Phase 1 - Retrieval & Reliability
- Chunking
- Metadata filtering
- Multi-tenant isolation
- Context management
- Hybrid retrieval
- Reranking
- LLM trust enforcement
- Query intent routing
- Retrieval failure recovery

---

## Phase 2 - Evaluation & Prompt Engineering
- Retrieval metrics
- Generation metrics
- Golden dataset
- Synthetic dataset expansion
- Prompt versioning
- Query rewriting
- Feedback learning loop

---

## Phase 3 - Performance & Scaling
- Caching
- Batching
- Model warm pooling
- Incremental indexing
- Blue/green index deployment

---

## Phase 4 - Enterprise Readiness
- Security
- Cost observability
- Observability
- Governance
- Compliance
- CI/CD
- API versioning
- Rate limiting
- SLA/SLO enforcement

---

# TARGET OUTCOME

Completion of this roadmap results in:

- Enterprise-ready RAG platform
- Research-capable RAG experimentation environment
- Production-scale GenAI service

