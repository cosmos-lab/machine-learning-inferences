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

## ** Better flow **
```
Document
   ↓
Chunk
   ↓
Retriever.build()
   ↓
Embed
   ↓
Vector DB
   ↓
FastAPI query
   ↓
Retriever.retrieve()
   ↓
Vector DB search
   ↓
Chunks
   ↓
Generator
```

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
**Complete Edition with Production, Research, and Enterprise Best Practices**

---

# 🟢 L1 — Foundational RAG (Proof of Concept)

> **Goal:** "System can answer questions using retrieved context"

This is portfolio / demo / hackathon level.

---

## Retrieval & Ingestion Basics

* Basic document ingestion (PDF, TXT, DOCX)
* Basic chunking (even if naive - fixed character/word count)
* Embedding generation (single model)
* Vector index creation
* Similarity search retrieval (top-k)

---

## Generation Basics

* Prompt template with retrieved context
* Single-shot generation
* Basic response formatting
* Simple streaming response (optional)

---

## Basic Engineering

* Simple API endpoint (Flask/FastAPI)
* Model loading pipeline
* Basic logging (print statements or basic logger)
* Environment configuration (.env files)
* Unit testing for core components

---

## Basic MLOps

* Artifact versioning (manual or git-based)
* Model registry basics (local storage)
* Basic evaluation experimentation (manual testing)

---

### 🔥 Outcome

✅ Working RAG proof of concept  
✅ Demonstrates core concept  
✅ Suitable for demos and prototypes  
❌ Low reliability and accuracy  
❌ No trust guarantees  
❌ No performance optimization  
❌ Not production-ready  

---

# 🟡 L2 — Reliable RAG System (Production MVP)

> **Goal:** "Answers are consistently relevant and stable enough for real users"

This is where real product usability begins.

---

## Retrieval Intelligence

* **Token-based chunking** (using tokenizer, not character count)
* **Configurable chunk size & overlap**
* **Paragraph-aware splitting** (respects document structure)
* **Multi-format ingestion** (PDF, DOCX, HTML, Markdown, CSV)
* **OCR ingestion** (for scanned documents/images)
* **Metadata storage** (source, timestamp, author, category)
* **Metadata-based filtering** (filter by date, source, category)
* **Document deduplication** (hash-based or content similarity)

---

## Context Management

* **Tokenizer-based truncation** (accurate token counting)
* **Max token guards** (prevent context overflow)
* **Dynamic chunk pruning** (remove least relevant chunks if needed)
* **Context overflow protection** (graceful degradation)
* **Context ordering strategies** (most relevant first or last)

---

## Multi-Tenant Retrieval Isolation

* **Tenant index isolation** (separate indexes per tenant)
* **Per-tenant embedding namespace** (logical separation)
* **Data leakage validation** (automated tests)
* **Tenant-specific metadata filtering**

---

## Retrieval Failure Handling

* **Retry with relaxed filters** (if strict filters return nothing)
* **Broader fallback retrieval** (expand search scope)
* **LLM-only fallback response** (when no context is found)
* **Empty result messaging** (transparent communication)
* **Error handling and retry logic** (network, API failures)

---

## Trust & Output Safety (Baseline)

* **Citation generation** (link answers to sources)
* **Answer-with-sources formatting**
* **Confidence threshold fallback** (refuse low-confidence answers)
* **Basic hallucination detection** (simple heuristics)
* **Prompt injection detection** (input sanitization)
* **Malicious document filtering** (content validation)
* **Output moderation guardrails** (toxicity, bias filters)

---

## Quality & Testing

* **Integration testing** (end-to-end RAG pipeline)
* **Regression test suite** (ensure changes don't break functionality)
* **Document freshness tracking** (timestamp-based)

---

## User Experience

* **Streaming responses** (token-by-token generation)
* **Loading indicators** (user feedback during retrieval)
* **Error messages** (user-friendly, actionable)

---

## Operations Basics

* **Basic cost tracking** (token usage, API calls)
* **Simple response time SLAs** (p95, p99 latency targets)
* **Health check endpoints**
* **Basic monitoring** (uptime, error rates)

---

### 🔥 Outcome

✅ Good answer quality  
✅ Safer responses  
✅ Usable for real applications  
✅ Basic production reliability  
❌ Performance still limited  
❌ Evaluation still immature  
❌ Scalability constraints  

---

# 🟠 L3 — Advanced Retrieval Intelligence (Research-Grade RAG)

> **Goal:** "System actively reasons about retrieval and understands complex queries"

This is where RAG becomes *intelligent*, not just retrieval + generation.

---

## Hybrid & Multi-Stage Retrieval

* **BM25 + dense retrieval** (keyword + semantic)
* **Retrieval weighting control** (adjust BM25/dense ratio)
* **Multi-stage retrieval pipelines** (coarse-to-fine)
* **Hierarchical retrieval** (document → section → chunk)
* **Fusion ranking** (combining multiple retrieval strategies)

---

## Advanced Chunking & Context

* **Contextual compression** (remove irrelevant parts of chunks)
* **Parent-child chunking** (retrieve small, provide large context)
* **Sentence window retrieval** (expand context around matched sentences)
* **Semantic chunking** (chunk by meaning, not just size)
* **Sliding window with smart boundaries** (respect semantic units)

---

## Reranking

* **Cross-encoder reranking** (deep semantic matching)
* **Optional reranking configuration** (performance vs accuracy trade-off)
* **Multi-stage reranking** (fast model then slow model)
* **LLM-based reranking** (using LLM to score relevance)

---

## Query Understanding & Expansion

* **Query rewriting** (rephrase for better retrieval)
* **Multi-query expansion** (generate variations of query)
* **HyDE (Hypothetical Document Embeddings)** (generate hypothetical answer, retrieve similar docs)
* **Self-ask / decomposition retrieval** (break complex queries into sub-queries)
* **Query intent classification** (categorize query type)
* **Dynamic retrieval routing** (route to specialized retrievers)
* **Query routing to specialized indexes** (different indexes for different content types)

---

## Cross-Domain & Multi-Modal

* **Cross-lingual retrieval** (query in one language, retrieve in another)
* **Multi-modal retrieval** (text, images, tables)
* **Table understanding** (structured data retrieval)

---

## Advanced Trust & Validation

* **Retrieval trust scoring** (confidence in retrieved chunks)
* **Context sanitization** (remove PII, sensitive info)
* **Structured output validation** (schema compliance)
* **Citation validation** (verify citations are accurate)
* **Response normalization** (consistent format)
* **Source credibility weighting** (prioritize authoritative sources)
* **Document quality scoring** (filter low-quality content)

---

## Multi-Hop & Complex Reasoning

* **Multi-hop retrieval pipelines** (retrieve, reason, retrieve again)
* **Complex query decomposition** (break down multi-part questions)
* **Chain-of-thought retrieval** (iterative reasoning)
* **GraphRAG / knowledge graph integration** (leverage entity relationships)
* **SQL + Vector hybrid search** (structured + unstructured)

---

## User Experience Enhancements

* **Suggested follow-up questions** (guide user exploration)
* **Query auto-completion** (based on document corpus)
* **Conversation memory** (multi-turn context)
* **Clarifying questions** (when query is ambiguous)

---

## Performance & Testing

* **Load testing** (stress test retrieval pipeline)
* **A/B testing infrastructure** (compare retrieval strategies)
* **Retrieval latency optimization** (caching, indexing strategies)

---

### 🔥 Outcome

✅ High retrieval precision and recall  
✅ Strong reasoning capability  
✅ Handles complex, multi-hop queries  
✅ Competitive with research systems  
❌ Expensive (compute, latency)  
❌ Hard to tune and maintain  
❌ Not yet scaled for high traffic  

---

# 🔵 L4 — Enterprise RAG Platform (Production Scale & Governed)

> **Goal:** "System is observable, scalable, compliant, secure, and operationally excellent"

This is where most serious SaaS AI products must reach.

---

## Performance & Scalability

* **Embedding caching** (cache embeddings for frequent queries)
* **Retrieval result caching** (cache top-k results)
* **Cache TTL controls** (manage cache freshness)
* **Batch embedding & generation** (process multiple requests together)
* **Transformer batching optimization** (dynamic batching)
* **Model warm pooling** (keep models loaded in memory)
* **Multi-worker inference** (parallel processing)
* **Worker health tracking** (monitor worker status)
* **Load balancing strategies** (distribute traffic efficiently)
* **Circuit breakers** (prevent cascading failures)
* **Request deduplication** (avoid redundant processing)

---

## Index Infrastructure

* **Incremental index updates** (add/update without full rebuild)
* **Streaming ingestion** (real-time document processing)
* **Background embedding workers** (async embedding generation)
* **Temporal retrieval weighting** (boost recent documents)
* **Offline index build pipelines** (batch processing)
* **Index validation pipelines** (ensure index integrity)
* **Blue/green index deployment** (zero-downtime updates)
* **Distributed vector database support** (Pinecone, Weaviate, Qdrant, Milvus)
* **Index sharding** (partition large indexes)
* **Hot/cold storage tiers** (optimize cost for infrequent data)

---

## Platform Engineering

* **Stateless API design** (horizontal scalability)
* **External artifact storage** (S3, GCS, Azure Blob)
* **Horizontal autoscaling** (scale based on load)
* **Rolling index updates** (gradual deployment)
* **Async inference services** (queue-based processing)
* **Token streaming responses** (real-time generation)
* **CLI ingestion tools** (developer productivity)
* **SDK integrations** (Python, Node.js, Go)
* **API versioning** (backward compatibility)
* **GraphQL / REST API** (flexible query interface)
* **WebSocket support** (real-time streaming)
* **Multi-region deployment** (global availability)
* **Disaster recovery** (backup and restore strategies)

---

## Security & Privacy

* **Authentication & authorization** (OAuth, JWT, API keys)
* **Role-based access control (RBAC)** (fine-grained permissions)
* **Encryption at rest** (data security)
* **Encryption in transit** (TLS/SSL)
* **PII redaction** (automatic sensitive data removal)
* **Safe structured logging** (no PII in logs)
* **Tenant rate limiting** (per-tenant quotas)
* **Query size enforcement** (prevent abuse)
* **Abuse throttling** (detect and block malicious usage)
* **API rate limiting per tenant/user** (prevent overuse)
* **Data residency controls** (compliance with local regulations)
* **Audit trails** (compliance logging)

---

## Observability & Reliability

* **Structured metrics collection** (Prometheus, StatsD)
* **Distributed tracing** (trace requests across services)
* **OpenTelemetry integration** (standardized observability)
* **Jaeger / Grafana tracing export** (visualize traces)
* **Timeout / empty retrieval alerting** (proactive monitoring)
* **Data drift alerting** (detect index degradation)
* **SLO & error budget tracking** (measure reliability)
* **Availability monitoring** (uptime SLAs)
* **Graceful shutdown handling** (zero request drops)
* **Model load error handling** (fallback strategies)
* **Index compatibility validation** (prevent breaking changes)
* **Latency percentile tracking** (p50, p95, p99)
* **Error categorization** (retryable vs non-retryable)
* **Dependency health checks** (monitor external services)

---

## Testing & Quality Assurance

* **A/B testing infrastructure** (compare system versions)
* **Canary releases** (gradual rollout)
* **Shadow deployment testing** (test in production without impact)
* **Chaos engineering practices** (resilience testing)
* **Performance benchmarking** (regression prevention)

---

### 🔥 Outcome

✅ Enterprise-ready platform  
✅ Scales to production load (millions of queries)  
✅ Meets reliability standards (99.9%+ uptime)  
✅ Secure and compliant  
✅ Observable and debuggable  
❌ Still reactive rather than self-improving  
❌ Manual optimization required  

---

# 🔴 L5 — Autonomous & Self-Improving RAG (Frontier / FAANG-Level)

> **Goal:** "System continuously measures, learns, and improves itself with minimal human intervention"

Very few organizations reach this maturity. This is the cutting edge.

---

## Evaluation Intelligence

* **Recall@K tracking** (measure retrieval quality)
* **MRR (Mean Reciprocal Rank) tracking** (rank quality)
* **NDCG (Normalized Discounted Cumulative Gain)** (ranking metric)
* **Precision@K tracking** (retrieval precision)
* **Benchmark dataset pipelines** (automated evaluation)
* **Evaluation report storage** (historical tracking)
* **Per-query-type evaluation** (segment performance)
* **Retrieval quality dashboards** (real-time monitoring)

---

## Generation Quality Measurement

* **Faithfulness scoring** (answers grounded in context)
* **Answer relevance scoring** (answers address query)
* **Offline hallucination metrics** (detect fabrication)
* **Reference-answer evaluation pipelines** (compare to gold standard)
* **Semantic similarity scoring** (answer quality)
* **Completeness scoring** (comprehensive answers)
* **LLM-as-judge evaluation** (automated quality assessment)

---

## Regression Safety & Dataset Governance

* **Golden Q&A datasets** (curated test sets)
* **Answer regression testing** (prevent quality degradation)
* **Drift detection across versions** (monitor changes)
* **Synthetic dataset generation** (augment test coverage)
* **Automatic evaluation coverage expansion** (identify gaps)
* **Dataset lineage tracking** (data provenance)
* **Dataset versioning** (reproducibility)
* **Dataset drift monitoring** (detect distribution shifts)

---

## Prompt Lifecycle Management

* **Versioned prompt templates** (track prompt evolution)
* **Prompt registry** (centralized prompt management)
* **Prompt history** (audit trail)
* **A/B prompt testing** (compare prompt variants)
* **Prompt benchmarking** (measure prompt effectiveness)
* **Automatic prompt optimization** (use LLMs to improve prompts)
* **Prompt performance analytics** (per-prompt metrics)

---

## Continuous Learning Systems

* **User feedback ingestion** (thumbs up/down, corrections)
* **Retrieval failure analytics** (identify patterns)
* **Feedback-driven dataset improvement** (use feedback to enhance data)
* **Real-time learning pipelines** (continuous model improvement)
* **Autonomous retrieval improvement** (self-optimizing retrieval)
* **Active learning loops** (identify uncertain predictions for human review)
* **RLHF for retrieval** (reinforce good retrieval behavior)
* **Embedding model fine-tuning** (domain adaptation)
* **Hard negative mining** (improve retrieval precision)

---

## ML Governance & Safety

* **Model version pinning** (control model updates)
* **Index version pinning** (reproducible retrieval)
* **Rollback systems** (revert to previous versions)
* **Dataset lineage tracking** (data provenance)
* **Embedding drift detection** (monitor embedding space changes)
* **Offline vs online pipeline separation** (safe experimentation)
* **CI/CD artifact validation** (prevent bad deployments)
* **Multi-armed bandit testing** (optimize retrieval strategies)
* **Adversarial testing frameworks** (red teaming)
* **Explainability dashboards** (why specific chunks were retrieved)
* **Model card generation** (document model characteristics)

---

## Compliance & Governance

* **Document deletion propagation** (cascade deletes to indexes)
* **GDPR workflows** (right to be forgotten)
* **CCPA compliance** (California privacy law)
* **Retention policy enforcement** (automatic data expiration)
* **Data access logs** (audit who accessed what)
* **Compliance reporting** (automated audit reports)

---

## Cost Intelligence & Optimization

* **Token usage monitoring** (per-query, per-tenant)
* **Embedding cost tracking** (embedding API costs)
* **Reranker cost tracking** (reranking compute costs)
* **LLM cost tracking** (generation costs)
* **Tenant cost attribution** (chargeback/showback)
* **Cost anomaly detection** (detect unexpected spikes)
* **Budget enforcement** (automatic throttling)
* **Cost optimization recommendations** (suggest cheaper alternatives)
* **ROI tracking** (cost vs value delivered)

---

## Advanced Analytics & Insights

* **Query pattern analysis** (identify common queries)
* **User journey analytics** (multi-turn conversation analysis)
* **Topic clustering** (discover user interests)
* **Content gap analysis** (identify missing knowledge)
* **Retrieval coverage analysis** (which docs are most/least used)
* **Performance correlation analysis** (link changes to outcomes)

---

### 🔥 Outcome

✅ Self-monitoring AI system  
✅ Automatically improving accuracy over time  
✅ Regulated, auditable, and compliant  
✅ Research + production convergence  
✅ Minimal manual intervention required  
✅ Continuous optimization and learning  
✅ Frontier-level capabilities  

---

# Visual Evolution Summary

```
L1 → Works (Proof of Concept)
L2 → Reliable (Production MVP)
L3 → Intelligent (Advanced Retrieval)
L4 → Scalable (Enterprise Platform)
L5 → Self-Improving (Autonomous System)
```

---

# Industry Mapping (Architecture Summery)

| Level | Industry Equivalent                                | Companies/Examples                        |
| ----- | -------------------------------------------------- | ----------------------------------------- |
| L1    | Hackathon / demo RAG                               | Weekend projects, PoCs                    |
| L2    | Early production chatbot                           | Startups, internal tools                  |
| L3    | Advanced retrieval research systems                | AI research labs, specialized applications|
| L4    | Enterprise AI SaaS platform                        | Notion AI, Glean, Harvey, Scale AI        |
| L5    | Frontier GenAI platform                            | OpenAI, Anthropic, Google DeepMind        |

---

# Effort vs Impact by Level

| Level | Time to Build | Team Size | Cost         | Business Impact      |
| ----- | ------------- | --------- | ------------ | -------------------- |
| L1    | 1-2 weeks     | 1-2       | $0-$1K       | Demo/prototype       |
| L2    | 1-3 months    | 2-4       | $1K-$10K     | MVP, early customers |
| L3    | 3-6 months    | 3-6       | $10K-$50K    | Competitive product  |
| L4    | 6-12 months   | 5-15      | $50K-$500K   | Enterprise scale     |
| L5    | 12-24+ months | 10-30+    | $500K-$5M+   | Industry-leading     |

---

# Progression Strategy

**For Individuals/Startups:**
- L1 → L2: Focus on reliability and basic safety
- L2 → L3: Add intelligent retrieval (reranking, query expansion)
- L3 → L4: Invest in infrastructure and observability
- L4 → L5: Build feedback loops and autonomous improvement

**For Enterprises:**
- Start at L2 (skip L1 prototype)
- Rapidly move to L4 (platform capabilities)
- Selectively adopt L3 features (advanced retrieval)
- L5 features only for critical competitive advantage

---

# Key Takeaway

**RAG maturity is not just about adding features—it's about:**
1. **Reliability** (L2): Can users trust it?
2. **Intelligence** (L3): Does it understand complex queries?
3. **Scale** (L4): Can it serve millions of users?
4. **Autonomy** (L5): Does it improve itself?

Most production RAG systems should target **L3-L4**. L5 is only necessary for companies where RAG is a core competitive advantage.







