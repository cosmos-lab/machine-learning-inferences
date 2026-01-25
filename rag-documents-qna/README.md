# RAG-Based Document Question Answering API

## Objective

This project provides a **production-ready API** for answering questions based on multiple documents or URLs using **Retrieval-Augmented Generation (RAG)** with Hugging Face models.

The API is designed for real-world use cases where users ask dynamic questions and receive **context-aware answers grounded strictly in the provided documents**.

---

## Key Features

* Retrieval-Augmented Generation (RAG) pipeline
* Document-based question answering
* FAISS-based vector search
* Hugging Face embedding and generation models
* Grounded, context-aware responses
* Suitable for production deployment

---

## Architecture Overview

The application follows a standard **three-layer RAG architecture**:

### 1. Document Ingestion & Embedding

* Load documents from files or URLs
* Split documents into chunks
* Convert chunks into vector embeddings
* Store embeddings in a vector database (FAISS)

### 2. Retrieval Layer

* Embed the user query
* Perform similarity search on the vector database
* Retrieve top-K relevant document chunks
* Assemble retrieved chunks as context

### 3. Generation Layer

* Combine retrieved context with the user query
* Pass both to a language model
* Generate a grounded answer based only on the context

---

## RAG Pipeline Flow

1. Load documents
2. Split documents into chunks
3. Generate embeddings
4. Store embeddings in FAISS
5. User submits a question
6. Embed the query
7. Retrieve top-K relevant chunks
8. Generate an answer using the LLM

---

## Production Considerations

* Persist FAISS indexes to avoid recomputing embeddings on restart
* Cache repeated queries for better performance
* Track source documents for every answer (important for trust and debugging)
* Design for scalability (LLM can later be moved to a remote server or API)

---

## Hardware Notes

This project is designed to run on modest hardware, such as:

* 16 GB RAM systems
* CPU-based inference or limited GPU support

For larger models or higher throughput, offloading the generation model to a dedicated server or API is recommended.

---

## Build and Run

```bash

podman build -t rag_api .

podman run --rm \
  -p 8000:8000 \
  -v $(pwd)/app:/app/app \
  -v $(pwd)/data:/app/data \
  rag_api

```

## API URLS

http://localhost:8000/ask?q=What documents are required for GST registration?


GET http://localhost:8000/reload?doc=data/doc1.txt

---

## Summary

This project demonstrates a clean and extensible implementation of a **Retrieval-Augmented Generation API**, suitable for document-based question answering systems in production environments.

---

# RAG Test Suite: UN & UNESCO

Use the following questions to verify the performance of the embedding retrieval and generative response quality.

### Document 1: United Nations (UN)
*   What is the primary articulated mission of the United Nations?
*   Who are the current Secretary-General and the President of the General Assembly as of January 2026?
*   How many member states and observer states are currently part of the UN?
*   What are the six principal organizations that make up the United Nations System?
*   In which five cities are the primary UN headquarters and offices located?
*   Which 1944 conference formulated the structure of the UN before its official 1945 founding?
*   How many active peacekeeping missions does the UN operate, and in which regions?

### Document 2: UNESCO
*   What are the five major program areas managed by UNESCO?
*   Where is UNESCO headquartered and how many field offices does it maintain?
*   On what date did the UNESCO Constitution come into force, and who was the first Director-General?
*   Which 1960 campaign led to the protection of the Abu Simbel monuments?
*   Which famous scientists were members of the International Committee on Intellectual Cooperation (ICIC)?
*   What occurred in 1950 regarding UNESCO's stance on racism and anthropologists?
*   How often does the UNESCO General Conference meet to govern the organization?



Absolutely! Here’s a **comprehensive, double-checked TODO checklist** for your RAG API repository, covering **functionality, MLOps, deployment, and best practices**. You can include this in your README under a **“TODO / Future Improvements”** section.

---


# TODO / Future Improvements for RAG API

### Core Functionality

* Ensure **RAG retrieval + generative answer** works for all queries.
* Deduplicate retrieved lines to avoid repetition in generated answers.
* Handle empty or missing documents gracefully with error messages.

### MLOps & Versioning

* Version **embedding models** (e.g., `all-MiniLM-L6-v2`) and track changes.
* Version **generative model** (e.g., `flan-t5-base`) and prompt configuration.
* Save and version **FAISS indices** for reproducibility.
* Maintain **model + embedding + document manifest** for reproducibility.

### Logging & Monitoring

* Log **incoming queries** with timestamps.
* Log **response times** for retrieval and generation.
* Log **errors** (missing documents, generation failures).
* Optional: Integrate **Prometheus/Grafana** for metrics.
* Optional: Integrate **Sentry** for error tracking.

### Testing & CI/CD

* Unit tests for:

  * `retrieve()` function correctness
  * `generate_answer()` formatting and completeness
* Integration tests for `/ask` endpoint (test API output for known queries).
* CI/CD workflow (GitHub Actions / GitLab CI):

  * Run tests automatically on push / pull requests
  * Optional: Build and push Docker image
* Docker image rebuild automation with minimal downtime.

### Deployment & Scaling

* Containerized deployment (Docker / Podman).
* Limit concurrency for CPU-bound inference (16 GB RAM constraint).
* Optional: Separate embedding and generation services for scaling.
* Optional: GPU support for faster generation.
* Add **reload endpoint** to dynamically switch documents without restarting.

### Prompt & Model Optimization

* Tune **MAX_NEW_TOKENS** for multi-line answers.
* Tune **num_beams**, `do_sample`, `top_p`, `temperature` for readability + completeness.
* Ensure **instructions prevent hallucination**.
* Optionally support **multi-document queries**.

### Documentation & Usability

* Add instructions for **adding new documents**.
* Add **MLOps / production practices** section.

### Optional Enhancements

* Add **caching layer** for repeated queries.
* Add **source tracking** for answers (which line/doc they came from).
* Add **multi-language support** if documents in multiple languages.
* Add **user-friendly front-end** (simple React/Vue/Streamlit interface).

