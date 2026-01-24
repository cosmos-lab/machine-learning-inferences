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
podman run --rm -p 8000:8000 rag_api
```

The API will be available at `http://localhost:8000`.

---

## Summary

This project demonstrates a clean and extensible implementation of a **Retrieval-Augmented Generation API**, suitable for document-based question answering systems in production environments.

---

