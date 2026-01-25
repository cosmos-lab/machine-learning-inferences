"""
Purpose:
Offline script to build FAISS index and metadata artifacts for the RAG system.
This is used to generate reproducible embeddings and index files for production deployment.
"""

from app.pipeline import RAGPipeline
from app.config import DATA_PATH
from app.observability import logger

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load_from_file(DATA_PATH, force_rebuild=True)
    logger.info("build_artifacts_complete", extra={"data_path": DATA_PATH})
    print(f"Artifacts built and saved for {DATA_PATH}")
