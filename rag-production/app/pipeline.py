import os
import json
import time
from datetime import datetime
import faiss

from app.retriever import Retriever
from app.generator import Generator
from app.observability import logger
from app.config import (
    EMBED_MODEL,
    GEN_MODEL,
    TOP_K,
    MAX_NEW_TOKENS,
    INDEX_PATH,
    META_PATH,
    INDEX_DIR,
    META_DIR,
)


class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever(EMBED_MODEL, TOP_K)
        self.generator = Generator(GEN_MODEL, MAX_NEW_TOKENS)
        self.current_doc_path = None

        os.makedirs(INDEX_DIR, exist_ok=True)
        os.makedirs(META_DIR, exist_ok=True)

    def load_from_file(self, file_path: str, force_rebuild: bool = False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            docs = [l.strip() for l in f if l.strip()]

        self.current_doc_path = file_path

        if os.path.exists(INDEX_PATH) and not force_rebuild:
            index = faiss.read_index(INDEX_PATH)
            self.retriever.load_index(index, docs)
            logger.info("index_loaded", extra={"path": INDEX_PATH})
        else:
            self.retriever.build(docs)
            faiss.write_index(self.retriever.index, INDEX_PATH)

            meta = {
                "embed_model": EMBED_MODEL,
                "generator_model": GEN_MODEL,
                "top_k": TOP_K,
                "doc_path": file_path,
                "created_at": datetime.utcnow().isoformat(),
            }

            with open(META_PATH, "w") as f:
                json.dump(meta, f, indent=2)

            logger.info("index_persisted", extra={"path": INDEX_PATH})

        logger.info(
            "document_load",
            extra={"file_path": file_path, "force_rebuild": force_rebuild},
        )

    def answer(self, question: str) -> str:
        start = time.time()

        context = self.retriever.retrieve(question)

        if not context:
            logger.info(
                "rag_request",
                extra={
                    "event": "no_results",
                    "latency_ms": int((time.time() - start) * 1000),
                },
            )
            return "No relevant information found."

        try:
            answer = self.generator.generate(question, context)
        except Exception as e:
            logger.error("generator_error", exc_info=True)
            return f"Error generating answer: {str(e)}"

        logger.info(
            "rag_request",
            extra={
                "event": "success",
                "retrieved_chunks": len(context),
                "latency_ms": int((time.time() - start) * 1000),
            },
        )

        return answer
