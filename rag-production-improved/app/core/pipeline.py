import os
import json
import faiss
from datetime import datetime
from app.retrieval.retriever import Retriever
from app.generation.generator import Generator
from app.observability.metrics import track
from app.config.settings import (
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

        os.makedirs(INDEX_DIR, exist_ok=True)
        os.makedirs(META_DIR, exist_ok=True)

    def load_from_file(self, file_path: str, force_rebuild: bool = False):
        with open(file_path, "r", encoding="utf-8") as f:
            docs = [l.strip() for l in f if l.strip()]

        if os.path.exists(INDEX_PATH) and not force_rebuild:
            index = faiss.read_index(INDEX_PATH)
            self.retriever.load_index(index, docs)
        else:
            self.retriever.build(docs)
            faiss.write_index(self.retriever.index, INDEX_PATH)

            with open(META_PATH, "w") as f:
                json.dump(
                    {
                        "embed_model": EMBED_MODEL,
                        "generator_model": GEN_MODEL,
                        "top_k": TOP_K,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )

    def answer(self, question: str) -> str:
        with track("retrieval"):
            context = self.retriever.retrieve(question)

        if not context:
            return "No relevant information found."

        with track("generation"):
            return self.generator.generate(question, context)


