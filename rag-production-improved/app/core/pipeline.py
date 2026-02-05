import os
import json
import faiss
from datetime import datetime
from app.retrieval.retriever import Retriever
from app.retrieval.chunker import DocumentChunker
from app.generation.generator import Generator
from app.observability.metrics import track
from app.config.settings import (
    EMBED_MODEL,
    GEN_MODEL,
    TOP_K,
    MAX_NEW_TOKENS,
    INDEX_PATH,
    META_PATH,
    CHUNKS_PATH,
    INDEX_DIR,
    META_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNKING_STRATEGY,
    ENABLE_METADATA,
)

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever(EMBED_MODEL, TOP_K, enable_metadata=ENABLE_METADATA)
        self.generator = Generator(GEN_MODEL, MAX_NEW_TOKENS)
        self.chunker = DocumentChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            strategy=CHUNKING_STRATEGY,
        )

        os.makedirs(INDEX_DIR, exist_ok=True)
        os.makedirs(META_DIR, exist_ok=True)

    def load_from_file(self, file_path: str, force_rebuild: bool = False):
        # Read the entire document
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Check if we can load cached index and chunks
        if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH) and not force_rebuild:
            index = faiss.read_index(INDEX_PATH)
            
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                chunks = chunks_data["chunks"]
                metadata = chunks_data.get("metadata", [])
            
            self.retriever.load_index(index, chunks, metadata)
        else:
            # Chunk the document using semantic chunking
            with track("chunking"):
                chunks = self.chunker.chunk_text(text)
            
            # Create metadata for each chunk
            metadata = []
            if ENABLE_METADATA:
                for i, chunk in enumerate(chunks):
                    metadata.append({
                        "chunk_id": i,
                        "source": file_path,
                        "chunk_size": len(chunk),
                        "strategy": CHUNKING_STRATEGY,
                    })
            
            # Build the index
            with track("indexing"):
                self.retriever.build(chunks, metadata)
                faiss.write_index(self.retriever.index, INDEX_PATH)
            
            # Save chunks with metadata
            with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "chunks": chunks,
                        "metadata": metadata,
                        "total_chunks": len(chunks),
                        "chunking_strategy": CHUNKING_STRATEGY,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                    f,
                    indent=2,
                )

            # Save index metadata
            with open(META_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "embed_model": EMBED_MODEL,
                        "generator_model": GEN_MODEL,
                        "top_k": TOP_K,
                        "chunking_strategy": CHUNKING_STRATEGY,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                        "total_chunks": len(chunks),
                        "metadata_enabled": ENABLE_METADATA,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                    f,
                    indent=2,
                )

    def answer(self, question: str, filters: dict = None) -> str:
        with track("retrieval"):
            context = self.retriever.retrieve(question, filters=filters)

        if not context:
            return "No relevant information found."

        with track("generation"):
            return self.generator.generate(question, context)


