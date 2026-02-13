# OS module for directory creation & file existence checks
import os

# JSON used for saving chunks + metadata to disk
import json

# FAISS → vector database used for similarity search
import faiss

# Used for saving index creation timestamp
from datetime import datetime


# Retriever:
# Responsible for converting text → embeddings
# and performing similarity search on FAISS index
from app.retrieval.retriever import Retriever


# DocumentChunker:
# Splits large documents into smaller semantic chunks
# (because embedding models have token limits)
from app.retrieval.chunker import DocumentChunker


# Generator:
# HuggingFace Seq2Seq model used to generate final answer
from app.generation.generator import Generator


# track():
# Observability tool for measuring latency of each pipeline step
from app.observability.metrics import track


# Load config values from settings
from app.config.settings import (
    EMBED_MODEL,          # embedding model name
    GEN_MODEL,            # generation model name
    TOP_K,                # number of chunks to retrieve
    MAX_NEW_TOKENS,       # max tokens for LLM output
    INDEX_PATH,           # FAISS index storage path
    META_PATH,            # metadata storage path
    CHUNKS_PATH,          # chunk storage path
    INDEX_DIR,            # directory for FAISS index
    META_DIR,             # directory for metadata
    CHUNK_SIZE,           # chunk length
    CHUNK_OVERLAP,        # overlapping tokens between chunks
    CHUNKING_STRATEGY,    # semantic / fixed chunking
    ENABLE_METADATA,      # enable chunk-level filtering
)


# -----------------------------
# Main RAG Pipeline Class
# -----------------------------
class RAGPipeline:

    # Constructor runs when pipeline object is created
    def __init__(self):

        # Retriever converts text → embeddings
        # and searches FAISS vector database
        self.retriever = Retriever(
            EMBED_MODEL,
            TOP_K,
            enable_metadata=ENABLE_METADATA
        )

        # Generator loads HuggingFace Seq2Seq model
        # used to generate answers from retrieved context
        self.generator = Generator(GEN_MODEL, MAX_NEW_TOKENS)

        # Chunker splits documents into smaller parts
        # before embedding (important for semantic search)
        self.chunker = DocumentChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            strategy=CHUNKING_STRATEGY,
        )

        # Ensure directories exist for storing index + metadata
        os.makedirs(INDEX_DIR, exist_ok=True)
        os.makedirs(META_DIR, exist_ok=True)


    # ------------------------------------------
    # Load document → Chunk → Embed → Index
    # ------------------------------------------
    def load_from_file(self, file_path: str, force_rebuild: bool = False):

        # Read entire document from disk
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()


        # If FAISS index already exists AND rebuild not forced
        if (
            os.path.exists(INDEX_PATH)
            and os.path.exists(CHUNKS_PATH)
            and not force_rebuild
        ):

            # Load FAISS vector index from disk
            index = faiss.read_index(INDEX_PATH)

            # Load saved text chunks + metadata
            with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)
                chunks = chunks_data["chunks"]
                metadata = chunks_data.get("metadata", [])

            # Load index + chunks into retriever memory
            self.retriever.load_index(index, chunks, metadata)


        else:
            # ----------------------------
            # Chunk the document
            # ----------------------------
            # Split text into smaller semantic pieces
            with track("chunking"):
                chunks = self.chunker.chunk_text(text)


            # ----------------------------
            # Create metadata per chunk
            # ----------------------------
            metadata = []

            if ENABLE_METADATA:
                for i, chunk in enumerate(chunks):
                    metadata.append({
                        "chunk_id": i,          # unique ID
                        "source": file_path,    # original file
                        "chunk_size": len(chunk),
                        "strategy": CHUNKING_STRATEGY,
                    })


            # ----------------------------
            # Build FAISS vector index
            # ----------------------------
            with track("indexing"):
                # Retriever internally converts:
                # chunk → embedding → stores in FAISS index
                self.retriever.build(chunks, metadata)

                # Save FAISS index to disk
                faiss.write_index(self.retriever.index, INDEX_PATH)


            # ----------------------------
            # Save chunk data
            # ----------------------------
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


            # ----------------------------
            # Save index metadata
            # ----------------------------
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


    # ------------------------------------------
    # RAG Question Answering Step
    # ------------------------------------------
    def answer(self, question: str, filters: dict = None) -> str:

        # Retrieve top_k most similar chunks from FAISS
        with track("retrieval"):
            context = self.retriever.retrieve(question, filters=filters)

        # If nothing retrieved
        if not context:
            return "No relevant information found."

        # Pass retrieved context to LLM for generation
        with track("generation"):
            return self.generator.generate(question, context)
