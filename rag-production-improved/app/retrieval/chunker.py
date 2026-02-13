import re                             # Regex used for fallback sentence splitting
from typing import List               # Type hint for returning list of chunks
from app.observability.logger import logger   # Logging for monitoring chunking performance

# Try to import NLTK, but make it optional
# Because production containers may not always have NLTK data preinstalled
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')   # Try new punkt_tab tokenizer (newer nltk)
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)   # Download silently
        except:
            # Fallback to old punkt if punkt_tab fails
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True          # Flag indicating high-quality sentence tokenizer is available
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using regex-based sentence splitting")
    # IMPORTANT:
    # Bad sentence splitting = bad semantic chunking
    # Bad chunking = bad embeddings
    # Bad embeddings = bad Vector DB retrieval accuracy later

class DocumentChunker:
    """
    Semantic document chunker that splits text intelligently based on:
    - Semantic boundaries (paragraphs, sentences)
    - Token/character limits
    - Overlap for context preservation
    
    NOTE:
    This class directly controls embedding quality
    Which later controls Vector DB recall performance in RAG
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, strategy: str = "semantic"):
        """
        chunk_size      -> max characters per chunk before embedding
        chunk_overlap   -> preserves context across chunks
        strategy        -> chunking strategy

        VECTOR DB NOTE:
        Chunk size strongly affects ANN search performance:
            too small  -> semantic meaning lost
            too large  -> embedding dilution

        Recommended for Vector DB:
            chunk_size ≈ 400–800
            overlap ≈ 10–25%
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy

    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Converts paragraph into list of sentences.
        Sentence boundary detection is critical for semantic embedding.
        """
        if NLTK_AVAILABLE:
            return nltk.sent_tokenize(text)   # NLP based tokenizer (better)
        else:
            # Regex-based fallback tokenizer
            # Splits on punctuation followed by capital letter
            pattern = r'(?<=[.!?])\s+(?=[A-Z])'
            sentences = re.split(pattern, text)
            return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> List[str]:
        """
        Main chunking pipeline entry point.
        This is what your Retriever should call BEFORE embedding.
        
        VECTOR DB INSERTION FLOW:
        Raw Doc → Chunk Here → Embed → Store in Vector DB
        """

        # Choose chunking strategy dynamically
        if self.strategy == "semantic":
            chunks = self._semantic_chunk(text)
        elif self.strategy == "recursive":
            chunks = self._recursive_chunk(text)
        elif self.strategy == "sentence":
            chunks = self._sentence_chunk(text)
        else:
            # Fallback basic sliding window
            chunks = self._simple_chunk(text)

        # Observability for RAG ingestion pipeline
        logger.info(
            "chunking_complete",
            extra={
                "strategy": self.strategy,
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
            }
        )

        # VECTOR DB NOTE:
        # These chunks will be passed to:
        # SentenceTransformer.encode()
        # Then stored using:
        # vector_db.upsert(embeddings, metadata)

        return chunks

    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Best strategy for Vector DB based RAG.

        Steps:
        Paragraph Split → Sentence Split → Overlap Preserve

        Produces semantically meaningful chunks
        Improves ANN recall inside FAISS / Milvus / Pinecone
        """

        # Split into paragraphs or markdown headers
        paragraphs = re.split(r'\n\s*\n+|(?=^#{1,6}\s)', text, flags=re.MULTILINE)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:

            # Try to append paragraph into current chunk
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:

                # Save previous chunk before overflow
                if current_chunk:
                    chunks.append(current_chunk)

                # If paragraph itself too large -> split by sentences
                if len(para) > self.chunk_size:
                    sentence_chunks = self._split_long_text_by_sentences(para)
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    # Add overlap from previous chunk
                    if chunks:
                        overlap_text = self._get_overlap(chunks[-1])
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _simple_chunk(self, text: str) -> List[str]:
        """
        Sliding window fallback chunking.
        FAST but worst for Vector DB semantic search.
        Should not be used in production RAG.
        """

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # Overlap helps semantic continuity across embeddings
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _get_overlap(self, text: str) -> str:
        """
        Extracts last overlap characters from previous chunk.
        Helps maintain semantic continuity.

        Without this:
        Vector DB ANN may miss relevant matches across chunk boundary
        """

        if len(text) <= self.chunk_overlap:
            return text

        overlap = text[-self.chunk_overlap:]

        # Prefer sentence boundary overlap
        sentences = self._tokenize_sentences(overlap)
        if len(sentences) > 1:
            return " ".join(sentences[1:])

        # fallback word boundary
        first_space = overlap.find(' ')
        if first_space != -1:
            return overlap[first_space + 1:]

        return overlap
