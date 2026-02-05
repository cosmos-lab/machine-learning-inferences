import re
from typing import List
from app.observability.logger import logger

# Try to import NLTK, but make it optional
try:
    import nltk
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except:
            # Fallback to old punkt if punkt_tab fails
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK not available, using regex-based sentence splitting")

class DocumentChunker:
    """
    Semantic document chunker that splits text intelligently based on:
    - Semantic boundaries (paragraphs, sentences)
    - Token/character limits
    - Overlap for context preservation
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, strategy: str = "semantic"):
        """
        Args:
            chunk_size: Maximum characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            strategy: Chunking strategy - "semantic", "recursive", or "sentence"
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
    
    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Tokenize text into sentences.
        Uses NLTK if available, otherwise falls back to regex.
        """
        if NLTK_AVAILABLE:
            return nltk.sent_tokenize(text)
        else:
            # Regex-based sentence splitting
            # Split on .!? followed by space and capital letter
            pattern = r'(?<=[.!?])\s+(?=[A-Z])'
            sentences = re.split(pattern, text)
            return [s.strip() for s in sentences if s.strip()]
        
    def chunk_text(self, text: str) -> List[str]:
        """Main entry point for chunking text."""
        if self.strategy == "semantic":
            chunks = self._semantic_chunk(text)
        elif self.strategy == "recursive":
            chunks = self._recursive_chunk(text)
        elif self.strategy == "sentence":
            chunks = self._sentence_chunk(text)
        else:
            # Fallback to simple splitting
            chunks = self._simple_chunk(text)
        
        logger.info(
            "chunking_complete",
            extra={
                "strategy": self.strategy,
                "total_chunks": len(chunks),
                "avg_chunk_size": sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
            }
        )
        
        return chunks
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Semantic chunking: splits on paragraphs first, then sentences if needed.
        Preserves semantic coherence by keeping related content together.
        """
        # Split into paragraphs (double newlines or markdown headers)
        paragraphs = re.split(r'\n\s*\n+|(?=^#{1,6}\s)', text, flags=re.MULTILINE)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # If paragraph fits in current chunk with overlap
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + para
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too large, split by sentences
                if len(para) > self.chunk_size:
                    sentence_chunks = self._split_long_text_by_sentences(para)
                    chunks.extend(sentence_chunks[:-1])
                    current_chunk = sentence_chunks[-1] if sentence_chunks else ""
                else:
                    # Start new chunk with overlap from previous
                    if chunks:
                        overlap_text = self._get_overlap(chunks[-1])
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _recursive_chunk(self, text: str) -> List[str]:
        """
        Recursive chunking: tries multiple separators in order of priority.
        Hierarchical: paragraphs -> sentences -> words
        """
        separators = [
            "\n\n",      # Paragraphs
            "\n",        # Lines
            ". ",        # Sentences
            "! ",        # Exclamations
            "? ",        # Questions
            "; ",        # Semi-colons
            ", ",        # Commas
            " ",         # Words
        ]
        
        return self._recursive_split(text, separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Helper for recursive chunking."""
        if not separators:
            # Base case: split by characters
            return self._split_by_chars(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        chunks = []
        splits = text.split(separator)
        
        current_chunk = ""
        for i, split in enumerate(splits):
            # Add separator back except for last split
            test_chunk = current_chunk + (separator if current_chunk else "") + split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If split itself is too large, recurse with next separator
                if len(split) > self.chunk_size:
                    sub_chunks = self._recursive_split(split, remaining_separators)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    # Start new chunk with overlap
                    if chunks:
                        overlap = self._get_overlap(chunks[-1])
                        current_chunk = overlap + (separator if overlap else "") + split
                    else:
                        current_chunk = split
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _sentence_chunk(self, text: str) -> List[str]:
        """Simple sentence-based chunking using NLTK or regex fallback."""
        sentences = self._tokenize_sentences(text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle very long sentences
                if len(sentence) > self.chunk_size:
                    sub_chunks = self._split_by_chars(sentence)
                    chunks.extend(sub_chunks[:-1])
                    current_chunk = sub_chunks[-1] if sub_chunks else ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _simple_chunk(self, text: str) -> List[str]:
        """Fallback: simple sliding window chunking."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def _split_long_text_by_sentences(self, text: str) -> List[str]:
        """Split long text by sentences while respecting chunk_size."""
        sentences = self._tokenize_sentences(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            test_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(sentence) > self.chunk_size:
                    # Split by words if sentence is too long
                    word_chunks = self._split_by_chars(sentence)
                    chunks.extend(word_chunks[:-1])
                    current_chunk = word_chunks[-1] if word_chunks else ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_chars(self, text: str) -> List[str]:
        """Split text by character count with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > self.chunk_size // 2:  # Only break if space is not too early
                    chunk = chunk[:last_space]
                    end = start + last_space
            
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        
        overlap = text[-self.chunk_overlap:]
        
        # Try to start at sentence boundary
        sentences = self._tokenize_sentences(overlap)
        if len(sentences) > 1:
            return " ".join(sentences[1:])
        
        # Otherwise try word boundary
        first_space = overlap.find(' ')
        if first_space != -1:
            return overlap[first_space + 1:]
        
        return overlap
