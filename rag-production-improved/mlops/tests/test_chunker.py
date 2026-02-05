import pytest
from app.retrieval.chunker import DocumentChunker

def test_semantic_chunking():
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20, strategy="semantic")
    text = """This is paragraph one. It has multiple sentences.

This is paragraph two. It also has content.

This is paragraph three with more information."""
    
    chunks = chunker.chunk_text(text)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) <= 120 for c in chunks)  # Allow some buffer

def test_recursive_chunking():
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10, strategy="recursive")
    text = "This is a test sentence. Another sentence here. And one more for good measure."
    
    chunks = chunker.chunk_text(text)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_sentence_chunking():
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=15, strategy="sentence")
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    
    chunks = chunker.chunk_text(text)
    
    assert isinstance(chunks, list)
    assert len(chunks) > 0

def test_long_text_handling():
    chunker = DocumentChunker(chunk_size=200, chunk_overlap=50, strategy="semantic")
    
    # Create a very long text
    text = " ".join(["This is sentence number " + str(i) + "." for i in range(100)])
    
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) > 1
    assert all(len(c) <= 250 for c in chunks)  # Allow buffer


