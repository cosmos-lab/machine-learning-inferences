import pytest
from app.retrieval.retriever import Retriever

@pytest.fixture
def sample_docs():
    return ["Hello world", "Test document", "Another sentence"]

@pytest.fixture
def sample_metadata():
    return [
        {"chunk_id": 0, "source": "doc1.txt", "chunk_size": 11},
        {"chunk_id": 1, "source": "doc2.txt", "chunk_size": 13},
        {"chunk_id": 2, "source": "doc1.txt", "chunk_size": 16},
    ]

def test_build_and_retrieve(sample_docs):
    retriever = Retriever("sentence-transformers/all-MiniLM-L6-v2", top_k=2)
    retriever.build(sample_docs)

    query = "Hello"
    results = retriever.retrieve(query)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(r, str) for r in results)

def test_retrieve_with_metadata_filter(sample_docs, sample_metadata):
    retriever = Retriever("sentence-transformers/all-MiniLM-L6-v2", top_k=2, enable_metadata=True)
    retriever.build(sample_docs, sample_metadata)

    query = "Hello"
    
    # Filter by source
    results = retriever.retrieve(query, filters={"source": "doc1.txt"})
    assert isinstance(results, list)
    assert len(results) <= 2

def test_retrieve_with_range_filter(sample_docs, sample_metadata):
    retriever = Retriever("sentence-transformers/all-MiniLM-L6-v2", top_k=2, enable_metadata=True)
    retriever.build(sample_docs, sample_metadata)

    query = "test"
    
    # Filter by chunk size range
    results = retriever.retrieve(query, filters={"chunk_size": {"$gte": 12}})
    assert isinstance(results, list)
