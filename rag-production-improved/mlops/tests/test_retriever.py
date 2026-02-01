import pytest
from app.retrieval.retriever import Retriever

@pytest.fixture
def sample_docs():
    return ["Hello world", "Test document", "Another sentence"]

def test_build_and_retrieve(sample_docs):
    retriever = Retriever("sentence-transformers/all-MiniLM-L6-v2", top_k=2)
    retriever.build(sample_docs)

    query = "Hello"
    results = retriever.retrieve(query)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(r, str) for r in results)


