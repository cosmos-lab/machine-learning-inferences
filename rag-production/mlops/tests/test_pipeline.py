import pytest
from app.pipeline import RAGPipeline

def test_pipeline_answer():
    pipeline = RAGPipeline()
    pipeline.load_from_file("data/doc1.txt")  # Make sure doc1.txt exists

    question = "What is the main topic?"
    answer = pipeline.answer(question)

    assert isinstance(answer, str)
    assert len(answer) > 0
