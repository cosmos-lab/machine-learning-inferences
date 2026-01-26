import pytest
from app.generator import Generator

def test_generate_simple():
    generator = Generator("google/flan-t5-base", max_new_tokens=10)
    question = "What is AI?"
    context = ["AI stands for artificial intelligence."]
    answer = generator.generate(question, context)

    assert isinstance(answer, str)
    assert len(answer) > 0
