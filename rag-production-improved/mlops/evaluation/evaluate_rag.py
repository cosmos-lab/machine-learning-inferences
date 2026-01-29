import time
from app.core.pipeline import RAGPipeline
from app.config.settings import DATA_PATH

SAMPLE_QUERIES = [
    "What is the purpose of this document?",
    "Explain the main topic",
]

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load_from_file(DATA_PATH)

    for q in SAMPLE_QUERIES:
        start = time.time()
        print(pipeline.answer(q))
        print(f"Latency: {(time.time() - start)*1000:.1f} ms")


