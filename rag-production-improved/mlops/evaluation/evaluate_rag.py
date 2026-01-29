import time
from app.core.pipeline import RAGPipeline
from app.config.settings import DATA_PATH
from app.observability.logger import logger

SAMPLE_QUERIES = [
    "What is the purpose of this document?",
    "Explain the main topic",
]

if __name__ == "__main__":
    pipeline = RAGPipeline()
    pipeline.load_from_file(DATA_PATH)

    for query in SAMPLE_QUERIES:
        start = time.time()
        answer = pipeline.answer(query)
        latency = (time.time() - start) * 1000
        print(f"Query: {query}\nAnswer: {answer}\nLatency: {latency:.1f} ms\n")
        logger.info("evaluate_rag_query", extra={"query": query, "latency_ms": int(latency)})

    print("Evaluation complete")


