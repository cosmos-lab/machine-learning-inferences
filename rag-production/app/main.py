from fastapi import FastAPI, Query
from app.pipeline import RAGPipeline
from app.config import DATA_PATH

app = FastAPI(title="Lightweight Extractive RAG API")

pipeline = RAGPipeline()
pipeline.load_from_file(DATA_PATH)


@app.get("/ask")
def ask(q: str = Query(..., description="Question to ask")):
    return {
        "question": q,
        "answer": pipeline.answer(q),
    }


@app.get("/reload")
def reload(doc: str = Query(None, description="Document path to reload")):
    path = doc if doc else DATA_PATH
    pipeline.load_from_file(path, force_rebuild=True)
    return {
        "status": "reloaded",
        "document": path,
    }


@app.get("/health")
def health():
    return {"status": "ok"}
