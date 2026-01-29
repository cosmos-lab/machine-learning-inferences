from fastapi import FastAPI, Query
from app.core.async_pipeline import AsyncRAGPipeline
from app.config.settings import DATA_PATH

app = FastAPI(title="Optimized RAG API")

pipeline = AsyncRAGPipeline()
pipeline.load_from_file(DATA_PATH)

@app.get("/ask")
async def ask(q: str = Query(...)):
    return {"question": q, "answer": await pipeline.a_answer(q)}

@app.get("/health")
async def health():
    return {"status": "ok"}
