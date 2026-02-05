from fastapi import FastAPI, Query, Body
from typing import Optional, Dict
from app.core.async_pipeline import AsyncRAGPipeline
from app.config.settings import DATA_PATH

app = FastAPI(title="Optimized RAG API")

pipeline = AsyncRAGPipeline()
pipeline.load_from_file(DATA_PATH)

@app.get("/ask")
async def ask(q: str = Query(...)):
    return {"question": q, "answer": await pipeline.a_answer(q)}

@app.post("/ask")
async def ask_with_filters(
    q: str = Body(..., embed=True),
    filters: Optional[Dict] = Body(None, embed=True)
):
    """
    Ask a question with optional metadata filters.
    
    Example:
    {
        "q": "What is AI?",
        "filters": {
            "source": "data/doc1.txt",
            "chunk_size": {"$gte": 100, "$lte": 600}
        }
    }
    """
    return {"question": q, "answer": await pipeline.a_answer(q, filters=filters)}

@app.get("/health")
async def health():
    return {"status": "ok"}


