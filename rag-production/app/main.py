from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.pipeline import RAGPipeline
from app.config import DATA_PATH

app = FastAPI(title="Lightweight Extractive RAG API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()
pipeline.load_from_file(DATA_PATH)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

# Async endpoints
@app.get("/ask")
async def ask(q: str = Query(..., description="Question to ask")):
    return {
        "question": q,
        "answer": pipeline.answer(q),
    }

@app.get("/reload")
async def reload(doc: str = Query(None, description="Document path to reload")):
    path = doc if doc else DATA_PATH
    pipeline.load_from_file(path, force_rebuild=True)
    return {
        "status": "reloaded",
        "document": path,
    }

@app.get("/health")
async def health():
    return {"status": "ok"}
