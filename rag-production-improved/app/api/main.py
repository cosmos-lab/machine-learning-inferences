from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.pipeline import RAGPipeline
from app.config.settings import DATA_PATH

app = FastAPI(title="Lightweight Extractive RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RAGPipeline()
pipeline.load_from_file(DATA_PATH)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})

@app.get("/ask")
async def ask(q: str = Query(...)):
    return {"question": q, "answer": pipeline.answer(q)}

@app.get("/reload")
async def reload(doc: str = Query(None)):
    pipeline.load_from_file(doc or DATA_PATH, force_rebuild=True)
    return {"status": "reloaded"}

@app.get("/health")
async def health():
    return {"status": "ok"}


