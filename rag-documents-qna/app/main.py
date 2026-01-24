# app/main.py

from fastapi import FastAPI, Query
from app.rag import answer

app = FastAPI(title="Lightweight RAG API")

@app.get("/ask")
def ask(q: str = Query(..., description="Question to ask the RAG system")):
    return {
        "question": q,
        "answer": answer(q)
    }
