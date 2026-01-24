from fastapi import FastAPI, Query
from app.rag import answer, reload_data

app = FastAPI(title="Lightweight Extractive RAG API")

@app.get("/ask")
def ask(q: str = Query(..., description="Question to ask")):
    return {
        "question": q,
        "answer": answer(q)
    }

@app.get("/reload")
def reload(doc: str = Query(None, description="Document filename to reload")):
    """
    Reload documents and rebuild FAISS index.
    Optionally provide a document filename: ?doc=doc1.txt
    """
    try:
        reload_data(doc)
        return {"status": "documents reloaded", "document": doc if doc else "default"}
    except FileNotFoundError as e:
        return {"status": "error", "message": str(e)}
