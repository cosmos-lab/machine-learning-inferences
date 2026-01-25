#rag.py

from app.index import build_index, load_documents, embed_model
from app.config import TOP_K, GEN_MODEL, MAX_NEW_TOKENS, DATA_PATH
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

_index = None
_documents = None
_lock = threading.Lock()
_current_data_path = DATA_PATH  # track current document

# Load the generative model for user-friendly answers
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
model.eval()

def is_fact(line: str) -> bool:
    return line.endswith(".")

def load_index(doc_path=None):
    """Load documents and build FAISS index. If doc_path is given, use that."""
    global _index, _documents, _current_data_path
    if doc_path:
        _current_data_path = doc_path
    _documents = load_documents(_current_data_path)
    _index, _ = build_index(_documents)

def get_index():
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                load_index()
    return _index, _documents

def retrieve(query, k=TOP_K):
    index, documents = get_index()
    q_vec = embed_model.encode([query], normalize_embeddings=True).astype("float32")
    _, I = index.search(q_vec, k)

    results = []
    for i in I[0]:
        line = documents[i]
        if is_fact(line):
            results.append(line)

    return list(dict.fromkeys(results))

def generate_answer(query, context_lines):
    context_text = "\n".join(context_lines)
    prompt = f"""
Answer the question clearly using ONLY the facts below.
If there are multiple items, list all of them in one answer in a readable way.
Do NOT add anything extra.
Context:
{context_text}

Question:
{query}

Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=4,
            top_p=0.9,         
            temperature=0.7,
            early_stopping=True 
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def answer(query):
    facts = retrieve(query)
    if not facts:
        return "No relevant information found."
    return generate_answer(query, facts)

def reload_data(doc_name=None):
    """Reload documents and rebuild FAISS index. Optionally specify doc_name."""
    global _index, _documents
    with _lock:
        _index = None
        _documents = None
        path_to_load = doc_name if doc_name else _current_data_path
        if not os.path.exists(path_to_load):
            raise FileNotFoundError(f"Document not found: {path_to_load}")
        load_index(path_to_load)
