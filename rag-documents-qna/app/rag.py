# app/rag.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from app.index import build_index, load_documents, embed_model
from app.config import GEN_MODEL, TOP_K, MAX_NEW_TOKENS

tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL)
model.eval()

index, documents = build_index(load_documents())

def retrieve(query, k=10): 
    q_vec = embed_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    _, I = index.search(q_vec, k)

    chunks = []
    for i in I[0]:
        chunks.append(f"- {documents[i]}")

    return "\n".join(dict.fromkeys(chunks)) 

def answer(query):
    context = retrieve(query)

    prompt = f"""
question: {query}

context:
{context}

answer:
"""

    with torch.no_grad():
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=30, 
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

