from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from flask import Flask, request, jsonify

app = Flask(__name__)

# Load documents from content.txt file
def load_documents_from_file(filename="content.txt"):
    with open(filename, 'r') as file:
        documents = file.readlines()
    # Remove any leading/trailing whitespace
    documents = [doc.strip() for doc in documents]
    return documents

documents = load_documents_from_file()

# Load a sentence transformer model for encoding documents and queries
retriever_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode documents into embeddings
document_embeddings = retriever_model.encode(documents)

# Build a FAISS index for fast retrieval
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
index.add(document_embeddings)

# Step 2: Set up a generative model (e.g., T5 for answer generation)
generative_model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(generative_model_name)
generative_model = AutoModelForSeq2SeqLM.from_pretrained(generative_model_name)

# Step 3: Define the RAG pipeline
def retrieve_relevant_documents(query, top_k=2):
    # Encode the query
    query_embedding = retriever_model.encode([query])
    # Search the FAISS index for the most relevant documents
    distances, indices = index.search(query_embedding, top_k)
    # Retrieve the top-k documents
    relevant_documents = [documents[i] for i in indices[0]]
    return relevant_documents

def generate_answer(query, relevant_documents):
    # Combine the query and retrieved documents into a single input
    input_text = f"question: {query} context: {' '.join(relevant_documents)}"
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    # Generate the answer
    outputs = generative_model.generate(inputs["input_ids"], max_length=50)
    # Decode the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

@app.route('/', methods=['GET'])
def answer_question():
    question = request.args.get('q', '')
    relevant_docs = retrieve_relevant_documents(question)
    answer = generate_answer(question, relevant_docs)
    return jsonify({'answer': answer})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
