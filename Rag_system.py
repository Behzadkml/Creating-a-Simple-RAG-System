documents = [
    "Python is a programming language created by Guido van Rossum.",
    "Machine learning is a field of artificial intelligence that uses statistical techniques.",
    "The capital city of France is Paris.",
    "TensorFlow and PyTorch are popular machine learning frameworks.",
    "FAISS is a library for efficient similarity search."
]
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for documents
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# Build FAISS index
embedding_dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(doc_embeddings)
def retrieve_docs(query, top_k=2):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    return [documents[idx] for idx in indices[0]]
from transformers import pipeline

generator = pipeline('text-generation', model='gpt2')

def generate_response(query, context_docs):
    context = " ".join(context_docs)
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = generator(prompt, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']
def rag(query):
    relevant_docs = retrieve_docs(query)
    answer = generate_response(query, relevant_docs)
    return answer

