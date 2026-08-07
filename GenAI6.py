import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------------
# 1. Knowledge Base
# -------------------------------

documents = [
"""Generative Artificial Intelligence is a branch of AI that creates
new content such as text, images, audio, video and computer programs.""",

"""Large Language Models are transformer-based models trained on massive
text datasets. They are used for text generation, summarization,
translation, question answering and conversational AI.""",

"""Retrieval-Augmented Generation combines information retrieval with
text generation. It retrieves relevant documents from an external
knowledge base and gives them to a language model as context.""",

"""Vector databases store high-dimensional embeddings and perform
similarity searches. Examples include FAISS, ChromaDB, Pinecone,
Weaviate and Milvus.""",

"""Prompt engineering is the process of designing clear instructions
that guide a language model to produce accurate and useful responses.""",

"""Fine-tuning adapts a pretrained language model to a specific domain
or task using a smaller dataset."""
]

# -------------------------------
# 2. Embedding Model
# -------------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert documents to embeddings
document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")

# Normalize embeddings
faiss.normalize_L2(document_embeddings)

# -------------------------------
# 3. FAISS Vector DB
# -------------------------------

dimension = document_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(document_embeddings)

# -------------------------------
# 4. Load FLAN-T5 (FIXED VERSION)
# -------------------------------

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# -------------------------------
# 5. Retrieval Function
# -------------------------------

def retrieve_documents(query, top_k=2):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for i, score in zip(indices[0], scores[0]):
        results.append({
            "document": documents[i],
            "score": float(score)
        })

    return results

# -------------------------------
# 6. Answer Generation
# -------------------------------

def generate_answer(query, retrieved_docs):
    context = "\n\n".join([doc["document"] for doc in retrieved_docs])

    prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------------
# 7. Run System
# -------------------------------

print("RAG SYSTEM")
print("=" * 50)

query = input("\nEnter your question: ")

retrieved = retrieve_documents(query)

answer = generate_answer(query, retrieved)

# -------------------------------
# 8. Output
# -------------------------------

print("\nRETRIEVED DOCUMENTS")
print("-" * 50)

for i, item in enumerate(retrieved, 1):
    print(f"\nDocument {i}:")
    print(item["document"])
    print(f"Score: {item['score']:.4f}")

print("\nGENERATED ANSWER")
print("-" * 50)
print(answer)