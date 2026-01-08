import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

INPUT_FILE = "data/processed/corpus.jsonl"
CHROMA_DIR = "data/chroma"
COLLECTION_NAME = "tunisia_archaeo"

# Load embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def chunk_text(text, size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks

def main():
    # Init ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Create or get collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Reading processed corpus...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]
            title = doc["title"]

            chunks = chunk_text(text)

            print(f"Embedding {len(chunks)} chunks from: {title}")

            embeddings = model.encode(chunks).tolist()

            ids = [f"{title}_{i}" for i in range(len(chunks))]

            # Add to Chroma
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=[{
                    "title": title,
                    "source": doc["source"],
                    "site": doc["site"],
                    "lang": doc["lang"]
                }] * len(chunks)
            )

    print("✔️ All chunks embedded and stored in ChromaDB!")
    print(f"Database saved in: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
