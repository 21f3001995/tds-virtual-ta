import json
import faiss
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Paths
DATA_PATH = "tds_discourse_posts.json"
FAISS_INDEX_PATH = "vector_index/faiss.index"
METADATA_PATH = "vector_index/metadata.pkl"

# Load scraped posts
with open(DATA_PATH, "r", encoding="utf-8") as f:
    posts = json.load(f)

# Filter and prepare texts for embedding
texts = []
metadata = []

for post in posts:
    text = post.get("content", "").strip()
    if text:
        texts.append(text)
        metadata.append({
            "url": post.get("post_url", ""),
            "text": text,
            "post_id": post.get("post_number", ""),
            "topic_id": post.get("topic_id", ""),
            "title": post.get("topic_title", "")
        })


print(f"[INFO] Found {len(texts)} valid posts to index.")

# Embed using SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

# Save index and metadata
os.makedirs("vector_index", exist_ok=True)
faiss.write_index(index, FAISS_INDEX_PATH)
with open(METADATA_PATH, "wb") as f:
    pickle.dump(metadata, f)

print(f"[DONE] Indexed {len(embeddings)} posts.")
print(f"→ FAISS index saved to {FAISS_INDEX_PATH}")
print(f"→ Metadata saved to {METADATA_PATH}")
