from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import base64, io, pickle, numpy as np
from PIL import Image
import pytesseract, json
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once on startup
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2")
faiss_index = faiss.read_index("vector_index/faiss.index")
with open("vector_index/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Models
class Link(BaseModel):
    url: str
    text: str

class Answer(BaseModel):
    answer: str
    links: List[Link]

# OCR
def extract_text_from_image(base64_image: str) -> str:
    try:
        image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        return ""

@app.get("/")
def home():
    return {
        "message": "✅ TDS Virtual TA is live. Use POST / with JSON: {\"question\": \"...\"}"
    }

@app.post("/", response_model=Answer)
async def answer_query(request: Request):
    try:
        body = await request.json()
    except:
        return {"answer": "Invalid JSON format", "links": []}

    question = body.get("question", "").strip()
    image_base64 = body.get("image")
    if image_base64:
        question += "\n" + extract_text_from_image(image_base64)

    if not question:
        return {"answer": "No valid question provided.", "links": []}

    embedding = bi_encoder.encode([question])[0].astype("float32")
    _, I = faiss_index.search(np.array([embedding]), k=5)
    top_chunks = [metadata[i] for i in I[0] if i < len(metadata)]

    if not top_chunks:
        return {"answer": "No relevant content found.", "links": []}

    rerank_scores = reranker.predict([(question, c["text"]) for c in top_chunks])
    ranked_chunks = [c for _, c in sorted(zip(rerank_scores, top_chunks), key=lambda x: -x[0])]
    relevant = ranked_chunks[:3]

    answer = "Based on the content, here's what I found relevant:\n"
    links = []
    for c in relevant:
        snippet = c["text"][:300].replace("\n", " ").strip()
        answer += f"- {snippet}...\n"
        if c.get("url"):
            links.append(Link(url=c["url"], text=c.get("title", "Source")))

    return Answer(answer=answer.strip(), links=links[:3])
