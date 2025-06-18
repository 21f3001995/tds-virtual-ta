from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import base64, io, pickle, gc
import numpy as np
from PIL import Image
import pytesseract
import faiss
from sentence_transformers import SentenceTransformer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Preload everything (safer than lazy under Render)
bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index("vector_index/faiss.index")
with open("vector_index/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

class Link(BaseModel):
    url: str
    text: str

class Answer(BaseModel):
    answer: str
    links: List[Link]

def extract_text_from_image(b64_img: str) -> str:
    try:
        image = Image.open(io.BytesIO(base64.b64decode(b64_img))).convert("RGB")
        return pytesseract.image_to_string(image).strip()
    except:
        return ""

@app.api_route("/", methods=["GET", "HEAD", "OPTIONS"])
async def root(request: Request):
    if request.method == "HEAD":
        return JSONResponse(status_code=200, content={})
    return {"message": "✅ TDS Virtual TA is live. POST to /api with {'question': '...'}"}

@app.post("/api", response_model=Answer)
async def handle_query(request: Request):
    try:
        data = await request.json()
    except:
        return {"answer": "Invalid JSON format", "links": []}

    question = data.get("question", "").strip()
    if not question and data.get("image"):
        question = extract_text_from_image(data["image"])
    elif data.get("image"):
        question += "\n" + extract_text_from_image(data["image"])

    if not question:
        return {"answer": "No valid question provided.", "links": []}

    # Encode and search
    emb = bi_encoder.encode([question])[0].astype("float32")
    _, I = faiss_index.search(np.array([emb]), k=3)
    top_chunks = [metadata[i] for i in I[0] if i < len(metadata)]

    if not top_chunks:
        return {"answer": "No relevant content found.", "links": []}

    # Basic scoring: top 2
    ranked = top_chunks[:2]
    answer = "Based on the content, here's what I found:\n"
    links = []
    for c in ranked:
        snippet = c["text"][:300].replace("\n", " ").strip()
        answer += f"- {snippet}...\n"
        if c.get("url"):
            links.append(Link(url=c["url"], text=c.get("title", "Source")))

    gc.collect()
    return Answer(answer=answer.strip(), links=links[:2])

@app.post("/", response_model=Answer)
async def alias_for_api(request: Request):
    return await handle_query(request)
