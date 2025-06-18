from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import base64, io, pickle, numpy as np
from PIL import Image
import pytesseract, json, gc
from json import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import faiss

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load models
bi_encoder = None
reranker = None
faiss_index = None
metadata = None

def extract_text_from_image(base64_image: str) -> str:
    try:
        image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
        return pytesseract.image_to_string(image).strip()
    except Exception:
        return ""

class Link(BaseModel):
    url: str
    text: str

class Answer(BaseModel):
    answer: str
    links: List[Link]

@app.api_route("/", methods=["GET", "HEAD", "OPTIONS"])
async def root(request: Request):
    if request.method == "HEAD":
        return JSONResponse(status_code=200, content={})
    return {"message": "✅ TDS Virtual TA is live. POST to /api/ with {'question': '...'}"}

@app.post("/api", response_model=Answer)
async def handle_query(request: Request):
    global bi_encoder, reranker, faiss_index, metadata

    # Load everything lazily
    if bi_encoder is None:
        from sentence_transformers import SentenceTransformer
        bi_encoder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

    if reranker is None:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-4-v2")

    if faiss_index is None or metadata is None:
        faiss_index = faiss.read_index("vector_index/faiss.index")
        with open("vector_index/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

    # Read input
    try:
        data = await request.json()
    except:
        return {"answer": "Invalid request format", "links": []}

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

    # Rerank
    pairs = [(question, c["text"]) for c in top_chunks]
    scores = reranker.predict(pairs)
    ranked = [c for _, c in sorted(zip(scores, top_chunks), key=lambda x: -x[0])][:2]

    # Prepare response
    answer = "Based on the content, here's what I found:\n"
    links = []
    for c in ranked:
        snippet = c["text"][:300].replace("\n", " ").strip()
        answer += f"- {snippet}...\n"
        if c.get("url"):
            links.append(Link(url=c["url"], text=c.get("title", "Source")))

    # Free temp memory
    del emb, I, top_chunks, pairs, scores
    gc.collect()

    return Answer(answer=answer.strip(), links=links[:2])

