from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import base64, io, pickle, gc
import numpy as np
import faiss

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bi_encoder = None
faiss_index = None
metadata = None

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
    return {"message": "✅ TDS Virtual TA is live. POST to /api with {'question': '...'}"}

@app.post("/api", response_model=Answer)
async def handle_query(request: Request):
    global bi_encoder, faiss_index, metadata

    try:
        data = await request.json()
    except:
        return {"answer": "Invalid request format", "links": []}

    question = data.get("question", "").strip()

    if not question:
        return {"answer": "No question provided.", "links": []}

    if bi_encoder is None:
        from sentence_transformers import SentenceTransformer
        bi_encoder = SentenceTransformer("paraphrase-MiniLM-L3-v2")  # ~40% smaller

    if faiss_index is None:
        faiss_index = faiss.read_index("vector_index/faiss.index")

    if metadata is None:
        with open("vector_index/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

    # Search
    emb = bi_encoder.encode([question])[0].astype("float32")
    _, I = faiss_index.search(np.array([emb]), k=3)
    top_chunks = [metadata[i] for i in I[0] if i < len(metadata)]

    if not top_chunks:
        return {"answer": "No relevant content found.", "links": []}

    # Prepare response
    answer = "Here's what I found:\n"
    links = []
    for c in top_chunks[:2]:
        snippet = c["text"][:300].replace("\n", " ").strip()
        answer += f"- {snippet}...\n"
        if c.get("url"):
            links.append(Link(url=c["url"], text=c.get("title", "Source")))

    gc.collect()
    return Answer(answer=answer.strip(), links=links[:2])

@app.post("/", response_model=Answer)
async def alias_for_api(request: Request):
    return await handle_query(request)
