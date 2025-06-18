from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List
import base64, io, pickle, numpy as np
from PIL import Image
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import pytesseract, os
from dotenv import load_dotenv
import json

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

app = FastAPI()

bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
faiss_index = faiss.read_index("vector_index/faiss.index")

with open("vector_index/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

class Link(BaseModel):
    url: str
    text: str

class Answer(BaseModel):
    answer: str
    links: List[Link]

def extract_text_from_image(base64_image: str) -> str:
    try:
        image = Image.open(io.BytesIO(base64.b64decode(base64_image))).convert("RGB")
        return pytesseract.image_to_string(image).strip()
    except Exception as e:
        print("OCR Error:", e)
        return ""


@app.post("/api/", response_model=Answer)
async def answer_query(request: Request):
    try:
        raw_body = await request.body()
        try:
            # Try parsing as actual JSON
            data = await request.json()
        except Exception:
            # Promptfoo sends stringified JSON, so we decode and parse it
            data = json.loads(raw_body.decode("utf-8"))

    except Exception as e:
        return {"answer": f"Invalid request format: {e}", "links": []}

    question = data.get("question", "")
    image_base64 = data.get("image")

    if image_base64:
        question += "\n" + extract_text_from_image(image_base64)

    if not question.strip():
        return {"answer": "No valid question provided.", "links": []}

    embedding = bi_encoder.encode([question])[0].astype("float32")
    D, I = faiss_index.search(np.array([embedding]), k=10)
    top_chunks = [metadata[i] for i in I[0] if i < len(metadata)]

    rerank_scores = reranker.predict([(question, c["text"]) for c in top_chunks])
    ranked_chunks = [c for _, c in sorted(zip(rerank_scores, top_chunks), key=lambda x: -x[0])]
    relevant = ranked_chunks[:3]

    answer = "Based on the content, here's what I found relevant: "
    links = []
    for c in relevant:
        snippet = c["text"][:300].replace("\n", " ").strip()
        answer += f"- {snippet}...\n"
        if c.get("url"):
            links.append(Link(url=c["url"], text=c.get("title", "Source")))

    return Answer(answer=answer.strip(), links=links[:3])

