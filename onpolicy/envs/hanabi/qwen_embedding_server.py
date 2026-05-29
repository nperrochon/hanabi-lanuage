import argparse
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn
from collections import OrderedDict

app = FastAPI(title="Qwen Embedding Server")

# Model Global Variable
MODEL = None
CACHE = OrderedDict()
CACHE_SIZE = 50000


class TextRequest(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global MODEL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[FASTAPI] Loading model on {device}...")
    # Loading in FP16 for immediate 2x speedup
    MODEL = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device=device,
        model_kwargs={"torch_dtype": torch.float16},
    )
    MODEL.eval()
    print("[FASTAPI] Model Ready.")


@app.get("/health")
def health():
    return {"status": "ok", "embedding_dim": 1024}


@app.post("/embed")
async def embed(request: TextRequest):
    global MODEL
    try:
        # FastAPI handles the async queueing automatically
        # To truly batch, you'd use a background worker, but this is already faster
        emb = MODEL.encode(request.text, normalize_embeddings=True)
        return {
            "ok": True,
            "embedding": emb.tolist(),
            "embedding_dim": int(emb.shape[0]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765)
