import argparse
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import uvicorn

app = FastAPI()
MODEL = None
EMBEDDING_DIM = None


class TextRequest(BaseModel):
    text: str


@app.get("/health")
async def health():
    if MODEL is None or EMBEDDING_DIM is None:
        return {
            "ok": False,
            "status": "loading",
            "embedding_dim": None,
        }

    return {
        "ok": True,
        "status": "online",
        "embedding_dim": int(EMBEDDING_DIM),
    }


@app.post("/embed")
async def embed(request: TextRequest):
    global MODEL
    try:
        emb = MODEL.encode(request.text, normalize_embeddings=True)
        return {
            "ok": True,
            "embedding": emb.tolist(),
            "embedding_dim": int(emb.shape[0]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cache_size", type=int, default=50000)
    args = parser.parse_args()

    print(f"[FASTAPI] Loading {args.model} on {args.device}...")
    MODEL = SentenceTransformer(
        args.model,
        device=args.device,
        model_kwargs={"torch_dtype": torch.float16},
    )

    test_emb = MODEL.encode("dimension check", normalize_embeddings=True)
    EMBEDDING_DIM = int(test_emb.shape[0])
    print(f"[FASTAPI] Model loaded. embedding_dim={EMBEDDING_DIM}")

    print(f"[FASTAPI] Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
