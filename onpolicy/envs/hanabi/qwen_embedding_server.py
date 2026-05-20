#!/usr/bin/env python3

import argparse
import hashlib
import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class LocalQwenEmbeddingModel:
    def __init__(
        self,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device=None,
        normalize=True,
        cache_size=50000,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.Lock()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device

        print(f"[QWEN SERVER] loading {model_name} on {device}", flush=True)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.eval()

        test = self.encode_one("dimension check")
        self.embedding_dim = int(test.shape[0])

        print(
            f"[QWEN SERVER] ready model={model_name} dim={self.embedding_dim} device={device}",
            flush=True,
        )

    def _cache_key(self, text):
        payload = {
            "model": self.model_name,
            "normalize": self.normalize,
            "text": text,
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @torch.no_grad()
    def encode_one(self, text):
        key = self._cache_key(text)

        with self.lock:
            if key in self.cache:
                self.cache_hits += 1
                emb = self.cache.pop(key)
                self.cache[key] = emb
                return emb.copy()

        with self.lock:
            self.cache_misses += 1
            emb = self.model.encode(
                [text],
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
            )[0].astype(np.float32)

            self.cache[key] = emb
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

            return emb.copy()

    def stats(self):
        with self.lock:
            return {
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_size": len(self.cache),
            }


def make_handler(model_wrapper):
    class QwenEmbeddingHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                payload = {
                    "ok": True,
                    "model": model_wrapper.model_name,
                    "device": model_wrapper.device,
                    "embedding_dim": model_wrapper.embedding_dim,
                    **model_wrapper.stats(),
                }
                self._send_json(200, payload)
            else:
                self._send_json(404, {"ok": False, "error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/embed":
                self._send_json(404, {"ok": False, "error": "unknown endpoint"})
                return

            try:
                content_len = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_len)
                request = json.loads(raw_body.decode("utf-8"))

                text = request.get("text")
                if not isinstance(text, str):
                    self._send_json(
                        400, {"ok": False, "error": "missing string field: text"}
                    )
                    return

                emb = model_wrapper.encode_one(text)

                payload = {
                    "ok": True,
                    "embedding": emb.tolist(),
                    "embedding_dim": int(emb.shape[0]),
                    **model_wrapper.stats(),
                }
                self._send_json(200, payload)

            except Exception as e:
                self._send_json(500, {"ok": False, "error": repr(e)})

        def log_message(self, format, *args):
            return

    return QwenEmbeddingHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache_size", type=int, default=50000)
    parser.add_argument("--no_normalize", action="store_true", default=False)
    args = parser.parse_args()

    model_wrapper = LocalQwenEmbeddingModel(
        model_name=args.model,
        device=args.device,
        normalize=not args.no_normalize,
        cache_size=args.cache_size,
    )

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(model_wrapper),
    )

    print(f"[QWEN SERVER] listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
