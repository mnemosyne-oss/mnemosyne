#!/usr/bin/env python3
"""
Shared embedding server for Mnemosyne in multi-gateway Hermes Agent setups.

Problem: 4 Hermes Agent profiles (gateways) on one machine, each loading
`intfloat/multilingual-e5-large` (~1.2 GB RAM per copy via fastembed/ONNX).
Total: ~4.8 GB just for embedding models — before LLM usage.

Solution: One process loads the model ONCE, exposes an OpenAI-compatible
POST /v1/embeddings endpoint, and all gateways point MNEMOSYNE_EMBEDDING_API_URL
to it. Result: 4.8 GB → ~1.5 GB (saves ~3.3 GB).

Usage:
    python3 embedding_server.py [--port 8765] [--model intfloat/multilingual-e5-large]

Each profile .env:
    MNEMOSYNE_EMBEDDING_API_URL=http://127.0.0.1:8765/v1
    MNEMOSYNE_EMBEDDING_MODEL=intfloat/multilingual-e5-large

Systemd (optional):
    [Unit]
    Description=Mnemosyne Shared Embedding Server
    After=network.target

    [Service]
    Type=simple
    ExecStart=/usr/bin/python3 /opt/embedding_server.py --port 8765
    Restart=always
    RestartSec=5

    [Install]
    WantedBy=multi-user.target

---

IMPORTANT — Production caveat:

stdlib http.server is single-threaded and blocking. If multiple gateways send
concurrent embedding requests, they will serialize: request B waits for request A
to finish before the model processes B. This adds latency under concurrent load.

For a 4-gateway homelab this is usually fine (embeddings are fast and requests
are sparse) but if you see recall latency spikes, upgrade to one of:

  - aiohttp + asyncio (keep it pure Python, zero new deps for fastembed users)
  - ThreadingMixIn (stdlib, drop-in: HTTPServer -> ThreadingHTTPServer)
  - uvicorn + fastapi/starlette (production-grade, adds deps)

This example ships the simple version on purpose — it works, it's readable,
and you can iterate from here.

Requires: fastembed (pip install fastembed)
No other dependencies beyond Python 3.10+ stdlib.
"""

import argparse
import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [embed-server] %(levelname)s %(message)s",
)
log = logging.getLogger("embed-server")


class EmbeddingServer:
    """Singleton that holds the fastembed model in memory."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._dim = None

    def _load(self):
        if self._model is not None:
            return
        log.info("Loading model %s via fastembed (ONNX)...", self.model_name)
        t0 = time.time()
        from fastembed import TextEmbedding

        cache_dir = os.path.join(
            os.path.expanduser("~/.hermes"), "cache", "fastembed"
        )
        os.makedirs(cache_dir, exist_ok=True)
        self._model = TextEmbedding(
            model_name=self.model_name,
            cache_dir=cache_dir,
        )
        # Get embedding dimension by encoding a test string
        test_vec = list(self._model.embed(["test"]))[0]
        self._dim = len(test_vec)
        dt = time.time() - t0
        log.info("Model loaded in %.1fs (%d-dim vectors)", dt, self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vectors = list(self._model.embed(texts))
        return [v.tolist() for v in vectors]

    @property
    def dim(self) -> int:
        self._load()
        return self._dim


class EmbeddingHandler(BaseHTTPRequestHandler):
    """OpenAI-compatible /v1/embeddings endpoint."""

    server_model: EmbeddingServer = None  # set by main()

    def log_message(self, format, *args):
        log.info("%s - %s", self.client_address[0], format % args)

    def do_POST(self):
        if self.path not in ("/v1/embeddings", "/embeddings"):
            self.send_error(404, "Not Found")
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            req = json.loads(body)
        except Exception:
            self.send_error(400, "Bad JSON")
            return

        model_requested = req.get("model", "")
        texts = req.get("input", [])
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            self.send_error(400, "Missing 'input' field")
            return

        log.info("Embedding %d text(s) [model=%s]", len(texts), model_requested)
        t0 = time.time()
        try:
            vectors = self.server_model.embed(texts)
        except Exception:
            log.exception("Embedding failed")
            self.send_error(500, "Embedding failed")
            return
        dt = time.time() - t0

        data = [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ]
        response = {
            "object": "list",
            "data": data,
            "model": self.server_model.model_name,
            "usage": {
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts),
            },
        }

        resp_body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)
        log.info("Served %d embeddings in %.1fms", len(texts), dt * 1000)

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok\n")
            return
        if self.path == "/":
            dim = (
                self.server_model.dim
                if self.server_model._model
                else "loading..."
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(
                f"Mnemosyne Embedding Server\n"
                f"model: {self.server_model.model_name}\n"
                f"dim: {dim}\n".encode()
            )
            return
        self.send_error(404, "Not Found")


def main():
    parser = argparse.ArgumentParser(
        description="Shared embedding server for Mnemosyne"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Listen port (default: 8765)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "MNEMOSYNE_EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
        ),
        help="Model name (default: intfloat/multilingual-e5-large)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    args = parser.parse_args()

    EmbeddingHandler.server_model = EmbeddingServer(args.model)
    httpd = HTTPServer((args.host, args.port), EmbeddingHandler)
    log.info("Embedding server listening on %s:%d", args.host, args.port)
    log.info("Model: %s", args.model)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
