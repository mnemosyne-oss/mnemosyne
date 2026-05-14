"""
Mnemosyne Dense Retrieval
Supports local fastembed (ONNX) and OpenAI-compatible API embeddings.
Falls back to keyword-only if neither is available.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import List, Optional
from functools import lru_cache

try:
    import numpy as np
except ImportError:
    np = None

# --- fastembed (local ONNX) ---
try:
    from fastembed import TextEmbedding
except Exception:
    TextEmbedding = None

_FASTEMBED_AVAILABLE = np is not None and TextEmbedding is not None
_FASTEMBED_CACHE_DIR = os.path.join(os.path.expanduser("~/.hermes"), "cache", "fastembed")

# --- OpenAI-compatible API ---
_OPENAI_API_KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
_OPENAI_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# --- Model selection ---
_DEFAULT_MODEL = os.environ.get("MNEMOSYNE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_embedding_model = None
_API_CALL_COUNT = 0


def _is_api_model(model_name: str) -> bool:
    """Check if the model should use the OpenAI-compatible API."""
    return (
        model_name.startswith("openai/")
        or "text-embedding" in model_name
        or model_name.startswith("text-embedding")
    )


def _get_embedding_dim(model_name: str) -> int:
    """Return the embedding dimension for a given model."""
    dims = {
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }
    return dims.get(model_name, 384)


def _get_model():
    """Lazy-load the embedding model (local fastembed)."""
    global _embedding_model
    if _is_api_model(_DEFAULT_MODEL):
        return "api"  # Sentinel for API mode
    if not _FASTEMBED_AVAILABLE:
        return None
    if _embedding_model is None:
        os.makedirs(_FASTEMBED_CACHE_DIR, exist_ok=True)
        _embedding_model = TextEmbedding(
            model_name=_DEFAULT_MODEL,
            cache_dir=_FASTEMBED_CACHE_DIR,
        )
    return _embedding_model


def _embed_api(texts: List[str]) -> Optional[np.ndarray]:
    """Embed texts via OpenAI-compatible API (OpenRouter)."""
    global _API_CALL_COUNT
    if not _OPENAI_API_KEY:
        return None

    url = f"{_OPENAI_BASE_URL.rstrip('/')}/embeddings"
    payload = json.dumps({
        "model": _DEFAULT_MODEL,
        "input": texts,
    }).encode()

    headers = {
        "Authorization": f"Bearer {_OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mnemosyne.site",
        "X-Title": "Mnemosyne Embedding",
    }

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            embeddings = [item["embedding"] for item in data["data"]]
            _API_CALL_COUNT += 1
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                import time
                time.sleep(2 ** attempt)
                continue
            return None

    return None


def available() -> bool:
    """Check if dense retrieval is available."""
    if os.environ.get("MNEMOSYNE_NO_EMBEDDINGS"):
        return False
    if _is_api_model(_DEFAULT_MODEL):
        return bool(_OPENAI_API_KEY)
    return _FASTEMBED_AVAILABLE


def available_api() -> bool:
    """Check if API-based embeddings are available."""
    return bool(_OPENAI_API_KEY)


@lru_cache(maxsize=512)
def embed_query(text: str) -> Optional[np.ndarray]:
    """Encode a single query text into a dense vector."""
    if not text:
        return None

    if _is_api_model(_DEFAULT_MODEL):
        result = _embed_api([text])
        return result[0] if result is not None else None

    model = _get_model()
    if model is None or model == "api":
        return None
    vectors = list(model.embed([text]))
    if not vectors:
        return None
    return vectors[0].astype(np.float32)


def embed(texts: List[str]) -> Optional[np.ndarray]:
    """Encode texts into dense vectors."""
    if not texts:
        return None

    if _is_api_model(_DEFAULT_MODEL):
        return _embed_api(texts)

    # Use cached single-query path for common case of 1 text
    if len(texts) == 1:
        v = embed_query(texts[0])
        if v is None:
            return None
        return np.stack([v])

    model = _get_model()
    if model is None or model == "api":
        return None
    vectors = list(model.embed(texts))
    return np.stack(vectors).astype(np.float32)


def serialize(vec: np.ndarray) -> str:
    """Serialize embedding to JSON string."""
    return json.dumps(vec.tolist())


# Export dimension for other modules
EMBEDDING_DIM = _get_embedding_dim(_DEFAULT_MODEL)
_DEFAULT_MODEL = _DEFAULT_MODEL  # Re-export for beam.py
