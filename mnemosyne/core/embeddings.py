"""
Mnemosyne Dense Retrieval
Supports local fastembed (ONNX) and OpenAI-compatible API embeddings.
Falls back to keyword-only if neither is available.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.request
from typing import List, Optional
from functools import lru_cache

try:
    import numpy as np
except ImportError:
    np = None

# --- fastembed (local ONNX) ---
import warnings

# fastembed >=0.7 switched multilingual-e5-large from CLS -> mean pooling.
# The new behaviour is correct for E5 models; suppress the noise.
warnings.filterwarnings(
    "ignore",
    message=".*multilingual-e5-large.*now uses mean pooling.*",
)

try:
    from fastembed import TextEmbedding
except Exception:
    TextEmbedding = None

def _is_fastembed_available() -> bool:
    """Check if fastembed is available. Evaluates lazily, so a correct
    sys.path ordering at call time won't be shadowed by an early import."""
    return np is not None and TextEmbedding is not None

# Backward-compatible alias for legacy users who import this constant.
# Use _is_fastembed_available() in new code — it re-evaluates on each call.
_FASTEMBED_AVAILABLE = _is_fastembed_available()
_FASTEMBED_CACHE_DIR = os.path.join(os.path.expanduser("~/.hermes"), "cache", "fastembed")

# --- OpenAI-compatible API ---
# Mnemosyne embedding config is independent of general OpenRouter/OpenAI settings.
# Embedding models may use local llama.cpp, OpenAI, Anthropic, or any other provider.
_OPENAI_API_KEY = os.environ.get("MNEMOSYNE_EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
_OPENAI_BASE_URL = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL", "https://openrouter.ai/api/v1")

# --- Model selection ---
_DEFAULT_MODEL = os.environ.get("MNEMOSYNE_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
# Local fallback model used when API embedding fails (network outage, rate limit, etc.).
# Only meaningful when _DEFAULT_MODEL is an API model (text-embedding-* / openai/*).
# Defaults to the original default so API→local fallback is transparent.
_FALLBACK_MODEL = os.environ.get("MNEMOSYNE_EMBEDDING_FALLBACK_MODEL", "BAAI/bge-small-en-v1.5")
_embedding_model = None
_API_CALL_COUNT = 0


def _is_api_model(model_name: str) -> bool:
    """Check if the model should use the OpenAI-compatible API."""
    if model_name.startswith("openai/") or "text-embedding" in model_name or model_name.startswith("text-embedding"):
        return True
    # Custom endpoint: if MNEMOSYNE_EMBEDDING_API_URL is set to a non-OpenRouter URL,
    # assume the user has their own API server and any model name should route there.
    base_url = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL", "")
    if base_url and "openrouter.ai" not in base_url:
        return True
    # Explicit opt-in for non-OpenAI embedding models hosted on OpenRouter
    # (qwen/qwen3-embedding-*, baai/bge-*, jina-embeddings-*, nvidia/*-embed-*, etc.).
    # Distinct from the substring/prefix checks above because the default fastembed
    # model id (BAAI/bge-small-en-v1.5) shares the same vendor-prefix shape as those
    # OpenRouter models — pure name-pattern matching would silently break fastembed
    # users that also have OPENROUTER_API_KEY set for chat. Requiring an explicit
    # env flag keeps local-first behavior the default while giving a clean opt-in
    # for OpenRouter-hosted embedding models.
    if os.environ.get("MNEMOSYNE_EMBEDDINGS_VIA_API", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return False


def _get_embedding_dim(model_name: str) -> int:
    """Return the embedding dimension for a given model.

    Supports English, Chinese, and multilingual embedding models.
    Falls back to 384 (bge-small dimension) for unknown models.
    Override with MNEMOSYNE_EMBEDDING_DIM env var for unsupported models.
    """
    dims = {
        # --- English BGE ---
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
        # --- Chinese BGE ---
        "BAAI/bge-small-zh-v1.5": 512,
        "BAAI/bge-base-zh-v1.5": 768,
        "BAAI/bge-large-zh-v1.5": 1024,
        # --- Multilingual E5 ---
        "intfloat/multilingual-e5-small": 384,
        "intfloat/multilingual-e5-base": 768,
        "intfloat/multilingual-e5-large": 1024,
        # --- SentenceTransformers multilingual / local fastembed ---
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 384,
        "sentence-transformers/all-MiniLM-L6-v2": 384,
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": 768,
        # --- Multilingual BGE ---
        "BAAI/bge-m3": 1024,            # M3: multilingual (100+ langs), 1024-dim
        "BAAI/bge-multilingual-gemma2": 3584,
        # --- OpenAI ---
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        # --- Jina ---
        "jina-embeddings-v5-omni-nano": 768,
        "jina-embeddings-v5-omni-small": 1024,
    }
    # Check env override first
    env_dim = os.environ.get("MNEMOSYNE_EMBEDDING_DIM")
    if env_dim is not None:
        try:
            return int(env_dim)
        except (ValueError, TypeError):
            pass
    return dims.get(model_name, 384)


def _get_fallback_model():
    """Lazy-load the fallback embedding model (local fastembed)."""
    global _embedding_model
    if _is_api_model(_FALLBACK_MODEL):
        return None  # Fallback must be local; skip if it's also API
    if not _is_fastembed_available():
        return None
    # If the fallback model is the same as the primary and already loaded (common case),
    # reuse the cached instance.
    if _embedding_model is not None and _DEFAULT_MODEL == _FALLBACK_MODEL:
        return _embedding_model
    # Separate lazy-load for fallback: cached on the global but re-bound
    # The singleton is safe because we never juggle two
    # models simultaneously; embed() is single-threaded in practice.
    if _embedding_model is not None and _DEFAULT_MODEL != _FALLBACK_MODEL:
        # Primary model already loaded but it's the wrong one for fallback.
        # That's fine; we create the fallback on-the-fly. This won't
        # stomp _embedding_model because fallback is only called after
        # the primary (API) path returns None.
        os.makedirs(_FASTEMBED_CACHE_DIR, exist_ok=True)
        return TextEmbedding(
            model_name=_FALLBACK_MODEL,
            cache_dir=_FASTEMBED_CACHE_DIR,
        )
    if _embedding_model is None:
        os.makedirs(_FASTEMBED_CACHE_DIR, exist_ok=True)
        _embedding_model = TextEmbedding(
            model_name=_FALLBACK_MODEL,
            cache_dir=_FASTEMBED_CACHE_DIR,
        )
    return _embedding_model


def _embed_local(texts: List[str], model_name: Optional[str] = None) -> Optional[np.ndarray]:
    """Embed texts using local fastembed (for fallback from API mode)."""
    if not _is_fastembed_available():
        return None
    model_name = model_name or _FALLBACK_MODEL
    if _is_api_model(model_name):
        return None  # Sanity: don't loop back
    model = _get_fallback_model() if model_name == _FALLBACK_MODEL else _get_model()
    if model is None or model == "api":
        return None
    vectors = list(model.embed(texts))
    if not vectors:
        return None
    return np.stack(vectors).astype(np.float32)


def _get_model():
    """Lazy-load the embedding model (local fastembed)."""
    global _embedding_model
    if _is_api_model(_DEFAULT_MODEL):
        return "api"  # Sentinel for API mode
    if not _is_fastembed_available():
        return None
    if _embedding_model is None:
        os.makedirs(_FASTEMBED_CACHE_DIR, exist_ok=True)
        _embedding_model = TextEmbedding(
            model_name=_DEFAULT_MODEL,
            cache_dir=_FASTEMBED_CACHE_DIR,
        )
    return _embedding_model


def _embed_api(texts: List[str]) -> Optional[np.ndarray]:
    """Embed texts via OpenAI-compatible API (OpenRouter or custom endpoint)."""
    global _API_CALL_COUNT
    # Require API key for OpenRouter; custom endpoints may not need one.
    base_url = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL", "https://openrouter.ai/api/v1")
    is_custom = "openrouter.ai" not in base_url
    if not is_custom and not _OPENAI_API_KEY:
        return None

    url = f"{base_url.rstrip('/')}/embeddings"
    payload = json.dumps({
        "model": _DEFAULT_MODEL,
        "input": texts,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mnemosyne.site",
        "X-Title": "Mnemosyne Embedding",
    }
    if _OPENAI_API_KEY:
        headers["Authorization"] = f"Bearer {_OPENAI_API_KEY}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=payload, headers=headers)
            ctx = ssl.create_default_context()
            # Support custom CA bundles (NixOS, enterprise proxies, etc.)
            # SSL_CERT_FILE takes priority, then REQUESTS_CA_BUNDLE.
            cert_file = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
            if cert_file:
                ctx.load_verify_locations(cert_file)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
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
    """Check if dense retrieval is available.

    When the primary model is API-based, returns True if either
    (a) the API is reachable (has key) OR (b) a local fallback
    model is configured via MNEMOSYNE_EMBEDDING_FALLBACK_MODEL
    and fastembed is available. This ensures recall doesn't skip
    vector search just because the API is unreachable.
    """
    if os.environ.get("MNEMOSYNE_NO_EMBEDDINGS"):
        return False
    if _is_api_model(_DEFAULT_MODEL):
        # API mode: available if either API works OR we can fall back to local
        base_url = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL", "")
        api_available = bool(base_url and "openrouter.ai" not in base_url) or bool(_OPENAI_API_KEY)
        fallback_available = (
            not _is_api_model(_FALLBACK_MODEL)
            and _is_fastembed_available()
        )
        return api_available or fallback_available
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
        if result is not None:
            return result[0]
        # API failed (network, rate limit, etc.). Fall through to local.
        logger = __import__("logging").getLogger(__name__)
        logger.debug("embed_query: API failed, falling back to local model %s", _FALLBACK_MODEL)
        local_result = _embed_local([text])
        if local_result is not None:
            return local_result[0]
        return None

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
        result = _embed_api(texts)
        if result is not None:
            return result
        # API failed (network, rate limit, etc.). Fall through to local.
        logger = __import__("logging").getLogger(__name__)
        logger.debug("embed: API failed, falling back to local model %s", _FALLBACK_MODEL)
        return _embed_local(texts)

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
