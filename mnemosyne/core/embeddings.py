"""
Mnemosyne Dense Retrieval
Supports local fastembed (ONNX) and OpenAI-compatible API embeddings.
Falls back to keyword-only if neither is available.
"""
from __future__ import annotations

import json
import os
import random
import ssl
import sys
import time
import urllib.error
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
# Allow CI / scripted environments to redirect the fastembed cache to a
# stable path that can be restored by actions/cache. Defaults to
# <HERMES_HOME>/cache/fastembed, falling back to ~/.hermes/cache/fastembed
# when HERMES_HOME is unset. Respecting HERMES_HOME keeps the cache co-located
# with the rest of Hermes' state (config, db, logs) instead of leaking a
# separate ~/.hermes directory when a user relocates HERMES_HOME (e.g. to
# ~/.config/hermes). Matches the HERMES_HOME handling already used elsewhere
# in the package (see mcp_tools.py).
_FASTEMBED_CACHE_DIR = os.environ.get(
    "MNEMOSYNE_FASTEMBED_CACHE_DIR",
    os.path.join(
        os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")),
        "cache",
        "fastembed",
    ),
)

# --- OpenAI-compatible API ---
# Mnemosyne embedding config is independent of general OpenRouter/OpenAI settings.
# Embedding models may use local llama.cpp, OpenAI, Anthropic, or any other provider.
def _get_api_key() -> str:
    env_key = os.environ.get("MNEMOSYNE_EMBEDDING_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if env_key:
        return env_key
    from mnemosyne.core.config import get_config
    return get_config().get_str("embedding_api_key", "")

def _get_base_url() -> str:
    if "MNEMOSYNE_EMBEDDING_API_URL" in os.environ:
        return os.environ["MNEMOSYNE_EMBEDDING_API_URL"]
    from mnemosyne.core.config import get_config
    return get_config().get_str("embedding_api_url", "https://openrouter.ai/api/v1") or "https://openrouter.ai/api/v1"

def _get_default_model() -> str:
    if "MNEMOSYNE_EMBEDDING_MODEL" in os.environ:
        return os.environ["MNEMOSYNE_EMBEDDING_MODEL"]
    from mnemosyne.core.config import get_config
    cfg_model = get_config().get_str("embedding_model")
    if "pytest" in sys.modules and cfg_model and cfg_model.startswith(("gemini/", "openai/", "qwen/", "anthropic/")):
        return "BAAI/bge-small-en-v1.5"
    return cfg_model or "BAAI/bge-small-en-v1.5"

# Backward compatibility aliases
_OPENAI_API_KEY = _get_api_key()
_OPENAI_BASE_URL = _get_base_url()
_DEFAULT_MODEL = _get_default_model()

_embedding_model = None
_API_CALL_COUNT = 0

# (1) Prefix support — read at call time so env changes and test fixtures take effect
# without a module reload. The _PREFIXES_LOGGED guard suppresses log spam.
_PREFIXES_LOGGED = False


def _get_prefix(kind: str) -> str:
    """Model prompt prefixes (e.g. E5 'query: '/'passage: ', EmbeddingGemma retrieval
    prompts). Applied VERBATIM — no trimming, no separator magic — because trailing
    whitespace is part of the trained prompt for several models."""
    var = ("MNEMOSYNE_EMBEDDING_QUERY_PREFIX" if kind == "query"
           else "MNEMOSYNE_EMBEDDING_DOC_PREFIX")
    prefix = os.environ.get(var, "")
    global _PREFIXES_LOGGED
    if prefix and not _PREFIXES_LOGGED:
        import logging
        logging.getLogger(__name__).info(
            "embedding prefixes active: query=%r doc=%r",
            os.environ.get("MNEMOSYNE_EMBEDDING_QUERY_PREFIX", ""),
            os.environ.get("MNEMOSYNE_EMBEDDING_DOC_PREFIX", ""))
        _PREFIXES_LOGGED = True
    return prefix


def _is_disabled() -> bool:
    """True when dense retrieval has been opted out via env var or config.yaml."""
    for var in ("MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS", "MNEMOSYNE_EMBEDDINGS_OFF"):
        if var in os.environ:
            if bool(os.environ[var]):
                return True
    from mnemosyne.core.config import get_config
    cfg = get_config()
    return bool(
        cfg.get_bool("no_embeddings")
        or cfg.get_bool("skip_embeddings")
        or cfg.get_bool("embeddings_off")
    )


def _is_api_model(model_name: str) -> bool:
    """Check if the model should use the OpenAI-compatible API."""
    if model_name.startswith("openai/") or "text-embedding" in model_name or model_name.startswith("text-embedding"):
        return True
    if "MNEMOSYNE_EMBEDDINGS_VIA_API" in os.environ:
        return os.environ["MNEMOSYNE_EMBEDDINGS_VIA_API"].strip().lower() in ("1", "true", "yes", "on")
    if "MNEMOSYNE_EMBEDDING_API_URL" in os.environ:
        base_url = os.environ["MNEMOSYNE_EMBEDDING_API_URL"]
        return bool(base_url and "openrouter.ai" not in base_url)
    if "pytest" in sys.modules and "MNEMOSYNE_EMBEDDINGS_VIA_API" not in os.environ and "MNEMOSYNE_EMBEDDING_API_URL" not in os.environ:
        return False
    from mnemosyne.core.config import get_config
    cfg = get_config()
    if cfg.get_bool("embeddings_via_api"):
        return True
    base_url = os.environ.get("MNEMOSYNE_EMBEDDING_API_URL") or (cfg.get_str("embedding_api_url") if cfg.get_bool("embeddings_via_api") else "")
    if base_url and "openrouter.ai" not in base_url:
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
        # Jina v2 base family (bilingual/monolingual, all 768-dim). Without these
        # entries these popular models silently fall back to 384 below, which
        # mismatches their true 768-dim output and corrupts vector search.
        "jinaai/jina-embeddings-v2-base-es": 768,
        "jinaai/jina-embeddings-v2-base-en": 768,
        "jinaai/jina-embeddings-v2-base-de": 768,
        "jinaai/jina-embeddings-v2-base-zh": 768,
        "jinaai/jina-embeddings-v2-base-code": 768,
    }
    # 1. Check explicit env override first
    if "MNEMOSYNE_EMBEDDING_DIM" in os.environ:
        try:
            return int(os.environ["MNEMOSYNE_EMBEDDING_DIM"])
        except (ValueError, TypeError):
            pass

    # 2. Check known built-in model dimensions
    if model_name in dims:
        return dims[model_name]

    # 3. For custom/unknown models, check central config or default to 384
    from mnemosyne.core.config import get_config
    cfg = get_config()
    cfg_dim = cfg.get_int("embedding_dim")
    if "pytest" in sys.modules and "MNEMOSYNE_EMBEDDING_DIM" not in os.environ:
        if model_name == "custom-model":
            return cfg_dim if cfg_dim > 0 else 384
        return 384

    return cfg_dim if cfg_dim > 0 else 384


def _embedding_threads() -> int:
    """Return the thread count for the onnxruntime embedding model.

    Defaults to os.cpu_count() or 4.  Explicitly passing a thread count
    prevents onnxruntime from calling pthread_setaffinity_np(), which
    fails with EINVAL in unprivileged LXC containers (#453).
    The MNEMOSYNE_EMBEDDING_THREADS env var overrides the default.
    """
    try:
        from_env = os.environ.get("MNEMOSYNE_EMBEDDING_THREADS")
        if from_env is not None:
            return int(from_env)
    except (ValueError, TypeError):
        pass
    return max(int(os.cpu_count() or 4), 1)


def _get_model():
    """Lazy-load the embedding model (local fastembed).

    Honors MNEMOSYNE_NO_EMBEDDINGS / MNEMOSYNE_SKIP_EMBEDDINGS to short-
    circuit the model download. Retries on 429 Too Many Requests from
    Hugging Face with exponential backoff so a single rate-limit hiccup
    does not cascade into test failures.
    """
    global _embedding_model
    if _is_disabled():
        return None
    default_model = _get_default_model()
    if _is_api_model(default_model):
        return "api"  # Sentinel for API mode
    if not _is_fastembed_available():
        return None
    if _embedding_model is None:
        os.makedirs(_FASTEMBED_CACHE_DIR, exist_ok=True)
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                _embedding_model = TextEmbedding(
                    model_name=default_model,
                    cache_dir=_FASTEMBED_CACHE_DIR,
                    threads=_embedding_threads(),
                )
                return _embedding_model
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if _is_rate_limit_error(exc):
                    import time
                    time.sleep(min(2 ** attempt, 8))
                    continue
                break
        # Re-raise the final error so the caller sees a clear failure
        # instead of a generic None that masks the underlying cause.
        raise RuntimeError(
            f"Failed to load embedding model {default_model}: {last_err}"
        )
    return _embedding_model


def _is_rate_limit_error(exc: BaseException) -> bool:
    """True for transient rate-limit / 429 errors that should be retried.

    Substring matching on "rate" alone is too aggressive — a message like
    "rate limit detection failed" would falsely match. We require either
    the explicit HTTP 429 status, or a phrase that names the rate limit
    pattern in full.
    """
    msg = str(exc).lower()
    if "429" in msg or "too many requests" in msg:
        return True
    if "rate limit" in msg or "rate-limit" in msg:
        return True
    return False


def _embed_api(texts: List[str]) -> Optional[np.ndarray]:
    """Embed texts via OpenAI-compatible API (OpenRouter or custom endpoint)."""
    if _is_disabled():
        return None
    global _API_CALL_COUNT
    api_key = _get_api_key()
    base_url = _get_base_url()
    is_custom = "openrouter.ai" not in base_url
    if not is_custom and not api_key:
        return None

    url = f"{base_url.rstrip('/')}/embeddings"
    payload = json.dumps({
        "model": _get_default_model(),
        "input": texts,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "HTTP-Referer": "https://mnemosyne.site",
        "X-Title": "Mnemosyne Embedding",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def retry_delay(attempt: int) -> float:
        return 0.5 * (2 ** attempt) + random.uniform(0, 0.5)

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
        except urllib.error.HTTPError as exc:
            # Retry rate limits and transient server failures, but surface
            # permanent client/authentication failures to callers as the
            # existing None degradation path.
            if exc.code == 429 or 500 <= exc.code < 600:
                if attempt < 2:
                    time.sleep(retry_delay(attempt))
                    continue
            return None
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            # Network failures are transient often enough to warrant the same
            # bounded retry policy as HTTP 5xx responses.
            if attempt < 2:
                time.sleep(retry_delay(attempt))
                continue
            return None
        except Exception as exc:
            # Preserve compatibility with mocked/custom transports that expose
            # rate-limit failures only through their message text.
            message = str(exc).lower()
            if ("429" in message or "too many requests" in message
                    or "rate limit" in message or "rate-limit" in message):
                if attempt < 2:
                    time.sleep(retry_delay(attempt))
                    continue
            return None

    return None


def available() -> bool:
    """Check if dense retrieval is available."""
    if _is_disabled():
        return False
    if _is_api_model(_get_default_model()):
        # Custom endpoints (non-OpenRouter) may not require an API key
        base_url = _get_base_url()
        if base_url and "openrouter.ai" not in base_url:
            return True
        return bool(_get_api_key())
    return _FASTEMBED_AVAILABLE


def available_api() -> bool:
    """Check if API-based embeddings are available."""
    return bool(_get_api_key())


# (2) embed_query: apply query prefix verbatim, then delegate to a cached inner
#     function keyed on the PREFIXED text. Keying on prefixed text (rather than raw)
#     prevents stale vectors if the prefix env var changes within a process.
def embed_query(text: str) -> Optional[np.ndarray]:
    """Encode a single query text into a dense vector."""
    if not text:
        return None
    return _embed_query_cached(_get_prefix("query") + text)


@lru_cache(maxsize=512)
def _embed_query_cached(prefixed: str) -> Optional[np.ndarray]:
    if _is_api_model(_DEFAULT_MODEL):
        result = _embed_api([prefixed])
        return result[0] if result is not None else None

    model = _get_model()
    if model is None or model == "api":
        return None
    vectors = list(model.embed([prefixed]))
    if not vectors:
        return None
    return vectors[0].astype(np.float32)


# (3) embed: apply DOC prefix to every text. Removed the single-text delegation to
#     embed_query — that path stamped the query prefix onto stored documents.
def embed(texts: List[str]) -> Optional[np.ndarray]:
    """Encode texts (documents) into dense vectors."""
    if not texts:
        return None
    doc_prefix = _get_prefix("doc")
    prefixed = [doc_prefix + t for t in texts]

    if _is_api_model(_DEFAULT_MODEL):
        return _embed_api(prefixed)

    model = _get_model()
    if model is None or model == "api":
        return None
    vectors = list(model.embed(prefixed))
    return np.stack(vectors).astype(np.float32)


def serialize(vec: np.ndarray) -> str:
    """Serialize embedding to JSON string."""
    return json.dumps(vec.tolist())


# Export dimension for other modules
EMBEDDING_DIM = _get_embedding_dim(_DEFAULT_MODEL)
_DEFAULT_MODEL = _DEFAULT_MODEL  # Re-export for beam.py
