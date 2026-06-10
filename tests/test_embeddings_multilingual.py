"""Tests for embedding multilingual model dimension detection and API model routing."""
import os

from mnemosyne.core import embeddings


def setup_module():
    """Clean env vars that would shadow dimension lookups."""
    os.environ.pop("MNEMOSYNE_EMBEDDING_DIM", None)
    os.environ.pop("OPENROUTER_BASE_URL", None)
    os.environ.pop("MNEMOSYNE_EMBEDDING_API_URL", None)


def test_get_embedding_dim_english_models():
    """English BGE models have correct dimensions."""
    assert embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5") == 384
    assert embeddings._get_embedding_dim("BAAI/bge-base-en-v1.5") == 768
    assert embeddings._get_embedding_dim("BAAI/bge-large-en-v1.5") == 1024


def test_get_embedding_dim_chinese_models():
    """Chinese BGE models have correct dimensions (different from English!)."""
    assert embeddings._get_embedding_dim("BAAI/bge-small-zh-v1.5") == 512
    assert embeddings._get_embedding_dim("BAAI/bge-base-zh-v1.5") == 768
    assert embeddings._get_embedding_dim("BAAI/bge-large-zh-v1.5") == 1024


def test_get_embedding_dim_multilingual_models():
    """Multilingual embedding models have correct dimensions."""
    assert embeddings._get_embedding_dim("intfloat/multilingual-e5-small") == 384
    assert embeddings._get_embedding_dim("intfloat/multilingual-e5-base") == 768
    assert embeddings._get_embedding_dim("intfloat/multilingual-e5-large") == 1024
    assert embeddings._get_embedding_dim("BAAI/bge-m3") == 1024


def test_get_embedding_dim_jina_models():
    """Jina v5 omni models have correct dimensions."""
    assert embeddings._get_embedding_dim("jina-embeddings-v5-omni-nano") == 768
    assert embeddings._get_embedding_dim("jina-embeddings-v5-omni-small") == 1024


def test_get_embedding_dim_env_override():
    """MNEMOSYNE_EMBEDDING_DIM env var overrides model-based detection."""
    os.environ["MNEMOSYNE_EMBEDDING_DIM"] = "768"
    try:
        assert embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5") == 768
        assert embeddings._get_embedding_dim("unknown-model") == 768
    finally:
        del os.environ["MNEMOSYNE_EMBEDDING_DIM"]


def test_get_embedding_dim_unknown_model_fallback():
    """Unknown models fall back to 384 (bge-small default)."""
    assert embeddings._get_embedding_dim("some/unknown-model") == 384
    assert embeddings._get_embedding_dim("") == 384


def test_get_embedding_dim_openai_models():
    """OpenAI API embedding models have correct dimensions."""
    assert embeddings._get_embedding_dim("openai/text-embedding-3-small") == 1536
    assert embeddings._get_embedding_dim("openai/text-embedding-3-large") == 3072
    assert embeddings._get_embedding_dim("text-embedding-3-small") == 1536


# ---------------------------------------------------------------------------
# _is_api_model tests — custom endpoint detection (PR #161)
# ---------------------------------------------------------------------------

def _clean_env():
    """Remove test env vars that influence _is_api_model()."""
    for key in ("OPENROUTER_BASE_URL", "MNEMOSYNE_EMBEDDING_API_URL"):
        os.environ.pop(key, None)


def test_is_api_model_openai_patterns():
    """Model names matching openai/text-embedding patterns return True."""
    _clean_env()
    assert embeddings._is_api_model("openai/text-embedding-3-small") is True
    assert embeddings._is_api_model("text-embedding-3-large") is True
    assert embeddings._is_api_model("openai/custom-model") is True


def test_is_api_model_unknown_without_custom_endpoint():
    """Unknown model names return False when no custom endpoint is set."""
    _clean_env()
    assert embeddings._is_api_model("BAAI/bge-small-en-v1.5") is False
    assert embeddings._is_api_model("jina-embeddings-v5-omni-nano") is False
    assert embeddings._is_api_model("some/random-model") is False


def test_is_api_model_custom_endpoint_non_openrouter():
    """Custom endpoint (non-OpenRouter URL) -> any model name returns True."""
    _clean_env()
    # intentional space for readability
    os.environ["MNEMOSYNE_EMBEDDING_API_URL"] = "https://llama.floory.uk/v1"
    try:
        assert embeddings._is_api_model("jina-embeddings-v5-omni-nano") is True
        assert embeddings._is_api_model("BAAI/bge-small-en-v1.5") is True
        assert embeddings._is_api_model("some/random-model") is True
    finally:
        _clean_env()


def test_is_api_model_openrouter_url_not_custom():
    """OpenRouter URL itself should NOT trigger custom endpoint detection."""
    _clean_env()
    os.environ["MNEMOSYNE_EMBEDDING_API_URL"] = "https://openrouter.ai/api/v1"
    try:
        # Unknown model on OpenRouter still uses fastembed (False)
        assert embeddings._is_api_model("jina-embeddings-v5-omni-nano") is False
        # But openai/ patterns still match on their own
        assert embeddings._is_api_model("openai/text-embedding-3-small") is True
    finally:
        _clean_env()


def test_is_api_model_text_embedding_substring():
    """Models containing 'text-embedding' anywhere in name return True."""
    _clean_env()
    assert embeddings._is_api_model("my-org/text-embedding-custom") is True
    assert embeddings._is_api_model("prefix-text-embedding-suffix") is True


# ---------------------------------------------------------------------------
# API → local fallback tests
# ---------------------------------------------------------------------------

def test_embed_local_returns_none_when_no_fastembed():
    """_embed_local handles missing fastembed gracefully (no crash)."""
    _clean_env()
    result = embeddings._embed_local(["hello world"])
    # Returns None when fastembed unavailable, or an ndarray when it is
    assert result is None or (hasattr(result, 'shape') and result.shape[0] == 1)
    assert result is None or result.dtype == np.float32


def test_embed_fallback_env_var_exists():
    """MNEMOSYNE_EMBEDDING_FALLBACK_MODEL env var is read correctly at module scope."""
    _clean_env()
    import os
    saved = os.environ.get("MNEMOSYNE_EMBEDDING_FALLBACK_MODEL")
    os.environ["MNEMOSYNE_EMBEDDING_FALLBACK_MODEL"] = "intfloat/multilingual-e5-small"
    try:
        import importlib
        import mnemosyne.core.embeddings as emb
        emb2 = importlib.reload(emb)
        assert emb2._FALLBACK_MODEL == "intfloat/multilingual-e5-small"
    finally:
        if saved is None:
            os.environ.pop("MNEMOSYNE_EMBEDDING_FALLBACK_MODEL", None)
        else:
            os.environ["MNEMOSYNE_EMBEDDING_FALLBACK_MODEL"] = saved


def test_embed_local_skips_api_model_fallback():
    """_embed_local returns None when fallback model is API-based (loop prevention)."""
    _clean_env()
    import os
    saved_key = os.environ.get("MNEMOSYNE_EMBEDDING_FALLBACK_MODEL")
    os.environ["MNEMOSYNE_EMBEDDING_FALLBACK_MODEL"] = "text-embedding-3-small"
    try:
        import importlib
        import mnemosyne.core.embeddings as emb
        emb2 = importlib.reload(emb)
        assert emb2._is_api_model(emb2._FALLBACK_MODEL) is True
        result = emb2._embed_local(["hello"])
        assert result is None
    finally:
        if saved_key is None:
            os.environ.pop("MNEMOSYNE_EMBEDDING_FALLBACK_MODEL", None)
        else:
            os.environ["MNEMOSYNE_EMBEDDING_FALLBACK_MODEL"] = saved_key


def test_available_accounts_for_fallback():
    """available() logic doesn't crash when API is configured with no key and fallback exists."""
    _clean_env()
    import os
    saved_model = os.environ.get("MNEMOSYNE_EMBEDDING_MODEL")
    saved_api_key = os.environ.get("MNEMOSYNE_EMBEDDING_API_KEY")
    saved_openai_key = os.environ.get("OPENAI_API_KEY")
    os.environ["MNEMOSYNE_EMBEDDING_MODEL"] = "text-embedding-3-small"
    os.environ["MNEMOSYNE_EMBEDDING_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    try:
        import importlib
        import mnemosyne.core.embeddings as emb
        emb2 = importlib.reload(emb)
        avail = emb2.available()
        assert isinstance(avail, bool)
    finally:
        if saved_model is None:
            os.environ.pop("MNEMOSYNE_EMBEDDING_MODEL", None)
        else:
            os.environ["MNEMOSYNE_EMBEDDING_MODEL"] = saved_model
        if saved_api_key is None:
            os.environ.pop("MNEMOSYNE_EMBEDDING_API_KEY", None)
        else:
            os.environ["MNEMOSYNE_EMBEDDING_API_KEY"] = saved_api_key
        if saved_openai_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = saved_openai_key
