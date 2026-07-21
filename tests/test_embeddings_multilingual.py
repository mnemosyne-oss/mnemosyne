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


def test_get_embedding_dim_jina_v2_base_family():
    """Jina v2 base models are all 768-dim, not the 384 unknown-model fallback.

    Regression guard: these popular bilingual/monolingual models (the ES one is
    a common Spanish/English choice) output 768-dim vectors. If they are absent
    from the dim table they silently resolve to 384, mismatching the real output
    and corrupting similarity search.
    """
    for model in (
        "jinaai/jina-embeddings-v2-base-es",
        "jinaai/jina-embeddings-v2-base-en",
        "jinaai/jina-embeddings-v2-base-de",
        "jinaai/jina-embeddings-v2-base-zh",
        "jinaai/jina-embeddings-v2-base-code",
    ):
        assert embeddings._get_embedding_dim(model) == 768, model


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
