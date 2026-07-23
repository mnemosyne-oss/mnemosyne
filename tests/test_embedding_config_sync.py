"""
Test that embeddings.py correctly respects get_config() / config.yaml settings.
"""

from mnemosyne.core.config import MnemosyneConfig, get_config


def test_embedding_config_sync(tmp_path, monkeypatch):
    """Test that embedding values via get_config()/config.yaml update embeddings resolution."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    MnemosyneConfig.reset_instance()
    config = get_config()

    # Import after pinning config path so module import doesn't seed ~/.hermes.
    from mnemosyne.core import embeddings

    # 1. Test embedding_model
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MODEL", "gemini/gemini-embedding-2-preview")
    assert embeddings._get_default_model() == "gemini/gemini-embedding-2-preview"

    # 2. Test embedding_dim
    config.set("embedding_dim", 3072)
    assert embeddings._get_embedding_dim("custom-model") == 3072

    # 3. Test embedding_api_url & via api
    config.set("embedding_api_url", "http://localhost:20128/v1")
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    assert embeddings._get_base_url() == "http://localhost:20128/v1"
    assert embeddings._is_api_model("custom-model") is True

    # 4. Test embedding_api_key
    config.set("embedding_api_key", "test-key-123")
    assert embeddings._get_api_key() == "test-key-123"
