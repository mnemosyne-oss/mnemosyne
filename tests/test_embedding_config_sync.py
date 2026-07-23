"""
Test that embeddings.py correctly respects get_config() / config.yaml settings.
"""

import pytest
from mnemosyne.core.config import get_config
from mnemosyne.core import embeddings


def test_embedding_config_sync(tmp_path, monkeypatch):
    """Test that setting embedding values via get_config() updates embeddings functions."""
    config = get_config()

    # 1. Test embedding_model
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MODEL", "gemini/gemini-embedding-2-preview")
    assert embeddings._get_default_model() == "gemini/gemini-embedding-2-preview"

    # 2. Test embedding_dim
    monkeypatch.setattr(config, "_yaml_cache", {**config._yaml_cache, "embedding_dim": 3072})
    assert embeddings._get_embedding_dim("custom-model") == 3072

    # 3. Test embedding_api_url & via api
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    monkeypatch.setattr(config, "_yaml_cache", {**config._yaml_cache, "embedding_api_url": "http://localhost:20128/v1"})
    assert embeddings._get_base_url() == "http://localhost:20128/v1"
    assert embeddings._is_api_model("custom-model") is True

    # 4. Test embedding_api_key
    monkeypatch.setattr(config, "_yaml_cache", {**config._yaml_cache, "embedding_api_key": "test-key-123"})
    assert embeddings._get_api_key() == "test-key-123"
