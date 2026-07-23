"""
Test that embeddings.py correctly respects get_config() / config.yaml settings.
"""
import yaml

from mnemosyne.core.config import MnemosyneConfig, get_config


def _write_config(hermes_home, data: dict):
    """Write a config.yaml file to the mnemosyne subdir under hermes_home.

    When HERMES_HOME is set, _default_config_path() resolves to
    <HERMES_HOME>/mnemosyne/config.yaml.
    """
    config_dir = hermes_home / "mnemosyne"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(data, f)
    return config_file


def test_embedding_config_sync(tmp_path, monkeypatch):
    """Test that embedding values via get_config()/config.yaml update embeddings resolution."""
    hermes_home = tmp_path / "hermes"

    # Clear all embedding-related env vars that may be set in the real environment
    # so that config-only code paths are exercised correctly.
    for _var in (
        "MNEMOSYNE_EMBEDDING_MODEL",
        "MNEMOSYNE_EMBEDDING_API_KEY",
        "MNEMOSYNE_EMBEDDING_API_URL",
        "MNEMOSYNE_EMBEDDINGS_VIA_API",
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(_var, raising=False)

    # Write a real config.yaml so _get_config_safe() finds and loads it
    _write_config(hermes_home, {
        "embedding_model": "",
        "embedding_dim": 0,
        "embedding_api_url": "https://openrouter.ai/api/v1",
        "embedding_api_key": "",
        "embeddings_via_api": False,
    })
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    MnemosyneConfig.reset_instance()
    config = get_config()

    # Import after pinning config path so module import doesn't seed ~/.hermes.
    from mnemosyne.core import embeddings

    # 1. Test embedding_model via env var (config-only tested below)
    monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MODEL", "gemini/gemini-embedding-2-preview")
    assert embeddings._get_default_model() == "gemini/gemini-embedding-2-preview"

    # 1b. Config-only (no env var): embedding_model from config.yaml
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_MODEL", raising=False)
    config.set("embedding_model", "BAAI/bge-base-en-v1.5")
    # Re-write config.yaml so _get_config_safe() can load fresh values
    _write_config(hermes_home, {"embedding_model": "BAAI/bge-base-en-v1.5"})
    MnemosyneConfig.reset_instance()
    assert embeddings._get_default_model() == "BAAI/bge-base-en-v1.5"
    # Restore
    MnemosyneConfig.reset_instance()

    # 2. Test embedding_dim via config.yaml
    _write_config(hermes_home, {"embedding_dim": 3072})
    MnemosyneConfig.reset_instance()
    assert embeddings._get_embedding_dim("custom-model") == 3072

    # 2b. Config-only: embedding_dim for another unknown model also gets config value
    assert embeddings._get_embedding_dim("unknown-custom-model") == 3072

    # 3. Test embedding_api_url via config.yaml & _is_api_model via env
    _write_config(hermes_home, {"embedding_api_url": "http://localhost:20128/v1"})
    MnemosyneConfig.reset_instance()
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_URL", raising=False)
    assert embeddings._get_base_url() == "http://localhost:20128/v1"

    # 3b. _is_api_model via env var
    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_VIA_API", "1")
    assert embeddings._is_api_model("custom-model") is True
    monkeypatch.delenv("MNEMOSYNE_EMBEDDINGS_VIA_API", raising=False)

    # 3c. Config-only: _is_api_model reads embedding_api_url from config.yaml
    # A custom (non-openrouter) URL in config should signal API mode
    _write_config(hermes_home, {"embedding_api_url": "http://localhost:20128/v1"})
    MnemosyneConfig.reset_instance()
    assert embeddings._is_api_model("custom-model") is True

    # 4. Test embedding_api_key — clear OPENAI_API_KEY first so env doesn't win
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_KEY", raising=False)
    _write_config(hermes_home, {"embedding_api_key": "test-key-123"})
    MnemosyneConfig.reset_instance()
    assert embeddings._get_api_key() == "test-key-123"

    # 5. Test _is_disabled: a truthy value for the env var disables embeddings
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    assert embeddings._is_disabled() is True
    monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS")

    monkeypatch.setenv("MNEMOSYNE_SKIP_EMBEDDINGS", "1")
    assert embeddings._is_disabled() is True
    monkeypatch.delenv("MNEMOSYNE_SKIP_EMBEDDINGS")

    monkeypatch.setenv("MNEMOSYNE_EMBEDDINGS_OFF", "on")
    assert embeddings._is_disabled() is True
    monkeypatch.delenv("MNEMOSYNE_EMBEDDINGS_OFF")

    # Not set → not disabled
    assert embeddings._is_disabled() is False

    # 6. available_api() respects _is_disabled()
    _write_config(hermes_home, {"embedding_api_key": "test-key-123"})
    MnemosyneConfig.reset_instance()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MNEMOSYNE_EMBEDDING_API_KEY", raising=False)
    assert embeddings.available_api() is True
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    assert embeddings.available_api() is False
    monkeypatch.delenv("MNEMOSYNE_NO_EMBEDDINGS")
