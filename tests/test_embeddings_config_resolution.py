"""Config-file resolution for embeddings (config.yaml > env > default).

``mnemosyne/core/embeddings.py`` used to read environment variables only, so a
Hermes-spawned process with a correct ``config.yaml`` (``embedding_dim: 1024``)
but no ``MNEMOSYNE_*`` env fell back to 384 and disabled vector search with an
"Embedding dimension mismatch". These tests pin the precedence
(``config.yaml`` wins over env, env wins over the built-in default), the
preserved fail-loud contract for unknown models, and the seeding behavior that
keeps per-profile (``HERMES_HOME``) configs from diverging from the shared
``HOME`` config.

Run with: MNEMOSYNE_NO_EMBEDDINGS=1 pytest tests/test_embeddings_config_resolution.py -v
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from mnemosyne.core import embeddings
from mnemosyne.core.config import MnemosyneConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Config singleton backed by a temp config.yaml with a clean env."""
    for key in list(os.environ):
        if key.startswith("MNEMOSYNE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")
    MnemosyneConfig.reset_instance()
    config = MnemosyneConfig(config_path=config_path)
    monkeypatch.setattr(
        "mnemosyne.core.config.MnemosyneConfig._instance", config
    )
    # Re-resolve the import-time snapshots so snapshot consumers see the
    # isolated state instead of the ambient config active at import.
    monkeypatch.setattr(
        embeddings, "_DEFAULT_MODEL", embeddings._resolve_default_model()
    )
    monkeypatch.setattr(
        embeddings, "_OPENAI_API_KEY", embeddings._resolve_api_key()
    )
    monkeypatch.setattr(
        embeddings, "_OPENAI_BASE_URL", embeddings._resolve_api_base_url()
    )
    yield config
    MnemosyneConfig.reset_instance()


def _write_config(config: MnemosyneConfig, data: dict) -> None:
    Path(config.config_path).write_text(yaml.safe_dump(data))


class TestDimPrecedence:
    def test_yaml_dim_wins_over_env(self, isolated_config, monkeypatch):
        _write_config(isolated_config, {"embedding_dim": 1024})
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DIM", "512")
        assert embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5") == 1024

    def test_env_dim_wins_over_builtin_table(self, isolated_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DIM", "512")
        assert embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5") == 512

    def test_builtin_table_when_nothing_set(self, isolated_config):
        assert embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5") == 384

    def test_blank_yaml_dim_treated_as_unset(self, isolated_config, monkeypatch):
        _write_config(isolated_config, {"embedding_dim": ""})
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DIM", "512")
        assert embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5") == 512

    def test_blank_env_dim_treated_as_unset(self, isolated_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DIM", "   ")
        assert embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5") == 384

    def test_invalid_yaml_dim_raises(self, isolated_config):
        _write_config(isolated_config, {"embedding_dim": "not-a-number"})
        with pytest.raises(ValueError, match="not a valid integer"):
            embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5")

    def test_nonpositive_yaml_dim_raises(self, isolated_config):
        _write_config(isolated_config, {"embedding_dim": 0})
        with pytest.raises(ValueError, match="positive integer"):
            embeddings._get_embedding_dim("BAAI/bge-small-en-v1.5")


class TestFailLoudPreserved:
    def test_unknown_model_without_dim_still_raises(self, isolated_config):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            embeddings._get_embedding_dim("some/unknown-local-model")

    def test_explicit_yaml_dim_satisfies_unknown_model(self, isolated_config):
        """A config.yaml dimension is explicit: it satisfies the contract."""
        _write_config(isolated_config, {"embedding_dim": 1024})
        assert embeddings._get_embedding_dim("some/unknown-local-model") == 1024

    def test_explicit_env_dim_satisfies_unknown_model(
        self, isolated_config, monkeypatch
    ):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_DIM", "1024")
        assert embeddings._get_embedding_dim("some/unknown-local-model") == 1024


class TestEndpointResolution:
    def test_model_from_yaml(self, isolated_config):
        _write_config(isolated_config, {"embedding_model": "openai/text-embedding-3-small"})
        assert (
            embeddings._resolve_default_model() == "openai/text-embedding-3-small"
        )

    def test_model_env_fallback(self, isolated_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_EMBEDDING_MODEL", "openai/text-embedding-3-small")
        assert (
            embeddings._resolve_default_model() == "openai/text-embedding-3-small"
        )

    def test_api_url_from_yaml_routes_custom_endpoint(self, isolated_config):
        _write_config(
            isolated_config,
            {"embedding_api_url": "http://localhost:8000/v1"},
        )
        assert embeddings._is_api_model("anything-local") is True

    def test_api_url_suffix_stripped(self, isolated_config):
        _write_config(
            isolated_config,
            {"embedding_api_url": "http://localhost:8000/v1/embeddings"},
        )
        assert embeddings._api_base_url() == "http://localhost:8000/v1"

    def test_via_api_flag_from_yaml(self, isolated_config):
        _write_config(isolated_config, {"embeddings_via_api": "1"})
        assert embeddings._is_api_model("BAAI/bge-small-en-v1.5") is True

    def test_api_key_from_yaml(self, isolated_config):
        _write_config(isolated_config, {"embedding_api_key": "yaml-key"})
        assert embeddings._resolve_api_key() == "yaml-key"


class TestSeedBehavior:
    def test_seed_inherits_embedding_config_from_shared_home(
        self, tmp_path, monkeypatch
    ):
        """Seeding a per-profile config copies embedding_*/vec_type from HOME."""
        home = tmp_path / "home"
        shared_dir = home / ".hermes" / "mnemosyne"
        shared_dir.mkdir(parents=True)
        (shared_dir / "config.yaml").write_text(
            yaml.safe_dump({"embedding_model": "BAAI/bge-m3", "embedding_dim": 1024})
        )
        for key in list(os.environ):
            if key.startswith("MNEMOSYNE_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HOME", str(home))
        MnemosyneConfig.reset_instance()
        try:
            profile_path = tmp_path / "profile" / "mnemosyne" / "config.yaml"
            MnemosyneConfig(config_path=profile_path)
            seeded = yaml.safe_load(profile_path.read_text())
        finally:
            MnemosyneConfig.reset_instance()
        assert seeded["embedding_model"] == "BAAI/bge-m3"
        assert seeded["embedding_dim"] == 1024

    def test_seed_sniffs_existing_db_dimension(self, tmp_path, monkeypatch):
        """With no env and no shared config, the DB vec dimension seeds the key."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "mnemosyne.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                'CREATE VIEW vec_working AS SELECT 1 AS "embedding float32[1024]"'
            )
            conn.commit()
        finally:
            conn.close()
        for key in list(os.environ):
            if key.startswith("MNEMOSYNE_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        MnemosyneConfig.reset_instance()
        try:
            config_path = tmp_path / "fresh" / "config.yaml"
            MnemosyneConfig(config_path=config_path)
            seeded = yaml.safe_load(config_path.read_text())
        finally:
            MnemosyneConfig.reset_instance()
        assert seeded["embedding_dim"] == 1024

    def test_seed_leaves_dim_unset_without_trustworthy_source(
        self, tmp_path, monkeypatch
    ):
        """No env, no shared config, no DB: the key stays unset (fail-loud)."""
        for key in list(os.environ):
            if key.startswith("MNEMOSYNE_"):
                monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        MnemosyneConfig.reset_instance()
        try:
            config_path = tmp_path / "fresh" / "config.yaml"
            MnemosyneConfig(config_path=config_path)
            seeded = yaml.safe_load(config_path.read_text())
        finally:
            MnemosyneConfig.reset_instance()
        assert "embedding_dim" not in seeded


def _run_fresh(code: str, tmp_path: Path, **env_overrides: str):
    """Run code in a fresh interpreter with isolated HOME and data dir."""
    env = os.environ.copy()
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    env["HOME"] = str(home)
    env.pop("MNEMOSYNE_EMBEDDING_DIM", None)
    # An empty file disables auto-seeding (which would otherwise bake
    # defaults into config.yaml and shadow the env under test); a test
    # that pre-writes a real config keeps it.
    if not (data_dir / "config.yaml").exists():
        (data_dir / "config.yaml").write_text("")
    env.pop("HERMES_HOME", None)
    for flag in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        env.pop(flag, None)
    env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        env=env,
        check=False,
        timeout=120,
    )


class TestFreshProcessYamlDim:
    def test_explicit_yaml_dim_boots_unknown_model(self, tmp_path):
        """A fresh process with an unknown model but an explicit YAML dim boots."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "config.yaml").write_text(
            yaml.safe_dump({"embedding_dim": 1024})
        )
        result = _run_fresh(
            "from mnemosyne.core.embeddings import _get_embedding_dim;"
            "print(_get_embedding_dim('some/unknown-local-model'))",
            tmp_path,
            MNEMOSYNE_EMBEDDING_MODEL="some/unknown-local-model",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1024"

    def test_profile_without_dim_falls_back_to_shared_home(self, tmp_path):
        """HERMES_HOME fragmentation: a dim-less profile resolves shared HOME."""
        home = tmp_path / "home"
        shared_dir = home / ".hermes" / "mnemosyne"
        shared_dir.mkdir(parents=True)
        (shared_dir / "config.yaml").write_text(
            yaml.safe_dump({"embedding_dim": 1024})
        )
        profile_home = tmp_path / "profile-home"
        profile_home.mkdir()
        env = os.environ.copy()
        env["HOME"] = str(home)
        env["HERMES_HOME"] = str(profile_home)
        env.pop("MNEMOSYNE_DATA_DIR", None)
        env.pop("MNEMOSYNE_EMBEDDING_DIM", None)
        for flag in (
            "MNEMOSYNE_NO_EMBEDDINGS",
            "MNEMOSYNE_SKIP_EMBEDDINGS",
            "MNEMOSYNE_EMBEDDINGS_OFF",
        ):
            env.pop(flag, None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from mnemosyne.core.embeddings import _get_embedding_dim;"
                "print(_get_embedding_dim('some/unknown-local-model'))",
            ],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            env=env,
            check=False,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "1024"
