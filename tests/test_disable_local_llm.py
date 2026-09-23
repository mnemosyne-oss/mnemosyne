"""Tests for the ``disable_local_llm`` config flag.

The flag blocks loading of the local GGUF model so deployments that rely
on remote/host LLMs (or on no LLM at all) never pay the download and
inference cost. Resolution follows the central precedence
``config.yaml > env > default`` via ``MnemosyneConfig.get_bool`` and is
read per call, so ``mnemosyne config set disable_local_llm true`` applies
without a process restart.

Run with: MNEMOSYNE_NO_EMBEDDINGS=1 pytest tests/test_disable_local_llm.py -v
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mnemosyne.core.config import (
    DEFAULTS,
    ENV_VAR_MAP,
    MnemosyneConfig,
)


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Config singleton backed by a temp config.yaml with a clean env."""
    for key in list(os.environ):
        if key.startswith("MNEMOSYNE_"):
            monkeypatch.delenv(key, raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")
    MnemosyneConfig.reset_instance()
    config = MnemosyneConfig(config_path=config_path)
    monkeypatch.setattr(
        "mnemosyne.core.config.MnemosyneConfig._instance", config
    )
    yield config
    MnemosyneConfig.reset_instance()


def _real_load_llm(monkeypatch):
    """Return the real ``_load_llm`` despite the conftest auto-stub."""
    import importlib

    from mnemosyne.core import local_llm

    reloaded = importlib.reload(local_llm)
    monkeypatch.setattr(reloaded, "_llm_instance", None, raising=False)
    monkeypatch.setattr(reloaded, "_llm_available", None, raising=False)
    monkeypatch.setattr(reloaded, "_llm_backend", None, raising=False)
    return reloaded


class TestKeyRegistration:
    def test_env_var_map_entry(self):
        assert ENV_VAR_MAP["disable_local_llm"] == "MNEMOSYNE_DISABLE_LOCAL_LLM"

    def test_default_is_false(self):
        assert DEFAULTS["disable_local_llm"] is False

    def test_template_key_entry(self):
        from mnemosyne.core.profiles import TEMPLATE_KEYS

        assert "disable_local_llm" in TEMPLATE_KEYS

    def test_seeded_config_contains_key(self, isolated_config, tmp_path):
        fresh_path = tmp_path / "fresh" / "config.yaml"
        MnemosyneConfig(config_path=fresh_path)
        assert fresh_path.exists()
        seeded = yaml.safe_load(fresh_path.read_text()) or {}
        assert seeded["disable_local_llm"] is False


class TestPrecedence:
    def test_default_off(self, isolated_config):
        from mnemosyne.core import local_llm

        assert local_llm._local_llm_disabled() is False

    def test_env_enables(self, isolated_config, monkeypatch):
        from mnemosyne.core import local_llm

        monkeypatch.setenv("MNEMOSYNE_DISABLE_LOCAL_LLM", "true")
        assert local_llm._local_llm_disabled() is True

    def test_yaml_enables(self, isolated_config):
        from mnemosyne.core import local_llm

        isolated_config.set("disable_local_llm", True)
        assert local_llm._local_llm_disabled() is True

    def test_yaml_wins_over_env(self, isolated_config, monkeypatch):
        """Central precedence is config.yaml > env: YAML false beats env true."""
        from mnemosyne.core import local_llm

        isolated_config.set("disable_local_llm", False)
        monkeypatch.setenv("MNEMOSYNE_DISABLE_LOCAL_LLM", "true")
        assert local_llm._local_llm_disabled() is False

    def test_hot_reload_without_restart(self, isolated_config):
        """Toggling the key via set() applies on the next call."""
        from mnemosyne.core import local_llm

        assert local_llm._local_llm_disabled() is False
        isolated_config.set("disable_local_llm", True)
        assert local_llm._local_llm_disabled() is True
        isolated_config.set("disable_local_llm", False)
        assert local_llm._local_llm_disabled() is False


class TestLoadGuard:
    def test_load_returns_none_when_disabled_via_env(
        self, isolated_config, monkeypatch
    ):
        monkeypatch.setenv("MNEMOSYNE_DISABLE_LOCAL_LLM", "true")
        local_llm = _real_load_llm(monkeypatch)
        assert local_llm._load_llm() is None
        assert local_llm._llm_available is False

    def test_call_returns_none_when_disabled_via_yaml(
        self, isolated_config, monkeypatch
    ):
        isolated_config.set("disable_local_llm", True)
        local_llm = _real_load_llm(monkeypatch)
        assert local_llm._call_local_llm("summarize this") is None

    def test_loaded_then_disabled_returns_none(
        self, isolated_config, monkeypatch
    ):
        """A model loaded before the flag is set must not be reused."""
        local_llm = _real_load_llm(monkeypatch)
        sentinel = object()
        monkeypatch.setattr(local_llm, "_llm_instance", sentinel, raising=False)
        isolated_config.set("disable_local_llm", True)
        assert local_llm._load_llm() is None
        assert local_llm._llm_available is False

    def test_fresh_process_env_disables_load(self, tmp_path):
        """Fresh interpreter: env flag alone blocks the loader."""
        code = (
            "from mnemosyne.core import local_llm;"
            "assert local_llm._local_llm_disabled() is True;"
            "assert local_llm._load_llm() is None;"
            "print('disabled-ok')"
        )
        env = {
            k: v for k, v in os.environ.items() if not k.startswith("MNEMOSYNE_")
        }
        env.pop("HERMES_HOME", None)
        isolated_dir = tmp_path / "data-isolated"
        isolated_dir.mkdir(exist_ok=True)
        env["MNEMOSYNE_DATA_DIR"] = str(isolated_dir)
        env["MNEMOSYNE_DISABLE_LOCAL_LLM"] = "true"
        env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "disabled-ok" in result.stdout

    def test_fresh_process_yaml_disables_load(self, tmp_path):
        """Fresh interpreter: config.yaml flag blocks the loader (no env)."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "config.yaml").write_text("disable_local_llm: true\n")
        code = (
            "from mnemosyne.core import local_llm;"
            "assert local_llm._local_llm_disabled() is True;"
            "assert local_llm._load_llm() is None;"
            "print('yaml-disabled-ok')"
        )
        env = {
            k: v for k, v in os.environ.items() if not k.startswith("MNEMOSYNE_")
        }
        env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
        env["MNEMOSYNE_NO_EMBEDDINGS"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "yaml-disabled-ok" in result.stdout
