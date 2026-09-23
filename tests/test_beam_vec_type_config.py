"""Config-aware resolution for the beam vector compression type.

``VEC_TYPE`` used to freeze at import from ``MNEMOSYNE_VEC_TYPE`` alone, so
``mnemosyne config set vec_type`` required a restart and a ``config.yaml``
value was ignored entirely. ``_resolve_vec_type()`` reads per call with the
central ``config.yaml > env > default`` precedence; ``VEC_TYPE`` remains as
an import-time snapshot for backward compatibility.

Run with: MNEMOSYNE_NO_EMBEDDINGS=1 pytest tests/test_beam_vec_type_config.py -v
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from mnemosyne.core import beam
from mnemosyne.core.config import MnemosyneConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Config singleton backed by a temp config.yaml with a clean env."""
    for key in list(os.environ):
        if key.startswith("MNEMOSYNE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    # Isolate the shared-HOME fallback: resolution is profile YAML > env
    # > shared YAML, so an ambient ~/.hermes config would otherwise shadow
    # the env/default values under test.
    monkeypatch.setenv("HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")
    MnemosyneConfig.reset_instance()
    config = MnemosyneConfig()
    monkeypatch.setattr(
        "mnemosyne.core.config.MnemosyneConfig._instance", config
    )
    yield config
    MnemosyneConfig.reset_instance()


def _write_config(config: MnemosyneConfig, data: dict) -> None:
    Path(config.config_path).write_text(yaml.safe_dump(data))


class TestResolvePrecedence:
    def test_default_is_int8(self, isolated_config):
        assert beam._resolve_vec_type() == "int8"

    def test_env_overrides_default(self, isolated_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "bit")
        assert beam._resolve_vec_type() == "bit"

    def test_yaml_overrides_env(self, isolated_config, monkeypatch):
        _write_config(isolated_config, {"vec_type": "float32"})
        monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "bit")
        assert beam._resolve_vec_type() == "float32"

    def test_values_are_case_insensitive(self, isolated_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "BIT")
        assert beam._resolve_vec_type() == "bit"

    def test_blank_yaml_treated_as_unset(self, isolated_config, monkeypatch):
        _write_config(isolated_config, {"vec_type": "  "})
        monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "bit")
        assert beam._resolve_vec_type() == "bit"

    def test_blank_env_falls_back_to_default(self, isolated_config, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "   ")
        assert beam._resolve_vec_type() == "int8"


class TestInvalidValues:
    def test_invalid_yaml_falls_back_to_float32_with_warning(
        self, isolated_config, caplog
    ):
        _write_config(isolated_config, {"vec_type": "fp16"})
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            assert beam._resolve_vec_type() == "float32"
        assert "vec_type" in caplog.text
        assert "fp16" in caplog.text

    def test_invalid_env_falls_back_to_float32(
        self, isolated_config, monkeypatch
    ):
        """Legacy mapping preserved: an invalid env value meant float32."""
        monkeypatch.setenv("MNEMOSYNE_VEC_TYPE", "fp16")
        assert beam._resolve_vec_type() == "float32"


class TestHotReload:
    def test_config_change_applies_without_restart(self, isolated_config):
        _write_config(isolated_config, {"vec_type": "int8"})
        assert beam._resolve_vec_type() == "int8"
        _write_config(isolated_config, {"vec_type": "bit"})
        assert beam._resolve_vec_type() == "bit"

    def test_snapshot_exists_and_is_valid(self):
        assert beam.VEC_TYPE in ("float32", "int8", "bit")

    def test_detect_vec_type_follows_live_config(self, isolated_config, monkeypatch):
        """``_detect_vec_type`` must consult the resolver, not the snapshot."""
        import sqlite3

        _write_config(isolated_config, {"vec_type": "float32"})
        monkeypatch.setattr(beam, "VEC_TYPE", "int8")
        # Force the resolver path even where sqlite-vec is not installed
        # (otherwise the ``not _SQLITE_VEC_AVAILABLE`` early return skips
        # the resolver and the test fails for environment reasons).
        monkeypatch.setattr(beam, "_SQLITE_VEC_AVAILABLE", True)
        calls = {"n": 0}
        real_resolve = beam._resolve_vec_type

        def _counting_resolve():
            calls["n"] += 1
            return real_resolve()

        monkeypatch.setattr(beam, "_resolve_vec_type", _counting_resolve)
        # float32 short-circuits before any sqlite-vec probe.
        assert beam._detect_vec_type(sqlite3.connect(":memory:")) == "float32"
        assert calls["n"] >= 1


def _run_fresh(code: str, tmp_path: Path, **env_overrides: str):
    """Run code in a fresh interpreter with an isolated data dir and HOME."""
    env = os.environ.copy()
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    env["HOME"] = str(home)
    env.pop("HERMES_HOME", None)
    for flag in (
        "MNEMOSYNE_NO_EMBEDDINGS",
        "MNEMOSYNE_SKIP_EMBEDDINGS",
        "MNEMOSYNE_EMBEDDINGS_OFF",
    ):
        env.pop(flag, None)
    # An empty file disables auto-seeding so the env under test governs.
    if not (data_dir / "config.yaml").exists():
        (data_dir / "config.yaml").write_text("")
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


class TestFreshProcess:
    def test_env_vec_type_visible_at_import(self, tmp_path):
        result = _run_fresh(
            "from mnemosyne.core.beam import VEC_TYPE, _resolve_vec_type;"
            "print(VEC_TYPE, _resolve_vec_type())",
            tmp_path,
            MNEMOSYNE_VEC_TYPE="bit",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "bit bit"

    def test_yaml_vec_type_visible_at_import(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "config.yaml").write_text(
            yaml.safe_dump({"vec_type": "float32"})
        )
        result = _run_fresh(
            "from mnemosyne.core.beam import VEC_TYPE, _resolve_vec_type;"
            "print(VEC_TYPE, _resolve_vec_type())",
            tmp_path,
            MNEMOSYNE_VEC_TYPE="bit",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "float32 float32"
