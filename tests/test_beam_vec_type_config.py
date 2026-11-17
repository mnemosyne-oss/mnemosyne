"""Config-aware resolution for the beam vector compression type.

``VEC_TYPE`` used to freeze at import from ``MNEMOSYNE_VEC_TYPE`` alone, so a
``vec_type`` written to config.yaml (``mnemosyne config set vec_type``,
``mnemosyne profile apply``) never reached the vector tables. The resolution
now follows the config layer's documented precedence:
``config.yaml > env > default``, resolved at import like the upstream
snapshot.

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    """Point the config layer at a temp config.yaml with a clean env."""
    for key in list(os.environ):
        if key.startswith("MNEMOSYNE_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("")
    monkeypatch.setattr(
        "mnemosyne.core.config._default_config_path", lambda: config_path
    )
    yield config_path
    monkeypatch.undo()


def _write_config(config_path: Path, data: dict) -> None:
    config_path.write_text(yaml.safe_dump(data))


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


class TestSnapshot:
    def test_snapshot_exists_and_is_valid(self):
        assert beam.VEC_TYPE in ("float32", "int8", "bit")


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
            "from mnemosyne.core.beam import VEC_TYPE;"
            "print(VEC_TYPE)",
            tmp_path,
            MNEMOSYNE_VEC_TYPE="bit",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "bit"

    def test_yaml_vec_type_visible_at_import(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "config.yaml").write_text(
            yaml.safe_dump({"vec_type": "float32"})
        )
        result = _run_fresh(
            "from mnemosyne.core.beam import VEC_TYPE;"
            "print(VEC_TYPE)",
            tmp_path,
            MNEMOSYNE_VEC_TYPE="bit",
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "float32"

    def test_yaml_config_file_is_not_created_by_import(self, tmp_path):
        """A read-only import must not seed config.yaml (peek contract)."""
        result = _run_fresh(
            "import mnemosyne.core.beam as beam;"
            "print(beam.VEC_TYPE)",
            tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert not (tmp_path / "data" / "config.yaml").exists()
