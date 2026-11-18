"""Fresh-process regressions for #1378: VEC_TYPE ignored config.yaml.

beam.py resolves its module-level VEC_TYPE once at import. Before the fix it
read MNEMOSYNE_VEC_TYPE directly, so a value written by `mnemosyne config set
vec_type` or `mnemosyne profile apply` — both of which store it in
config.yaml — never reached the vector tables despite the documented
`config.yaml > env vars > defaults` precedence and the CLI telling users a
restart applies it. Each case runs a fresh interpreter because the constant
is bound at import, exactly like the reported reproduction.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _vec_type_in_subprocess(tmp_path: Path, env: dict[str, str]) -> str:
    environment = os.environ.copy()
    environment.pop("MNEMOSYNE_VEC_TYPE", None)
    environment.update(env)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", "from mnemosyne.core.beam import VEC_TYPE; print(VEC_TYPE)"],
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )
    return proc.stdout.strip()


@pytest.mark.parametrize("yaml_value,expected", [("float32", "float32"), ("bit", "bit")])
def test_vec_type_reads_config_yaml(tmp_path: Path, yaml_value: str, expected: str):
    """A config.yaml value reaches the import-time resolution (#1378)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(f"vec_type: {yaml_value}\n")
    assert (
        _vec_type_in_subprocess(tmp_path, {"MNEMOSYNE_DATA_DIR": str(data_dir)})
        == expected
    )


def test_vec_type_yaml_beats_env(tmp_path: Path):
    """Documented precedence is config.yaml > env vars, so yaml wins."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("vec_type: bit\n")
    assert (
        _vec_type_in_subprocess(
            tmp_path,
            {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_VEC_TYPE": "int8"},
        )
        == "bit"
    )


def test_vec_type_env_still_applies_without_yaml(tmp_path: Path):
    """The env var keeps working when config.yaml does not set the key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert (
        _vec_type_in_subprocess(
            tmp_path,
            {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_VEC_TYPE": "bit"},
        )
        == "bit"
    )


def test_vec_type_invalid_yaml_falls_back(tmp_path: Path):
    """An unrecognized value keeps the existing float32 fallback."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("vec_type: bogus\n")
    assert (
        _vec_type_in_subprocess(tmp_path, {"MNEMOSYNE_DATA_DIR": str(data_dir)})
        == "float32"
    )
