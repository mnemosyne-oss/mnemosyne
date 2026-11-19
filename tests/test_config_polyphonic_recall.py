"""Fresh-process regressions for #1285: polyphonic_recall ignored config.yaml.

The polyphonic gate in recall() and the enhanced-recall cache key read
MNEMOSYNE_POLYPHONIC_RECALL directly, so a value written by `mnemosyne
config set polyphonic_recall` or `mnemosyne profile apply` never enabled
the engine despite the documented `config.yaml > env vars > defaults`
precedence. The gate now resolves through resolve_beam_runtime() per
call — the same hot-reloadable path cross_session uses — so these cases
probe the real gate, not the config layer alone: an uninitialised
BeamMemory is enough because the flag check precedes any instance state.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_GATE_SNIPPET = """
from mnemosyne.core import beam
_hits = []
beam.BeamMemory._recall_polyphonic = lambda self, *a, **k: _hits.append(1) or []
_m = beam.BeamMemory.__new__(beam.BeamMemory)
try:
    _m.recall("q")
except Exception:
    pass
print("POLY" if _hits else "LINEAR")
"""


def _gate_in_subprocess(env: dict[str, str]) -> str:
    environment = os.environ.copy()
    environment.pop("MNEMOSYNE_POLYPHONIC_RECALL", None)
    environment.update(env)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _GATE_SNIPPET],
        text=True,
        capture_output=True,
        env=environment,
        check=True,
    )
    return proc.stdout.strip()


def test_polyphonic_gate_reads_config_yaml(tmp_path: Path):
    """A config.yaml value enables the engine without the env var (#1285)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("polyphonic_recall: true\n")
    assert _gate_in_subprocess({"MNEMOSYNE_DATA_DIR": str(data_dir)}) == "POLY"


def test_polyphonic_gate_yaml_beats_env(tmp_path: Path):
    """Documented precedence is config.yaml > env vars, so yaml wins."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("polyphonic_recall: true\n")
    assert (
        _gate_in_subprocess(
            {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_POLYPHONIC_RECALL": "0"}
        )
        == "POLY"
    )


def test_polyphonic_gate_env_still_applies(tmp_path: Path):
    """The env var keeps working when config.yaml does not set the key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert (
        _gate_in_subprocess(
            {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_POLYPHONIC_RECALL": "1"}
        )
        == "POLY"
    )


@pytest.mark.parametrize("env_value,expected", [("", "LINEAR"), ("0", "LINEAR")])
def test_polyphonic_gate_default_off(tmp_path: Path, env_value: str, expected: str):
    """Unset or '0' leaves the linear path unchanged."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    env = {"MNEMOSYNE_DATA_DIR": str(data_dir)}
    if env_value:
        env["MNEMOSYNE_POLYPHONIC_RECALL"] = env_value
    assert _gate_in_subprocess(env) == expected
