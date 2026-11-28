"""Fresh-process regressions for #1285: polyphonic_recall ignored config.yaml.

The polyphonic gate in recall() and the enhanced-recall cache key read
MNEMOSYNE_POLYPHONIC_RECALL directly, so a value written by `mnemosyne
config set polyphonic_recall` or `mnemosyne profile apply` never enabled
the engine despite the documented `config.yaml > env vars > defaults`
precedence. The gate now resolves through resolve_beam_runtime() per
call — the same hot-reloadable path cross_session uses — so these cases
exercise the real dispatch with an initialized BeamMemory and a local store.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_GATE_SETUP = """
import os
from pathlib import Path
from mnemosyne.core.beam import BeamMemory

root = Path(os.environ["MNEMOSYNE_DATA_DIR"])
memory = BeamMemory(db_path=root / "beam.db")
memory._recall_polyphonic = lambda *args, **kwargs: []
"""

_GATE_SNIPPET = _GATE_SETUP + """
print(memory.recall("synthetic query", explain=True)["engine"])
"""


def _gate_in_subprocess(env: dict[str, str], code: str = _GATE_SNIPPET) -> str:
    environment = os.environ.copy()
    environment.pop("MNEMOSYNE_POLYPHONIC_RECALL", None)
    environment["MNEMOSYNE_EMBEDDINGS_OFF"] = "1"
    environment.update(env)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), environment.get("PYTHONPATH", "")]
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=environment,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_polyphonic_gate_reads_config_yaml(tmp_path: Path):
    """A config.yaml value enables the engine without the env var (#1285)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("polyphonic_recall: true\n")
    assert _gate_in_subprocess({"MNEMOSYNE_DATA_DIR": str(data_dir)}) == "polyphonic"


def test_polyphonic_gate_yaml_beats_env(tmp_path: Path):
    """Documented precedence is config.yaml > env vars, so yaml wins."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("polyphonic_recall: true\n")
    assert (
        _gate_in_subprocess(
            {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_POLYPHONIC_RECALL": "0"}
        )
        == "polyphonic"
    )


def test_polyphonic_gate_env_still_applies(tmp_path: Path):
    """The env var keeps working when config.yaml does not set the key."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    assert (
        _gate_in_subprocess(
            {"MNEMOSYNE_DATA_DIR": str(data_dir), "MNEMOSYNE_POLYPHONIC_RECALL": "1"}
        )
        == "polyphonic"
    )


@pytest.mark.parametrize("env_value,expected", [("", "linear"), ("0", "linear")])
def test_polyphonic_gate_default_off(tmp_path: Path, env_value: str, expected: str):
    """Unset or '0' leaves the linear path unchanged."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    env = {"MNEMOSYNE_DATA_DIR": str(data_dir)}
    if env_value:
        env["MNEMOSYNE_POLYPHONIC_RECALL"] = env_value
    assert _gate_in_subprocess(env) == expected


def test_polyphonic_yaml_false_overrides_env(tmp_path: Path):
    """A YAML false value takes precedence over an enabled environment value."""
    (tmp_path / "config.yaml").write_text("polyphonic_recall: false\n")
    assert _gate_in_subprocess({
        "MNEMOSYNE_DATA_DIR": str(tmp_path), "MNEMOSYNE_POLYPHONIC_RECALL": "1",
    }) == "linear"


def test_polyphonic_gate_reloads_between_calls(tmp_path: Path):
    """An initialized Beam observes each YAML update on the next direct recall."""
    snippet = _GATE_SETUP + """
for mode in ("false", "true", "false"):
    (root / "config.yaml").write_text(f"polyphonic_recall: {mode}\\n")
    print(memory.recall("synthetic query", explain=True)["engine"])
"""
    assert _gate_in_subprocess({"MNEMOSYNE_DATA_DIR": str(tmp_path)}, snippet).splitlines() == [
        "linear", "polyphonic", "linear",
    ]


@pytest.mark.parametrize("change_at", ["cache-key", "cache-lookup"])
def test_enhanced_recall_keeps_cache_key_and_execution_in_same_mode(
    tmp_path: Path, change_at: str,
):
    """A YAML edit during a cache miss takes effect on the following request."""
    snippet = """
import os
from pathlib import Path
from unittest.mock import patch
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.query_cache import QueryCache

root = Path(os.environ["MNEMOSYNE_DATA_DIR"])
config = root / "config.yaml"
config.write_text("polyphonic_recall: false\\n")
memory = BeamMemory(db_path=root / "beam.db", session_id="mode-test")
linear_id = memory.remember("orchid greenhouse irrigation", extract=False)
assert linear_id
cache = QueryCache(db_path=root / "query_cache.db")
memory._query_cache = cache
if os.environ["MNEMOSYNE_TEST_MODE_CHANGE_AT"] == "cache-key":
    target, method = memory, "_enhanced_recall_cache_key"
else:
    target, method = cache, "get_opaque"
original = getattr(target, method)

def switch_mode(*args, **kwargs):
    config.write_text("polyphonic_recall: true\\n")
    return original(*args, **kwargs)

def recall_ids():
    results = memory.recall_enhanced(
        "orchid greenhouse irrigation", top_k=3,
        use_weibull=False, use_mmr=False, use_intent=False, use_synonyms=False,
    )
    return [result["id"] for result in results]

poly_rows = [{"id": "poly-only", "content": "synthetic polyphonic result", "score": 1.0}]
with patch.object(memory, "_recall_polyphonic", return_value=poly_rows) as polyphonic:
    with patch.object(target, method, side_effect=switch_mode):
        assert recall_ids() == [linear_id]
        assert not polyphonic.called
    assert recall_ids() == ["poly-only"]
    config.write_text("polyphonic_recall: false\\n")
    assert recall_ids() == [linear_id]
    assert polyphonic.call_count == 1
"""
    _gate_in_subprocess({
        "MNEMOSYNE_DATA_DIR": str(tmp_path), "MNEMOSYNE_ENHANCED_RECALL": "1",
        "MNEMOSYNE_TEST_MODE_CHANGE_AT": change_at,
    }, snippet)
