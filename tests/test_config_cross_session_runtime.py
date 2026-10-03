"""Fresh-process regressions for cross-session runtime config resolution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_SRC = ROOT / "integrations" / "hermes" / "src"


def _run(
    script: str, *, env: dict[str, str], unset: tuple[str, ...] = ()
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(env)
    for name in unset:
        environment.pop(name, None)
    environment.pop("PYTHONHOME", None)
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(HERMES_SRC), existing_path])
    return subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, env=environment, check=True
    )


@pytest.mark.parametrize(
    ("yaml_value", "env_value", "expected"),
    [("true", "0", "True"), ("false", "1", "False")],
)
def test_cross_session_direct_core_honors_yaml_over_env(tmp_path: Path, yaml_value: str, env_value: str, expected: str):
    """Beam's real scope helpers honor config.yaml over a conflicting env var."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(f"cross_session: {yaml_value}\n")
    result = _run(
        """
import os
from pathlib import Path
from mnemosyne.core.beam import BeamMemory

db_path = Path(os.environ["TEST_DB"])
writer = BeamMemory(session_id="session-a", db_path=db_path)
reader = BeamMemory(session_id="session-b", db_path=db_path)
writer.remember("cross session runtime sentinel", source="test", importance=0.9)
results = reader.recall("cross session runtime sentinel", top_k=10)
print(any("runtime sentinel" in row.get("content", "") for row in results))
writer.conn.close()
reader.conn.close()
""",
        env={
            "MNEMOSYNE_DATA_DIR": str(data_dir),
            "MNEMOSYNE_CROSS_SESSION": env_value,
            "TEST_DB": str(tmp_path / "direct.db"),
        },
    )
    assert result.stdout.strip() == expected


@pytest.mark.parametrize(
    ("yaml_value", "env_value", "expected"),
    [("true", "0", "True"), ("false", "1", "False")],
)
def test_cross_session_hermes_provider_honors_yaml_over_env(tmp_path: Path, yaml_value: str, env_value: str, expected: str):
    """Hermes initialization uses the same cross-session resolver contract."""
    hermes_home = tmp_path / "hermes"
    config_dir = hermes_home / "mnemosyne"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text(f"cross_session: {yaml_value}\n")
    result = _run(
        """
import os
from mnemosyne_hermes import MnemosyneMemoryProvider

writer = MnemosyneMemoryProvider()
reader = MnemosyneMemoryProvider()
home = os.environ["HERMES_HOME"]
writer.initialize("session-a", hermes_home=home)
reader.initialize("session-b", hermes_home=home)
assert writer._beam is not None and reader._beam is not None
writer._beam.remember("cross session provider sentinel", source="test", importance=0.9)
results = reader._beam.recall("cross session provider sentinel", top_k=10)
print(any("provider sentinel" in row.get("content", "") for row in results))
writer._beam.conn.close()
reader._beam.conn.close()
""",
        env={"HERMES_HOME": str(hermes_home), "MNEMOSYNE_CROSS_SESSION": env_value, "MNEMOSYNE_DATA_DIR": ""},
    )
    assert result.stdout.strip() == expected


def test_direct_core_recall_weights_honor_yaml_env_and_defaults(tmp_path: Path):
    """The real Beam scoring consumer resolves weights as YAML > env > defaults."""
    script = """
import json
import os
from pathlib import Path
from mnemosyne.core.beam import BeamMemory

memory = BeamMemory(session_id="weights", db_path=Path(os.environ["TEST_DB"]))
try:
    memory.remember("recall weight runtime sentinel", source="test", importance=0.9)
    payload = memory.recall("recall weight runtime sentinel", top_k=3, explain=True)
    print(json.dumps(payload["explain"]["weights"], sort_keys=True))
finally:
    memory.conn.close()
"""

    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    (yaml_dir / "config.yaml").write_text(
        "vec_weight: 0\nfts_weight: 1\nimportance_weight: 0\n"
    )
    yaml_result = _run(
        script,
        env={
            "MNEMOSYNE_DATA_DIR": str(yaml_dir),
            "MNEMOSYNE_VEC_WEIGHT": "1",
            "MNEMOSYNE_FTS_WEIGHT": "0",
            "MNEMOSYNE_IMPORTANCE_WEIGHT": "1",
            "MNEMOSYNE_NO_EMBEDDINGS": "1",
            "TEST_DB": str(tmp_path / "yaml.db"),
        },
    )
    assert json.loads(yaml_result.stdout) == {
        "fts": 1.0,
        "importance": 0.0,
        "temporal": 0.0,
        "vec": 0.0,
    }

    env_dir = tmp_path / "env"
    env_dir.mkdir()
    # An existing non-weight config prevents auto-seeding from copying the
    # environment into YAML, so these values must come from env fallback.
    (env_dir / "config.yaml").write_text("cross_session: false\n")
    env_result = _run(
        script,
        env={
            "MNEMOSYNE_DATA_DIR": str(env_dir),
            "MNEMOSYNE_VEC_WEIGHT": "0",
            "MNEMOSYNE_FTS_WEIGHT": "1",
            "MNEMOSYNE_IMPORTANCE_WEIGHT": "0",
            "MNEMOSYNE_NO_EMBEDDINGS": "1",
            "TEST_DB": str(tmp_path / "env.db"),
        },
    )
    assert json.loads(env_result.stdout) == {
        "fts": 1.0,
        "importance": 0.0,
        "temporal": 0.0,
        "vec": 0.0,
    }

    defaults_dir = tmp_path / "defaults"
    defaults_dir.mkdir()
    # An existing config that omits the weight keys prevents auto-seeding from
    # materializing their defaults into YAML. The child clears inherited values,
    # so this case exercises get_many's true default fallback.
    (defaults_dir / "config.yaml").write_text("cross_session: false\n")
    defaults_result = _run(
        script,
        env={
            "MNEMOSYNE_DATA_DIR": str(defaults_dir),
            "MNEMOSYNE_NO_EMBEDDINGS": "1",
            "TEST_DB": str(tmp_path / "defaults.db"),
        },
        unset=(
            "MNEMOSYNE_VEC_WEIGHT",
            "MNEMOSYNE_FTS_WEIGHT",
            "MNEMOSYNE_IMPORTANCE_WEIGHT",
        ),
    )
    assert json.loads(defaults_result.stdout) == {
        "fts": 0.3,
        "importance": 0.2,
        "temporal": 0.0,
        "vec": 0.5,
    }


def test_both_hermes_provider_recall_surfaces_honor_yaml_over_env(tmp_path: Path):
    """Both provider tool routes reach Beam with the YAML-resolved weights."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(
        "vec_weight: 0\nfts_weight: 1\nimportance_weight: 0\n"
    )
    script = """
import importlib
import json
import os

Provider = importlib.import_module(os.environ["PROVIDER_MODULE"]).MnemosyneMemoryProvider
provider = Provider()
provider.initialize("weights", hermes_home=os.environ["HERMES_HOME"])
assert provider._beam is not None
try:
    provider._beam.remember("provider recall weight sentinel", source="test", importance=0.9)
    response = json.loads(provider._handle_recall({
        "query": "provider recall weight sentinel", "limit": 3, "explain": True,
    }))
    print(json.dumps(response["explain"]["weights"], sort_keys=True))
finally:
    provider._beam.conn.close()
"""
    for provider_module in ("hermes_memory_provider", "mnemosyne_hermes"):
        provider_home = tmp_path / provider_module
        provider_home.mkdir()
        result = _run(
            script,
            env={
                "PROVIDER_MODULE": provider_module,
                "HERMES_HOME": str(provider_home),
                "MNEMOSYNE_DATA_DIR": str(data_dir),
                "MNEMOSYNE_VEC_WEIGHT": "1",
                "MNEMOSYNE_FTS_WEIGHT": "0",
                "MNEMOSYNE_IMPORTANCE_WEIGHT": "1",
                "MNEMOSYNE_NO_EMBEDDINGS": "1",
            },
        )
        assert json.loads(result.stdout) == {
            "fts": 1.0,
            "importance": 0.0,
            "temporal": 0.0,
            "vec": 0.0,
        }
