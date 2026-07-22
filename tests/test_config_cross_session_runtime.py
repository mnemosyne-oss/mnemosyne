"""Fresh-process regressions for cross-session runtime config resolution."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_SRC = ROOT / "integrations" / "hermes" / "src"


def _run(script: str, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(env)
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
