"""Runtime-consumer regressions for the #482 ``degrade_batch`` slice."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.config import MnemosyneConfig

ROOT = Path(__file__).resolve().parents[1]
HERMES_SRC = ROOT / "integrations" / "hermes" / "src"


def _run(script: str, *, env: dict[str, str], unset: tuple[str, ...] = ()) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(env)
    for name in unset:
        environment.pop(name, None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(HERMES_SRC), environment.get("PYTHONPATH", "")]
    )
    return subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, env=environment, check=True
    )


_DEGRADE_SCRIPT = """
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from mnemosyne.core.beam import BeamMemory

memory = BeamMemory(session_id="degrade", db_path=Path(os.environ["TEST_DB"]))
try:
    old = (datetime.now() - timedelta(days=365)).isoformat()
    memory.conn.executemany(
        "INSERT INTO episodic_memory (id, content, created_at, tier) VALUES (?, ?, ?, 1)",
        [(f"row-{index}", f"runtime degrade sentinel {index}", old) for index in range(101)],
    )
    memory.conn.commit()
    print(json.dumps(memory.degrade_episodic(dry_run=True), sort_keys=True))
finally:
    memory.conn.close()
"""


@pytest.mark.parametrize(
    ("yaml", "env_value", "expected"),
    [
        ("degrade_batch: 3\n", "5", 3),
        ("cross_session: false\n", "2", 2),
        ("cross_session: false\n", None, 100),
    ],
)
def test_degrade_batch_direct_consumer_honors_yaml_env_and_default(
    tmp_path: Path, yaml: str, env_value: str | None, expected: int
):
    """The real degradation query uses YAML > env > default, not an import cache."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(yaml)
    env = {
        "MNEMOSYNE_DATA_DIR": str(data_dir),
        "MNEMOSYNE_NO_EMBEDDINGS": "1",
        "TEST_DB": str(tmp_path / "memory.db"),
    }
    if env_value is not None:
        env["MNEMOSYNE_DEGRADE_BATCH"] = env_value
    result = _run(
        _DEGRADE_SCRIPT,
        env=env,
        unset=("MNEMOSYNE_DEGRADE_BATCH",) if env_value is None else (),
    )
    assert json.loads(result.stdout)["tier1_to_tier2"] == expected


def _seed(memory: BeamMemory, *, tier: int, count: int) -> None:
    old = (datetime.now() - timedelta(days=365)).isoformat()
    memory.conn.executemany(
        "INSERT INTO episodic_memory (id, content, created_at, tier) VALUES (?, ?, ?, ?)",
        [(f"tier-{tier}-{index}", f"degrade snapshot {tier} {index}", old, tier) for index in range(count)],
    )
    memory.conn.commit()


def test_degrade_batch_reload_applies_to_next_complete_operation(tmp_path: Path, monkeypatch):
    """A YAML reload changes the next pass, not the config reader alone."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("degrade_batch: 1\n")
    config = MnemosyneConfig(config_path)
    monkeypatch.setattr("mnemosyne.core.config.get_config", lambda: config)
    memory = BeamMemory(session_id="degrade", db_path=tmp_path / "memory.db")
    try:
        _seed(memory, tier=1, count=5)
        assert memory.degrade_episodic(dry_run=True)["tier1_to_tier2"] == 1
        config_path.write_text("degrade_batch: 3\n")
        assert config.reload() == {"degrade_batch"}
        assert memory.degrade_episodic(dry_run=True)["tier1_to_tier2"] == 3
    finally:
        memory.conn.close()


def test_degrade_batch_is_snapshotted_once_for_tier_queries(tmp_path: Path, monkeypatch):
    """A complete pass cannot mix generations between tier candidate queries."""
    values = iter((2, 5))

    class ChangingConfig:
        calls = 0

        def get_int(self, key: str, default: int) -> int:
            assert key == "degrade_batch"
            self.calls += 1
            return next(values)

    config = ChangingConfig()
    monkeypatch.setattr("mnemosyne.core.config.get_config", lambda: config)
    memory = BeamMemory(session_id="degrade", db_path=tmp_path / "memory.db")
    try:
        _seed(memory, tier=1, count=5)
        _seed(memory, tier=2, count=5)
        result = memory.degrade_episodic(dry_run=True)
        assert result["tier1_to_tier2"] == 2
        assert result["tier2_to_tier3"] == 1
        assert config.calls == 1
    finally:
        memory.conn.close()


def test_hermes_sleep_surface_reaches_degrade_batch(tmp_path: Path):
    """Both public Hermes providers reach the same YAML-resolved degradation pass."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("degrade_batch: 2\n")
    script = """
import importlib
import json
import os
from datetime import datetime, timedelta

Provider = importlib.import_module(os.environ["PROVIDER_MODULE"]).MnemosyneMemoryProvider
provider = Provider()
provider.initialize("degrade", hermes_home=os.environ["HERMES_HOME"])
assert provider._beam is not None
try:
    old = (datetime.now() - timedelta(days=365)).isoformat()
    provider._beam.conn.executemany(
        "INSERT INTO episodic_memory (id, content, created_at, tier) VALUES (?, ?, ?, 1)",
        [(f"provider-{index}", f"provider degrade {index}", old) for index in range(5)],
    )
    provider._beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, importance, session_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("sleep-trigger", "trigger", "test", old, 0.5, provider._beam.session_id),
    )
    provider._beam.conn.commit()
    # Exercise each provider's public tool routing. The provider returns a
    # JSON string whose result contains Beam's sleep/degradation payload.
    print(provider.handle_tool_call("mnemosyne_sleep", {"dry_run": True, "force": True}))
finally:
    provider._beam.conn.close()
"""
    for module in ("hermes_memory_provider", "mnemosyne_hermes"):
        home = tmp_path / module
        home.mkdir()
        result = _run(
            script,
            env={
                "PROVIDER_MODULE": module,
                "HERMES_HOME": str(home),
                "MNEMOSYNE_DATA_DIR": str(data_dir),
                "MNEMOSYNE_DEGRADE_BATCH": "5",
                "MNEMOSYNE_LLM_ENABLED": "0",
                "MNEMOSYNE_NO_EMBEDDINGS": "1",
            },
        )
        response = json.loads(result.stdout)
        assert response["result"]["degradation"]["tier1_to_tier2"] == 2
