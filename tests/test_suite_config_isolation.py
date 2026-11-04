"""The test harness must not read ambient developer configuration (#1321)."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("ambient_source", ["home", "hermes_home", "data_dir"])
def test_ambient_quality_profile_cannot_change_collection_or_test_defaults(tmp_path, ambient_source):
    developer = tmp_path / "developer"
    hermes = developer / ".hermes"
    data = tmp_path / "external-data"
    for directory in (hermes / "mnemosyne", data):
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(
            "cross_session: true\ndefault_scope: global\nenhanced_recall: true\n",
            encoding="utf-8",
        )
    project = tmp_path / "isolated-project"
    project.mkdir()
    # Exercise the real shared harness when present. Without the fix, the
    # child still runs and fails on the leaked configuration, not a missing file.
    if (ROOT / "conftest.py").exists():
        shutil.copyfile(ROOT / "conftest.py", project / "conftest.py")
    (project / "test_probe.py").write_text(
        '''from pathlib import Path
from mnemosyne.core.config import get_config, MnemosyneConfig
from mnemosyne.core.beam import BeamMemory

collection_cross_session = get_config().get_bool("cross_session")

def test_collection_and_beam_defaults(tmp_path):
    assert collection_cross_session is False
    assert get_config().get_bool("cross_session") is False
    assert get_config().get("default_scope", "session") == "session"
    memory = BeamMemory(session_id="probe", db_path=tmp_path / "probe.db")
    try:
        assert get_config().get_bool("cross_session") is False
    finally:
        memory.conn.close()

def test_explicit_test_configuration_is_honored(tmp_path, monkeypatch):
    directory = tmp_path / "explicit"
    directory.mkdir()
    (directory / "config.yaml").write_text("cross_session: true\\ndefault_scope: global\\n")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(directory))
    MnemosyneConfig.reset_instance()
    assert get_config().get_bool("cross_session") is True
    assert get_config().get("default_scope") == "global"

def test_next_test_gets_clean_defaults():
    assert get_config().get_bool("cross_session") is False
    assert get_config().get("default_scope", "session") == "session"
''', encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update(HOME=str(developer), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment.pop("MNEMOSYNE_DATA_DIR", None)
    environment.pop("HERMES_HOME", None)
    if ambient_source == "hermes_home":
        environment["HERMES_HOME"] = str(hermes)
    elif ambient_source == "data_dir":
        environment["MNEMOSYNE_DATA_DIR"] = str(data)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(project)],
        cwd=project, env=environment, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The synthetic developer profiles must remain intact.
    assert "cross_session: true" in (hermes / "mnemosyne" / "config.yaml").read_text()
    assert "cross_session: true" in (data / "config.yaml").read_text()
