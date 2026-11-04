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
    helper = ROOT / "tests" / "_config_isolation.py"
    if helper.exists():
        shutil.copyfile(helper, project / "conftest.py")
    elif (ROOT / "conftest.py").exists():
        shutil.copyfile(ROOT / "conftest.py", project / "conftest.py")
    (project / "test_probe.py").write_text(
        '''from pathlib import Path
import os
from mnemosyne.core.config import get_config, MnemosyneConfig
from mnemosyne.core.beam import BeamMemory

def assert_isolated_fallback_paths():
    original = {key: os.environ.get(key) for key in ("MNEMOSYNE_DATA_DIR", "HERMES_HOME")}
    data = Path(os.environ["MNEMOSYNE_DATA_DIR"])
    hermes = Path(os.environ["HERMES_HOME"])
    home = Path(os.environ["HOME"])
    try:
        for expected in (data / "config.yaml", hermes / "mnemosyne" / "config.yaml", home / ".hermes" / "mnemosyne" / "config.yaml"):
            MnemosyneConfig.reset_instance()
            assert get_config().config_path == expected
            assert get_config().get_bool("cross_session") is False
            if "MNEMOSYNE_DATA_DIR" in os.environ:
                os.environ.pop("MNEMOSYNE_DATA_DIR")
            else:
                os.environ.pop("HERMES_HOME", None)
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        MnemosyneConfig.reset_instance()

assert_isolated_fallback_paths()
collection_cross_session = get_config().get_bool("cross_session")

def test_collection_and_beam_defaults(tmp_path):
    assert Path(os.environ["HOME"]) == tmp_path / "home"
    assert_isolated_fallback_paths()
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


@pytest.mark.parametrize("suite", ["tests", "integrations/hermes/tests", "both"])
def test_checkout_named_mnemosyne_does_not_load_plugin_as_core(tmp_path, suite):
    """CI's checkout basename must not let root plugin packaging shadow Core."""
    project = tmp_path / "checkouts" / "mnemosyne"
    project.mkdir(parents=True)
    (project / "integrations").mkdir()
    shutil.copyfile(ROOT / "__init__.py", project / "__init__.py")
    if (ROOT / "conftest.py").exists():
        shutil.copyfile(ROOT / "conftest.py", project / "conftest.py")
    helper = ROOT / "tests" / "_config_isolation.py"
    if helper.exists():
        (project / "tests").mkdir(exist_ok=True)
        shutil.copyfile(helper, project / "tests" / "_config_isolation.py")
    suites = ["tests", "integrations/hermes/tests"] if suite == "both" else [suite]
    for name in suites:
        directory = project / name
        directory.mkdir(parents=True, exist_ok=True)
        source = ROOT / name / "conftest.py"
        if source.exists():
            shutil.copyfile(source, directory / "conftest.py")
        filename = "test_core_imports.py" if name == "tests" else "test_hermes_imports.py"
        (directory / filename).write_text(
            "from mnemosyne.integrations import hermes_persona_prompt, auto_save_openwebui\n"
            "def test_core_integrations_are_importable():\n"
            "    assert hermes_persona_prompt.HermesPersonaPromptMixin\n"
            "    assert callable(auto_save_openwebui.main)\n",
            encoding="utf-8",
        )
    environment = os.environ.copy()
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    code = (
        "import os, pytest, sys\n"
        "keys = ('HOME', 'HERMES_HOME', 'MNEMOSYNE_DATA_DIR')\n"
        "previous = {key: os.environ.get(key) for key in keys}\n"
        f"result = pytest.main({['-q', *suites]!r})\n"
        "assert {key: os.environ.get(key) for key in keys} == previous\n"
        "sys.exit(result)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project, env=environment, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
