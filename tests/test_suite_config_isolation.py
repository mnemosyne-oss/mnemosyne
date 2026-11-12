"""The test harness must not read ambient developer configuration (#1321)."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("inherited_overrides", [False, True], ids=["plain-env", "scope-overrides"])
@pytest.mark.parametrize("ambient_source", ["home", "hermes_home", "data_dir", "windows_home"])
def test_ambient_quality_profile_cannot_change_collection_or_test_defaults(
    tmp_path, monkeypatch, ambient_source, inherited_overrides,
):
    """Keep inherited profiles out of collection and per-test defaults."""
    if inherited_overrides:
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "true")
        monkeypatch.setenv("MNEMOSYNE_DEFAULT_SCOPE", "global")
        monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
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
import ntpath

# Exercise the Windows stdlib home-selection rule on every test platform.
if __WINDOWS_HOME__:
    Path.home = classmethod(lambda cls: cls(ntpath.expanduser("~")))

from mnemosyne.core.config import get_config, MnemosyneConfig
from mnemosyne.core.beam import BeamMemory

def assert_isolated_fallback_paths():
    original = {key: os.environ.get(key) for key in ("MNEMOSYNE_DATA_DIR", "HERMES_HOME")}
    data = Path(os.environ["MNEMOSYNE_DATA_DIR"])
    hermes = Path(os.environ["HERMES_HOME"])
    home = Path(os.environ["HOME"])
    assert Path.home() == home
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
'''.replace("__WINDOWS_HOME__", repr(ambient_source == "windows_home")), encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment.update(HOME=str(developer), USERPROFILE=str(developer), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    environment.pop("MNEMOSYNE_DATA_DIR", None)
    environment.pop("HERMES_HOME", None)
    environment.pop("MNEMOSYNE_CROSS_SESSION", None)
    environment.pop("MNEMOSYNE_DEFAULT_SCOPE", None)
    if ambient_source == "hermes_home":
        environment["HERMES_HOME"] = str(hermes)
    elif ambient_source == "data_dir":
        environment["MNEMOSYNE_DATA_DIR"] = str(data)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(project)],
        cwd=project, env=environment, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout, result.stdout + result.stderr
    # The synthetic developer profiles must remain intact.
    assert "cross_session: true" in (hermes / "mnemosyne" / "config.yaml").read_text()
    assert "cross_session: true" in (data / "config.yaml").read_text()


@pytest.mark.parametrize("suite", ["tests", "integrations/hermes/tests", "both"])
def test_checkout_named_mnemosyne_does_not_load_plugin_as_core(tmp_path, suite):
    """CI's checkout basename must not let root plugin packaging shadow Core."""
    result = _run_checkout_probe(tmp_path, suite)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("suite", ["tests", "integrations/hermes/tests", "both"])
def test_collection_rejects_precached_ambient_singleton(tmp_path, monkeypatch, suite):
    """A preload before pytest.main must be detected during collection."""
    monkeypatch.setenv("PYTEST_ADDOPTS", "--continue-on-collection-errors")
    result = _run_checkout_probe(tmp_path, suite, preload_config=True)
    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "AssertionError: cached collection config escaped isolation" in output
    assert "during collection" in output


def _run_checkout_probe(tmp_path, suite, *, preload_config=False):
    """Run real suite setup with a synthetic ambient profile in a fresh child."""
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
    ambient = tmp_path / "ambient-profile"
    for config_dir in (ambient / ".hermes" / "mnemosyne", ambient / "data"):
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text(
            "cross_session: true\ndefault_scope: global\n", encoding="utf-8",
        )
    suites = ["tests", "integrations/hermes/tests"] if suite == "both" else [suite]
    for name in suites:
        directory = project / name
        directory.mkdir(parents=True, exist_ok=True)
        source = ROOT / name / "conftest.py"
        if source.exists():
            shutil.copyfile(source, directory / "conftest.py")
        filename = "test_core_imports.py" if name == "tests" else "test_hermes_imports.py"
        # Probe before function fixtures can hide a missing suite configure hook.
        # The ntpath rule also covers an absent incoming USERPROFILE on POSIX.
        collection_probe = (
            "from pathlib import Path\n"
            "import os\n"
            "import ntpath\n"
            "from mnemosyne.core.config import get_config\n"
            "Path.home = classmethod(lambda cls: cls(ntpath.expanduser('~')))\n"
            "home = Path(os.environ['HOME'])\n"
            f"assert home != Path({str(ambient)!r}), 'collection home escaped isolation'\n"
            "assert Path.home() == home\n"
            "assert Path(os.environ['USERPROFILE']) == home\n"
            "assert Path(os.environ['HERMES_HOME']) == home / '.hermes'\n"
            "data = Path(os.environ['MNEMOSYNE_DATA_DIR'])\n"
            "assert data == home / 'data'\n"
            # Inspect the existing singleton before any reset can hide an
            # ambient configuration read during conftest import.
            "config = get_config()\n"
            "assert config.config_path == data / 'config.yaml', 'cached collection config escaped isolation'\n"
            "assert config.get_bool('cross_session') is False\n"
            "assert config.get('default_scope', 'session') == 'session'\n"
        )
        (directory / filename).write_text(
            collection_probe
            + "from mnemosyne.integrations import hermes_persona_prompt, auto_save_openwebui\n"
            "def test_core_integrations_are_importable():\n"
            "    assert hermes_persona_prompt.HermesPersonaPromptMixin\n"
            "    assert callable(auto_save_openwebui.main)\n",
            encoding="utf-8",
        )
    environment = os.environ.copy()
    environment.pop("PYTEST_ADDOPTS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    environment.update(
        HOME=str(ambient), USERPROFILE=str(ambient),
        HERMES_HOME=str(ambient / ".hermes"), MNEMOSYNE_DATA_DIR=str(ambient / "data"),
    )
    environment.pop("MNEMOSYNE_CROSS_SESSION", None)
    environment.pop("MNEMOSYNE_DEFAULT_SCOPE", None)
    if suite == "tests":
        environment.pop("USERPROFILE")
        drive, tail = os.path.splitdrive(str(ambient))
        environment.update(HOMEDRIVE=drive, HOMEPATH=tail)
    preload = (
        "from pathlib import Path\n"
        "from mnemosyne.core.config import get_config\n"
        "cached = get_config()\n"
        "assert cached.config_path == Path(os.environ['MNEMOSYNE_DATA_DIR']) / 'config.yaml'\n"
        "assert cached.get_bool('cross_session') is True\n"
        "assert cached.get('default_scope') == 'global'\n"
        if preload_config else ""
    )
    code = (
        "import os, pytest, sys\n"
        + preload
        + "keys = ('HOME', 'USERPROFILE', 'HERMES_HOME', 'MNEMOSYNE_DATA_DIR', 'HOMEDRIVE', 'HOMEPATH')\n"
        "previous = {key: os.environ.get(key) for key in keys}\n"
        f"result = pytest.main({['-q', *suites]!r})\n"
        "assert {key: os.environ.get(key) for key in keys} == previous\n"
        "sys.exit(result)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=project, env=environment, text=True, capture_output=True, timeout=60,
    )
    return result
