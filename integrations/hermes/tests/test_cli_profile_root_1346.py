"""Hermes CLI and provider must select the same profile database (#1346)."""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

from mnemosyne_hermes.cli import mnemosyne_command, register_cli


def test_cli_help_names_profile_root_precedence_and_standalone_fallback(capsys):
    parser = argparse.ArgumentParser()
    register_cli(parser)
    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "HERMES_HOME" in help_text
    assert "mnemosyne/data" in help_text
    assert "MNEMOSYNE_DATA_DIR" in help_text
    assert "standalone" in help_text.lower()


def test_doctor_reports_install_command_when_engine_is_missing(monkeypatch, capsys):
    from types import SimpleNamespace
    import mnemosyne_hermes.cli as cli

    def missing_engine(*_args, **_kwargs):
        raise ModuleNotFoundError("No module named 'mnemosyne'", name="mnemosyne")

    monkeypatch.setattr(cli, "_run_cli_diagnostics", missing_engine)
    result = cli.mnemosyne_command(
        SimpleNamespace(mnemosyne_cmd="doctor", bank=None, no_fix=True)
    )

    output = capsys.readouterr().out
    assert result == 1
    assert "Mnemosyne core is missing" in output
    assert f'--python "{sys.executable}"' in output
    assert "--upgrade mnemosyne-hermes" in output


@pytest.mark.parametrize("outcome", ["nonzero", "timeout", "malformed", "wrong-shape", "success"])
def test_legacy_doctor_subprocess_reports_safe_failure_categories(
    tmp_path, monkeypatch, capsys, outcome
):
    """Exercise the CLI boundary with real selected and competing stores."""
    from types import SimpleNamespace
    from mnemosyne.core.beam import BeamMemory
    import mnemosyne.diagnose as diagnose

    home = tmp_path / "profile"
    home.mkdir()
    selected_dir = home / "mnemosyne" / "data"
    selected_db = selected_dir / "mnemosyne.db"
    competing_dir = tmp_path / "ambient"
    competing_db = competing_dir / "mnemosyne.db"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(competing_dir))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    BeamMemory(db_path=selected_db).scratchpad_write("selected-marker")
    BeamMemory(db_path=competing_db).scratchpad_write("competing-marker")
    assert selected_db.is_file() and competing_db.is_file()
    with sqlite3.connect(selected_db) as conn:
        assert conn.execute("SELECT content FROM scratchpad").fetchone() == ("selected-marker",)
    with sqlite3.connect(competing_db) as conn:
        assert conn.execute("SELECT content FROM scratchpad").fetchone() == ("competing-marker",)

    def legacy_diagnostics(*, bank=None):
        raise AssertionError("legacy diagnostics must run in a child")

    monkeypatch.setattr(diagnose, "run_diagnostics", legacy_diagnostics)
    secret = f"\x1b[31m{selected_db} private memory text\ncommand/script\x1b[0m"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert argv[0] == sys.executable and argv[1] == "-c" and argv[3] == ""
        assert "run_diagnostics" in argv[2]
        assert kwargs["env"]["MNEMOSYNE_DATA_DIR"] == str(selected_dir)
        assert kwargs["check"] is True
        if outcome == "nonzero":
            raise subprocess.CalledProcessError(9, argv, output=secret, stderr=secret)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, 120, output=secret, stderr=secret)
        payload = (
            "{malformed" if outcome == "malformed" else
            json.dumps([secret]) if outcome == "wrong-shape" else
            json.dumps({"resolved_db": str(selected_db), "checks_passed": 1, "checks_total": 1})
        )
        return subprocess.CompletedProcess(argv, 0, payload, secret)

    monkeypatch.setattr(subprocess, "run", fake_run)
    original_dir = os.environ["MNEMOSYNE_DATA_DIR"]
    result = mnemosyne_command(
        SimpleNamespace(mnemosyne_cmd="doctor", bank=None, no_fix=True)
    )
    output = capsys.readouterr().out
    assert len(calls) == 1
    assert os.environ["MNEMOSYNE_DATA_DIR"] == original_dir
    if outcome == "success":
        assert result == 0
        assert f"resolved_db: {selected_db}" in output
    else:
        assert result == 1
        category = {
            "nonzero": "exited unsuccessfully", "timeout": "timed out",
            "malformed": "returned invalid JSON", "wrong-shape": "returned invalid JSON",
        }[outcome]
        assert output == f"Diagnostic failed: legacy diagnostic subprocess {category}\n"
        assert secret not in output
        assert str(selected_db) not in output
        assert "private memory text" not in output
        assert "command/script" not in output
        assert "\x1b" not in output
        assert "-c" not in output


_PROBE = r'''
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
from types import SimpleNamespace

socket.socket.connect = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network disabled"))
from mnemosyne.core.beam import BeamMemory
from mnemosyne_hermes import MnemosyneMemoryProvider
from mnemosyne_hermes.cli import mnemosyne_command
if os.environ.get("TEST_LEGACY_DIAGNOSTICS") == "1":
    import mnemosyne.diagnose as diagnose
    def legacy_signature(*, bank=None):
        raise AssertionError("legacy diagnostic must run in a separate process")
    diagnose.run_diagnostics = legacy_signature

home = Path(os.environ["HERMES_HOME"])
isolated = os.environ.get("TEST_PROFILE_ISOLATION") == "1"
if isolated:
    (home / "config.yaml").write_text("memory:\n  mnemosyne:\n    profile_isolation: true\n")
provider = MnemosyneMemoryProvider()
provider.initialize(session_id="default", hermes_home=str(home))
assert provider._beam is not None, provider._init_error
provider._beam.remember("lamp calibration marker", source="preference", scope="global")
provider_path = Path(provider._beam.db_path).resolve()
expected = home / "mnemosyne" / "data"
if isolated:
    expected = expected / "banks" / home.name
assert provider_path == expected / "mnemosyne.db"

if os.environ.get("MNEMOSYNE_DATA_DIR") == str(home / "mnemosyne"):
    other = BeamMemory(session_id="hermes_default")
    other.remember("alternate marker", source="preference", scope="global")
    other.remember("another alternate marker", source="preference", scope="global")
    assert Path(other.db_path).resolve() != provider_path

responses = {}
for command, opts in (("stats", {}), ("inspect", {"query": "lamp", "limit": 5}),
                      ("doctor", {"no_fix": True})):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = mnemosyne_command(SimpleNamespace(mnemosyne_cmd=command, bank=None, **opts))
    responses[command] = {"rc": rc, "out": out.getvalue()}

if isolated:
    export_path = home.parent / "selected.json"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = mnemosyne_command(SimpleNamespace(mnemosyne_cmd="export", bank=None,
                                               output=str(export_path)))
    assert rc == 0, out.getvalue()
    payload = json.loads(export_path.read_text())
    assert "lamp calibration marker" in json.dumps(payload["working_memory"])
    missing = expected.parent / "missing" / "mnemosyne.db"
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = mnemosyne_command(SimpleNamespace(mnemosyne_cmd="doctor", bank="missing", no_fix=True))
    assert rc == 1 and "Bank not found: missing" in out.getvalue()
    assert not missing.exists()

if os.environ.get("MNEMOSYNE_DATA_DIR") == str(home / "mnemosyne"):
    from mnemosyne.core.memory import Mnemosyne
    source = home.parent / "alternate.json"
    Mnemosyne(session_id="hermes_default", db_path=other.db_path).export_to_file(source)
    alternate_before = other.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
    for dry_run in (True, False):
        out = io.StringIO()
        args = SimpleNamespace(mnemosyne_cmd="import", bank=None, input=str(source),
                               from_provider=None, force=False, dry_run=dry_run)
        with contextlib.redirect_stdout(out):
            rc = mnemosyne_command(args)
        assert rc == 0, out.getvalue()
        count = provider._beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        assert count == (1 if dry_run else 3), (out.getvalue(), count)
        with sqlite3.connect(other.db_path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == alternate_before

# A standalone core caller still honors its deliberate override.
standalone = BeamMemory(session_id="standalone")
print(json.dumps({"provider_db": str(provider_path), "standalone_db": str(standalone.db_path),
                  "responses": responses}))
provider.shutdown()
'''


@pytest.mark.parametrize("override", ["unset", "same", "different"])
@pytest.mark.parametrize("isolated", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_packaged_cli_selects_provider_root_for_stats_inspect_doctor(tmp_path, override, isolated, legacy):
    home = tmp_path / "profile"
    home.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MNEMOSYNE_", "HERMES_", "OPENAI_", "ANTHROPIC_",
                                "AZURE_", "GEMINI_", "HF_"))}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home),
               MNEMOSYNE_NO_EMBEDDINGS="1", MNEMOSYNE_HOST_LLM_ENABLED="0",
               TEST_PROFILE_ISOLATION="1" if isolated else "0",
               TEST_LEGACY_DIAGNOSTICS="1" if legacy else "0")
    if override != "unset":
        env["MNEMOSYNE_DATA_DIR"] = str(
            home / "mnemosyne" / ("data" if override == "same" else "")
        )
    completed = subprocess.run([sys.executable, "-c", _PROBE], env=env,
                               capture_output=True, text=True, check=True)
    result = json.loads(completed.stdout)
    responses = result["responses"]
    assert all(item["rc"] == 0 for item in responses.values())
    assert json.loads(responses["stats"]["out"])["working"]["total"] == 1
    assert "lamp calibration marker" in responses["inspect"]["out"]
    assert f"resolved_db: {result['provider_db']}" in responses["doctor"]["out"]
    if override == "different" or isolated:
        assert result["standalone_db"] != result["provider_db"]
    else:
        assert result["standalone_db"] == result["provider_db"]


_RELEASE_PROBE = r'''
import contextlib
import inspect
import io
import os
from pathlib import Path
import socket
from types import SimpleNamespace

socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network disabled"))
import mnemosyne
from mnemosyne.core.beam import BeamMemory
from mnemosyne.diagnose import run_diagnostics
from mnemosyne_hermes.cli import mnemosyne_command

home = Path(os.environ["HERMES_HOME"])
site = Path(os.environ["TEST_CORE_SITE"])
assert Path(mnemosyne.__file__).resolve().is_relative_to(site)
assert mnemosyne.__version__ == "4.0.0"
assert "db_path" not in inspect.signature(run_diagnostics).parameters
root = home / "mnemosyne" / "data"
for bank, selected in ((None, root / "mnemosyne.db"),
                       (home.name, root / "banks" / home.name / "mnemosyne.db")):
    BeamMemory(db_path=selected, session_id="hermes_default").remember("marker", scope="global")
    if bank:
        (home / "config.yaml").write_text("memory:\n  mnemosyne:\n    profile_isolation: true\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = mnemosyne_command(SimpleNamespace(mnemosyne_cmd="doctor", bank=None, no_fix=True))
    assert rc == 0, out.getvalue()
    assert f"resolved_db: {selected}" in out.getvalue()
    assert f"resolved_bank: {bank or 'default'}" in out.getvalue()
    assert os.environ["MNEMOSYNE_DATA_DIR"] == str(home / "mnemosyne")
'''


def test_published_core_400_cli_doctor_compatibility(tmp_path):
    """Opt-in offline check of the minimum supported Core wheel, not a mock."""
    wheel = os.environ.get("MNEMOSYNE_CORE_400_WHEEL")
    if not wheel:
        pytest.skip("set MNEMOSYNE_CORE_400_WHEEL to a local published wheel")
    site = tmp_path / "site"
    site.mkdir()
    with ZipFile(wheel) as archive:
        archive.extractall(site)
    home = tmp_path / "profile"
    home.mkdir()
    provider_src = Path(__file__).resolve().parents[1] / "src"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("MNEMOSYNE_", "HERMES_", "OPENAI_", "ANTHROPIC_",
                                "AZURE_", "GEMINI_", "HF_"))}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home),
               MNEMOSYNE_DATA_DIR=str(home / "mnemosyne"), MNEMOSYNE_NO_EMBEDDINGS="1",
               TEST_CORE_SITE=str(site), PYTHONPATH=os.pathsep.join((str(site), str(provider_src))))
    result = subprocess.run([sys.executable, "-c", _RELEASE_PROBE], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


_MUTATION_PROBE = r'''
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
from types import SimpleNamespace

socket.socket.connect = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network disabled"))
from mnemosyne.core.beam import BeamMemory
from mnemosyne_hermes import MnemosyneMemoryProvider
from mnemosyne_hermes.cli import mnemosyne_command

home = Path(os.environ["HERMES_HOME"])
ambient = Path(os.environ["MNEMOSYNE_DATA_DIR"])
isolated = os.environ["TEST_PROFILE_ISOLATION"] == "1"
if isolated:
    (home / "config.yaml").write_text("memory:\n  mnemosyne:\n    profile_isolation: true\n")
provider = MnemosyneMemoryProvider()
provider.initialize(session_id="default", hermes_home=str(home))
assert provider._beam is not None, provider._init_error
selected = home / "mnemosyne" / "data"
if isolated:
    selected = selected / "banks" / home.name
selected_db = selected / "mnemosyne.db"
competing_db = ambient / "mnemosyne.db"
assert Path(provider._beam.db_path).resolve() == selected_db.resolve()
assert selected_db != competing_db
provider._beam.remember("selected sleep marker", scope="global")
provider._beam.scratchpad_write("selected scratch marker")
competitor = BeamMemory(db_path=competing_db, session_id="hermes_default")
competitor.remember("competing sleep marker", scope="global")
competitor.scratchpad_write("competing scratch marker")

def contents(db, table):
    with sqlite3.connect(db) as conn:
        return [row[0] for row in conn.execute(f"SELECT content FROM {table} ORDER BY content")]

assert contents(selected_db, "working_memory") == ["selected sleep marker"]
assert contents(competing_db, "working_memory") == ["competing sleep marker"]
assert contents(selected_db, "scratchpad") == ["selected scratch marker"]
assert contents(competing_db, "scratchpad") == ["competing scratch marker"]

# Stub consolidation work, but retain real CLI Beam construction and real SQLite reads.
def observed_sleep(self, *, dry_run=False):
    return {"db": str(Path(self.db_path).resolve()), "rows": contents(self.db_path, "working_memory"),
            "dry_run": dry_run, "mode": "session"}

def observed_all(self, *, dry_run=False):
    return {**observed_sleep(self, dry_run=dry_run), "mode": "all"}

BeamMemory.sleep = observed_sleep
BeamMemory.sleep_all_sessions = observed_all
for all_sessions in (False, True):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = mnemosyne_command(SimpleNamespace(mnemosyne_cmd="sleep", bank=None,
                                               all_sessions=all_sessions, dry_run=True))
    assert rc == 0, out.getvalue()
    result = json.loads(out.getvalue())
    assert result == {"db": str(selected_db.resolve()), "rows": ["selected sleep marker"],
                      "dry_run": True, "mode": "all" if all_sessions else "session"}
assert contents(competing_db, "working_memory") == ["competing sleep marker"]

builtins.input = lambda prompt: "y"
out = io.StringIO()
with contextlib.redirect_stdout(out):
    rc = mnemosyne_command(SimpleNamespace(mnemosyne_cmd="clear", bank=None))
assert rc == 0 and "Scratchpad cleared." in out.getvalue()
assert contents(selected_db, "scratchpad") == []
assert contents(competing_db, "scratchpad") == ["competing scratch marker"]
assert contents(selected_db, "working_memory") == ["selected sleep marker"]
assert contents(competing_db, "working_memory") == ["competing sleep marker"]
assert os.environ["MNEMOSYNE_DATA_DIR"] == str(ambient)
provider.shutdown()
'''


@pytest.mark.parametrize("isolated", [False, True])
def test_sleep_and_clear_use_selected_profile_not_ambient_root(tmp_path, isolated):
    home = tmp_path / "profile"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith((
        "MNEMOSYNE_", "HERMES_", "OPENAI_", "ANTHROPIC_", "AZURE_", "GEMINI_", "HF_",
    ))}
    env.update(HOME=str(tmp_path), HERMES_HOME=str(home),
               MNEMOSYNE_DATA_DIR=str(tmp_path / "ambient"),
               MNEMOSYNE_NO_EMBEDDINGS="1", MNEMOSYNE_HOST_LLM_ENABLED="0",
               TEST_PROFILE_ISOLATION="1" if isolated else "0")
    completed = subprocess.run([sys.executable, "-c", _MUTATION_PROBE], env=env,
                               cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
