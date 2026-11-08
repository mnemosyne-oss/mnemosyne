"""Hermes CLI and provider must select the same profile database (#1346)."""

import json
import os
import subprocess
import sys

import pytest


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
           if not k.startswith(("MNEMOSYNE_", "HERMES_"))}
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
