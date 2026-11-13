"""Public CLI regressions: explicit missing recall banks must not be initialized."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _snapshot(root):
    return {
        str(path.relative_to(root)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        if path.is_file()
        else None
        for path in root.rglob("*")
    }


@pytest.fixture
def cli(tmp_path):
    """Use fresh processes and synthetic roots, never ambient user configuration."""
    temp = tmp_path / "tmp"
    temp.mkdir()
    env = {
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.defpath,
        "LANG": "C.UTF-8",
        "PYTHONUTF8": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT),
        "HOME": str(tmp_path / "home"),
        "HERMES_HOME": str(tmp_path / "hermes"),
        "MNEMOSYNE_HOME": str(tmp_path / "mnemo-home"),
        "MNEMOSYNE_DATA_DIR": str(tmp_path / "data"),
        "FLEET_ROOT": str(tmp_path / "fleet"),
        "XDG_CONFIG_HOME": str(tmp_path / "xdg-config"),
        "XDG_DATA_HOME": str(tmp_path / "xdg-data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "HF_HOME": str(tmp_path / "cache" / "hf"),
        "MNEMOSYNE_MODEL_DIR": str(tmp_path / "cache" / "models"),
        "TMPDIR": str(temp),
        "TMP": str(temp),
        "TEMP": str(temp),
        "MNEMOSYNE_NO_EMBEDDINGS": "1",
        "MNEMOSYNE_AUTO_SLEEP_ENABLED": "0",
        "MNEMOSYNE_DEFAULT_SCOPE": "global",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    # Permit an external test-only audit harness, not ambient app settings.
    if "TRIAGE_ROOT" in os.environ:
        for key in ("TRIAGE_ROOT", "TRIAGE_LOG", "PYTHONPATH"):
            env[key] = os.environ[key]

    def run(args, bank=None, overrides=None):
        child_env = dict(env)
        if bank is not None:
            child_env["MNEMOSYNE_BANK"] = bank
        for key, value in (overrides or {}).items():
            if value is None:
                child_env.pop(key, None)
            else:
                child_env[key] = str(value)
        return subprocess.run(
            [sys.executable, "-m", "mnemosyne.cli", *args],
            cwd=tmp_path,
            env=child_env,
            capture_output=True,
            text=True,
            timeout=30,
        )

    return run


@pytest.mark.parametrize("command", ["recall", "search"])
@pytest.mark.parametrize("state", ["missing", "directory-without-db"])
def test_missing_explicit_bank_fails_before_initialization(
    cli, tmp_path, command, state
):
    if state == "directory-without-db":
        (tmp_path / "data" / "banks" / "does-not-exist").mkdir(parents=True)
    before = _snapshot(tmp_path)
    result = cli([command, "x", "5", "--json"], "does-not-exist")
    assert result.returncode == 2, "missing bank must not look like an empty search"
    message = (
        "Bank 'does-not-exist' does not exist"
        if state == "missing"
        else "Database for bank 'does-not-exist' does not exist"
    )
    assert result.stderr == f"Error: {message}\n"
    assert result.stdout == ""
    after = _snapshot(tmp_path)
    # Dispatch may defensively create DATA_DIR, but never a bank, config or DB.
    assert {key: value for key, value in after.items() if key != "data"} == {
        key: value for key, value in before.items() if key != "data"
    }


@pytest.mark.parametrize("command", ["recall", "search"])
@pytest.mark.parametrize("bank", [None, "default", " default ", "", "   "])
def test_default_bootstrap_is_preserved(cli, tmp_path, command, bank):
    result = cli([command, "x", "5", "--json"], bank)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"query": "x", "top_k": 5, "results": []}
    assert (tmp_path / "data" / "mnemosyne.db").is_file()


@pytest.mark.parametrize("command", ["recall", "search"])
def test_existing_bank_match_and_no_match(cli, command):
    stored = cli(["store", "syntheticquartz unique memory", "cli", "0.83"], "known")
    assert stored.returncode == 0, stored.stderr
    matched = cli([command, "syntheticquartz", "5", "--json"], " known ")
    assert matched.returncode == 0, matched.stderr
    results = json.loads(matched.stdout)["results"]
    assert [row["content"] for row in results] == ["syntheticquartz unique memory"]
    missing = cli([command, "zzzzunmatchedzzzz", "5", "--json"], "known")
    assert missing.returncode == 0, missing.stderr
    assert json.loads(missing.stdout)["results"] == []


@pytest.mark.parametrize("command", ["recall", "search"])
def test_bank_create_empty_container_can_be_initialized(cli, tmp_path, command):
    created = cli(["bank", "create", "known"])
    assert created.returncode == 0, created.stderr
    db = tmp_path / "data" / "banks" / "known" / "mnemosyne.db"
    assert db.is_file() and db.stat().st_size == 0
    result = cli([command, "x", "5", "--json"], "known")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["results"] == []
    assert db.stat().st_size > 0


@pytest.mark.parametrize("command", ["store", "remember"])
def test_write_to_missing_bank_still_creates_it(cli, tmp_path, command):
    stored = cli([command, "syntheticquartz unique memory", "cli"], "new-write")
    assert stored.returncode == 0, stored.stderr
    assert "Stored:" in stored.stdout
    assert (tmp_path / "data" / "banks" / "new-write" / "mnemosyne.db").is_file()
    found = cli(["recall", "syntheticquartz", "5", "--json"], "new-write")
    assert found.returncode == 0, found.stderr
    assert len(json.loads(found.stdout)["results"]) == 1


@pytest.mark.parametrize("command", ["recall", "search"])
def test_invalid_bank_is_rejected_before_memory_import(cli, tmp_path, command):
    before = _snapshot(tmp_path)
    result = cli([command, "x", "5", "--json"], "../escape")
    assert result.returncode == 2
    assert "Invalid bank name '../escape'" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.stdout == ""
    after = _snapshot(tmp_path)
    assert {k: v for k, v in after.items() if k != "data"} == before


@pytest.mark.parametrize("command", ["recall", "search"])
def test_missing_bank_does_not_fall_back_to_foreign_root(cli, tmp_path, command):
    foreign = tmp_path / "hermes" / "mnemosyne" / "data"
    stored = cli(
        ["store", "syntheticquartz foreign memory", "cli"],
        "known",
        {"MNEMOSYNE_DATA_DIR": None},
    )
    assert stored.returncode == 0, stored.stderr
    assert (foreign / "banks" / "known" / "mnemosyne.db").is_file()
    before = _snapshot(foreign)
    result = cli([command, "syntheticquartz", "5", "--json"], " known ")
    assert result.returncode == 2
    assert result.stderr == "Error: Bank 'known' does not exist\n"
    assert result.stdout == ""
    assert _snapshot(foreign) == before
    assert not (tmp_path / "data" / "banks").exists()
