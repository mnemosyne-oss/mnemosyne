"""Sleep/consolidate usage must be resolved before any store initialization."""

import itertools
import sqlite3
import sys

import pytest

import mnemosyne.cli as cli


ENTRYPOINTS = ["sleep", "consolidate", None]


def _invoke(monkeypatch, entrypoint, args):
    if entrypoint is None:
        return cli.cmd_sleep(args)
    monkeypatch.setattr(sys, "argv", ["mnemosyne", entrypoint, *args])
    return cli.run_cli()


@pytest.fixture
def guarded_store(tmp_path, monkeypatch):
    from mnemosyne.core import banks, config, memory

    data_dir = tmp_path / "absent-data"
    assert not data_dir.exists()
    monkeypatch.setattr(cli, "DATA_DIR", str(data_dir))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("usage must not initialize memory or open a database")

    monkeypatch.setattr(cli, "_get_memory", forbidden)
    monkeypatch.setattr(banks, "BankManager", forbidden)
    monkeypatch.setattr(memory, "Mnemosyne", forbidden)
    monkeypatch.setattr(config.MnemosyneConfig, "__init__", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr(sqlite3.dbapi2, "connect", forbidden)
    yield data_dir
    assert calls == []
    assert not data_dir.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("flag", ["--help", "-h"])
def test_sleep_help_is_side_effect_free(monkeypatch, capsys, guarded_store, entrypoint, flag):
    assert _invoke(monkeypatch, entrypoint, [flag]) is None
    output = capsys.readouterr()
    assert "Usage: mnemosyne" in output.out
    assert "sleep" in output.out
    for option in ("--force", "-f", "--all-sessions", "--dry-run"):
        assert option in output.out
    assert "Consolidation complete" not in output.out
    assert output.err == ""


INVALID_ARGS = [
    ["--unknown-1222"],
    ["--dry-run", "--unknown-1222"],
    ["--unknown-1222", "--all-sessions", "--force"],
    ["--all-sessions", "--dry-run", "-f", "--unknown-1222"],
    ["unexpected-positional"],
    ["--force", "unexpected-positional"],
    ["--for"],
    ["--all-session"],
    ["--dry-ru"],
    ["-ff"],
    ["--force=true"],
]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("args", INVALID_ARGS)
def test_sleep_rejects_unknown_tokens_before_writes(
    monkeypatch, capsys, guarded_store, entrypoint, args
):
    valid = {"--force", "-f", "--all-sessions", "--dry-run"}
    token = next(arg for arg in args if arg not in valid)
    with pytest.raises(SystemExit) as error:
        _invoke(monkeypatch, entrypoint, args)
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert token in output.err
    assert "Unknown" in output.err
    assert "Traceback" not in output.err


VALID_ARGS = [
    [*force, *all_sessions, *dry_run]
    for force, all_sessions, dry_run in itertools.product(
        [[], ["--force"], ["-f"]], [[], ["--all-sessions"]], [[], ["--dry-run"]]
    )
]


@pytest.mark.parametrize("entrypoint", ENTRYPOINTS)
@pytest.mark.parametrize("args", VALID_ARGS)
def test_sleep_preserves_valid_dispatch(monkeypatch, capsys, tmp_path, entrypoint, args):
    data_dir = tmp_path / "absent-data"
    monkeypatch.setattr(cli, "DATA_DIR", str(data_dir))
    calls = []

    class RecordingMemory:
        def sleep(self, **kwargs):
            calls.append(("sleep", kwargs))
            return {"instrumented": True}

        def sleep_all_sessions(self, **kwargs):
            calls.append(("sleep_all_sessions", kwargs))
            return {"instrumented": True}

    def get_memory():
        calls.append("_get_memory")
        return RecordingMemory()

    monkeypatch.setattr(cli, "_get_memory", get_memory)
    assert _invoke(monkeypatch, entrypoint, args) is None
    method = "sleep_all_sessions" if "--all-sessions" in args else "sleep"
    assert calls == [
        "_get_memory",
        (method, {"dry_run": "--dry-run" in args, "force": bool({"--force", "-f"} & set(args))}),
    ]
    if entrypoint is not None:
        assert data_dir.is_dir()
    output = capsys.readouterr()
    assert "Consolidation complete:" in output.out
    assert output.err == ""


@pytest.mark.parametrize("flag", ["--help", "-h", "help"])
def test_root_help_remains_side_effect_free(monkeypatch, capsys, guarded_store, flag):
    monkeypatch.setattr(sys, "argv", ["mnemosyne", flag])
    assert cli.run_cli() is None
    output = capsys.readouterr()
    assert "Usage: mnemosyne <command> [args]" in output.out
    assert output.err == ""
