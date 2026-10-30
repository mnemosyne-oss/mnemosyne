"""Tests for read-only source connections in backup/verify paths."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne import cli
from mnemosyne.dr import recovery


def _make_simple_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()


def test_create_backup_opens_source_read_only(tmp_path, monkeypatch):
    db_path = tmp_path / "src.db"
    _make_simple_db(db_path)
    bdir = tmp_path / "bk"

    calls = []
    original = recovery.sqlite3.connect

    def spy(connectable, *args, **kwargs):
        calls.append((connectable, dict(kwargs)))
        return original(connectable, *args, **kwargs)

    monkeypatch.setattr(recovery.sqlite3, "connect", spy)

    recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sqlite3")
    recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sql")

    ro_calls = [
        c for c in calls
        if isinstance(c[0], str) and c[0].startswith("file:" + str(db_path))
    ]
    assert any(c[1].get("uri") is True and "mode=ro" in c[0] for c in ro_calls), calls


def test_verify_integrity_opens_source_read_only(tmp_path, monkeypatch):
    db_path = tmp_path / "src.db"
    _make_simple_db(db_path)

    calls = []
    original = recovery.sqlite3.connect

    def spy(connectable, *args, **kwargs):
        calls.append((connectable, dict(kwargs)))
        return original(connectable, *args, **kwargs)

    monkeypatch.setattr(recovery.sqlite3, "connect", spy)

    recovery.verify_integrity(db_path)

    ro_calls = [
        c for c in calls
        if isinstance(c[0], str) and c[0].startswith("file:" + str(db_path))
    ]
    assert any(c[1].get("uri") is True and "mode=ro" in c[0] for c in ro_calls), calls


def test_cmd_verify_quick_positional(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "src.db"
    _make_simple_db(db_path)

    # make sure DATA_DIR points somewhere (not needed for positional)
    monkeypatch.setattr(cli, "DATA_DIR", str(tmp_path / "data"))

    cli.cmd_verify(["--quick", str(db_path)])
    out = capsys.readouterr()
    assert "Database integrity check passed" in out.out

    cli.cmd_verify([str(db_path), "--quick"])
    out = capsys.readouterr()
    assert "Database integrity check passed" in out.out


def test_cmd_verify_quick_nonexistent_does_not_create_or_succeed(tmp_path, monkeypatch, capsys):
    nonexistent = tmp_path / "nope.db"
    assert not nonexistent.exists()
    monkeypatch.setattr(cli, "DATA_DIR", str(tmp_path / "data"))

    with pytest.raises(SystemExit) as exc:
        cli.cmd_verify(["--quick", str(nonexistent)])
    assert exc.value.code == 1
    assert not nonexistent.exists()
    out = capsys.readouterr()
    assert "passed" not in out.out.lower()
