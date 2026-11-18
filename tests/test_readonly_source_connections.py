"""Tests for read-only source connections in backup/verify paths."""

from __future__ import annotations

import sqlite3
from contextlib import closing
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
        if isinstance(c[0], str) and c[0].startswith("file:") and "mode=ro" in c[0]
    ]
    assert ro_calls, calls
    assert any(c[1].get("uri") is True for c in ro_calls), calls


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
        if isinstance(c[0], str) and c[0].startswith("file:") and "mode=ro" in c[0]
    ]
    assert ro_calls, calls
    assert any(c[1].get("uri") is True for c in ro_calls), calls


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


def test_corrupt_verify_points_to_wal_restore_procedure(tmp_path, capsys):
    db = tmp_path / "corrupt.db"
    db.write_bytes(b"not a SQLite database")
    before = db.read_bytes()
    with pytest.raises(SystemExit) as caught:
        cli.cmd_verify([str(db)])
    assert caught.value.code == 1
    output = capsys.readouterr().out
    assert "restore" in output.lower()
    assert "WAL" in output
    assert "docs/cli-reference.md" in output
    assert "Restoring a WAL store" in output
    assert db.read_bytes() == before


@pytest.mark.parametrize("operation", ["verify", "sqlite3", "sql"])
def test_idle_wal_inspection_requires_quiesced_maintenance(tmp_path, operation):
    db = tmp_path / "idle.db"
    with closing(sqlite3.connect(db)) as writer:
        assert writer.execute("PRAGMA journal_mode=WAL").fetchone()[0] == "wal"
        writer.execute("CREATE TABLE t(v TEXT)")
        writer.execute("INSERT INTO t VALUES ('original')")
        writer.commit()
        backup = Path(recovery.create_backup(db, tmp_path / "fixture")["backup_path"])
    sidecars = [db.with_name(db.name + suffix) for suffix in ("-wal", "-shm")]
    assert not any(p.exists() for p in sidecars)
    before = db.read_bytes()
    if operation == "verify":
        assert recovery.verify_integrity(db)
    else:
        recovery.create_backup(db, tmp_path / "new-backup", format=operation)
    assert all(p.exists() for p in sidecars)
    with pytest.raises(RuntimeError, match="read-only") as caught:
        recovery.restore_backup(backup, db)
    assert "all processes" in str(caught.value)
    assert db.read_bytes() == before
    # The fixture has stopped ALL clients. This maintenance is an explicit
    # operator action, not a quiescence detector or part of read-only backup.
    with closing(sqlite3.connect(db.as_uri() + "?mode=rw", uri=True, timeout=0)) as maintenance:
        assert maintenance.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    assert not any(p.exists() for p in sidecars)
    assert recovery.restore_backup(backup, db)["restored"]
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT v FROM t").fetchall() == [("original",)]


@pytest.mark.parametrize("live_kind", ["reader", "writer"])
def test_checkpoint_does_not_waive_live_store_sidecar_refusal(tmp_path, live_kind):
    db = tmp_path / "live.db"
    writer = sqlite3.connect(db)
    peer = None
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE t(v TEXT)")
        writer.execute("INSERT INTO t VALUES ('committed')")
        writer.commit()
        backup = Path(recovery.create_backup(db, tmp_path / "backup")["backup_path"])
        if live_kind == "reader":
            peer = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True)
            assert peer.execute("SELECT v FROM t").fetchall() == [("committed",)]
            writer.close()
        else:
            writer.execute("INSERT INTO t VALUES ('uncommitted')")
        with closing(sqlite3.connect(db.as_uri() + "?mode=rw", uri=True, timeout=0)) as maintenance:
            busy = maintenance.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]
        assert busy == (1 if live_kind == "writer" else 0)
        sidecars = [db.with_name(db.name + suffix) for suffix in ("-wal", "-shm")]
        assert all(p.exists() for p in sidecars)
        before = [db.read_bytes(), *(p.read_bytes() for p in sidecars)]
        with pytest.raises(RuntimeError, match="sidecar"):
            recovery.restore_backup(backup, db)
        assert [db.read_bytes(), *(p.read_bytes() for p in sidecars)] == before
    finally:
        if peer is not None:
            peer.close()
        writer.close()


@pytest.mark.parametrize("format", ["sqlite3", "sql"])
def test_readonly_backup_preserves_live_wal_and_committed_rows(tmp_path, format):
    db = tmp_path / "source.db"
    with closing(sqlite3.connect(db)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("CREATE TABLE t(v TEXT)")
        writer.execute("INSERT INTO t VALUES ('committed')")
        writer.commit()
        wal = db.with_name(db.name + "-wal")
        shm = db.with_name(db.name + "-shm")
        wal_size = wal.stat().st_size
        writer.execute("PRAGMA cache_size=1")
        writer.execute("PRAGMA cache_spill=ON")
        writer.execute("INSERT INTO t VALUES (?)", ("uncommitted" + "x" * 2_000_000,))
        assert wal.stat().st_size > wal_size
        before = (db.read_bytes(), wal.read_bytes(), wal.stat().st_ino, shm.stat().st_ino)
        backup = Path(recovery.create_backup(db, tmp_path / "backup", format=format)["backup_path"])
        assert (db.read_bytes(), wal.read_bytes(), wal.stat().st_ino, shm.stat().st_ino) == before
        target = tmp_path / "other.db"
        assert recovery.restore_backup(backup, target)["restored"]
        with closing(sqlite3.connect(target)) as conn:
            assert conn.execute("SELECT v FROM t").fetchall() == [("committed",)]
