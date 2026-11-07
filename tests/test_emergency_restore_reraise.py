"""Tests for emergency_restore re-raising restore refusals."""

from __future__ import annotations

import json
import gzip
import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from mnemosyne.dr import recovery


def _make_simple_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()


def test_emergency_restore_reraise_on_refusal(tmp_path):
    db_path = tmp_path / "src.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sqlite3")
    backup_path = Path(backup["backup_path"])
    meta_path = backup_path.with_suffix(".gz.json")

    # delete backup_checksum from metadata sidecar
    meta = json.loads(meta_path.read_text())
    meta.pop("backup_checksum", None)
    meta_path.write_text(json.dumps(meta, indent=2))

    target = tmp_path / "target.db"
    _make_simple_db(target)
    db_path_bytes_before = db_path.read_bytes()

    with pytest.raises(RuntimeError) as excinfo:
        recovery.emergency_restore(backup_dir=bdir, db_path=db_path)

    msg = str(excinfo.value)
    assert "refusing to restore" in msg.lower()
    assert "all backups failed integrity check" not in msg.lower()

    # The actual restoration target is db_path — the refusal must not touch it.
    assert db_path.read_bytes() == db_path_bytes_before


def test_emergency_restore_skips_corrupt_backup_restores_valid(tmp_path):
    db_path = tmp_path / "src.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)

    # older backup B first
    backup_b = recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sqlite3")
    # newer backup A
    backup_a = recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sqlite3")
    backup_a_path = Path(backup_a["backup_path"])

    # overwrite A's .gz with non-gzip bytes (keep sidecar)
    backup_a_path.write_bytes(b"this is not gzip data at all")

    result = recovery.emergency_restore(backup_dir=bdir, db_path=db_path)
    assert result["restored"] is True
    assert result["backup_used"] == backup_b["backup_path"]


def test_emergency_restore_valid_backup_restored(tmp_path):
    db_path = tmp_path / "src.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sqlite3")

    result = recovery.emergency_restore(backup_dir=bdir, db_path=db_path)
    assert result["restored"] is True


def _incomplete_snapshot(db_path, backup_dir):
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("CREATE VIRTUAL TABLE ft USING fts5(body)")
        conn.execute("INSERT INTO ft VALUES ('orchid')")
    backup = Path(recovery.create_backup(db_path, backup_dir, format="sql")["backup_path"])
    text = gzip.decompress(backup.read_bytes()).decode()
    lines = text.splitlines(keepends=True)
    removed = [line for line in lines if line.startswith("CREATE TABLE 'ft_idx'")]
    assert len(removed) == 1
    payload = "".join(line for line in lines if line not in removed).encode()
    compressed = gzip.compress(payload)
    backup.write_bytes(compressed)
    path = backup.with_suffix(".gz.json")
    meta = json.loads(path.read_text())
    meta.update(backup_checksum=hashlib.sha256(compressed).hexdigest()[:16],
                dump_checksum=hashlib.sha256(payload).hexdigest())
    path.write_text(json.dumps(meta))
    return backup


def test_emergency_restore_keeps_snapshot_refusal_and_cause(tmp_path):
    db_path = tmp_path / "source.db"
    _make_simple_db(db_path)
    backup = _incomplete_snapshot(db_path, tmp_path / "backups")
    before = db_path.read_bytes()
    with pytest.raises(RuntimeError, match="Incomplete virtual-table snapshot") as caught:
        recovery.emergency_restore(backup.parent, db_path)
    assert isinstance(caught.value.__cause__, recovery._RestoreRefusedError)
    assert "Incomplete virtual-table snapshot" in str(caught.value.__cause__)
    assert db_path.read_bytes() == before


def test_emergency_restore_skips_snapshot_refusal_and_restores_older_rows(tmp_path):
    db_path = tmp_path / "source.db"
    _make_simple_db(db_path)
    older = recovery.create_backup(db_path, tmp_path / "backups")
    _incomplete_snapshot(db_path, tmp_path / "backups")
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("UPDATE t SET v = 'changed'")
    result = recovery.emergency_restore(tmp_path / "backups", db_path)
    assert result["backup_used"] == older["backup_path"]
    assert result["attempts"] == 2
    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT id, v FROM t ORDER BY id").fetchall() == [(1, "a"), (2, "b")]


@pytest.mark.parametrize("newest_kind", ["file", "payload"])
def test_emergency_restore_reports_newest_of_two_refusals(tmp_path, newest_kind):
    db = tmp_path / "source.db"
    _make_simple_db(db)
    directory = tmp_path / "backups"
    older = Path(recovery.create_backup(db, directory)["backup_path"])
    newer = Path(recovery.create_backup(db, directory)["backup_path"])
    assert newer.name > older.name
    for backup, kind in ((older, "payload" if newest_kind == "file" else "file"),
                         (newer, newest_kind)):
        path = backup.with_suffix(".gz.json")
        metadata = json.loads(path.read_text())
        metadata["backup_checksum" if kind == "file" else "dump_checksum"] = "0" * 64
        path.write_text(json.dumps(metadata))
    before = db.read_bytes()
    with pytest.raises(recovery._RestoreRefusedError) as caught:
        recovery.emergency_restore(directory, db)
    expected = f"Backup {newest_kind} checksum mismatch"
    other = "payload" if newest_kind == "file" else "file"
    assert "after trying 2 backup(s)" in str(caught.value)
    assert f"attempt 1, {newer.name}" in str(caught.value)
    assert expected in str(caught.value)
    cause = caught.value.__cause__
    assert isinstance(cause, recovery._RestoreRefusedError)
    assert expected in str(cause)
    assert f"Backup {other} checksum mismatch" not in str(cause)
    assert db.read_bytes() == before


def test_emergency_restore_logs_and_chains_non_refusal_failures(tmp_path, caplog):
    db_path = tmp_path / "source.db"
    _make_simple_db(db_path)
    backup = Path(recovery.create_backup(db_path, tmp_path / "backups")["backup_path"])
    backup.write_bytes(b"not gzip")
    with pytest.raises(RuntimeError, match="All backups failed integrity check") as caught:
        recovery.emergency_restore(backup.parent, db_path)
    assert isinstance(caught.value.__cause__, OSError)
    assert any(backup.name in record.message and record.exc_info for record in caplog.records)
