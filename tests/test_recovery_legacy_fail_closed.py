"""Tests for fail-closed legacy SQL dump replay in restore_backup."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from mnemosyne.dr import recovery


def _make_simple_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()


def _rewrite_backup_payload(backup_path: Path, payload: bytes) -> None:
    import gzip as _gzip

    meta_path = backup_path.with_suffix(".gz.json")
    compressed = _gzip.compress(payload)
    backup_path.write_bytes(compressed)
    meta = json.loads(meta_path.read_text())
    meta["backup_checksum"] = hashlib.sha256(compressed).hexdigest()[:16]
    meta["dump_checksum"] = hashlib.sha256(payload).hexdigest()
    meta_path.write_text(json.dumps(meta, indent=2))


def test_legacy_fts5_restore_skips_shadow_ddl(tmp_path, caplog):
    db_path = tmp_path / "src.db"
    backup_dir = tmp_path / "backups"
    restore_target = tmp_path / "fts_restored.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, v TEXT)")
    conn.execute("CREATE VIRTUAL TABLE items_fts USING fts5(v)")
    conn.execute("INSERT INTO items (id, v) VALUES (1, 'hello world')")
    conn.execute("INSERT INTO items_fts (rowid, v) VALUES (1, 'hello world')")
    conn.commit()
    conn.close()

    backup = recovery.create_backup(
        db_path=db_path, backup_dir=backup_dir, format="sql"
    )
    backup_path = Path(backup["backup_path"])

    # shadow DDL present in dump
    assert b"items_fts_config" in gzip.decompress(backup_path.read_bytes())

    import logging

    caplog.set_level(logging.WARNING)
    result = recovery.restore_backup(backup_path, restore_target)
    assert result["restored"] is True
    assert result["integrity_check"] is True

    target_conn = sqlite3.connect(str(restore_target))
    try:
        rows = target_conn.execute("SELECT id, v FROM items").fetchall()
        assert rows == [(1, "hello world")]
        match_rows = target_conn.execute(
            "SELECT rowid FROM items_fts WHERE items_fts MATCH 'hello'"
        ).fetchall()
        assert len(match_rows) == 1
        assert match_rows[0][0] == 1
    finally:
        target_conn.close()

    assert any("Skipping shadow-table DDL" in r.message for r in caplog.records)


def test_legacy_empty_payload_refused(tmp_path):
    db_path = tmp_path / "src.db"
    backup_dir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=backup_dir, format="sql")
    backup_path = Path(backup["backup_path"])

    target = tmp_path / "target.db"
    _make_simple_db(target)
    target_bytes_before = target.read_bytes()

    _rewrite_backup_payload(backup_path, b"")

    with pytest.raises(RuntimeError, match="Refusing to restore"):
        recovery.restore_backup(backup_path, target)

    assert target.read_bytes() == target_bytes_before


def test_legacy_comments_only_payload_refused(tmp_path):
    db_path = tmp_path / "src.db"
    backup_dir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=backup_dir, format="sql")
    backup_path = Path(backup["backup_path"])

    target = tmp_path / "target.db"
    _make_simple_db(target)
    target_bytes_before = target.read_bytes()

    _rewrite_backup_payload(backup_path, b"-- legacy dump\n-- comments only\n")

    with pytest.raises(RuntimeError, match="Refusing to restore"):
        recovery.restore_backup(backup_path, target)

    assert target.read_bytes() == target_bytes_before
