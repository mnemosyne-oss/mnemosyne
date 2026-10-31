"""Tests for emergency_restore re-raising restore refusals."""

from __future__ import annotations

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
