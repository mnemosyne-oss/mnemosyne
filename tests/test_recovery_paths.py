"""Tests for recovery.get_default_paths() honoring the same path configuration
as the live store (mnemosyne.core.beam).

The disaster-recovery helpers (backup/restore, and `mnemosyne reindex`'s
auto-backup) must resolve the database to the same location the store actually
uses. Previously they hardcoded ``~/.mnemosyne/data`` and ignored
MNEMOSYNE_DATA_DIR / HERMES_HOME, so they operated on (or failed to find) the
wrong database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.dr import recovery


def test_get_default_paths_honors_data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MNEMOSYNE_BACKUP_DIR", raising=False)
    data_dir, backup_dir, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "data"
    assert db_path == tmp_path / "data" / "mnemosyne.db"
    # backups land alongside the data dir, not under ~/.mnemosyne
    assert backup_dir == tmp_path / "backups"


def test_get_default_paths_honors_hermes_home(monkeypatch, tmp_path):
    monkeypatch.delenv("MNEMOSYNE_DATA_DIR", raising=False)
    monkeypatch.delenv("MNEMOSYNE_BACKUP_DIR", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    data_dir, backup_dir, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "home" / "mnemosyne" / "data"
    assert db_path == data_dir / "mnemosyne.db"
    assert backup_dir == tmp_path / "home" / "mnemosyne" / "backups"


def test_get_default_paths_backup_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_BACKUP_DIR", str(tmp_path / "custom_backups"))
    _, backup_dir, _ = recovery.get_default_paths()
    assert backup_dir == tmp_path / "custom_backups"


def test_get_default_paths_data_dir_takes_precedence_over_hermes_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "explicit"))
    data_dir, _, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "explicit"
    assert db_path == tmp_path / "explicit" / "mnemosyne.db"


def test_create_backup_succeeds_with_sqlite_vec_tables(tmp_path):
    """The binary snapshot must remain queryable with sqlite-vec loaded.

    A SQL dump is not a valid recovery format for vec0/FTS shadow tables;
    decompress the native SQLite file and exercise the virtual table instead.
    """
    pytest.importorskip("sqlite_vec")

    db_path = tmp_path / "vec_test.db"
    backup_dir = tmp_path / "backups"

    # Build a tiny DB that has a vec0 virtual table — the exact schema
    # shape that triggered the original bug in 3.10.x.
    conn = sqlite3.connect(str(db_path))
    conn.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(conn)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_items USING vec0("
        "embedding float[4] distance_metric=cosine)"
    )
    conn.execute("CREATE TABLE meta (k TEXT PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO meta VALUES (?, ?)", [("a", "1"), ("b", "2")])
    conn.commit()
    conn.close()

    # Act: this is the call path `mnemosyne backup` uses. Pre-fix it
    # raised sqlite3.OperationalError: no such module: vec0.
    result = recovery.create_backup(db_path=db_path, backup_dir=backup_dir)

    # Assert: backup file exists, is non-empty and decompresses to a native
    # SQLite database whose vec0 table can be queried.
    assert Path(result["backup_path"]).exists()
    assert result["backup_size"] > 0
    import gzip
    restored = tmp_path / "snapshot.db"
    with gzip.open(result["backup_path"], "rb") as source, restored.open("wb") as target:
        target.write(source.read())
    assert restored.read_bytes().startswith(b"SQLite format 3\x00")
    check = sqlite3.connect(str(restored))
    check.enable_load_extension(True)
    sqlite_vec.load(check)
    try:
        assert check.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0] == 0
        assert check.execute("SELECT COUNT(*) FROM meta").fetchone()[0] == 2
    finally:
        check.close()
