"""Regression tests for atomic, SQLite-native disaster recovery.

These tests intentionally exercise FTS5, sqlite-vec, WAL frames, corrupt input,
and pre-existing restore targets.  A backup is only useful when it can be
restored without replaying virtual-table shadow DDL.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path
from unittest.mock import Mock

import pytest

from mnemosyne.dr import recovery
from mnemosyne import cli


SQLITE_HEADER = b"SQLite format 3\x00"


def _load_vec(conn: sqlite3.Connection) -> None:
    sqlite_vec = pytest.importorskip("sqlite_vec")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    _load_vec(conn)
    return conn


def _build_feature_db(path: Path, *, keep_wal_open: bool = False):
    """Create a small DB with ordinary, FTS5 and vec0 tables.

    When ``keep_wal_open`` is true, the last committed row exists only in WAL
    while the returned connection remains open.
    """
    conn = _open(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
    conn.execute("CREATE VIRTUAL TABLE items_fts USING fts5(content, content='items', content_rowid='id')")
    conn.execute("CREATE VIRTUAL TABLE vec_items USING vec0(embedding float[4] distance_metric=cosine)")
    conn.execute("CREATE TRIGGER items_ai AFTER INSERT ON items BEGIN INSERT INTO items_fts(rowid, content) VALUES (new.id, new.content); END")
    conn.execute("INSERT INTO items(content) VALUES ('pierwszy rekord')")
    conn.execute("INSERT INTO vec_items(rowid, embedding) VALUES (1, '[1, 0, 0, 0]')")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("INSERT INTO items(content) VALUES ('rekord tylko w wal')")
    conn.execute("INSERT INTO vec_items(rowid, embedding) VALUES (2, '[0, 1, 0, 0]')")
    conn.commit()
    if keep_wal_open:
        return conn
    conn.close()
    return None


def _decompress(backup: Path, target: Path) -> None:
    with gzip.open(backup, "rb") as src, target.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)


def _logical_state(path: Path) -> dict:
    conn = _open(path)
    try:
        return {
            "quick": conn.execute("PRAGMA quick_check").fetchone()[0],
            "integrity": conn.execute("PRAGMA integrity_check").fetchone()[0],
            "fk": conn.execute("PRAGMA foreign_key_check").fetchall(),
            "items": conn.execute("SELECT id, content FROM items ORDER BY id").fetchall(),
            "fts": conn.execute("SELECT rowid FROM items_fts WHERE items_fts MATCH 'rekord' ORDER BY rowid").fetchall(),
            "vec_count": conn.execute("SELECT COUNT(*) FROM vec_items").fetchone()[0],
        }
    finally:
        conn.close()


def test_backup_is_gzipped_binary_sqlite_and_preserves_wal_fts_vec(tmp_path):
    source = tmp_path / "source.db"
    writer = _build_feature_db(source, keep_wal_open=True)
    assert writer is not None
    try:
        result = recovery.create_backup(source, tmp_path / "backups")
    finally:
        writer.close()

    backup = Path(result["backup_path"])
    unpacked = tmp_path / "unpacked.db"
    _decompress(backup, unpacked)

    assert unpacked.read_bytes().startswith(SQLITE_HEADER)
    state = _logical_state(unpacked)
    assert state == {
        "quick": "ok",
        "integrity": "ok",
        "fk": [],
        "items": [(1, "pierwszy rekord"), (2, "rekord tylko w wal")],
        "fts": [(1,), (2,)],
        "vec_count": 2,
    }
    assert result["format"] == "sqlite-binary-gzip-v1"
    assert len(result["db_checksum"]) == 64
    assert hashlib.sha256(unpacked.read_bytes()).hexdigest() == result["db_checksum"]


def test_backup_restore_roundtrip_preserves_fts_vec_and_wal(tmp_path):
    source = tmp_path / "source.db"
    writer = _build_feature_db(source, keep_wal_open=True)
    assert writer is not None
    try:
        result = recovery.create_backup(source, tmp_path / "backups")
    finally:
        writer.close()

    restored = tmp_path / "restored" / "mnemosyne.db"
    outcome = recovery.restore_backup(Path(result["backup_path"]), restored)

    assert outcome["integrity_check"] is True
    assert _logical_state(restored) == _logical_state(source)


def test_backup_and_restore_preserve_paths_with_uri_delimiters(tmp_path):
    source = tmp_path / "source?#.db"
    _build_feature_db(source)

    backup = Path(recovery.create_backup(source, tmp_path / "backups")["backup_path"])
    restored = tmp_path / "restore?#" / "target?#.db"
    outcome = recovery.restore_backup(backup, restored)

    assert outcome["integrity_check"] is True
    assert _logical_state(restored) == _logical_state(source)


def test_legacy_sql_dump_backup_remains_discoverable(tmp_path, monkeypatch):
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()
    backup = backup_dir / "mnemosyne_backup_20260701_120000.db.gz"
    sql = "BEGIN TRANSACTION;\nCREATE TABLE marker(value TEXT);\nCOMMIT;\n"
    with gzip.open(backup, "wb") as handle:
        handle.write(sql.encode("utf-8"))
    metadata = {
        "timestamp": "20260701_120000",
        "original_size": len(sql),
        "backup_size": backup.stat().st_size,
        "backup_checksum": hashlib.sha256(backup.read_bytes()).hexdigest()[:16],
        "compressed": True,
    }
    backup.with_suffix(".gz.json").write_text(json.dumps(metadata))

    listed = recovery.list_backups(backup_dir)
    assert len(listed) == 1
    assert listed[0]["metadata"]["format"] == "legacy-sql-dump-gzip"

    database = tmp_path / "mnemosyne.db"
    sqlite3.connect(database).close()
    monkeypatch.setattr(
        recovery,
        "get_default_paths",
        lambda: (tmp_path, backup_dir, database),
    )
    assert recovery.health_check()["backups"]["total"] == 1


def test_corrupt_backup_never_replaces_existing_target(tmp_path):
    target = tmp_path / "target.db"
    conn = sqlite3.connect(target)
    conn.execute("CREATE TABLE marker(value TEXT NOT NULL)")
    conn.execute("INSERT INTO marker VALUES ('original')")
    conn.commit()
    conn.close()
    before = hashlib.sha256(target.read_bytes()).hexdigest()

    corrupt = tmp_path / "broken.db.gz"
    with gzip.GzipFile(filename=str(corrupt), mode="wb") as f:
        f.write(b"not a sqlite database")

    with pytest.raises((sqlite3.DatabaseError, ValueError, RuntimeError)):
        recovery.restore_backup(corrupt, target)

    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
    conn = sqlite3.connect(target)
    try:
        assert conn.execute("SELECT value FROM marker").fetchone()[0] == "original"
    finally:
        conn.close()


def test_emergency_backup_includes_existing_targets_wal(tmp_path):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    result = recovery.create_backup(source, tmp_path / "backups")

    target = tmp_path / "target.db"
    conn = sqlite3.connect(target)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("CREATE TABLE marker(value TEXT NOT NULL)")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("INSERT INTO marker VALUES ('committed only in wal')")
    conn.commit()
    try:
        recovery.restore_backup(Path(result["backup_path"]), target)
    finally:
        conn.close()

    emergency = target.with_suffix(".emergency_backup.db")
    assert emergency.exists()
    check = sqlite3.connect(emergency)
    try:
        assert check.execute("SELECT value FROM marker").fetchone()[0] == "committed only in wal"
    finally:
        check.close()


def test_restore_keeps_active_connections_on_the_restored_database(tmp_path):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    result = recovery.create_backup(source, tmp_path / "backups")

    target = tmp_path / "target.db"
    active = sqlite3.connect(target)
    active.execute("PRAGMA journal_mode=WAL")
    active.execute("CREATE TABLE old_marker(value TEXT)")
    active.commit()
    try:
        recovery.restore_backup(Path(result["backup_path"]), target)
        active.execute("CREATE TABLE post_restore(value TEXT)")
        active.execute("INSERT INTO post_restore VALUES ('visible')")
        active.commit()

        fresh = _open(target)
        try:
            assert fresh.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
            assert fresh.execute("SELECT value FROM post_restore").fetchone()[0] == "visible"
        finally:
            fresh.close()
    finally:
        active.close()


def test_verify_integrity_rejects_external_content_fts_mismatch(tmp_path):
    db = tmp_path / "fts-mismatch.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE items(id INTEGER PRIMARY KEY, content TEXT NOT NULL)")
    conn.execute(
        "CREATE VIRTUAL TABLE items_fts USING "
        "fts5(content, content='items', content_rowid='id')"
    )
    conn.executemany("INSERT INTO items(content) VALUES (?)", [("alpha",), ("beta",)])
    conn.execute("INSERT INTO items_fts(items_fts) VALUES ('rebuild')")
    conn.commit()
    conn.execute(
        "INSERT INTO items_fts(items_fts, rowid, content) "
        "VALUES ('delete', 2, 'beta')"
    )
    conn.commit()
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()

    assert recovery.verify_integrity(db) is False


def test_verify_integrity_rejects_orphaned_mnemosyne_vec_row(tmp_path):
    db = tmp_path / "vec-orphan.db"
    conn = _open(db)
    conn.execute("CREATE TABLE episodic_memory(id TEXT)")
    conn.execute("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding float[4])")
    conn.execute("INSERT INTO vec_episodes(rowid, embedding) VALUES (1, '[1,0,0,0]')")
    conn.commit()
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    conn.close()

    assert recovery.verify_integrity(db) is False


def test_two_backups_created_in_same_second_do_not_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    conn = sqlite3.connect(source)
    conn.execute("CREATE TABLE marker(value TEXT)")
    conn.commit()
    conn.close()

    class FrozenDateTime:
        @classmethod
        def now(cls):
            from datetime import datetime
            return datetime(2026, 7, 14, 12, 0, 0)

    monkeypatch.setattr(recovery, "datetime", FrozenDateTime)
    first = recovery.create_backup(source, tmp_path / "backups")
    second = recovery.create_backup(source, tmp_path / "backups")

    assert first["backup_path"] != second["backup_path"]
    assert Path(first["backup_path"]).exists()
    assert Path(second["backup_path"]).exists()


def test_metadata_matches_binary_snapshot_not_live_db_file(tmp_path):
    source = tmp_path / "source.db"
    writer = _build_feature_db(source, keep_wal_open=True)
    assert writer is not None
    try:
        result = recovery.create_backup(source, tmp_path / "backups")
    finally:
        writer.close()

    backup = Path(result["backup_path"])
    metadata = json.loads(Path(result["metadata_path"]).read_text())
    unpacked = tmp_path / "snapshot.db"
    _decompress(backup, unpacked)

    assert metadata["db_checksum"] == hashlib.sha256(unpacked.read_bytes()).hexdigest()
    assert metadata["backup_checksum"] == hashlib.sha256(backup.read_bytes()).hexdigest()
    assert metadata["format"] == "sqlite-binary-gzip-v1"


def test_restore_cli_accepts_explicit_target(tmp_path, monkeypatch, capsys):
    backup = tmp_path / "backup.db.gz"
    backup.write_bytes(b"placeholder")
    target = tmp_path / "isolated" / "mnemosyne.db"
    restore = Mock(return_value={
        "backup_used": str(backup),
        "database_path": str(target),
        "integrity_check": True,
    })
    monkeypatch.setattr(recovery, "restore_backup", restore)

    cli.cmd_restore([str(backup), "--target", str(target)])

    restore.assert_called_once_with(backup, target)
    assert str(target) in capsys.readouterr().out


def test_restore_transfer_failure_rolls_back_existing_target(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    result = recovery.create_backup(source, tmp_path / "backups")

    target = tmp_path / "target.db"
    conn = sqlite3.connect(target)
    conn.execute("CREATE TABLE marker(value TEXT)")
    conn.execute("INSERT INTO marker VALUES ('do not replace')")
    conn.commit()
    conn.close()
    real_restore = recovery._restore_snapshot
    target_attempts = 0

    def fail_first_target_restore(source_path, destination_path):
        nonlocal target_attempts
        if Path(destination_path) == target:
            target_attempts += 1
            if target_attempts == 1:
                raise OSError("simulated SQLite restore failure")
        return real_restore(Path(source_path), Path(destination_path))

    monkeypatch.setattr(recovery, "_restore_snapshot", fail_first_target_restore)
    with pytest.raises(OSError, match="simulated SQLite restore failure"):
        recovery.restore_backup(Path(result["backup_path"]), target)

    check = sqlite3.connect(target)
    try:
        assert check.execute("SELECT value FROM marker").fetchone()[0] == "do not replace"
    finally:
        check.close()
    assert target_attempts == 2


def test_failed_final_verification_rolls_back_existing_target(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    result = recovery.create_backup(source, tmp_path / "backups")

    target = tmp_path / "target.db"
    conn = sqlite3.connect(target)
    conn.execute("CREATE TABLE marker(value TEXT)")
    conn.execute("INSERT INTO marker VALUES ('rollback me')")
    conn.commit()
    conn.close()

    real_verify = recovery.verify_integrity
    target_checks = 0

    def fail_first_target_check(path=None):
        nonlocal target_checks
        assert path is not None
        if Path(path) == target:
            target_checks += 1
            if target_checks == 1:
                return False
        return real_verify(path)

    monkeypatch.setattr(recovery, "verify_integrity", fail_first_target_check)
    with pytest.raises(RuntimeError, match="final integrity verification"):
        recovery.restore_backup(Path(result["backup_path"]), target)

    check = sqlite3.connect(target)
    try:
        assert check.execute("SELECT value FROM marker").fetchone()[0] == "rollback me"
    finally:
        check.close()
    assert target_checks >= 2


def test_keyboard_interrupt_during_final_verification_rolls_back_existing_target(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    result = recovery.create_backup(source, tmp_path / "backups")
    target = tmp_path / "target.db"
    conn = sqlite3.connect(target)
    conn.execute("CREATE TABLE marker(value TEXT)")
    conn.execute("INSERT INTO marker VALUES ('original')")
    conn.commit()
    conn.close()
    real_verify = recovery.verify_integrity
    target_checks = 0

    def interrupt_first_target_check(path=None):
        nonlocal target_checks
        assert path is not None
        if Path(path) == target:
            target_checks += 1
            if target_checks == 1:
                raise KeyboardInterrupt("final verification interrupted")
        return real_verify(path)

    monkeypatch.setattr(recovery, "verify_integrity", interrupt_first_target_check)
    with pytest.raises(KeyboardInterrupt, match="final verification interrupted"):
        recovery.restore_backup(Path(result["backup_path"]), target)

    check = sqlite3.connect(target)
    try:
        assert check.execute("SELECT value FROM marker").fetchone()[0] == "original"
    finally:
        check.close()
    assert target_checks >= 2


def test_system_exit_during_final_verification_removes_new_target(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    result = recovery.create_backup(source, tmp_path / "backups")
    target = tmp_path / "new-target.db"
    real_verify = recovery.verify_integrity

    def interrupt_target_check(path=None):
        assert path is not None
        if Path(path) == target:
            raise SystemExit("final verification interrupted")
        return real_verify(path)

    monkeypatch.setattr(recovery, "verify_integrity", interrupt_target_check)
    with pytest.raises(SystemExit, match="final verification interrupted"):
        recovery.restore_backup(Path(result["backup_path"]), target)

    assert not target.exists()
    assert not Path(f"{target}-wal").exists()
    assert not Path(f"{target}-shm").exists()


def test_backup_metadata_survives_source_disappearing_after_archive_publication(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.db"
    conn = sqlite3.connect(source)
    conn.execute("CREATE TABLE marker(value TEXT)")
    conn.commit()
    conn.close()
    source_size = source.stat().st_size
    real_replace = recovery.os.replace

    def remove_source_after_archive_publish(src, dst):
        real_replace(src, dst)
        if Path(dst).suffix == ".gz":
            source.unlink()

    monkeypatch.setattr(recovery.os, "replace", remove_source_after_archive_publish)
    result = recovery.create_backup(source, tmp_path / "backups")

    assert result["source_file_size"] == source_size
    assert Path(result["backup_path"]).exists()
    assert Path(result["metadata_path"]).exists()


def test_metadata_interrupt_removes_published_backup_pair(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    conn = sqlite3.connect(source)
    conn.execute("CREATE TABLE marker(value TEXT)")
    conn.commit()
    conn.close()
    backup_dir = tmp_path / "backups"

    def interrupt(_path, _metadata):
        raise KeyboardInterrupt("metadata interrupted")

    monkeypatch.setattr(recovery, "_atomic_json", interrupt)
    with pytest.raises(KeyboardInterrupt, match="metadata interrupted"):
        recovery.create_backup(source, backup_dir)

    assert list(backup_dir.glob("*.db.gz")) == []
    assert list(backup_dir.glob("*.json")) == []


def test_archive_is_the_last_publication_marker(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    backup_dir = tmp_path / "backups"
    real_replace = recovery.os.replace
    observed_window = False

    def inspect_before_archive_publish(src, dst):
        nonlocal observed_window
        destination = Path(dst)
        if destination.name.endswith(".db.gz"):
            observed_window = True
            metadata = destination.with_suffix(".gz.json")
            assert metadata.exists()
            assert recovery.list_backups(backup_dir) == []
            with pytest.raises(FileNotFoundError):
                recovery.restore_backup(destination, tmp_path / "must-not-exist.db")
        real_replace(src, dst)

    monkeypatch.setattr(recovery.os, "replace", inspect_before_archive_publish)
    result = recovery.create_backup(source, backup_dir)

    assert observed_window is True
    assert len(recovery.list_backups(backup_dir)) == 1
    assert Path(result["backup_path"]).exists()
    assert Path(result["metadata_path"]).exists()


def test_binary_backup_without_metadata_is_hidden_and_restore_fails_closed(tmp_path):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    backup_dir = tmp_path / "backups"
    result = recovery.create_backup(source, backup_dir)
    Path(result["metadata_path"]).unlink()

    assert recovery.list_backups(backup_dir) == []
    with pytest.raises(ValueError, match="metadata"):
        recovery.restore_backup(Path(result["backup_path"]), tmp_path / "target.db")


def test_backup_with_invalid_checksum_is_hidden_and_restore_fails_closed(tmp_path):
    source = tmp_path / "source.db"
    _build_feature_db(source)
    backup_dir = tmp_path / "backups"
    result = recovery.create_backup(source, backup_dir)
    metadata_path = Path(result["metadata_path"])
    metadata = json.loads(metadata_path.read_text())
    metadata["backup_checksum"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))

    assert recovery.list_backups(backup_dir) == []
    with pytest.raises(ValueError, match="checksum"):
        recovery.restore_backup(Path(result["backup_path"]), tmp_path / "target.db")
