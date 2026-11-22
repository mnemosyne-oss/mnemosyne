"""Tests for recovery.get_default_paths() honoring the same path configuration
as the live store (mnemosyne.core.beam).

The disaster-recovery helpers (backup/restore, and `mnemosyne reindex`'s
auto-backup) must resolve the database to the same location the store actually
uses. Previously they hardcoded ``~/.mnemosyne/data`` and ignored
MNEMOSYNE_DATA_DIR / HERMES_HOME, so they operated on (or failed to find) the
wrong database.
"""

from __future__ import annotations

import json
import os
import sqlite3
import hashlib
import sys
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


def test_get_default_paths_data_dir_takes_precedence_over_hermes_home(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "explicit"))
    data_dir, _, db_path = recovery.get_default_paths()
    assert data_dir == tmp_path / "explicit"
    assert db_path == tmp_path / "explicit" / "mnemosyne.db"


def test_create_backup_succeeds_with_sqlite_vec_tables(tmp_path):
    """Regression: create_backup() must load sqlite-vec on the source AND
    destination connections, otherwise sqlite3.Connection.backup() and
    Connection.iterdump() both fail with ``no such module: vec0`` on
    databases that use vec0 virtual tables.

    Pre-fix: this test fails with ``sqlite3.OperationalError: no such
    module: vec0`` raised from inside the backup serialization path.
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
    conn.execute(
        "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
        (1, json.dumps([1.0, 0.0, 0.0, 0.0])),
    )
    conn.execute(
        "INSERT INTO vec_items (rowid, embedding) VALUES (?, ?)",
        (2, json.dumps([0.0, 1.0, 0.0, 0.0])),
    )
    conn.commit()
    conn.close()

    # Act: this is the call path `mnemosyne backup` uses. Pre-fix it
    # raised sqlite3.OperationalError: no such module: vec0.
    result = recovery.create_backup(db_path=db_path, backup_dir=backup_dir)

    # Assert: backup file exists, is non-empty, gzipped, and is a valid
    # binary SQLite database preserving the vec0 table definition (#640).
    assert Path(result["backup_path"]).exists()
    assert result["backup_size"] > 0
    assert result["format"] == "sqlite3"
    import gzip

    decompressed = gzip.decompress(Path(result["backup_path"]).read_bytes())
    assert decompressed[:16] == b"SQLite format 3\x00"

    # Roundtrip restore into a target database
    target = tmp_path / "vec_restored.db"
    res = recovery.restore_backup(Path(result["backup_path"]), target)
    assert res["restored"] is True
    assert res["integrity_check"] is True

    target_conn = sqlite3.connect(str(target))
    try:
        target_conn.enable_load_extension(True)
        sqlite_vec.load(target_conn)
        target_conn.enable_load_extension(False)

        master_rows = target_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_items'"
        ).fetchall()
        assert len(master_rows) == 1

        rows = target_conn.execute("SELECT k, v FROM meta ORDER BY k").fetchall()
        assert rows == [("a", "1"), ("b", "2")]

        knn = target_conn.execute(
            "SELECT rowid, distance FROM vec_items WHERE embedding MATCH ? AND k = 1",
            (json.dumps([0.9, 0.1, 0.0, 0.0]),),
        ).fetchall()
        assert len(knn) == 1
        assert knn[0][0] == 1
    finally:
        target_conn.close()


# ---------------------------------------------------------------------------
# Task 1 / Wave 1 P0: backup unique filename + fail-closed restore
# ---------------------------------------------------------------------------

import gzip as _gzip


def _make_simple_db(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()


def test_two_rapid_backups_get_unique_filenames(tmp_path):
    """Two backups created within the same second must not overwrite each
    other."""
    db_path = tmp_path / "src.db"
    _make_simple_db(db_path)
    bdir = tmp_path / "backups"

    r1 = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    r2 = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    assert Path(r1["backup_path"]).name != Path(r2["backup_path"]).name, (
        "two backups in the same second collided on filename"
    )
    assert Path(r1["backup_path"]).exists()
    assert Path(r2["backup_path"]).exists()


def test_restore_rejects_active_wal_sidecar(tmp_path):
    """A stale -wal sidecar means uncommitted frames would be silently
    dropped by a main-file replace; restore must refuse."""
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    (db_path.parent / (db_path.name + "-wal")).write_bytes(b"\x00" * 64)
    with pytest.raises(RuntimeError, match="sidecar"):
        recovery.restore_backup(Path(backup["backup_path"]), db_path)


def test_restore_rejects_active_shm_sidecar(tmp_path):
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    (db_path.parent / (db_path.name + "-shm")).write_bytes(b"\x00" * 64)
    with pytest.raises(RuntimeError, match="sidecar"):
        recovery.restore_backup(Path(backup["backup_path"]), db_path)


def test_restore_payload_checksum_mismatch_rejected_and_target_preserved(tmp_path):
    """Corruption that still decompresses as gzip but changes the dump payload
    must be caught by the payload checksum and rejected; the target is
    preserved."""
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    backup_path = Path(backup["backup_path"])

    raw = _gzip.decompress(backup_path.read_bytes())
    corrupted = raw.replace(b"VALUES", b"VALOOS")
    if corrupted == raw:
        corrupted = raw.replace(b"CREATE", b"CREAT")
    backup_path.write_bytes(_gzip.compress(corrupted))

    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO t VALUES (99, 'preserved')")
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="checksum"):
        recovery.restore_backup(backup_path, db_path)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT v FROM t WHERE id = 99").fetchone()
    conn.close()
    assert row is not None and row[0] == "preserved", (
        "target was corrupted by a failed restore"
    )


def test_restore_failed_integrity_preserves_original(tmp_path):
    """A dump that rebuilds but fails integrity_check must not replace the
    target."""
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    backup_path = Path(backup["backup_path"])

    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO t VALUES (99, 'keepme')")
    conn.commit()
    conn.close()

    # Break the dump so it produces an invalid DB but still parses as SQL.
    raw = _gzip.decompress(backup_path.read_bytes())
    backup_path.write_bytes(
        _gzip.compress(raw.replace(b"CREATE TABLE", b"BREAK TABLE"))
    )

    with pytest.raises(Exception):
        recovery.restore_backup(backup_path, db_path)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT v FROM t WHERE id = 99").fetchone()
    conn.close()
    assert row is not None and row[0] == "keepme"


def test_successful_restore_replaces_target_and_preserves_original(tmp_path):
    db_path = tmp_path / "target.db"
    bdir = tmp_path / "backups"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    # Mutate the live target after the backup.
    conn = sqlite3.connect(str(db_path))
    conn.execute("INSERT INTO t VALUES (42, 'post-backup')")
    conn.commit()
    conn.close()

    result = recovery.restore_backup(Path(backup["backup_path"]), db_path)

    assert result["integrity_check"] is True
    conn = sqlite3.connect(str(db_path))
    total = conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    post_backup_gone = conn.execute("SELECT COUNT(*) FROM t WHERE id = 42").fetchone()[
        0
    ]
    conn.close()
    assert total == 2, "target not restored to backup contents"
    assert post_backup_gone == 0
    preserved = Path(result["preserved_original"])
    assert preserved.exists(), "original target must be preserved as a sidecar"


# ---------------------------------------------------------------------------
# Task 1 fix round 1: fail-closed metadata, post-replace integrity, held
# exclusive writer lock, atomic filename allocation, unique staged files,
# parent-dir fsync, staged cleanup on failure.
# ---------------------------------------------------------------------------

import threading as _threading


def _make_db_simple(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "a"), (2, "b")])
    conn.commit()
    conn.close()


class TestRestoreMetadataFailClosed:
    def test_missing_metadata_sidecar_rejected_and_target_preserved(self, tmp_path):
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
        backup_path = Path(backup["backup_path"])
        backup_path.with_suffix(".gz.json").unlink()

        target = tmp_path / "target.db"
        _make_db_simple(target)
        conn = sqlite3.connect(str(target))
        conn.execute("INSERT INTO t VALUES (99, 'keep')")
        conn.commit()
        conn.close()

        with pytest.raises(RuntimeError, match="metadata"):
            recovery.restore_backup(backup_path, target)

        conn = sqlite3.connect(str(target))
        row = conn.execute("SELECT v FROM t WHERE id = 99").fetchone()
        conn.close()
        assert row and row[0] == "keep"

    def test_malformed_metadata_rejected(self, tmp_path):
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
        backup_path = Path(backup["backup_path"])
        backup_path.with_suffix(".gz.json").write_text("{not valid json")

        target = tmp_path / "target.db"
        _make_db_simple(target)
        with pytest.raises(RuntimeError, match="metadata"):
            recovery.restore_backup(backup_path, target)

    def test_metadata_without_backup_checksum_rejected(self, tmp_path):
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
        backup_path = Path(backup["backup_path"])
        meta_file = backup_path.with_suffix(".gz.json")
        meta = json.loads(meta_file.read_text())
        meta.pop("backup_checksum")
        meta_file.write_text(json.dumps(meta))

        target = tmp_path / "target.db"
        _make_db_simple(target)
        with pytest.raises(RuntimeError, match="checksum"):
            recovery.restore_backup(backup_path, target)


class TestRestorePostReplaceIntegrityFailure:
    def test_failed_post_replace_check_restores_original_and_raises(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

        target = tmp_path / "target.db"
        _make_db_simple(target)
        conn = sqlite3.connect(str(target))
        conn.execute("INSERT INTO t VALUES (77, 'preserved-me')")
        conn.commit()
        conn.close()

        # Force the post-replace integrity check to fail.
        monkeypatch.setattr(recovery, "verify_integrity", lambda p: False)

        with pytest.raises(RuntimeError, match="integrity"):
            recovery.restore_backup(Path(backup["backup_path"]), target)

        conn = sqlite3.connect(str(target))
        row = conn.execute("SELECT v FROM t WHERE id = 77").fetchone()
        ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
        assert row and row[0] == "preserved-me"
        assert ok == "ok", "original was not restored after post-replace failure"


class TestExclusiveWriterLockHeld:
    def test_competing_writer_cannot_enter_staging_window(self, tmp_path, monkeypatch):
        """A second connection must be unable to BEGIN IMMEDIATE while a
        restore holds the writer lock across staging and os.replace."""
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

        target = tmp_path / "target.db"
        _make_db_simple(target)

        events = {"locked": False, "competitor_entered": False}
        ready = _threading.Event()
        original_replace = os.replace

        def slow_replace(staged, dest):
            if str(dest) == str(target):
                events["locked"] = True
                ready.set()
                import time

                time.sleep(0.3)
            return original_replace(staged, dest)

        monkeypatch.setattr("mnemosyne.dr.recovery.os.replace", slow_replace)

        def compete():
            ready.wait(timeout=2)
            try:
                c = sqlite3.connect(str(target), timeout=0.1)
                c.execute("BEGIN IMMEDIATE")
                events["competitor_entered"] = True
                c.execute("ROLLBACK")
                c.close()
            except sqlite3.OperationalError:
                pass

        thread = _threading.Thread(target=compete)
        thread.start()
        recovery.restore_backup(Path(backup["backup_path"]), target)
        thread.join(timeout=3)

        assert events["locked"] is True
        assert events["competitor_entered"] is False, (
            "a competing writer entered the staging window despite the lock"
        )


class TestBackupAndStagingRaces:
    def test_backup_filename_allocation_is_exclusive_under_frozen_clock(
        self, tmp_path, monkeypatch
    ):
        """With the timestamp frozen, four concurrent backups must still get
        distinct files via exclusive (O_CREAT|O_EXCL) allocation."""
        db_path = tmp_path / "src.db"
        _make_db_simple(db_path)
        bdir = tmp_path / "bk"

        from datetime import datetime as _dt

        frozen = _dt(2026, 1, 1, 0, 0, 0)
        fake_datetime = type(
            "D",
            (),
            {
                "now": staticmethod(lambda: frozen),
                "max": _dt.max,
                "strftime": _dt.strftime,
                "fromisoformat": _dt.fromisoformat,
            },
        )
        monkeypatch.setattr(recovery, "datetime", fake_datetime)

        results = []
        errors = []

        def make():
            try:
                results.append(recovery.create_backup(db_path=db_path, backup_dir=bdir))
            except Exception as exc:
                errors.append(exc)

        threads = [_threading.Thread(target=make) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        names = [Path(r["backup_path"]).name for r in results]
        assert len(names) == len(set(names)), f"filenames collided: {names}"
        for name in names:
            assert (bdir / name).stat().st_size > 0
        assert not errors

    def test_staged_filename_unique_per_invocation_same_target(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
        target = tmp_path / "same.db"
        _make_db_simple(target)

        staged_names = []
        original_connect = sqlite3.connect

        def spy_connect(connectable, *args, **kwargs):
            conn = original_connect(connectable, *args, **kwargs)
            if isinstance(connectable, str) and "restore_staged" in connectable:
                staged_names.append(Path(connectable).name)
            return conn

        monkeypatch.setattr("mnemosyne.dr.recovery.sqlite3.connect", spy_connect)

        recovery.restore_backup(Path(backup["backup_path"]), target)
        recovery.restore_backup(Path(backup["backup_path"]), target)

        assert len(staged_names) == 2
        assert staged_names[0] != staged_names[1], (
            f"staged filenames collided on same target: {staged_names}"
        )

    def test_parent_dir_fsynced_after_replace(self, tmp_path, monkeypatch):
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
        target = tmp_path / "target.db"
        _make_db_simple(target)

        fsynced_dirs = []
        real_open = os.open

        def spy_open(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if (flags & os.O_DIRECTORY) and str(target.parent) == str(path):
                fsynced_dirs.append(str(path))
            return fd

        monkeypatch.setattr("mnemosyne.dr.recovery.os.open", spy_open)

        recovery.restore_backup(Path(backup["backup_path"]), target)

        assert fsynced_dirs, "parent directory was not fsynced after replace"

    def test_backup_staging_files_not_visible_to_rotation_race(
        self, tmp_path, monkeypatch
    ):
        """rotate_backups runs while create_backup is writing staged files.
        Because in-progress backup artifacts use private staging names,
        rotation must not delete or disrupt the in-flight backup."""
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)

        prior_backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
        assert Path(prior_backup["backup_path"]).exists()

        rotation_ran = _threading.Event()
        real_flush_and_fsync = recovery._flush_and_fsync

        def slow_fsync(handle):
            if ".staging-" in str(handle.name) and not rotation_ran.is_set():
                rotation_ran.set()
                recovery.rotate_backups(bdir, keep=0)
            return real_flush_and_fsync(handle)

        monkeypatch.setattr(recovery, "_flush_and_fsync", slow_fsync)

        new_backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

        assert rotation_ran.is_set()
        assert not Path(prior_backup["backup_path"]).exists()
        assert Path(new_backup["backup_path"]).exists()
        assert Path(new_backup["backup_path"]).stat().st_size > 0
        assert Path(new_backup["metadata_path"]).exists()

    def test_staged_file_cleaned_up_on_failure(self, tmp_path):
        db_path = tmp_path / "src.db"
        bdir = tmp_path / "bk"
        _make_db_simple(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
        backup_path = Path(backup["backup_path"])
        target = tmp_path / "target.db"
        _make_db_simple(target)

        raw = _gzip.decompress(backup_path.read_bytes())
        backup_path.write_bytes(
            _gzip.compress(raw.replace(b"CREATE TABLE", b"BREAK TABLE"))
        )

        with pytest.raises(Exception):
            recovery.restore_backup(backup_path, target)

        leftovers = list(target.parent.glob("*restore_staged*"))
        assert leftovers == [], f"staged file not cleaned up: {leftovers}"


# ---------------------------------------------------------------------------
# #1160: on Windows os.fsync maps to FlushFileBuffers, which needs a handle
# with write access, and os.O_DIRECTORY does not exist. Backup staging files
# are fsynced through the handles that wrote them, and a missing O_DIRECTORY
# skips the directory fsync instead of raising after publication.
# ---------------------------------------------------------------------------

import errno as _errno


class TestRestoreWritableFsync:
    """Portable descriptor checks, not a native Windows restore acceptance run."""

    @pytest.fixture
    def backup_target(self, tmp_path):
        # Valid legacy SQL isolates handle lifecycle from vec0 serialization (#640).
        backup = tmp_path / "legacy.db.gz"
        backup.write_bytes(_gzip.compress(
            b"CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);"
            b"INSERT INTO t VALUES (1, 'backup');"
        ))
        backup.with_suffix(".gz.json").write_text(json.dumps({
            "backup_checksum": hashlib.sha256(backup.read_bytes()).hexdigest()[:16],
        }))
        target = tmp_path / "target.db"
        _make_db_simple(target)
        return backup, target

    def test_helper_writable_nontruncating(self, tmp_path, monkeypatch):
        path = tmp_path / "existing.db"
        original = b"existing database bytes\x00\xff"
        path.write_bytes(original)
        mode = path.stat().st_mode
        real_fsync = os.fsync
        synced = []

        def check(fd):
            assert os.write(fd, b"") == 0  # real OS rejects read-only fds
            assert path.read_bytes() == original
            synced.append(fd)
            real_fsync(fd)

        monkeypatch.setattr(recovery.os, "fsync", check)
        recovery._fsync_path(path)
        assert len(synced) == 1
        with pytest.raises(OSError) as exc:
            os.fstat(synced[0])
        assert exc.value.errno == _errno.EBADF
        assert path.read_bytes() == original
        assert path.stat().st_mode == mode
        with pytest.raises(FileNotFoundError):
            recovery._fsync_path(tmp_path / "missing.db")
        assert not (tmp_path / "missing.db").exists()

    @pytest.mark.parametrize("outcome", ["success", "fsync_failure", "rollback"])
    def test_public_restore_handle_lifecycle(self, backup_target, monkeypatch, outcome):
        import stat

        backup, target = backup_target
        original = target.read_bytes()
        real_fsync, real_replace = os.fsync, os.replace
        synced, published = [], []

        def assert_closed(fd):
            with pytest.raises(OSError) as exc:
                os.fstat(fd)
            assert exc.value.errno == _errno.EBADF

        def check(fd):
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                return real_fsync(fd)
            # Let staging pass on base for rollback: RED proves the second caller.
            if outcome != "rollback" or synced:
                assert os.write(fd, b"") == 0
            synced.append((fd, info.st_ino))
            if outcome == "fsync_failure":
                raise OSError(_errno.EIO, "injected restore fsync failure")
            return real_fsync(fd)

        def replace(src, dst):
            assert len(synced) == 1
            assert_closed(synced[0][0])
            assert Path(src).stat().st_ino == synced[0][1]
            published.append(str(dst))
            return real_replace(src, dst)

        monkeypatch.setattr(recovery.os, "fsync", check)
        monkeypatch.setattr(recovery.os, "replace", replace)
        if outcome == "rollback":
            monkeypatch.setattr(recovery, "verify_integrity", lambda p: False)
        if outcome == "success":
            assert recovery.restore_backup(backup, target)["restored"] is True
            conn = sqlite3.connect(str(target))
            assert conn.execute("SELECT v FROM t").fetchall() == [("backup",)]
            conn.close()
        else:
            error, message = ((OSError, "injected restore fsync failure")
                              if outcome == "fsync_failure" else
                              (RuntimeError, "Post-replace integrity_check failed"))
            with pytest.raises(error, match=message):
                recovery.restore_backup(backup, target)
            assert target.read_bytes() == original
        assert len(synced) == (2 if outcome == "rollback" else 1)
        for fd, _ in synced:
            assert_closed(fd)
        assert published == ([] if outcome == "fsync_failure" else [str(target)])
        if outcome == "rollback":
            assert synced[1][1] == target.stat().st_ino
            assert target.with_name(target.name + ".restore_preserved").read_bytes() == original
        assert list(target.parent.glob("*restore_staged*")) == []
        conn = sqlite3.connect(str(target), timeout=0)
        conn.execute("BEGIN IMMEDIATE")  # cleanup released the writer lock
        conn.rollback()
        conn.close()


def _require_write_access_for_fsync(monkeypatch):
    """Make os.fsync refuse descriptors opened without write access, as
    FlushFileBuffers does on Windows. Returns the list of synced inodes."""
    fcntl = pytest.importorskip("fcntl")
    real_fsync = os.fsync
    synced_inodes = []

    def fsync(fd):
        if fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE == os.O_RDONLY:
            raise OSError(_errno.EBADF, os.strerror(_errno.EBADF))
        synced_inodes.append(os.fstat(fd).st_ino)
        return real_fsync(fd)

    monkeypatch.setattr(recovery.os, "fsync", fsync)
    return synced_inodes


def test_backup_fsyncs_staging_files_through_writable_handles(tmp_path, monkeypatch):
    db_path = tmp_path / "src.db"
    _make_db_simple(db_path)
    synced_inodes = _require_write_access_for_fsync(monkeypatch)

    backup = recovery.create_backup(db_path=db_path, backup_dir=tmp_path / "bk")

    backup_path = Path(backup["backup_path"])
    meta_path = Path(backup["metadata_path"])
    # os.replace keeps the inode, so the published files are the synced ones.
    assert backup_path.stat().st_ino in synced_inodes
    assert meta_path.stat().st_ino in synced_inodes
    assert recovery._file_sha256(backup_path)[:16] == backup["backup_checksum"]
    dump = _gzip.decompress(backup_path.read_bytes())
    assert hashlib.sha256(dump).hexdigest() == backup["dump_checksum"]
    metadata = {
        key: value
        for key, value in backup.items()
        if key not in ("backup_path", "metadata_path")
    }
    assert json.loads(meta_path.read_text()) == metadata


def test_backup_staging_fsync_failure_publishes_nothing(tmp_path, monkeypatch):
    db_path = tmp_path / "src.db"
    bdir = tmp_path / "bk"
    _make_db_simple(db_path)

    def failing_fsync(fd):
        raise OSError(_errno.EIO, "simulated staging fsync failure")

    monkeypatch.setattr(recovery.os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="simulated staging fsync failure"):
        recovery.create_backup(db_path=db_path, backup_dir=bdir)

    assert list(bdir.iterdir()) == []


def test_backup_without_o_directory_returns_published_backup(tmp_path, monkeypatch):
    db_path = tmp_path / "src.db"
    _make_db_simple(db_path)
    monkeypatch.delattr(recovery.os, "O_DIRECTORY", raising=False)

    backup = recovery.create_backup(db_path=db_path, backup_dir=tmp_path / "bk")

    assert Path(backup["backup_path"]).exists()
    assert Path(backup["metadata_path"]).exists()
    assert sorted(p.name for p in (tmp_path / "bk").iterdir()) == sorted(
        [Path(backup["backup_path"]).name, Path(backup["metadata_path"]).name]
    )


def test_restore_without_o_directory_reports_success(tmp_path, monkeypatch):
    db_path = tmp_path / "src.db"
    _make_db_simple(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=tmp_path / "bk")
    target = tmp_path / "target.db"
    _make_db_simple(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (99, 'replaced')")
    conn.commit()
    conn.close()
    monkeypatch.delattr(recovery.os, "O_DIRECTORY", raising=False)

    result = recovery.restore_backup(Path(backup["backup_path"]), target)

    assert result["restored"] is True
    conn = sqlite3.connect(str(target))
    rows = conn.execute("SELECT id, v FROM t ORDER BY id").fetchall()
    conn.close()
    assert rows == [(1, "a"), (2, "b")]


# ---------------------------------------------------------------------------
# Security I-1: restore must not execute untrusted dump SQL with SQLite
# extension loading left enabled. The optional sqlite-vec loader calls
# enable_load_extension(True); it must be disabled again before any
# caller runs executescript on backup-provided SQL, and it must stay
# disabled even when the loader raised ImportError/OperationalError.
#
# These tests stub the sqlite_vec module so the leak is exercised
# deterministically regardless of whether real sqlite_vec is installed
# (the bug is latent when sqlite_vec is absent because the ImportError
# fires before enable_load_extension runs).
# ---------------------------------------------------------------------------


def _install_sqlite_vec_stub(monkeypatch, load_side_effect=None):
    """Install a fake ``sqlite_vec`` module whose ``load(conn)`` mirrors the
    real extension's behavior: it requires (and enables) extension loading on
    the connection to register itself.

    If ``load_side_effect`` is given, ``load`` raises it instead, simulating
    sqlite3.OperationalError from a failed dlopen (another path where the
    current loader calls enable_load_extension(True) and never restores it).
    """
    import sys, types

    fake = types.ModuleType("sqlite_vec")

    def _load(conn):
        if load_side_effect is not None:
            raise load_side_effect
        conn.enable_load_extension(True)  # mirror real sqlite_vec.load

    fake.load = _load
    monkeypatch.setitem(sys.modules, "sqlite_vec", fake)
    return fake


def _assert_load_extension_blocked(conn):
    """Extension loading must be DISABLED: SQLite rejects load_extension up
    front with 'not authorized'. If it is still enabled, the call instead
    reaches the filesystem and fails with 'cannot open shared object' — that
    is the leak signature.
    """
    with pytest.raises(sqlite3.OperationalError) as excinfo:
        conn.execute("SELECT load_extension('definitely_not_a_real_extension')")
    assert "not authorized" in str(excinfo.value), (
        f"extension loading still enabled after _load_sqlite_vec "
        f"(load_extension was reachable): {excinfo.value!r}"
    )


def test_load_sqlite_vec_disables_extension_loading_after_success(monkeypatch):
    """RED driver: when sqlite_vec.load() succeeds (and itself enables
    extension loading to register), _load_sqlite_vec must re-disable
    extension loading before returning.

    Pre-fix the call to enable_load_extension(True) inside the loader is never
    reverted, so load_extension remains callable afterward.
    """
    _install_sqlite_vec_stub(monkeypatch)
    conn = sqlite3.connect(":memory:")
    recovery._load_sqlite_vec(conn)
    _assert_load_extension_blocked(conn)


def test_load_sqlite_vec_disables_extension_loading_after_operational_error(
    monkeypatch,
):
    """If sqlite_vec.load() itself raises OperationalError (e.g. a failed
    dlopen on a present-but-broken build), enable_load_extension(True) has
    already run and must still be re-disabled.
    """
    _install_sqlite_vec_stub(
        monkeypatch, load_side_effect=sqlite3.OperationalError("dlopen failed")
    )
    conn = sqlite3.connect(":memory:")
    recovery._load_sqlite_vec(conn)
    _assert_load_extension_blocked(conn)


def test_load_sqlite_vec_disables_extension_loading_when_absent(monkeypatch):
    """When sqlite-vec is NOT installed, _load_sqlite_vec swallows the
    ImportError and must leave extension loading disabled. (Today this passes
    incidentally because the ImportError fires before the toggle; the stub
    makes the assertion load-bearing for the future.)
    """
    import sys

    monkeypatch.setitem(sys.modules, "sqlite_vec", None)
    conn = sqlite3.connect(":memory:")
    recovery._load_sqlite_vec(conn)  # must not raise
    _assert_load_extension_blocked(conn)


def test_restore_backup_rejects_load_extension_in_dump(tmp_path, monkeypatch):
    """End-to-end: a tampered-but-checksum-valid legacy dump whose dump SQL invokes
    load_extension must NOT execute that call. The staged-restore connection
    must have extension loading disabled when executescript runs.

    We stub sqlite_vec so the leak is deterministic, craft a backup whose dump
    embeds ``SELECT load_extension(...)``, and recompute both checksums so it
    passes restore's integrity gate — isolating the extension-loading vector
    from the checksum gate (checksums are integrity, not authenticity).
    """
    import gzip as _gz

    _install_sqlite_vec_stub(monkeypatch)

    db_path = tmp_path / "src.db"
    bdir = tmp_path / "bk"
    _make_simple_db(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    backup_path = Path(backup["backup_path"])
    meta_path = backup_path.with_suffix(".gz.json")

    # Legacy text SQL dump format containing load_extension
    malicious = b"BEGIN TRANSACTION;\nCREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT);\nSELECT load_extension('evil_extension_payload');\nCOMMIT;\n"
    backup_path.write_bytes(_gz.compress(malicious))

    new_backup_checksum = hashlib.sha256(backup_path.read_bytes()).hexdigest()[:16]
    new_dump_checksum = hashlib.sha256(malicious).hexdigest()
    meta = json.loads(meta_path.read_text())
    meta["backup_checksum"] = new_backup_checksum
    meta["dump_checksum"] = new_dump_checksum
    meta_path.write_text(json.dumps(meta, indent=2))

    target = tmp_path / "target.db"
    _make_simple_db(target)

    with pytest.raises(sqlite3.OperationalError) as excinfo:
        recovery.restore_backup(backup_path, target)
    assert "not authorized" in str(excinfo.value), (
        f"untrusted dump SQL reached load_extension during restore "
        f"(extension loading was enabled on the staged connection): "
        f"{excinfo.value!r}"
    )


def test_backup_and_restore_roundtrip_with_fts5(tmp_path):
    """[#640] Database with FTS5 virtual tables backs up and restores successfully."""
    db_path = tmp_path / "fts_source.db"
    backup_dir = tmp_path / "backups"
    restore_target = tmp_path / "fts_restored.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE items (id INTEGER PRIMARY KEY, title TEXT)")
    conn.execute("CREATE VIRTUAL TABLE items_fts USING fts5(title)")
    conn.execute("INSERT INTO items (id, title) VALUES (1, 'FTS5 virtual table backup test')")
    conn.execute("INSERT INTO items_fts (rowid, title) VALUES (1, 'FTS5 virtual table backup test')")
    conn.commit()
    conn.close()

    # 1. create_backup produces a binary SQLite backup
    backup = recovery.create_backup(db_path=db_path, backup_dir=backup_dir)
    assert Path(backup["backup_path"]).exists()
    assert backup.get("format") == "sqlite3"

    decompressed = _gzip.decompress(Path(backup["backup_path"]).read_bytes())
    assert decompressed[:16] == b"SQLite format 3\x00"

    # 2. restore_backup successfully restores without shadow-table operational errors
    result = recovery.restore_backup(Path(backup["backup_path"]), restore_target)
    assert result["restored"] is True
    assert result["integrity_check"] is True

    # 3. Assert restored DB integrity, table counts, and FTS retrievability
    assert recovery.verify_integrity(restore_target) is True
    target_conn = sqlite3.connect(str(restore_target))
    try:
        items_count = target_conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        assert items_count == 1
        match_rows = target_conn.execute("SELECT rowid, title FROM items_fts WHERE items_fts MATCH 'backup'").fetchall()
        assert len(match_rows) == 1
        assert match_rows[0][0] == 1
    finally:
        target_conn.close()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX mode-bit permissions are not supported on Windows")
def test_backup_and_metadata_files_created_with_private_permissions(tmp_path):
    """[#640] Staged and published backup files and metadata are created with mode 0600."""
    import stat

    orig_umask = os.umask(0o022)
    try:
        db_path = tmp_path / "perm_test.db"
        bdir = tmp_path / "bk"
        _make_simple_db(db_path)
        backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)
        backup_path = Path(backup["backup_path"])
        meta_path = Path(backup["metadata_path"])

        assert stat.S_IMODE(backup_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(meta_path.stat().st_mode) == 0o600
    finally:
        os.umask(orig_umask)


def test_create_backup_format_selection_and_validation(tmp_path, monkeypatch):
    """[#640] create_backup format validation, normalization, and environment precedence."""
    db_path = tmp_path / "src.db"
    bdir = tmp_path / "bk"
    _make_simple_db(db_path)

    # 1. Default format is sqlite3
    monkeypatch.delenv("MNEMOSYNE_BACKUP_FORMAT", raising=False)
    b_default = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    assert b_default["format"] == "sqlite3"
    assert _gzip.decompress(Path(b_default["backup_path"]).read_bytes()).startswith(b"SQLite format 3\x00")

    # 2. Whitespace-padded and mixed-case normalization
    b_norm_sql = recovery.create_backup(db_path=db_path, backup_dir=bdir, format="  SQL  ")
    assert b_norm_sql["format"] == "sql"
    assert not _gzip.decompress(Path(b_norm_sql["backup_path"]).read_bytes()).startswith(b"SQLite format 3\x00")

    b_norm_sqlite = recovery.create_backup(db_path=db_path, backup_dir=bdir, format=" Sqlite3 ")
    assert b_norm_sqlite["format"] == "sqlite3"

    # 3. Environment variable MNEMOSYNE_BACKUP_FORMAT
    monkeypatch.setenv("MNEMOSYNE_BACKUP_FORMAT", "sql")
    b_env_sql = recovery.create_backup(db_path=db_path, backup_dir=bdir)
    assert b_env_sql["format"] == "sql"

    # 4. Explicit format overrides environment variable
    b_override = recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sqlite3")
    assert b_override["format"] == "sqlite3"

    # 5. Invalid format fails closed (raises ValueError)
    with pytest.raises(ValueError, match="Invalid backup format: 'sql3'"):
        recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sql3")

    monkeypatch.setenv("MNEMOSYNE_BACKUP_FORMAT", "invalid_fmt")
    with pytest.raises(ValueError, match="Invalid backup format: 'invalid_fmt'"):
        recovery.create_backup(db_path=db_path, backup_dir=bdir)


def test_restore_backup_accepts_valid_legacy_sql_dump(tmp_path):
    """[#640] restore_backup() seamlessly restores valid legacy text SQL dumps."""
    db_path = tmp_path / "src.db"
    bdir = tmp_path / "bk"
    _make_simple_db(db_path)

    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir, format="sql")
    backup_path = Path(backup["backup_path"])
    assert backup["format"] == "sql"

    decompressed = _gzip.decompress(backup_path.read_bytes())
    assert not decompressed.startswith(b"SQLite format 3\x00")
    assert b"CREATE TABLE" in decompressed

    target = tmp_path / "legacy_restored.db"
    res = recovery.restore_backup(backup_path, target)
    assert res["restored"] is True
    assert res["integrity_check"] is True

    target_conn = sqlite3.connect(str(target))
    try:
        rows = target_conn.execute("SELECT id, v FROM t ORDER BY id").fetchall()
        assert rows == [(1, "a"), (2, "b")]
    finally:
        target_conn.close()


def test_backup_and_restore_captures_committed_wal_frames(tmp_path):
    """[#640] create_backup() captures committed WAL frames and restore delivers them."""
    src = tmp_path / "wal_src.db"
    bdir = tmp_path / "bk"
    target = tmp_path / "wal_restored.db"

    conn = sqlite3.connect(str(src))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE wal_items (id INTEGER PRIMARY KEY, note TEXT)")
    conn.execute("INSERT INTO wal_items VALUES (1, 'initial')")
    conn.commit()
    conn.execute("INSERT INTO wal_items VALUES (2, 'committed_wal_frame')")
    conn.commit()

    wal_file = src.with_name(src.name + "-wal")
    assert wal_file.exists()
    assert wal_file.stat().st_size > 0

    backup = recovery.create_backup(db_path=src, backup_dir=bdir)
    res = recovery.restore_backup(Path(backup["backup_path"]), target)
    assert res["restored"] is True
    assert res["integrity_check"] is True

    target_conn = sqlite3.connect(str(target))
    try:
        rows = target_conn.execute("SELECT id, note FROM wal_items ORDER BY id").fetchall()
        assert rows == [(1, "initial"), (2, "committed_wal_frame")]
    finally:
        target_conn.close()
        conn.close()


def test_restore_backup_rejects_corrupt_binary_payload_and_cleans_stage(tmp_path):
    """[#640] Corrupted binary payload fails restore, preserves target byte-identically, and cleans staged files."""
    target = tmp_path / "target.db"
    _make_simple_db(target)
    initial_target_bytes = target.read_bytes()

    src = tmp_path / "src.db"
    bdir = tmp_path / "bk"
    _make_simple_db(src)

    backup = recovery.create_backup(db_path=src, backup_dir=bdir)
    backup_path = Path(backup["backup_path"])
    meta_path = Path(backup["metadata_path"])

    corrupt_dump = b"SQLite format 3\x00" + b"\xff" * 4096
    compressed_corrupt = _gzip.compress(corrupt_dump)
    backup_path.write_bytes(compressed_corrupt)

    meta = json.loads(meta_path.read_text())
    meta["backup_checksum"] = hashlib.sha256(compressed_corrupt).hexdigest()[:16]
    meta["dump_checksum"] = hashlib.sha256(corrupt_dump).hexdigest()
    meta_path.write_text(json.dumps(meta, indent=2))

    with pytest.raises((RuntimeError, sqlite3.DatabaseError)):
        recovery.restore_backup(backup_path, target)

    assert target.read_bytes() == initial_target_bytes
    assert list(target.parent.glob("*restore_staged*")) == []


# ---------------------------------------------------------------------------
# Task 9: re-acquire the writer lock on the replacement inode after
# os.replace(). A BEGIN IMMEDIATE lock binds to the inode opened at connect
# time; after os.replace the path resolves to a NEW inode that the held lock
# does not cover. These tests prove a competing writer (a separate process
# with busy_timeout=0) cannot enter the post-replace verify or rollback
# window, that re-acquisition failure is fail-closed and visible, and that
# rollback remains an in-place copy (inode-stable) so the new-inode lock
# covers it.
#
# Competitors run in a SEPARATE PROCESS (multiprocessing) coordinated by
# multiprocessing.Event / Queue — no arbitrary sleeps. This catches a lock
# regression that same-process threads would not on a POSIX-fcntl-lock build.
# ---------------------------------------------------------------------------

import multiprocessing as _mp
import shutil as _shutil


def _backup_and_target(tmp_path):
    """Build a valid backup and an existing target with a distinguishing row."""
    db_path = tmp_path / "src.db"
    bdir = tmp_path / "bk"
    _make_db_simple(db_path)
    backup = recovery.create_backup(db_path=db_path, backup_dir=bdir)

    target = tmp_path / "target.db"
    _make_db_simple(target)
    conn = sqlite3.connect(str(target))
    conn.execute("INSERT INTO t VALUES (777, 'target-original')")
    conn.commit()
    conn.close()
    return backup, target


def _competitor_worker(target_str, enter_event, allow_event, result_queue):
    """Separate-process competitor: wait for enter_event, then try to acquire
    a writer lock with busy_timeout=0 and insert a row. Reports (entered, err)
    via result_queue."""
    try:
        enter_event.wait(timeout=10)
        conn = sqlite3.connect(target_str, timeout=0)
        conn.execute("PRAGMA busy_timeout=0")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("INSERT INTO t VALUES (8888, 'competitor')")
        conn.commit()
        conn.close()
        result_queue.put((True, None))
    except sqlite3.OperationalError as exc:
        result_queue.put((False, str(exc)))
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put((False, repr(exc)))


def test_competing_writer_cannot_enter_post_replace_verify_window(
    tmp_path, monkeypatch
):
    """A separate-process writer must NOT be able to BEGIN IMMEDIATE on the
    target while post-replace integrity verification runs. Before the fix the
    held lock belongs to the OLD inode (replaced away), so the competitor
    succeeds."""
    backup, target = _backup_and_target(tmp_path)

    verify_entered = _threading.Event()
    allow_verify = _threading.Event()

    def gated_verify(path):
        verify_entered.set()
        allow_verify.wait(timeout=10)
        return True

    monkeypatch.setattr(recovery, "verify_integrity", gated_verify)

    # Use a multiprocessing Event/Queue visible to the child process.
    ctx = _mp.get_context("spawn")
    m_enter = ctx.Event()
    m_allow = ctx.Event()
    m_queue = ctx.Queue()
    proc = ctx.Process(
        target=_competitor_worker,
        args=(str(target), m_enter, m_allow, m_queue),
    )
    proc.start()

    # Bridge the in-process threading.Event to the multiprocessing.Event so
    # the competitor is released exactly when verify_integrity is entered.
    def gated_verify_mp(path):
        verify_entered.set()
        m_enter.set()
        allow_verify.wait(timeout=10)
        return True

    monkeypatch.setattr(recovery, "verify_integrity", gated_verify_mp)

    try:
        # Run restore in a thread so we can handshake events.
        result_holder = {"exc": None}

        def do_restore():
            try:
                recovery.restore_backup(Path(backup["backup_path"]), target)
            except Exception as exc:
                result_holder["exc"] = exc

        t = _threading.Thread(target=do_restore)
        t.start()
        # Wait until verify is entered (competitor now released).
        verify_entered.wait(timeout=10)
        # Give the competitor a moment to attempt (it will block/fail under
        # the fix; under the bug it commits instantly). Poll the queue with a
        # short timeout instead of sleeping.
        competitor_result = m_queue.get(timeout=5)
        allow_verify.set()
        t.join(timeout=10)

        assert result_holder["exc"] is None, (
            f"restore raised unexpectedly: {result_holder['exc']!r}"
        )
        entered, err = competitor_result
        assert entered is False, (
            "competing writer entered the post-replace VERIFY window and "
            f"committed a row (err={err})"
        )
        assert err is not None and "locked" in err.lower(), (
            f"expected 'database is locked', got: {err}"
        )
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def test_competing_writer_cannot_enter_post_replace_rollback_window(
    tmp_path, monkeypatch
):
    """When post-replace verify fails and the rollback copy runs, a separate-
    process writer must NOT be able to BEGIN IMMEDIATE on the target. Before
    the fix the new inode is unlocked during rollback."""
    backup, target = _backup_and_target(tmp_path)
    original_bytes = target.read_bytes()

    monkeypatch.setattr(recovery, "verify_integrity", lambda p: False)

    ctx = _mp.get_context("spawn")
    m_enter = ctx.Event()
    m_allow = ctx.Event()
    m_queue = ctx.Queue()
    proc = ctx.Process(
        target=_competitor_worker,
        args=(str(target), m_enter, m_allow, m_queue),
    )
    proc.start()

    rollback_entered = _threading.Event()
    allow_rollback = _threading.Event()
    real_copy2 = _shutil.copy2

    def gated_copy2(src, dst, *args, **kwargs):
        # Only the rollback direction: preserved -> target.
        if str(src).endswith(".restore_preserved") and str(dst) == str(target):
            rollback_entered.set()
            m_enter.set()
            allow_rollback.wait(timeout=10)
        return real_copy2(src, dst, *args, **kwargs)

    monkeypatch.setattr("mnemosyne.dr.recovery.shutil.copy2", gated_copy2)

    try:
        result_holder = {"exc": None}

        def do_restore():
            try:
                recovery.restore_backup(Path(backup["backup_path"]), target)
            except Exception as exc:
                result_holder["exc"] = exc

        t = _threading.Thread(target=do_restore)
        t.start()
        rollback_entered.wait(timeout=10)
        competitor_result = m_queue.get(timeout=5)
        allow_rollback.set()
        t.join(timeout=10)

        assert result_holder["exc"] is not None, (
            "restore did not raise after forced post-replace integrity failure"
        )
        entered, err = competitor_result
        assert entered is False, (
            "competing writer entered the post-replace ROLLBACK window and "
            f"committed a row (err={err})"
        )
        assert err is not None and "locked" in err.lower(), (
            f"expected 'database is locked', got: {err}"
        )
        # Rollback restored original bytes in place.
        assert target.read_bytes() == original_bytes, (
            "target bytes differ from original after rollback"
        )
    finally:
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)


def test_competing_writer_cannot_enter_during_inode_replacement_handoff(
    tmp_path, monkeypatch
):
    """A competing writer attempting BEGIN IMMEDIATE during the inode handoff
    (after os.replace, while the old-inode lock is released and new-inode lock is
    acquired) must be blocked because the replacement inode lock is held."""
    backup, target = _backup_and_target(tmp_path)

    handoff_entered = _threading.Event()
    competitor_done = _threading.Event()
    competitor_result = {"entered": False, "err": None}
    release_calls = 0

    real_release = recovery._release_writer_lock

    def gated_release(conn):
        nonlocal release_calls
        release_calls += 1
        if release_calls == 1:
            handoff_entered.set()
            competitor_done.wait(timeout=10)
        return real_release(conn)

    monkeypatch.setattr(recovery, "_release_writer_lock", gated_release)

    def compete():
        handoff_entered.wait(timeout=10)
        try:
            conn = sqlite3.connect(str(target), timeout=0)
            conn.execute("PRAGMA busy_timeout=0")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("INSERT INTO t VALUES (9999, 'competitor')")
            conn.commit()
            conn.close()
            competitor_result["entered"] = True
        except sqlite3.OperationalError as exc:
            competitor_result["err"] = str(exc)
        finally:
            competitor_done.set()

    comp_thread = _threading.Thread(target=compete)
    comp_thread.start()

    recovery.restore_backup(Path(backup["backup_path"]), target)
    comp_thread.join(timeout=5)

    assert competitor_result["entered"] is False
    assert (
        competitor_result["err"] is not None
        and "locked" in competitor_result["err"].lower()
    )


def test_restore_backup_post_replace_reacquire_failure_is_visible_and_valid(
    tmp_path, monkeypatch
):
    """If the SECOND (post-replace) writer-lock acquisition fails, the restore
    must raise visibly (fail-closed) and never claim success. The target holds
    the staged image (verified + fsynced before replace). This test fakes the
    failure; the real-race contract is a visible failure + retained preserved
    original, not a proof that a competitor could not have changed the target.
    """
    backup, target = _backup_and_target(tmp_path)

    original_acquire = recovery._acquire_writer_lock
    call_count = {"n": 0}

    def fail_second_acquire(db_path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise sqlite3.OperationalError("database is locked")
        return original_acquire(db_path)

    monkeypatch.setattr(recovery, "_acquire_writer_lock", fail_second_acquire)

    with pytest.raises(Exception) as excinfo:
        recovery.restore_backup(Path(backup["backup_path"]), target)

    # Fail-closed: a visible error, not a silent success.
    assert (
        "locked" in str(excinfo.value).lower()
        or "restore" in str(excinfo.value).lower()
    ), f"unexpected error shape: {excinfo.value!r}"

    # The staged image at the path is a valid database (it was verified and
    # fsynced before the replace). Pinned only because this test fakes the
    # failure with no real competitor.
    conn = sqlite3.connect(str(target))
    ok = conn.execute("PRAGMA integrity_check").fetchone()[0]
    conn.close()
    assert ok == "ok", "staged image left at target was not a valid database"

    # The preserved original is retained (recovery never cleans it).
    preserved = target.with_name(target.name + ".restore_preserved")
    assert preserved.exists(), "preserved original was lost on reacquire failure"


def test_restore_backup_rollback_copy_keeps_target_inode(tmp_path, monkeypatch):
    """The rollback copy (shutil.copy2 preserved -> target) must be in-place:
    it must NOT replace the target inode. This pins the assumption that lets
    the replacement-inode lock cover the rollback copy."""
    backup, target = _backup_and_target(tmp_path)

    monkeypatch.setattr(recovery, "verify_integrity", lambda p: False)

    inodes = {}
    real_copy2 = _shutil.copy2

    def inode_tracking_copy2(src, dst, *args, **kwargs):
        if str(src).endswith(".restore_preserved") and str(dst) == str(target):
            inodes["before"] = os.stat(str(target)).st_ino
        result = real_copy2(src, dst, *args, **kwargs)
        if str(src).endswith(".restore_preserved") and str(dst) == str(target):
            inodes["after"] = os.stat(str(target)).st_ino
        return result

    monkeypatch.setattr("mnemosyne.dr.recovery.shutil.copy2", inode_tracking_copy2)

    with pytest.raises(Exception):
        recovery.restore_backup(Path(backup["backup_path"]), target)

    assert "before" in inodes and "after" in inodes, (
        "rollback copy was not observed (inode tracking did not fire)"
    )
    assert inodes["before"] == inodes["after"], (
        f"rollback copy changed the target inode ({inodes['before']} -> "
        f"{inodes['after']}); the new-inode lock would NOT cover the rollback"
    )
