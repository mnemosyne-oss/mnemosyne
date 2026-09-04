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
    conn.commit()
    conn.close()

    # Act: this is the call path `mnemosyne backup` uses. Pre-fix it
    # raised sqlite3.OperationalError: no such module: vec0.
    result = recovery.create_backup(db_path=db_path, backup_dir=backup_dir)

    # Assert: backup file exists, is non-empty, gzipped, and the gz
    # contents contain the vec0 table definition.
    assert Path(result["backup_path"]).exists()
    assert result["backup_size"] > 0
    import gzip

    with gzip.open(result["backup_path"], "rt") as f:
        dump = f.read()
    assert "vec_items" in dump
    assert "CREATE VIRTUAL TABLE" in dump


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
    """End-to-end: a tampered-but-checksum-valid backup whose dump SQL invokes
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

    raw = _gz.decompress(backup_path.read_bytes())
    malicious = raw + b"\nSELECT load_extension('evil_extension_payload');\n"
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
