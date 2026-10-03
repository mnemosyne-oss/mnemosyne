"""
Mnemosyne Disaster Recovery System

Comprehensive backup, restore, and integrity verification for Mnemosyne.
"""

import contextlib
import gzip
import io
import os
import json
import hashlib
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def get_default_paths():
    """Get default Mnemosyne paths.

    These MUST resolve to the same location the live store uses (see
    ``mnemosyne.core.beam``), or backup/restore -- and ``mnemosyne reindex``'s
    auto-backup -- operate on a different database than the one in use. The
    precedence mirrors beam:

    * data dir: ``MNEMOSYNE_DATA_DIR`` if set, else
      ``$HERMES_HOME/mnemosyne/data`` (``HERMES_HOME`` defaults to ``~/.hermes``).
    * backups: ``MNEMOSYNE_BACKUP_DIR`` if set, else a ``backups`` dir alongside
      the data dir.

    Previously this hardcoded ``~/.mnemosyne/data``, which disagreed with the
    store whenever ``MNEMOSYNE_DATA_DIR`` or ``HERMES_HOME`` was set, so
    operations failed with "Database not found".
    """
    if os.environ.get("MNEMOSYNE_DATA_DIR"):
        data_dir = Path(os.environ["MNEMOSYNE_DATA_DIR"])
    else:
        root = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
        data_dir = root / "mnemosyne" / "data"
    backup_dir = Path(
        os.environ.get("MNEMOSYNE_BACKUP_DIR", data_dir.parent / "backups")
    )
    db_path = data_dir / "mnemosyne.db"
    return data_dir, backup_dir, db_path


def _allocate_unique_backup_path(backup_dir: Path, timestamp: str) -> tuple[Path, Path, Path]:
    """Atomically allocate a unique backup filename and private staging paths.

    Uses ``O_CREAT | O_EXCL`` on a hidden reservation file and private staging
    paths so in-progress backups are never visible to ``rotate_backups()``,
    ``list_backups()``, or other glob consumers until fully published.
    """
    import secrets

    suffix = ""
    for _ in range(64):
        name = f"mnemosyne_backup_{timestamp}{suffix}.db.gz"
        final_path = backup_dir / name
        reservation = backup_dir / f".{name}.reserve"
        staged_path = backup_dir / f".{name}.staging-{os.getpid()}-{secrets.token_hex(4)}"

        if final_path.exists():
            suffix = "_" + secrets.token_hex(3)
            continue

        try:
            fd = os.open(str(reservation), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.close(fd)
        except FileExistsError:
            suffix = "_" + secrets.token_hex(3)
            continue

        if final_path.exists():
            try:
                reservation.unlink()
            except OSError:
                pass
            suffix = "_" + secrets.token_hex(3)
            continue

        try:
            fd = os.open(str(staged_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
            os.close(fd)
            return final_path, staged_path, reservation
        except BaseException:
            try:
                reservation.unlink()
            except OSError:
                pass
            raise
    raise RuntimeError(f"Could not allocate a unique backup filename in {backup_dir}")


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_staged_path(db_path: Path) -> Path:
    """A staged-restore path unique per invocation, so concurrent restores to
    the same target cannot clobber each other's staging file."""
    import secrets

    token = secrets.token_hex(4)
    return db_path.with_name(f"{db_path.name}.{os.getpid()}.{token}.restore_staged")


def _fsync_dir(path: Path) -> None:
    """fsync a directory so rename/replace metadata reaches durable storage."""
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    except (OSError, ValueError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def create_backup(db_path: Path = None, backup_dir: Path = None) -> Dict:
    """
    Create a compressed backup of the database.

    Returns:
        Dict with backup_path, size, checksum, and timestamp
    """
    _, default_backup_dir, default_db = get_default_paths()
    db_path = db_path or default_db
    backup_dir = backup_dir or default_backup_dir

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path, staged_backup_path, reservation_path = _allocate_unique_backup_path(
        backup_dir, timestamp
    )
    meta_path = backup_path.with_suffix(".gz.json")
    staged_meta_path = staged_backup_path.with_suffix(".gz.json")

    try:
        # Use sqlite3 online backup API instead of shutil.copyfileobj.
        # sqlite3.backup() is lock-aware (acquires read-lock), includes
        # uncommitted WAL frames, and is atomic — it won't produce a torn
        # file if a checkpoint runs partway through. The old copyfileobj
        # approach only copied the .db file, missed .db-wal frames, and
        # could produce corrupted backups under concurrent write load.
        src = sqlite3.connect(str(db_path))
        try:
            _load_sqlite_vec(src)
            dst = sqlite3.connect(":memory:")
            try:
                _load_sqlite_vec(dst)
                src.backup(dst)

                # Serialize the in-memory backup → gzip → disk
                buf = io.BytesIO()
                for line in dst.iterdump():
                    buf.write((line + "\n").encode("utf-8"))
            finally:
                dst.close()
        finally:
            src.close()

        dump_bytes = buf.getvalue()
        with gzip.open(staged_backup_path, "wb") as f_out:
            f_out.write(dump_bytes)

        # Calculate checksums. ``dump_checksum`` covers the decompressed SQL dump
        # payload so restore can detect corruption that survives gzip decompression
        # (the older file-level checksums only covered the compressed container).
        db_checksum = _file_sha256(db_path)[:16]
        backup_checksum = _file_sha256(staged_backup_path)[:16]
        dump_checksum = hashlib.sha256(dump_bytes).hexdigest()

        # Create metadata
        metadata = {
            "timestamp": timestamp,
            "original_size": db_path.stat().st_size,
            "backup_size": staged_backup_path.stat().st_size,
            "db_checksum": db_checksum,
            "backup_checksum": backup_checksum,
            "dump_checksum": dump_checksum,
            "compressed": True,
        }

        # Save metadata to staging path then publish both atomically
        with open(staged_meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        _fsync_path(staged_backup_path)
        _fsync_path(staged_meta_path)
        os.replace(staged_meta_path, meta_path)
        os.replace(staged_backup_path, backup_path)
        try:
            reservation_path.unlink()
        except OSError:
            pass
        _fsync_dir(backup_dir)

        return {
            "backup_path": str(backup_path),
            "metadata_path": str(meta_path),
            **metadata,
        }
    except BaseException:
        for stale in (staged_backup_path, staged_meta_path, reservation_path):
            try:
                stale.unlink()
            except OSError:
                pass
        raise


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec on a connection when the extension is available.

    Mirrors core/beam.py's graceful fallback: absence of sqlite-vec just means
    no vec0 virtual tables to introspect.

    Security: sqlite_vec.load() registers itself via SQLite's extension-loading
    facility, which requires enable_load_extension(True). We MUST re-disable
    extension loading before returning, regardless of whether the load
    succeeded, failed with OperationalError, or the module was absent — callers
    (notably restore_backup's executescript of backup-provided SQL) must never
    run with extension loading left enabled. See security-privacy-audit-6a I-1.
    """
    try:
        conn.enable_load_extension(True)
        import sqlite_vec

        sqlite_vec.load(conn)
    except (ImportError, AttributeError, sqlite3.NotSupportedError, sqlite3.OperationalError):
        pass  # optional extra; absence/breakage just means no vec0 tables
    finally:
        try:
            conn.enable_load_extension(False)
        except (AttributeError, sqlite3.NotSupportedError, sqlite3.OperationalError):
            pass


def _reject_active_sidecars(db_path: Path) -> None:
    """Fail closed if WAL/SHM sidecars exist at the target.

    A stale -wal/-shm alongside the main file means the on-disk state is the
    union of the main file and uncommitted WAL frames. Atomically replacing
    only the main file would silently drop those frames, so we refuse and let
    an operator checkpoint/quiesce the writer first.
    """
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            raise RuntimeError(
                f"Refusing to restore while an active sidecar exists: {sidecar}. "
                f"Quiesce live writers and run 'PRAGMA wal_checkpoint(TRUNCATE)' "
                f"before restoring."
            )


def _acquire_writer_lock(db_path: Path) -> Optional[sqlite3.Connection]:
    """Acquire and HOLD an exclusive writer lock on the target.

    A probe that begins and immediately rolls back does not cover the staging
    window — a writer can start after the probe and race the replace. Instead
    this takes ``BEGIN IMMEDIATE`` on a kept-open connection and returns it;
    the caller releases it in a ``finally`` only after ``os.replace``. That
    keeps SQLite's write lock held across the whole check-then-replace window
    so a competing writer cannot enter.

    Returns the locked connection (to be closed by the caller), or None when
    the target does not exist yet (nothing to lock).
    """
    if not db_path.exists():
        return None
    lock_conn = sqlite3.connect(f"file:{db_path}?mode=rwc", uri=True)
    try:
        lock_conn.execute("PRAGMA busy_timeout=0")
        lock_conn.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        lock_conn.close()
        raise RuntimeError(
            f"Refusing to restore: a live writer appears to hold {db_path} "
            f"({exc}). Stop all Mnemosyne processes before restoring."
        ) from exc
    return lock_conn


def _release_writer_lock(conn: Optional[sqlite3.Connection]) -> None:
    """Safely roll back and close a held writer-lock connection.

    Returns None so call sites rebind (lock_conn = _release_writer_lock(
    lock_conn)), making cleanup idempotent: the finally can call it on
    a connection that was already released. Tolerates an already-closed
    connection by swallowing sqlite3.Error. Replaces the duplicated
    rollback()/close() blocks in the restore paths.
    """
    if conn is None:
        return None
    try:
        conn.rollback()
    except sqlite3.Error:
        pass
    try:
        conn.close()
    except sqlite3.Error:
        pass
    return None


def _reacquire_writer_lock(db_path: Path) -> sqlite3.Connection:
    """Re-acquire the exclusive writer lock on the *replacement* inode.

    os.replace swaps the path to a new inode; the connection held since
    before the replace still locks the OLD (now-unlinked) inode. This must be
    called immediately after os.replace so the new inode is covered for
    post-replace verify and rollback. It delegates to _acquire_writer_lock
    (no second lock implementation) and raises a visible RuntimeError if
    the new inode cannot be locked — never silently continuing.
    """
    return _acquire_writer_lock(db_path)


def _fsync_path(path: Path) -> None:
    """fsync a file path so the staged bytes reach durable storage."""
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def restore_backup(backup_path: Path, db_path: Path = None) -> Dict:
    """Restore database from a compressed backup. Fail-closed.

    Sequence:

      1. Read the backup and its metadata sidecar. Metadata MUST be a readable
         JSON object containing ``backup_checksum``; missing/malformed/unreadable
         metadata or a missing file checksum is rejected (never silently
         restored). ``dump_checksum`` is optional for compatibility with older
         valid backups.
      2. Verify the gzip file checksum and the decompressed dump checksum (when
         recorded).
      3. Reject active WAL/SHM sidecars. Acquire and HOLD an exclusive SQLite
         writer lock on the target from this point through ``os.replace`` so a
         competing writer cannot enter the staging window.
      4. Rebuild the dump into a uniquely-named staged DB in the target
         directory, load sqlite-vec where supported, and run
         ``PRAGMA integrity_check``. fsync the staged file.
      5. Preserve the original as ``.restore_preserved``, ``os.replace`` the
         staged file onto the target, then immediately release the old-inode
         writer lock and re-acquire ``BEGIN IMMEDIATE`` on the replacement
         inode (the held lock does not follow the path across replace). Fsync
         the parent dir after re-acquisition.
      6. Run ``PRAGMA integrity_check`` under the new-inode lock. If it fails,
         restore the preserved original in place, fsync, and raise — never
         report success after a failed post-replace check.

    Any error before the atomic replace leaves the original target untouched.
    The held writer lock and the staged file are cleaned up in all cases.

    Args:
        backup_path: Path to the .gz backup file.
        db_path: Destination database path.

    Returns:
        Dict with restore status and details. Raises on any validation
        failure so the caller can report a structured error.
    """
    _, _, default_db = get_default_paths()
    db_path = Path(db_path or default_db)
    backup_path = Path(backup_path)

    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    # --- 1. Read backup + require readable metadata ------------------------
    meta_path = backup_path.with_suffix(".gz.json")
    if not meta_path.exists():
        raise RuntimeError(
            f"Backup metadata sidecar not found: {meta_path}. Refusing to "
            f"restore without checksum verification."
        )
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Backup metadata is unreadable or malformed ({exc}). Refusing "
            f"to restore without checksum verification."
        ) from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("Backup metadata is not a JSON object; refusing to restore.")

    with gzip.open(backup_path, "rb") as f_in:
        dump_bytes = f_in.read()

    backup_checksum = _file_sha256(backup_path)[:16]
    dump_checksum = hashlib.sha256(dump_bytes).hexdigest()

    expected_backup = metadata.get("backup_checksum")
    if not expected_backup:
        raise RuntimeError(
            "Backup metadata lacks a backup_checksum; refusing to restore "
            "without file checksum verification."
        )
    if expected_backup != backup_checksum:
        raise RuntimeError(
            f"Backup file checksum mismatch: metadata={expected_backup} "
            f"actual={backup_checksum}. The backup file is corrupted or was "
            f"modified; refusing to restore."
        )
    # dump_checksum is optional: older valid backups predate it. Only verify
    # when present, so this stays backward compatible.
    expected_dump = metadata.get("dump_checksum")
    if expected_dump and expected_dump != dump_checksum:
        raise RuntimeError(
            f"Backup payload checksum mismatch: metadata={expected_dump} "
            f"actual={dump_checksum}. The decompressed dump is corrupted; "
            f"refusing to restore."
        )

    # --- 2. Reject active sidecars ----------------------------------------
    _reject_active_sidecars(db_path)

    # --- 3. Acquire + hold the writer lock through replace ----------------
    lock_conn = _acquire_writer_lock(db_path)
    staged_path = _unique_staged_path(db_path)
    preserved_path = db_path.with_name(db_path.name + ".restore_preserved")
    preserved_existed = db_path.exists()
    try:
        # --- 4. Rebuild into the staged DB and validate ------------------
        tmp_db = sqlite3.connect(str(staged_path))
        _load_sqlite_vec(tmp_db)
        tmp_db.set_authorizer(
            lambda action, *_args: (
                sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_ATTACH
                else sqlite3.SQLITE_OK
            )
        )
        try:
            tmp_db.executescript(dump_bytes.decode("utf-8"))
            integrity = tmp_db.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise RuntimeError(
                    f"Staged restore failed integrity_check: {integrity}. "
                    f"Refusing to replace target."
                )
            tmp_db.commit()
            try:
                tmp_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError:
                pass
            tmp_db.close()
        except Exception:
            try:
                tmp_db.close()
            except Exception:
                pass
            raise

        _fsync_path(staged_path)

        # --- 5. Preserve original, atomic replace -------------------------
        if preserved_existed:
            shutil.copy2(db_path, preserved_path)
        os.replace(staged_path, db_path)
        staged_path = None  # consumed by replace
        # Acquire writer lock on the replacement inode BEFORE releasing the
        # old-inode lock, ensuring no window where both inodes are unlocked.
        new_lock = _reacquire_writer_lock(db_path)
        lock_conn = _release_writer_lock(lock_conn)
        lock_conn = new_lock
        _fsync_dir(db_path.parent)

        # --- 6. Post-replace integrity check; restore original on failure
        if not verify_integrity(db_path):
            # Never report success after a failed post-replace check. Restore
            # the preserved original in place, fsync, and raise.
            if preserved_existed and preserved_path.exists():
                shutil.copy2(preserved_path, db_path)
                _fsync_path(db_path)
                _fsync_dir(db_path.parent)
            raise RuntimeError(
                f"Post-replace integrity_check failed for {db_path}. The "
                f"original target was restored from {preserved_path}."
            )
    finally:
        lock_conn = _release_writer_lock(lock_conn)
        if staged_path is not None and staged_path.exists():
            try:
                staged_path.unlink()
            except OSError:
                pass

    return {
        "restored": True,
        "backup_used": str(backup_path),
        "database_path": str(db_path),
        "integrity_check": True,
        "backup_checksum": backup_checksum,
        "dump_checksum": dump_checksum,
        "preserved_original": str(preserved_path) if preserved_existed else None,
    }


def emergency_restore(backup_dir: Path = None, db_path: Path = None) -> Dict:
    """
    Automatically restore from the most recent valid backup.

    Returns:
        Dict with restore status
    """
    _, default_backup_dir, default_db = get_default_paths()
    backup_dir = backup_dir or default_backup_dir
    db_path = db_path or default_db

    # Find all backups
    backups = sorted(backup_dir.glob("mnemosyne_backup_*.db.gz"), reverse=True)

    if not backups:
        raise FileNotFoundError("No backups found in " + str(backup_dir))

    # Try each backup until one works
    for attempt, backup in enumerate(backups, start=1):
        try:
            result = restore_backup(backup, db_path)
            if result["integrity_check"]:
                return {"restored": True, "backup_used": str(backup), "attempts": attempt}
        except Exception:
            continue

    raise RuntimeError("All backups failed integrity check")


def verify_integrity(db_path: Path = None) -> bool:
    """
    Verify SQLite database integrity.

    Returns:
        True if database is valid, False otherwise
    """

    _, _, default_db = get_default_paths()
    db_path = db_path or default_db

    if not db_path.exists():
        return False

    try:
        with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()
            return bool(result) and result[0] == "ok"
    except Exception:
        return False


def list_backups(backup_dir: Path = None) -> List[Dict]:
    """
    List all available backups with metadata.

    Returns:
        List of backup information dictionaries
    """
    _, default_backup_dir, _ = get_default_paths()
    backup_dir = backup_dir or default_backup_dir

    backups = []
    for backup_file in sorted(
        backup_dir.glob("mnemosyne_backup_*.db.gz"), reverse=True
    ):
        meta_file = backup_file.with_suffix(".gz.json")

        info = {
            "file": str(backup_file),
            "name": backup_file.name,
            "size": backup_file.stat().st_size,
            "modified": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
        }

        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    info["metadata"] = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

        backups.append(info)

    return backups


def rotate_backups(backup_dir: Path = None, keep: int = 10) -> Dict:
    """
    Rotate backups, keeping only the most recent N.

    Args:
        keep: Number of backups to retain

    Returns:
        Dict with rotation results
    """
    _, default_backup_dir, _ = get_default_paths()
    backup_dir = backup_dir or default_backup_dir

    if not isinstance(keep, int) or keep < 0:
        raise ValueError("keep must be a non-negative integer")

    backups = sorted(backup_dir.glob("mnemosyne_backup_*.db.gz"))

    to_delete = backups if keep == 0 else (backups[:-keep] if len(backups) > keep else [])
    deleted = []

    for backup in to_delete:
        # Delete backup and metadata
        try:
            backup.unlink()
        except FileNotFoundError:
            continue
        meta = backup.with_suffix(".gz.json")
        if meta.exists():
            try:
                meta.unlink()
            except FileNotFoundError:
                pass
        deleted.append(backup.name)

    return {
        "total_backups": len(backups),
        "kept": keep,
        "deleted": len(deleted),
        "deleted_files": deleted,
    }


def health_check() -> Dict:
    """
    Comprehensive health check of Mnemosyne system.

    Returns:
        Dict with health status of all components
    """
    data_dir, backup_dir, db_path = get_default_paths()

    # Check database
    db_exists = db_path.exists()
    db_valid = verify_integrity(db_path) if db_exists else False

    # Check backups
    backups = sorted(
        backup_dir.glob("mnemosyne_backup_*.db.gz"),
        key=lambda path: path.stat().st_mtime,
    ) if backup_dir.exists() else []

    return {
        "database": {
            "exists": db_exists,
            "valid": db_valid,
            "path": str(db_path),
            "message": "Database integrity verified"
            if db_valid
            else "Database missing or corrupt",
        },
        "backups": {
            "total": len(backups),
            "latest": str(backups[-1]) if backups else None,
            "directory": str(backup_dir),
        },
        "status": "healthy" if db_valid else "unhealthy",
    }


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "Usage: python -m mnemosyne.dr [backup|restore|emergency|verify|list|health|rotate]"
        )
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "backup":
        result = create_backup()
        print(json.dumps(result, indent=2))

    elif cmd == "restore":
        if len(sys.argv) <= 2:
            print("restore requires a backup path")
            sys.exit(2)
        result = restore_backup(Path(sys.argv[2]))
        print(json.dumps(result, indent=2))

    elif cmd == "emergency":
        result = emergency_restore()
        print(json.dumps(result, indent=2))

    elif cmd == "verify":
        valid = verify_integrity()
        print(json.dumps({"valid": valid}))

    elif cmd == "list":
        backups = list_backups()
        print(json.dumps(backups, indent=2))

    elif cmd == "health":
        status = health_check()
        print(json.dumps(status, indent=2))

    elif cmd == "rotate":
        result = rotate_backups()
        print(json.dumps(result, indent=2))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
