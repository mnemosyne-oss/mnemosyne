"""
Mnemosyne Disaster Recovery System

Comprehensive backup, restore, and integrity verification for Mnemosyne.
"""

import gzip
import os
import json
import hashlib
import shutil
import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


BACKUP_FORMAT = "sqlite-binary-gzip-v1"


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Load sqlite-vec when installed so vec0 databases can be inspected."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except (ImportError, sqlite3.OperationalError):
        # sqlite-vec is optional.  Databases without vec0 tables remain usable;
        # databases that need the extension will fail the subsequent check.
        pass


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Persist directory entries on POSIX; best effort on other platforms."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _readonly_sqlite_uri(path: Path) -> str:
    """Build a read-only SQLite URI without treating path delimiters as query text."""
    return f"{path.resolve().as_uri()}?mode=ro"


def _online_snapshot(source: Path, destination: Path) -> None:
    """Create a self-contained SQLite snapshot, including committed WAL frames."""
    source = source.resolve()
    src = sqlite3.connect(_readonly_sqlite_uri(source), uri=True)
    dst = sqlite3.connect(str(destination))
    try:
        _load_sqlite_vec(src)
        _load_sqlite_vec(dst)
        src.backup(dst)
        # A restored file must not depend on a sidecar left beside the source.
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.commit()
    finally:
        dst.close()
        src.close()
    _fsync_file(destination)


def _restore_snapshot(source: Path, destination: Path) -> None:
    """Restore a validated snapshot through SQLite without replacing its inode.

    Existing SQLite connections remain attached to the restored database.  The
    online-backup API owns the destination write transaction, so a failed copy
    is rolled back by SQLite instead of exposing a partially replaced file.
    """
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(_readonly_sqlite_uri(source), uri=True)
    dst = sqlite3.connect(str(destination))
    try:
        _load_sqlite_vec(src)
        _load_sqlite_vec(dst)
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    if destination.exists():
        _fsync_file(destination)
    wal = Path(str(destination) + "-wal")
    if wal.exists():
        _fsync_file(wal)
    _fsync_directory(destination.parent)


def _atomic_json(path: Path, payload: Dict) -> None:
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def _read_backup_metadata(backup_path: Path, *, verify_archive: bool = True) -> Dict:
    """Load and validate the metadata that commits a v1 backup pair."""
    meta_path = backup_path.with_suffix(".gz.json")
    if not meta_path.exists():
        raise ValueError("Backup metadata is required for sqlite-binary-gzip-v1")
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Backup metadata is unreadable or invalid") from exc
    if metadata.get("format") != BACKUP_FORMAT:
        raise ValueError("Backup metadata format is missing or unsupported")
    if metadata.get("compressed") is not True:
        raise ValueError("Backup metadata does not describe a compressed payload")
    for field in ("backup_checksum", "db_checksum"):
        value = metadata.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Backup metadata field {field} is missing or invalid")
    if verify_archive:
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        if metadata.get("backup_size") != backup_path.stat().st_size:
            raise ValueError("Backup size does not match metadata")
        if _sha256(backup_path) != metadata["backup_checksum"]:
            raise ValueError("Backup checksum does not match metadata")
    return metadata


def _remove_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)


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
    backup_dir = Path(os.environ.get("MNEMOSYNE_BACKUP_DIR", data_dir.parent / "backups"))
    db_path = data_dir / "mnemosyne.db"
    return data_dir, backup_dir, db_path


def create_backup(db_path: Optional[Path] = None, backup_dir: Optional[Path] = None) -> Dict:
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
    source_file_size = db_path.stat().st_size
    
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"mnemosyne_backup_{timestamp}_{uuid.uuid4().hex[:8]}.db.gz"
    backup_path = backup_dir / backup_name

    snapshot_fd, snapshot_raw = tempfile.mkstemp(
        prefix=".mnemosyne-snapshot-", suffix=".db", dir=backup_dir
    )
    os.close(snapshot_fd)
    snapshot_path = Path(snapshot_raw)
    gzip_fd, gzip_raw = tempfile.mkstemp(
        prefix=f".{backup_name}.", suffix=".tmp", dir=backup_dir
    )
    os.close(gzip_fd)
    gzip_tmp = Path(gzip_raw)

    try:
        _online_snapshot(db_path, snapshot_path)
        if not verify_integrity(snapshot_path):
            raise RuntimeError("SQLite snapshot failed integrity verification")

        with snapshot_path.open("rb") as f_in, gzip.open(gzip_tmp, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        _fsync_file(gzip_tmp)

        db_checksum = _sha256(snapshot_path)
        backup_checksum = _sha256(gzip_tmp)
        snapshot_size = snapshot_path.stat().st_size
        metadata = {
            "timestamp": timestamp,
            "original_size": snapshot_size,
            "source_file_size": source_file_size,
            "backup_size": gzip_tmp.stat().st_size,
            "db_checksum": db_checksum,
            "backup_checksum": backup_checksum,
            "compressed": True,
            "format": BACKUP_FORMAT,
        }
        meta_path = backup_path.with_suffix('.gz.json')
        try:
            _atomic_json(meta_path, metadata)
            # The archive is the commit marker. Metadata is durable first, so
            # readers can never observe a listed/restorable v1 archive without
            # its validation record.
            os.replace(gzip_tmp, backup_path)
            _fsync_directory(backup_dir)
        except BaseException:
            meta_path.unlink(missing_ok=True)
            backup_path.unlink(missing_ok=True)
            _fsync_directory(backup_dir)
            raise
    finally:
        snapshot_path.unlink(missing_ok=True)
        gzip_tmp.unlink(missing_ok=True)
    
    return {
        "backup_path": str(backup_path),
        "metadata_path": str(meta_path),
        **metadata
    }


def restore_backup(backup_path: Path, db_path: Optional[Path] = None) -> Dict:
    """
    Restore database from a compressed backup.
    
    Args:
        backup_path: Path to the .gz backup file
        db_path: Destination database path (default: ~/.mnemosyne/data/mnemosyne.db)
        
    Returns:
        Dict with restore status and details
    """
    _, _, default_db = get_default_paths()
    db_path = db_path or default_db
    
    if not backup_path.exists():
        raise FileNotFoundError(f"Backup not found: {backup_path}")

    meta_path = backup_path.with_suffix(".gz.json")
    metadata = None
    if meta_path.exists():
        try:
            raw_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Backup metadata is unreadable or invalid") from exc
        if raw_metadata.get("format") == BACKUP_FORMAT:
            metadata = _read_backup_metadata(backup_path)
    
    db_path.parent.mkdir(parents=True, exist_ok=True)

    payload_fd, payload_raw = tempfile.mkstemp(
        prefix=f".{db_path.name}.restore-payload-", suffix=".tmp", dir=db_path.parent
    )
    os.close(payload_fd)
    payload = Path(payload_raw)
    candidate: Optional[Path] = None

    try:
        with gzip.open(backup_path, "rb") as f_in, payload.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
            f_out.flush()
            os.fsync(f_out.fileno())

        with payload.open("rb") as handle:
            header = handle.read(16)

        if header == b"SQLite format 3\x00":
            if metadata is None:
                metadata = _read_backup_metadata(backup_path)
            candidate = payload
        else:
            # Backward-compatible best effort for pre-v1 SQL-dump backups.
            # Complex FTS shadow dumps may fail, but the live target remains
            # untouched because replay happens in a temporary candidate.
            sql_dump = payload.read_text(encoding="utf-8")
            candidate_fd, candidate_raw = tempfile.mkstemp(
                prefix=f".{db_path.name}.restore-db-", suffix=".tmp", dir=db_path.parent
            )
            os.close(candidate_fd)
            candidate = Path(candidate_raw)
            conn = sqlite3.connect(str(candidate))
            try:
                _load_sqlite_vec(conn)
                conn.executescript(sql_dump)
                conn.commit()
                conn.execute("PRAGMA journal_mode=DELETE")
            finally:
                conn.close()

        if not verify_integrity(candidate):
            raise RuntimeError("Restored candidate failed SQLite integrity verification")

        if metadata is not None:
            if header != b"SQLite format 3\x00":
                raise ValueError("Backup payload does not match sqlite-binary metadata")
            if _sha256(candidate) != metadata["db_checksum"]:
                raise ValueError("Restored database checksum does not match metadata")

        # Preserve the previous target with the same WAL-aware mechanism.
        emergency_path: Optional[Path] = None
        if db_path.exists():
            emergency_path = db_path.with_suffix(".emergency_backup.db")
            emergency_fd, emergency_raw = tempfile.mkstemp(
                prefix=f".{emergency_path.name}.", suffix=".tmp", dir=db_path.parent
            )
            os.close(emergency_fd)
            emergency_tmp = Path(emergency_raw)
            try:
                _online_snapshot(db_path, emergency_tmp)
                if not verify_integrity(emergency_tmp):
                    raise RuntimeError("Emergency backup failed integrity verification")
                os.replace(emergency_tmp, emergency_path)
                _fsync_directory(db_path.parent)
            finally:
                emergency_tmp.unlink(missing_ok=True)

        had_target = db_path.exists()
        try:
            _restore_snapshot(candidate, db_path)
            is_valid = verify_integrity(db_path)
            if not is_valid:
                raise RuntimeError("Restored target failed final integrity verification")
        except BaseException as restore_error:
            try:
                if had_target and emergency_path is not None:
                    _restore_snapshot(emergency_path, db_path)
                    if not verify_integrity(db_path):
                        raise RuntimeError("Rolled-back target failed integrity verification")
                elif not had_target:
                    db_path.unlink(missing_ok=True)
                    _remove_sidecars(db_path)
                    _fsync_directory(db_path.parent)
            except BaseException as rollback_error:
                raise RuntimeError(
                    f"Restore failed and automatic rollback also failed: {rollback_error}"
                ) from restore_error
            raise
    finally:
        payload.unlink(missing_ok=True)
        if candidate is not None:
            candidate.unlink(missing_ok=True)
    
    return {
        "restored": True,
        "backup_used": str(backup_path),
        "database_path": str(db_path),
        "integrity_check": is_valid
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
    for backup in backups:
        try:
            result = restore_backup(backup, db_path)
            if result["integrity_check"]:
                return {
                    "restored": True,
                    "backup_used": str(backup),
                    "attempts": 1
                }
        except Exception as e:
            continue
    
    raise RuntimeError("All backups failed integrity check")


def verify_integrity(db_path: Optional[Path] = None) -> bool:
    """
    Verify SQLite database integrity.
    
    Returns:
        True if database is valid, False otherwise
    """
    import sqlite3
    
    _, _, default_db = get_default_paths()
    db_path = db_path or default_db
    
    if not db_path.exists():
        return False
    
    try:
        with sqlite3.connect(str(db_path)) as conn:
            _load_sqlite_vec(conn)
            cursor = conn.cursor()
            quick = cursor.execute("PRAGMA quick_check").fetchone()
            integrity = cursor.execute("PRAGMA integrity_check").fetchone()
            foreign_keys = cursor.execute("PRAGMA foreign_key_check").fetchall()
            if quick[0] != "ok" or integrity[0] != "ok" or foreign_keys:
                return False

            fts_tables = cursor.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'table' AND lower(sql) LIKE '%using fts5%'"
            ).fetchall()
            for table_name, table_sql in fts_tables:
                quoted = table_name.replace('"', '""')
                cursor.execute(
                    f'INSERT INTO "{quoted}"("{quoted}") VALUES (\'integrity-check\')'
                )
                sql_lower = (table_sql or "").lower().replace(" ", "")
                if "content='" in sql_lower or 'content="' in sql_lower:
                    if "content=''" not in sql_lower and 'content=""' not in sql_lower:
                        cursor.execute(
                            f'INSERT INTO "{quoted}"("{quoted}", rank) '
                            "VALUES ('integrity-check', 1)"
                        )

            for vec_table, source_table in (
                ("vec_episodes", "episodic_memory"),
                ("vec_working", "working_memory"),
            ):
                exists = cursor.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (vec_table,),
                ).fetchone()
                if not exists:
                    continue
                cursor.execute(f'SELECT COUNT(*) FROM "{vec_table}"').fetchone()
                orphan = cursor.execute(
                    f'SELECT COUNT(*) FROM "{vec_table}" v '
                    f'LEFT JOIN "{source_table}" s ON s.rowid = v.rowid '
                    "WHERE s.rowid IS NULL"
                ).fetchone()[0]
                if orphan:
                    raise ValueError(f"orphaned rows in {vec_table}")

            conn.rollback()
            return True
    except Exception:
        return False


def _read_backup_listing_metadata(backup_path: Path) -> Dict:
    """Validate current backups and recognize pre-v1 SQL-dump archives."""
    meta_path = backup_path.with_suffix(".gz.json")
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Backup metadata is unreadable or invalid") from exc
    if raw.get("format") == BACKUP_FORMAT:
        return _read_backup_metadata(backup_path)
    if raw.get("format") is not None or raw.get("compressed") is not True:
        raise ValueError("Unsupported backup metadata format")

    expected = str(raw.get("backup_checksum") or "")
    actual = _sha256(backup_path)
    if expected and actual[: len(expected)] != expected:
        raise ValueError("Legacy backup checksum does not match metadata")
    try:
        with gzip.open(backup_path, "rb") as handle:
            prefix = handle.read(64 * 1024).decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("Legacy backup payload is unreadable") from exc
    if "BEGIN TRANSACTION" not in prefix:
        raise ValueError("Legacy backup payload is not a SQLite SQL dump")

    metadata = dict(raw)
    metadata["format"] = "legacy-sql-dump-gzip"
    return metadata


def list_backups(backup_dir: Path = None) -> List[Dict]:
    """
    List all available backups with metadata.
    
    Returns:
        List of backup information dictionaries
    """
    _, default_backup_dir, _ = get_default_paths()
    backup_dir = backup_dir or default_backup_dir
    
    backups = []
    for backup_file in sorted(backup_dir.glob("mnemosyne_backup_*.db.gz"), reverse=True):
        try:
            metadata = _read_backup_listing_metadata(backup_file)
        except (OSError, ValueError):
            continue
        info = {
            "file": str(backup_file),
            "name": backup_file.name,
            "size": backup_file.stat().st_size,
            "modified": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat(),
            "metadata": metadata,
        }
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
    
    backups = sorted(backup_dir.glob("mnemosyne_backup_*.db.gz"))
    
    to_delete = backups[:-keep] if len(backups) > keep else []
    deleted = []
    
    for backup in to_delete:
        # Delete backup and metadata
        backup.unlink()
        meta = backup.with_suffix('.gz.json')
        if meta.exists():
            meta.unlink()
        deleted.append(backup.name)
    
    return {
        "total_backups": len(backups),
        "kept": keep,
        "deleted": len(deleted),
        "deleted_files": deleted
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
    backups = list_backups(backup_dir) if backup_dir.exists() else []
    
    return {
        "database": {
            "exists": db_exists,
            "valid": db_valid,
            "path": str(db_path),
            "message": "Database integrity verified" if db_valid else "Database missing or corrupt"
        },
        "backups": {
            "total": len(backups),
            "latest": backups[0]["file"] if backups else None,
            "directory": str(backup_dir)
        },
        "status": "healthy" if db_valid else "unhealthy"
    }


# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m mnemosyne.dr [backup|restore|emergency|verify|list|health|rotate]")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "backup":
        result = create_backup()
        print(json.dumps(result, indent=2))
    
    elif cmd == "restore" and len(sys.argv) > 2:
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
