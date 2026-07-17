"""
Mnemosyne Disaster Recovery System

Comprehensive backup, restore, and integrity verification for Mnemosyne.
"""

import gzip
import io
import os
import json
import hashlib
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List


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
    backup_name = f"mnemosyne_backup_{timestamp}.db.gz"
    backup_path = backup_dir / backup_name
    
    # Use sqlite3 online backup API instead of shutil.copyfileobj.
    # sqlite3.backup() is lock-aware (acquires read-lock), includes
    # uncommitted WAL frames, and is atomic — it won't produce a torn
    # file if a checkpoint runs partway through. The old copyfileobj
    # approach only copied the .db file, missed .db-wal frames, and
    # could produce corrupted backups under concurrent write load.
    src = sqlite3.connect(str(db_path))
    # Load sqlite-vec on BOTH connections involved in the backup.
    # Without this, src.backup(dst) fails with "no such module: vec0"
    # when copying vec0 virtual tables, AND dst.iterdump() (used to
    # serialize the in-memory backup to gzipped SQL) fails the same
    # way when introspecting the destination's vec0 schema.
    # Mirrors the graceful-fallback pattern in core/beam.py.
    def _load_sqlite_vec(conn):
        try:
            import sqlite_vec
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
        except (ImportError, sqlite3.OperationalError):
            pass  # optional extra; absence just means no vec0 tables
    _load_sqlite_vec(src)
    dst = sqlite3.connect(":memory:")
    _load_sqlite_vec(dst)
    src.backup(dst)
    src.close()

    # Serialize the in-memory backup → gzip → disk
    buf = io.BytesIO()
    for line in dst.iterdump():
        buf.write((line + "\n").encode("utf-8"))
    dst.close()

    with gzip.open(backup_path, "wb") as f_out:
        f_out.write(buf.getvalue())
    
    # Calculate checksums
    db_checksum = hashlib.sha256(db_path.read_bytes()).hexdigest()[:16]
    backup_checksum = hashlib.sha256(backup_path.read_bytes()).hexdigest()[:16]
    
    # Create metadata
    metadata = {
        "timestamp": timestamp,
        "original_size": db_path.stat().st_size,
        "backup_size": backup_path.stat().st_size,
        "db_checksum": db_checksum,
        "backup_checksum": backup_checksum,
        "compressed": True
    }
    
    # Save metadata
    meta_path = backup_path.with_suffix('.gz.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        "backup_path": str(backup_path),
        "metadata_path": str(meta_path),
        **metadata
    }


def restore_backup(backup_path: Path, db_path: Path = None) -> Dict:
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
    
    # Create emergency backup of current DB
    if db_path.exists():
        emergency_path = db_path.with_suffix('.emergency_backup.db')
        shutil.copy2(db_path, emergency_path)
    
    # Decompress and restore using sqlite3 backup API
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(backup_path, "rb") as f_in:
        sql_dump = f_in.read().decode("utf-8")

    # Rebuild DB in-memory, then backup to target file
    mem_db = sqlite3.connect(":memory:")
    mem_db.executescript(sql_dump)
    target_db = sqlite3.connect(str(db_path))
    mem_db.backup(target_db)
    mem_db.close()
    target_db.close()
    
    # Verify restored database
    is_valid = verify_integrity(db_path)
    
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
        except Exception:
            continue
    
    raise RuntimeError("All backups failed integrity check")


def verify_integrity(db_path: Path = None) -> bool:
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
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Run PRAGMA integrity_check
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        
        conn.close()
        
        return result[0] == "ok"
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
    for backup_file in sorted(backup_dir.glob("mnemosyne_backup_*.db.gz"), reverse=True):
        meta_file = backup_file.with_suffix('.gz.json')
        
        info = {
            "file": str(backup_file),
            "name": backup_file.name,
            "size": backup_file.stat().st_size,
            "modified": datetime.fromtimestamp(backup_file.stat().st_mtime).isoformat()
        }
        
        if meta_file.exists():
            with open(meta_file) as f:
                info["metadata"] = json.load(f)
        
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
    backups = list(backup_dir.glob("mnemosyne_backup_*.db.gz")) if backup_dir.exists() else []
    
    return {
        "database": {
            "exists": db_exists,
            "valid": db_valid,
            "path": str(db_path),
            "message": "Database integrity verified" if db_valid else "Database missing or corrupt"
        },
        "backups": {
            "total": len(backups),
            "latest": str(backups[-1]) if backups else None,
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
