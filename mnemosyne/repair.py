"""Narrow, fail-closed repairs authorized by a Doctor JSON manifest.

This module deliberately supports only an explicitly selected working-memory
row: backfilling its already-stored fallback embedding into a confirmed vec0
working index, or expiring that one row.  It never initializes Mnemosyne,
reads memory content, re-embeds, deletes, changes schema, or touches another
table's data.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import uuid
from typing import Any

from mnemosyne.doctor import (
    _VEC0_VIRTUAL_TABLE,
    build_doctor_report,
    inspect_schema_fingerprint,
    open_readonly_doctor_db,
)


BACKFILL_VEC_WORKING = "backfill-vec-working"
EXPIRE = "expire"
_ALLOWED_ACTIONS = frozenset({BACKFILL_VEC_WORKING, EXPIRE})
_SELECTION_TABLE = "working_memory"
_VECTOR_KIND = re.compile(
    r"\bembedding\s+(float(?:32)?|int8|bit)\s*\[\s*(\d+)\s*\]", re.IGNORECASE
)


class RepairError(ValueError):
    """A bounded, content-safe repair validation error suitable for the CLI."""


def parse_selection(value: str) -> tuple[str, str]:
    """Parse one strictly scoped ``working_memory:ID`` selection."""

    if not isinstance(value, str) or ":" not in value:
        raise RepairError("--select must use working_memory:ID")
    table, memory_id = value.split(":", 1)
    if table != _SELECTION_TABLE or not memory_id or "\x00" in memory_id:
        raise RepairError("--select must use a non-empty working_memory:ID")
    return table, memory_id


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


_MAX_MANIFEST_BYTES = 1_000_000


@dataclass(frozen=True)
class _RepairManifest:
    bank_name: str
    fingerprint: dict[str, Any]
    database_identity: dict[str, int]


@dataclass
class _BoundDatabase:
    """A private hardlink and retained FDs for the authorized database inode."""

    stage_path: Path
    source_fd: int
    parent_fd: int
    stage_dir_fd: int
    stage_name: str
    identity: dict[str, int]


@dataclass
class _AnchoredDatabase:
    """Original parent/source FDs retained across the final binding gate."""

    parent_fd: int
    source_fd: int
    name: str
    identity: dict[str, int]


@dataclass
class _ReservedBackup:
    """An exclusively created backup inode anchored to its opened parent FD."""

    parent_fd: int
    name: str
    descriptor: int
    identity: dict[str, int]
    display_path: Path
    installed: bool = False

    def __fspath__(self) -> str:
        """Support test-only SQLite verification without reopening in repair code."""

        return os.fspath(self.display_path)

    def is_file(self) -> bool:
        try:
            return stat.S_ISREG(os.fstat(self.descriptor).st_mode)
        except OSError:
            return False


def _load_manifest(report_path: str | Path) -> _RepairManifest:
    """Read one bounded regular-file manifest snapshot without following links."""

    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        # A FIFO must be rejected from its FD metadata, not block the CLI while
        # waiting for a writer.
        flags |= os.O_NONBLOCK
    try:
        descriptor = os.open(Path(report_path), flags)
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_MANIFEST_BYTES:
                raise RepairError("Repair report could not be safely read")
            chunks: list[bytes] = []
            remaining = _MAX_MANIFEST_BYTES + 1
            while remaining:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(descriptor)
        if len(raw) > _MAX_MANIFEST_BYTES:
            raise RepairError("Repair report could not be safely read")
        payload = json.loads(raw.decode("utf-8"))
    except RepairError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RepairError("Repair report could not be parsed") from error
    if not isinstance(payload, dict):
        raise RepairError("Repair report has no valid authorization manifest")
    bank_name = payload.get("bank_name")
    fingerprint = payload.get("schema_fingerprint")
    identity = payload.get("database_identity")
    if (
        not isinstance(bank_name, str)
        or not bank_name
        or not isinstance(fingerprint, dict)
        or not isinstance(identity, dict)
        or any(
            not isinstance(identity.get(key), int) or isinstance(identity.get(key), bool)
            for key in ("st_dev", "st_ino")
        )
    ):
        raise RepairError("Repair report has no valid authorization manifest")
    try:
        _canonical_json(fingerprint)
    except (TypeError, ValueError) as error:
        raise RepairError("Repair report has no valid authorization manifest") from error
    return _RepairManifest(bank_name, fingerprint, {"st_dev": identity["st_dev"], "st_ino": identity["st_ino"]})


def _verifiable_fingerprint(fingerprint: dict[str, Any]) -> bool:
    """Reject incomplete or degraded schema snapshots instead of trusting them."""

    if (
        fingerprint.get("status") != "ok"
        or fingerprint.get("tables_truncated") is not False
        or fingerprint.get("triggers_truncated") is not False
    ):
        return False
    tables = fingerprint.get("tables")
    triggers = fingerprint.get("triggers")
    if not isinstance(tables, list) or not isinstance(triggers, list):
        return False
    return all(
        isinstance(table, dict)
        and table.get("status") == "ok"
        and table.get("columns_truncated") is False
        and table.get("foreign_keys_truncated") is False
        for table in tables
    ) and all(
        isinstance(trigger, dict)
        and isinstance(trigger.get("name"), str)
        and isinstance(trigger.get("table_name"), str)
        and isinstance(trigger.get("sql_digest"), str)
        for trigger in triggers
    )


def _database_identity(path: Path) -> dict[str, int]:
    """Read one regular database identity without following pathname links."""

    descriptor: int | None = None
    parent_fd: int | None = None
    try:
        parent_fd = _open_safe_directory_fd(path.parent)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
        info = os.fstat(descriptor)
    except (OSError, ValueError) as error:
        raise RepairError("Database identity could not be verified") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)
    if not stat.S_ISREG(info.st_mode):
        raise RepairError("Database identity could not be verified")
    return _stat_identity(info)


def _verify_report_gate(
    manifest: _RepairManifest,
    db_path: Path,
    bank_name: str,
    *,
    conn: sqlite3.Connection | None = None,
    bound_identity: dict[str, int] | None = None,
) -> None:
    """Fail closed unless this exact bank, file, and schema still match."""

    if manifest.bank_name != bank_name:
        raise RepairError("Repair report bank does not match the selected database")
    # Before binding, the requested path is the only available identity.  Once
    # a private hardlink has been made, *never* re-stat that attacker-mutable
    # path while using the locked connection: the connection is authorized by
    # the inode captured in ``bound_identity`` instead.
    if conn is None:
        current_identity = _database_identity(db_path)
    elif bound_identity is None:
        raise RepairError("Database identity could not be verified")
    else:
        current_identity = bound_identity
    if manifest.database_identity != current_identity:
        raise RepairError("Repair report does not authorize the selected database")
    expected_fingerprint = manifest.fingerprint
    if not _verifiable_fingerprint(expected_fingerprint):
        raise RepairError("Repair report fingerprint is incomplete and cannot authorize a repair")
    try:
        current_fingerprint = (
            asdict(inspect_schema_fingerprint(conn))
            if conn is not None
            else build_doctor_report(bank_name, db_path).to_dict()["schema_fingerprint"]
        )
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError) as error:
        raise RepairError("Current database fingerprint could not be verified") from error
    if not isinstance(current_fingerprint, dict) or not _verifiable_fingerprint(current_fingerprint):
        raise RepairError("Current database fingerprint could not be verified")
    if _canonical_json(expected_fingerprint) != _canonical_json(current_fingerprint):
        raise RepairError("Repair report fingerprint does not match the current database")


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
    except sqlite3.Error:
        return set()


def _working_row(conn: sqlite3.Connection, memory_id: str) -> tuple[sqlite3.Row | None, str | None]:
    """Read the one selected active row only; malformed time data is inactive."""

    columns = _table_columns(conn, _SELECTION_TABLE)
    if not {"id", "valid_until"}.issubset(columns):
        return None, "unloadable"
    superseded = ", superseded_by" if "superseded_by" in columns else ""
    try:
        row = conn.execute(
            "SELECT rowid, valid_until" + superseded + ", "
            "CASE WHEN valid_until IS NULL OR julianday(valid_until) > julianday('now') "
            "THEN 1 ELSE 0 END AS active "
            "FROM working_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
    except sqlite3.Error:
        return None, "unloadable"
    if row is None:
        return None, "missing"
    if "superseded_by" in row.keys() and row["superseded_by"] is not None:
        return None, "already_inactive"
    if not bool(row["active"]):
        return None, "already_inactive"
    return row, None


def _confirmed_vec_working_kind(conn: sqlite3.Connection) -> tuple[str, int] | None:
    """Confirm a live *virtual* vec0 table and its declared write contract."""

    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'vec_working'"
        ).fetchone()
        if row is None or not isinstance(row[0], str) or not _VEC0_VIRTUAL_TABLE.search(row[0]):
            return None
        kind = _VECTOR_KIND.search(row[0])
        if kind is None or int(kind.group(2)) < 1:
            return None
        # A normal table with forged sqlite_master DDL is not a vec0 contract.
        table_list = conn.execute("PRAGMA table_list").fetchall()
        if not any(str(item[1]) == "vec_working" and str(item[2]) == "virtual" for item in table_list):
            return None
        conn.execute('SELECT rowid, embedding FROM "vec_working" LIMIT 0').fetchone()
        return kind.group(1).lower(), int(kind.group(2))
    except (sqlite3.Error, ValueError, IndexError):
        return None


def _decode_embedding(value: Any) -> str | None:
    """Validate and normalize fallback JSON without exposing it to callers."""

    if not isinstance(value, str):
        return None
    try:
        raw = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(raw, list) or not raw:
        return None
    values: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        values.append(number)
    magnitude = math.sqrt(sum(item * item for item in values))
    if magnitude:
        values = [item / magnitude for item in values]
    return json.dumps(values, separators=(",", ":"))


def _backfill_plan(conn: sqlite3.Connection, memory_id: str) -> tuple[dict[str, Any] | None, str | None]:
    row, reason = _working_row(conn, memory_id)
    if reason or row is None:
        return None, reason
    vec_contract = _confirmed_vec_working_kind(conn)
    if vec_contract is None:
        return None, "unloadable"
    vector_kind, vector_dimension = vec_contract
    if not {"memory_id", "embedding_json"}.issubset(_table_columns(conn, "memory_embeddings")):
        return None, "unloadable"
    try:
        exists = conn.execute(
            "SELECT 1 FROM vec_working WHERE rowid = ? LIMIT 1", (int(row["rowid"]),)
        ).fetchone()
        if exists is not None:
            return None, "already_present"
        embedding_row = conn.execute(
            "SELECT embedding_json FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        ).fetchone()
    except sqlite3.Error:
        return None, "unloadable"
    if embedding_row is None:
        return None, "missing_fallback_embedding"
    embedding = _decode_embedding(embedding_row[0])
    if embedding is None:
        return None, "unloadable"
    try:
        if len(json.loads(embedding)) != vector_dimension:
            return None, "unloadable"
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, "unloadable"
    return {
        "rowid": int(row["rowid"]),
        "embedding": embedding,
        "vector_kind": vector_kind,
        "vector_dimension": vector_dimension,
    }, None


def _expire_plan(conn: sqlite3.Connection, memory_id: str) -> tuple[dict[str, Any] | None, str | None]:
    row, reason = _working_row(conn, memory_id)
    if reason or row is None:
        return None, reason
    return {"rowid": int(row["rowid"])}, None


def _plan_selection(conn: sqlite3.Connection, action: str, memory_id: str) -> tuple[dict[str, Any] | None, str | None]:
    if action == BACKFILL_VEC_WORKING:
        return _backfill_plan(conn, memory_id)
    return _expire_plan(conn, memory_id)


def _open_safe_directory_fd(path: Path) -> int:
    """Open every directory component without resolving or following a link."""

    flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    absolute = os.path.abspath(os.fspath(path))
    descriptor = os.open(os.sep, flags)
    try:
        for component in (part for part in absolute.split(os.sep) if part):
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            try:
                if not stat.S_ISDIR(os.fstat(next_descriptor).st_mode):
                    raise NotADirectoryError(component)
            except BaseException:
                os.close(next_descriptor)
                raise
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _reject_original_sidecars(anchored: _AnchoredDatabase) -> None:
    """Reject original-name SQLite sidecars through a verified parent FD.

    A hardlink has a different basename, so SQLite cannot safely recover an
    original rollback journal (or WAL/SHM pair) through the private pathname.
    Recheck the retained source inode before inspecting its siblings; no public
    pathname is used after this point.
    """

    try:
        source_info = os.fstat(anchored.source_fd)
        if not stat.S_ISREG(source_info.st_mode) or _stat_identity(source_info) != anchored.identity:
            raise RepairError("Repair report does not authorize the selected database")
        for suffix in ("-journal", "-wal", "-shm"):
            try:
                os.stat(anchored.name + suffix, dir_fd=anchored.parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise RepairError("Database sidecars prevent a safe repair")
    except RepairError:
        raise
    except (OSError, ValueError) as error:
        raise RepairError("Database sidecars could not be safely inspected") from error


def _backup_is_owned(backup: _ReservedBackup) -> bool:
    """Verify the final name still addresses the exclusively reserved inode."""

    try:
        actual = os.stat(backup.name, dir_fd=backup.parent_fd, follow_symlinks=False)
        descriptor = os.fstat(backup.descriptor)
    except OSError:
        return False
    return (
        stat.S_ISREG(actual.st_mode)
        and _stat_identity(actual) == backup.identity
        and stat.S_ISREG(descriptor.st_mode)
        and _stat_identity(descriptor) == backup.identity
    )


def _reserve_backup_destination(
    db_path: Path, requested: str | Path | None
) -> _ReservedBackup:
    """Exclusively reserve a non-aliased backup inode through its parent FD."""

    backup = (
        Path(requested).expanduser()
        if requested is not None
        else db_path.with_name(f"{db_path.stem}.repair-backup-{uuid.uuid4().hex}.sqlite")
    )
    if os.path.abspath(os.fspath(backup)) == os.path.abspath(os.fspath(db_path)):
        raise RepairError("Backup path must not be the inspected database")
    if backup.name in {"", ".", ".."}:
        raise RepairError("Backup path could not be safely inspected")
    parent_fd: int | None = None
    descriptor: int | None = None
    try:
        parent_fd = _open_safe_directory_fd(backup.parent)
        try:
            existing = os.stat(backup.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if stat.S_ISLNK(existing.st_mode):
                raise RepairError("Backup path must not use a symlink")
            raise RepairError("Backup path must be a new file and must not overwrite an existing path")
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        descriptor = os.open(backup.name, flags, 0o600, dir_fd=parent_fd)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RepairError("Backup could not be safely created")
        reservation = _ReservedBackup(parent_fd, backup.name, descriptor, _stat_identity(info), backup)
        parent_fd = None
        descriptor = None
        return reservation
    except RepairError:
        raise
    except FileNotFoundError as error:
        raise RepairError("Backup directory does not exist") from error
    except (OSError, ValueError) as error:
        raise RepairError("Backup path could not be safely inspected") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def _release_backup_destination(backup: _ReservedBackup, *, keep: bool) -> None:
    """Close retained FDs and remove only our verified reservation on failure."""

    try:
        if not keep and _backup_is_owned(backup):
            os.unlink(backup.name, dir_fd=backup.parent_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(backup.descriptor)
        except OSError:
            pass
        try:
            os.close(backup.parent_fd)
        except OSError:
            pass


def _open_anchored_database(db_path: Path, identity: dict[str, int]) -> _AnchoredDatabase:
    """Retain verified original parent/source FDs until binding consumes them."""

    source_fd: int | None = None
    parent_fd: int | None = None
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        parent_fd = _open_safe_directory_fd(db_path.parent)
        source_fd = os.open(db_path.name, flags, dir_fd=parent_fd)
        info = os.fstat(source_fd)
        if not stat.S_ISREG(info.st_mode) or _stat_identity(info) != identity:
            raise RepairError("Repair report does not authorize the selected database")
        anchored = _AnchoredDatabase(parent_fd, source_fd, db_path.name, identity)
        parent_fd = None
        source_fd = None
        return anchored
    except RepairError:
        raise
    except (OSError, ValueError) as error:
        raise RepairError("Database identity could not be verified") from error
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if parent_fd is not None:
            os.close(parent_fd)



def _release_anchored_database(anchored: _AnchoredDatabase) -> None:
    """Close original FDs unless successful binding transferred ownership."""

    for descriptor_name in ("source_fd", "parent_fd"):
        descriptor = getattr(anchored, descriptor_name)
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
            setattr(anchored, descriptor_name, -1)


_AT_EMPTY_PATH = 0x1000


def _link_open_fd(source_fd: int, directory_fd: int, name: str) -> None:
    """Hardlink an already-open Linux file descriptor into a private directory.

    ``/proc/self/fd/N`` is not a safe substitute: on common Linux systems it
    crosses the procfs mount and ``link(2)`` fails with EXDEV.  ``linkat`` with
    ``AT_EMPTY_PATH`` names the actual open inode, not a pathname that can be
    swapped after it was checked.
    """

    try:
        linkat = ctypes.CDLL(None, use_errno=True).linkat
    except (AttributeError, OSError) as error:
        raise OSError("linkat(AT_EMPTY_PATH) is unavailable") from error
    linkat.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    linkat.restype = ctypes.c_int
    if linkat(source_fd, b"", directory_fd, os.fsencode(name), _AT_EMPTY_PATH) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _stat_identity(info: os.stat_result) -> dict[str, int]:
    return {"st_dev": int(info.st_dev), "st_ino": int(info.st_ino)}


def _remove_private_binding(bound: _BoundDatabase) -> None:
    """Close and remove only files owned through retained private directory FDs."""

    try:
        os.close(bound.source_fd)
    except OSError:
        pass
    for name in ("database.sqlite", "database.sqlite-journal", "database.sqlite-wal", "database.sqlite-shm"):
        try:
            info = os.stat(name, dir_fd=bound.stage_dir_fd, follow_symlinks=False)
            if stat.S_ISREG(info.st_mode):
                os.unlink(name, dir_fd=bound.stage_dir_fd)
        except OSError:
            pass
    try:
        stage_info = os.fstat(bound.stage_dir_fd)
        named_info = os.stat(bound.stage_name, dir_fd=bound.parent_fd, follow_symlinks=False)
        if stat.S_ISDIR(named_info.st_mode) and _stat_identity(named_info) == _stat_identity(stage_info):
            os.rmdir(bound.stage_name, dir_fd=bound.parent_fd)
    except OSError:
        pass
    finally:
        for descriptor in (bound.stage_dir_fd, bound.parent_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _discard_private_stage(parent_fd: int, stage_name: str, stage_dir_fd: int | None) -> None:
    """Best-effort cleanup for a binding that failed before ownership transfer."""

    if stage_dir_fd is not None:
        for name in ("database.sqlite", "database.sqlite-journal", "database.sqlite-wal", "database.sqlite-shm"):
            try:
                info = os.stat(name, dir_fd=stage_dir_fd, follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    os.unlink(name, dir_fd=stage_dir_fd)
            except OSError:
                pass
        try:
            stage_info = os.fstat(stage_dir_fd)
            named_info = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISDIR(named_info.st_mode) and _stat_identity(named_info) == _stat_identity(stage_info):
                os.rmdir(stage_name, dir_fd=parent_fd)
        except OSError:
            pass
        return
    try:
        info = os.stat(stage_name, dir_fd=parent_fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            os.rmdir(stage_name, dir_fd=parent_fd)
    except OSError:
        pass


def _bind_authorized_database(anchored: _AnchoredDatabase) -> _BoundDatabase:
    """Link retained authorized FDs only after the final sidecar recheck."""

    directory_fd: int | None = None
    bound: _BoundDatabase | None = None
    stage_name = f".mnemosyne-repair-{uuid.uuid4().hex}"
    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
    try:
        source_info = os.fstat(anchored.source_fd)
        if not stat.S_ISREG(source_info.st_mode) or _stat_identity(source_info) != anchored.identity:
            raise RepairError("Repair report does not authorize the selected database")
        os.mkdir(stage_name, 0o700, dir_fd=anchored.parent_fd)
        directory_fd = os.open(stage_name, directory_flags, dir_fd=anchored.parent_fd)
        stage_info = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(stage_info.st_mode)
            or stat.S_IMODE(stage_info.st_mode) != 0o700
            or stage_info.st_dev != source_info.st_dev
        ):
            raise RepairError("Database could not be safely bound for repair")
        # Give SQLite a path anchored to this retained private directory FD,
        # not its mutable parent-directory spelling.
        stage_path = Path("/proc/self/fd") / str(directory_fd) / "database.sqlite"
        # This is deliberately the final original-name operation: no public DB
        # path is reopened between this dirfd recheck and linkat(AT_EMPTY_PATH).
        _reject_original_sidecars(anchored)
        _link_open_fd(anchored.source_fd, directory_fd, stage_path.name)
        linked_info = os.stat(stage_path.name, dir_fd=directory_fd, follow_symlinks=False)
        source_after_link = os.fstat(anchored.source_fd)
        if (
            not stat.S_ISREG(linked_info.st_mode)
            or _stat_identity(linked_info) != anchored.identity
            or _stat_identity(source_after_link) != anchored.identity
            or linked_info.st_nlink < 2
        ):
            raise RepairError("Database could not be safely bound for repair")
        bound = _BoundDatabase(
            stage_path,
            anchored.source_fd,
            anchored.parent_fd,
            directory_fd,
            stage_name,
            anchored.identity,
        )
        anchored.source_fd = -1
        anchored.parent_fd = -1
        directory_fd = None
        return bound
    except RepairError:
        raise
    except (OSError, ValueError) as error:
        raise RepairError("Database could not be safely bound for repair") from error
    finally:
        if bound is None:
            if anchored.parent_fd >= 0:
                _discard_private_stage(anchored.parent_fd, stage_name, directory_fd)
            _release_anchored_database(anchored)
        if directory_fd is not None:
            try:
                os.close(directory_fd)
            except OSError:
                pass


def _private_connection_is_safe(conn: sqlite3.Connection) -> bool:
    """A private hardlink cannot safely replay another pathname's WAL file."""

    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        return row is not None and str(row[0]).lower() != "wal"
    except sqlite3.Error:
        return False


def _has_working_memory_trigger(conn: sqlite3.Connection) -> bool:
    """Do not execute selected-row UPDATEs when table triggers already exist."""

    try:
        return (
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'trigger' AND tbl_name = ? COLLATE NOCASE LIMIT 1",
                (_SELECTION_TABLE,),
            ).fetchone()
            is not None
        )
    except sqlite3.Error as error:
        raise RepairError("working_memory trigger safety could not be verified") from error


def _create_validated_backup(source_fd: int, backup: _ReservedBackup) -> None:
    """Populate the exclusively reserved destination through its retained FD.

    SQLite never opens the caller's destination pathname. The destination FD
    was created with ``O_CREAT|O_EXCL|O_NOFOLLOW`` through its retained parent
    directory FD, so a later parent-directory rename cannot redirect the copy.

    The caller holds ``BEGIN IMMEDIATE`` and supplies an already verified file
    descriptor for the authorized database. Backing up from that same write
    transaction deadlocks CPython's SQLite backup API, so the snapshot uses a
    separate read-only connection through that FD while the write lock prevents
    another writer from changing the planned database.
    """

    source: sqlite3.Connection | None = None
    destination: sqlite3.Connection | None = None
    try:
        destination_info = os.fstat(backup.descriptor)
        if not stat.S_ISREG(destination_info.st_mode) or _stat_identity(destination_info) != backup.identity:
            raise RepairError("Backup could not be safely created")
        # /proc/self/fd duplicates this exact open file description on Linux,
        # keeping SQLite's destination bound to the reserved inode.
        destination = sqlite3.connect(f"file:/proc/self/fd/{backup.descriptor}?mode=rw", uri=True)
        source = sqlite3.connect(f"file:/proc/self/fd/{source_fd}?mode=ro", uri=True)
        source.backup(destination)
        row = destination.execute("PRAGMA quick_check").fetchone()
        if row is None or str(row[0]).lower() != "ok":
            raise RepairError("Backup validation failed")
        destination.close()
        destination = None
        os.fsync(backup.descriptor)
        if not _backup_is_owned(backup):
            raise RepairError("Backup could not be safely created")
        backup.installed = True

    except RepairError:
        raise
    except (OSError, sqlite3.Error) as error:
        raise RepairError("Backup could not be created or validated") from error
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()


def _result(status: str, reason: str | None = None) -> dict[str, str]:
    """Use a fixed content-free result schema; never echo selected IDs/paths."""

    result = {"table": _SELECTION_TABLE, "status": status}
    if reason is not None:
        result["reason"] = reason
    return result


def _read_only_plans(db_path: Path, action: str, selections: list[tuple[str, str]]) -> list[dict[str, str]]:
    try:
        conn = open_readonly_doctor_db(db_path)
    except (OSError, ValueError, sqlite3.Error) as error:
        raise RepairError("Database could not be safely opened for repair") from error
    try:
        results = []
        for _table, memory_id in selections:
            _plan, reason = _plan_selection(conn, action, memory_id)
            results.append(_result("planned" if reason is None else "skipped", reason))
        return results
    except sqlite3.Error as error:
        raise RepairError("Selected rows could not be safely read") from error
    finally:
        conn.close()


def _open_writable_repair_db(database: Path) -> sqlite3.Connection:
    """Open only an existing DB, mapping filesystem/SQLite failures to RepairError."""

    try:
        conn = sqlite3.connect(f"{database.absolute().as_uri()}?mode=rw", uri=True, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn
    except (OSError, ValueError, sqlite3.Error) as error:
        raise RepairError("Database could not be safely opened for repair") from error


def _insert_vec_working(conn: sqlite3.Connection, plan: dict[str, Any]) -> None:
    """Use the existing fallback embedding with the vec0 encoding declared in DDL."""

    vector_kind = plan["vector_kind"]
    if vector_kind == "int8":
        conn.execute(
            "INSERT INTO vec_working(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
            (plan["rowid"], plan["embedding"]),
        )
    elif vector_kind == "bit":
        conn.execute(
            "INSERT INTO vec_working(rowid, embedding) VALUES (?, vec_quantize_binary(?))",
            (plan["rowid"], plan["embedding"]),
        )
    else:
        conn.execute(
            "INSERT INTO vec_working(rowid, embedding) VALUES (?, ?)",
            (plan["rowid"], plan["embedding"]),
        )


def _preflight_vec_write(conn: sqlite3.Connection, plan: dict[str, Any]) -> bool:
    """Exercise the exact vec0 write in a rolled-back savepoint before backup."""

    try:
        conn.execute("SAVEPOINT repair_vec_write_preflight")
        _insert_vec_working(conn, plan)
        present = conn.execute(
            "SELECT 1 FROM vec_working WHERE rowid = ? LIMIT 1", (plan["rowid"],)
        ).fetchone()
        conn.execute("ROLLBACK TO repair_vec_write_preflight")
        conn.execute("RELEASE repair_vec_write_preflight")
        return present is not None
    except sqlite3.Error:
        try:
            conn.execute("ROLLBACK TO repair_vec_write_preflight")
            conn.execute("RELEASE repair_vec_write_preflight")
        except sqlite3.Error:
            pass
        return False


def run_repair(
    *,
    db_path: str | Path,
    bank_name: str,
    report_path: str | Path,
    selections: list[str],
    action: str = BACKFILL_VEC_WORKING,
    apply: bool = False,
    backup_path: str | Path | None = None,
) -> dict[str, Any]:
    """Plan or apply only individually selected working-memory repairs.

    The Doctor manifest is validated even for dry-run so it cannot become a
    misleading plan preview.  Apply opens a write lock, re-reads every selected
    condition under that lock, and creates a validated backup only if at least
    one selected row remains actionable.
    """

    if sys.platform != "linux":
        raise RepairError("Repair is supported only on Linux")
    if action not in _ALLOWED_ACTIONS:
        raise RepairError("--action must be backfill-vec-working or expire")
    if not selections:
        raise RepairError("At least one complete --select working_memory:ID is required")
    parsed: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for value in selections:
        selection = parse_selection(value)
        if selection not in seen:
            parsed.append(selection)
            seen.add(selection)
    database = Path(db_path).expanduser()
    try:
        _database_identity(database)
    except RepairError:
        raise RepairError("Database not found")
    manifest = _load_manifest(report_path)
    anchored: _AnchoredDatabase | None = None
    if apply:
        # Keep this exact original parent/source pair live while Doctor performs
        # its permitted preflight reopen.  Binding later consumes the same FDs.
        anchored = _open_anchored_database(database, manifest.database_identity)
        try:
            _reject_original_sidecars(anchored)
            _verify_report_gate(manifest, database, bank_name)
        except BaseException:
            _release_anchored_database(anchored)
            raise
    else:
        _verify_report_gate(manifest, database, bank_name)
    base: dict[str, Any] = {
        "mode": "apply" if apply else "dry_run",
        "action": action,
        "backup": False,
        "applied": [],
        "skipped": [],
    }
    if not apply:
        for item in _read_only_plans(database, action, parsed):
            (base["applied"] if item["status"] == "planned" else base["skipped"]).append(item)
        return base

    if anchored is None:  # pragma: no cover - apply establishes it above.
        raise RepairError("Database could not be safely bound for repair")
    bound = _bind_authorized_database(anchored)
    conn: sqlite3.Connection | None = None
    backup: _ReservedBackup | None = None
    try:
        conn = _open_writable_repair_db(bound.stage_path)
        if not _private_connection_is_safe(conn):
            raise RepairError("Database journal mode cannot be safely bound for repair")
        conn.execute("BEGIN IMMEDIATE")
        _verify_report_gate(
            manifest,
            database,
            bank_name,
            conn=conn,
            bound_identity=bound.identity,
        )
        if _has_working_memory_trigger(conn):
            raise RepairError("working_memory triggers prevent a safe repair")
        plans: list[dict[str, Any]] = []
        for _table, memory_id in parsed:
            plan, reason = _plan_selection(conn, action, memory_id)
            if reason is None and plan is not None:
                plans.append(plan)
            else:
                base["skipped"].append(_result("skipped", reason or "unloadable"))
        if not plans:
            conn.rollback()
            return base
        if action == BACKFILL_VEC_WORKING and not all(_preflight_vec_write(conn, plan) for plan in plans):
            conn.rollback()
            return {
                **base,
                "skipped": base["skipped"] + [_result("skipped", "unloadable") for _ in plans],
            }

        backup = _reserve_backup_destination(database, backup_path)
        _create_validated_backup(bound.source_fd, backup)
        base["backup"] = True

        for plan in plans:
            try:
                if action == BACKFILL_VEC_WORKING:
                    _insert_vec_working(conn, plan)
                    present = conn.execute(
                        "SELECT 1 FROM vec_working WHERE rowid = ? LIMIT 1", (plan["rowid"],)
                    ).fetchone()
                    if present is None:
                        raise sqlite3.DatabaseError("postcheck")
                else:
                    updated = conn.execute(
                        "UPDATE working_memory SET valid_until = CURRENT_TIMESTAMP "
                        "WHERE rowid = ? AND (valid_until IS NULL OR julianday(valid_until) > julianday('now'))",
                        (plan["rowid"],),
                    )
                    if updated.rowcount != 1:
                        raise sqlite3.DatabaseError("stale_selected_row")
                    active = conn.execute(
                        "SELECT CASE WHEN valid_until IS NULL OR julianday(valid_until) > julianday('now') "
                        "THEN 1 ELSE 0 END FROM working_memory WHERE rowid = ?",
                        (plan["rowid"],),
                    ).fetchone()
                    if active is None or bool(active[0]):
                        raise sqlite3.DatabaseError("postcheck")
                base["applied"].append(_result("applied"))
            except sqlite3.Error:
                # Do not continue after an uncertain selected mutation. Roll
                # back the narrow transaction and keep errors content-free.
                conn.rollback()
                raise RepairError("Selected repair could not be applied safely")
        conn.commit()
        return base
    except RepairError:
        if conn is not None and conn.in_transaction:
            conn.rollback()
        raise
    except sqlite3.Error as error:
        if conn is not None and conn.in_transaction:
            conn.rollback()
        raise RepairError("Selected rows could not be safely locked or re-read") from error
    finally:
        if conn is not None:
            conn.close()
        if backup is not None:
            _release_backup_destination(backup, keep=backup.installed)
        _remove_private_binding(bound)


def render_repair_json(result: dict[str, Any]) -> str:
    """Render the fixed-schema, content-free structured repair result."""

    return json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
