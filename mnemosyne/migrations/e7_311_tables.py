"""
Mnemosyne E7 Migration — 3.11.1 schema additions
===============================================

Adds ``memory_events`` and ``sync_meta`` when absent, and backfills
``memory_events.device_id`` before creating its index when the table
already exists. The table definitions below are a 3.11-era snapshot,
not the current full SyncEngine schema. Historical event rows and other
legacy sync fields are not converted.

Safe to re-run: missing tables, columns and indices are checked before
DDL. Index and unrelated column DDL failures propagate; earlier DDL may
remain applied after a later failure, so retry completes only the missing
steps. The migration does not delete data or drop tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Literal, TypedDict, Union, overload


# 3.11-era table definition; not a copy of the current SyncEngine schema.
_MEMORY_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    operation TEXT NOT NULL CHECK(operation IN ('CREATE','UPDATE','DELETE','CONSOLIDATE')),
    timestamp TEXT NOT NULL,
    device_id TEXT NOT NULL,
    payload TEXT,
    parent_event_ids TEXT DEFAULT '[]',
    importance REAL DEFAULT 0.5,
    expiry TEXT,
    event_hash TEXT,
    synced_at TEXT
)
""".strip()

_SYNC_META_DDL = """
CREATE TABLE IF NOT EXISTS sync_meta (
    key TEXT PRIMARY KEY,
    value TEXT
)
""".strip()

# Indices created by this migration; SyncEngine may create additional ones.
_MEMORY_EVENTS_INDICES = [
    (
        "idx_me_timestamp",
        "CREATE INDEX IF NOT EXISTS idx_me_timestamp ON memory_events(timestamp)",
    ),
    (
        "idx_me_memory_id",
        "CREATE INDEX IF NOT EXISTS idx_me_memory_id ON memory_events(memory_id)",
    ),
    (
        "idx_me_device_id",
        "CREATE INDEX IF NOT EXISTS idx_me_device_id ON memory_events(device_id)",
    ),
]

_MEMORY_EVENTS_LEGACY_COLUMNS = [
    ("device_id", "device_id TEXT NOT NULL DEFAULT ''"),
]


# The new tables this migration adds (in 3.11.1).
NEW_TABLES = ("memory_events", "sync_meta")
_TABLES = (
    ("memory_events", _MEMORY_EVENTS_DDL),
    ("sync_meta", _SYNC_META_DDL),
)


class MigrationReport(TypedDict):
    added: int
    tables_added: list[str]
    tables_already_present: list[str]
    columns_added: list[str]
    indices_added: int


class MigrationDryRunReport(MigrationReport):
    # Report-only (dry-run) fields; present in every dry-run report.
    would_add: int
    tables_would_add: list[str]
    columns_would_add: list[str]
    indices_would_add: int


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cursor.fetchone() is not None


def _has_index(conn: sqlite3.Connection, name: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
        (name,),
    )
    return cursor.fetchone() is not None


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())


def _missing_memory_events_columns(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    if not _has_table(conn, "memory_events"):
        return []
    return [
        (name, ddl)
        for name, ddl in _MEMORY_EVENTS_LEGACY_COLUMNS
        if not _has_column(conn, "memory_events", name)
    ]


@overload
def migrate_311_tables(db_path: Path, dry_run: Literal[True]) -> MigrationDryRunReport:
    ...


@overload
def migrate_311_tables(db_path: Path, dry_run: Literal[False] = False) -> MigrationReport:
    ...


@overload
def migrate_311_tables(
    db_path: Path, dry_run: bool = False
) -> Union[MigrationReport, MigrationDryRunReport]:
    ...


def migrate_311_tables(
    db_path: Path, dry_run: bool = False
) -> Union[MigrationReport, MigrationDryRunReport]:
    """Add the 3.11.1 schema tables to an existing bank at the older
    54-table schema. Idempotent.

    With ``dry_run=True`` the database is opened read-only
    (``mode=ro`` + ``PRAGMA query_only=ON``), no DDL is executed and no
    commit happens. The report keeps ``added`` / ``tables_added`` /
    ``indices_added`` at zero and instead exposes ``would_add`` /
    ``tables_would_add`` / ``columns_would_add`` /
    ``indices_would_add`` for the pending DDL (all zero/empty when
    the bank does not exist yet).

    Returns a report dict with:
      - added: int (number of tables added in this call)
      - tables_added: List[str] (names of tables added in this call)
      - tables_already_present: List[str] (names already in the schema)
      - columns_added: List[str] (legacy columns added in this call)
      - indices_added: int (number of indices added in this call)
      - dry-run reports additionally carry would_add /
        tables_would_add / columns_would_add / indices_would_add
        describing the DDL a real run would execute.
    """
    db_path = Path(db_path)
    if dry_run:
        dry_report: MigrationDryRunReport = {
            "added": 0,
            "tables_added": [],
            "tables_already_present": [],
            "columns_added": [],
            "indices_added": 0,
            "would_add": 0,
            "tables_would_add": [],
            "columns_would_add": [],
            "indices_would_add": 0,
        }
        if not db_path.exists():
            return dry_report

        conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            conn.execute("PRAGMA query_only=ON")
            for name, _ddl in _TABLES:
                if _has_table(conn, name):
                    dry_report["tables_already_present"].append(name)
                else:
                    dry_report["tables_would_add"].append(name)
                    dry_report["would_add"] += 1
            for name, _ddl in _missing_memory_events_columns(conn):
                dry_report["columns_would_add"].append(f"memory_events.{name}")
            for index_name, _index_ddl in _MEMORY_EVENTS_INDICES:
                if not _has_index(conn, index_name):
                    dry_report["indices_would_add"] += 1
        finally:
            conn.close()
        return dry_report

    applied_report: MigrationReport = {
        "added": 0,
        "tables_added": [],
        "tables_already_present": [],
        "columns_added": [],
        "indices_added": 0,
    }
    if not db_path.exists():
        return applied_report

    conn = sqlite3.connect(str(db_path))
    try:
        for name, ddl in _TABLES:
            if _has_table(conn, name):
                applied_report["tables_already_present"].append(name)
                continue
            conn.execute(ddl)
            applied_report["tables_added"].append(name)
            applied_report["added"] += 1

        for name, ddl in _missing_memory_events_columns(conn):
            try:
                conn.execute(f"ALTER TABLE memory_events ADD COLUMN {ddl}")
            except sqlite3.OperationalError as exc:
                if str(exc).lower() != f"duplicate column name: {name}".lower():
                    raise
                # A concurrent migrator may have added the column after our
                # schema read. Only accept the exact declaration we add here.
                matches = [
                    row for row in conn.execute("PRAGMA table_info(memory_events)")
                    if row[1] == name
                ]
                if len(matches) != 1 or not (
                    matches[0][2].strip().upper() == "TEXT"
                    and matches[0][3] == 1
                    and matches[0][4] == "''"
                ):
                    raise
                continue
            applied_report["columns_added"].append(f"memory_events.{name}")

        # Indices. Let DDL errors propagate so a successful migration means the
        # selected runtime can open the resulting schema.
        for index_name, index_ddl in _MEMORY_EVENTS_INDICES:
            if _has_index(conn, index_name):
                continue
            conn.execute(index_ddl)
            applied_report["indices_added"] += 1
        conn.commit()
    finally:
        conn.close()
    return applied_report
