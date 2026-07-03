"""
Mnemosyne E7 Migration — 3.11.1 schema additions
===============================================

Idempotent migration that adds the two tables introduced in
mnemosyne-memory 3.11.1 to an existing bank at the older 54-table
schema:

  - memory_events  (created by SyncManager._init_events_table)
  - sync_meta       (created by SyncManager._init_events_table)

The DDL is copied verbatim from the canonical source at
``mnemosyne/core/sync.py:641-665`` (SyncManager._init_events_table)
and the indices that method creates (``sync.py:668-672``). We do
NOT invent DDL here; if the upstream source changes its DDL, this
migration should be updated to match.

Safe to re-run (idempotent — uses ``CREATE TABLE IF NOT EXISTS``
and try/except for indices, matching the upstream behavior). The
migration does not delete any data or drop any tables.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TypedDict


# Canonical DDL from mnemosyne/core/sync.py:641-665
# (SyncManager._init_events_table). If the upstream source changes
# its DDL, update this migration to match.
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

# Canonical index DDL from mnemosyne/core/sync.py:668-672
# (try/except because IF NOT EXISTS for indices is not supported
# in all SQLite versions, matching the upstream behavior).
_MEMORY_EVENTS_INDICES = [
    (
        "idx_me_timestamp",
        "CREATE INDEX IF NOT EXISTS idx_me_timestamp "
        "ON memory_events(timestamp)",
    ),
    (
        "idx_me_memory_id",
        "CREATE INDEX IF NOT EXISTS idx_me_memory_id "
        "ON memory_events(memory_id)",
    ),
    (
        "idx_me_device_id",
        "CREATE INDEX IF NOT EXISTS idx_me_device_id "
        "ON memory_events(device_id)",
    ),
]


# The new tables this migration adds (in 3.11.1).
NEW_TABLES = ("memory_events", "sync_meta")


class MigrationReport(TypedDict):
    added: int
    tables_added: list[str]
    tables_already_present: list[str]
    indices_added: int


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


def migrate_311_tables(db_path: Path) -> MigrationReport:
    """Add the 3.11.1 schema tables to an existing bank at the older
    54-table schema. Idempotent.

    Returns a report dict with:
      - added: int (number of tables added in this call)
      - tables_added: List[str] (names of tables added in this call)
      - tables_already_present: List[str] (names already in the schema)
      - indices_added: int (number of indices added in this call)
    """
    db_path = Path(db_path)
    report: MigrationReport = {
        "added": 0,
        "tables_added": [],
        "tables_already_present": [],
        "indices_added": 0,
    }
    if not db_path.exists():
        # Nothing to migrate (the bank doesn't exist yet).
        return report

    conn = sqlite3.connect(str(db_path))
    try:
        for name, ddl in (
            ("memory_events", _MEMORY_EVENTS_DDL),
            ("sync_meta", _SYNC_META_DDL),
        ):
            if _has_table(conn, name):
                report["tables_already_present"].append(name)
                continue
            conn.execute(ddl)
            report["tables_added"].append(name)
            report["added"] += 1

        # Indices (best-effort; IF NOT EXISTS may not be supported
        # in all SQLite versions, matching the upstream behavior).
        for index_name, index_ddl in _MEMORY_EVENTS_INDICES:
            if _has_index(conn, index_name):
                continue
            try:
                conn.execute(index_ddl)
                report["indices_added"] += 1
            except sqlite3.OperationalError:
                # IF NOT EXISTS not supported in this SQLite version.
                # Indices are best-effort; not fatal.
                pass
        conn.commit()
    finally:
        conn.close()
    return report
