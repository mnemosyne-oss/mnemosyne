"""Regression coverage for issue #1047 legacy memory_events.device_id migration."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest


LEGACY_MEMORY_EVENTS_DDL = """
CREATE TABLE memory_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    action TEXT NOT NULL,
    memory_id TEXT,
    bank TEXT,
    scope TEXT,
    profile TEXT,
    session_id TEXT,
    source_tool TEXT,
    tokens_used INTEGER,
    reason TEXT,
    metadata_json TEXT,
    event_hash TEXT,
    synced_at TEXT,
    parent_event_ids TEXT DEFAULT '[]',
    expiry TEXT
)
""".strip()


@pytest.fixture
def _before_connection_reset(tmp_path, monkeypatch):
    """Keep memory.py's import-time default database inside this test tempdir."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "ambient-data"))


def _create_reported_legacy_db(db_path):
    with sqlite3.connect(db_path) as conn:
        conn.execute(LEGACY_MEMORY_EVENTS_DDL)
        conn.execute(
            "INSERT INTO memory_events (timestamp, action, memory_id) "
            "VALUES (1.0, 'CREATE', 'legacy-memory')"
        )
        conn.commit()


def _columns(db_path, table="memory_events"):
    with sqlite3.connect(db_path) as conn:
        return {
            row[1]: row
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }


def _has_index(db_path, index_name):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
            (index_name,),
        ).fetchone() is not None


def _schema_fingerprint(db_path):
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    return hashlib.sha256(repr((rows, user_version)).encode()).hexdigest()


def test_init_db_adds_device_id_before_index_on_reported_legacy_schema(tmp_path):
    from mnemosyne.core.memory import Mnemosyne, init_db

    db_path = tmp_path / "legacy.db"
    _create_reported_legacy_db(db_path)

    init_db(db_path)
    mem = Mnemosyne(db_path=db_path)

    cols = _columns(db_path)
    assert cols["device_id"][2].upper() == "TEXT"
    assert cols["device_id"][3] == 1  # NOT NULL
    assert cols["device_id"][4] == "''"
    assert _has_index(db_path, "idx_me_device_id")
    assert mem.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert mem.conn.execute(
        "SELECT device_id FROM memory_events WHERE memory_id='legacy-memory'"
    ).fetchone()[0] == ""


def test_migrate_311_dry_run_reports_device_id_without_writing(tmp_path):
    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    db_path = tmp_path / "legacy.db"
    _create_reported_legacy_db(db_path)
    before = _schema_fingerprint(db_path)

    report = migrate_311_tables(db_path, dry_run=True)

    assert report["added"] == 0
    assert report["would_add"] == 1
    assert report["tables_would_add"] == ["sync_meta"]
    assert report["columns_would_add"] == ["memory_events.device_id"]
    assert report["indices_would_add"] == 3
    assert "device_id" not in _columns(db_path)
    assert _schema_fingerprint(db_path) == before


def test_migrate_311_adds_device_id_and_runtime_can_open(tmp_path):
    from mnemosyne.core.memory import Mnemosyne, init_db
    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    db_path = tmp_path / "legacy.db"
    _create_reported_legacy_db(db_path)

    first = migrate_311_tables(db_path)
    second = migrate_311_tables(db_path)

    assert first["tables_added"] == ["sync_meta"]
    assert first["columns_added"] == ["memory_events.device_id"]
    assert first["indices_added"] == 3
    assert second["added"] == 0
    assert second["columns_added"] == []
    assert second["indices_added"] == 0
    assert _has_index(db_path, "idx_me_device_id")

    init_db(db_path)
    mem = Mnemosyne(db_path=db_path)
    assert mem.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_fresh_init_db_creates_device_id_and_index(tmp_path):
    from mnemosyne.core.memory import init_db

    db_path = tmp_path / "fresh.db"
    init_db(db_path)

    cols = _columns(db_path)
    assert "device_id" in cols
    assert _has_index(db_path, "idx_me_device_id")


def test_migrate_311_device_id_ddl_errors_propagate(tmp_path, monkeypatch):
    from mnemosyne.migrations import e7_311_tables

    db_path = tmp_path / "legacy.db"
    _create_reported_legacy_db(db_path)
    real_connect = sqlite3.connect

    class FailingConnection:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, params=()):
            if sql.startswith("ALTER TABLE memory_events ADD COLUMN device_id"):
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, params)

        def commit(self):
            return self._conn.commit()

        def close(self):
            return self._conn.close()

    def connect(*args, **kwargs):
        return FailingConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(e7_311_tables.sqlite3, "connect", connect)

    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        e7_311_tables.migrate_311_tables(db_path)
