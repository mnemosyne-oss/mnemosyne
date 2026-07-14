"""Regression test for [Issue 3]: an existing mnemosyne bank at the
older 54-table schema (no `memory_events`, no `sync_meta`) must get
those two tables added by an explicit migration that uses the
canonical schema-init helpers (not invented DDL).

The fix lives in `mnemosyne.migrations.e7_311_tables.migrate_311_tables`,
which opens the bank and runs the same `CREATE TABLE IF NOT EXISTS`
statements that the upstream `SyncManager._init_events_table` does
(canonical source at `mnemosyne/core/sync.py:641-677`). The test
asserts:

1. A bank at the 54-table schema (memory_events and sync_meta
   missing) is the input.
2. After the migration, the two tables exist with the canonical
   columns and types from the upstream source.
3. The other 54 (or more) tables are NOT dropped (no data loss).
4. The migration is idempotent (re-running is a no-op).
5. A bank at the current schema (already has the two tables) is
   unchanged by the migration.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


# Tables added in 3.11.1 (the gap the migration closes).
NEW_TABLES = ("memory_events", "sync_meta")


def _all_tables(db_path: Path) -> set[str]:
    con = sqlite3.connect(str(db_path))
    rows = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    con.close()
    return {r[0] for r in rows}


def _drop_to_old_schema(db_path: Path) -> set[str]:
    """Initialize to the full schema, then drop the 2 new tables
    unconditionally (DROP TABLE IF EXISTS) to simulate the older
    54-table bank state. The fork's init_db creates memory_events
    during init, but sync_meta is created lazily by the SyncManager;
    we drop both unconditionally so the simulation works regardless
    of which tables init_db actually creates. Returns the pre-drop
    table set for the test to assert against (proves we did not
    drop unrelated tables)."""
    from mnemosyne.core.memory import init_db
    init_db(db_path)
    pre = _all_tables(db_path)
    con = sqlite3.connect(str(db_path))
    for tbl in NEW_TABLES:
        con.execute(f"DROP TABLE IF EXISTS {tbl}")
    con.commit()
    con.close()
    return pre


def test_old_schema_bank_gets_missing_tables_via_migration(tmp_path):
    """The migration adds memory_events + sync_meta to a 54-table bank
    without dropping any existing tables."""
    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    db_path = tmp_path / "old_bank.db"
    pre = _drop_to_old_schema(db_path)  # pre = full schema set

    # Sanity: after the drop, the two new tables are gone.
    after_drop = _all_tables(db_path)
    assert "memory_events" not in after_drop
    assert "sync_meta" not in after_drop

    # Run the migration.
    result = migrate_311_tables(db_path)
    assert result["added"] == 2, (
        f"expected 2 tables added, got {result['added']}: {result}"
    )
    assert set(result["tables_added"]) == set(NEW_TABLES)
    assert result["indices_added"] == 3

    # The two new tables are now present.
    after = _all_tables(db_path)
    assert "memory_events" in after
    assert "sync_meta" in after
    # No data loss: every table that was in pre is still in after.
    missing = pre - after
    assert not missing, f"migration dropped tables: {missing}"


def test_migration_is_idempotent(tmp_path):
    """Running the migration twice does not fail or duplicate data."""
    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    db_path = tmp_path / "old_bank.db"
    _drop_to_old_schema(db_path)
    first = migrate_311_tables(db_path)
    second = migrate_311_tables(db_path)
    assert first["added"] == 2
    assert second["added"] == 0, (
        f"second run should be a no-op, got {second['added']} added"
    )
    assert second["indices_added"] == 0
    # Tables still present.
    after = _all_tables(db_path)
    assert "memory_events" in after
    assert "sync_meta" in after


def test_migration_on_already_current_bank_is_a_noop(tmp_path):
    """A bank that already has the two new tables is unchanged."""
    from mnemosyne.core.memory import init_db
    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    db_path = tmp_path / "current_bank.db"
    init_db(db_path)
    # Bring sync_meta into existence so the bank matches a 3.11.1+ state
    # (init_db may or may not create sync_meta; the SyncManager does on
    # first meta access, but for the assertion we just need the table
    # to exist so the migration sees it as already-present).
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TABLE IF NOT EXISTS sync_meta (key TEXT PRIMARY KEY, value TEXT)")
    con.commit()
    con.close()
    pre = _all_tables(db_path)
    result = migrate_311_tables(db_path)
    assert result["added"] == 0
    post = _all_tables(db_path)
    assert pre == post, "migration changed the table set on a current bank"
    # No duplicate indices, no schema drift.
    con = sqlite3.connect(str(db_path))
    ntables = con.execute(
        "SELECT count(*) FROM sqlite_master WHERE type='table'"
    ).fetchone()[0]
    con.close()
    assert ntables == len(post)


def test_migration_creates_canonical_columns_for_memory_events(tmp_path):
    """The migration must use the canonical DDL (CREATE TABLE IF NOT
    EXISTS) so existing data in the other tables is preserved and
    the new tables match the upstream column layout."""
    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    db_path = tmp_path / "old_bank.db"
    _drop_to_old_schema(db_path)
    migrate_311_tables(db_path)

    con = sqlite3.connect(str(db_path))
    cols = [
        r[1] for r in con.execute("PRAGMA table_info(memory_events)").fetchall()
    ]
    # Canonical columns from upstream _init_events_table (sync.py).
    assert "event_id" in cols
    assert "memory_id" in cols
    assert "operation" in cols
    assert "device_id" in cols
    assert "timestamp" in cols

    cols = [r[1] for r in con.execute("PRAGMA table_info(sync_meta)").fetchall()]
    assert "key" in cols
    assert "value" in cols
    con.close()



def test_migration_noop_when_database_does_not_exist(tmp_path):
    """A missing database path returns the unchanged no-op report."""
    from mnemosyne.migrations.e7_311_tables import migrate_311_tables

    db_path = tmp_path / "missing.db"
    result = migrate_311_tables(db_path)

    assert result == {
        "added": 0,
        "tables_added": [],
        "tables_already_present": [],
        "indices_added": 0,
    }
    assert not db_path.exists()
