"""Task 4: report-only migration support — fingerprint tests.

Proves E6 / E7 / CLI dry runs never write: the schema fingerprint
(ordered ``sqlite_master(type, name, sql)`` rows + ``PRAGMA user_version``)
and the on-disk byte size stay identical, even for a WAL-mode bank with a
committed seed write.
"""

from __future__ import annotations

import hashlib
import sqlite3

import mnemosyne.cli as cli
from mnemosyne.migrations.e6_triplestore_split import migrate as e6_migrate
from mnemosyne.migrations.e7_311_tables import migrate_311_tables


def _schema_fingerprint(db_path) -> str:
    """Test-internal hash of schema rows + user_version, read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        rows = conn.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()
    return hashlib.sha256(repr((rows, user_version)).encode()).hexdigest()


def _wal_seed(db_path) -> None:
    """Switch to WAL and perform one committed seed write."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS seed_probe (x TEXT)")
    conn.execute("DELETE FROM seed_probe")
    conn.execute("INSERT INTO seed_probe VALUES ('seed')")
    conn.commit()
    conn.close()


def _fresh_e7_bank(tmp_path):
    """54-table-era bank: full init minus the two 3.11.1 tables."""
    from mnemosyne.core.memory import init_db

    db_path = tmp_path / "e7_bank.db"
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    for tbl in ("memory_events", "sync_meta"):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.commit()
    conn.close()
    _wal_seed(db_path)
    return db_path


def _fresh_e6_bank(tmp_path):
    """Legacy bank with one pending annotation-flavored triples row."""
    db_path = tmp_path / "e6_bank.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE triples (
            id INTEGER PRIMARY KEY,
            subject TEXT,
            predicate TEXT,
            object TEXT,
            source TEXT,
            confidence REAL,
            created_at TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO triples (subject, predicate, object, source, confidence)"
        " VALUES ('mem-1', 'mentions', 'Alice', 'extraction', 0.9)"
    )
    conn.commit()
    conn.close()
    _wal_seed(db_path)
    return db_path


def _fresh_cli_bank(tmp_path, monkeypatch):
    """Bank on disk at the old schema, wired into the CLI module."""
    from mnemosyne.core.banks import BankManager
    from mnemosyne.core.memory import init_db

    data_dir = tmp_path / "data"
    db_path = BankManager(data_dir).create_bank("drybank")
    init_db(db_path)
    conn = sqlite3.connect(str(db_path))
    for tbl in ("memory_events", "sync_meta"):
        conn.execute(f"DROP TABLE IF EXISTS {tbl}")
    conn.commit()
    conn.close()
    _wal_seed(db_path)
    monkeypatch.setattr(cli, "DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_BANK", "drybank")
    return db_path


def test_e7_dry_run_leaves_schema_fingerprint_unchanged(tmp_path, monkeypatch):
    db_path = _fresh_e7_bank(tmp_path)
    before = _schema_fingerprint(db_path)
    before_size = db_path.stat().st_size

    connect_calls = []
    real_connect = sqlite3.connect

    def spy_connect(*args, **kwargs):
        connect_calls.append((args, kwargs))
        return real_connect(*args, **kwargs)

    from mnemosyne.migrations import e7_311_tables

    monkeypatch.setattr(e7_311_tables.sqlite3, "connect", spy_connect)

    report = migrate_311_tables(db_path, dry_run=True)

    # Verify connect was called with mode=ro and uri=True
    assert len(connect_calls) == 1
    c_args, c_kwargs = connect_calls[0]
    assert "mode=ro" in str(c_args[0])
    assert c_kwargs.get("uri") is True

    assert report["added"] == 0
    assert report["indices_added"] == 0
    assert report["would_add"] == 2
    assert sorted(report["tables_would_add"]) == ["memory_events", "sync_meta"]
    assert report["columns_would_add"] == []
    assert report["indices_would_add"] == 3
    assert _schema_fingerprint(db_path) == before
    assert db_path.stat().st_size == before_size

    # Truthful: nothing was actually created.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        tables = {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()
    assert "sync_meta" not in tables


def test_e7_dry_run_missing_bank_reports_empty_would_add(tmp_path):
    """A missing bank still returns a truthful dry-run report."""
    db_path = tmp_path / "missing.db"

    report = migrate_311_tables(db_path, dry_run=True)

    assert report["added"] == 0
    assert report["would_add"] == 0
    assert report["tables_would_add"] == []
    assert report["indices_would_add"] == 0


def test_e7_dry_run_report_fields_are_not_notrequired():
    """Regression: typing.NotRequired is 3.11+; project supports 3.10.

    The dry-run fields must be plain required TypedDict keys (no
    typing_extensions, no 3.11-only typing import) so the module keeps
    importing on Python 3.10.
    """
    from mnemosyne.migrations.e7_311_tables import MigrationDryRunReport

    assert MigrationDryRunReport.__required_keys__ >= {
        "added",
        "tables_added",
        "tables_already_present",
        "columns_added",
        "indices_added",
        "would_add",
        "tables_would_add",
        "columns_would_add",
        "indices_would_add",
    }


def test_e6_dry_run_leaves_schema_fingerprint_unchanged(tmp_path):
    db_path = _fresh_e6_bank(tmp_path)
    before = _schema_fingerprint(db_path)
    before_size = db_path.stat().st_size

    written = e6_migrate(db_path, dry_run=True, backup=False, log_fn=lambda _line: None)

    assert written == 1  # pending work was reported...
    assert _schema_fingerprint(db_path) == before
    assert db_path.stat().st_size == before_size


def test_cli_migrate_dry_run_leaves_schema_fingerprint_unchanged(
    tmp_path, monkeypatch, capsys
):
    db_path = _fresh_cli_bank(tmp_path, monkeypatch)
    before = _schema_fingerprint(db_path)
    before_size = db_path.stat().st_size

    cli.cmd_migrate(["--dry-run"])

    out = capsys.readouterr().out
    assert "DRY RUN" in out
    assert "memory_events" in out
    assert "sync_meta" in out
    assert "would add columns: (none)" in out
    assert "would add indices: 3" in out
    assert _schema_fingerprint(db_path) == before
    assert db_path.stat().st_size == before_size
