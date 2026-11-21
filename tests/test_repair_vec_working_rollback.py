"""repair_vec_working must not leave an open transaction behind on collision.

A "database is locked" collision surfaces after the backfill has already
written into an implicit transaction. Without a rollback the passed connection
stays inside the half-finished repair, and every later write on it fails for
the life of the connection (in the Hermes provider that connection was the
session's long-lived one, so one collision wedged all later remember/recall).
"""

from __future__ import annotations

import sqlite3

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import repair_vec_working


@pytest.fixture
def conn(tmp_path):
    handle = sqlite3.connect(str(tmp_path / "m.db"))
    handle.execute(
        "CREATE TABLE working_memory (id TEXT PRIMARY KEY, content TEXT,"
        " timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, importance REAL,"
        " source TEXT)"
    )
    yield handle
    handle.close()


def _exploding_backfill(c: sqlite3.Connection) -> int:
    # One row lands, then the rival writer wins the lock: exactly the state a
    # caller must never inherit.
    c.execute(
        "INSERT INTO working_memory (id, content, source)"
        " VALUES ('half-written', 'x', 'test')"
    )
    raise sqlite3.OperationalError("database is locked")


def test_failed_backfill_rolls_back_the_connection(
    conn, monkeypatch
):
    monkeypatch.setattr(
        beam_module, "_backfill_vec_working_from_memory_embeddings", _exploding_backfill
    )
    monkeypatch.setattr(
        beam_module,
        "vec_working_coverage",
        lambda _conn: {"vec_working_available": True},
    )

    result = repair_vec_working(conn)

    assert result["status"] == "error"
    assert "database is locked" in result["error"]
    assert not conn.in_transaction, "failed repair left the connection mid-transaction"
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
    # And the connection is usable again - the wedge is the regression here.
    conn.execute(
        "INSERT INTO working_memory (id, content, source) VALUES ('later', 'y', 'test')"
    )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1


def test_dry_run_never_writes(conn, monkeypatch):
    monkeypatch.setattr(
        beam_module,
        "vec_working_coverage",
        lambda _conn: {"vec_working_available": True},
    )
    result = repair_vec_working(conn, dry_run=True)
    assert result["status"] == "dry_run"
    assert not conn.in_transaction
