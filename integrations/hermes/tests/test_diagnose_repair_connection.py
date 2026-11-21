"""The vec_working repair must never run on the provider's long-lived connection.

A "database is locked" collision during an in-session diagnose repair left the
connection inside the half-finished transaction, so every later remember/recall
on that session failed with "database is locked" for the life of the process.
These tests pin the containment: repair gets a dedicated short-lived
connection, failures on it leave the provider connection writable, and the
base diagnostics stay read-only when the provider DB is the active bank.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

import mnemosyne.core.beam as beam_module
import mnemosyne.diagnose as diagnose_module
from mnemosyne_hermes import MnemosyneMemoryProvider


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.setattr(
        diagnose_module, "run_diagnostics", lambda **kw: {"key_findings": []}
    )
    instance = MnemosyneMemoryProvider()
    instance.initialize("diagnose-repair", hermes_home=str(tmp_path), agent_identity="test")
    assert instance._beam is not None
    try:
        yield instance
    finally:
        try:
            instance.shutdown()
        except Exception:
            pass


def test_repair_runs_on_dedicated_connection_and_leaves_session_writable(
    provider, monkeypatch
):
    beam = provider._beam
    active = beam.conn
    seen = {}

    def fake_repair(conn, *, dry_run=False):
        seen["conn"] = conn
        # Emulate the collision: a write has begun and the lock error lands
        # mid-transaction, exactly like the real backfill under a rival writer.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
            ("mid-transaction", "wedge probe", "test"),
        )
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(beam_module, "repair_vec_working", fake_repair)

    result = json.loads(
        provider.handle_tool_call("mnemosyne_diagnose", {"repair_vec_working": True})
    )

    assert "database is locked" in result["active_provider_vec_working_error"]
    assert seen["conn"] is not None
    assert seen["conn"] is not active, "repair must not receive the provider connection"
    with pytest.raises(sqlite3.ProgrammingError):
        seen["conn"].execute("SELECT 1")  # short-lived: closed, not leaked
    # The provider session survived the failed repair.
    assert not active.in_transaction
    active.execute(
        "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
        ("after-wedge", "session still writable", "test"),
    )
    active.commit()
    assert (
        active.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'after-wedge'"
        ).fetchone()[0]
        == 1
    )
    # The half-finished write never reached the database either.
    assert (
        active.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'mid-transaction'"
        ).fetchone()[0]
        == 0
    )


def test_public_tools_still_work_after_the_failed_repair(provider, monkeypatch):
    """Direct SQL proves the connection is writable; the tools prove the session is.

    The production incident surfaced as every later mnemosyne_remember /
    mnemosyne_recall returning "database is locked" for the life of the
    process. Reproduce the contained lock collision here, then pin the
    actual public entry points, not just the connection.
    """
    def fake_repair(conn, *, dry_run=False):
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
            ("mid-transaction", "wedge probe", "test"),
        )
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(beam_module, "repair_vec_working", fake_repair)
    result = json.loads(provider.handle_tool_call(
        "mnemosyne_diagnose", {"repair_vec_working": True}
    ))
    assert "database is locked" in result["active_provider_vec_working_error"]

    remember = json.loads(provider.handle_tool_call(
        "mnemosyne_remember",
        {"content": "the session kept working after the wedged repair",
         "source": "test", "importance": 0.5},
    ))
    assert "error" not in remember, f"remember failed after contained repair error: {remember}"

    recall = json.loads(provider.handle_tool_call(
        "mnemosyne_recall",
        {"query": "session kept working wedged repair", "limit": 5},
    ))
    assert "error" not in recall, f"recall failed after contained repair error: {recall}"
    assert isinstance(recall.get("results"), list)


def test_real_repair_backfills_vec_working_and_leaves_provider_healthy(
    provider, monkeypatch
):
    """Success path of the repair on the DEDICATED connection, not a stub.

    The lock-collision test replaces repair_vec_working entirely, so nothing
    there exercises the real backfill running on the short-lived connection.
    Seed a memory embedding whose vec_working row is missing, repair, and
    demand the documented result: repaired, one row inserted, coverage
    complete - and the session still writable afterwards.
    """
    np = pytest.importorskip("numpy")
    pytest.importorskip("sqlite_vec")
    if not beam_module._wm_vec_available(provider._beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    now = "2026-01-01T00:00:00"
    memory_id = "active-gap"
    embedding = np.array([1.0] + [0.0] * (beam_module.EMBEDDING_DIM - 1), dtype=np.float32)
    provider._beam.conn.execute(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, "active provider vector gap", "test", now,
         "diagnose-repair", "session", 0.5),
    )
    provider._beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        (memory_id, beam_module._embeddings.serialize(embedding), "test"),
    )
    provider._beam.conn.commit()

    result = json.loads(provider.handle_tool_call(
        "mnemosyne_diagnose", {"repair_vec_working": True}
    ))
    assert result["active_provider_vec_working_repair"]["status"] == "repaired"
    assert result["active_provider_vec_working_repair"]["inserted"] == 1
    assert result["active_provider_vec_working"]["missing_vec_working_rows"] == 0
    assert result["active_provider_vec_working"]["status"] == "complete"

    # Public write path on the provider's own connection, after the repair ran.
    remember = json.loads(provider.handle_tool_call(
        "mnemosyne_remember", {"content": "written after real repair", "source": "test"}
    ))
    assert "error" not in remember


def test_base_diagnostics_are_read_only_when_provider_db_active(
    provider, monkeypatch
):
    calls = []

    def spy(**kw):
        calls.append(kw)
        return {"key_findings": []}

    monkeypatch.setattr(diagnose_module, "run_diagnostics", spy)
    provider.handle_tool_call("mnemosyne_diagnose", {"repair_vec_working": True})
    assert calls, "run_diagnostics was not called"
    assert all(call.get("repair_vec_working") is False for call in calls), (
        "run_diagnostics must not repair the active provider bank: its "
        "BeamMemory resolves to the provider's own thread-local connection"
    )
