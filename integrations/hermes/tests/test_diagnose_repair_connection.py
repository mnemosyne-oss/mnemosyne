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
