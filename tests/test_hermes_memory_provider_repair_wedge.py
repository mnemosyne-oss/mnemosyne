"""Legacy provider: the vec_working repair must not touch the session connection.

Mirrors integrations/hermes/tests/test_diagnose_repair_connection.py for the
``hermes_memory_provider`` path, which carries the same ``_handle_diagnose``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory


def _provider_with_beam(tmp_path: Path):
    db_path = tmp_path / "banks" / "legacy" / "mnemosyne.db"
    beam = BeamMemory(session_id="repair-wedge", db_path=db_path)
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider._session_id = "repair-wedge"
    provider._agent_context = "primary"
    provider._profile_isolation_enabled = True
    return provider, beam.conn


def test_legacy_repair_uses_dedicated_connection(tmp_path, monkeypatch):
    provider, active = _provider_with_beam(tmp_path)
    monkeypatch.setattr(
        "mnemosyne.diagnose.run_diagnostics", lambda **kw: {"key_findings": []}
    )
    seen = {}

    def fake_repair(conn, *, dry_run=False):
        seen["conn"] = conn
        conn.execute(
            "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
            ("mid-transaction", "wedge probe", "test"),
        )
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(beam_module, "repair_vec_working", fake_repair)

    result = json.loads(provider._handle_diagnose({"repair_vec_working": True}))

    assert "database is locked" in result["active_provider_vec_working_error"]
    assert seen["conn"] is not active
    with pytest.raises(sqlite3.ProgrammingError):
        seen["conn"].execute("SELECT 1")
    assert not active.in_transaction
    active.execute(
        "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
        ("after-wedge", "session still writable", "test"),
    )
    active.commit()
    assert (
        active.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'mid-transaction'"
        ).fetchone()[0]
        == 0
    )
