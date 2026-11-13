"""Standalone diagnostic provenance; all databases are synthetic and local."""

from __future__ import annotations

import hashlib
import json
import sqlite3

import pytest

import mnemosyne_hermes
from mnemosyne.core.llm_backends import get_host_llm_backend
from mnemosyne_hermes import MnemosyneMemoryProvider


def _snapshot(conn, db_path):
    schema = conn.execute("SELECT * FROM sqlite_master ORDER BY name").fetchall()
    # Snapshot physical tables, including FTS shadow storage, rather than
    # querying virtual tables that can legitimately be broken in error tests.
    rows = {
        name: conn.execute('SELECT * FROM "' + name.replace('"', '""') + '"').fetchall()
        for kind, name, _table, _root, sql in schema
        if kind == "table" and sql and "VIRTUAL TABLE" not in sql.upper()
    }
    return {
        "rows": rows,
        "schema": schema,
        "user_version": conn.execute("PRAGMA user_version").fetchone()[0],
        "hashes": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (db_path, db_path.with_name(db_path.name + "-wal"))
            if path.exists()
        },
        "foreign_keys": conn.execute("PRAGMA foreign_keys").fetchone()[0],
        "total_changes": conn.total_changes,
    }


@pytest.fixture
def provider(tmp_path, monkeypatch):
    # Only general host diagnostics are replaced. Initialization, Beam, orphan
    # queries, tool discovery, dispatch and both SQLite connections are real.
    monkeypatch.setattr("mnemosyne.diagnose.run_diagnostics", lambda **kw: {"key_findings": []})
    instance = MnemosyneMemoryProvider()
    instance.initialize("diagnose-provenance", hermes_home=str(tmp_path), agent_identity="test")
    assert instance._beam is not None
    conn = instance._beam.conn
    try:
        yield instance
    finally:
        try:
            instance.shutdown()
        finally:
            conn.close()


def test_provider_fixture_releases_lifecycle_state(tmp_path, monkeypatch):
    before = (
        mnemosyne_hermes._active_provider_count,
        mnemosyne_hermes._host_llm_owner_count,
        mnemosyne_hermes._provider_active,
    )
    lifecycle = provider.__wrapped__(tmp_path, monkeypatch)
    instance = next(lifecycle)
    conn = instance._beam.conn
    try:
        assert instance._is_active_in_module
        assert instance._owns_host_llm_backend
        assert mnemosyne_hermes._active_provider_count == before[0] + 1
        assert mnemosyne_hermes._host_llm_owner_count == before[1] + 1
        assert get_host_llm_backend() is not None
    finally:
        with pytest.raises(StopIteration):
            next(lifecycle)

    assert (
        mnemosyne_hermes._active_provider_count,
        mnemosyne_hermes._host_llm_owner_count,
        mnemosyne_hermes._provider_active,
    ) == before
    assert not instance._is_active_in_module
    assert not instance._owns_host_llm_backend
    assert instance._beam is None
    if before[1] == 0:
        assert get_host_llm_backend() is None
    with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
        conn.execute("SELECT 1")


@pytest.mark.parametrize("active_fk", [0, 1], ids=["active-off", "active-on"])
@pytest.mark.parametrize("read_only", [False, True], ids=["normal", "read-only"])
def test_diagnose_labels_secondary_connection_without_mutation(
    provider, monkeypatch, active_fk, read_only
):
    beam = provider._beam
    active = beam.conn
    # Fixture preparation is complete before the before-snapshot; retain the
    # provider's normal journal mode and hash both DB and WAL bytes.
    active.execute(f"PRAGMA foreign_keys={active_fk}")
    active.execute("PRAGMA user_version=37")
    active.execute(
        "INSERT INTO working_memory (id, content, source) VALUES (?, ?, ?)",
        ("live", "synthetic row", "test"),
    )
    active.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, ?, ?)",
        ("orphan-gist", "synthetic gist", "missing"),
    )
    active.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        ("missing", "[0.0]", "test"),
    )
    active.commit()
    if read_only:
        active.execute("PRAGMA query_only=ON")
    before = _snapshot(active, beam.db_path)
    original_connect = sqlite3.connect
    observed_fk = []
    secondary = []

    def connect(*args, **kwargs):
        conn = original_connect(*args, **kwargs)
        assert conn is not active
        if read_only:
            conn.execute("PRAGMA query_only=ON")
        observed_fk.append(conn.execute("PRAGMA foreign_keys").fetchone()[0])
        secondary.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", connect)
    assert provider.has_tool("mnemosyne_diagnose")
    assert "mnemosyne_diagnose" in {s["name"] for s in provider.get_tool_schemas()}
    result = json.loads(provider.handle_tool_call("mnemosyne_diagnose", {}))

    assert "error" not in result
    assert len(secondary) == 1
    assert _snapshot(active, beam.db_path) == before
    assert active.execute("PRAGMA query_only").fetchone()[0] == int(read_only)
    diagnostic = result["active_provider_orphan_diagnostics"]
    assert diagnostic["foreign_keys_enabled"] == observed_fk[0]
    assert diagnostic["foreign_keys_connection"] == "separate_diagnostic_connection"
    assert diagnostic["gists_orphan_memory_id"] == 1
    assert diagnostic["memory_embeddings_orphan_memory_id"] == 1
    assert diagnostic["orphan_memory_id_overlap"] == 1
    assert result["active_provider_counts"]["working_memory"] == 1


def test_diagnose_count_error_does_not_claim_connection_state(provider):
    active = provider._beam.conn
    active.execute("DROP TABLE facts")
    active.commit()
    before = _snapshot(active, provider._beam.db_path)

    result = json.loads(provider.handle_tool_call("mnemosyne_diagnose", {}))

    assert "no such table: facts" in result["active_provider_counts_error"]
    assert "active_provider_orphan_diagnostics" not in result
    assert _snapshot(active, provider._beam.db_path) == before


def test_diagnose_without_beam_preserves_base_result(monkeypatch):
    base = {"checks_total": 1, "key_findings": ["base finding"]}
    monkeypatch.setattr("mnemosyne.diagnose.run_diagnostics", lambda **kw: base)
    provider = MnemosyneMemoryProvider()

    assert json.loads(provider._handle_diagnose({})) == base
