"""Supplemental MEMORIA source hydration obeys working-row eligibility."""
from dataclasses import replace
import json
import sqlite3

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.verbatim_ledger import (
    CAPTURE_KEY, CaptureProof, ExclusionSnapshot, _Generation,
    content_hash, resolve_exclusions,
)

QUERY = "What is the Telemetry API latency?"
TEXT = "[USER] Telemetry API latency is 240ms after the cache migration."


@pytest.fixture
def captured(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    memory = BeamMemory(session_id="fact-capture", db_path=tmp_path / "memory.db")
    nonce = "owned-metric-capture"
    mid = memory.remember(TEXT, source="conversation", metadata={CAPTURE_KEY: nonce})
    stored = memory.conn.execute("SELECT content FROM working_memory WHERE id=?", (mid,)).fetchone()[0]
    proof = CaptureProof(mid, memory.session_id, content_hash(stored), content_hash(stored), nonce)
    snapshot = ExclusionSnapshot(_Generation(), (proof,))
    # Both extraction and retrieval are real: no injected MEMORIA result IDs.
    assert mid in memory.memoria_retrieve(QUERY, top_k=3)["source_memory_ids"]
    assert resolve_exclusions(memory.conn, snapshot) == {mid}
    ordinary = memory.recall(QUERY, top_k=20, _cross_session=False)
    assert any(r["id"] == f"memoria_source_{mid}" and r["content"] == stored for r in ordinary)
    yield memory, mid, snapshot
    memory.conn.close()


def recall(memory, **kwargs):
    return memory.recall(QUERY, top_k=20, _cross_session=False, **kwargs)


def assert_no_raw_source(results, mid):
    assert not any(r["id"] in (mid, f"memoria_source_{mid}") for r in results)
    assert not any(r.get("source_memory_id") == mid for r in results)


def test_proof_excluded_capture_cannot_return_as_memoria_source(captured):
    memory, mid, snapshot = captured
    results = recall(memory, exclude_captures=snapshot)
    assert_no_raw_source(results, mid)
    assert all(r["content"] != TEXT for r in results)
    # Structured facts are independently eligible; the raw source is not.
    assert any(r["tier"] == "memoria" and "240ms" in r["content"] for r in results)
    # Revocation and ordinary explicit recall release both raw-row paths.
    snapshot.generation.valid = False
    for released in (recall(memory, exclude_captures=snapshot), recall(memory)):
        assert any(r["id"] == f"memoria_source_{mid}" for r in released)


@pytest.mark.parametrize("column,value,filters", [
    ("valid_until", "2000-01-01T00:00:00Z", {}),
    ("superseded_by", "replacement", {}),
    ("session_id", "unrelated-session", {}),
    ("source", "import", {"source": "conversation"}),
    ("source", "import", {"topic": "conversation"}),
    ("author_id", "different-author", {"author_id": "requested-author"}),
    ("author_type", "agent", {"author_type": "human"}),
    ("channel_id", "different-channel", {"channel_id": "requested-channel"}),
    ("veracity", "unknown", {"veracity": "verified"}),
    ("memory_type", "observation", {"memory_type": "decision"}),
    ("timestamp", "2000-01-01T00:00:00", {"from_date": "2001-01-01"}),
    ("timestamp", "2099-01-01T00:00:00", {"to_date": "2098-01-01"}),
])
def test_memoria_sources_obey_same_working_filters(captured, column, value, filters):
    memory, mid, _ = captured
    memory.conn.execute(f"UPDATE working_memory SET {column}=? WHERE id=?", (value, mid))
    memory.conn.commit()
    assert mid in memory.memoria_retrieve(QUERY, top_k=3)["source_memory_ids"]
    results = recall(memory, **filters)
    assert_no_raw_source(results, mid)
    assert any(r["tier"] == "memoria" for r in results)


def test_valid_source_parameters_preserve_memoria_hydration(captured):
    memory, mid, _ = captured
    memory.conn.execute(
        "UPDATE working_memory SET author_id=?, author_type=?, channel_id=?, "
        "veracity=?, memory_type=? WHERE id=?",
        ("requested-author", "human", "requested-channel", "verified", "decision", mid),
    )
    memory.conn.commit()
    results = recall(memory, source="conversation", topic="conversation", author_id="requested-author",
                     author_type="human", channel_id="requested-channel", veracity="verified",
                     memory_type="decision", from_date="2000-01-01", to_date="2099-01-01")
    assert any(r["id"] == f"memoria_source_{mid}" for r in results)


def test_exclusions_leave_same_text_unowned_memoria_source(captured):
    memory, mid, snapshot = captured
    copy_id = "independent-unmarked-source"
    memory.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, importance, session_id) "
        "SELECT ?, content, source, timestamp, importance, session_id FROM working_memory WHERE id=?",
        (copy_id, mid),
    )
    memory.conn.execute("UPDATE memoria_facts SET source_memory_id=? WHERE source_memory_id=?", (copy_id, mid))
    memory.conn.commit()
    assert resolve_exclusions(memory.conn, snapshot) == {mid}
    assert copy_id in memory.memoria_retrieve(QUERY, top_k=3)["source_memory_ids"]
    results = recall(memory, exclude_captures=snapshot)
    assert_no_raw_source(results, mid)
    assert any(r["id"] == f"memoria_source_{copy_id}" and r["content"] == TEXT for r in results)


def test_memoria_sources_fit_legacy_sqlite_variable_budget(captured):
    memory, mid, snapshot = captured
    setlimit = getattr(memory.conn, "setlimit", None)
    if not callable(setlimit):
        pytest.skip("SQLite runtime variable limits require Python 3.11+")
    proofs = list(snapshot.captures)
    for i in range(159):
        other_id = f"budget-capture-{i}"
        memory.conn.execute(
            "INSERT INTO working_memory (id,content,source,timestamp,session_id,metadata_json) "
            "VALUES (?,?,'conversation','2026-01-01',?,?)",
            (other_id, TEXT, memory.session_id, json.dumps({CAPTURE_KEY: proofs[0].nonce})),
        )
        proofs.append(replace(proofs[0], memory_id=other_id))
    memory.conn.commit()
    snapshot = ExclusionSnapshot(_Generation(), tuple(proofs))
    previous = setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    try:
        assert resolve_exclusions(memory.conn, snapshot) == {p.memory_id for p in proofs}
        results = recall(memory, exclude_captures=snapshot, source="conversation", topic="conversation",
                         from_date="2000-01-01", to_date="2099-01-01")
        assert_no_raw_source(results, mid)
        assert any(r["tier"] == "memoria" for r in results)
        snapshot.generation.valid = False
        assert any(r["id"] == f"memoria_source_{mid}" for r in recall(memory, exclude_captures=snapshot))
    finally:
        setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)
