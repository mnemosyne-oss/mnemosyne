"""Regression tests for opt-in recall explain traces (#322)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def test_recall_default_return_shape_unchanged(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam.remember("Alice prefers Vim editor", source="pref", importance=0.7)

    results = beam.recall("Alice Vim", top_k=5)

    assert isinstance(results, list)
    assert results
    assert all(isinstance(row, dict) for row in results)


def test_recall_explain_returns_structured_payload(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam.remember("Alice prefers Vim editor", source="pref", importance=0.7)

    payload = beam.recall("Alice Vim", top_k=5, explain=True)

    assert payload["query"] == "Alice Vim"
    assert payload["top_k"] == 5
    assert payload["engine"] == "linear"
    assert isinstance(payload["results"], list)
    explain = payload["explain"]
    assert set(explain) >= {"filters", "weights", "embedding", "stages", "candidates", "truncation"}
    json.dumps(payload)


def test_explain_contains_stage_counts_and_topk_truncation(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    for i in range(4):
        beam.remember(f"Alice project note {i}", source="note", importance=0.5)

    payload = beam.recall("Alice project", top_k=1, explain=True)
    explain = payload["explain"]

    assert explain["stages"]
    assert explain["truncation"]["top_k"] == 1
    assert explain["truncation"]["dropped_count"] >= 1
    assert any(c["drop_reason"] == "top_k_truncated" for c in explain["candidates"])


def test_explain_candidate_has_source_path_and_scores(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam.remember("Alice keeps architecture notes", source="note", importance=0.6)

    payload = beam.recall("Alice architecture", top_k=5, explain=True)
    candidates = payload["explain"]["candidates"]

    assert candidates
    candidate = candidates[0]
    assert candidate["source_path"] in {"wm_fts", "wm_vec", "wm_fallback", "em_fts", "em_vec", "em_fallback", "memoria", "memoria_source"}
    assert set(candidate["scores"]) >= {"keyword", "fts", "dense", "importance", "recency_decay", "final"}


def test_explain_does_not_leak_cross_session_filtered_rows(temp_db):
    foreign = BeamMemory(session_id="foreign", db_path=temp_db)
    foreign.remember("SECRET foreign Alice content", source="note", importance=0.9)
    local = BeamMemory(session_id="local", db_path=temp_db)
    local.remember("Alice local visible content", source="note", importance=0.5)

    payload = local.recall("Alice", top_k=5, explain=True)
    serialized_explain = json.dumps(payload["explain"])

    assert "SECRET foreign" not in serialized_explain


def test_explain_has_no_raw_sql_or_embedding_values(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam.remember("Alice prefers Vim editor", source="pref", importance=0.7)

    payload = beam.recall("Alice Vim", top_k=5, explain=True)
    serialized_explain = json.dumps(payload["explain"])

    assert "SELECT " not in serialized_explain.upper()
    assert " MATCH " not in serialized_explain.upper()
    assert "embedding" in payload["explain"]
    assert "values" not in payload["explain"]["embedding"]


def test_linear_explain_distinguishes_dense_candidates_with_zero_scores(temp_db, monkeypatch):
    import numpy as np

    from mnemosyne.core import embeddings

    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setattr(embeddings, "available", lambda: False)
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    controls = (
        ("Quartz telescope mirror shipped from Zurich last April.", None, (False, 0.0)),
        ("Quartz telescope calibration tracks drift near Orion.", [0.0, 1.0], (True, 0.0)),
        ("Quartz telescope instruments record diffraction through the solar filter.", [0.00001, 1.0], (True, 0.0)),
        ("Quartz telescope tracking campaign uses the main detector.", [0.8, 0.6], (True, 0.8)),
    )
    expected = {}
    for content, embedding, values in controls:
        mid = beam.remember(content, source="fact", importance=0.5)
        expected[mid] = values
        if embedding is not None:
            beam.conn.execute(
                "INSERT INTO memory_embeddings(memory_id, embedding_json, model) VALUES (?, ?, ?)",
                (mid, json.dumps(embedding + [0.0] * 382), "test"),
            )
    beam.conn.commit()
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(
        embeddings, "embed_query", lambda *_a, **_kw: np.array([1.0] + [0.0] * 383, dtype=np.float32),
    )

    payload = beam.recall("quartz telescope", top_k=10, explain=True)

    assert payload["engine"] == "linear"
    assert {r["id"]: r["dense_score"] for r in payload["results"]} == {
        mid: scores[1] for mid, scores in expected.items()
    }
    assert {
        c["id"]: (c["dense_candidate"], c["scores"]["dense"])
        for c in payload["explain"]["candidates"]
    } == expected
    assert all("dense_candidate" not in r for r in payload["results"])

    beam.conn.execute("DELETE FROM memory_embeddings")
    beam.conn.commit()
    without_candidates = beam.recall("quartz telescope", top_k=10, explain=True)
    assert {
        c["id"]: c["dense_candidate"] for c in without_candidates["explain"]["candidates"]
    } == dict.fromkeys(expected, False)


@pytest.mark.parametrize("native_available", [False, True], ids=["unavailable", "empty-native"])
def test_linear_explain_omits_dense_membership_after_backend_error(temp_db, monkeypatch, native_available):
    """A failed compatibility scan leaves lexical results and no membership flag."""
    import sqlite3

    np = pytest.importorskip("numpy")
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings

    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setattr(embeddings, "available", lambda: False)
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    mid = beam.remember("Quartz telescope tracks stars in Orion.", source="fact", scope="session")
    dim = beam_module.EMBEDDING_DIM
    query_vec = np.zeros(dim, dtype=np.float32)
    query_vec[0] = 1
    if native_available and not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec working-memory table unavailable")
    beam.conn.execute("DROP TABLE IF EXISTS vec_working")
    if native_available:
        beam.conn.execute(f"CREATE VIRTUAL TABLE vec_working USING vec0(embedding float32[{dim}])")
        foreign = beam.remember("Quartz telescope tracks stars over Chile.", source="fact", scope="session")
        beam.conn.execute(
            "UPDATE working_memory SET session_id='other', channel_id='other' WHERE id=?", (foreign,),
        )
        rowid = beam.conn.execute("SELECT rowid FROM working_memory WHERE id=?", (foreign,)).fetchone()[0]
        beam_module._vec_table_insert(beam.conn, "vec_working", rowid, query_vec, commit=False)
    beam.conn.execute(
        "INSERT INTO memory_embeddings(memory_id, embedding_json, model) VALUES (?, ?, 'test')",
        (mid, json.dumps(query_vec.tolist())),
    )
    beam.conn.commit()
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "embed_query", lambda *_a, **_kw: query_vec)

    denied = False
    statements = []

    def authorize(action, table, column, _db, _trigger):
        """Reject only the actual compatibility-scan payload read."""
        nonlocal denied
        if action == sqlite3.SQLITE_READ and table == "memory_embeddings" and column == "embedding_json":
            denied = True
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    beam.conn.set_authorizer(authorize)
    beam.conn.set_trace_callback(statements.append)
    try:
        payload = beam.recall("quartz telescope", explain=True)
        ordinary = beam.recall("quartz telescope")
    finally:
        beam.conn.set_trace_callback(None)
        # Disabling an authorizer with None requires Python 3.11.
        beam.conn.set_authorizer(lambda *_: sqlite3.SQLITE_OK)

    assert denied
    assert any("vw.embedding MATCH" in statement for statement in statements) == native_available
    assert [r["id"] for r in payload["results"]] == [mid]
    assert [r["id"] for r in ordinary] == [mid]
    candidate, = payload["explain"]["candidates"]
    assert candidate["id"] == mid
    assert "dense_candidate" not in candidate

    recovered = beam.recall("quartz telescope", explain=True)
    recovered_candidate, = recovered["explain"]["candidates"]
    assert recovered_candidate["id"] == mid
    assert recovered_candidate["dense_candidate"] is True
    assert recovered_candidate["scores"]["dense"] == pytest.approx(1.0)


@pytest.mark.parametrize("vec_type", ["float32", "int8"])
def test_linear_explain_native_dense_membership_survives_truncation(temp_db, monkeypatch, vec_type):
    """Keep native WM membership in the trace through truncation and tier lookup."""
    np = pytest.importorskip("numpy")
    from mnemosyne.core import beam as beam_module
    from mnemosyne.core import embeddings

    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setattr(embeddings, "available", lambda: False)
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec working-memory table unavailable")
    dim = beam_module.EMBEDDING_DIM
    beam.conn.execute("DROP TABLE vec_working")
    beam.conn.execute(f"CREATE VIRTUAL TABLE vec_working USING vec0(embedding {vec_type}[{dim}])")
    aligned = np.zeros(dim, dtype=np.float32)
    aligned[0] = 1
    orthogonal = np.zeros(dim, dtype=np.float32)
    orthogonal[1] = 1
    strongest = beam.remember("Quartz telescope observes Orion from the mountaintop.", source="fact", importance=0.9)
    zero = beam.remember("Quartz telescope equipment shipped by rail last April.", source="fact", importance=0.2)
    for mid, embedding in ((strongest, aligned), (zero, orthogonal)):
        rowid = beam.conn.execute("SELECT rowid FROM working_memory WHERE id=?", (mid,)).fetchone()[0]
        beam_module._vec_table_insert(beam.conn, "vec_working", rowid, embedding, commit=False)
    beam.conn.commit()
    assert beam.conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0] == 0
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "embed_query", lambda *_a, **_kw: aligned)

    payload = beam.recall("quartz telescope", top_k=1, explain=True)

    assert [r["id"] for r in payload["results"]] == [strongest]
    kept, truncated = payload["explain"]["candidates"]
    assert kept["id"] == strongest and kept["dense_candidate"] is True
    assert kept["scores"]["dense"] == pytest.approx(1.0)
    assert truncated["dense_candidate"] is True and truncated["scores"]["dense"] == 0.0
    assert truncated["kept"] is False and truncated["drop_reason"] == "top_k_truncated"
    assert truncated["id"] is None and truncated["content_preview"] is None

    beam.conn.execute(
        "INSERT INTO episodic_memory(id, content, source, timestamp, session_id, importance) "
        "VALUES (?, 'Quartz telescope repair completed in the workshop.', 'test', CURRENT_TIMESTAMP, 's1', 0.5)",
        (zero,),
    )
    beam.conn.commit()

    shared_id = beam.recall("quartz telescope", top_k=10, explain=True)
    twins = {
        c["tier"]: c for c in shared_id["explain"]["candidates"] if c["id"] == zero
    }
    assert set(twins) == {"episodic", "working"}
    assert twins["working"]["dense_candidate"] is True
    assert "dense_candidate" not in twins["episodic"]
