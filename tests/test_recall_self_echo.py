"""Recall excludes only provider-owned unchanged ROWS, before WM selection."""
import json
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.polyphonic_recall import RecallResult
from mnemosyne.core.verbatim_ledger import (
    CAPTURE_KEY,
    MAX_CAPTURES,
    CaptureProof,
    ExclusionSnapshot,
    _Generation,
    content_hash,
    resolve_exclusions,
)


@pytest.fixture(params=[False, True], ids=["linear", "polyphonic"])
def beam(request, tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", str(int(request.param)))
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "0")
    monkeypatch.setenv("MNEMOSYNE_VOICE_FACT", "0")
    return BeamMemory(session_id="actual-beam-session", db_path=tmp_path / "memory.db")


def row(beam, mid, content, session="actual-beam-session", nonce=None, source="conversation", age=0):
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, session_id, source, timestamp, importance, metadata_json)"
        " VALUES (?, ?, ?, ?, ?, 0.5, ?)",
        (mid, content, session, source, (datetime.now() - timedelta(hours=age)).isoformat(),
         json.dumps({CAPTURE_KEY: nonce} if nonce else {})),
    )
    beam.conn.commit()


def owned(beam, mid="owned", content="[USER] orchard botanical archive itinerary"):
    nonce = "fresh-capture-" + mid
    row(beam, mid, content, nonce=nonce)
    return CaptureProof(mid, beam.session_id, content_hash(content), content_hash(content), nonce)


def snapshot(*proofs):
    return ExclusionSnapshot(_Generation(), tuple(proofs))


def recall(beam, proofs=None, top_k=100, **kw):
    return beam.recall("recent orchard botanical archive itinerary", top_k=top_k,
                       exclude_captures=proofs, **kw)


def test_owned_row_excluded_explicit_recall_unchanged(beam):
    proof = owned(beam)
    assert {r["id"] for r in recall(beam)} == {proof.memory_id}
    assert recall(beam, snapshot(proof)) == []


@pytest.mark.parametrize("mid,session,source", [
    ("null-row", None, "conversation"),
    ("import-row", "actual-beam-session", "import"),
    ("unmarked-row", "actual-beam-session", "conversation"),
    ("other-session-row", "other", "conversation"),
])
def test_equal_text_different_rows_null_import_unmarked_remain(beam, mid, session, source):
    proof = owned(beam)
    content = "[USER] orchard botanical archive itinerary"
    row(beam, mid, content, session=session, source=source)
    results = recall(beam, snapshot(proof), _cross_session=True)
    assert {r["id"] for r in results} == {mid}


@pytest.mark.parametrize("column,value", [("content", "Edited orchard botanical archive itinerary"),
                                         ("source", "import"), ("session_id", None),
                                         ("metadata_json", "{}")])
def test_edited_or_unowned_rows_remain_recallable(beam, column, value):
    proof = owned(beam)
    beam.conn.execute(f"UPDATE working_memory SET {column}=? WHERE id=?", (value, proof.memory_id))
    beam.conn.commit()
    assert {r["id"] for r in recall(beam, snapshot(proof))} == {proof.memory_id}


def test_released_cached_snapshot_is_ordinary_recall(beam):
    proof = owned(beam)
    cached = snapshot(proof)
    assert recall(beam, cached) == []
    cached.generation.valid = False
    assert {r["id"] for r in recall(beam, cached)} == {proof.memory_id}


def test_over_cap_and_unsupported_contract_fail_open(beam):
    proof = owned(beam)
    too_many = snapshot(*([proof] * (MAX_CAPTURES + 1)))
    assert {r["id"] for r in recall(beam, too_many)} == {proof.memory_id}
    assert {r["id"] for r in recall(beam, {content_hash("same text")})} == {proof.memory_id}


def test_many_equal_content_owned_rows_cannot_starve_survivor(beam):
    # One distinct content hash, MANY excluded row IDs above the survivor.
    proofs = [owned(beam, f"owned-{i}") for i in range(65)]
    row(beam, "survivor", "orchard botanical archive itinerary plus extensive historical notes " * 8, age=1)
    results = recall(beam, snapshot(*proofs), top_k=1)
    assert {r["id"] for r in results} == {"survivor"}


def test_duplicate_unowned_rows_are_not_counted_as_excluded(beam):
    proof = owned(beam)
    for i in range(65):
        row(beam, f"copy-{i}", "[USER] orchard botanical archive itinerary")
    results = recall(beam, snapshot(proof))
    assert results
    assert all(r["id"].startswith("copy-") for r in results)


def episodic(beam, mid, content):
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, importance, session_id)"
        " VALUES (?, ?, 'summary', ?, 0.5, ?)",
        (mid, content, datetime.now().isoformat(), beam.session_id),
    )
    beam.conn.execute("INSERT INTO memory_embeddings(memory_id, embedding_json) VALUES (?, ?)",
                     (mid, json.dumps([1.0, 0.0, 0.0])))
    beam.conn.commit()


def test_same_id_episodic_twin_keeps_own_content_tier_and_score(beam, monkeypatch):
    proof = owned(beam)
    summary = "Distilled orchard botanical archive itinerary decision"
    episodic(beam, proof.memory_id, summary)
    from mnemosyne.core import embeddings
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "embed", lambda _: np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    engine = beam._get_polyphonic_engine()
    # Establish the actual episodic vector signal, without a temporal WM boost.
    only_episode = engine._vector_voice(np.array([1.0, 0.0, 0.0]), source=None,
                                       excluded_wm_ids={proof.memory_id})
    assert len(only_episode) == 1
    assert only_episode[0].metadata["embedding_tier"] == "episodic"
    filtered = engine.recall("recent orchard", np.array([1.0, 0.0, 0.0]),
                             source=None, exclude_captures=snapshot(proof))
    assert len(filtered) == 1
    assert filtered[0].content == summary
    assert filtered[0].metadata["embedding_tier"] == "episodic"
    assert set(filtered[0].voice_scores) == {"vector"}
    assert filtered[0].combined_score == pytest.approx(1 / 61)
    results = recall(beam, snapshot(proof), source=None)
    assert len(results) == 1
    assert results[0]["content"] == summary
    assert results[0]["tier"] == "episodic"


@pytest.mark.parametrize("column,value", [("valid_until", "2000-01-01T00:00:00Z"),
                                         ("superseded_by", "new-summary")])
def test_ineligible_episodic_twin_never_rescues_working_content(beam, monkeypatch, column, value):
    proof = owned(beam)
    episodic(beam, proof.memory_id, "Invalidated orchard botanical archive summary")
    beam.conn.execute(f"UPDATE episodic_memory SET {column}=? WHERE id=?", (value, proof.memory_id))
    beam.conn.commit()
    from mnemosyne.core import embeddings
    monkeypatch.setattr(embeddings, "available", lambda: True)
    monkeypatch.setattr(embeddings, "embed", lambda _: np.array([[1.0, 0.0, 0.0]], dtype=np.float32))
    assert recall(beam, snapshot(proof)) == []


def test_vector_excludes_before_cross_tier_dedup_and_top20(beam):
    proofs = [owned(beam, f"owned-vector-{i}") for i in range(25)]
    for proof in proofs:
        beam.conn.execute("INSERT INTO memory_embeddings(memory_id, embedding_json) VALUES (?, ?)",
                          (proof.memory_id, json.dumps([1.0, 0.0, 0.0])))
    row(beam, "dense-survivor", "orchard archive notes")
    beam.conn.execute("INSERT INTO memory_embeddings(memory_id, embedding_json) VALUES (?, ?)",
                      ("dense-survivor", json.dumps([0.8, 0.6, 0.0])))
    beam.conn.commit()
    engine = beam._get_polyphonic_engine()
    results = engine._vector_voice(np.array([1.0, 0.0, 0.0]), source="conversation",
                                    excluded_wm_ids=resolve_exclusions(beam.conn, snapshot(*proofs)))
    assert [r.memory_id for r in results] == ["dense-survivor"]


def test_untyped_graph_hits_abstain_instead_of_relabeling_working_score(beam, monkeypatch):
    proof = owned(beam)
    engine = beam._get_polyphonic_engine()
    monkeypatch.setattr(engine, "_graph_voice", lambda _: [RecallResult(proof.memory_id, 0.9, "graph", {})])
    results = engine.recall("recent orchard", exclude_captures=snapshot(proof))
    assert [r.memory_id for r in results] == [proof.memory_id]
    assert "graph" in results[0].voice_scores


def test_content_equality_is_not_proof_of_session_or_nonce(beam):
    proof = owned(beam)
    wrong = replace(proof, session_id="caller-not-beam-session")
    assert resolve_exclusions(beam.conn, snapshot(wrong)) == set()
    wrong = replace(proof, nonce="unrelated-capture")
    assert resolve_exclusions(beam.conn, snapshot(wrong)) == set()


def test_small_sqlite_variable_budget_abstains_without_recall_loss(beam):
    import sqlite3
    if not hasattr(beam.conn, "setlimit"):
        pytest.skip("SQLite runtime-limit mutation requires Python 3.11+")
    proof = owned(beam)
    previous = beam.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 100)
    try:
        assert resolve_exclusions(beam.conn, snapshot(proof)) == set()
        assert {r["id"] for r in recall(beam, snapshot(proof))} == {proof.memory_id}
    finally:
        beam.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)


def test_resolution_uses_bounded_primary_key_queries_not_full_bank_scan(beam):
    proof = owned(beam)
    sql = []
    beam.conn.set_trace_callback(sql.append)
    assert resolve_exclusions(beam.conn, snapshot(proof)) == {proof.memory_id}
    beam.conn.set_trace_callback(None)
    assert len(sql) == 1
    assert "WHERE id IN (" in sql[0]
    plan = beam.conn.execute("EXPLAIN QUERY PLAN SELECT * FROM working_memory WHERE id IN (?)", (proof.memory_id,)).fetchall()
    assert any("SEARCH" in r[3] and "INDEX" in r[3] for r in plan)
