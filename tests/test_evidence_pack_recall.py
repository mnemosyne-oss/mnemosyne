"""Read-only candidate-recall contract for future evidence packs."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core import beam as beam_module


def _state(db_path: Path, memory_id: str) -> tuple[int, str | None]:
    with sqlite3.connect(str(db_path)) as conn:
        count, recalled_at = conn.execute(
            "SELECT recall_count, last_recalled FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()
    return (count or 0, recalled_at)




def test_evidence_pack_rejects_invalid_internal_controls(tmp_path: Path):
    beam = BeamMemory(session_id="guards", db_path=tmp_path / "guards.db")
    with pytest.raises(ValueError, match="candidate_k"):
        beam.recall_with_evidence_pack("q", top_k=2, candidate_k=1)
    with pytest.raises(ValueError, match="pack_k"):
        beam.recall_with_evidence_pack("q", pack_k=-1)
    with pytest.raises(ValueError, match="internal recall controls"):
        beam.recall_with_evidence_pack("q", _track_recall=False)


def test_candidate_only_recall_does_not_mutate_usage_state(tmp_path: Path):
    db_path = tmp_path / "evidence-pack.db"
    beam = BeamMemory(session_id="evidence-pack", db_path=db_path)
    memory_id = beam.remember("Orion evidence-pack candidate fixture", source="test", importance=0.8)

    before = _state(db_path, memory_id)
    results = beam.recall("Orion evidence-pack", top_k=5, _track_recall=False)
    after_candidate = _state(db_path, memory_id)
    assert memory_id in [row["id"] for row in results]
    assert after_candidate == before

    beam.recall("Orion evidence-pack", top_k=5)
    after_normal = _state(db_path, memory_id)
    assert after_normal[0] == before[0] + 1
    assert after_normal[1] is not None




def test_candidate_only_recall_does_not_mutate_episodic_usage_state(tmp_path: Path):
    db_path = tmp_path / "episodic-telemetry.db"
    beam = BeamMemory(session_id="episodic", db_path=db_path)
    memory_id = "episodic-telemetry-id"
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (memory_id, "episodic telemetry unique phrase", "test", "2026-01-01T00:00:00", "episodic", 0.8, "session"),
    )
    beam.conn.commit()
    before = beam.conn.execute("SELECT recall_count, last_recalled FROM episodic_memory WHERE id = ?", (memory_id,)).fetchone()
    results = beam.recall("episodic telemetry unique phrase", top_k=5, _track_recall=False)
    after = beam.conn.execute("SELECT recall_count, last_recalled FROM episodic_memory WHERE id = ?", (memory_id,)).fetchone()
    assert memory_id in [row["id"] for row in results]
    assert after == before




def test_real_recall_uses_working_and_episodic_tier_labels(tmp_path: Path):
    db_path = tmp_path / "tier-labels.db"
    beam = BeamMemory(session_id="tiers", db_path=db_path)
    working_id = beam.remember("working tier unique phrase", source="test", importance=0.8)
    episodic_id = "episodic-tier-id"
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (episodic_id, "episodic tier unique phrase", "test", "2026-01-01T00:00:00", "tiers", 0.8, "session"),
    )
    beam.conn.commit()

    working = beam.recall("working tier unique phrase", top_k=5, _track_recall=False)
    episodic = beam.recall("episodic tier unique phrase", top_k=5, _track_recall=False)
    assert next(row for row in working if row["id"] == working_id)["tier"] == "working"
    assert next(row for row in episodic if row["id"] == episodic_id)["tier"] == "episodic"


def test_evidence_pack_keeps_primary_ranking_and_bounds_supplement(tmp_path: Path):
    db_path = tmp_path / "pack.db"
    beam = BeamMemory(session_id="evidence-pack", db_path=db_path)
    for index in range(4):
        beam.remember(f"Orion evidence fixture {index}", source="test", importance=0.8)

    primary = beam.recall("Orion evidence", top_k=2)
    packed = beam.recall_with_evidence_pack("Orion evidence", top_k=2, candidate_k=4, pack_k=1)

    assert [row["id"] for row in packed["primary"]] == [row["id"] for row in primary]
    assert len(packed["evidence_pack"]) <= 1
    assert {row["id"] for row in packed["primary"]}.isdisjoint(
        {row["id"] for row in packed["evidence_pack"]}
    )


def test_evidence_pack_does_not_leak_other_session_rows(tmp_path: Path):
    db_path = tmp_path / "scope.db"
    local = BeamMemory(session_id="local", db_path=db_path)
    other = BeamMemory(session_id="other", db_path=db_path)
    local_id = local.remember("Orion local scope fixture", source="test", importance=0.8)
    other_id = other.remember("Orion other scope fixture", source="test", importance=0.8)

    packed = local.recall_with_evidence_pack("Orion scope", top_k=1, candidate_k=5, pack_k=5)
    returned_ids = {row["id"] for row in packed["primary"] + packed["evidence_pack"]}

    assert local_id in returned_ids
    assert other_id not in returned_ids


def test_cross_session_scope_can_produce_a_supplemental_pack(tmp_path: Path):
    db_path = tmp_path / "cross-session-pack.db"
    reader = BeamMemory(session_id="reader", db_path=db_path)
    first = BeamMemory(session_id="first", db_path=db_path)
    second = BeamMemory(session_id="second", db_path=db_path)
    first.remember("Orion cross session alpha", source="test", importance=0.8, scope="global")
    second_id = second.remember("Orion cross session beta", source="test", importance=0.8, scope="global")

    packed = reader.recall_with_evidence_pack(
        "Orion cross session", top_k=1, candidate_k=5, pack_k=5, _cross_session=True
    )
    assert len(packed["evidence_pack"]) == 1
    assert [row["evidence_rank"] for row in packed["evidence_pack"]] == [2]
    assert second_id in {row["id"] for row in packed["primary"] + packed["evidence_pack"]}


def test_vector_only_candidate_is_opt_in(tmp_path: Path, monkeypatch):
    np = pytest.importorskip("numpy")
    db_path = tmp_path / "vector-only.db"
    beam = BeamMemory(session_id="local", db_path=db_path)
    memory_id = beam.remember("semantic-only candidate", source="test", importance=0.8)

    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda query: np.array([1.0], dtype=np.float32))
    monkeypatch.setattr(beam_module, "_vec_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        beam_module, "_wm_vec_search",
        lambda conn, emb, k=20, **kwargs: [{"id": memory_id, "sim": 0.9}],
    )

    primary = beam.recall("unrelated lexical query", top_k=5)
    candidates = beam.recall(
        "unrelated lexical query", top_k=5,
        _track_recall=False, _include_vector_only_candidates=True,
    )
    assert memory_id not in [row["id"] for row in primary]
    assert memory_id in [row["id"] for row in candidates]

    monkeypatch.setattr(
        beam_module, "_wm_vec_search",
        lambda conn, emb, k=20, **kwargs: [{"id": memory_id, "sim": 0.83}],
    )
    low_similarity = beam.recall(
        "unrelated lexical query", top_k=5,
        _track_recall=False, _include_vector_only_candidates=True,
    )
    assert memory_id not in [row["id"] for row in low_similarity]


def test_vector_only_candidate_obeys_session_scope(tmp_path: Path, monkeypatch):
    np = pytest.importorskip("numpy")
    db_path = tmp_path / "vector-scope.db"
    local = BeamMemory(session_id="local", db_path=db_path)
    foreign = BeamMemory(session_id="foreign", db_path=db_path)
    foreign_id = foreign.remember("foreign semantic candidate", source="test", importance=0.8)
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda query: np.array([1.0], dtype=np.float32))
    monkeypatch.setattr(beam_module, "_vec_search", lambda *args, **kwargs: [])
    monkeypatch.setattr(beam_module, "_wm_vec_search", lambda *args, **kwargs: [{"id": foreign_id, "sim": 0.9}])
    results = local.recall("unrelated lexical query", top_k=5, _track_recall=False, _include_vector_only_candidates=True)
    assert foreign_id not in [row["id"] for row in results]


def test_provenance_backfill_keeps_same_ids_separate_by_tier(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "collision.db"
    beam = BeamMemory(session_id="reader", db_path=db_path)
    shared_id = "same-id"
    now = "2026-01-01T00:00:00"
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (shared_id, "working", "test", now, "primary-session", 0.5, "global"),
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (shared_id, "episodic", "test", now, "candidate-session", 0.5, "global"),
    )
    beam.conn.commit()
    calls = iter([
        [{"id": shared_id, "tier": "working", "timestamp": now}],
        [{"id": shared_id, "tier": "episodic", "timestamp": now}],
    ])
    monkeypatch.setattr(beam, "recall", lambda *args, **kwargs: next(calls))

    packed = beam.recall_with_evidence_pack("q", top_k=1, candidate_k=2, pack_k=1)
    assert packed["primary"][0]["session_id"] == "primary-session"
    assert packed["evidence_pack"][0]["session_id"] == "candidate-session"
