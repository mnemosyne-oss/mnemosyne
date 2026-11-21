"""Per-voice retention floor and declared voice weights reach fusion.

Two opt-in knobs on the polyphonic engine, default behavior unchanged:

- `MNEMOSYNE_VOICE_RETENTION` raises the per-voice retention FLOOR
  (default 20). A target ranked deeper than the cap can never surface no
  matter how good its score, so the cap is itself a recall ceiling for
  voice arms with long candidate tails.
- `MNEMOSYNE_VOICE_WEIGHING` multiplies each RRF contribution by
  `self.voice_weights`, the weights the constructor declares ("learned
  from validation") but `_combine_voices` never read. Without them fusion
  is flat equal-weight RRF: a distractor surfaced by two weaker voices
  outranks a target the dense voice ranks first.
"""

from __future__ import annotations

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.polyphonic_recall import (
    PolyphonicRecallEngine,
    RecallResult,
    _voice_retention,
    _voice_weighting_enabled,
)

RRF_K = 60  # local constant inside _combine_voices


def test_voice_retention_default_and_validation(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_VOICE_RETENTION", raising=False)
    assert _voice_retention() == 20
    monkeypatch.setenv("MNEMOSYNE_VOICE_RETENTION", "60")
    assert _voice_retention() == 60
    monkeypatch.setenv("MNEMOSYNE_VOICE_RETENTION", "0")
    assert _voice_retention() == 20
    monkeypatch.setenv("MNEMOSYNE_VOICE_RETENTION", "many")
    assert _voice_retention() == 20


def test_weighting_toggle_opt_in(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_VOICE_WEIGHING", raising=False)
    assert _voice_weighting_enabled() is False
    monkeypatch.setenv("MNEMOSYNE_VOICE_WEIGHING", "on")
    assert _voice_weighting_enabled() is True


def _engine(tmp_path) -> PolyphonicRecallEngine:
    beam = BeamMemory(session_id="knobs", db_path=tmp_path / "p.db")
    return PolyphonicRecallEngine(db_path=beam.db_path, conn=beam.conn)


def _results():
    # One target ranked first by the vector voice.
    vector_results = [
        RecallResult(memory_id="target", score=0.9, voice="vector", metadata={})
    ]
    graph_results = [
        RecallResult(memory_id="distractor-a", score=0.5, voice="graph", metadata={})
    ]
    return vector_results, graph_results


def test_fusion_is_flat_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_VOICE_WEIGHING", raising=False)
    engine = _engine(tmp_path)
    vector_results, graph_results = _results()
    combined = engine._combine_voices(vector_results, graph_results, [], [])
    assert abs(combined["target"].voice_scores["vector"] - 1.0 / (RRF_K + 1)) < 1e-12


def test_declared_weights_reach_fusion_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_VOICE_WEIGHING", "1")
    engine = _engine(tmp_path)
    vector_results, graph_results = _results()
    combined = engine._combine_voices(vector_results, graph_results, [], [])
    expected = (1.0 / (RRF_K + 1)) * engine.voice_weights["vector"]
    assert abs(combined["target"].voice_scores["vector"] - expected) < 1e-12
    assert combined["target"].combined_score == pytest.approx(expected)


def test_recall_uses_retention_floor(tmp_path, monkeypatch):
    engine = _engine(tmp_path)
    seen = {}
    real = engine._vector_voice

    def spy(*args, **kwargs):
        seen["limit"] = kwargs.get("limit")
        return real(*args, **kwargs)

    engine._vector_voice = spy
    monkeypatch.delenv("MNEMOSYNE_VOICE_RETENTION", raising=False)
    engine.recall("zulu yankee", query_embedding=None, top_k=5)
    assert seen["limit"] == 20  # default floor, unchanged
    monkeypatch.setenv("MNEMOSYNE_VOICE_RETENTION", "64")
    engine.recall("zulu yankee", query_embedding=None, top_k=5)
    assert seen["limit"] == 64  # floor raised above max(20, top_k)
    monkeypatch.setenv("MNEMOSYNE_VOICE_RETENTION", "8")
    engine.recall("zulu yankee", query_embedding=None, top_k=30)
    assert seen["limit"] == 30  # top_k still wins over a low floor


# ---------------------------------------------------------------------------
# Result-level coverage: the voices must ACT on the limit, not just receive it
# ---------------------------------------------------------------------------

import json  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

from mnemosyne.core import beam as beam_module  # noqa: E402


def _seed_vector_candidates(beam: BeamMemory, query_vec: np.ndarray,
                            n_distractors: int, target_sim_rank: int) -> None:
    """Seed distractors nearly parallel to the query and one target below them.

    target_sim_rank only fixes the ORDER: the target sits at
    n_distractors + 1 by cosine similarity, so a retention limit at or above
    n_distractors + 1 admits it and a lower limit must drop it.
    """
    rng = np.random.RandomState(7)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    orth = rng.randn(beam_module.EMBEDDING_DIM).astype(np.float32)
    orth /= np.linalg.norm(orth)
    vectors = {}
    for i in range(n_distractors):
        noise = rng.randn(beam_module.EMBEDDING_DIM).astype(np.float32) * 0.01
        v = query_vec + noise
        vectors[f"distractor-{i}"] = v / np.linalg.norm(v)
    v = 0.5 * query_vec + np.sqrt(0.75) * orth
    vectors["deep-target"] = v / np.linalg.norm(v)
    for memory_id, vec in vectors.items():
        beam.conn.execute(
            "INSERT INTO working_memory "
            "(id, content, source, timestamp, importance, session_id, scope) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (memory_id, f"wug qlorf body {memory_id}", "fact", now,
             0.6, "knobs", "session"),
        )
        beam.conn.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding_json, model) "
            "VALUES (?, ?, ?)",
            (memory_id, json.dumps(vec.astype(np.float32).tolist()), "test-model"),
        )
    beam.conn.commit()


def test_vector_voice_actually_honors_the_retention_limit(tmp_path):
    """The limit must cut candidates, not just be forwarded."""
    beam = BeamMemory(session_id="knobs", db_path=tmp_path / "v.db")
    query_vec = np.zeros(beam_module.EMBEDDING_DIM, dtype=np.float32)
    query_vec[0] = 1.0
    _seed_vector_candidates(beam, query_vec, n_distractors=24, target_sim_rank=25)
    engine = PolyphonicRecallEngine(db_path=beam.db_path, conn=beam.conn)

    capped = engine._vector_voice(query_vec, limit=20)
    capped_ids = {r.memory_id for r in capped}
    assert len(capped) == 20, "the vector voice ignored the limit"
    assert "deep-target" not in capped_ids, (
        "rank 25 must not survive a retention of 20"
    )

    deep = engine._vector_voice(query_vec, limit=64)
    deep_ids = {r.memory_id for r in deep}
    assert len(deep) == 25
    assert "deep-target" in deep_ids, (
        "a retention of 64 must reach the rank-25 candidate"
    )


def test_temporal_voice_actually_honors_the_retention_limit(tmp_path):
    beam = BeamMemory(session_id="knobs", db_path=tmp_path / "t.db")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    for i in range(25):
        beam.conn.execute(
            "INSERT INTO working_memory "
            "(id, content, source, timestamp, importance, session_id, scope) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (f"row-{i}", f"recent note {i} alpha omega", "fact", now,
             0.6, "knobs", "session"),
        )
    beam.conn.commit()
    engine = PolyphonicRecallEngine(db_path=beam.db_path, conn=beam.conn)

    capped = engine._temporal_voice("recent note", limit=10)
    assert 0 < len(capped) == 10, "the temporal voice ignored the limit"
    deep = engine._temporal_voice("recent note", limit=64)
    assert len(deep) == 25, "the temporal voice capped below the seeded rows"


# ---------------------------------------------------------------------------
# Weighting regressions
# ---------------------------------------------------------------------------

def test_graph_traversal_results_carry_the_graph_weight(tmp_path, monkeypatch):
    """graph_traversal is graph evidence with its own attribution label.

    Before the mapping, the label missed voice_weights and fell back to 1.0:
    a traversal rank-2 hit (1/62 ~= 0.0161) then OUTRANKED the vector voice's
    rank 1 (0.35/61 ~= 0.0057) - weighting inverted what flat fusion scored.
    """
    monkeypatch.setenv("MNEMOSYNE_VOICE_WEIGHING", "1")
    engine = _engine(tmp_path)
    vector_results = [
        RecallResult(memory_id="target", score=0.9, voice="vector", metadata={})
    ]
    graph_results = [
        RecallResult(memory_id="graph-seed", score=0.9, voice="graph", metadata={}),
        RecallResult(memory_id="graph-walk", score=0.8, voice="graph_traversal", metadata={}),
    ]
    combined = engine._combine_voices(vector_results, graph_results, [], [])

    graph_weight = engine.voice_weights["graph"]
    # Attribution label preserved, weight read from "graph".
    assert combined["graph-walk"].voice_scores["graph_traversal"] == pytest.approx(
        (1.0 / (RRF_K + 2)) * graph_weight
    )
    assert combined["graph-seed"].voice_scores["graph"] == pytest.approx(
        (1.0 / (RRF_K + 1)) * graph_weight
    )
    # A traversal rank 2 must not outrank the vector voice's rank 1.
    assert combined["target"].combined_score > combined["graph-walk"].combined_score


def test_weighted_fusion_sums_multi_voice_contributions(tmp_path, monkeypatch):
    """A memory surfaced by two voices accumulates both weighted RRF terms."""
    monkeypatch.setenv("MNEMOSYNE_VOICE_WEIGHING", "1")
    engine = _engine(tmp_path)
    vector_results = [
        RecallResult(memory_id="shared", score=0.9, voice="vector", metadata={})
    ]
    fact_results = [
        RecallResult(memory_id="shared", score=0.8, voice="fact", metadata={}),
        RecallResult(memory_id="fact-only", score=0.7, voice="fact", metadata={}),
    ]
    combined = engine._combine_voices(vector_results, [], fact_results, [])

    expected = (1.0 / (RRF_K + 1)) * engine.voice_weights["vector"] + (
        1.0 / (RRF_K + 1)
    ) * engine.voice_weights["fact"]
    assert combined["shared"].combined_score == pytest.approx(expected)
    # The multi-voice memory still outranks a single-voice fact hit ranked below it.
    assert combined["shared"].combined_score > combined["fact-only"].combined_score
