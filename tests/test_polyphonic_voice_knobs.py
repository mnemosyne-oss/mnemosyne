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
