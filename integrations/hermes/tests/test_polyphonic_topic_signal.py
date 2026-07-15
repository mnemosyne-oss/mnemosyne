"""Polyphonic-to-provider topical signal contract."""
from __future__ import annotations

from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine, RecallResult
from mnemosyne_hermes import _prefetch_topic_signal


def test_combine_voices_preserves_raw_per_voice_scores():
    engine = object.__new__(PolyphonicRecallEngine)
    combined = engine._combine_voices(
        [RecallResult(memory_id="m1", score=0.82, voice="vector", metadata={})],
        [RecallResult(memory_id="m1", score=0.61, voice="fact", metadata={})],
    )

    assert combined["m1"].metadata["raw_voice_scores"] == {
        "vector": 0.82,
        "fact": 0.61,
    }


def test_provider_prefetch_uses_explicit_topic_signal():
    assert _prefetch_topic_signal({
        "score": 0.04,
        "keyword_score": 0.0,
        "fts_score": 0.0,
        "dense_score": 0.0,
        "topic_signal": 0.73,
    }) == 0.73
