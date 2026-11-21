"""The prefetch gates must understand the 4.0.0 RRF score scale.

mnemosyne-memory 4.0.0 recall rows carry reciprocal-rank-fusion scores
(~1/(60+rank), so rank 1 in one voice is ~0.0164) and polyphonic rows put
their per-voice contributions under ``voice_scores`` instead of
``keyword_score``/``fts_score``/``dense_score``. The prefetch admission gates
(min_score 0.20, min_topic_signal 0.08) were calibrated against the older 0-1
weighted-sum scale, so every row failed both gates and the automatic
injection block silently rendered empty: on a long-lived store this starved
the bank source completely (measured in production: prefetch carried no
memory rows for days and nothing errored).
"""
from __future__ import annotations

import pytest

from mnemosyne.core.prefetch import (
    PrefetchProfile,
    _prefetch_rrf_norm,
    _prefetch_topic_signal,
    render_bank_source,
)


def test_rrf_norm_maps_rank_one_to_one():
    # rank 1 in one voice: 1/(60+1) -> 1.0 on the gate scale
    assert abs(_prefetch_rrf_norm(1.0 / 61.0) - 1.0) < 1e-9


def test_rrf_norm_env_override_and_garbage(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_RRF_NORM_K", "101")
    assert abs(_prefetch_rrf_norm(1.0 / 101.0) - 1.0) < 1e-9
    monkeypatch.setenv("MNEMOSYNE_RRF_NORM_K", "not-a-number")
    assert abs(_prefetch_rrf_norm(1.0 / 61.0) - 1.0) < 1e-9
    monkeypatch.setenv("MNEMOSYNE_RRF_NORM_K", "0")  # clamped to >= 1
    assert _prefetch_rrf_norm(1.0) == 1.0


def test_topic_signal_reads_voice_scores():
    row = {
        "voice_scores": {
            "keyword": 0.016393,  # rank 1
            "fts": 0.015873,      # rank 2
        }
    }
    assert _prefetch_topic_signal(row) >= 0.08  # general profile gate floor


def test_topic_signal_ignores_non_numeric_voice_entries():
    row = {"voice_scores": {"vec": True, "fts": "high", "keyword": 0.0164}}
    signal = _prefetch_topic_signal(row)
    assert abs(signal - _prefetch_rrf_norm(0.0164)) < 1e-9


def test_topic_signal_legacy_rows_unchanged():
    row = {"keyword_score": 0.5, "dense_score": 0.3}
    assert _prefetch_topic_signal(row) == 0.5


class _RrfBeam:
    """A 4.0.0-style beam: fused RRF score, per-voice RRF contributions."""

    def recall(self, **kwargs):
        return [
            {
                "id": "a1",
                "content": "Paris is the capital of France",
                "timestamp": "2026-11-03T09:00:00",
                "importance": 0.6,
                "score": 0.0325,  # fused RRF: rank 1 in two voices
                "voice_scores": {"fts": 0.0164, "dense_score": 0.0161},
                "trust_tier": "STATED",
                "source": "fact",
            }
        ]


def test_bank_source_is_not_starved_by_rrf_scale(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_RRF_NORM_K", raising=False)
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", raising=False)
    block = render_bank_source(
        _RrfBeam(), "capital of France", "s1", PrefetchProfile(name="test")
    )
    assert "Paris is the capital of France" in block
