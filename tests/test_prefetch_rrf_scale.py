"""The prefetch gates must understand the 4.0.0 RRF score scale.

mnemosyne-memory 4.0.0 polyphonic recall rows carry reciprocal-rank-fusion
scores (~1/(60+rank), so rank 1 in one voice is ~0.0164) and put their
per-voice contributions under ``voice_scores`` keyed by VOICE name
(vector/temporal/graph/graph_traversal/fact) instead of
``keyword_score``/``fts_score``/``dense_score``. The prefetch admission gates
(min_score 0.20, min_topic_signal 0.08) were calibrated against the older 0-1
weighted-sum scale, so every row failed both gates and the automatic
injection block silently rendered empty: on a long-lived store this starved
the bank source completely (measured in production: prefetch carried no
memory rows for days and nothing errored).

The linear path also emits ``voice_scores`` (beam.py "Gap G" parity block),
keyed by SIGNAL name (vec/fts/keyword) plus the ranking inputs ``importance``
and ``recency_decay``, with 0-1 values. Both shapes are exercised here with
producer-faithful fixtures: rescaling must apply only to RRF-shaped rows, and
recency/importance-flavoured entries must never satisfy the topic gate.
"""
from __future__ import annotations

import pytest

from mnemosyne.core.prefetch import (
    PrefetchProfile,
    _prefetch_gate_score,
    _prefetch_is_rrf_row,
    _prefetch_rrf_norm,
    _prefetch_topic_signal,
    render_bank_source,
)


@pytest.fixture(autouse=True)
def _default_norm_k(monkeypatch):
    """Default-mapping tests must not inherit an operator override."""
    monkeypatch.delenv("MNEMOSYNE_RRF_NORM_K", raising=False)
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", raising=False)


def _linear_row(**overrides):
    """A row shaped by the LINEAR producer (beam.py 0-1 weighted-sum path)."""
    row = {
        "id": "lin1",
        "content": "Kegeling presses were rebuilt in 2026-05",
        "timestamp": "2026-11-01T09:00:00",
        "importance": 0.5,
        "recency_decay": 0.9,
        "score": 0.45,
        "keyword_score": 0.4,
        "fts_score": 0.5,
        "dense_score": 0.45,
        "trust_tier": "STATED",
        "source": "fact",
    }
    # Gap G parity block: same numbers, mirrored into voice_scores.
    row["voice_scores"] = {
        "vec": row["dense_score"],
        "fts": row["fts_score"],
        "keyword": row["keyword_score"],
        "importance": row["importance"],
        "recency_decay": row["recency_decay"],
    }
    row.update(overrides)
    return row


def _rrf_row(**overrides):
    """A row shaped by the POLYPHONIC RRF producer (4.0.0 fusion path)."""
    row = {
        "id": "rrf1",
        "content": "Paris is the capital of France",
        "timestamp": "2026-11-03T09:00:00",
        "importance": 0.6,
        "score": 0.0325,  # fused RRF: rank 1 + rank 2 across two voices
        "voice_scores": {"vector": 1.0 / 61.0, "fact": 1.0 / 62.0},
        "trust_tier": "STATED",
        "source": "fact",
    }
    row.update(overrides)
    return row


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


def test_scale_contract_distinguishes_the_two_producers():
    assert _prefetch_is_rrf_row(_rrf_row())
    assert not _prefetch_is_rrf_row(_linear_row())
    # A row without voice_scores at all (older builds) stays on the 0-1 scale.
    assert not _prefetch_is_rrf_row({"score": 0.3})


def test_gate_score_rescales_only_rrf_rows():
    assert _prefetch_gate_score(_rrf_row()) == pytest.approx(0.0325 * 61.0)
    # A legacy 0.10 must stay 0.10, not inflate past min_score.
    assert _prefetch_gate_score(_linear_row(score=0.10)) == pytest.approx(0.10)


def test_topic_signal_reads_topical_rrf_voices():
    row = _rrf_row(voice_scores={"vector": 1.0 / 61.0})  # rank 1 in vector
    assert _prefetch_topic_signal(row) >= 0.9  # ~1.0 after rescale


def test_topic_signal_rejects_rrf_temporal_only_row():
    # The temporal voice is recency, not topicality: rank 1 there must not
    # manufacture a topic signal.
    row = _rrf_row(voice_scores={"temporal": 1.0 / 61.0})
    assert _prefetch_topic_signal(row) == 0.0


def test_topic_signal_rejects_linear_importance_only_row():
    # No keyword/fts/dense evidence, but high importance + recency: the
    # linear voice_scores mirror those ranking inputs, and they are NOT
    # topical evidence ("importance alone is not enough").
    row = _linear_row(
        keyword_score=0.0, fts_score=0.0, dense_score=0.0,
        importance=0.9, recency_decay=0.99,
    )
    assert _prefetch_topic_signal(row) == 0.0


def test_topic_signal_linear_rows_use_top_level_fields_unchanged():
    # Linear values are already 0-1 and must NOT be rescaled via voice_scores.
    row = _linear_row(keyword_score=0.02, fts_score=0.05, dense_score=0.03)
    assert _prefetch_topic_signal(row) == pytest.approx(0.05)


def test_topic_signal_ignores_non_numeric_voice_entries():
    row = _rrf_row(voice_scores={"vector": True, "graph": "high", "fact": 0.0164})
    signal = _prefetch_topic_signal(row)
    assert signal == pytest.approx(_prefetch_rrf_norm(0.0164))


def test_topic_signal_legacy_rows_unchanged():
    row = {"keyword_score": 0.5, "dense_score": 0.3}
    assert _prefetch_topic_signal(row) == 0.5


class _StubBeam:
    """A beam returning fixed rows, shaped by a chosen producer."""

    def __init__(self, rows):
        self._rows = rows

    def recall(self, **kwargs):
        return [dict(r) for r in self._rows]


def _render(rows, profile=None):
    return render_bank_source(
        _StubBeam(rows), "capital of France", "s1",
        profile or PrefetchProfile(name="test"),
    )


def test_bank_source_is_not_starved_by_rrf_scale():
    assert "Paris is the capital of France" in _render([_rrf_row()])


def test_bank_source_admits_rrf_graph_and_traversal_voices():
    row = _rrf_row(
        content="The graph linked the D91SE to the DAC noise floor",
        voice_scores={"graph": 1.0 / 63.0, "graph_traversal": 1.0 / 64.0},
        score=1.0 / 63.0,
    )
    assert "D91SE" in _render([row])


def test_bank_source_rejects_low_score_legacy_row():
    # Pre-PR regression: an unconditional x61 made this legacy row (score
    # 0.10, topic 0.10) pass min_score. It must stay rejected: both score
    # and importance are below the fallback pair's thresholds.
    row = _linear_row(
        score=0.10, keyword_score=0.10, fts_score=0.10, dense_score=0.10,
        importance=0.10, recency_decay=0.10,
        content="Legacy row with only weak evidence",
    )
    assert "Legacy row" not in _render([row])


def test_bank_source_rejects_importance_only_linear_row():
    row = _linear_row(
        keyword_score=0.0, fts_score=0.0, dense_score=0.0,
        importance=0.9, recency_decay=0.99, score=0.5,
        content="Important but topically irrelevant",
    )
    assert "Important but topically irrelevant" not in _render([row])


def test_bank_source_rejects_recency_only_rrf_row():
    row = _rrf_row(
        content="Recently touched, topically irrelevant",
        voice_scores={"temporal": 1.0 / 61.0}, score=1.0 / 61.0,
    )
    assert "Recently touched" not in _render([row])


def test_bank_source_raw_rows_need_stronger_signal():
    # signal 0.12 passes the distilled floor (0.08) but not the raw floor
    # (0.18): same evidence, different source, different admission.
    distilled = _linear_row(
        source="fact", keyword_score=0.12, fts_score=0.12, dense_score=0.12,
        score=0.30, content="Distilled note about the French press",
    )
    raw = _linear_row(
        source="conversation", keyword_score=0.12, fts_score=0.12,
        dense_score=0.12, score=0.30,
        content="Raw transcript about the French press",
    )
    assert "Distilled note" in _render([distilled])
    assert "Raw transcript" not in _render([raw])


def test_bank_source_orders_by_adjusted_score_on_the_shared_scale():
    # Same voice, different ranks: rescaling puts rank 1 clearly above rank 5
    # on the gate scale even though the raw RRF numbers differ by ~6%.
    strong = _rrf_row(
        id="a", content="Paris is a capital",
        voice_scores={"vector": 1.0 / 61.0}, score=1.0 / 61.0,
    )
    weak = _rrf_row(
        id="b", content="Lyon is a city",
        voice_scores={"vector": 1.0 / 65.0}, score=1.0 / 65.0,
    )
    block = _render([weak, strong])
    assert block.index("Paris") < block.index("Lyon")


def test_bank_source_caps_at_top_k():
    rows = [
        _rrf_row(id=f"r{i}", content=f"alpha {['bridge','canyon','delta','ember','flint'][i]} anchor")
        for i in range(5)
    ]
    block = _render(rows, PrefetchProfile(name="test", top_k=2))
    assert block.count("anchor") == 2
