"""Tests for the MNEMOSYNE_LEXICAL_GATE_MIN recall knob.

The lexical admission gate (_minimum_recall_relevance) historically returns
query-length-dependent thresholds (0.15 / 0.5 / 0.3). The knob overrides the
gate entirely (float 0.0-1.0), defaulting to the historical behaviour when
unset, so existing deployments are unaffected unless they opt in.
"""
from mnemosyne.core.beam import _minimum_recall_relevance


def test_default_thresholds_unchanged_when_env_unset(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_LEXICAL_GATE_MIN", raising=False)
    assert _minimum_recall_relevance([]) == 0.15
    assert _minimum_recall_relevance(["a"]) == 0.15
    assert _minimum_recall_relevance(["a", "b"]) == 0.15
    assert _minimum_recall_relevance(["a", "b", "c"]) == 0.5
    assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 0.3
    assert _minimum_recall_relevance(["a"] * 8) == 0.3


def test_zero_knob_admits_pure_vector_candidates(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", "0")
    assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 0.0
    assert _minimum_recall_relevance(["a", "b", "c"]) == 0.0
    assert _minimum_recall_relevance([]) == 0.0


def test_explicit_threshold_override(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", "0.4")
    assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 0.4
    assert _minimum_recall_relevance(["a"]) == 0.4


def test_knob_is_clamped_to_valid_domain(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", "1.7")
    assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 1.0
    monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", "-0.5")
    assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 0.0


def test_invalid_knob_falls_back_to_defaults(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", "not-a-number")
    assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 0.3
    assert _minimum_recall_relevance(["a", "b", "c"]) == 0.5
    assert _minimum_recall_relevance(["a"]) == 0.15


def test_non_finite_knob_falls_back_to_defaults(monkeypatch):
    # NaN and infinities are not valid gates: fall through to the defaults.
    for bad in ("nan", "inf", "-inf"):
        monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", bad)
        assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 0.3
        assert _minimum_recall_relevance(["a", "b", "c"]) == 0.5
        assert _minimum_recall_relevance(["a"]) == 0.15


def test_env_is_read_per_call(monkeypatch):
    # Changing the env between calls takes effect immediately (no caching).
    monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", "0.0")
    assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 0.0
    monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", "0.6")
    assert _minimum_recall_relevance(["a", "b", "c", "d"]) == 0.6
