"""Tests for the prefetch low-quality fragment filter.

The regex fact extractor can emit bare single-token "facts" (a stray adverb, a
particle, or a truncated word). Those tokens FTS-match common query words and can
score above real memories, so they must not dominate the per-turn prefetch window.
A real, injectable memory is a phrase; a lone short/stopword token is dropped.
"""
from __future__ import annotations

import pytest

from hermes_memory_provider import _is_low_quality_prefetch as is_junk


@pytest.mark.parametrize("frag", ["", "  ", "what", "very", "is", "still", "almost", "over"])
def test_lone_fragments_are_dropped(frag):
    assert is_junk(frag) is True


@pytest.mark.parametrize("real", [
    "Paris is the capital of France",
    "water boils at 100C",
    "film school",                 # two-word value survives
    "the meeting is on Tuesday",
])
def test_real_phrases_are_kept(real):
    assert is_junk(real) is False


class FakeBeam:
    """Returns a fixed mix of junk fragments and real phrases regardless of query."""
    author_id = "test-author"

    def recall(self, query, top_k, temporal_weight, temporal_halflife, author_id):
        return [
            # bare fragments scoring high on keyword match — must be filtered out
            {"content": "what", "timestamp": "2026-05-14T12:00:00Z",
             "importance": 0.1, "score": 0.63, "keyword_score": 0.63, "trust_tier": "STATED"},
            {"content": "still", "timestamp": "2026-05-14T12:00:00Z",
             "importance": 0.1, "score": 0.62, "keyword_score": 0.62, "trust_tier": "STATED"},
            # a real, sentence-length memory scoring lower — must survive
            {"content": "Paris is the capital of France", "timestamp": "2026-05-14T12:00:00Z",
             "importance": 0.6, "score": 0.40, "keyword_score": 0.40, "trust_tier": "STATED"},
        ]


def test_prefetch_drops_fragments_and_keeps_real_memory(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", raising=False)
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    provider._beam = FakeBeam()

    block = provider.prefetch("what matters most")

    assert "Paris is the capital of France" in block
    # The bare fragments must not appear as injected memory lines.
    for line in block.splitlines():
        assert line.strip() not in {"what", "still"}
