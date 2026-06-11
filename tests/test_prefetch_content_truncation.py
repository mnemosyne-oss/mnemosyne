"""Regression tests for issue #132: prefetch memory content truncation.

The provider used to hard-cap each recalled memory at 200 characters inside
``<memory-context>``. That drops the most specific facts from long,
LLM-authored memories. The default should preserve the full recalled content,
while still allowing operators to opt into a prompt-budget cap.
"""
from __future__ import annotations


class FakeBeam:
    author_id = "test-author"

    def __init__(self, content: str):
        self.content = content

    def recall(self, query, top_k, temporal_weight, temporal_halflife, author_id):
        return [
            {
                "content": self.content,
                "timestamp": "2026-05-14T12:00:00Z",
                "importance": 0.9,
                "score": 0.9,
                "keyword_score": 0.9,
                "source": "correction",
                "trust_tier": "STATED",
            }
        ]


def test_prefetch_preserves_full_memory_content_by_default(monkeypatch):
    """Facts after char 200 must survive in the injected memory context."""
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", raising=False)

    from hermes_memory_provider import MnemosyneMemoryProvider

    long_content = f"{'context ' * 35}critical-tail-fact-survives"
    assert len(long_content) > 250

    provider = MnemosyneMemoryProvider()
    provider._beam = FakeBeam(long_content)

    rendered = provider.prefetch("critical tail fact")

    assert "critical-tail-fact-survives" in rendered
    assert long_content in rendered
    assert "..." not in rendered


def test_prefetch_content_limit_is_opt_in_and_word_boundary(monkeypatch):
    """Positive env var values cap content without splitting mid-word."""
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", "12")

    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    provider._beam = FakeBeam("alpha beta gamma delta")

    rendered = provider.prefetch("alpha")

    assert "alpha beta..." in rendered
    assert "alpha beta g..." not in rendered
    assert "gamma delta" not in rendered


def test_prefetch_content_limit_zero_means_no_truncation(monkeypatch):
    """Explicit zero is the documented escape hatch for complete content."""
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", "0")

    from hermes_memory_provider import _format_prefetch_content

    assert _format_prefetch_content("alpha beta gamma", 0) == "alpha beta gamma"
