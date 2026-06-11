from __future__ import annotations

from mnemosyne_hermes import MnemosyneMemoryProvider


class FakeBeam:
    author_id = "test-author"

    def __init__(self, results=None):
        self.results = results or []
        self.writes = []

    def recall(self, **kwargs):
        self.last_kwargs = kwargs
        return self.results

    def remember(self, **kwargs):
        self.writes.append(kwargs)


def _provider(results):
    p = MnemosyneMemoryProvider()
    p._beam = FakeBeam(results)
    p._agent_context = "primary"
    return p


def test_prefetch_excludes_assistant_transcript_rows():
    p = _provider([
        {"content": "[ASSISTANT] stale answer that should not inject", "source": "conversation",
         "timestamp": "2026-06-11T09:00:00Z", "importance": 1.0, "score": 1.0,
         "keyword_score": 1.0, "trust_tier": "STATED"},
        {"content": "Mnemosyne injection should prefer distilled correction memories.",
         "source": "correction", "timestamp": "2026-06-11T09:01:00Z",
         "importance": 0.8, "score": 0.7, "keyword_score": 0.7, "trust_tier": "STATED"},
    ])

    block = p.prefetch("Mnemosyne injection correction")

    assert "distilled correction" in block
    assert "[ASSISTANT]" not in block


def test_prefetch_requires_topic_signal_not_importance_only():
    p = _provider([
        {"content": "[USER] unrelated minecraft watcher cleanup", "source": "conversation",
         "timestamp": "2026-06-10T11:33:00Z", "importance": 0.99, "score": 0.9,
         "keyword_score": 0.02, "trust_tier": "STATED"},
        {"content": "Mnemosyne memory-context injection should be selected by topical relevance.",
         "source": "correction", "timestamp": "2026-06-11T09:01:00Z",
         "importance": 0.7, "score": 0.6, "keyword_score": 0.6, "trust_tier": "STATED"},
    ])

    block = p.prefetch("make Mnemosyne memory-context injection more relevant")

    assert "topical relevance" in block
    assert "minecraft watcher" not in block


def test_sync_roles_can_disable_assistant_autosave():
    p = MnemosyneMemoryProvider()
    p._beam = FakeBeam()
    p._agent_context = "primary"
    p._sync_roles = {"user"}

    p.sync_turn("please remember user side", "assistant side should not be stored")

    written = [w["content"] for w in p._beam.writes]
    assert any(c.startswith("[USER]") for c in written)
    assert not any(c.startswith("[ASSISTANT]") for c in written)
