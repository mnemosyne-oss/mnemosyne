from __future__ import annotations

import json

import pytest

from mnemosyne_hermes import (
    MnemosyneMemoryProvider,
    _canonical_prefetch_rows,
    _prefetch_canonical_generic_tokens,
    _prefetch_min_query_coverage,
)


class FakeBeam:
    author_id = "test-author"

    def __init__(self, results=None):
        self.results = results or []
        self.writes = []

    def recall(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return self.results

    def remember(self, **kwargs):
        self.writes.append(kwargs)


class FakeCanonicalStore:
    def __init__(self, rows_or_owners):
        if isinstance(rows_or_owners, dict):
            self.rows_by_owner = rows_or_owners
        else:
            self.rows_by_owner = {"default": rows_or_owners}
        self.requested_owner_ids = []

    def list(self, owner_id):
        self.requested_owner_ids.append(owner_id)
        return self.rows_by_owner.get(owner_id, [])


@pytest.fixture(autouse=True)
def clear_prefetch_configuration(monkeypatch):
    for key in (
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE",
        "MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY",
    ):
        monkeypatch.delenv(key, raising=False)


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


def test_prefetch_requires_two_distinctive_lexical_terms(monkeypatch):
    monkeypatch.setenv(
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "sampleowner,assistant",
    )
    p = _provider([
        {"content": "A lightweight workflow unrelated to tea.", "source": "workflow",
         "timestamp": "2026-06-11T09:00:00Z", "importance": 0.95, "score": 0.9,
         "keyword_score": 0.5, "trust_tier": "STATED"},
        {"content": "SampleOwner prefers Cedar Bakery over Harbor Bakery.", "source": "preference",
         "timestamp": "2026-06-11T09:01:00Z", "importance": 0.9, "score": 0.8,
         "keyword_score": 0.8, "trust_tier": "STATED"},
    ])

    tea = p.prefetch("tea preference jasmine oolong floral citrus clear infusion")
    bakery = p.prefetch("Cedar Bakery preference")

    assert tea == ""
    assert "Cedar Bakery" in bakery


def test_prefetch_generic_token_configuration_replaces_defaults(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", "sampleowner,assistant")

    tokens = _prefetch_canonical_generic_tokens()

    assert tokens == {"sampleowner", "assistant"}


def test_prefetch_extra_generic_token_configuration_is_additive(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", "sampleowner,assistant")
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS", "local,ownername")

    tokens = _prefetch_canonical_generic_tokens()

    assert tokens == {"sampleowner", "assistant", "local", "ownername"}


def test_prefetch_lexical_thresholds_are_configurable(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS", "1")
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE", "0.10")
    p = _provider([
        {"content": "A lightweight workflow unrelated to tea.", "source": "workflow",
         "timestamp": "2026-06-11T09:00:00Z", "importance": 0.95, "score": 0.9,
         "keyword_score": 0.5, "trust_tier": "STATED"},
    ])

    block = p.prefetch("tea floral jasmine clear infusion")

    assert "lightweight workflow" in block


def test_prefetch_nonfinite_query_coverage_uses_conservative_default(monkeypatch, caplog):
    for value in ("NaN", "inf", "-inf"):
        monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE", value)

        assert _prefetch_min_query_coverage() == 0.30

    assert caplog.text.count("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE") == 3


def test_sync_roles_can_disable_assistant_autosave():
    p = MnemosyneMemoryProvider()
    p._beam = FakeBeam()
    p._agent_context = "primary"
    p._sync_roles = {"user"}

    p.sync_turn("please remember user side", "assistant side should not be stored")

    written = [w["content"] for w in p._beam.writes]
    assert any(c.startswith("[USER]") for c in written)
    assert not any(c.startswith("[ASSISTANT]") for c in written)


def test_canonical_prefetch_rejects_common_single_token_matches(monkeypatch):
    monkeypatch.setenv(
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "user,owner,assistant,agent,system,profile,identity,default,sampleowner,prefers,preference",
    )
    store = FakeCanonicalStore([
        {"name": "tea", "body": "SampleOwner likes fragrant jasmine tea.", "category": "model:user"},
        {"name": "unrelated-a", "body": "SampleOwner studies orbital mechanics.", "category": "model:user"},
        {"name": "unrelated-b", "body": "SampleOwner collects antique maps.", "category": "model:user"},
    ])

    rows = _canonical_prefetch_rows(store, "default", "SampleOwner tea preference")

    assert [row["canonical_name"] for row in rows] == ["tea"]


def test_canonical_prefetch_allows_rare_single_token_and_two_token_matches(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", raising=False)
    store = FakeCanonicalStore([
        {"name": "archive", "body": "Archive boundaries apply to expired records.", "category": "model:user"},
        {"name": "archive-noise", "body": "External indexes require archiving.", "category": "model:workflow"},
        {"name": "geology", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
        {"name": "other", "body": "Copper alloys resist corrosion.", "category": "model:user"},
    ])

    archive = _canonical_prefetch_rows(store, "default", "archive boundaries")
    caldrin = _canonical_prefetch_rows(store, "default", "Caldrin research lens")

    assert [row["canonical_name"] for row in archive] == ["archive"]
    assert [row["canonical_name"] for row in caldrin] == ["geology"]


def test_canonical_prefetch_rarity_is_scoped_to_requested_owner(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY", "1")
    store = FakeCanonicalStore({
        "default": [
            {"name": "geology", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
        ],
        "other-owner": [
            {"name": "geology-a", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
            {"name": "geology-b", "body": "Caldrin studies coastal erosion.", "category": "model:user"},
        ],
    })

    default_rows = _canonical_prefetch_rows(store, "default", "Caldrin research lens")
    other_rows = _canonical_prefetch_rows(store, "other-owner", "Caldrin research lens")

    assert store.requested_owner_ids == ["default", "other-owner"]
    assert [row["canonical_name"] for row in default_rows] == ["geology"]
    assert other_rows == []


def test_explicit_recall_preserves_broad_canonical_merge():
    p = _provider([])
    p._beam.canonical = FakeCanonicalStore([
        {"name": "geology-a", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
        {"name": "geology-b", "body": "Caldrin studies coastal erosion.", "category": "model:user"},
    ])

    response = json.loads(p.handle_tool_call(
        "mnemosyne_recall",
        {"query": "Caldrin research lens", "limit": 5},
    ))

    assert p._beam.last_args == ("Caldrin research lens",)
    assert {row["canonical_name"] for row in response["results"]} == {"geology-a", "geology-b"}


def test_canonical_prefetch_can_disable_single_token_exception(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY", "0")
    store = FakeCanonicalStore([
        {"name": "geology", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
    ])

    rows = _canonical_prefetch_rows(store, "default", "Caldrin research lens")

    assert rows == []


def test_canonical_prefetch_keeps_singleton_coverage_guard_when_minimum_is_one(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS", "1")
    store = FakeCanonicalStore([
        {"name": "geology", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
    ])

    rows = _canonical_prefetch_rows(
        store,
        "default",
        "Caldrin research archive survey analysis history",
    )

    assert rows == []


def test_canonical_prefetch_respects_higher_distinctive_token_minimum(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS", "3")
    store = FakeCanonicalStore([
        {"name": "tea", "body": "Fragrant jasmine tea is preferred.", "category": "model:user"},
    ])

    rows = _canonical_prefetch_rows(store, "default", "fragrant jasmine soup")

    assert rows == []
