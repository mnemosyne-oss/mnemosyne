"""Regression coverage for enhanced-recall cache request isolation (#513)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.query_cache import QueryCache


@pytest.fixture
def enhanced(monkeypatch, tmp_path: Path):
    """A deterministic enhanced-recall instance whose base pipeline is observable."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    memory = BeamMemory(session_id="session-a", db_path=tmp_path / "memories.db")
    calls = []

    def fake_recall(query, top_k=40, **kwargs):
        calls.append((query, top_k, kwargs.copy()))
        if kwargs.get("explain"):
            return {
                "query": query,
                "top_k": top_k,
                "engine": "linear",
                "results": [{"id": "explained", "content": query, "score": 1.0}],
                "explain": {"trace": "kept"},
            }
        return [{"id": f"result-{len(calls)}", "content": query, "score": 1.0}]

    memory.recall = fake_recall
    yield memory, calls
    memory.conn.close()
    if getattr(memory, "_query_cache", None) is not None:
        memory._query_cache.close()


def _call(memory: BeamMemory, query: str = "private query", **kwargs):
    return memory.recall_enhanced(
        query,
        use_weibull=False,
        use_mmr=False,
        use_intent=False,
        use_synonyms=False,
        **kwargs,
    )


def test_v2_request_digest_is_opaque_persisted_and_exact_hits_once(enhanced):
    memory, calls = enhanced

    first = _call(memory, "private query", top_k=3)
    second = _call(memory, "private query", top_k=3)

    assert first == second
    assert len(calls) == 1
    assert memory._query_cache is not None
    rows = memory._query_cache._conn.execute("SELECT normalized FROM query_cache").fetchall()
    assert len(rows) == 1
    key = rows[0][0]
    assert key.startswith("v2:")
    assert len(key) == len("v2:") + 64
    assert all(character in "0123456789abcdef" for character in key[3:])
    assert "private query" not in key
    assert "session-a" not in key
    assert str(memory.db_path) not in key


def test_v2_prefixed_natural_language_uses_normalization_after_reload(tmp_path: Path):
    db_path = tmp_path / "query-cache.db"
    cache = QueryCache(db_path=db_path)
    natural_query = "v2: hello world"
    equivalent_query = "WORLD v2: HELLO"
    legacy_query = "v2: zebra"
    legacy_equivalent = "zebra v2:"
    natural_results = [{"id": "natural", "content": natural_query, "score": 1.0}]
    legacy_results = [{"id": "legacy", "content": legacy_query, "score": 1.0}]

    cache.put(natural_query, natural_results, embedding=[1.0])
    cache.put(legacy_query, legacy_results, embedding=[1.0])

    assert cache.get(equivalent_query, embedding=[1.0]) == natural_results
    assert cache.get(legacy_equivalent, embedding=[1.0]) == legacy_results
    assert cache._opaque == {}
    assert cache._normalize(natural_query) in cache._tier1
    assert cache._normalize(legacy_query) in cache._tier1
    cache.close()

    reloaded = QueryCache(db_path=db_path)
    try:
        assert reloaded.get(equivalent_query, embedding=[1.0]) == natural_results
        assert reloaded.get(legacy_equivalent, embedding=[1.0]) == legacy_results
        assert reloaded._opaque == {}
        assert reloaded._normalize(natural_query) in reloaded._tier1
        assert reloaded._normalize(legacy_query) in reloaded._tier1
    finally:
        reloaded.close()


def test_effective_request_variants_do_not_cross_hit(enhanced):
    memory, calls = enhanced
    base = {
        "top_k": 3,
        "source": "email",
        "channel_id": "channel-a",
        "author_id": "author-a",
        "author_type": "agent",
        "veracity": "stated",
        "memory_type": "fact",
        "topic": "security",
        "from_date": "2026-01-01",
        "to_date": "2026-01-31",
        "temporal_weight": 0.3,
        "query_time": datetime(2026, 1, 15, tzinfo=timezone.utc),
        "temporal_halflife": 12.0,
        "vec_weight": 0.2,
        "fts_weight": 0.7,
        "importance_weight": 0.1,
        "use_associative": True,
        "associative_depth": 2,
        "mmr_lambda": 0.4,
    }
    _call(memory, **base)
    assert len(calls) == 1

    variants = [
        {"top_k": 4},
        {"source": "slack"},
        {"channel_id": "channel-b"},
        {"author_id": "author-b"},
        {"author_type": "human"},
        {"veracity": "inferred"},
        {"memory_type": "preference"},
        {"topic": "operations"},
        {"from_date": "2026-01-02"},
        {"to_date": "2026-02-01"},
        {"temporal_weight": 0.4},
        {"query_time": datetime(2026, 1, 16, tzinfo=timezone.utc)},
        {"temporal_halflife": 24.0},
        {"vec_weight": 0.3},
        {"use_associative": False},
        {"associative_depth": 3},
        {"mmr_lambda": 0.8},
    ]
    for change in variants:
        _call(memory, **(base | change))

    assert len(calls) == 1 + len(variants)


def test_pipeline_flags_and_process_ranking_configuration_change_digest(enhanced, monkeypatch):
    memory, _ = enhanced
    runtime = SimpleNamespace(cross_session=False)
    common = dict(
        original_query="private query",
        expanded_query="private query",
        top_k=3,
        runtime=runtime,
        use_weibull=False,
        use_mmr=False,
        use_intent=False,
        use_synonyms=False,
        use_associative=False,
        associative_depth=1,
        mmr_lambda=0.7,
        recall_kwargs={},
    )
    baseline = memory._enhanced_recall_cache_key(**common)

    assert memory._enhanced_recall_cache_key(**(common | {"use_mmr": True})) != baseline
    assert memory._enhanced_recall_cache_key(**(common | {"use_weibull": True})) != baseline
    assert memory._enhanced_recall_cache_key(**(common | {"use_intent": True})) != baseline
    assert memory._enhanced_recall_cache_key(**(common | {"use_synonyms": True})) != baseline
    assert memory._enhanced_recall_cache_key(**(common | {"use_associative": True})) != baseline

    monkeypatch.setattr(beam_module, "TIER2_DAYS", beam_module.TIER2_DAYS + 1)
    assert memory._enhanced_recall_cache_key(**common) != baseline

    monkeypatch.undo()
    monkeypatch.setenv("MNEMOSYNE_CROSS_TIER_DEDUP", "0")
    assert memory._enhanced_recall_cache_key(**common) != baseline

    monkeypatch.undo()
    monkeypatch.setattr(beam_module, "weibull_boost", None)
    assert memory._enhanced_recall_cache_key(**common) != baseline


def test_sessions_cross_session_and_sibling_databases_are_isolated(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    runtime = SimpleNamespace(cross_session=False)
    monkeypatch.setattr(beam_module, "resolve_beam_runtime", lambda: runtime)
    db_a = tmp_path / "a.db"
    session_a = BeamMemory(session_id="session-a", db_path=db_a)
    session_b = BeamMemory(session_id="session-b", db_path=db_a)
    sibling = BeamMemory(session_id="session-a", db_path=tmp_path / "b.db")
    calls = {"a": 0, "b": 0, "sibling": 0}

    def fake(label):
        def recall(query, top_k=40, **kwargs):
            calls[label] += 1
            return [{"id": label, "content": label, "score": 1.0}]
        return recall

    session_a.recall = fake("a")
    session_b.recall = fake("b")
    sibling.recall = fake("sibling")
    try:
        _call(session_a)
        _call(session_a)
        _call(session_b)
        _call(sibling)
        assert calls == {"a": 1, "b": 1, "sibling": 1}

        runtime.cross_session = True
        _call(session_a)
        assert calls["a"] == 2
    finally:
        for memory in (session_a, session_b, sibling):
            memory.conn.close()
            if getattr(memory, "_query_cache", None) is not None:
                memory._query_cache.close()


def test_bypass_and_explain_never_read_or_write_the_cache(enhanced):
    memory, calls = enhanced

    cached = _call(memory, "bypass query")
    bypassed = _call(memory, "bypass query", use_cache=False)
    assert len(calls) == 2
    assert bypassed[0]["id"] != cached[0]["id"]
    assert _call(memory, "bypass query") == cached
    assert len(calls) == 2

    normal = _call(memory, "explain query")
    explained = _call(memory, "explain query", explain=True)
    assert explained["engine"] == "linear"
    assert explained["explain"] == {"trace": "kept"}
    assert len(calls) == 4
    assert _call(memory, "explain query") == normal
    assert len(calls) == 4


def test_legacy_entries_and_every_opaque_access_path_are_exact_only(enhanced):
    memory, calls = enhanced
    cache = QueryCache(max_size=20)
    legacy_key = "session-a\x1f0\x1fprivate query"
    cache.put(legacy_key, [{"id": "legacy", "content": "legacy", "score": 1.0}], embedding=[1.0])
    memory._query_cache = cache

    assert _call(memory)[0]["id"] != "legacy"
    assert len(calls) == 1

    opaque = "v2:" + hashlib.sha256(b"request-a").hexdigest()
    different = "v2:" + hashlib.sha256(b"request-b").hexdigest()
    cache.put(opaque, [{"id": "opaque", "content": "opaque", "score": 1.0}])
    cache.put("semantic source", [{"id": "semantic", "content": "semantic", "score": 1.0}], embedding=[1.0])

    assert cache.get(opaque, embedding=[1.0])[0]["id"] == "opaque"
    assert cache.get_opaque(opaque)[0]["id"] == "opaque"
    assert cache.get(different, embedding=[1.0]) is None
    assert cache.get_opaque(different) is None
    assert cache.tier2_hits == 0
    assert cache.tier3_hits == 0
    assert cache.tier4_hits == 0
