"""Regression coverage for enhanced-recall cache request isolation (#513)."""

from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.config import MnemosyneConfig, get_config
from mnemosyne.core.query_cache import QueryCache


@pytest.fixture(autouse=True)
def reset_config_singleton():
    """Do not leak a temporary config directory into later test modules."""
    MnemosyneConfig.reset_instance()
    yield
    MnemosyneConfig.reset_instance()


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


def _close_memory(memory: BeamMemory | None) -> None:
    if memory is None:
        return
    try:
        cache = getattr(memory, "_query_cache", None)
        if cache is not None:
            cache.close()
    finally:
        memory.conn.close()


def test_successful_invalidate_clears_persisted_enhanced_recall_cache(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    memory = None
    fresh = None
    try:
        memory = BeamMemory(session_id="session-a", db_path=db_path)
        memory_id = memory.remember(
            "issue 550 persistent cache invalidation sentinel",
            source="test",
            importance=1.0,
        )
        warm = _call(memory, "issue 550 persistent cache invalidation sentinel", top_k=3)
        assert memory_id in {result["id"] for result in warm}
        assert _call(memory, "issue 550 persistent cache invalidation sentinel", top_k=3) == warm
        assert memory._query_cache is not None

        cache_version = memory._query_cache.stats()["version"]
        assert memory.invalidate("missing-memory") is False
        assert memory._query_cache.stats()["version"] == cache_version

        assert memory.invalidate(memory_id) is True
        assert memory._query_cache.stats()["version"] == cache_version + 1
        current = _call(memory, "issue 550 persistent cache invalidation sentinel", top_k=3)
        assert memory_id not in {result["id"] for result in current}

        fresh = BeamMemory(session_id="session-a", db_path=db_path)
        reloaded = _call(fresh, "issue 550 persistent cache invalidation sentinel", top_k=3)
        assert memory_id not in {result["id"] for result in reloaded}
    finally:
        try:
            _close_memory(fresh)
        finally:
            _close_memory(memory)


def test_successful_forget_working_clears_persisted_enhanced_recall_cache(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    memory = forgetter = fresh = None
    try:
        memory = BeamMemory(session_id="session-a", db_path=db_path)
        memory_id = memory.remember("issue 553 forget cache sentinel", source="test")
        warm = _call(memory, "issue 553 forget cache sentinel", top_k=3)
        assert memory_id in {result["id"] for result in warm}
        assert memory._query_cache is not None
        warm_cache_stats = memory._query_cache.stats()
        assert _call(memory, "issue 553 forget cache sentinel", top_k=3) == warm
        assert memory._query_cache.stats()["hits"] == warm_cache_stats["hits"] + 1
        cache_version = memory._query_cache.stats()["version"]

        assert memory.forget_working("missing-memory") is False
        assert memory._query_cache.stats()["version"] == cache_version

        local_id = memory.remember("issue 553 local-cache forget sentinel", source="test")
        local_warm = _call(memory, "issue 553 local-cache forget sentinel", top_k=3)
        assert local_id in {result["id"] for result in local_warm}
        local_version = memory._query_cache.stats()["version"]
        assert memory.forget_working(local_id) is True
        assert memory._query_cache.stats()["version"] == local_version + 1
        assert local_id not in {
            result["id"]
            for result in _call(memory, "issue 553 local-cache forget sentinel", top_k=3)
        }

        forgetter = BeamMemory(session_id="session-a", db_path=db_path)
        assert not hasattr(forgetter, "_query_cache")
        assert forgetter.forget_working(memory_id) is True

        fresh = BeamMemory(session_id="session-a", db_path=db_path)
        assert memory_id not in {
            result["id"] for result in _call(fresh, "issue 553 forget cache sentinel", top_k=3)
        }
    finally:
        try:
            _close_memory(fresh)
        finally:
            try:
                _close_memory(forgetter)
            finally:
                _close_memory(memory)


def test_failed_cross_session_forget_keeps_cache_and_memory(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    owner = other = None
    try:
        db_path = tmp_path / "memories.db"
        owner = BeamMemory(session_id="session-a", db_path=db_path)
        memory_id = owner.remember("issue 553 private forget sentinel", source="test")
        assert memory_id in {
            row["id"] for row in _call(owner, "issue 553 private forget sentinel", top_k=3)
        }

        other = BeamMemory(session_id="session-b", db_path=db_path)
        other_warm = _call(other, "issue 553 private forget sentinel", top_k=3)
        assert memory_id not in {result["id"] for result in other_warm}
        assert other._query_cache is not None
        cache_version = other._query_cache.stats()["version"]
        assert other.forget_working(memory_id) is False
        assert other._query_cache.stats()["version"] == cache_version
        assert memory_id not in {
            result["id"]
            for result in _call(other, "issue 553 private forget sentinel", top_k=3)
        }
        assert owner.get(memory_id) is not None
        assert memory_id in {
            row["id"] for row in _call(owner, "issue 553 private forget sentinel", top_k=3)
        }
    finally:
        _close_memory(other)
        _close_memory(owner)


def test_invalidate_without_local_query_cache_clears_persisted_enhanced_recall_cache(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    source = None
    invalidator = None
    fresh = None
    try:
        source = BeamMemory(session_id="session-a", db_path=db_path)
        memory_id = source.remember(
            "issue 554 cross-instance persistent cache invalidation sentinel",
            source="test",
            importance=1.0,
        )
        warm = _call(source, "issue 554 cross-instance persistent cache invalidation sentinel", top_k=3)
        assert memory_id in {result["id"] for result in warm}
        assert source._query_cache is not None

        cache_db = db_path.parent / "query_cache.db"
        assert cache_db.is_file()
        invalidator = BeamMemory(session_id="session-a", db_path=db_path)
        assert not hasattr(invalidator, "_query_cache")
        assert invalidator.invalidate(memory_id) is True

        fresh = BeamMemory(session_id="session-a", db_path=db_path)
        reloaded = _call(fresh, "issue 554 cross-instance persistent cache invalidation sentinel", top_k=3)
        assert memory_id not in {result["id"] for result in reloaded}
    finally:
        try:
            _close_memory(fresh)
        finally:
            try:
                _close_memory(invalidator)
            finally:
                _close_memory(source)


def test_remember_without_local_query_cache_clears_persisted_enhanced_recall_cache(
    monkeypatch, tmp_path: Path
):
    """A fresh writer must evict A's persisted result before fresh C recalls."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    source = None
    writer = None
    fresh = None
    try:
        source = BeamMemory(session_id="session-a", db_path=db_path)
        source_id = source.remember(
            "issue 556 persisted remember cache sentinel baseline",
            source="test",
            importance=1.0,
        )
        warm = _call(source, "issue 556 persisted remember cache sentinel", top_k=3)
        assert source_id in {result["id"] for result in warm}
        assert source._query_cache is not None
        assert (db_path.parent / "query_cache.db").is_file()

        writer = BeamMemory(session_id="session-a", db_path=db_path)
        assert not hasattr(writer, "_query_cache")
        inserted_id = writer.remember(
            "issue 556 persisted remember cache sentinel inserted",
            source="test",
            importance=1.0,
        )

        fresh = BeamMemory(session_id="session-a", db_path=db_path)
        reloaded = _call(fresh, "issue 556 persisted remember cache sentinel", top_k=3)
        assert inserted_id in {result["id"] for result in reloaded}
    finally:
        try:
            _close_memory(fresh)
        finally:
            try:
                _close_memory(writer)
            finally:
                _close_memory(source)


def test_dedup_remember_without_local_query_cache_evicts_persisted_result(
    monkeypatch, tmp_path: Path
):
    """A fresh dedup writer must refresh the persisted result that exposes its update."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    source = None
    writer = None
    fresh = None
    try:
        source = BeamMemory(session_id="session-a", db_path=db_path)
        memory_id = source.remember(
            "issue 556 dedup persisted cache sentinel",
            source="initial",
            importance=0.1,
        )
        warm = _call(source, "issue 556 dedup persisted cache sentinel", top_k=3)
        assert next(result for result in warm if result["id"] == memory_id)["source"] == "initial"
        assert source._query_cache is not None
        assert (db_path.parent / "query_cache.db").is_file()

        writer = BeamMemory(session_id="session-a", db_path=db_path)
        assert not hasattr(writer, "_query_cache")
        assert writer.remember(
            "issue 556 dedup persisted cache sentinel",
            source="updated",
            importance=1.0,
        ) == memory_id

        fresh = BeamMemory(session_id="session-a", db_path=db_path)
        reloaded = _call(fresh, "issue 556 dedup persisted cache sentinel", top_k=3)
        updated = next(result for result in reloaded if result["id"] == memory_id)
        assert updated["source"] == "updated"
        assert updated["importance"] == 1.0
    finally:
        try:
            _close_memory(fresh)
        finally:
            try:
                _close_memory(writer)
            finally:
                _close_memory(source)


def test_remember_invalidates_persisted_cache_when_post_commit_trim_raises(
    monkeypatch, tmp_path: Path
):
    """A trim failure must propagate only after the committed write evicts cache."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    source = None
    writer = None
    fresh = None
    query = "issue 556 trim failure persisted cache sentinel"
    try:
        source = BeamMemory(session_id="session-a", db_path=db_path)
        source_id = source.remember(f"{query} baseline", source="test", importance=1.0)
        warm = _call(source, query, top_k=3)
        assert source_id in {result["id"] for result in warm}
        assert source._query_cache is not None
        assert (db_path.parent / "query_cache.db").is_file()

        writer = BeamMemory(session_id="session-a", db_path=db_path)
        assert not hasattr(writer, "_query_cache")

        def fail_trim():
            raise RuntimeError("trim failed after insert commit")

        monkeypatch.setattr(writer, "_trim_working_memory", fail_trim)
        with pytest.raises(RuntimeError, match="trim failed after insert commit"):
            writer.remember(f"{query} inserted", source="test", importance=1.0)

        fresh = BeamMemory(session_id="session-a", db_path=db_path)
        reloaded = _call(fresh, query, top_k=3)
        assert {result["content"] for result in reloaded} != {
            result["content"] for result in warm
        }
        assert f"{query} inserted" in {result["content"] for result in reloaded}
    finally:
        try:
            _close_memory(fresh)
        finally:
            try:
                _close_memory(writer)
            finally:
                _close_memory(source)


def test_dedup_invalidates_before_enrichment(monkeypatch, tmp_path: Path):
    """Dedup writes evict cache before their post-commit enrichment begins."""
    memory = BeamMemory(session_id="session-a", db_path=tmp_path / "memories.db")
    calls = []
    try:
        monkeypatch.setattr(
            memory,
            "_invalidate_query_cache_after_remember_commit",
            lambda: calls.append("invalidate"),
        )
        monkeypatch.setattr(
            memory,
            "_ingest_graph_and_veracity",
            lambda *args: calls.append("enrich"),
        )
        memory.remember("issue 556 dedup invalidation timing", source="test")
        calls.clear()

        memory.remember("issue 556 dedup invalidation timing", source="updated")

        assert calls.index("invalidate") < calls.index("enrich")
    finally:
        _close_memory(memory)


def test_remember_finally_invalidates_cache_refilled_during_enrichment(monkeypatch, tmp_path: Path):
    """Post-commit enrichment cannot leave a newly warmed result behind."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    memory = None
    try:
        memory = BeamMemory(session_id="session-a", db_path=db_path)
        query = "issue 566 post commit enrichment cache refill"
        baseline = f"{query} baseline"
        memory.remember(baseline, source="initial", importance=0.1)
        _call(memory, query, top_k=3)
        assert memory._query_cache._conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 1

        refilled_rows = []

        def refill_during_enrichment(*args):
            _call(memory, query, top_k=3)
            refilled_rows.append(
                memory._query_cache._conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            )

        monkeypatch.setattr(memory, "_ingest_graph_and_veracity", refill_during_enrichment)

        assert memory.remember(baseline, source="updated", importance=1.0)
        assert refilled_rows == [1]
        assert memory._query_cache._conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 0
    finally:
        _close_memory(memory)


def test_new_remember_finally_invalidates_cache_refilled_during_enrichment(monkeypatch, tmp_path: Path):
    """New inserts receive the same final eviction after enrichment refills cache."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    memory = None
    try:
        memory = BeamMemory(session_id="session-a", db_path=db_path)
        query = "issue 566 new insert enrichment cache refill"
        memory.remember(f"{query} baseline", source="initial", importance=0.1)
        _call(memory, query, top_k=3)
        assert memory._query_cache._conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 1

        refilled_rows = []

        def refill_during_enrichment(*args):
            _call(memory, query, top_k=3)
            refilled_rows.append(
                memory._query_cache._conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0]
            )

        monkeypatch.setattr(memory, "_ingest_graph_and_veracity", refill_during_enrichment)

        memory.remember(f"{query} inserted", source="new", importance=1.0)
        assert refilled_rows == [1]
        assert memory._query_cache._conn.execute("SELECT COUNT(*) FROM query_cache").fetchone()[0] == 0
    finally:
        _close_memory(memory)


def test_new_remember_finally_invalidates_cache_refilled_during_enrichment_exception(
    monkeypatch, tmp_path: Path
):
    """A post-commit enrichment error must evict its real cache refill for fresh C."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    writer = None
    fresh = None
    try:
        writer = BeamMemory(session_id="session-a", db_path=db_path)
        query = "issue 566 new insert enrichment exception cache refill"
        inserted_id = "issue-566-new-enrichment-exception"
        writer.remember(f"{query} baseline", source="initial", importance=0.1)
        _call(writer, query, top_k=3)
        assert writer._query_cache._conn.execute(
            "SELECT COUNT(*) FROM query_cache"
        ).fetchone()[0] == 1

        refilled_rows = []

        def refill_then_raise(memory_id, *args):
            refilled = _call(writer, query, top_k=3)
            assert memory_id in {result["id"] for result in refilled}
            refilled_rows.append(
                writer._query_cache._conn.execute(
                    "SELECT COUNT(*) FROM query_cache"
                ).fetchone()[0]
            )
            raise RuntimeError("new enrichment failed after cache refill")

        monkeypatch.setattr(writer, "_ingest_graph_and_veracity", refill_then_raise)
        with pytest.raises(RuntimeError, match="new enrichment failed after cache refill"):
            writer.remember(
                f"{query} inserted",
                source="new",
                importance=1.0,
                memory_id=inserted_id,
            )

        assert refilled_rows == [1]
        assert writer.conn.execute(
            "SELECT content FROM working_memory WHERE id = ?", (inserted_id,)
        ).fetchone()["content"] == f"{query} inserted"

        fresh = BeamMemory(session_id="session-a", db_path=db_path)
        assert not hasattr(fresh, "_query_cache")
        fresh._query_cache = QueryCache(db_path=db_path.parent / "query_cache.db")
        assert fresh._query_cache._conn.execute(
            "SELECT COUNT(*) FROM query_cache"
        ).fetchone()[0] == 0
        reloaded = _call(fresh, query, top_k=3)
        assert inserted_id in {result["id"] for result in reloaded}
    finally:
        try:
            _close_memory(fresh)
        finally:
            _close_memory(writer)


def test_dedup_remember_finally_invalidates_cache_refilled_during_enrichment_exception(
    monkeypatch, tmp_path: Path
):
    """A dedup enrichment error must evict its real cache refill for fresh C."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    db_path = tmp_path / "memories.db"
    writer = None
    fresh = None
    try:
        writer = BeamMemory(session_id="session-a", db_path=db_path)
        query = "issue 566 dedup enrichment exception cache refill"
        memory_id = writer.remember(query, source="initial", importance=0.1)
        _call(writer, query, top_k=3)
        assert writer._query_cache._conn.execute(
            "SELECT COUNT(*) FROM query_cache"
        ).fetchone()[0] == 1

        refilled_rows = []

        def refill_then_raise(enriched_id, *args):
            refilled = _call(writer, query, top_k=3)
            assert enriched_id in {result["id"] for result in refilled}
            refilled_rows.append(
                writer._query_cache._conn.execute(
                    "SELECT COUNT(*) FROM query_cache"
                ).fetchone()[0]
            )
            raise RuntimeError("dedup enrichment failed after cache refill")

        monkeypatch.setattr(writer, "_ingest_graph_and_veracity", refill_then_raise)
        with pytest.raises(RuntimeError, match="dedup enrichment failed after cache refill"):
            writer.remember(query, source="updated", importance=1.0)

        assert refilled_rows == [1]
        assert dict(
            writer.conn.execute(
                "SELECT source, importance FROM working_memory WHERE id = ?", (memory_id,)
            ).fetchone()
        ) == {"source": "updated", "importance": 1.0}

        fresh = BeamMemory(session_id="session-a", db_path=db_path)
        assert not hasattr(fresh, "_query_cache")
        fresh._query_cache = QueryCache(db_path=db_path.parent / "query_cache.db")
        assert fresh._query_cache._conn.execute(
            "SELECT COUNT(*) FROM query_cache"
        ).fetchone()[0] == 0
        reloaded = _call(fresh, query, top_k=3)
        updated = next(result for result in reloaded if result["id"] == memory_id)
        assert updated["source"] == "updated"
        assert updated["importance"] == 1.0
    finally:
        try:
            _close_memory(fresh)
        finally:
            _close_memory(writer)


def test_new_remember_write_survives_post_commit_cache_invalidation_failure(
    monkeypatch, tmp_path: Path, caplog
):
    memory = BeamMemory(session_id="session-a", db_path=tmp_path / "memories.db")

    def fail_invalidation():
        raise RuntimeError("cache unavailable")

    try:
        monkeypatch.setattr(memory, "_invalidate_query_cache", fail_invalidation)
        with caplog.at_level(logging.WARNING, logger=beam_module.__name__):
            memory_id = memory.remember(
                "issue 556 new write survives cache invalidation failure",
                source="test",
                importance=1.0,
            )

        row = memory.conn.execute(
            "SELECT content, source, importance FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()
        assert row is not None
        assert dict(row) == {
            "content": "issue 556 new write survives cache invalidation failure",
            "source": "test",
            "importance": 1.0,
        }
        assert "query-cache invalidation failed after commit" in caplog.text
    finally:
        _close_memory(memory)


def test_dedup_remember_write_survives_post_commit_cache_invalidation_failure(
    monkeypatch, tmp_path: Path, caplog
):
    memory = BeamMemory(session_id="session-a", db_path=tmp_path / "memories.db")

    def fail_invalidation():
        raise RuntimeError("cache unavailable")

    try:
        memory_id = memory.remember(
            "issue 556 dedup write survives cache invalidation failure",
            source="initial",
            importance=0.1,
        )
        monkeypatch.setattr(memory, "_invalidate_query_cache", fail_invalidation)
        with caplog.at_level(logging.WARNING, logger=beam_module.__name__):
            assert memory.remember(
                "issue 556 dedup write survives cache invalidation failure",
                source="updated",
                importance=1.0,
            ) == memory_id

        row = memory.conn.execute(
            "SELECT source, importance FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()
        assert row is not None
        assert dict(row) == {"source": "updated", "importance": 1.0}
        assert "query-cache invalidation failed after commit" in caplog.text
    finally:
        _close_memory(memory)


def test_successful_invalidate_episodic_memory_invalidates_enhanced_recall_cache(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    memory = None
    fresh = None
    try:
        memory = BeamMemory(session_id="session-a", db_path=tmp_path / "memories.db")
        episodic_id = memory.consolidate_to_episodic(
            "issue 550 episodic cache invalidation sentinel",
            source_wm_ids=[],
            source="test",
            importance=1.0,
        )

        warm = _call(memory, "issue 550 episodic cache invalidation sentinel", top_k=3)
        episodic_result = next(result for result in warm if result["id"] == episodic_id)
        assert episodic_result["tier"] == "episodic"
        assert memory._query_cache is not None

        cache_version = memory._query_cache.stats()["version"]
        assert memory.invalidate(episodic_id) is True
        assert memory._query_cache.stats()["version"] == cache_version + 1
        fresh = BeamMemory(session_id="session-a", db_path=memory.db_path)
        current = _call(fresh, "issue 550 episodic cache invalidation sentinel", top_k=3)
        assert fresh._query_cache is not None
        assert episodic_id not in {result["id"] for result in current}
    finally:
        try:
            _close_memory(fresh)
        finally:
            _close_memory(memory)


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


def test_private_enhanced_cache_key_weight_fallback_honors_yaml_over_env(
    enhanced, monkeypatch, tmp_path: Path
):
    """Private cache-key fallback uses the same YAML-resolved weights as scoring."""
    memory, _ = enhanced
    data_dir = tmp_path / "config"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(
        "vec_weight: 0\nfts_weight: 1\nimportance_weight: 0\n"
    )
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "1")
    monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0")
    monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "1")
    MnemosyneConfig.reset_instance()
    common = dict(
        original_query="private query",
        expanded_query="private query",
        top_k=3,
        runtime=SimpleNamespace(cross_session=False),
        use_weibull=False,
        use_mmr=False,
        use_intent=False,
        use_synonyms=False,
        use_associative=False,
        associative_depth=1,
        mmr_lambda=0.7,
        recall_kwargs={},
    )
    try:
        expected_weights = beam_module._resolve_recall_weights(None, None, None).as_tuple()
        assert expected_weights == (0.0, 1.0, 0.0)

        fallback_key = memory._enhanced_recall_cache_key(**common)
        expected_key = memory._enhanced_recall_cache_key(**common, weights=expected_weights)

        assert fallback_key == expected_key
    finally:
        MnemosyneConfig.reset_instance()


def test_private_enhanced_cache_key_normalizes_intent_adjustment_fallback(
    enhanced, monkeypatch, tmp_path: Path
):
    """The direct key helper retains scorer-equivalent safe weights."""
    memory, _ = enhanced
    data_dir = tmp_path / "config"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text("cross_session: false\n")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_QUERY_INTENT", "1")
    monkeypatch.setattr(beam_module, "classify_intent", lambda query: object())
    monkeypatch.setattr(beam_module, "adjust_weights", lambda **kwargs: (0.0, 0.0, 0.0))
    MnemosyneConfig.reset_instance()
    common = dict(
        original_query="private query",
        expanded_query="private query",
        top_k=3,
        runtime=SimpleNamespace(cross_session=False),
        use_weibull=False,
        use_mmr=False,
        use_intent=True,
        use_synonyms=False,
        use_associative=False,
        associative_depth=1,
        mmr_lambda=0.7,
        recall_kwargs={},
    )
    try:
        fallback_key = memory._enhanced_recall_cache_key(**common)
        expected_key = memory._enhanced_recall_cache_key(
            **common, weights=(0.5, 0.3, 0.2)
        )
        assert fallback_key == expected_key
    finally:
        MnemosyneConfig.reset_instance()


def test_enhanced_recall_skips_env_intent_adjustment_when_disabled(enhanced, monkeypatch):
    """An explicit use_intent=False wins over the process-level intent flag."""
    memory, calls = enhanced
    monkeypatch.setenv("MNEMOSYNE_QUERY_INTENT", "1")
    monkeypatch.setattr(beam_module, "classify_intent", lambda query: object())
    adjust_calls = []

    def record_adjustment(*args, **kwargs):
        adjust_calls.append((args, kwargs))
        return (0.0, 1.0, 0.0)

    monkeypatch.setattr(beam_module, "adjust_weights", record_adjustment)

    results = memory.recall_enhanced(
        "explicitly disabled query intent",
        use_weibull=False,
        use_mmr=False,
        use_intent=False,
        use_synonyms=False,
    )

    assert results
    assert adjust_calls == []
    assert calls[0][2]["_resolved_weights"].as_tuple() == (
        beam_module._resolve_recall_weights(None, None, None).as_tuple()
    )


def test_enhanced_recall_reloaded_weight_snapshot_misses_cache_and_changes_key(
    enhanced, monkeypatch, tmp_path: Path
):
    """A cache key is bound to the one effective config-resolved weight snapshot."""
    memory, calls = enhanced
    data_dir = tmp_path / "config"
    data_dir.mkdir()
    config_path = data_dir / "config.yaml"
    config_path.write_text("vec_weight: 0\nfts_weight: 1\nimportance_weight: 0\n")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    monkeypatch.setenv("MNEMOSYNE_VEC_WEIGHT", "1")
    monkeypatch.setenv("MNEMOSYNE_FTS_WEIGHT", "0")
    monkeypatch.setenv("MNEMOSYNE_IMPORTANCE_WEIGHT", "1")
    MnemosyneConfig.reset_instance()

    original_key = memory._enhanced_recall_cache_key
    original_resolver = beam_module._resolve_recall_weights
    observed = []
    resolutions = []

    def capture_resolver(*args, **kwargs):
        snapshot = original_resolver(*args, **kwargs)
        resolutions.append(snapshot)
        return snapshot

    def capture_key(*args, **kwargs):
        observed.append(kwargs["weights"])
        return original_key(*args, **kwargs)

    monkeypatch.setattr(beam_module, "_resolve_recall_weights", capture_resolver)
    monkeypatch.setattr(memory, "_enhanced_recall_cache_key", capture_key)
    _call(memory, "reloadable cache weights")
    _call(memory, "reloadable cache weights")
    assert len(calls) == 1
    assert len(resolutions) == 2
    assert observed == [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)]

    config_path.write_text("vec_weight: 1\nfts_weight: 0\nimportance_weight: 0\n")
    get_config().reload()
    _call(memory, "reloadable cache weights")
    assert len(calls) == 2
    assert len(resolutions) == 3
    assert observed[-1] == (1.0, 0.0, 0.0)


def test_enhanced_recall_reload_boundary_uses_one_complete_weight_generation(
    enhanced, monkeypatch, tmp_path: Path
):
    """A reload at the snapshot boundary reaches cache and recall as one generation."""
    memory, calls = enhanced
    data_dir = tmp_path / "config"
    data_dir.mkdir()
    config_path = data_dir / "config.yaml"
    new_generation = (0.0, 1.0, 0.0)
    config_path.write_text("vec_weight: 1\nfts_weight: 0\nimportance_weight: 0\n")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    MnemosyneConfig.reset_instance()
    config = get_config()
    original_maybe_reload = config._maybe_reload
    original_resolver = beam_module._resolve_recall_weights
    reads = 0
    weight_resolution_reads = []
    in_weight_resolution = False
    observed_key_weights = []
    original_key = memory._enhanced_recall_cache_key

    def reload_at_snapshot_boundary():
        nonlocal reads
        reads += 1
        original_maybe_reload()
        if in_weight_resolution:
            config_path.write_text("vec_weight: 0\nfts_weight: 1\nimportance_weight: 0\n")
            config.reload()

    def capture_resolver(*args, **kwargs):
        nonlocal in_weight_resolution
        start_reads = reads
        in_weight_resolution = True
        try:
            return original_resolver(*args, **kwargs)
        finally:
            in_weight_resolution = False
            weight_resolution_reads.append(reads - start_reads)

    def capture_key(*args, **kwargs):
        observed_key_weights.append(kwargs["weights"])
        return original_key(*args, **kwargs)

    monkeypatch.setattr(config, "_maybe_reload", reload_at_snapshot_boundary)
    monkeypatch.setattr(beam_module, "_resolve_recall_weights", capture_resolver)
    monkeypatch.setattr(memory, "_enhanced_recall_cache_key", capture_key)
    _call(memory, "atomic enhanced snapshot")

    base_weights = calls[0][2]["_resolved_weights"].as_tuple()
    assert weight_resolution_reads == [1]
    assert base_weights == new_generation
    assert observed_key_weights == [base_weights]


def test_enhanced_nonfinite_yaml_weight_uses_finite_cache_material_and_score(
    enhanced, monkeypatch, tmp_path: Path
):
    """A YAML .inf weight cannot reach enhanced cache material or base scoring."""
    memory, calls = enhanced
    data_dir = tmp_path / "config"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(
        "vec_weight: .inf\nfts_weight: 0.3\nimportance_weight: 0.2\n"
    )
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
    MnemosyneConfig.reset_instance()
    observed = []
    original_key = memory._enhanced_recall_cache_key

    def capture_key(*args, **kwargs):
        observed.append((kwargs["weights"], kwargs["recall_kwargs"].copy()))
        return original_key(*args, **kwargs)

    monkeypatch.setattr(memory, "_enhanced_recall_cache_key", capture_key)
    results = _call(memory, "finite enhanced cache")

    base_weights = calls[0][2]["_resolved_weights"].as_tuple()
    assert base_weights == pytest.approx((0.5, 0.3, 0.2))
    assert observed[0][0] == pytest.approx(base_weights)
    assert all(math.isfinite(value) for value in observed[0][0])
    assert all(math.isfinite(observed[0][1][key]) for key in (
        "vec_weight", "fts_weight", "importance_weight",
    ))
    assert math.isfinite(results[0]["score"])


def test_enhanced_cache_key_uses_effective_finite_temporal_halflife(enhanced):
    """Invalid temporal inputs use the same safe default in recall and cache."""
    memory, calls = enhanced
    common = dict(
        original_query="nonfinite temporal cache key",
        expanded_query="nonfinite temporal cache key",
        top_k=3,
        runtime=SimpleNamespace(cross_session=False),
        use_weibull=False,
        use_mmr=False,
        use_intent=False,
        use_synonyms=False,
        use_associative=False,
        associative_depth=1,
        mmr_lambda=0.7,
        weights=(0.5, 0.3, 0.2),
    )

    def key_for(temporal_halflife: float) -> str:
        return memory._enhanced_recall_cache_key(
            **common,
            recall_kwargs={"temporal_halflife": temporal_halflife},
        )

    default_key = key_for(24.0)
    for invalid in (float("inf"), float("-inf"), float("nan"), 0.0):
        assert key_for(invalid) == default_key
        _call(memory, f"invalid temporal sentinel {invalid!r}", temporal_halflife=invalid)
        assert calls[-1][2]["temporal_halflife"] == 24.0

    assert key_for(48.0) != default_key


@pytest.mark.parametrize(
    "huge_weight", [10 ** 10_000, -(10 ** 10_000)], ids=["positive", "negative"]
)
def test_enhanced_extreme_python_int_uses_defaults_and_finite_cache_material(
    enhanced, monkeypatch, huge_weight
):
    """Enhanced recall sanitizes one overflowing explicit weight before cache use."""
    memory, calls = enhanced
    observed = []
    original_key = memory._enhanced_recall_cache_key

    def capture_key(*args, **kwargs):
        key = original_key(*args, **kwargs)
        observed.append((key, kwargs["weights"], kwargs["recall_kwargs"].copy()))
        return key

    monkeypatch.setattr(memory, "_enhanced_recall_cache_key", capture_key)
    results = _call(
        memory,
        "extreme enhanced weight sentinel",
        vec_weight=huge_weight,
        fts_weight=0.8,
        importance_weight=0.2,
    )

    base_weights = calls[0][2]["_resolved_weights"].as_tuple()
    assert base_weights == (0.5, 0.3, 0.2)
    assert observed[0][1] == base_weights
    assert all(math.isfinite(value) for value in observed[0][1])
    assert all(math.isfinite(observed[0][2][key]) for key in (
        "vec_weight", "fts_weight", "importance_weight",
    ))
    assert "nan" not in repr(observed[0]).lower()
    assert "inf" not in repr(observed[0]).lower()
    assert all(math.isfinite(result["score"]) for result in results)



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
