"""Invalidating a sole working source retires its existing summaries."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from mnemosyne.core import beam, embeddings
from mnemosyne.core.beam import BeamMemory

WORKING = "The Atlas team works from its Massachusetts headquarters in Boston."
SUMMARY = "Atlas office location: Boston waterfront campus."
QUERY = "Atlas office location Boston waterfront"


@pytest.fixture
def memory(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    return BeamMemory(session_id="atl-s1", db_path=tmp_path / "memory.db")


def _state(memory, table, ident):
    return tuple(memory.conn.execute(
        f"SELECT valid_until, superseded_by FROM {table} WHERE id=?", (ident,)
    ).fetchone())


def _episode(memory, text, *sources, **kwargs):
    ident = memory.consolidate_to_episodic(
        text, source_wm_ids=list(sources), tier=1, importance=0.9, **kwargs,
    )
    assert ident
    return ident


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("polyphonic", [False, True], ids=["linear", "polyphonic"])
def test_invalidate_working_source_retires_summary_in_recall_and_cache(
    memory, monkeypatch, replacement, polyphonic,
):
    old = memory.remember(WORKING, source="note", importance=0.9)
    summary = _episode(memory, SUMMARY, old)
    successor = memory.remember("The Atlas office relocated to Denver.", source="note") if replacement else None

    if polyphonic:
        np = pytest.importorskip("numpy")
        if not beam._vec_available(memory.conn):
            pytest.skip("SQLite vector extension unavailable")
        vec = np.zeros(embeddings.EMBEDDING_DIM, dtype=np.float32)
        vec[0] = 1.0
        rowid, = memory.conn.execute(
            "SELECT rowid FROM episodic_memory WHERE id=?", (summary,)
        ).fetchone()
        beam._vec_table_insert(memory.conn, "vec_episodes", rowid, vec.tolist(), commit=False)
        memory.conn.commit()
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        monkeypatch.setattr(embeddings, "available", lambda: True)
        monkeypatch.setattr(embeddings, "embed_query", lambda _query: vec)

    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")

    def enhanced():
        return memory.recall_enhanced(
            QUERY, top_k=10, use_intent=False, use_synonyms=False,
            use_mmr=False, use_weibull=False, _track_recall=False,
        )

    warm = enhanced()
    assert summary in {row["id"] for row in warm}
    assert enhanced() == warm
    stats = memory._query_cache.stats()
    assert stats["size"] == 1
    version = stats["version"]

    assert memory.invalidate(old, replacement_id=successor)

    source_expiry, source_replacement = _state(memory, "working_memory", old)
    assert source_expiry is not None and source_replacement == successor
    assert memory._query_cache.stats()["version"] == version + 1
    assert _state(memory, "episodic_memory", summary) == (source_expiry, None)
    assert not {old, summary} & {row["id"] for row in enhanced()}
    hits = memory.recall(QUERY, top_k=10)
    assert not {old, summary} & {row["id"] for row in hits}


@pytest.mark.parametrize("replacement", [False, True])
def test_cascade_keeps_mixed_and_unrelated_summaries_and_prior_retirement(memory, replacement):
    old = memory.remember(WORKING, source="note")
    other = memory.remember("Badge pickup is at the Saffron Atrium.", source="note")
    successor = memory.remember("The Atlas office relocated to Denver.", source="note") if replacement else None
    first = _episode(memory, SUMMARY, old)
    second = _episode(memory, "The former Boston waterfront office.", old)
    mixed = _episode(memory, SUMMARY + " Pick up badges in the Saffron Atrium.", other, old)
    independent = _episode(memory, "Saffron Atrium badge pickup.", other)
    historical = _episode(memory, "An outdated version of the Atlas summary.", old, valid_until="2000-01-01T00:00:00")
    historical_state = _state(memory, "episodic_memory", historical)

    assert memory.invalidate(old, replacement_id=successor, bank="working_memory")

    stamp, _ = _state(memory, "working_memory", old)
    assert stamp is not None
    assert _state(memory, "episodic_memory", first) == (stamp, None)
    assert _state(memory, "episodic_memory", second) == (stamp, None)
    assert _state(memory, "episodic_memory", mixed) == (None, None)
    assert _state(memory, "episodic_memory", independent) == (None, None)
    assert _state(memory, "episodic_memory", historical) == historical_state


@pytest.mark.parametrize("replacement", [False, True])
@pytest.mark.parametrize("caller_owned", [False, True])
def test_cascade_failure_rolls_back_source_update(memory, replacement, caller_owned):
    old = memory.remember(WORKING, source="note")
    first = _episode(memory, SUMMARY, old)
    successor = memory.remember("The Atlas office relocated to Denver.", source="note") if replacement else None
    memory.conn.executescript("""
        CREATE TRIGGER reject_summary_invalidate
        BEFORE UPDATE OF valid_until ON episodic_memory
        BEGIN SELECT RAISE(ABORT, 'summary update failed'); END;
    """)
    if caller_owned:
        memory.conn.execute("BEGIN")
        memory.conn.execute("UPDATE working_memory SET importance = 0.7 WHERE id = ?", (old,))

    with pytest.raises(sqlite3.IntegrityError, match="summary update failed"):
        memory.invalidate(old, replacement_id=successor)

    assert _state(memory, "working_memory", old) == (None, None)
    assert _state(memory, "episodic_memory", first) == (None, None)
    assert memory.conn.in_transaction is caller_owned
    if caller_owned:
        assert memory.conn.execute(
            "SELECT importance FROM working_memory WHERE id=?", (old,)
        ).fetchone()[0] == 0.7
        memory.conn.rollback()


def test_failed_replacement_and_episodic_target_do_not_cascade(memory):
    old = memory.remember(WORKING, source="note")
    summary = _episode(memory, SUMMARY, old)
    same_id = _episode(memory, "Independent episode with the same id.")
    memory.conn.execute("UPDATE episodic_memory SET id=? WHERE id=?", (old, same_id))
    memory.conn.commit()

    assert not memory.invalidate(old, replacement_id="missing")
    assert _state(memory, "working_memory", old) == (None, None)
    assert _state(memory, "episodic_memory", summary) == (None, None)

    assert memory.invalidate(old, bank="episodic_memory")
    assert _state(memory, "episodic_memory", old)[0] is not None
    assert _state(memory, "working_memory", old) == (None, None)
    assert _state(memory, "episodic_memory", summary) == (None, None)


def test_invalidation_requires_access_to_source_but_reaches_global_derivation(memory):
    local = memory.remember(WORKING, source="note", scope="session")
    local_summary = _episode(memory, SUMMARY, local)
    global_source = memory.remember(
        "This contact address is shared across sessions.", source="note", scope="global",
    )
    foreign = BeamMemory(session_id="atl-s2", db_path=memory.db_path)
    foreign_summary = _episode(foreign, "All sessions use the shared contact address.", global_source)

    assert not foreign.invalidate(local)
    assert _state(memory, "episodic_memory", local_summary) == (None, None)
    assert memory.invalidate(global_source)
    stamp, _ = _state(memory, "working_memory", global_source)
    assert stamp is not None
    assert _state(memory, "episodic_memory", foreign_summary) == (stamp, None)


def test_sleep_discards_summary_when_its_sole_source_is_invalidated_during_summarization(
    memory, monkeypatch,
):
    from mnemosyne.core import local_llm

    old = memory.remember(WORKING, source="note")
    calls = []

    sleep_connection = memory.conn
    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(BeamMemory, session_id=memory.session_id, db_path=memory.db_path).result(5)

        def invalidate_on_other_connection():
            assert writer.conn is not sleep_connection
            writer.conn.execute("PRAGMA busy_timeout=250")
            claim, = writer.conn.execute(
                "SELECT consolidation_claimed_at FROM working_memory WHERE id = ?", (old,)
            ).fetchone()
            assert claim is not None
            return writer.invalidate(old)

        def summarize(lines, **kwargs):
            assert executor.submit(invalidate_on_other_connection).result(5)
            calls.append(lines)
            return SUMMARY

        monkeypatch.setattr(local_llm, "llm_available", lambda: True)
        monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda lines, **kwargs: [lines])
        monkeypatch.setattr(local_llm, "_summarize_memories", summarize)

        result = memory.sleep(force=True, _fleet_census=False)

    assert len(calls) == 1
    assert result["claims_lost"] == 1
    assert _state(memory, "working_memory", old)[0] is not None
    assert memory.conn.execute(
        "SELECT 1 FROM episodic_memory WHERE summary_of=?", (old,)
    ).fetchone() is None


@pytest.mark.parametrize("repeated", [False, True], ids=["single-reference", "legacy-repeated"])
def test_rejects_own_episodic_summary_as_replacement(memory, repeated):
    old = memory.remember(WORKING, source="note")
    dependent = _episode(memory, SUMMARY, old)
    independent = _episode(memory, "The Atlas office has relocated to Denver.")
    if repeated:
        memory.conn.execute(
            "UPDATE episodic_memory SET summary_of=? WHERE id=?", (f"{old},{old}", dependent),
        )
        memory.conn.commit()

    assert not memory.invalidate(old, replacement_id=dependent)
    assert _state(memory, "working_memory", old) == (None, None)
    assert _state(memory, "episodic_memory", dependent) == (None, None)

    assert memory.invalidate(old, replacement_id=independent)
    stamp, successor = _state(memory, "working_memory", old)
    assert stamp is not None and successor == independent
    assert _state(memory, "episodic_memory", dependent) == ((None, None) if repeated else (stamp, None))
    assert _state(memory, "episodic_memory", independent) == (None, None)


def test_preserves_source_order_and_leaves_ambiguous_legacy_provenance(memory):
    old = memory.remember(WORKING, source="note", memory_id="atlas")
    other = memory.remember("Badge pickup is at the Saffron Atrium.", source="note", memory_id="atlas-north")
    single = _episode(memory, SUMMARY, old, old)
    mixed = _episode(memory, SUMMARY + " Pick up badges at the Saffron Atrium.", other, old, other)
    legacy = _episode(memory, "Atlas office and its older waterfront location.", old)
    prefix = _episode(memory, "Atlas north badge pickup.", other)
    memory.conn.execute(
        "UPDATE episodic_memory SET summary_of=? WHERE id=?", (f"{old},{old},{old}", legacy),
    )
    memory.conn.commit()

    def provenance(ident):
        return memory.conn.execute(
            "SELECT summary_of FROM episodic_memory WHERE id=?", (ident,),
        ).fetchone()[0]

    assert provenance(single) == old
    assert provenance(mixed) == f"{other},{old}"
    assert memory.invalidate(old)
    stamp, _ = _state(memory, "working_memory", old)
    assert stamp is not None
    assert _state(memory, "episodic_memory", single) == (stamp, None)
    assert _state(memory, "episodic_memory", legacy) == (None, None)
    assert _state(memory, "episodic_memory", mixed) == (None, None)
    assert _state(memory, "episodic_memory", prefix) == (None, None)


@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_successful_caller_owned_invalidation_uses_outer_transaction(memory, monkeypatch, commit):
    old = memory.remember(WORKING, source="note")
    summary = _episode(memory, SUMMARY, old)
    invalidations = []
    original = memory._invalidate_query_cache

    def track_cache_invalidation():
        invalidations.append(memory.conn.in_transaction)
        original()

    monkeypatch.setattr(memory, "_invalidate_query_cache", track_cache_invalidation)
    memory.conn.execute("BEGIN")
    assert memory.invalidate(old)
    assert memory.conn.in_transaction
    assert invalidations == [True]
    stamp, _ = _state(memory, "working_memory", old)
    assert stamp is not None
    assert _state(memory, "episodic_memory", summary) == (stamp, None)
    if commit:
        memory.conn.commit()
    else:
        memory.conn.rollback()
    assert invalidations == ([True, False] if commit else [True])
    final = (stamp if commit else None, None)
    assert _state(memory, "working_memory", old) == final
    assert _state(memory, "episodic_memory", summary) == final


@pytest.mark.parametrize("refill_after_commit", [False, True], ids=["before-commit", "late-write"])
def test_enhanced_cache_drops_summary_from_concurrent_reader(
    memory, monkeypatch, refill_after_commit,
):
    old = memory.remember(WORKING, source="note")
    summary = _episode(memory, SUMMARY, old)
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")

    def hits(instance):
        return {row["id"] for row in instance.recall_enhanced(
            QUERY, top_k=10, use_intent=False, use_synonyms=False,
            use_mmr=False, use_weibull=False, _track_recall=False,
        )}

    ready, start, before_write, written, committed = (Event() for _ in range(5))
    original_put = beam.QueryCache.put_opaque

    def sync_put(cache, key, rows):
        if any(row["id"] == summary for row in rows):
            before_write.set()
            if refill_after_commit:
                assert committed.wait(5)
        original_put(cache, key, rows)
        written.set()

    monkeypatch.setattr(beam.QueryCache, "put_opaque", sync_put)

    def read_across_commit():
        reader = BeamMemory(session_id=memory.session_id, db_path=memory.db_path)
        ready.set()
        assert start.wait(5)
        assert summary in hits(reader)
        assert committed.wait(5)
        return hits(reader)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(read_across_commit)
        try:
            assert ready.wait(5)
            memory.conn.execute("BEGIN")
            assert memory.invalidate(old)
            start.set()
            assert before_write.wait(5)
            if not refill_after_commit:
                assert written.wait(5)
            memory.conn.commit()
        finally:
            if memory.conn.in_transaction:
                memory.conn.rollback()
            committed.set()
            start.set()

        stale_reader = future.result(5)
    assert summary not in stale_reader

    fresh = BeamMemory(session_id=memory.session_id, db_path=memory.db_path)
    assert summary not in hits(fresh)


@pytest.mark.parametrize("target", ["atlas,atlas", "atlas"], ids=["actual-source", "other-id"])
def test_summary_with_comma_bearing_source_expires_only_for_that_source(memory, target):
    old = memory.remember(WORKING, source="note", memory_id="atlas,atlas")
    other = memory.remember("Atlas garden contact is the groundskeeper.", source="note", memory_id="atlas")
    assert (old, other) == ("atlas,atlas", "atlas")
    summary = _episode(memory, SUMMARY, old, old)
    mixed = _episode(memory, "Atlas office and garden contacts.", old, other)

    assert memory.invalidate(target)

    stamp, _ = _state(memory, "working_memory", target)
    assert stamp is not None
    assert _state(memory, "episodic_memory", summary) == ((stamp, None) if target == old else (None, None))
    assert _state(memory, "episodic_memory", mixed) == (None, None)


def test_ambiguous_legacy_summary_stays_active_after_other_possible_source_is_pruned(memory):
    regular = memory.remember(WORKING, source="note", memory_id="atlas")
    compound = memory.remember("Atlas office is beside the river.", source="note", memory_id="atlas,atlas")
    legacy = _episode(memory, SUMMARY)
    memory.conn.execute(
        "UPDATE episodic_memory SET summary_of=? WHERE id=?", ("atlas,atlas", legacy),
    )
    memory.conn.execute("DELETE FROM working_memory WHERE id=?", (compound,))
    memory.conn.commit()

    assert not memory.invalidate(regular, replacement_id=legacy)
    assert memory.invalidate(regular)
    assert _state(memory, "episodic_memory", legacy) == (None, None)


def test_dedup_uses_exact_sources_for_new_comma_id_summary(memory):
    old = memory.remember(WORKING, source="note", memory_id="atlas,atlas")
    other = memory.remember("Atlas garden contact is the groundskeeper.", source="note", memory_id="atlas")
    summary = _episode(memory, SUMMARY, old)
    results = [
        {"id": summary, "tier": "episodic", "score": 1.0},
        {"id": old, "tier": "working", "score": 0.5},
        {"id": other, "tier": "working", "score": 0.5},
    ]

    hits = memory._dedup_cross_tier_summary_links(results)

    assert {row["id"] for row in hits} == {summary, other}


def test_export_import_preserves_exact_comma_id_provenance(memory, tmp_path):
    old = memory.remember(WORKING, source="note", memory_id="atlas,atlas")
    assert old == "atlas,atlas"
    summary = _episode(memory, SUMMARY, old)
    exported = memory.export_to_dict()
    restored = BeamMemory(session_id=memory.session_id, db_path=tmp_path / "restored" / "memory.db")
    restored.import_from_dict(exported)

    assert restored.invalidate(old)
    stamp, _ = _state(restored, "working_memory", old)
    assert stamp is not None
    assert _state(restored, "episodic_memory", summary) == (stamp, None)


def test_reclaim_keeps_completed_comma_id_claim_and_releases_unrelated_claim(memory):
    old = memory.remember(WORKING, source="note", memory_id="atlas,atlas")
    other = memory.remember("Atlas garden contact is the groundskeeper.", source="note", memory_id="atlas")
    stamp = "2000-01-01T00:00:00"
    memory.conn.executemany(
        "UPDATE working_memory SET consolidated_at=?, consolidation_claimed_at=? WHERE id=?",
        [(stamp, stamp, ident) for ident in (old, other)],
    )
    memory.conn.commit()
    _episode(memory, SUMMARY, old)

    result = memory.reclaim_orphans(stale_after_seconds=1)

    assert result["candidate_ids"] == [other]
    assert result["reclaimed"] == 1
    claims = dict(memory.conn.execute("SELECT id, consolidation_claimed_at FROM working_memory"))
    assert claims == {old: stamp, other: None}


def test_comma_id_summary_is_not_a_valid_replacement_for_its_source(memory):
    old = memory.remember(WORKING, source="note", memory_id="atlas,atlas")
    dependent = _episode(memory, SUMMARY, old)

    assert not memory.invalidate(old, replacement_id=dependent)
    assert _state(memory, "working_memory", old) == (None, None)
    assert _state(memory, "episodic_memory", dependent) == (None, None)


def test_new_mixed_replacement_is_distinct_from_a_comma_bearing_source(memory):
    old = memory.remember(WORKING, source="note", memory_id="atlas,north")
    first = memory.remember("The Atlas garden contact is the groundskeeper.", memory_id="atlas")
    second = memory.remember("The North garden access code was rotated.", memory_id="north")
    independent = _episode(memory, "Separate garden contacts for Atlas and North.", first, second)

    assert memory.invalidate(old, replacement_id=independent)
    assert _state(memory, "working_memory", old)[1] == independent
    assert _state(memory, "episodic_memory", independent) == (None, None)
