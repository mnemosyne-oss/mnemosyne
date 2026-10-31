"""Polyphonic recall must return min(top_k, eligible) results, never fewer.

Grid: tier (working, episodic) x top_k (at or under the old per-voice cap of
20, over it, over the point where the removed metadata-sized context budget
cut the engine's 2x over-fetch below top_k, and over the eligible count).
"""
import sqlite3

import numpy as np
import pytest

import mnemosyne.core.beam as bm
from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine, PolyphonicResult

ELIGIBLE = 100


def _distinct(i):
    # Distinct words per row so the diversity re-rank keeps every one.
    return f"record{i} alpha{i} bravo{i} charlie{i} delta{i}"


@pytest.fixture
def beam(tmp_path, monkeypatch):
    query = np.ones(384, dtype=np.float32)
    query /= np.linalg.norm(query)
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(bm._embeddings, "embed_query", lambda text: query)
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: np.stack([query for _ in texts]))
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    # Only the vector voice: the others would add rows and hide a short vector voice.
    for voice in ("TEMPORAL", "GRAPH", "FACT"):
        monkeypatch.setenv(f"MNEMOSYNE_VOICE_{voice}", "0")
    monkeypatch.delenv("MNEMOSYNE_VOICE_VECTOR", raising=False)
    return bm.BeamMemory(session_id="top-k-session", db_path=tmp_path / "memory.db")


def _seed_working(beam):
    for i in range(ELIGIBLE):
        beam.remember(_distinct(i), source="user", dedupe=False)


def _seed_episodic(beam):
    pytest.importorskip("sqlite_vec")
    for i in range(ELIGIBLE):
        beam.consolidate_to_episodic(_distinct(i), [], source="user")


@pytest.mark.parametrize("seed,tier", [(_seed_working, "working"), (_seed_episodic, "episodic")],
                         ids=["working", "episodic"])
@pytest.mark.parametrize("top_k", [10, 20, 50, 80, 150])
def test_polyphonic_recall_returns_full_top_k(beam, seed, tier, top_k):
    seed(beam)
    results = beam.recall("unrelated opaque query", top_k=top_k)
    expected = min(top_k, ELIGIBLE)
    assert len(results) == expected
    assert len({r["id"] for r in results}) == expected
    assert {r["tier"] for r in results} == {tier}


@pytest.mark.parametrize("top_k", [1, 50, ELIGIBLE])
def test_memoria_supplement_does_not_displace_memories(beam, monkeypatch, top_k):
    # The MEMORIA entry is added on top of the top_k memories. It has no
    # relevance check yet, so letting it take a slot would drop a real match.
    _seed_working(beam)
    monkeypatch.setattr(beam, "memoria_retrieve",
                        lambda query, top_k: {"source": "facts", "context": "record0 is alpha0"})
    results = beam.recall("unrelated opaque query", top_k=top_k)
    assert results[0]["tier"] == "memoria"
    assert [r["tier"] for r in results[1:]] == ["working"] * top_k
    # Only memories that are returned get credited with a recall.
    credited = dict(beam.conn.execute(
        "SELECT id, recall_count FROM working_memory WHERE recall_count > 0").fetchall())
    assert credited == {r["id"]: 1 for r in results[1:]}


def _reference_diversity_rerank(results, top_k):
    # The pairwise word-Jaccard selection as first written, kept as the oracle
    # for the precomputed-word-set version.
    def similarity(a, b):
        words_a = (a.content or a.metadata.get("content") or "").lower().split()
        words_b = (b.content or b.metadata.get("content") or "").lower().split()
        if not words_a or not words_b:
            return 0.0
        set_a, set_b = set(words_a), set(words_b)
        return len(set_a & set_b) / len(set_a | set_b)

    selected = []
    for result in sorted(results.values(), key=lambda x: x.combined_score, reverse=True):
        if len(selected) >= top_k:
            break
        if all(similarity(result, sel) <= 0.8 for sel in selected):
            selected.append(result)
    return selected


@pytest.mark.parametrize("seed", range(50))
def test_diversity_rerank_matches_reference(seed):
    import random

    from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine, PolyphonicResult

    rng = random.Random(seed)
    # A small vocabulary makes near-duplicates (Jaccard > 0.8) common.
    vocab = [f"w{i}" for i in range(8)]
    results = {}
    for i in range(rng.randint(0, 120)):
        words = rng.sample(vocab, rng.randint(0, 6))
        content = " ".join(words) if rng.random() < 0.8 else ""
        meta = {} if content or rng.random() < 0.5 else {"content": " ".join(words).upper()}
        results[f"m{i}"] = PolyphonicResult(memory_id=f"m{i}", combined_score=rng.random(),
                                            voice_scores={}, metadata=meta, content=content)
    top_k = rng.randint(1, 100)
    engine = PolyphonicRecallEngine.__new__(PolyphonicRecallEngine)
    got = [r.memory_id for r in engine._diversity_rerank(results, top_k)]
    want = [r.memory_id for r in _reference_diversity_rerank(results, top_k)]
    assert got == want


def test_content_hydration_stays_under_sqlite_variable_limit(beam):
    # Older SQLite builds allow 999 bound variables, and a large top_k now lets
    # the vector voice pass more candidates than that. A failed hydration query
    # leaves rows without content, and the diversity re-rank can't compare them.
    ids = [f"m{i}" for i in range(1200)]
    beam.conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, ?, 'user', datetime('now'), 'top-k-session', 0.5, 'session')",
        [(mid, _distinct(i)) for i, mid in enumerate(ids)],
    )
    beam.conn.commit()
    if hasattr(beam.conn, "setlimit"):  # Python 3.11+
        beam.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 1000)
    results = {mid: PolyphonicResult(memory_id=mid, combined_score=0.0, voice_scores={}, metadata={})
               for mid in ids}

    PolyphonicRecallEngine(db_path=beam.db_path, conn=beam.conn)._hydrate_result_content(results)

    assert all(r.content == _distinct(i) for i, r in enumerate(results.values()))


LARGE = 1000


def _seed_large_working_under_limit(beam):
    # Distinct rows, so all of them survive the re-rank and reach the
    # per-result queries; duplicates would be dropped before them.
    if not hasattr(beam.conn, "setlimit"):  # Python 3.11+
        pytest.skip("needs Connection.setlimit")
    vec = bm._embeddings.embed_query("unrelated opaque query")
    beam.conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, ?, 'user', datetime('now'), 'top-k-session', 0.5, 'session')",
        [(f"m{i}", _distinct(i)) for i in range(LARGE)],
    )
    for i in range(LARGE):
        bm._store_working_embedding(beam.conn, f"m{i}", vec)
    beam.conn.commit()
    beam.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)


def test_recall_tracking_stays_under_sqlite_variable_limit(beam):
    count = LARGE
    _seed_large_working_under_limit(beam)

    results = beam.recall("unrelated opaque query", top_k=count)

    assert len(results) == count
    credited = dict(beam.conn.execute("SELECT id, recall_count FROM working_memory").fetchall())
    assert credited == {r["id"]: 1 for r in results}


def _record_id_lookups(monkeypatch):
    # The rows these lookups return can also arrive on the results, so a
    # lookup that silently returned nothing would still pass a result check.
    lookups = []
    real = bm._select_by_ids

    def record(conn, select, ids):
        rows = real(conn, select, ids)
        lookups.append((select, len(ids), len(rows)))
        return rows

    monkeypatch.setattr(bm, "_select_by_ids", record)
    return lookups


def test_enhanced_recall_weibull_stays_under_sqlite_variable_limit(beam, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    _seed_large_working_under_limit(beam)
    lookups = _record_id_lookups(monkeypatch)

    results = beam.recall_enhanced("unrelated opaque query", top_k=LARGE, use_cache=False,
                                   use_mmr=False, use_intent=False, use_synonyms=False)

    assert len({r["id"] for r in results}) == LARGE
    assert ("SELECT id, memory_type FROM working_memory WHERE id IN", LARGE, LARGE) in lookups


def test_cross_tier_dedup_stays_under_sqlite_variable_limit(beam):
    if not hasattr(beam.conn, "setlimit"):  # Python 3.11+
        pytest.skip("needs Connection.setlimit")
    summary = "ep-summary"
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, summary_of) "
        "VALUES (?, 'summary', 'user', datetime('now'), 'top-k-session', 'w-covered')", (summary,))
    beam.conn.commit()
    beam.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
    results = ([{"id": summary, "tier": "episodic", "score": 0.9},
                {"id": "w-covered", "tier": "working", "score": 0.1}]
               + [{"id": f"e{i}", "tier": "episodic", "score": 0.5} for i in range(LARGE)])

    kept = beam._dedup_cross_tier_summary_links(results)

    # The summary outscores its source, so only the covered working row goes.
    assert [r["id"] for r in kept] == [r["id"] for r in results if r["id"] != "w-covered"]


def test_evidence_pack_stays_under_sqlite_variable_limit(beam, monkeypatch):
    _seed_large_working_under_limit(beam)
    lookups = _record_id_lookups(monkeypatch)

    pack = beam.recall_with_evidence_pack("unrelated opaque query", top_k=10,
                                          candidate_k=LARGE, pack_k=5)

    assert len(pack["primary"]) == 10
    # Every candidate is checked for consolidation and given its session.
    assert ("SELECT id, consolidated_at FROM working_memory WHERE id IN", LARGE, LARGE) in lookups
    assert ("SELECT id, session_id FROM working_memory WHERE id IN", LARGE, LARGE) in lookups


@pytest.mark.parametrize("top_k", [20, 50])
def test_temporal_voice_supplies_full_top_k_without_embeddings(beam, monkeypatch, top_k):
    # With no embedding backend the temporal voice is the only source for a
    # time-worded query, so its own limit decides how many rows come back.
    monkeypatch.setattr(bm._embeddings, "available", lambda: False)
    monkeypatch.setenv("MNEMOSYNE_VOICE_VECTOR", "0")
    monkeypatch.delenv("MNEMOSYNE_VOICE_TEMPORAL", raising=False)
    _seed_working(beam)

    results = beam.recall("what happened today", top_k=top_k)

    assert len({r["id"] for r in results}) == top_k
    assert {r["tier"] for r in results} == {"working"}


@pytest.mark.parametrize("count", [0, 1, 900, 901, 1800])
def test_select_by_ids_returns_every_row_across_chunks(beam, count):
    ids = [f"s{i}" for i in range(count)]
    beam.conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, scope) "
        "VALUES (?, ?, 'user', datetime('now'), 'top-k-session', 'session')",
        [(mid, _distinct(i)) for i, mid in enumerate(ids)],
    )

    rows = bm._select_by_ids(beam.conn, "SELECT id FROM working_memory WHERE id IN", ids)

    assert sorted(r["id"] for r in rows) == sorted(ids)
