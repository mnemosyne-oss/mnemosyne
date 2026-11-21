"""#979: consolidated working-memory rows must be reachable by the dense voice.

The default dense candidate predicate excludes `consolidated_at IS NOT NULL`
rows. On an additive store, where consolidation is the normal state of older
working memory, that shrinks the vector voice to "written since the last
sleep" while the archive stays reachable only through FTS5 - a silent loss of
the dense arm exactly where history lives. `MNEMOSYNE_DENSE_INCLUDE_CONSOLIDATED=1`
opts the consolidated rows back into the dense pool of both arms; the default
is unchanged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory, _dense_consolidated_exclusion

ENV = "MNEMOSYNE_DENSE_INCLUDE_CONSOLIDATED"


def _unit(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(beam_module.EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _seed_consolidated_row(beam: BeamMemory, query_vec: np.ndarray) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    # No shared tokens with the query: the lexical arms cannot surface it.
    # The embedding matches the query exactly: only the dense voice can.
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, importance, session_id, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("archived", "wug qlorf mboseo", "fact", now, 0.7, "s-979", "session"),
    )
    beam.conn.execute(
        "UPDATE working_memory SET consolidated_at = ? WHERE id = 'archived'", (now,)
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) "
        "VALUES (?, ?, ?)",
        ("archived", json.dumps(query_vec.astype(np.float32).tolist()), "test-model"),
    )
    beam.conn.commit()


def test_helper_default_excludes_opt_in_clears(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert _dense_consolidated_exclusion() == " AND consolidated_at IS NULL"
    monkeypatch.setenv(ENV, "1")
    assert _dense_consolidated_exclusion() == ""


def test_polyphonic_shares_the_same_switch(monkeypatch):
    from mnemosyne.core.polyphonic_recall import _env_enabled

    monkeypatch.delenv(ENV, raising=False)
    assert _env_enabled(ENV) is False
    monkeypatch.setenv(ENV, "on")
    assert _env_enabled(ENV) is True


@pytest.mark.parametrize("opt_in", [False, True])
def test_linear_recall_dense_predicate_follows_the_switch(tmp_path, monkeypatch, opt_in):
    beam = BeamMemory(session_id="s-979", db_path=tmp_path / "m.db")
    query_vec = _unit(11)
    _seed_consolidated_row(beam, query_vec)
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda _q: query_vec)
    if opt_in:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising=False)

    seen = {}
    real = beam_module._wm_vec_search

    def spy(conn, query_embedding, k=20, **kwargs):
        seen["where_sql"] = kwargs.get("where_sql") or ""
        return real(conn, query_embedding, k=k, **kwargs)

    monkeypatch.setattr(beam_module, "_wm_vec_search", spy)
    # Second consolidated row WITH lexical overlap on the query: the linear
    # scorer blends dense evidence into lexically matched rows only, so this
    # is the row that can show a functional dense_score difference. The
    # zero-overlap "archived" row stays for the polyphonic end-to-end test.
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, importance, session_id, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("archived-lexical", "zulu yankee archiving notes", "fact", now, 0.7, "s-979", "session"),
    )
    beam.conn.execute(
        "UPDATE working_memory SET consolidated_at = ? WHERE id = 'archived-lexical'", (now,)
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) "
        "VALUES (?, ?, ?)",
        ("archived-lexical", json.dumps(query_vec.astype(np.float32).tolist()), "test-model"),
    )
    beam.conn.commit()
    results = beam.recall("zulu yankee", top_k=5)
    assert seen["where_sql"], "the dense pool was never queried"
    assert ("consolidated_at IS NULL" in seen["where_sql"]) is (not opt_in)
    # Functional half of the predicate assertion: with the opt-in the dense
    # voice must score the consolidated row; without it the row may still
    # surface lexically via FTS but must carry no dense evidence.
    archived = [r for r in results if r.get("id") == "archived-lexical"]
    if opt_in:
        assert archived, "the consolidated row was not recalled despite the dense match"
        assert float(archived[0].get("dense_score") or 0.0) > 0.0
    else:
        assert not archived or float(archived[0].get("dense_score") or 0.0) == 0.0


def test_enhanced_recall_cache_is_fresh_across_toggle_flips(monkeypatch, tmp_path):
    """unset -> enabled -> disabled must each compute, never replay a cached state.

    The toggle changes which rows the dense pool admits inside recall(),
    which recall_enhanced() may serve from its request cache, so the
    effective boolean belongs in the cache key next to the other toggles.
    """
    from types import SimpleNamespace

    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setattr(
        beam_module, "resolve_beam_runtime", lambda: SimpleNamespace(cross_session=False)
    )
    memory = BeamMemory(session_id="s-979", db_path=tmp_path / "cache.db")
    calls: list = []

    def fake_recall(query, top_k=40, **kwargs):
        calls.append((query, top_k))
        return [{"id": f"result-{len(calls)}", "content": query, "score": 1.0}]

    memory.recall = fake_recall
    try:
        def run() -> tuple[list, int]:
            out = memory.recall_enhanced(
                "cache isolation query",
                use_weibull=False, use_mmr=False, use_intent=False,
                use_synonyms=False,
            )
            return [r["id"] for r in out], len(calls)

        # A same-state repeat may replay; a state TRANSITION must never
        # replay the OTHER state's entry. Before the toggle entered the key
        # material, enabling returned the unset-state result from cache and
        # disabling returned the enabled-state result (consolidated rows the
        # current setting would not admit).
        monkeypatch.delenv(ENV, raising=False)
        assert run() == (["result-1"], 1), "first call must compute"
        assert run() == (["result-1"], 1), "identical request must hit the cache (fixture sanity)"
        monkeypatch.setenv(ENV, "1")
        assert run() == (["result-2"], 2), "enabling the toggle must not replay the unset-state cache entry"
        monkeypatch.delenv(ENV, raising=False)
        result, computed = run()
        assert result == ["result-1"], "disabling must not serve the enabled-state entry"
        assert computed == 2, "the original unset-state entry may be replayed again"
        monkeypatch.setenv(ENV, "1")
        assert run() == (["result-2"], 2), "the enabled-state entry must still be intact"
    finally:
        cache = getattr(memory, "_query_cache", None)
        if cache is not None:
            cache.close()
        memory.conn.close()


def test_dense_native_vec_working_selection_not_json_fallback(tmp_path, monkeypatch):
    """Native sqlite-vec selection is covered SEPARATELY from the JSON fallback.

    The functional test above seeds memory_embeddings, which is exactly what
    the compatibility fallback reads, so a broken native query could fall
    through to the fallback and still pass there. Here the vec_working table
    is populated with the same backfill the diagnose repair uses, the JSON
    fallback is trapped, and recall under the opt-in must score the
    consolidated row from the native table alone.
    """
    pytest.importorskip("sqlite_vec")
    beam = BeamMemory(session_id="s-979", db_path=tmp_path / "native.db")
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")
    query_vec = _unit(11)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, importance, session_id, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("archived-native", "zulu yankee archive note", "fact", now, 0.7, "s-979", "session"),
    )
    beam.conn.execute(
        "UPDATE working_memory SET consolidated_at = ? WHERE id = 'archived-native'", (now,)
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) "
        "VALUES (?, ?, ?)",
        ("archived-native", json.dumps(query_vec.astype(np.float32).tolist()), "test-model"),
    )
    beam.conn.commit()
    report = beam_module.repair_vec_working(beam.conn)
    assert report.get("inserted", 0) >= 1, "backfill must put the row in vec_working"

    monkeypatch.setenv(ENV, "1")
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda _q: query_vec)
    fallback_calls: list = []
    real_fallback = beam_module._wm_vec_search_fallback

    def spy(*args, **kwargs):
        fallback_calls.append(args)
        return real_fallback(*args, **kwargs)

    monkeypatch.setattr(beam_module, "_wm_vec_search_fallback", spy)

    results = beam.recall("zulu yankee", top_k=5)
    native = [r for r in results if r.get("id") == "archived-native"]
    assert native, "the consolidated row was not recalled via native vec_working"
    assert float(native[0].get("dense_score") or 0.0) > 0.0
    assert not fallback_calls, (
        "native vec_working selection must not consult the JSON fallback"
    )


@pytest.mark.parametrize("opt_in", [False, True])
def test_polyphonic_dense_voice_follows_the_switch(tmp_path, monkeypatch, opt_in):
    beam = BeamMemory(session_id="s-979", db_path=tmp_path / "m.db")
    query_vec = _unit(11)
    _seed_consolidated_row(beam, query_vec)
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(beam_module._embeddings, "embed_query", lambda _q: query_vec)
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    if opt_in:
        monkeypatch.setenv(ENV, "1")
    else:
        monkeypatch.delenv(ENV, raising=False)

    results = beam.recall("zulu yankee", top_k=5)
    ids = {r.get("id") for r in results}
    assert ("archived" in ids) is opt_in
