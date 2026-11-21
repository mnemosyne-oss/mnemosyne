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
    beam.recall("zulu yankee", top_k=5)
    assert seen["where_sql"], "the dense pool was never queried"
    assert ("consolidated_at IS NULL" in seen["where_sql"]) is (not opt_in)


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
