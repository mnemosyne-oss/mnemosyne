"""Tests for query-side vec0 dimension-mismatch degradation in _vec_search().

The write path already degrades: ``_wm_vec_upsert`` failures are logged and
the vector is dropped (``vec_working upsert failed ...``). The working-memory
query path degrades too: ``_wm_vec_search_sqlite`` wraps its KNN in
try/except and returns ``[]``. ``_vec_search`` (the episodic KNN over
``vec_episodes``) was the one unprotected path: a query embedding whose
dimension disagreed with the table's raised ``sqlite3.OperationalError``
straight out of ``recall()``, crashing the agent process. It must degrade the
same way, logging an actionable pointer to the existing mismatch guidance.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import mnemosyne.core.beam as beam


def _store_at(monkeypatch, tmp_path: Path, name: str, dim: int):
    """Create a store whose vec0 tables are dimensioned at `dim`.

    Uses monkeypatch so the module global is restored after the test: other
    suites assert on the default EMBEDDING_DIM and must not observe ours.
    """
    db = tmp_path / name
    monkeypatch.setattr(beam, "EMBEDDING_DIM", dim)
    beam.init_beam(db)
    return db


def test_vec_search_degrades_on_dimension_mismatch_with_guidance(tmp_path, monkeypatch, caplog):
    """A 384-dim query against a 768-dim vec_episodes must return [] and log
    the self-heal guidance, not raise out of recall()."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = _store_at(monkeypatch, tmp_path, "store.db", 768)
    # The misconfiguration: the process is now configured for 384.
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)

    with caplog.at_level("ERROR", logger="mnemosyne.core.beam"):
        rows = beam._vec_search(beam._get_connection(db), [0.01] * 384, k=5)
    assert rows == []
    logged = " ".join(r.message for r in caplog.records)
    expected = beam._dim_mismatch_message(768, 384)
    assert "dimension mismatch" in logged.lower() and "vec_episodes" in logged.lower(), logged
    assert expected.strip() in logged, logged


def test_vec_search_guidance_when_config_agrees_but_query_differs(tmp_path, monkeypatch, caplog):
    """The issue's actual failure shape: table AND configured EMBEDDING_DIM
    agree (both 1024) while the endpoint serves a 384-dim query vector. The
    guidance branch must classify against the submitted query vector's
    dimension, not the config, and emit the self-heal guidance."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = _store_at(monkeypatch, tmp_path, "agree.db", 1024)
    # Config agrees with the table (1024): a config-vs-table comparison would
    # take the generic branch and drop the guidance.
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 1024)

    with caplog.at_level("ERROR", logger="mnemosyne.core.beam"):
        rows = beam._vec_search(beam._get_connection(db), [0.01] * 384, k=5)
    assert rows == []
    logged = " ".join(r.message for r in caplog.records)
    # All three dimensions are reported separately and truthfully...
    assert "query vector is 384-dim" in logged, logged
    assert "table is 1024-dim" in logged, logged
    assert "process configured 1024-dim" in logged, logged
    # ...and the guidance points at the embedding endpoint, NOT at a reindex:
    # the store and the configuration agree, so reindex advice would be false.
    assert "embedding endpoint/model served a 384-dim query vector" in logged, logged
    assert "no reindex is needed" in logged, logged
    assert "mnemosyne reindex" not in logged, logged


def test_classification_requires_dim_error_signal():
    """Only sqlite-vec's own dimension-mismatch error may take the guidance
    path. An unrelated OperationalError (locked database) must stay on the
    generic warning path even when the submitted and stored dimensions
    disagree, or it would emit false reindex guidance."""
    import sqlite3

    dim_error = sqlite3.OperationalError(
        'Dimension mismatch for query vector for the "embedding" column. '
        "Expected 1024 dimensions but received 384."
    )
    locked_error = sqlite3.OperationalError("database is locked")
    assert beam._is_query_dim_mismatch(dim_error, 384, 1024) is True
    assert beam._is_query_dim_mismatch(locked_error, 384, 1024) is False
    assert beam._is_query_dim_mismatch(dim_error, 1024, 1024) is False
    assert beam._is_query_dim_mismatch(dim_error, 384, None) is False


def test_vec_search_degrades_on_non_operational_sqlite_error(monkeypatch, caplog):
    """A sqlite3.Error that is not an OperationalError (DatabaseError,
    InternalError, ...) must also degrade: recall() does not guard this call,
    so narrowing the guard to OperationalError alone would let siblings crash
    the process again."""
    import sqlite3

    class _BoomConn:
        def execute(self, *a, **k):
            raise sqlite3.DatabaseError("disk I/O error")

    monkeypatch.setattr(beam, "_effective_vec_type", lambda conn, table=None: "float32")
    with caplog.at_level("WARNING", logger="mnemosyne.core.beam"):
        rows = beam._vec_search(_BoomConn(), [0.01] * 768, k=5)
    assert rows == []
    assert any("vec_episodes query failed" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


def test_vec_search_generic_warning_when_table_dim_unreadable(tmp_path, monkeypatch, caplog):
    """When the table's declared dimension cannot be read, the mismatch
    cannot be classified: degrade with the generic warning, never a raise."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = _store_at(monkeypatch, tmp_path, "unreadable.db", 768)
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    monkeypatch.setattr(beam, "_existing_vec_dim", lambda conn: None)

    with caplog.at_level("WARNING", logger="mnemosyne.core.beam"):
        rows = beam._vec_search(beam._get_connection(db), [0.01] * 384, k=5)
    assert rows == []
    assert any("vec_episodes query failed" in r.message for r in caplog.records), [
        r.message for r in caplog.records
    ]


def test_vec_search_returns_stored_row_when_dimension_matches(tmp_path, monkeypatch, caplog):
    """The guard must not weaken healthy retrieval: with the table populated
    at the configured dimension, the stored rowid comes back, warning-free."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = _store_at(monkeypatch, tmp_path, "match.db", 768)
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    conn = beam._get_connection(db)

    # Insert one episodic memory and its 768-dim vector through the production
    # helper (normalization + quantization), bypassing only the embedding
    # layer: the KNN is what is under test. vec_episodes rowids index
    # episodic_memory, so the probe belongs there (a working_memory row would
    # be an orphan vector).
    cur = conn.execute(
        "INSERT INTO episodic_memory (id, content) VALUES ('probe-ep', 'healthy retrieval probe')"
    )
    probe_rowid = cur.lastrowid
    vec = [0.01] * 768
    beam._vec_insert(conn, probe_rowid, vec)
    conn.commit()

    with caplog.at_level("WARNING", logger="mnemosyne.core.beam"):
        rows = beam._vec_search(conn, vec, k=5)
    assert [r["rowid"] for r in rows] == [probe_rowid], rows
    assert not [r for r in caplog.records if "mismatch" in r.message.lower() or "failed" in r.message.lower()], [
        r.message for r in caplog.records
    ]


def test_linear_recall_serves_lexical_candidates_under_mismatch(tmp_path, monkeypatch):
    """End-to-end contract (MNEMOSYNE_POLYPHONIC_RECALL unset, linear path):
    with a dimension mismatch disabling the vector voices, recall() must still
    return the FTS/keyword candidates. Importance is a scoring input here, not
    an independent voice."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)
    db = _store_at(monkeypatch, tmp_path, "e2e.db", 768)

    mem = beam.BeamMemory(session_id="s", db_path=db)
    wm_id = mem.remember("quantum harmonic oscillator lecture notes", source="e2e")
    mem.consolidate_to_episodic(
        summary="quantum harmonic oscillator summary",
        source_wm_ids=[wm_id],
        importance=0.8,
    )

    # The misconfiguration: query-time embedding resolves 384 against the
    # 768-dim tables. Stub the query embedding to the wrong dimension rather
    # than standing up an HTTP endpoint: the KNN rejection is identical.
    import numpy as np

    monkeypatch.setattr(
        beam._embeddings, "embed_query", lambda q: np.array([0.01] * 384, dtype=np.float32)
    )
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)

    results = mem.recall("quantum harmonic oscillator", top_k=5)
    assert results, "lexical voices must still serve recall under a dimension mismatch"
    # The recalled rows must actually be the seeded memories, not just
    # "something came back".
    contents = [(r.get("content") or "") if isinstance(r, dict) else getattr(r, "content", "") for r in results]
    assert any("quantum harmonic oscillator" in c for c in contents), contents
