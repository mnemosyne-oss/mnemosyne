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

import sqlite3

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
    expected = beam._dim_mismatch_message(
        (("vec_episodes", 768), ("vec_working", 768), ("vec_facts", 768)), 384
    )
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


def test_vec_search_uses_vec_episodes_dim_in_mixed_schema(tmp_path, monkeypatch, caplog):
    """Mixed/partially migrated stores can carry vec tables at different
    dimensions. The episodic KNN's mismatch classification must read
    vec_episodes' OWN dimension, not whichever vec table the generic lookup
    happens to return first: a vec_working at 384 alongside a vec_episodes at
    768 must not steer the guidance at 384."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = _store_at(monkeypatch, tmp_path, "mixed.db", 768)
    conn = beam._get_connection(db)
    # Partially migrated shape: rebuild vec_working at 384 while vec_episodes
    # stays at 768 (same trick test_vec_dim_guard uses to simulate stores
    # written under a different dimension).
    conn.execute("DROP TABLE vec_working")
    conn.commit()
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)
    beam.init_beam(db)  # guard leaves vec_episodes at 768 untouched
    # Main's guarded init no longer recreates vec_working under a mismatch, so
    # build the mixed shape explicitly: vec_working at 384, vec_episodes at 768.
    conn.execute(
        "CREATE VIRTUAL TABLE vec_working USING vec0(embedding float32[384])"
    )
    conn.commit()
    assert beam._existing_vec_dim(conn) is None  # mixed store has no honest scalar
    assert dict(beam._existing_vec_dims(conn))["vec_episodes"] == 768
    assert dict(beam._existing_vec_dims(conn))["vec_working"] == 384

    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    with caplog.at_level("ERROR", logger="mnemosyne.core.beam"):
        rows = beam._vec_search(conn, [0.01] * 384, k=5)
    assert rows == []
    logged = " ".join(r.message for r in caplog.records)
    assert "table is 768-dim" in logged, logged
    # A mixed store gets the every-table reindex guidance even though
    # vec_episodes agrees with the configuration; the endpoint branch's
    # "stored vectors are fine" claim is only true for a uniform store.
    assert "vec_episodes=768" in logged and "vec_working=384" in logged, logged
    assert "384-dim, table is 384" not in logged, logged


def test_classification_requires_dim_error_signal():
    """Only sqlite-vec's own dimension-mismatch error may take the degrade
    path. An unrelated OperationalError (locked database) must propagate even
    when the submitted and stored dimensions disagree, or a real storage
    failure would be misread as a mismatch."""
    dim_error = sqlite3.OperationalError(
        'Dimension mismatch for query vector for the "embedding" column. '
        "Expected 1024 dimensions but received 384."
    )
    locked_error = sqlite3.OperationalError("database is locked")
    assert beam._is_query_dim_mismatch(dim_error, 384, 1024) is True
    assert beam._is_query_dim_mismatch(locked_error, 384, 1024) is False
    assert beam._is_query_dim_mismatch(dim_error, 1024, 1024) is False
    assert beam._is_query_dim_mismatch(dim_error, 384, None) is False


class _ProbeResult:
    """Cursor-like result for the sqlite_master probes (type read and
    dimension read alike): a 768-dim float32 vec_episodes DDL."""

    def fetchone(self):
        return ("CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding float32[768])",)

    def fetchall(self):
        return [self.fetchone()]


class _BoomConn:
    """Connection stub that fails on one side of the KNN only.

    fail_schema=False: serves the sqlite_master probe (float32 DDL) and
    raises `exc` on the KNN query itself. fail_schema=True: raises `exc` on
    the schema probe, before any KNN query is built.
    """

    def __init__(self, exc, fail_schema=False):
        self._exc = exc
        self._fail_schema = fail_schema

    def execute(self, sql, *a, **k):
        if "sqlite_master" in sql:
            if self._fail_schema:
                raise self._exc
            return _ProbeResult()
        if self._fail_schema:
            raise AssertionError("KNN query reached after schema probe failure")
        raise self._exc


_PROPAGATION_EXCS = (
    sqlite3.OperationalError("database is locked"),
    sqlite3.DatabaseError("disk I/O error"),
    sqlite3.DatabaseError("database disk image is malformed"),
)


def test_vec_search_propagates_unrelated_sqlite_errors():
    """Only a confirmed dimension mismatch may degrade. Unrelated
    OperationalError / DatabaseError (locked database, disk I/O, corruption)
    raised by the KNN itself must propagate unchanged: silently returning []
    there would convert a real storage failure into an unexplained loss of
    the vector voice."""
    for exc in _PROPAGATION_EXCS:
        with pytest.raises(type(exc)) as caught:
            beam._vec_search(_BoomConn(exc), [0.01] * 768, k=5)
        assert caught.value is exc


def test_vec_search_propagates_schema_probe_failures():
    """A lock / I/O / corruption failure in the pre-KNN schema lookup (the
    strict vec_episodes type read) must surface with the original exception
    and message, not be swallowed into a float32 fallback that later
    resurfaces as a misleading vector-type or dimension error."""
    for exc in _PROPAGATION_EXCS:
        with pytest.raises(type(exc), match=str(exc)) as caught:
            beam._vec_search(_BoomConn(exc, fail_schema=True), [0.01] * 768, k=5)
        assert caught.value is exc


@pytest.mark.parametrize("vec_type", ["float32", "int8", "bit"])
def test_vec_search_degrades_on_confirmed_mismatch_for_every_vec_type(
    tmp_path, monkeypatch, caplog, vec_type
):
    """Confirmed dimension mismatches still fall back safely across all three
    vec encodings: the KNN returns [] and the diagnostic names both dims."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    # The declared vec0 type must match the query path: create the store
    # under VEC_TYPE so the DDL itself carries the encoding, rather than
    # monkeypatching _effective_vec_type against a table declared int8
    # (sqlite-vec would then reject the vector TYPE, not the dimension).
    monkeypatch.setattr(beam, "VEC_TYPE", vec_type)
    db = _store_at(monkeypatch, tmp_path, f"mismatch_{vec_type}.db", 768)
    # _detect_vec_type silently falls back (bit -> int8 -> float32) when the
    # installed sqlite-vec rejects the requested encoding; assert the premise
    # so the parametrization cannot quietly degrade into weaker runs.
    assert beam._vec_table_type_strict(beam._get_connection(db)) == vec_type

    with caplog.at_level("ERROR", logger="mnemosyne.core.beam"):
        rows = beam._vec_search(beam._get_connection(db), [0.01] * 384, k=5)
    assert rows == []
    logged = " ".join(r.message for r in caplog.records)
    assert "query vector is 384-dim" in logged, logged
    assert "table is 768-dim" in logged, logged


def test_vec_search_raises_when_table_dim_unreadable(tmp_path, monkeypatch, caplog):
    """When a successful dimension read yields no declared dimension, the
    mismatch cannot be confirmed, so the KNN error propagates rather than
    being silently swallowed."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    db = _store_at(monkeypatch, tmp_path, "unreadable.db", 768)
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 768)
    monkeypatch.setattr(beam, "_vec_table_dim_strict", lambda conn, table=None: None)

    with caplog.at_level("ERROR", logger="mnemosyne.core.beam"):
        with pytest.raises(sqlite3.OperationalError, match="(?i)dimension mismatch"):
            beam._vec_search(beam._get_connection(db), [0.01] * 384, k=5)
    assert "Dimension mismatch querying vec_episodes" not in caplog.text


def test_vec_search_dim_probe_failure_replaces_knn_exception():
    """When the KNN raises a dimension mismatch but the follow-up dimension
    probe fails with its own SQLite error, the probe error is the one that
    propagates: it names the real storage problem, and silently re-raising
    the KNN exception instead would discard it."""
    knn_exc = sqlite3.OperationalError(
        'Dimension mismatch for query vector for the "embedding" column. '
        "Expected 768 dimensions but received 384."
    )
    probe_exc = sqlite3.DatabaseError("database disk image is malformed")

    class _DimProbeBoomConn:
        """Serves the type probe (first sqlite_master read), raises the KNN
        dimension mismatch, then fails the dimension probe (second
        sqlite_master read) with a distinct error."""

        def __init__(self):
            self._schema_reads = 0

        def execute(self, sql, *a, **k):
            if "sqlite_master" in sql:
                self._schema_reads += 1
                if self._schema_reads == 1:
                    return _ProbeResult()
                raise probe_exc
            raise knn_exc

    conn = _DimProbeBoomConn()
    with pytest.raises(sqlite3.DatabaseError) as caught:
        beam._vec_search(conn, [0.01] * 384, k=5)
    assert caught.value is probe_exc


def test_vec_search_lock_error_survives_dim_probe_failure():
    """Two-failure identity: when the KNN raises an unrelated lock error and
    the follow-up dimension probe would fail too, the ORIGINAL lock error is
    the one that propagates. The handler classifies the KNN error's own text
    before any further schema probe, so a failing probe can never replace an
    unrelated failure."""
    lock_exc = sqlite3.OperationalError("database is locked")
    probe_exc = sqlite3.DatabaseError("database disk image is malformed")

    class _LockThenBoomConn:
        """Serves the type probe (first sqlite_master read), raises the lock
        error on the KNN, and fails every later sqlite_master read."""

        def __init__(self):
            self._schema_reads = 0

        def execute(self, sql, *a, **k):
            if "sqlite_master" in sql:
                self._schema_reads += 1
                if self._schema_reads == 1:
                    return _ProbeResult()
                raise probe_exc
            raise lock_exc

    with pytest.raises(sqlite3.OperationalError) as caught:
        beam._vec_search(_LockThenBoomConn(), [0.01] * 384, k=5)
    assert caught.value is lock_exc


def test_vec_search_guidance_catalog_failure_propagates():
    """After a CONFIRMED dimension mismatch, the guidance's all-tables
    catalog read is strict: a diagnostic lock/I/O/corruption failure there
    propagates instead of being swallowed into degraded guidance and a
    silent [] return."""
    knn_exc = sqlite3.OperationalError(
        'Dimension mismatch for query vector for the "embedding" column. '
        "Expected 768 dimensions but received 384."
    )
    diag_exc = sqlite3.DatabaseError("database disk image is malformed")

    class _GuidanceBoomConn:
        """Serves the type and dimension probes (first two sqlite_master
        reads), raises a confirmed KNN dimension mismatch, then fails the
        guidance's all-tables catalog read (third sqlite_master read)."""

        def __init__(self):
            self._schema_reads = 0

        def execute(self, sql, *a, **k):
            if "sqlite_master" in sql:
                self._schema_reads += 1
                if self._schema_reads <= 2:
                    return _ProbeResult()
                raise diag_exc
            raise knn_exc

    with pytest.raises(sqlite3.DatabaseError) as caught:
        beam._vec_search(_GuidanceBoomConn(), [0.01] * 384, k=5)
    assert caught.value is diag_exc


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


def test_linear_recall_serves_lexical_candidates_under_mismatch(tmp_path, monkeypatch, caplog):
    """End-to-end contract (MNEMOSYNE_POLYPHONIC_RECALL unset, linear path):
    with a dimension mismatch disabling the vector voices, recall() must still
    return the FTS/keyword candidates, and the episodic KNN degradation must
    actually have run (the vec_episodes mismatch diagnostic in the log), so
    the lexical results are proven to be the fallback, not a vector path that
    silently never executed. Importance is a scoring input here, not an
    independent voice."""
    if not beam._SQLITE_VEC_AVAILABLE:
        pytest.skip("sqlite-vec unavailable")

    monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)
    # The two candidates are a (working-memory source, episodic summary) pair
    # linked by source_wm_ids; the cross-tier summary dedup would collapse
    # exactly that pair before results are returned. This test proves each
    # lexical voice serves its own row, so the dedup is disabled for it.
    monkeypatch.setenv("MNEMOSYNE_CROSS_TIER_DEDUP", "0")
    db = _store_at(monkeypatch, tmp_path, "e2e.db", 768)

    mem = beam.BeamMemory(session_id="s", db_path=db)
    wm_id = mem.remember("quantum harmonic oscillator lecture notes", source="e2e")
    mem.consolidate_to_episodic(
        summary="quantum harmonic oscillator summary",
        source_wm_ids=[wm_id],
        importance=0.8,
    )

    # The misconfiguration: query-time embedding resolves 384 against the
    # 768-dim tables. Stub available() too: under MNEMOSYNE_NO_EMBEDDINGS the
    # vector voices are gated off before embed_query is ever consulted, and
    # the KNN degradation path this test exists to prove would never run.
    import numpy as np

    monkeypatch.setattr(beam._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam._embeddings, "embed_query", lambda q: np.array([0.01] * 384, dtype=np.float32)
    )
    monkeypatch.setattr(beam, "EMBEDDING_DIM", 384)

    with caplog.at_level("ERROR", logger="mnemosyne.core.beam"):
        results = mem.recall("quantum harmonic oscillator", top_k=5)
    assert results, "lexical voices must still serve recall under a dimension mismatch"
    # The recalled rows must be both seeded lexical candidates, each asserted
    # on its own: the working-memory record and the episodic summary. A
    # single "any" match here would pass if either voice silently stopped
    # serving its candidate.
    contents = [(r.get("content") or "") if isinstance(r, dict) else getattr(r, "content", "") for r in results]
    assert any("lecture notes" in c for c in contents), contents
    assert any("summary" in c for c in contents), contents
    # Proof the KNN degradation path ran (not that the vector voice was merely
    # absent): the guard's own diagnostic names the table and the query dim.
    logged = " ".join(r.message for r in caplog.records)
    assert "query vector is 384-dim" in logged and "table is 768-dim" in logged, logged
