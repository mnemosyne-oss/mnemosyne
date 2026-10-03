"""Tests for reindex_vectors() — rebuilding vector stores after a model/dim change.

Simulates the motivating bug (sqlite-vec tables stuck at the old dimension after an
embedding-model swap) by recreating the vec0 tables at a wrong dimension, then
verifies reindex_vectors() recreates them at the active dimension, repopulates
working + episodic vectors, refreshes the episodic binary_vector, and leaves recall
working. Also checks that --dry-run writes nothing.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory, reindex_vectors, _effective_vec_type
import mnemosyne.core.embeddings as E


class _Array:
    def __init__(self, vector):
        self.vector = vector

    def tolist(self):
        return self.vector


class _NumpyStub:
    float32 = object()
    asarray = staticmethod(_Array)
    array = staticmethod(lambda vector, dtype=None: _Array(vector))


def _ddl(conn, table):
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    return row[0] if row and row[0] else ""


def _reindex_fixture_beam(tmp_path, *, working=0, episodic=0):
    beam = BeamMemory(session_id="reindex-failure", db_path=str(tmp_path / "m.db"))
    for index in range(working):
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"wm-{index}", f"working source {index}", "test", "2026-01-01T00:00:00", "reindex-failure"),
        )
    for index in range(episodic):
        beam.conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"ep-{index}", f"episodic source {index}", "test", "2026-01-01T00:00:00", "reindex-failure", 0.5),
        )
    beam.conn.commit()
    return beam


@pytest.mark.parametrize(
    "batch_response",
    [None, [[0.1] * 384], [[0.1] * 384, [0.1] * 384, [0.1] * 384]],
)
def test_reindex_rejects_failed_or_partial_embedding_batches(tmp_path, monkeypatch, batch_response):
    """A failed or short embedding batch must never yield a reindexed result."""
    beam = _reindex_fixture_beam(tmp_path, working=2)
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "embed", lambda _contents: batch_response)

    with pytest.raises(RuntimeError, match="working_memory embedding batch"):
        reindex_vectors(beam.conn)


@pytest.mark.parametrize(
    ("invalid_vector_factory", "reason"),
    [
        (lambda: ["not-a-number"] * E.EMBEDDING_DIM, "convertible numeric"),
        (lambda: [[0.1]] * E.EMBEDDING_DIM, "one-dimensional"),
        (lambda: [float("nan")] * E.EMBEDDING_DIM, "finite values"),
        (lambda: [0.1] * (E.EMBEDDING_DIM - 1), "expected"),
    ],
)
def test_reindex_rejects_invalid_individual_embedding_vectors(
    tmp_path, monkeypatch, invalid_vector_factory, reason
):
    """Each vector must be numeric, 1-D, finite, and at the active dimension."""
    beam = _reindex_fixture_beam(tmp_path, working=1)
    invalid_vector = invalid_vector_factory()
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "embed", lambda _contents: [invalid_vector])

    with pytest.raises(
        RuntimeError, match=rf"working_memory embedding vector 0.*{reason}"
    ):
        reindex_vectors(beam.conn)


def test_reindex_rejects_numeric_string_embedding_vector(tmp_path, monkeypatch):
    """Numeric-looking strings must not be persisted as JSON string vectors."""
    beam = _reindex_fixture_beam(tmp_path, working=1)
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "embed", lambda _contents: [["0.1"] * E.EMBEDDING_DIM])

    with pytest.raises(RuntimeError, match="working_memory embedding vector 0.*convertible numeric"):
        reindex_vectors(beam.conn)


def test_reindex_requires_episodic_vector_backend_before_writes(tmp_path, monkeypatch):
    """Episodic rows cannot be counted as reindexed without a writable backend."""
    beam = _reindex_fixture_beam(tmp_path, episodic=1)
    embed_calls = []
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "embed", lambda contents: embed_calls.append(contents))
    monkeypatch.setattr("mnemosyne.core.beam._vec_available", lambda _conn: False)
    monkeypatch.setattr("mnemosyne.core.beam._mib", None)

    with pytest.raises(RuntimeError, match="episodic_memory vector backend unavailable"):
        reindex_vectors(beam.conn)

    assert embed_calls == []


def test_reindex_rejects_invalid_later_episodic_batch(tmp_path, monkeypatch):
    """A later invalid episodic batch must fail before it is counted as written."""
    beam = _reindex_fixture_beam(tmp_path, episodic=2)
    responses = iter([
        [[0.1] * E.EMBEDDING_DIM],
        [[float("nan")] * E.EMBEDDING_DIM],
    ])
    calls = []
    monkeypatch.setattr(E, "available", lambda: True)

    def embed(contents):
        calls.append(contents)
        return next(responses)

    monkeypatch.setattr(E, "embed", embed)
    monkeypatch.setattr("mnemosyne.core.beam.np", _NumpyStub())
    monkeypatch.setattr("mnemosyne.core.beam._vec_available", lambda _conn: False)
    monkeypatch.setattr("mnemosyne.core.beam._mib", lambda _array: b"vector")

    with pytest.raises(RuntimeError, match="episodic_memory embedding vector 0.*finite values"):
        reindex_vectors(beam.conn, batch_size=1)

    assert len(calls) == 2


def test_reindex_does_not_mask_episodic_binary_vector_write_failure(tmp_path, monkeypatch):
    """A binary-vector update failure after a destructive rebuild must fail loudly."""
    beam = _reindex_fixture_beam(tmp_path, episodic=1)
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "embed", lambda contents: [[0.1] * E.EMBEDDING_DIM for _ in contents])
    monkeypatch.setattr("mnemosyne.core.beam.np", _NumpyStub())
    monkeypatch.setattr("mnemosyne.core.beam._mib", lambda _array: b"vector")
    beam.conn.execute(
        "CREATE TRIGGER fail_binary_vector BEFORE UPDATE OF binary_vector ON episodic_memory "
        "BEGIN SELECT RAISE(ABORT, 'binary vector write failed'); END"
    )

    with pytest.raises(sqlite3.IntegrityError, match="binary vector write failed"):
        reindex_vectors(beam.conn)


def test_reindex_does_not_mask_working_vec_write_failure(tmp_path, monkeypatch):
    """A strict vec_working write failure during reindex must fail loudly."""
    beam = _reindex_fixture_beam(tmp_path, working=1)
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "embed", lambda contents: [[0.1] * E.EMBEDDING_DIM for _ in contents])
    monkeypatch.setattr("mnemosyne.core.beam.np", _NumpyStub())
    monkeypatch.setattr("mnemosyne.core.beam._vec_available", lambda _conn: False)
    monkeypatch.setattr("mnemosyne.core.beam._wm_vec_available", lambda _conn: True)

    def fail_vec_working(_conn, table, _rowid, _embedding, **_kwargs):
        assert table == "vec_working"
        raise sqlite3.IntegrityError("vec_working write failed")

    monkeypatch.setattr("mnemosyne.core.beam._vec_table_insert", fail_vec_working)

    with pytest.raises(sqlite3.IntegrityError, match="vec_working write failed"):
        reindex_vectors(beam.conn)


def test_reindex_real_commits_deferred_working_batches_before_next_embedding(tmp_path, monkeypatch):
    """Deferred BEAM writes must be committed before the next embedding request."""
    beam = _reindex_fixture_beam(tmp_path, working=2)
    observed_transactions = []
    real_commits = []
    original_real_commit = type(beam.conn)._real_commit

    def record_real_commit(conn):
        real_commits.append(conn.in_transaction)
        return original_real_commit(conn)

    def embed(contents):
        observed_transactions.append(beam.conn.in_transaction)
        return [[0.1] * E.EMBEDDING_DIM for _ in contents]

    monkeypatch.setattr(type(beam.conn), "_real_commit", record_real_commit)
    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "embed", embed)
    monkeypatch.setattr("mnemosyne.core.beam.np", _NumpyStub())
    monkeypatch.setattr("mnemosyne.core.beam._vec_available", lambda _conn: False)
    monkeypatch.setattr("mnemosyne.core.beam._wm_vec_available", lambda _conn: False)
    beam.conn._defer_commit = True

    result = reindex_vectors(beam.conn, batch_size=1)

    assert result["working_memory_reindexed"] == 2
    assert observed_transactions == [False, False]
    assert real_commits == [True, True]
    assert not beam.conn.in_transaction


def test_reindex_success_rebuilds_exact_source_vectors_and_recalls_target(tmp_path, monkeypatch):
    """A deterministic core rebuild maps every derived vector back to its source row."""
    import json
    import numpy as np
    import mnemosyne.core.beam as beam_module

    beam = BeamMemory(session_id="reindex-success", db_path=str(tmp_path / "memory.db"))
    conn = beam.conn
    if not beam_module._vec_available(conn) or not beam_module._wm_vec_available(conn):
        pytest.skip("sqlite-vec vec_episodes and vec_working tables unavailable in this build")
    if beam_module._mib is None:
        pytest.skip("binary episodic vector writer unavailable in this build")

    working_sources = {
        "wm-orchid": "working source orchid signal",
        "wm-copper": "working source copper signal",
    }
    episodic_sources = {
        "ep-harbor": "episodic source harbor signal",
        "ep-forest": "episodic source forest signal",
    }
    source_vectors = {}
    for position, text in enumerate((*working_sources.values(), *episodic_sources.values())):
        vector = [-1.0] * E.EMBEDDING_DIM
        vector[position] = 1.0
        source_vectors[text] = vector
    target_id, target_content = next(iter(episodic_sources.items()))
    probe = "unrelated recall probe"

    for memory_id, content in working_sources.items():
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (memory_id, content, "test", "2026-01-01T00:00:00", "reindex-success"),
        )
        conn.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
            (memory_id, "[99.0]", "stale-model"),
        )
    for memory_id, content in episodic_sources.items():
        conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (memory_id, content, "test", "2026-01-01T00:00:00", "reindex-success", 0.5),
        )
    conn.commit()

    monkeypatch.setattr(E, "available", lambda: True)
    monkeypatch.setattr(E, "embed", lambda contents: [source_vectors[text] for text in contents])
    monkeypatch.setattr(
        E,
        "embed_query",
        lambda text: np.asarray(source_vectors[target_content if text == probe else text], dtype=np.float32),
    )

    result = reindex_vectors(conn, batch_size=1)

    assert result["status"] == "reindexed"
    assert result["working_memory_reindexed"] == len(working_sources)
    assert result["episodic_memory_reindexed"] == len(episodic_sources)

    embedding_rows = {
        row["memory_id"]: json.loads(row["embedding_json"])
        for row in conn.execute(
            "SELECT memory_id, embedding_json FROM memory_embeddings ORDER BY memory_id"
        )
    }
    assert embedding_rows == {
        memory_id: source_vectors[content] for memory_id, content in working_sources.items()
    }

    working_rowids = dict(conn.execute("SELECT id, rowid FROM working_memory"))
    episodic_rowids = dict(conn.execute("SELECT id, rowid FROM episodic_memory"))
    assert {row[0] for row in conn.execute("SELECT rowid FROM vec_working")} == set(working_rowids.values())
    assert {row[0] for row in conn.execute("SELECT rowid FROM vec_episodes")} == set(episodic_rowids.values())

    for memory_id, content in working_sources.items():
        matches = beam_module._wm_vec_search_sqlite(
            conn, np.asarray(source_vectors[content], dtype=np.float32), k=1, where_sql="1=1"
        )
        assert matches[0]["id"] == memory_id
    for memory_id, content in episodic_sources.items():
        matches = beam_module._vec_search(conn, source_vectors[content], k=1)
        assert matches[0]["rowid"] == episodic_rowids[memory_id]

    binary_rows = dict(conn.execute("SELECT id, binary_vector FROM episodic_memory"))
    assert binary_rows == {
        memory_id: beam_module._mib(np.asarray(source_vectors[content]))
        for memory_id, content in episodic_sources.items()
    }

    recalled = beam.recall(probe, top_k=5)
    assert recalled[0]["id"] == target_id


def test_reindex_rebuilds_all_vector_stores_at_active_dim():
    if not E.available():
        import pytest  # type: ignore
        pytest.skip("embedding model unavailable")

    with tempfile.TemporaryDirectory() as tmp:
        beam = BeamMemory(session_id="t", db_path=str(Path(tmp) / "m.db"))
        conn = beam.conn

        # working memory via the public store path (populates working_memory,
        # memory_embeddings, and vec_working).
        for text in ("the cat sat on the mat",
                     "python is a programming language",
                     "paris is the capital of france"):
            beam.remember(text)

        # one episodic row with content for reindex to re-embed.
        conn.execute(
            "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("ep1", "a long sunny day at the beach with friends", "test",
             "2026-01-01T00:00:00", "t", 0.8),
        )
        conn.commit()

        dim = int(E.EMBEDDING_DIM)
        vt = _effective_vec_type(conn)
        wrong = 384 if dim != 384 else 256

        # simulate the stale-dimension state after a model swap.
        for table in ("vec_episodes", "vec_working", "vec_facts"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(f"CREATE VIRTUAL TABLE {table} USING vec0(embedding {vt}[{wrong}])")
        conn.commit()

        # dry-run: reports the plan, writes nothing.
        plan = reindex_vectors(conn, dry_run=True)
        assert plan["dim"] == dim
        assert plan["working_memory"] >= 3
        assert plan["episodic_memory"] >= 1
        assert f"[{wrong}]" in _ddl(conn, "vec_episodes")  # unchanged by dry-run

        # real reindex.
        result = reindex_vectors(conn)
        assert result["status"] == "reindexed"
        assert result["working_memory_reindexed"] >= 3
        assert result["episodic_memory_reindexed"] >= 1

        # vec tables recreated at the active dim and repopulated.
        for table in ("vec_episodes", "vec_working", "vec_facts"):
            assert f"[{dim}]" in _ddl(conn, table), (table, _ddl(conn, table))
        assert conn.execute("SELECT COUNT(*) FROM vec_working").fetchone()[0] >= 3
        assert conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0] >= 1

        # episodic binary_vector refreshed.
        assert conn.execute(
            "SELECT COUNT(*) FROM episodic_memory WHERE binary_vector IS NOT NULL"
        ).fetchone()[0] >= 1

        # recall works (no dimension error) and the model/dim matches the query path.
        results = beam.recall("programming language", top_k=5)
        assert isinstance(results, list)


if __name__ == "__main__":  # allow direct execution without pytest
    test_reindex_rebuilds_all_vector_stores_at_active_dim()
    print("ok")
