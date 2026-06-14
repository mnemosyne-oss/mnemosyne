"""Tests for reindex_vectors() — rebuilding vector stores after a model/dim change.

Simulates the motivating bug (sqlite-vec tables stuck at the old dimension after an
embedding-model swap) by recreating the vec0 tables at a wrong dimension, then
verifies reindex_vectors() recreates them at the active dimension, repopulates
working + episodic vectors, refreshes the episodic binary_vector, and leaves recall
working. Also checks that --dry-run writes nothing.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from mnemosyne.core.beam import BeamMemory, reindex_vectors, _effective_vec_type
import mnemosyne.core.embeddings as E


def _ddl(conn, table):
    row = conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (table,)).fetchone()
    return row[0] if row and row[0] else ""


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
