"""
The working-memory vector backfill and its coverage counter must find the
same gap without joining the ``vec_working`` virtual table.

``LEFT JOIN vec_working ... WHERE vw.rowid IS NULL`` (and the equivalent
correlated ``NOT EXISTS``) is planned as a full scan of the vec0 table, which
costs seconds on a multi-thousand-row bank even when nothing needs inserting.
The materialized ``NOT IN (SELECT rowid FROM vec_working)`` form returns the
same rows in milliseconds. These tests pin the functional contract of that
rewrite: same rows found, existing vectors preserved, repeated runs idempotent,
and the coverage counter agreeing with the backfill.
"""

from datetime import datetime
from pathlib import Path
import tempfile

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _require_vec_working(conn):
    pytest.importorskip("sqlite_vec")
    if not beam_module._wm_vec_available(conn):
        pytest.skip("sqlite-vec vec_working table unavailable")


def _unit_embedding():
    np = pytest.importorskip("numpy")
    return np.array([1.0] + [0.0] * (beam_module.EMBEDDING_DIM - 1), dtype=np.float32)


def _insert_working(beam, memory_id, embedding):
    """Insert one working memory plus its fallback embedding row; return its rowid."""
    beam.conn.execute(
        """
        INSERT INTO working_memory
            (id, content, source, timestamp, session_id, scope, importance)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (memory_id, f"content {memory_id}", "test", datetime.now().isoformat(),
         "vec-backfill-scan", "session", 0.5),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) VALUES (?, ?, ?)",
        (memory_id, beam_module._embeddings.serialize(embedding), "test"),
    )
    beam.conn.commit()
    row = beam.conn.execute(
        "SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()
    return row["rowid"]


def test_backfill_finds_only_missing_rows_and_preserves_existing_vectors(temp_db):
    beam = BeamMemory(session_id="vec-backfill-scan", db_path=temp_db)
    _require_vec_working(beam.conn)
    embedding = _unit_embedding()

    kept_rowid = _insert_working(beam, "already-indexed", embedding)
    missing_rowid = _insert_working(beam, "needs-backfill", embedding)
    beam_module._vec_table_insert(beam.conn, "vec_working", int(kept_rowid), embedding)
    beam.conn.commit()

    coverage = beam_module.vec_working_coverage(beam.conn)
    assert coverage["missing_vec_working_rows"] == 1
    assert coverage["orphan_vec_working_rows"] == 0

    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 1
    beam.conn.commit()

    for rowid in (kept_rowid, missing_rowid):
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)
        ).fetchone()[0] == 1

    # Second run is a no-op, and the coverage counter now agrees with it.
    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 0
    assert beam_module.vec_working_coverage(beam.conn)["missing_vec_working_rows"] == 0


def test_coverage_counts_every_unmirrored_embedding(temp_db):
    beam = BeamMemory(session_id="vec-backfill-coverage", db_path=temp_db)
    _require_vec_working(beam.conn)
    embedding = _unit_embedding()

    assert beam_module.vec_working_coverage(beam.conn)["status"] == "empty"

    rowids = [_insert_working(beam, f"wm-{index}", embedding) for index in range(3)]
    beam_module._vec_table_insert(beam.conn, "vec_working", int(rowids[0]), embedding)
    beam.conn.commit()

    coverage = beam_module.vec_working_coverage(beam.conn)
    assert coverage["working_embedding_rows"] == 3
    assert coverage["vec_working_rows"] == 1
    assert coverage["missing_vec_working_rows"] == 2
    assert coverage["status"] == "partial"

    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 2
    beam.conn.commit()
    assert beam_module.vec_working_coverage(beam.conn)["status"] == "complete"
