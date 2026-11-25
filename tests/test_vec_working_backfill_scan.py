"""
The working-memory vector backfill and its coverage counter must find the
same gap without joining the ``vec_working`` virtual table.

``LEFT JOIN vec_working ... WHERE vw.rowid IS NULL`` (and the equivalent
correlated ``NOT EXISTS``) is planned as a full scan of the vec0 table, which
costs seconds on a multi-thousand-row bank even when nothing needs inserting.
The materialized ``NOT IN (SELECT rowid FROM vec_working)`` form returns the
same rows in milliseconds.

These tests pin the functional contract of that rewrite: the same gap is found,
a vector already stored in ``vec_working`` keeps its exact bytes (the backfill
only inserts, it never rewrites), an orphan vector row neither suppresses a
missing row nor gets deleted, an empty vector table is filled completely, and
repeated runs are no-ops.
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


def _unit_embedding(axis: int = 0):
    """A distinct unit vector per ``axis`` so two embeddings never collide."""
    np = pytest.importorskip("numpy")
    vector = [0.0] * beam_module.EMBEDDING_DIM
    vector[axis] = 1.0
    return np.array(vector, dtype=np.float32)


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


def _stored_vector(conn, rowid):
    """Exact bytes currently stored in ``vec_working`` for ``rowid``."""
    row = conn.execute(
        "SELECT embedding FROM vec_working WHERE rowid = ?", (rowid,)
    ).fetchone()
    return None if row is None else row[0]


def test_backfill_preserves_stored_bytes_and_keeps_orphans(temp_db):
    beam = BeamMemory(session_id="vec-backfill-preserve", db_path=temp_db)
    _require_vec_working(beam.conn)

    compat = _unit_embedding(0)       # compatibility embedding in memory_embeddings
    already_stored = _unit_embedding(1)  # deliberately different from compat
    orphan_vector = _unit_embedding(2)
    orphan_rowid = 987_654  # no working_memory row behind it

    kept_rowid = _insert_working(beam, "already-indexed", compat)
    missing_rowid = _insert_working(beam, "needs-backfill", compat)
    beam_module._vec_table_insert(beam.conn, "vec_working", int(kept_rowid), already_stored)
    beam_module._vec_table_insert(beam.conn, "vec_working", orphan_rowid, orphan_vector)
    beam.conn.commit()

    coverage = beam_module.vec_working_coverage(beam.conn)
    assert coverage["missing_vec_working_rows"] == 1
    assert coverage["orphan_vec_working_rows"] == 1

    kept_before = _stored_vector(beam.conn, kept_rowid)
    orphan_before = _stored_vector(beam.conn, orphan_rowid)
    assert kept_before is not None and orphan_before is not None
    assert kept_before != orphan_before

    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 1
    beam.conn.commit()

    # The stored vector was not rewritten, and the orphan survived untouched.
    assert _stored_vector(beam.conn, kept_rowid) == kept_before
    assert _stored_vector(beam.conn, orphan_rowid) == orphan_before

    # The gap was filled from the compatibility embedding, not from the
    # already-stored vector: the two differ, so this also proves the backfill
    # did not simply copy the neighbouring row.
    filled = _stored_vector(beam.conn, missing_rowid)
    assert filled is not None
    assert filled != kept_before

    after = beam_module.vec_working_coverage(beam.conn)
    assert after["missing_vec_working_rows"] == 0
    assert after["orphan_vec_working_rows"] == 1

    # Second run is a no-op and rewrites nothing.
    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 0
    assert _stored_vector(beam.conn, kept_rowid) == kept_before
    assert _stored_vector(beam.conn, missing_rowid) == filled


def test_backfill_fills_an_empty_vector_table(temp_db):
    beam = BeamMemory(session_id="vec-backfill-empty", db_path=temp_db)
    _require_vec_working(beam.conn)

    assert beam_module.vec_working_coverage(beam.conn)["status"] == "empty"

    rowids = [_insert_working(beam, f"wm-{axis}", _unit_embedding(axis)) for axis in range(3)]

    coverage = beam_module.vec_working_coverage(beam.conn)
    assert coverage["working_embedding_rows"] == 3
    assert coverage["vec_working_rows"] == 0
    assert coverage["missing_vec_working_rows"] == 3
    assert coverage["status"] == "partial"

    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 3
    beam.conn.commit()

    assert beam_module._backfill_vec_working_from_memory_embeddings(beam.conn) == 0
    after = beam_module.vec_working_coverage(beam.conn)
    assert after["vec_working_rows"] == 3
    assert after["missing_vec_working_rows"] == 0
    assert after["status"] == "complete"
    for rowid in rowids:
        assert _stored_vector(beam.conn, rowid) is not None
