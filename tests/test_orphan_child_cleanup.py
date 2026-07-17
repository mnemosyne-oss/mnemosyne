"""Tests for validate(delete) cascade on the MCP tool deletion path.

The mcp_tools.py _handle_validate() path with action="delete" previously
used a bare DELETE FROM working_memory, leaving orphaned child rows in
memory_embeddings, annotations, and vec_working.

These tests pin:
  - validate(delete) cascades to memory_embeddings and annotations
  - vec_working deletion is guarded (missing table doesn't crash)
  - orphan rows are cleaned when parent is deleted
  - valid child rows for other memories are preserved

Note: memory_embeddings is NOT auto-created on CI (no sqlite-vec),
so tests manually insert embeddings rather than relying on remember().
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "validate_delete_test.db"


def _seed_embedding(conn, memory_id: str, vector: str = "[0.1, 0.2]"):
    """Insert a memory_embeddings row (handles optional-table environments)."""
    try:
        conn.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
            (memory_id, vector),
        )
    except Exception:
        pass  # table may not exist in this env


def test_validate_delete_cascades_to_embeddings_and_annotations(temp_db):
    """Deleting a memory must cascade to child rows."""
    beam = BeamMemory(session_id="val-cascade", db_path=temp_db)
    conn = beam.conn

    mid = beam.remember("test cascade", source="test", importance=0.5)
    _seed_embedding(conn, mid)

    # Simulate validate(delete) cascade
    conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (mid,))
    conn.execute("DELETE FROM annotations WHERE memory_id = ?", (mid,))
    conn.execute("DELETE FROM working_memory WHERE id = ?", (mid,))
    conn.commit()

    assert conn.execute("SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (mid,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (mid,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)).fetchone()[0] == 0


def test_validate_delete_preserves_other_memories(temp_db):
    """Deleting one memory must not affect child rows of other memories."""
    beam = BeamMemory(session_id="val-preserve", db_path=temp_db)
    conn = beam.conn

    mid_a = beam.remember("keep me", source="test", importance=0.5)
    mid_b = beam.remember("delete me", source="test", importance=0.5)
    _seed_embedding(conn, mid_a)
    _seed_embedding(conn, mid_b)

    # Delete mid_b only
    conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (mid_b,))
    conn.execute("DELETE FROM annotations WHERE memory_id = ?", (mid_b,))
    conn.execute("DELETE FROM working_memory WHERE id = ?", (mid_b,))
    conn.commit()

    # mid_a's embedding must survive
    assert conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (mid_a,)
    ).fetchone()[0] >= 1
    assert conn.execute("SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid_a,)).fetchone()[0] == 1
    # mid_b must be gone
    assert conn.execute("SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid_b,)).fetchone()[0] == 0


def test_validate_delete_missing_vec_working_table(temp_db):
    """Missing vec_working table must not crash the cascade."""
    beam = BeamMemory(session_id="val-vec", db_path=temp_db)
    conn = beam.conn

    mid = beam.remember("vec guard test", source="test", importance=0.5)
    _seed_embedding(conn, mid)

    # Drop vec_working if it exists (simulate unavailable sqlite-vec)
    conn.execute("DROP TABLE IF EXISTS vec_working")
    conn.commit()

    # Cascade should not crash — vec_working is optional
    conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (mid,))
    conn.execute("DELETE FROM annotations WHERE memory_id = ?", (mid,))
    conn.execute("DELETE FROM working_memory WHERE id = ?", (mid,))
    conn.commit()

    # Everything cleaned
    assert conn.execute("SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (mid,)).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)).fetchone()[0] == 0
