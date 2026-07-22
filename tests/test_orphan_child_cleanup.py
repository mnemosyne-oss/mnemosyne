"""Regression tests for MCP validate(delete) child cleanup and rollback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.mcp_tools import handle_tool_call


@pytest.fixture
def beam(tmp_path: Path) -> BeamMemory:
    return BeamMemory(session_id="validate-delete", db_path=tmp_path / "validate_delete_test.db")


def _seed_children(beam: BeamMemory, memory_id: str) -> None:
    """Add child records that the MCP delete handler must remove."""
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
        (memory_id, "[0.1, 0.2]"),
    )
    beam.conn.execute(
        "INSERT INTO annotations "
        "(memory_id, kind, value, source, confidence, created_at) "
        "VALUES (?, 'fact', 'test annotation', 'test', 1.0, CURRENT_TIMESTAMP)",
        (memory_id,),
    )
    beam.conn.commit()


def _validate_delete(beam: BeamMemory, memory_id: str) -> dict:
    """Dispatch the public MCP tool through its production handler."""
    memory = SimpleNamespace(beam=beam)
    with patch("mnemosyne.mcp_tools._create_instance", return_value=memory):
        return handle_tool_call(
            "mnemosyne_validate",
            {"memory_id": memory_id, "action": "delete"},
        )


def _count(beam: BeamMemory, table: str, memory_id: str) -> int:
    return beam.conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]


def test_mcp_validate_delete_cascades_only_target_children(beam: BeamMemory):
    """The public MCP dispatch deletes target children and preserves others."""
    keep_id = beam.remember("keep this memory", source="test", importance=0.5)
    delete_id = beam.remember("delete this memory", source="test", importance=0.5)
    _seed_children(beam, keep_id)
    _seed_children(beam, delete_id)
    keep_annotation_count = _count(beam, "annotations", keep_id)

    result = _validate_delete(beam, delete_id)

    assert result["status"] == "validation_delete"
    assert _count(beam, "memory_embeddings", delete_id) == 0
    assert _count(beam, "annotations", delete_id) == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (delete_id,)
    ).fetchone()[0] == 0
    assert _count(beam, "memory_embeddings", keep_id) == 1
    assert _count(beam, "annotations", keep_id) == keep_annotation_count
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (keep_id,)
    ).fetchone()[0] == 1


def test_mcp_validate_delete_rolls_back_on_child_failure(beam: BeamMemory):
    """A failed child delete leaves no partial cleanup or stale transaction."""
    memory_id = beam.remember("must survive failed delete", source="test", importance=0.5)
    _seed_children(beam, memory_id)
    annotation_count = _count(beam, "annotations", memory_id)
    beam.conn.execute(
        "CREATE TRIGGER fail_annotation_delete "
        "BEFORE DELETE ON annotations "
        "BEGIN SELECT RAISE(ABORT, 'forced annotation failure'); END"
    )
    beam.conn.commit()

    result = _validate_delete(beam, memory_id)

    assert result["error"] == "validation_failed"
    assert "forced annotation failure" in result["reason"]
    assert not beam.conn.in_transaction
    assert _count(beam, "memory_embeddings", memory_id) == 1
    assert _count(beam, "annotations", memory_id) == annotation_count
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 1
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_validations WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 0


def test_mcp_validate_delete_handles_missing_vec_working(beam: BeamMemory):
    """The production MCP dispatch tolerates an unavailable sqlite-vec table."""
    memory_id = beam.remember("delete without vec", source="test", importance=0.5)
    _seed_children(beam, memory_id)
    beam.conn.execute("DROP TABLE IF EXISTS vec_working")
    beam.conn.commit()

    result = _validate_delete(beam, memory_id)

    assert result["status"] == "validation_delete"
    assert _count(beam, "memory_embeddings", memory_id) == 0
    assert _count(beam, "annotations", memory_id) == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 0
