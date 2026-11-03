"""Proactive linking's duplicate check must not read every `references` edge.

`references` can become the most common edge type, so idx_edges_type selects
most of the table for a lookup that already has both endpoints.
"""
import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def beam(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PROACTIVE_LINKING", "1")
    return BeamMemory(session_id="s", db_path=tmp_path / "memory.db")


def _references(beam):
    return [tuple(r) for r in beam.conn.execute(
        "SELECT source, target FROM graph_edges WHERE edge_type = 'references' ORDER BY id")]


def test_duplicate_check_uses_the_endpoint_indexes(beam):
    beam.remember("Alice met Bob in Paris.", source="user", extract_entities=True)
    statements = []
    beam.conn.set_trace_callback(statements.append)
    beam.remember("Alice called Bob from Paris.", source="user", extract_entities=True)
    beam.conn.set_trace_callback(None)

    checks = [s for s in statements if "FROM graph_edges" in s and "references" in s]
    assert checks
    plan = " ".join(row[3] for row in beam.conn.execute("EXPLAIN QUERY PLAN " + checks[0]))
    assert "idx_edges_type" not in plan


def test_check_still_finds_an_existing_references_edge(beam):
    # Calling the linker again for the same memory must not add a second edge.
    beam.remember("Alice met Bob in Paris.", source="user", extract_entities=True)
    second = beam.remember("Alice called Bob from Paris.", source="user", extract_entities=True)
    before = _references(beam)
    assert before

    beam._proactively_link(second, "Alice called Bob from Paris.")

    assert _references(beam) == before


def test_check_ignores_other_edge_types_between_the_same_memories(beam):
    # The filter must still apply: a related_to edge is not a references edge.
    first = beam.remember("Alice met Bob in Paris.", source="user", extract_entities=True)
    second = beam.remember("Alice called Bob from Paris.", source="user", extract_entities=True)
    beam.conn.execute("DELETE FROM graph_edges WHERE source = ? AND target = ?", (second, first))
    beam.conn.execute(
        "INSERT INTO graph_edges (source, target, edge_type, weight, timestamp) "
        "VALUES (?, ?, 'other', 0.5, 't')", (second, first))
    beam.conn.commit()

    beam._proactively_link(second, "Alice called Bob from Paris.")

    assert (second, first) in _references(beam)
