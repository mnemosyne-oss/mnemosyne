"""Proactive linking's duplicate check must not read every `references` edge.

`references` can become the most common edge type, so idx_edges_type selects
most of the table for a lookup that already has both endpoints.
"""
import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def beam(tmp_path, monkeypatch):
    """Keep linker writes local and independent of deployment filters/backends."""
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("MNEMOSYNE_IGNORE_PATTERNS", raising=False)
    monkeypatch.delenv("MNEMOSYNE_WRITE_CLASSIFIER", raising=False)
    for name in ("MNEMOSYNE_NO_EMBEDDINGS", "MNEMOSYNE_SKIP_EMBEDDINGS", "MNEMOSYNE_EMBEDDINGS_OFF"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("MNEMOSYNE_PROACTIVE_LINKING", "1")
    return BeamMemory(session_id="s", db_path=tmp_path / "memory.db")


def _references(beam):
    """Return reference-edge endpoints in stable insertion order."""
    return [tuple(r) for r in beam.conn.execute(
        "SELECT source, target FROM graph_edges WHERE edge_type = 'references' ORDER BY id")]


def test_duplicate_check_searches_the_new_memorys_edges(beam):
    """Require the duplicate lookup to use the new memory's source index."""
    beam.remember("Alice met Bob in Paris.", source="user", extract_entities=True)
    statements = []
    beam.conn.set_trace_callback(statements.append)
    beam.remember("Alice called Bob from Paris.", source="user", extract_entities=True)
    beam.conn.set_trace_callback(None)

    checks = [s for s in statements if "FROM graph_edges" in s and "references" in s]
    assert checks
    plan = " ".join(row[3] for row in beam.conn.execute("EXPLAIN QUERY PLAN " + checks[0]))
    # The target can be a hub, since many memories link to the same few. The
    # source is the new memory, which has only the edges this call is adding.
    assert "SEARCH graph_edges USING INDEX idx_edges_source (source=?)" in plan


def test_check_still_finds_an_existing_references_edge(beam):
    """Re-running the linker must preserve an existing reference edge."""
    # Calling the linker again for the same memory must not add a second edge.
    beam.remember("Alice met Bob in Paris.", source="user", extract_entities=True)
    second = beam.remember("Alice called Bob from Paris.", source="user", extract_entities=True)
    before = _references(beam)
    assert before

    beam._proactively_link(second, "Alice called Bob from Paris.")

    assert _references(beam) == before


def test_check_ignores_other_edge_types_between_the_same_memories(beam):
    """An edge of another type must not suppress a reference between the same pair."""
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


def test_edge_to_one_target_does_not_stop_a_link_to_another(beam):
    """Restore a missing target edge without duplicating a different target edge."""
    # The check must compare the target too, not only look for any edge from the source.
    first = beam.remember("Alice met Bob in Paris.", source="user", extract_entities=True)
    second = beam.remember("Alice saw Bob in Paris.", source="user", extract_entities=True)
    third = beam.remember("Alice called Bob from Paris.", source="user", extract_entities=True)
    assert {(third, first), (third, second)} <= set(_references(beam))
    beam.conn.execute("DELETE FROM graph_edges WHERE source = ? AND target = ?", (third, second))
    beam.conn.commit()

    beam._proactively_link(third, "Alice called Bob from Paris.")

    edges = _references(beam)
    assert edges.count((third, second)) == 1
    assert edges.count((third, first)) == 1
