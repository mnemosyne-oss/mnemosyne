"""Graph voice retains the shortest eligible route to each memory (#1303)."""

from datetime import datetime, timezone

import pytest

from mnemosyne.core.episodic_graph import EpisodicGraph, Gist, GraphEdge
from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine


@pytest.mark.parametrize("query", ["Alice then Bob", "Bob then Alice"])
def test_graph_voice_keeps_shorter_path_across_seeds(tmp_path, query):
    db = tmp_path / "graph.db"
    graph = EpisodicGraph(db_path=db)
    ts = datetime.now(timezone.utc).isoformat()
    for name in ("Alice", "Bob"):
        graph.store_gist(
            Gist(f"gist_{name}", f"{name} was here", ts, [name], None, None, None),
            name,
        )
    for source, target in (("Alice", "middle"), ("middle", "target"),
                           ("Bob", "target"), ("Alice", "Bob")):
        graph.add_edge(GraphEdge(source, target, "ctx", 0.9, ts))

    engine = PolyphonicRecallEngine(db_path=db, conn=graph.conn)
    results = engine._graph_voice(query)
    target = [r for r in results if r.memory_id == "target"]
    assert len(target) == 1
    assert target[0].score == 0.4
    assert target[0].metadata["depth"] == 1
    assert target[0].metadata["seed"] == "Bob"
    assert not [r for r in results if r.voice == "graph_traversal" and
                r.memory_id in {"Alice", "Bob"}]
    graph.close()
