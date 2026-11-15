"""find_related_memories' typed walk must not scan edges unrelated to the node.

Nearly every edge shares one type, so idx_edges_type selects the whole table.
Invariant: a typed walk returns what a scan in insertion order would, and its
cost does not grow with edges that don't touch the walked nodes.
"""
import random

import pytest

from mnemosyne.core.episodic_graph import EpisodicGraph


@pytest.fixture
def graph(tmp_path):
    return EpisodicGraph(db_path=tmp_path / "graph.db")


def _add(graph, edges):
    graph.conn.executemany(
        "INSERT INTO graph_edges (source, target, edge_type, weight, timestamp) "
        "VALUES (?, ?, ?, ?, 't')", edges)
    graph.conn.commit()


def _reference_walk(edges, start, depth, edge_type, min_weight):
    # The behavior being preserved, as plain Python over insertion order.
    results, level, seen = [], [start], {start}
    for hop in range(1, depth + 1):
        next_level = []
        for node in level:
            for source, target, kind, weight in edges:
                if node in (source, target) and kind == edge_type and weight >= min_weight:
                    neighbor = target if source == node else source
                    if neighbor not in seen:
                        seen.add(neighbor)
                        next_level.append(neighbor)
                        results.append({"memory_id": neighbor, "edge_type": kind,
                                        "weight": weight, "depth": hop})
        level = next_level
    return results


def _walk_statement(graph, *args, **kwargs):
    statements = []
    graph.conn.set_trace_callback(statements.append)
    graph.find_related_memories(*args, **kwargs)
    graph.conn.set_trace_callback(None)
    return next(s for s in statements if "FROM graph_edges" in s)


def test_typed_walk_uses_the_endpoint_indexes(graph):
    _add(graph, [("a", "b", "ctx", 1.0)])

    sql = _walk_statement(graph, "a", depth=1, edge_type="ctx", min_weight=0.3)

    plan = " ".join(row[3] for row in graph.conn.execute("EXPLAIN QUERY PLAN " + sql))
    assert "idx_edges_source" in plan and "idx_edges_target" in plan
    assert "idx_edges_type" not in plan


def test_typed_walk_cost_ignores_unrelated_edges(graph):
    _add(graph, [("a", "b", "ctx", 0.9), ("b", "c", "ctx", 0.9)])

    def add_unrelated(start, stop):
        _add(graph, ((f"gist_{i}", f"fact_{i}_0", "ctx", 1.0) for i in range(start, stop)))

    def vm_steps_for_a_walk():
        steps = [0]

        def count():
            steps[0] += 1
            return 0

        graph.conn.set_progress_handler(count, 10)
        related = graph.find_related_memories("a", depth=2, edge_type="ctx", min_weight=0.3)
        graph.conn.set_progress_handler(None, 10)
        assert [r["memory_id"] for r in related] == ["b", "c"]
        return steps[0]

    add_unrelated(0, 1000)
    at_1k = vm_steps_for_a_walk()
    add_unrelated(1000, 10000)
    at_10k = vm_steps_for_a_walk()

    # Ten times the edges; a little index depth is fine, a scan is ten times the work.
    assert at_10k <= at_1k * 2


def test_typed_walk_matches_a_scan_in_insertion_order(graph):
    nodes = [f"n{i}" for i in range(12)]
    for seed in range(25):
        rng = random.Random(seed)
        edges = [(rng.choice(nodes), rng.choice(nodes), rng.choice(["ctx", "ctx", "ctx", "rel", "CTX"]),
                  rng.choice([0.1, 0.3, 0.5, 0.9])) for _ in range(40)]
        edges = [e for e in edges if e[0] != e[1]]
        graph.conn.execute("DELETE FROM graph_edges")
        _add(graph, edges)
        start, depth, min_weight = rng.choice(nodes), rng.choice([1, 2, 3]), rng.choice([0.0, 0.3, 0.6])

        got = graph.find_related_memories(start, depth=depth, edge_type="ctx", min_weight=min_weight)

        assert got == _reference_walk(edges, start, depth, "ctx", min_weight), f"seed {seed}"


def test_non_string_edge_type_still_matches(graph):
    # The tool handler passes edge_type through uncoerced.
    _add(graph, [("a", "b", "5", 1.0), ("a", "near-miss", "05", 1.0)])

    assert [r["memory_id"] for r in graph.find_related_memories("a", depth=1, edge_type=5)] == ["b"]


def test_shared_neighbour_is_claimed_by_the_earliest_edge(graph):
    # Insertion order decides which edge's weight a shared neighbour reports,
    # including when the later edge is an outgoing one the source index reads first.
    _add(graph, [("b", "a", "ctx", 0.8), ("c", "shared", "ctx", 0.5), ("a", "c", "ctx", 0.9),
                 ("b", "shared", "ctx", 0.4), ("a", "d", "ctx", 0.7)])

    related = graph.find_related_memories("a", depth=2, edge_type="ctx")

    assert [(r["memory_id"], r["depth"], r["weight"]) for r in related] == [
        ("b", 1, 0.8), ("c", 1, 0.9), ("d", 1, 0.7), ("shared", 2, 0.4)]
