"""Graph voice retains the shortest eligible route to each memory (#1303)."""

import sqlite3
from datetime import datetime, timezone

import pytest

from mnemosyne.core.beam import _polyphonic_scope_where
from mnemosyne.core.episodic_graph import EpisodicGraph, Fact, Gist, GraphEdge
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


@pytest.mark.parametrize("query", ["Alice then Bob then Ghost", "Bob then Alice then Ghost"])
@pytest.mark.parametrize("seed_kind", ["gist", "fact"])
@pytest.mark.parametrize("tier", ["working", "episodic"])
def test_real_recall_excludes_foreign_and_orphan_seeds_before_traversal(
    tmp_path, query, seed_kind, tier
):
    # Explicit raw-ID edges exercise seed eligibility, not ingestion's
    # gist_/fact_ node-ID format (which the current voice does not traverse).
    graph = EpisodicGraph(db_path=tmp_path / "recall.db")
    ts = datetime.now(timezone.utc).isoformat()
    for table in ("working_memory", "episodic_memory"):
        graph.conn.execute(
            f"CREATE TABLE {table} (id TEXT PRIMARY KEY, content TEXT, "
            "session_id TEXT, scope TEXT)"
        )
    for mid, session in (("Alice", "a"), ("Bob", "b"),
                         ("middle", "a"), ("target", "a")):
        graph.conn.execute(
            f"INSERT INTO {tier}_memory (id, content, session_id, scope) "
            "VALUES (?, ?, ?, 'session')",
            (mid, f"content for {mid}", session),
        )
    graph.conn.commit()
    for name in ("Alice", "Bob", "Ghost"):
        if seed_kind == "gist":
            graph.store_gist(
                Gist(f"gist_{name}", f"{name} gist", ts, [name], None, None, None),
                name,
            )
        else:
            graph.store_fact(
                Fact(f"fact_seed_{name}", name, "uses", "Python", ts, 0.9), name
            )
    for source, target, edge_type, weight in (
        ("Alice", "middle", "ctx", 0.9),
        ("middle", "target", "ctx", 0.8),
        ("Bob", "target", "ctx", 0.9),
        ("Ghost", "target", "ctx", 0.95),
        ("Alice", "target", "rel", 1.0),
        ("Alice", "target", "ctx", 0.2),
    ):
        graph.add_edge(GraphEdge(source, target, edge_type, weight, ts))
    engine = PolyphonicRecallEngine(db_path=graph.db_path, conn=graph.conn)
    scope_where, scope_params = _polyphonic_scope_where(
        session_id="a", cross_session=False
    )
    try:
        # A real engine call combines voices, prunes scope, hydrates content,
        # reranks, and assembles context; a foreign seed must not set metadata.
        scoped = engine.recall(
            query, scope_where=scope_where, scope_params=scope_params
        )
        target = next(r for r in scoped if r.memory_id == "target")
        assert target.metadata["seed"] == "Alice"
        assert target.metadata["depth"] == 2
        assert target.metadata["edge_type"] == "ctx"
        assert target.metadata["weight"] == 0.8
        assert target.content == "content for target"
        assert target.voice_scores["graph_traversal"] > 0
        assert target.combined_score == pytest.approx(1 / 63)
        assert not {"Bob", "Ghost"} & {r.memory_id for r in scoped}

        graph_target = next(r for r in engine._graph_voice(
            query, scope_where=scope_where, scope_params=scope_params
        ) if r.memory_id == "target")
        assert graph_target.score == 0.2
        assert graph_target.metadata["seed"] == "Alice"

        # An unscoped caller still selects the stronger Bob path.
        unscoped = engine.recall(query)
        target = next(r for r in unscoped if r.memory_id == "target")
        assert target.metadata["seed"] == "Bob"
        assert target.metadata["depth"] == 1
    finally:
        graph.close()


@pytest.mark.parametrize("seed_kind", ["gist", "fact"])
def test_scoped_graph_seed_with_no_backing_row_or_missing_tier(tmp_path, seed_kind):
    graph = EpisodicGraph(db_path=tmp_path / "one-tier.db")
    graph.conn.execute(
        "CREATE TABLE working_memory (id TEXT PRIMARY KEY, content TEXT, "
        "session_id TEXT, scope TEXT)"
    )
    graph.conn.executemany(
        "INSERT INTO working_memory VALUES (?, ?, 'a', 'session')",
        [("Alice", "Alice content"), ("target", "target content")],
    )
    ts = datetime.now(timezone.utc).isoformat()
    for name in ("Alice", "Ghost"):
        if seed_kind == "gist":
            graph.store_gist(
                Gist(f"gist_{name}", name, ts, [name], None, None, None), name
            )
        else:
            graph.store_fact(
                Fact(f"fact_seed_{name}", name, "uses", "Python", ts, 0.9), name
            )
        graph.add_edge(GraphEdge(name, "target", "ctx", 0.9, ts))
    engine = PolyphonicRecallEngine(db_path=graph.db_path, conn=graph.conn)
    scope_where, scope_params = _polyphonic_scope_where(
        session_id="a", cross_session=False
    )
    try:
        results = engine.recall(
            "Ghost then Alice", scope_where=scope_where, scope_params=scope_params
        )
        target = next(r for r in results if r.memory_id == "target")
        assert target.metadata["seed"] == "Alice"
        assert target.metadata["depth"] == 1
        assert target.content == "target content"
        assert "Ghost" not in {r.memory_id for r in results}
    finally:
        graph.close()


@pytest.mark.parametrize("visible_tier", ["working", "episodic"])
def test_scoped_seed_kept_when_either_backing_tier_is_visible(tmp_path, visible_tier):
    graph = EpisodicGraph(db_path=tmp_path / "dual-tier.db")
    for tier in ("working", "episodic"):
        graph.conn.execute(
            f"CREATE TABLE {tier}_memory (id TEXT PRIMARY KEY, content TEXT, "
            "session_id TEXT, scope TEXT)"
        )
        graph.conn.execute(
            f"INSERT INTO {tier}_memory VALUES ('Alice', 'Alice content', ?, 'session')",
            ("a" if tier == visible_tier else "b",),
        )
    graph.conn.execute(
        "INSERT INTO working_memory VALUES ('target', 'target content', 'a', 'session')"
    )
    ts = datetime.now(timezone.utc).isoformat()
    graph.store_gist(Gist("gist_Alice", "Alice gist", ts, ["Alice"], None, None, None), "Alice")
    graph.add_edge(GraphEdge("Alice", "target", "ctx", 0.9, ts))
    engine = PolyphonicRecallEngine(db_path=graph.db_path, conn=graph.conn)
    scope_where, scope_params = _polyphonic_scope_where(session_id="a", cross_session=False)
    try:
        results = engine.recall(
            "Alice", scope_where=scope_where, scope_params=scope_params
        )
        assert next(r for r in results if r.memory_id == "target").metadata["seed"] == "Alice"
    finally:
        graph.close()


def test_scoped_seed_lookup_does_not_swallow_sql_errors(tmp_path):
    graph = EpisodicGraph(db_path=tmp_path / "sql-error.db")
    graph.conn.execute(
        "CREATE TABLE working_memory (id TEXT PRIMARY KEY, content TEXT, "
        "session_id TEXT, scope TEXT)"
    )
    graph.conn.execute(
        "INSERT INTO working_memory VALUES ('Alice', 'Alice content', 'a', 'session')"
    )
    graph.store_gist(Gist("gist_Alice", "Alice gist", datetime.now(timezone.utc).isoformat(),
                          ["Alice"], None, None, None), "Alice")
    engine = PolyphonicRecallEngine(db_path=graph.db_path, conn=graph.conn)
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such column"):
            engine.recall("Alice", scope_where="missing_scope_column = ?", scope_params=("a",))
    finally:
        graph.close()
