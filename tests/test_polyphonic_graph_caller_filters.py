"""Caller filters constrain graph seeds before path selection and fusion."""

from datetime import datetime, timezone

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.episodic_graph import EpisodicGraph, Fact, Gist, GraphEdge

STAMP = "2026-01-01T12:00:00"
CASES = [
    ({"from_date": "2026-01-01"}, "2025-12-31T23:59:59", "wanted", False),
    ({"from_date": "2026-01-01"}, "2026-01-01", "wanted", True),
    ({"from_date": "2026-01-01"}, None, "wanted", False),
    ({"to_date": "2026-01-01"}, "2026-01-02T00:00:00", "wanted", False),
    ({"to_date": "2026-01-01"}, "2026-01-01T23:59:59", "wanted", True),
    ({"to_date": "2026-01-01"}, "2026-01-01", "wanted", True),
    ({"to_date": "2026-01-01"}, None, "wanted", True),
    ({"source": "wanted"}, STAMP, "other", False),
    ({"source": "wanted"}, STAMP, "wanted", True),
    ({"topic": "wanted"}, STAMP, "wanted_archive", False),
    ({"topic": "wanted"}, STAMP, "wanted", True),
    ({"from_date": "2026-01-01", "to_date": "2026-01-01", "source": "wanted"},
     STAMP, "other", False),
]


def _store(tmp_path, monkeypatch, tier, kind, stamp, source):
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "1")
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
    for voice in ("VECTOR", "TEMPORAL", "FACT"):
        monkeypatch.setenv(f"MNEMOSYNE_VOICE_{voice}", "0")
    memory = BeamMemory(session_id="a", db_path=tmp_path / "filters.db")
    graph = EpisodicGraph(db_path=memory.db_path, conn=memory.conn)
    for mid in ("Alice", "middle", "target"):
        _row(memory, "working", mid, STAMP, "wanted")
    _row(memory, tier, "Bob", stamp, source)
    graph_stamp = datetime.now(timezone.utc).isoformat()
    for name in ("Alice", "Bob"):
        if kind == "gist":
            graph.store_gist(Gist(f"gist_{name}", name, graph_stamp, [name], None, None, None), name)
        else:
            graph.store_fact(Fact(f"fact_seed_{name}", name, "uses", "Python", graph_stamp, .9), name)
    for start, end in (("Alice", "middle"), ("middle", "target"), ("Bob", "target")):
        graph.add_edge(GraphEdge(start, end, "ctx", .9, graph_stamp))
    memory.conn.commit()
    engine = memory._get_polyphonic_engine()
    actual = engine._graph_voice
    seen = []

    def recording(*args, **kwargs):
        rows = actual(*args, **kwargs)
        seen.append(rows)
        return rows

    monkeypatch.setattr(engine, "_graph_voice", recording)
    return memory, seen


def _row(memory, tier, mid, stamp, source, session="a"):
    memory.conn.execute(
        f"INSERT INTO {tier}_memory(id,content,source,timestamp,session_id,scope) "
        "VALUES(?,?,?,?,?,'session')",
        (mid, f"unique {mid}", source, stamp, session),
    )


def _ask(memory, query, filters):
    return memory.recall(query, top_k=10, _track_recall=False, _skip_provenance=True, **filters)


def _target(rows):
    return next(row for row in rows if row.memory_id == "target")


@pytest.mark.parametrize("tier", ["working", "episodic"])
@pytest.mark.parametrize("kind", ["gist", "fact"])
@pytest.mark.parametrize("query", ["Alice then Bob", "Bob then Alice"])
@pytest.mark.parametrize("filters,stamp,source,eligible", CASES)
def test_public_recall_filters_seed_before_selecting_path(
    tmp_path, monkeypatch, tier, kind, query, filters, stamp, source, eligible
):
    memory, seen = _store(tmp_path, monkeypatch, tier, kind, stamp, source)
    try:
        rows = _ask(memory, query, filters)
        raw = _target(seen[-1])
        assert raw.metadata["seed"] == ("Bob" if eligible else "Alice")
        assert raw.metadata["depth"] == (1 if eligible else 2)
        assert raw.score == pytest.approx(.4 if eligible else .2)
        assert ("Bob" in {row["id"] for row in rows}) is eligible
        target = next(row for row in rows if row["id"] == "target")
        assert target["content"] == "unique target"
        if not eligible:
            # A caller-filtered seed must affect the final score no more than
            # an already-invisible seed. Both calls run the real engine/fusion.
            memory.conn.execute(f"UPDATE {tier}_memory SET session_id='foreign' WHERE id='Bob'")
            memory.conn.commit()
            reference = _ask(memory, query, filters)
            assert target["score"] == pytest.approx(
                next(row["score"] for row in reference if row["id"] == "target")
            )
    finally:
        memory.conn.close()


@pytest.mark.parametrize("tier", ["working", "episodic"])
@pytest.mark.parametrize("kind", ["gist", "fact"])
@pytest.mark.parametrize("query", ["Alice then Bob", "Bob then Alice"])
@pytest.mark.parametrize("twin", ["split", "foreign", "valid"])
def test_graph_seed_needs_all_filters_and_scope_on_one_backing_row(
    tmp_path, monkeypatch, tier, kind, query, twin
):
    memory, seen = _store(tmp_path, monkeypatch, tier, kind, STAMP, "wanted" if twin == "valid" else "other")
    other = "episodic" if tier == "working" else "working"
    try:
        _row(memory, other, "Bob", "2026-01-02" if twin == "split" else STAMP,
             "wanted", session="a" if twin == "split" else "foreign")
        memory.conn.commit()
        rows = _ask(memory, query, {"to_date": "2026-01-01", "source": "wanted"})
        raw = _target(seen[-1])
        assert raw.metadata["seed"] == ("Bob" if twin == "valid" else "Alice")
        assert raw.score == pytest.approx(.4 if twin == "valid" else .2)
        assert ("Bob" in {row["id"] for row in rows}) is (twin == "valid")
    finally:
        memory.conn.close()


@pytest.mark.parametrize("tier", ["working", "episodic"])
@pytest.mark.parametrize("kind", ["gist", "fact"])
@pytest.mark.parametrize("query", ["Alice then Bob", "Bob then Alice"])
def test_graph_seed_must_satisfy_source_and_topic_together(
    tmp_path, monkeypatch, tier, kind, query
):
    memory, seen = _store(tmp_path, monkeypatch, tier, kind, STAMP, "wanted")
    try:
        assert _ask(memory, query, {"source": "wanted", "topic": "other"}) == []
        assert seen[-1] == []
    finally:
        memory.conn.close()
