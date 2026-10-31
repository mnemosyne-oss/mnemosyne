"""remember() commits the row once and its local enrichment once.

Every commit is an fsync, and the enrichment helpers (trim, embedding,
temporal annotations, entities, MEMORIA, graph) each commit on their own.

Invariant: with no caller transaction, remember() issues at most two COMMITs
(the row, then all local enrichment) and the LLM fact step never runs inside
a write transaction. With a caller transaction open, remember() behaves as it
did before batching: the helpers' own commits close the caller's transaction.

Grid: path (new row, dedup hit) x embeddings (available, unavailable)
x caller transaction (none, open), plus the dedupe=False and
extract_entities=False write paths, the LLM fact step and a failing
enrichment step on each path.
"""
import itertools

import numpy as np
import pytest

import mnemosyne.core.beam as bm

TEXT = "Alice met Bob in Paris on 3 March 2026 to plan the launch at 5pm."

@pytest.fixture
def beam(tmp_path, monkeypatch):
    vec = np.ones(bm._embeddings.EMBEDDING_DIM, dtype=np.float32)
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: np.stack([vec for _ in texts]))
    memory = bm.BeamMemory(session_id="commit-count", db_path=tmp_path / "memory.db")
    memory.remember("warm-up row so lazy schema setup is not counted", source="user")
    return memory


def _count_commits(beam, fn):
    # Counts this connection's COMMITs only; that is where the fsyncs under
    # test happen.
    statements = []
    beam.conn.set_trace_callback(statements.append)
    try:
        result = fn()
    finally:
        beam.conn.set_trace_callback(None)
    return result, sum(1 for s in statements if s.strip().upper().startswith("COMMIT"))


def _open_caller_transaction(beam):
    beam.conn.execute("BEGIN")
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES ('caller-marker', 'caller row', 'user', datetime('now'), 'commit-count')")


CELLS = list(itertools.product(("new", "dedup"), (True, False), (False, True)))


@pytest.mark.parametrize("path,embeddings,caller_txn", CELLS,
                         ids=[f"{p}-{'emb' if e else 'noemb'}-{'caller_txn' if c else 'own'}"
                              for p, e, c in CELLS])
def test_remember_commit_count(beam, monkeypatch, path, embeddings, caller_txn):
    monkeypatch.setattr(bm._embeddings, "available", lambda: embeddings)
    if path == "dedup":
        beam.remember(TEXT, source="user", extract_entities=True)
    if caller_txn:
        _open_caller_transaction(beam)

    memory_id, commits = _count_commits(
        beam, lambda: beam.remember(TEXT, source="user", extract_entities=True))

    if caller_txn:
        # Unchanged from before batching: the helpers' own commits close the
        # caller's transaction. Their exact number depends on optional
        # backends such as sqlite-vec, so only the outcome is pinned.
        assert commits >= 1
    else:
        assert commits == 2
    assert not beam.conn.in_transaction
    if caller_txn:
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'caller-marker'").fetchone()[0] == 1
    # Enrichment landed: entity mentions and the gist for this row.
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ? AND kind = 'mentions'",
        (memory_id,)).fetchone()[0] > 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (memory_id,)).fetchone()[0] == 1
    if path == "new":
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0] == (1 if embeddings else 0)


@pytest.mark.parametrize("kwargs", [{"dedupe": False}, {"extract_entities": False}],
                         ids=["no-dedupe", "no-entities"])
def test_other_write_paths_commit_twice(beam, monkeypatch, kwargs):
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    beam.remember(TEXT, source="user")
    call = {"extract_entities": True, **kwargs}

    _, commits = _count_commits(beam, lambda: beam.remember(TEXT, source="user", **call))

    assert commits == 2


@pytest.mark.parametrize("path,caller_txn", list(itertools.product(("new", "dedup"), (False, True))),
                         ids=["new-own", "new-caller_txn", "dedup-own", "dedup-caller_txn"])
def test_llm_fact_step_runs_outside_a_write_transaction(beam, monkeypatch, path, caller_txn):
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    if path == "dedup":
        beam.remember(TEXT, source="user")
    if caller_txn:
        _open_caller_transaction(beam)
    seen = []
    monkeypatch.setattr(bm, "_extract_and_store_facts",
                        lambda b, *args: seen.append(b.conn.in_transaction))

    beam.remember(TEXT, source="user", extract=True)

    assert seen == [False]


@pytest.mark.parametrize("path", ["new", "dedup"])
def test_failing_enrichment_keeps_the_row(beam, monkeypatch, path):
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    if path == "dedup":
        beam.remember("unrelated earlier row", source="user")
        beam.remember(TEXT, source="user")
    before = {t: beam.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ("annotations", "memory_embeddings", "memoria_facts", "memoria_timelines")}

    def fail(*args, **kwargs):
        raise RuntimeError("graph store down")

    monkeypatch.setattr(beam, "_ingest_graph_and_veracity", fail)

    with pytest.raises(RuntimeError):
        beam.remember(TEXT, source="user", extract_entities=True)

    assert not beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?", (TEXT,)).fetchone()[0] == 1
    # The graph step runs last, so the earlier enrichment rolled back with it.
    for table, count in before.items():
        assert beam.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count


def test_embedding_runs_outside_a_write_transaction(beam, monkeypatch):
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    seen = []
    vec = np.ones(bm._embeddings.EMBEDDING_DIM, dtype=np.float32)

    def embed(texts):
        seen.append(beam.conn.in_transaction)
        return np.stack([vec for _ in texts])

    monkeypatch.setattr(bm._embeddings, "embed", embed)

    beam.remember(TEXT, source="user")

    assert seen == [False]
