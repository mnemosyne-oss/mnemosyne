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
import sqlite3

import numpy as np
import pytest

import mnemosyne.core.beam as bm

TEXT = "Alice met Bob in Paris on March 3, 2026 to plan the launch at 5pm."

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
    assert bm.extract_temporal is not None
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
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memoria_facts WHERE source_memory_id = ?", (memory_id,)
    ).fetchone()[0] > 0
    if path == "new":
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0] == (1 if embeddings else 0)
        assert beam.conn.execute(
            "SELECT event_date FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0]


@pytest.mark.parametrize("kwargs", [{"dedupe": False}, {"extract_entities": False}],
                         ids=["no-dedupe", "no-entities"])
def test_other_write_paths_commit_twice(beam, monkeypatch, kwargs):
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    if kwargs.get("dedupe") is False:
        beam.remember(TEXT, source="user")
    call = {"extract_entities": True, **kwargs}

    memory_id, commits = _count_commits(beam, lambda: beam.remember(TEXT, source="user", **call))

    assert commits == 2
    # The one enrichment commit has to carry the writes, not land empty.
    assert _gist_count(beam, memory_id) == 1
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memoria_facts WHERE source_memory_id = ?", (memory_id,)).fetchone()[0] > 0
    mentions = beam.conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ? AND kind = 'mentions'",
        (memory_id,)).fetchone()[0]
    assert (mentions > 0) == (kwargs != {"extract_entities": False})


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


ENRICHMENT_TABLES = ("annotations", "memory_embeddings", "memoria_facts",
                     "memoria_timelines", "gists", "consolidated_facts")

# Every enrichment step remember() runs inside its BEAM-owned transaction, as
# (owner, name). Steps the dedup path never runs are listed in DEDUP_SKIPS.
INJECTION_POINTS = [
    ("beam", "_trim_working_memory"),
    ("module", "_store_working_embedding"),
    ("beam", "_add_temporal_triple"),
    ("module", "extract_temporal"),
    ("module", "_extract_and_store_entities"),
    ("beam", "extract_and_store_facts"),
    ("beam", "_ingest_graph_and_veracity"),
]
DEDUP_SKIPS = {"_trim_working_memory", "_store_working_embedding",
               "_add_temporal_triple", "extract_temporal"}
INJECTION_CELLS = [
    (path, point, error)
    for path in ("new", "dedup")
    for point in INJECTION_POINTS
    for error in (RuntimeError, KeyboardInterrupt)
    if not (path == "dedup" and point[1] in DEDUP_SKIPS)
]


def _assert_no_transaction_left_behind(beam, db_path):
    assert not beam.conn.in_transaction
    assert not beam.conn._defer_commit
    assert beam.conn._after_commit_hooks == []
    other = sqlite3.connect(db_path, timeout=0)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.rollback()
    finally:
        other.close()


@pytest.mark.parametrize("path,point,error", INJECTION_CELLS,
                         ids=[f"{p}-{n}-{e.__name__}" for p, (_, n), e in INJECTION_CELLS])
def test_failure_after_each_enrichment_step(beam, monkeypatch, tmp_path, path, point, error):
    # Each step runs for real, then fails. Steps whose ordinary errors
    # remember() deliberately swallows still have to leave no transaction
    # behind; a failure that escapes rolls back all of this call's enrichment
    # and keeps the row.
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    text = "Alice works at Initech. " + TEXT
    beam.remember("Alice works at Acme.", source="user")
    if path == "dedup":
        beam.remember(text, source="user")
        beam.conn.execute("DELETE FROM consolidated_facts WHERE object = 'Initech'")
        beam.conn.commit()
    before = {t: beam.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
              for t in ENRICHMENT_TABLES}
    owner, name = point
    target = beam if owner == "beam" else bm
    real = getattr(target, name)
    reached = []

    def run_then_fail(*args, **kwargs):
        real(*args, **kwargs)
        reached.append(name)
        raise error(f"stopped after {name}")

    monkeypatch.setattr(target, name, run_then_fail)

    try:
        beam.remember(text, source="user", extract_entities=True)
        escaped = False
    except error:
        escaped = True

    assert reached == [name]
    _assert_no_transaction_left_behind(beam, tmp_path / "memory.db")
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?", (text,)).fetchone()[0] == 1
    if error is KeyboardInterrupt:
        assert escaped
    if escaped:
        for table, count in before.items():
            assert beam.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == count, table
    else:
        # The swallowed failure must not stop the steps after it: the graph
        # step runs last, so its gist and the queued conflict prove they ran.
        memory_id = beam.conn.execute(
            "SELECT id FROM working_memory WHERE content = ?", (text,)).fetchone()[0]
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (memory_id,)).fetchone()[0] == 1
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM consolidated_facts WHERE object = 'Initech'").fetchone()[0] == 1


def _seed_expired_row(beam):
    vec = np.ones(bm._embeddings.EMBEDDING_DIM, dtype=np.float32)
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES ('expired', 'expired row', 'user', datetime('now', '-30 days'), 'commit-count')")
    bm._store_working_embedding(beam.conn, "expired", vec)
    beam.conn.commit()


def _expired_row_counts(beam):
    return tuple(beam.conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {column} = 'expired'").fetchone()[0]
        for table, column in (("working_memory", "id"), ("memory_embeddings", "memory_id")))


def test_trim_removes_the_expired_row(beam, monkeypatch):
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    _seed_expired_row(beam)

    beam.remember(TEXT, source="user")

    assert _expired_row_counts(beam) == (0, 0)


def test_failing_enrichment_rolls_back_the_trim(beam, monkeypatch):
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    _seed_expired_row(beam)

    def fail(*args, **kwargs):
        raise RuntimeError("graph store down")

    monkeypatch.setattr(beam, "_ingest_graph_and_veracity", fail)

    with pytest.raises(RuntimeError):
        beam.remember(TEXT, source="user")

    assert _expired_row_counts(beam) == (1, 1)


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


@pytest.mark.parametrize("path", ["new", "dedup"])
def test_conflict_probe_runs_after_the_enrichment_commit(beam, monkeypatch, path):
    # The fleet probe does peer I/O, so it must not hold the home bank's
    # write lock; queued pairs still get probed once the enrichment commits.
    conflicting = "Alice works at Initech."
    if path == "dedup":
        # Store the row first, then forget its fact so the dedup hit below
        # consolidates it again against the conflicting one.
        beam.remember(conflicting, source="user")
        beam.conn.execute("DELETE FROM consolidated_facts")
        beam.conn.commit()
    beam.remember("Alice works at Acme.", source="user")
    consolidator = beam.veracity_consolidator
    real_retry = consolidator.retry_pending_conflicts
    seen = []

    def retry():
        seen.append(beam.conn.in_transaction)
        return real_retry()

    monkeypatch.setattr(consolidator, "retry_pending_conflicts", retry)

    beam.remember(conflicting, source="user")

    assert seen == [False]
    assert beam.conn.execute("SELECT COUNT(*) FROM conflict_probe_pending").fetchone()[0] == 0


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt], ids=["exception", "interrupt"])
@pytest.mark.parametrize("path", ["new", "dedup"])
def test_failure_after_a_queued_conflict_rolls_it_back(beam, monkeypatch, tmp_path, path, error):
    # Fails after the graph step has queued a conflict pair and its probe
    # hook, but before the enrichment commits.
    conflicting = "Alice works at Initech."
    if path == "dedup":
        beam.remember(conflicting, source="user")
        beam.conn.execute("DELETE FROM consolidated_facts")
        beam.conn.commit()
    beam.remember("Alice works at Acme.", source="user")
    consolidator = beam.veracity_consolidator
    retries = []
    monkeypatch.setattr(consolidator, "retry_pending_conflicts", lambda: retries.append(1))
    real_ingest = beam._ingest_graph_and_veracity
    queued = []

    def ingest_then_fail(*args, **kwargs):
        real_ingest(*args, **kwargs)
        queued.append(len(beam.conn._after_commit_hooks))
        raise error("stopped after graph ingestion")

    monkeypatch.setattr(beam, "_ingest_graph_and_veracity", ingest_then_fail)

    with pytest.raises(error):
        beam.remember(conflicting, source="user")

    assert queued == [1]
    assert not beam.conn.in_transaction
    assert beam.conn._after_commit_hooks == []
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM consolidated_facts WHERE object = 'Initech'").fetchone()[0] == 0
    # The queue table is created lazily, so the rollback can remove it too.
    if beam.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'conflict_probe_pending'").fetchone():
        assert beam.conn.execute("SELECT COUNT(*) FROM conflict_probe_pending").fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?", (conflicting,)).fetchone()[0] == 1
    other = sqlite3.connect(tmp_path / "memory.db", timeout=0)
    try:
        other.execute("BEGIN IMMEDIATE")
        other.rollback()
    finally:
        other.close()
    beam.conn.execute("UPDATE working_memory SET importance = importance WHERE 0")
    beam.conn.commit()
    assert retries == []


def test_interrupt_inside_a_caller_transaction_rolls_back_only_the_savepoint(beam):
    _open_caller_transaction(beam)

    with pytest.raises(KeyboardInterrupt):
        with bm._deferred_commits(beam.conn, immediate=True):
            beam.conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
                "VALUES ('scoped', 'scoped row', 'user', datetime('now'), 'commit-count')")
            raise KeyboardInterrupt

    assert beam.conn.in_transaction
    assert not beam.conn._defer_commit
    ids = {r[0] for r in beam.conn.execute(
        "SELECT id FROM working_memory WHERE id IN ('caller-marker', 'scoped')")}
    assert ids == {"caller-marker"}
    beam.conn.rollback()


def _hold_lock_after_row_commit(beam, monkeypatch):
    # Another writer takes the lock as soon as remember()'s row commit lands.
    beam.conn.execute("PRAGMA busy_timeout=1")
    other = sqlite3.connect(beam.db_path, timeout=0, isolation_level=None)
    real_commit = bm._BeamConnection.commit

    def commit(self):
        had_transaction = self.in_transaction
        real_commit(self)
        if had_transaction and not other.in_transaction:
            other.execute("BEGIN IMMEDIATE")

    monkeypatch.setattr(bm._BeamConnection, "commit", commit)
    return other


def _gist_count(beam, memory_id):
    return beam.conn.execute(
        "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (memory_id,)).fetchone()[0]


def test_busy_lock_skips_enrichment_on_a_dedup_hit(beam, monkeypatch, caplog):
    # The row update already committed, so a duplicate write must not report failure.
    existing = beam.remember(TEXT, source="user")
    beam.conn.execute("DELETE FROM gists WHERE memory_id = ?", (existing,))
    beam.conn.commit()
    sleeps = []
    monkeypatch.setattr(bm.time, "sleep", sleeps.append)
    other = _hold_lock_after_row_commit(beam, monkeypatch)
    try:
        assert beam.remember(TEXT, source="user") == existing
    finally:
        other.rollback()
        other.close()

    assert len(sleeps) == 2
    assert not beam.conn.in_transaction
    assert _gist_count(beam, existing) == 0
    assert "enrichment skipped" in caplog.text


def test_busy_lock_on_a_dedup_hit_skips_the_llm_step(beam, monkeypatch):
    # The LLM call would only wait on the same lock after spending model time.
    existing = beam.remember(TEXT, source="user")
    calls = []
    monkeypatch.setattr(bm, "_extract_and_store_facts", lambda *args: calls.append(args))
    monkeypatch.setattr(bm.time, "sleep", lambda seconds: None)
    other = _hold_lock_after_row_commit(beam, monkeypatch)
    try:
        assert beam.remember(TEXT, source="user", extract=True) == existing
    finally:
        other.rollback()
        other.close()

    assert calls == []


def test_busy_lock_still_raises_on_a_new_row(beam, monkeypatch):
    monkeypatch.setattr(bm.time, "sleep", lambda seconds: None)
    other = _hold_lock_after_row_commit(beam, monkeypatch)
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            beam.remember(TEXT, source="user")
    finally:
        other.rollback()
        other.close()

    assert not beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?", (TEXT,)).fetchone()[0] == 1


@pytest.mark.parametrize("path", ["new", "dedup"])
def test_lock_released_during_backoff_runs_enrichment(beam, monkeypatch, path):
    existing = beam.remember(TEXT, source="user") if path == "dedup" else None
    if existing:
        beam.conn.execute("DELETE FROM gists WHERE memory_id = ?", (existing,))
        beam.conn.commit()
    other = _hold_lock_after_row_commit(beam, monkeypatch)
    sleeps = []

    def release(seconds):
        sleeps.append(seconds)
        other.rollback()
        # Stop the commit hook from taking the lock again.
        monkeypatch.undo()
        monkeypatch.setattr(bm.time, "sleep", lambda seconds: None)

    monkeypatch.setattr(bm.time, "sleep", release)
    try:
        memory_id = beam.remember(TEXT, source="user")
    finally:
        other.close()

    assert len(sleeps) == 1
    assert _gist_count(beam, memory_id) == 1


def test_lock_released_during_backoff_still_runs_the_llm_step(beam, monkeypatch):
    existing = beam.remember(TEXT, source="user")
    in_transaction = []
    monkeypatch.setattr(bm, "_extract_and_store_facts",
                        lambda b, *args: in_transaction.append(b.conn.in_transaction))
    with monkeypatch.context() as patch:
        other = _hold_lock_after_row_commit(beam, patch)
        sleeps = []

        def release(seconds):
            sleeps.append(seconds)
            other.rollback()
            # Stop the commit hook from taking the lock again.
            patch.undo()

        monkeypatch.setattr(bm.time, "sleep", release)
        try:
            assert beam.remember(TEXT, source="user", extract=True) == existing
        finally:
            other.close()

    assert len(sleeps) == 1
    assert in_transaction == [False]


def test_busy_error_without_an_error_code_counts_as_busy():
    # What Python 3.10 raises: its sqlite3 exceptions carry no error code.
    assert bm._is_busy_error(sqlite3.OperationalError("database is locked"))
    assert not bm._is_busy_error(sqlite3.OperationalError("no such table: x"))


def _sqlite_error(code):
    error = sqlite3.OperationalError("injected sqlite failure")
    error.sqlite_errorcode = code
    return error


BUSY, BUSY_RECOVERY, BUSY_TIMEOUT, LOCKED, LOCKED_SHAREDCACHE = 5, 261, 773, 6, 262


@pytest.mark.parametrize("code,busy", [(BUSY, True), (BUSY_RECOVERY, True), (BUSY_TIMEOUT, True),
                                       (LOCKED, False), (LOCKED_SHAREDCACHE, False)])
def test_busy_classification_uses_the_primary_result_code(code, busy):
    assert bm._is_busy_error(_sqlite_error(code)) is busy


def _fail_enrichment_entry(beam, monkeypatch, code):
    # remember()'s row write takes the first BEGIN IMMEDIATE; enrichment takes the next.
    real_execute = beam.conn.execute
    begins = []

    def execute(sql, *args, **kwargs):
        if sql == "BEGIN IMMEDIATE":
            begins.append(sql)
            if len(begins) > 1:
                raise _sqlite_error(code)
        return real_execute(sql, *args, **kwargs)

    monkeypatch.setattr(beam.conn, "execute", execute)
    return begins


@pytest.mark.parametrize("code", [BUSY_RECOVERY, BUSY_TIMEOUT])
def test_extended_busy_at_enrichment_entry_retries_then_skips(beam, monkeypatch, code):
    existing = beam.remember(TEXT, source="user")
    sleeps = []
    monkeypatch.setattr(bm.time, "sleep", sleeps.append)
    begins = _fail_enrichment_entry(beam, monkeypatch, code)

    assert beam.remember(TEXT, source="user") == existing

    assert len(begins) == 4
    assert len(sleeps) == 2
    assert not beam.conn.in_transaction


def test_non_busy_error_at_enrichment_entry_is_not_retried(beam, monkeypatch):
    beam.remember(TEXT, source="user")
    sleeps = []
    monkeypatch.setattr(bm.time, "sleep", sleeps.append)
    begins = _fail_enrichment_entry(beam, monkeypatch, LOCKED)

    with pytest.raises(sqlite3.OperationalError, match="injected"):
        beam.remember(TEXT, source="user")

    assert len(begins) == 2
    assert sleeps == []


def test_extended_busy_inside_the_block_is_not_retried(beam, monkeypatch):
    beam.remember(TEXT, source="user")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise _sqlite_error(BUSY_TIMEOUT)

    monkeypatch.setattr(beam, "_ingest_graph_and_veracity", fail)

    with pytest.raises(sqlite3.OperationalError, match="injected"):
        beam.remember(TEXT, source="user")

    assert calls == [1]
    assert not beam.conn.in_transaction


def test_busy_error_inside_the_block_is_not_retried_or_skipped(beam, monkeypatch):
    # Only failing to enter is a busy lock; a failure inside still rolls back and raises.
    beam.remember(TEXT, source="user")
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(beam, "_ingest_graph_and_veracity", fail)

    with pytest.raises(sqlite3.OperationalError, match="locked"):
        beam.remember(TEXT, source="user")

    assert calls == [1]
    assert not beam.conn.in_transaction
