"""Public regression coverage for #726 batch transaction ownership."""

import contextlib
import sqlite3
from pathlib import Path

import pytest

import mnemosyne.core.beam as beam_module
from mnemosyne.batch_tool import apply_beam_batch, validate_batch_operations
from mnemosyne.core.beam import BeamMemory, _deferred_commits, _guarded_transaction


def _beam(tmp_path):
    return BeamMemory(
        session_id="batch-owner-726", db_path=Path(tmp_path) / "batch-owner.db"
    )


def _count_content(conn, content):
    return conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?", (content,)
    ).fetchone()[0]


def test_caller_owned_batch_releases_savepoint_and_outer_rollback_removes_batch_writes(
    tmp_path,
):
    beam = _beam(tmp_path)
    existing_id = beam.remember("#726 before update")

    beam.conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    beam.conn.commit()
    beam.conn.execute("INSERT INTO caller_marker VALUES ('outer marker')")

    result = apply_beam_batch(
        beam,
        validate_batch_operations(
            [
                {"action": "remember", "content": "#726 remembered in batch"},
                {
                    "action": "update",
                    "memory_id": existing_id,
                    "content": "#726 updated in batch",
                },
            ]
        ),
    )

    assert result["status"] == "ok"
    assert beam.conn.in_transaction is True
    assert (
        beam.conn.execute("SELECT value FROM caller_marker").fetchone()[0]
        == "outer marker"
    )

    beam.conn.rollback()
    assert _count_content(beam.conn, "#726 remembered in batch") == 0
    assert beam.get(existing_id)["content"] == "#726 before update"
    assert beam.conn.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0


def test_failed_batch_rolls_back_its_savepoint_without_events_or_caller_data_loss(
    tmp_path,
):
    beam = _beam(tmp_path)
    events = []

    beam.conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    beam.conn.commit()
    beam.conn.execute("INSERT INTO caller_marker VALUES ('survives')")

    result = apply_beam_batch(
        beam,
        validate_batch_operations(
            [
                {"action": "remember", "content": "#726 must roll back"},
                {
                    "action": "update",
                    "memory_id": "missing",
                    "content": "never written",
                },
            ]
        ),
        audit_event=lambda name, **kwargs: events.append((name, kwargs)),
    )

    assert result["status"] == "error"
    assert result["failed_index"] == 1
    assert events == []
    assert beam.conn.in_transaction is True
    assert (
        beam.conn.execute("SELECT value FROM caller_marker").fetchone()[0] == "survives"
    )
    assert _count_content(beam.conn, "#726 must roll back") == 0

    beam.conn.rollback()
    assert beam.conn.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0


def test_forget_cascade_failure_after_guard_preserves_caller_transaction(
    tmp_path, monkeypatch
):
    beam = _beam(tmp_path)
    target_id = beam.remember("#726 cascade target")
    beam.annotations.add(target_id, "test", "annotation")
    annotation_count = beam.conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (target_id,)
    ).fetchone()[0]
    beam.conn.execute(
        f"""
        CREATE TRIGGER forced_annotation_failure
        BEFORE DELETE ON annotations
        WHEN OLD.memory_id = '{target_id}'
        BEGIN
            SELECT RAISE(ABORT, 'forced cascade failure');
        END
        """
    )
    beam.conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    beam.conn.commit()
    beam.conn.execute("INSERT INTO caller_marker VALUES ('survives guarded rollback')")

    entered_guard = []
    original_guard = beam_module._guarded_transaction

    @contextlib.contextmanager
    def observing_guard(conn):
        entered_guard.append(True)
        with original_guard(conn):
            yield

    monkeypatch.setattr(beam_module, "_guarded_transaction", observing_guard)
    events = []
    result = apply_beam_batch(
        beam,
        validate_batch_operations(
            [
                {"action": "remember", "content": "#726 must roll back after guard"},
                {"action": "forget", "memory_id": target_id},
            ]
        ),
        audit_event=lambda name, **kwargs: events.append((name, kwargs)),
    )

    assert entered_guard == [True]  # Failure is downstream of forget_working's guard.
    assert result == {
        "status": "error",
        "error": "batch_failed",
        "failed_index": 1,
        "action": "forget",
    }
    assert "forced cascade failure" not in str(result)
    assert events == []
    assert beam.conn.in_transaction is True
    assert beam.conn.execute("SELECT value FROM caller_marker").fetchone()[0] == (
        "survives guarded rollback"
    )
    assert _count_content(beam.conn, "#726 must roll back after guard") == 0
    assert beam.get(target_id)["content"] == "#726 cascade target"
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (target_id,)
        ).fetchone()[0]
        == annotation_count
    )


def test_nested_deferred_commits_use_distinct_savepoints_and_restore_deferral(tmp_path):
    beam = _beam(tmp_path)
    conn = beam.conn
    conn.execute("CREATE TABLE deferred_probe (value TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO deferred_probe VALUES ('caller')")
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        with _deferred_commits(conn):
            assert conn._defer_commit is True
            conn.execute("INSERT INTO deferred_probe VALUES ('outer')")
            with _deferred_commits(conn):
                assert conn._defer_commit is True
                conn.execute("INSERT INTO deferred_probe VALUES ('inner')")
            assert conn._defer_commit is True
    finally:
        conn.set_trace_callback(None)

    savepoints = [
        statement
        for statement in statements
        if statement.startswith("SAVEPOINT mnemosyne_deferred_commits_")
    ]
    assert len(savepoints) == 2
    assert len(set(savepoints)) == 2
    assert conn._defer_commit is False
    assert [
        row[0]
        for row in conn.execute("SELECT value FROM deferred_probe ORDER BY rowid")
    ] == [
        "caller",
        "outer",
        "inner",
    ]
    conn.rollback()


def test_nested_deferred_failure_restores_outer_deferral_and_rolls_back_only_inner(
    tmp_path,
):
    beam = _beam(tmp_path)
    conn = beam.conn
    conn.execute("CREATE TABLE deferred_probe (value TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO deferred_probe VALUES ('caller')")

    with _deferred_commits(conn):
        conn.execute("INSERT INTO deferred_probe VALUES ('outer')")
        try:
            with _deferred_commits(conn):
                conn.execute("INSERT INTO deferred_probe VALUES ('inner')")
                raise RuntimeError("nested failure")
        except RuntimeError:
            pass
        assert conn._defer_commit is True
        assert [
            row[0]
            for row in conn.execute("SELECT value FROM deferred_probe ORDER BY rowid")
        ] == [
            "caller",
            "outer",
        ]

    assert conn._defer_commit is False
    assert [
        row[0]
        for row in conn.execute("SELECT value FROM deferred_probe ORDER BY rowid")
    ] == [
        "caller",
        "outer",
    ]
    conn.rollback()


def test_guarded_transaction_success_preserves_non_deferred_caller_transaction(
    tmp_path,
):
    beam = _beam(tmp_path)
    conn = beam.conn
    conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    conn.execute("CREATE TABLE guarded_probe (value TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO caller_marker VALUES ('caller')")

    with _guarded_transaction(conn):
        conn.execute("INSERT INTO guarded_probe VALUES ('guarded')")

    assert conn.in_transaction is True
    assert conn.execute("SELECT value FROM caller_marker").fetchone()[0] == "caller"
    with sqlite3.connect(beam.db_path) as outside:
        assert outside.execute("SELECT COUNT(*) FROM guarded_probe").fetchone()[0] == 0

    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM guarded_probe").fetchone()[0] == 0


def test_guarded_transaction_failure_preserves_non_deferred_caller_transaction(
    tmp_path,
):
    beam = _beam(tmp_path)
    conn = beam.conn
    conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    conn.execute("CREATE TABLE guarded_probe (value TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO caller_marker VALUES ('caller')")

    try:
        with _guarded_transaction(conn):
            conn.execute("INSERT INTO guarded_probe VALUES ('discarded')")
            raise RuntimeError("forced guarded failure")
    except RuntimeError as exc:
        assert str(exc) == "forced guarded failure"

    assert conn.in_transaction is True
    assert conn.execute("SELECT value FROM caller_marker").fetchone()[0] == "caller"
    assert conn.execute("SELECT COUNT(*) FROM guarded_probe").fetchone()[0] == 0

    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0


def test_get_context_preserves_caller_owned_transaction_without_nested_begin(tmp_path):
    beam = _beam(tmp_path)
    beam.remember("#726 context item")
    conn = beam.conn
    conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO caller_marker VALUES ('before context')")
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        rows = beam.get_context()
    finally:
        conn.set_trace_callback(None)

    assert [row["content"] for row in rows] == ["#726 context item"]
    assert "BEGIN TRANSACTION" not in statements
    assert conn.in_transaction is True
    assert (
        conn.execute("SELECT value FROM caller_marker").fetchone()[0]
        == "before context"
    )
    conn.execute("INSERT INTO caller_marker VALUES ('after context')")

    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0


def test_active_sqlite_vec_path_defers_commit_inside_caller_owned_batch(
    tmp_path, monkeypatch
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sqlite_vec")
    beam = _beam(tmp_path)
    if not beam_module._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")

    beam.conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    beam.conn.commit()

    real_commits = []
    original_real_commit = beam_module._BeamConnection._real_commit

    def observing_real_commit(self):
        real_commits.append(True)
        return original_real_commit(self)

    # Keep the embedding backend deterministic while exercising production's
    # real float conversion, sqlite-vec SQL, and vec_working virtual table.
    embedding = np.array(
        [1.0] + [0.0] * (beam_module.EMBEDDING_DIM - 1), dtype=np.float32
    )

    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda contents: [embedding.copy() for _ in contents],
    )
    monkeypatch.setattr(
        beam_module._BeamConnection, "_real_commit", observing_real_commit
    )

    beam.conn.execute("INSERT INTO caller_marker VALUES ('outer vector marker')")
    result = apply_beam_batch(
        beam,
        validate_batch_operations(
            [{"action": "remember", "content": "#726 forced active vec batch"}]
        ),
    )

    assert result["status"] == "ok"
    assert real_commits == []
    assert beam.conn.in_transaction is True
    memory_id = result["results"][0]["memory_id"]
    rowid = beam.conn.execute(
        "SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0]
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        == 1
    )
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)
        ).fetchone()[0]
        == 1
    )

    with sqlite3.connect(beam.db_path) as outside:
        assert outside.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0
        assert (
            outside.execute(
                "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            outside.execute(
                "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()[0]
            == 0
        )

    beam.conn.rollback()
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        == 0
    )
    assert (
        beam.conn.execute(
            "SELECT COUNT(*) FROM vec_working WHERE rowid = ?", (rowid,)
        ).fetchone()[0]
        == 0
    )
