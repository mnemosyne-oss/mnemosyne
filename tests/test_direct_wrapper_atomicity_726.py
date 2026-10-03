"""Public durable-row atomicity coverage for #726 direct wrappers.

This file deliberately does not assert streaming behavior.  Option 3 scopes
#726 to durable BEAM/legacy rows; delivery for direct wrapper calls inside a
caller-owned transaction remains a separate contract.
"""

import sqlite3
from pathlib import Path

import pytest

import mnemosyne.core.beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.memory import Mnemosyne


def _memory(tmp_path):
    return Mnemosyne(
        session_id="direct-wrapper-726", db_path=Path(tmp_path) / "direct.db"
    )


def _row_count(conn, table, memory_id):
    return conn.execute(
        f"SELECT COUNT(*) FROM {table} WHERE id = ?", (memory_id,)
    ).fetchone()[0]


def _contents(conn, table, memory_id):
    return conn.execute(
        f"SELECT content FROM {table} WHERE id = ?", (memory_id,)
    ).fetchone()[0]


def test_direct_wrapper_dual_writes_share_beam_connection_and_commit_together(tmp_path):
    memory = _memory(tmp_path)

    assert memory.conn is memory.beam.conn
    memory_id = memory.remember("#726 direct wrapper durable row")
    assert _row_count(memory.conn, "working_memory", memory_id) == 1
    assert _row_count(memory.conn, "memories", memory_id) == 1

    assert memory.update(memory_id, content="#726 direct wrapper updated") is True
    assert (
        _contents(memory.conn, "working_memory", memory_id)
        == "#726 direct wrapper updated"
    )
    assert (
        _contents(memory.conn, "memories", memory_id) == "#726 direct wrapper updated"
    )

    assert memory.forget(memory_id) is True
    assert _row_count(memory.conn, "working_memory", memory_id) == 0
    assert _row_count(memory.conn, "memories", memory_id) == 0


def test_direct_wrapper_reopens_closed_shared_cache_for_same_path(tmp_path):
    path = Path(tmp_path) / "closed-shared-cache.db"
    first = Mnemosyne(session_id="first", db_path=path)
    first.conn.close()

    fresh = Mnemosyne(session_id="fresh", db_path=path)

    assert fresh.conn is fresh.beam.conn
    assert fresh.conn.execute("SELECT 1").fetchone()[0] == 1
    memory_id = fresh.remember("#726 durable rows after shared-cache reopen")
    assert _row_count(fresh.conn, "working_memory", memory_id) == 1
    assert _row_count(fresh.conn, "memories", memory_id) == 1


def test_direct_invalidate_is_beam_only_and_leaves_legacy_row_unchanged(tmp_path):
    memory = _memory(tmp_path)
    memory_id = memory.remember("#726 direct invalidate BEAM-only boundary")
    legacy_before = tuple(
        memory.conn.execute(
            """
            SELECT id, content, source, timestamp, session_id, importance, metadata_json
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()
    )

    assert memory.invalidate(memory_id) is True
    assert memory.conn.in_transaction is False
    working_row = memory.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()
    assert working_row[0] is not None
    assert working_row[1] is None
    legacy_after = memory.conn.execute(
        """
        SELECT id, content, source, timestamp, session_id, importance, metadata_json
        FROM memories
        WHERE id = ?
        """,
        (memory_id,),
    ).fetchone()
    assert legacy_after is not None
    assert tuple(legacy_after) == legacy_before


def test_direct_remember_failure_after_beam_write_rolls_back_both_durable_rows(
    tmp_path, monkeypatch
):
    memory = _memory(tmp_path)
    original_remember = memory.beam.remember

    def fail_after_beam(*args, **kwargs):
        original_remember(*args, **kwargs)
        raise RuntimeError("forced failure after BEAM write")

    monkeypatch.setattr(memory.beam, "remember", fail_after_beam)
    with pytest.raises(RuntimeError, match="forced failure after BEAM write"):
        memory.remember("#726 fail after beam")

    assert memory.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
    assert memory.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    assert memory.conn.in_transaction is False


def test_direct_remember_failure_after_legacy_write_rolls_back_both_durable_rows(
    tmp_path, monkeypatch
):
    memory = _memory(tmp_path)

    def fail_finalization(self):
        raise sqlite3.OperationalError("forced failure after legacy write")

    monkeypatch.setattr(beam_module._BeamConnection, "_real_commit", fail_finalization)
    with pytest.raises(
        sqlite3.OperationalError, match="forced failure after legacy write"
    ):
        memory.remember("#726 fail after legacy")

    assert memory.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
    assert memory.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0
    assert memory.conn.in_transaction is False


@pytest.mark.parametrize("decision", ["commit", "rollback"])
def test_direct_wrapper_caller_owned_transaction_controls_both_durable_rows(
    tmp_path, decision
):
    memory = _memory(tmp_path)
    conn = memory.conn
    conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    conn.commit()
    conn.execute("INSERT INTO caller_marker VALUES ('caller owns outcome')")

    memory_id = memory.remember(f"#726 caller-owned {decision}")
    assert conn.in_transaction is True
    assert _row_count(conn, "working_memory", memory_id) == 1
    assert _row_count(conn, "memories", memory_id) == 1
    with sqlite3.connect(memory.db_path) as outside:
        assert _row_count(outside, "working_memory", memory_id) == 0
        assert _row_count(outside, "memories", memory_id) == 0

    getattr(conn, decision)()
    expected = 1 if decision == "commit" else 0
    with sqlite3.connect(memory.db_path) as outside:
        assert _row_count(outside, "working_memory", memory_id) == expected
        assert _row_count(outside, "memories", memory_id) == expected


@pytest.mark.parametrize("operation", ["update", "forget"])
def test_direct_wrapper_failures_after_legacy_mutation_restore_both_rows(
    tmp_path, monkeypatch, operation
):
    memory = _memory(tmp_path)
    memory_id = memory.remember(f"#726 {operation} original")
    target = "update_working" if operation == "update" else "forget_working"

    def fail_after_legacy(*args, **kwargs):
        raise RuntimeError(f"forced {operation} failure after legacy mutation")

    monkeypatch.setattr(memory.beam, target, fail_after_legacy)
    with pytest.raises(RuntimeError, match="after legacy mutation"):
        if operation == "update":
            memory.update(memory_id, content="#726 replacement")
        else:
            memory.forget(memory_id)

    assert (
        _contents(memory.conn, "working_memory", memory_id)
        == f"#726 {operation} original"
    )
    assert _contents(memory.conn, "memories", memory_id) == f"#726 {operation} original"


@pytest.mark.parametrize("operation", ["update", "forget"])
@pytest.mark.parametrize("decision", ["commit", "rollback"])
def test_direct_wrapper_caller_decides_update_and_forget_rows(
    tmp_path, operation, decision
):
    memory = _memory(tmp_path)
    original_content = f"#726 caller {operation} original"
    memory_id = memory.remember(original_content)
    memory.conn.execute(
        "INSERT INTO memories (id, content, session_id) VALUES ('marker', 'marker', ?)",
        (memory.session_id,),
    )
    if operation == "update":
        memory.update(memory_id, content="#726 caller changed")
    else:
        memory.forget(memory_id)
    if operation == "update":
        # The caller's transaction owns visibility as well as final outcome.
        with sqlite3.connect(memory.db_path) as outside:
            assert _contents(outside, "working_memory", memory_id) == original_content
            assert _contents(outside, "memories", memory_id) == original_content
    getattr(memory.conn, decision)()
    expected = 0 if operation == "forget" and decision == "commit" else 1
    assert _row_count(memory.conn, "working_memory", memory_id) == expected
    assert _row_count(memory.conn, "memories", memory_id) == expected
    if operation == "update" and decision == "commit":
        assert _contents(memory.conn, "working_memory", memory_id) == "#726 caller changed"
        assert _contents(memory.conn, "memories", memory_id) == "#726 caller changed"
    if operation == "update" and decision == "rollback":
        assert _contents(memory.conn, "working_memory", memory_id) == original_content
        assert _contents(memory.conn, "memories", memory_id) == original_content


def test_direct_wrapper_cross_session_forget_removes_global_dual_rows(tmp_path):
    path = Path(tmp_path) / "global-forget.db"
    creator = Mnemosyne(session_id="session-a", db_path=path)
    memory_id = creator.remember("#726 global forget", scope="global")
    caller = Mnemosyne(session_id="session-b", db_path=path)

    assert caller.forget(memory_id) is True
    assert _row_count(caller.conn, "working_memory", memory_id) == 0
    assert _row_count(caller.conn, "memories", memory_id) == 0


def test_direct_wrapper_cross_session_forget_preserves_private_rows(tmp_path):
    path = Path(tmp_path) / "private-forget.db"
    creator = Mnemosyne(session_id="session-a", db_path=path)
    memory_id = creator.remember("#726 private forget")
    caller = Mnemosyne(session_id="session-b", db_path=path)

    assert caller.forget(memory_id) is False
    assert _row_count(caller.conn, "working_memory", memory_id) == 1
    assert _row_count(caller.conn, "memories", memory_id) == 1


def test_direct_wrapper_vector_path_does_not_commit_a_caller_transaction(
    tmp_path, monkeypatch
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("sqlite_vec")
    memory = _memory(tmp_path)
    if not beam_module._wm_vec_available(memory.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")

    real_commits = []
    original_real_commit = beam_module._BeamConnection._real_commit

    def observing_real_commit(self):
        real_commits.append(True)
        return original_real_commit(self)

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

    memory.conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    memory.conn.commit()
    memory.conn.execute("INSERT INTO caller_marker VALUES ('outer vector marker')")
    memory_id = memory.remember("#726 direct active vec caller transaction")

    assert real_commits == []
    assert memory.conn.in_transaction is True
    with sqlite3.connect(memory.db_path) as outside:
        assert _row_count(outside, "working_memory", memory_id) == 0
        assert _row_count(outside, "memories", memory_id) == 0

    memory.conn.rollback()
    assert _row_count(memory.conn, "working_memory", memory_id) == 0
    assert _row_count(memory.conn, "memories", memory_id) == 0


def test_direct_wrapper_republishes_live_core_connection_after_beam_other_path(
    tmp_path, monkeypatch
):
    path_a = Path(tmp_path) / "a.db"
    path_b = Path(tmp_path) / "b.db"
    first = Mnemosyne(session_id="first-a", db_path=path_a)
    BeamMemory(session_id="standalone-b", db_path=path_b)

    second = Mnemosyne(session_id="second-a", db_path=path_a)

    assert second.conn is first.conn
    assert second.conn is second.beam.conn
    assert second.conn.execute("SELECT 1").fetchone()[0] == 1
    content = "#726 A B A cache repair recall smoke"
    memory_id = second.remember(content)
    assert _row_count(second.conn, "working_memory", memory_id) == 1
    assert _row_count(second.conn, "memories", memory_id) == 1
    recalled = second.recall("A B A cache repair", top_k=5)
    assert any(
        result["id"] == memory_id and result["content"] == content
        for result in recalled
    )

    original_remember = second.beam.remember

    def fail_after_beam(*args, **kwargs):
        original_remember(*args, **kwargs)
        raise RuntimeError("forced A B A BEAM failure")

    monkeypatch.setattr(second.beam, "remember", fail_after_beam)
    with pytest.raises(RuntimeError, match="forced A B A BEAM failure"):
        second.remember("#726 A B A rollback")
    assert second.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1
    assert second.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 1


def test_owner_forget_removes_legacy_only_row_but_foreign_caller_cannot(tmp_path):
    path = Path(tmp_path) / "legacy-only-forget.db"
    owner = Mnemosyne(session_id="owner", db_path=path)
    memory_id = owner.remember("#726 legacy-only owner row")
    owner.conn.execute("DELETE FROM working_memory WHERE id = ?", (memory_id,))
    owner.conn.commit()
    caller = Mnemosyne(session_id="other", db_path=path)

    assert caller.forget(memory_id) is False
    assert _row_count(caller.conn, "memories", memory_id) == 1
    assert owner.forget(memory_id) is False
    assert _row_count(owner.conn, "memories", memory_id) == 0


def test_owner_legacy_only_forget_keeps_caller_transaction_rollbackable(
    tmp_path, monkeypatch
):
    path = Path(tmp_path) / "legacy-only-caller-transaction.db"
    owner = Mnemosyne(session_id="owner", db_path=path)
    memory_id = owner.remember("#764 legacy-only owner row")

    # Simulate BEAM trimming: the legacy mirror remains after working_memory
    # has been removed, so the public fallback takes its owner-only path.
    owner.conn.execute("DELETE FROM working_memory WHERE id = ?", (memory_id,))
    owner.conn.commit()
    assert _row_count(owner.conn, "working_memory", memory_id) == 0
    assert _row_count(owner.conn, "memories", memory_id) == 1

    owner.conn.execute("CREATE TABLE caller_marker (value TEXT NOT NULL)")
    owner.conn.commit()
    owner.conn.execute("INSERT INTO caller_marker VALUES ('caller owns rollback')")

    real_commits = []

    def fail_if_real_commit(self):
        real_commits.append(True)
        raise AssertionError("forget must not commit a caller-owned transaction")

    monkeypatch.setattr(beam_module._BeamConnection, "_real_commit", fail_if_real_commit)

    # The documented legacy-only fallback return remains False even when its
    # owner-only legacy delete succeeds.
    assert owner.forget(memory_id) is False
    assert owner.conn.in_transaction is True
    assert _row_count(owner.conn, "working_memory", memory_id) == 0
    assert _row_count(owner.conn, "memories", memory_id) == 0

    # A distinct file-backed connection proves neither the marker nor delete
    # escaped the caller-owned outer transaction.
    with sqlite3.connect(str(path)) as outside:
        assert outside.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0
        assert _row_count(outside, "working_memory", memory_id) == 0
        assert _row_count(outside, "memories", memory_id) == 1

    owner.conn.rollback()

    with sqlite3.connect(str(path)) as outside:
        assert outside.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0
        assert _row_count(outside, "working_memory", memory_id) == 0
        assert _row_count(outside, "memories", memory_id) == 1
    assert owner.conn.execute("SELECT COUNT(*) FROM caller_marker").fetchone()[0] == 0
    assert _row_count(owner.conn, "working_memory", memory_id) == 0
    assert _row_count(owner.conn, "memories", memory_id) == 1
    assert real_commits == []
