"""Authorized episodic forget and cross-tier cascade safety (#959/#1002)."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from mnemosyne.batch_tool import apply_beam_batch, validate_batch_operations
from mnemosyne.core.beam import BeamMemory, _vec_available, _vec_insert
from mnemosyne.core.memory import Mnemosyne


def _seed_episodic(
    conn, memory_id: str, session_id: str, *, scope: str = "session"
) -> int:
    conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, 'episodic content', 'test', datetime('now'), ?, 0.6, ?)",
        (memory_id, session_id, scope),
    )
    conn.commit()
    return conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0]


def _seed_children(conn, memory_id: str) -> None:
    conn.execute(
        "INSERT INTO annotations (memory_id, kind, value) "
        "VALUES (?, 'mentions', 'child')",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, '[]')",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, 'gist', ?)",
        (f"gist-{memory_id}", memory_id),
    )
    conn.commit()


def _child_counts(conn, memory_id: str) -> tuple[int, int, int]:
    return (
        conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0],
        conn.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()[0],
        conn.execute(
            "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0],
    )


def _seed_same_id_parents(beam: BeamMemory, memory_id: str) -> None:
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, 'working content', 'test', datetime('now'), ?, 0.6, 'session')",
        (memory_id, beam.session_id),
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, 'episodic content', 'test', datetime('now'), ?, 0.6, 'session')",
        (memory_id, beam.session_id),
    )
    beam.conn.commit()
    _seed_children(beam.conn, memory_id)


def _seed_legacy_parent(conn, memory_id: str, session_id: str) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS memories ("
        "id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT, "
        "session_id TEXT, importance REAL, metadata_json TEXT)"
    )
    conn.execute(
        "INSERT INTO memories (id, content, source, timestamp, session_id, importance) "
        "VALUES (?, 'legacy content', 'test', datetime('now'), ?, 0.6)",
        (memory_id, session_id),
    )
    conn.commit()


def test_forget_working_preserves_children_when_legacy_parent_shares_id(
    tmp_path: Path,
):
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_legacy_parent(beam.conn, "collision", "owner")
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES ('collision', 'working', 'test', datetime('now'), 'owner', 0.6, 'session')"
    )
    beam.conn.commit()
    _seed_children(beam.conn, "collision")

    assert beam.forget_working("collision") is True
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = 'collision'"
    ).fetchone()[0] == 1
    assert _child_counts(beam.conn, "collision") == (1, 1, 1)


def test_forget_episodic_preserves_children_when_legacy_parent_shares_id(
    tmp_path: Path,
):
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_legacy_parent(beam.conn, "collision", "owner")
    _seed_episodic(beam.conn, "collision", "owner")
    _seed_children(beam.conn, "collision")

    assert beam.forget_episodic("collision") is True
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = 'collision'"
    ).fetchone()[0] == 1
    assert _child_counts(beam.conn, "collision") == (1, 1, 1)


def test_forget_deletes_owned_episodic_row_and_unambiguous_children(
    tmp_path: Path,
):
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    rowid = _seed_episodic(beam.conn, "episode", "owner")
    _seed_children(beam.conn, "episode")
    if _vec_available(beam.conn):
        _vec_insert(beam.conn, rowid, [0.1] * 384)

    assert beam.forget_episodic("episode") is True
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'episode'"
    ).fetchone()[0] == 0
    assert _child_counts(beam.conn, "episode") == (0, 0, 0)
    if _vec_available(beam.conn):
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM vec_episodes WHERE rowid = ?", (rowid,)
        ).fetchone()[0] == 0


def test_forget_episodic_refuses_foreign_private_row_and_keeps_children(
    tmp_path: Path,
):
    db_path = tmp_path / "forget.db"
    owner = BeamMemory(session_id="owner", db_path=db_path)
    _seed_episodic(owner.conn, "private-episode", "owner")
    _seed_children(owner.conn, "private-episode")

    foreign = BeamMemory(session_id="foreign", db_path=db_path)
    assert foreign.forget_episodic("private-episode") is False
    assert _child_counts(foreign.conn, "private-episode") == (1, 1, 1)


def test_forget_episodic_allows_global_row_cross_session(tmp_path: Path):
    db_path = tmp_path / "forget.db"
    owner = BeamMemory(session_id="owner", db_path=db_path)
    _seed_episodic(owner.conn, "global-episode", "owner", scope="global")

    foreign = BeamMemory(session_id="foreign", db_path=db_path)
    assert foreign.forget_episodic("global-episode") is True


def test_forget_episodic_preserves_children_when_working_parent_shares_id(
    tmp_path: Path,
):
    """Ambiguous legacy children stay with the surviving working parent.

    The shipped child schema has only memory_id, so it cannot prove which tier
    owns a same-ID child. The compatible forget boundary must preserve such
    rows rather than guess and irreversibly delete another tier's data.
    """
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_same_id_parents(beam, "collision")

    assert beam.forget_episodic("collision") is True
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'collision'"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = 'collision'"
    ).fetchone()[0] == 1
    assert _child_counts(beam.conn, "collision") == (1, 1, 1)


def test_forget_working_preserves_children_when_episodic_parent_shares_id(
    tmp_path: Path,
):
    """The ownership guard is symmetric for the pre-existing working path."""
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_same_id_parents(beam, "collision")

    assert beam.forget_working("collision") is True
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = 'collision'"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'collision'"
    ).fetchone()[0] == 1
    assert _child_counts(beam.conn, "collision") == (1, 1, 1)


def test_forget_working_preserves_foreign_episodic_parent_and_children(
    tmp_path: Path,
):
    db_path = tmp_path / "forget.db"
    owner = BeamMemory(session_id="owner", db_path=db_path)
    owner.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES ('collision', 'working', 'test', datetime('now'), "
        "'owner', 0.6, 'session')"
    )
    owner.conn.commit()
    _seed_children(owner.conn, "collision")

    foreign = BeamMemory(session_id="foreign", db_path=db_path)
    _seed_episodic(foreign.conn, "collision", "foreign")

    assert owner.forget_working("collision") is True
    assert owner.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = 'collision'"
    ).fetchone()[0] == 0
    assert owner.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'collision'"
    ).fetchone()[0] == 1
    assert _child_counts(owner.conn, "collision") == (1, 1, 1)


def test_forget_shared_children_after_final_same_id_parent_deleted(
    tmp_path: Path,
):
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_same_id_parents(beam, "collision")

    assert beam.forget_episodic("collision") is True
    assert beam.forget_working("collision") is True
    assert _child_counts(beam.conn, "collision") == (0, 0, 0)


@pytest.mark.parametrize(
    ("forget_method", "insert_table", "delete_sql"),
    [
        ("forget_working", "episodic_memory", "DELETE FROM working_memory"),
        ("forget_episodic", "working_memory", "DELETE FROM episodic_memory"),
    ],
)
def test_forget_locks_before_a_competing_parent_can_arrive(
    tmp_path: Path,
    forget_method: str,
    insert_table: str,
    delete_sql: str,
):
    db_path = tmp_path / "forget.db"
    beam = BeamMemory(session_id="owner", db_path=db_path)
    if forget_method == "forget_working":
        beam.conn.execute(
            "INSERT INTO working_memory "
            "(id, content, source, timestamp, session_id, importance, scope) "
            "VALUES ('race', 'working', 'test', datetime('now'), 'owner', 0.6, 'session')"
        )
        beam.conn.commit()
    else:
        _seed_episodic(beam.conn, "race", "owner")
    _seed_children(beam.conn, "race")

    attempts: list[str] = []
    competitor_thread: list[threading.Thread] = []

    def compete() -> None:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 0")
        try:
            conn.execute(
                f"INSERT INTO {insert_table} "
                "(id, content, source, timestamp, session_id, importance, scope) "
                "VALUES ('race', 'competitor', 'test', datetime('now'), "
                "'competitor', 0.6, 'session')"
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            conn.rollback()
            attempts.append(str(exc))
        else:
            attempts.append("committed")
        finally:
            conn.close()

    def insert_competing_parent(sql: str) -> None:
        if competitor_thread or delete_sql not in sql:
            return
        thread = threading.Thread(target=compete)
        competitor_thread.append(thread)
        thread.start()
        thread.join(timeout=2)

    beam.conn.set_trace_callback(insert_competing_parent)
    try:
        assert getattr(beam, forget_method)("race") is True
    finally:
        beam.conn.set_trace_callback(None)

    assert competitor_thread and not competitor_thread[0].is_alive()
    assert attempts and "locked" in attempts[0]


def test_mnemosyne_forget_locks_before_episodic_fallback_reads(tmp_path: Path):
    db_path = tmp_path / "forget.db"
    mem = Mnemosyne(session_id="owner", db_path=db_path)
    attempts: list[str] = []
    competitor_thread: list[threading.Thread] = []

    def compete() -> None:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 0")
        try:
            conn.execute(
                "INSERT INTO episodic_memory "
                "(id, content, source, timestamp, session_id, importance, scope) "
                "VALUES ('race', 'competitor', 'test', datetime('now'), "
                "'competitor', 0.6, 'session')"
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            conn.rollback()
            attempts.append(str(exc))
        else:
            attempts.append("committed")
        finally:
            conn.close()

    def insert_before_fallback(sql: str) -> None:
        if competitor_thread or "SELECT 1 FROM memories WHERE id" not in sql:
            return
        thread = threading.Thread(target=compete)
        competitor_thread.append(thread)
        thread.start()
        thread.join(timeout=2)

    mem.conn.set_trace_callback(insert_before_fallback)
    try:
        assert mem.forget("race") is False
    finally:
        mem.conn.set_trace_callback(None)

    assert competitor_thread and not competitor_thread[0].is_alive()
    assert attempts and "locked" in attempts[0]
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'race'"
    ).fetchone()[0] == 0


def test_mnemosyne_forget_falls_back_to_episodic_and_emits_only_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mem = Mnemosyne(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_episodic(mem.conn, "episode", "owner")
    events: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        mem, "_emit_wrapper", lambda *args, **kwargs: events.append((args, kwargs))
    )

    assert mem.forget("episode") is True
    assert mem.forget("missing") is False
    assert events == [(("MEMORY_INVALIDATED", "episode"), {})]
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'episode'"
    ).fetchone()[0] == 0
    fresh = Mnemosyne(session_id="owner", db_path=mem.db_path)
    assert fresh.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'episode'"
    ).fetchone()[0] == 0


def test_mnemosyne_forget_removes_episodic_after_legacy_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mem = Mnemosyne(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_episodic(mem.conn, "collision", "owner")
    _seed_legacy_parent(mem.conn, "collision", "owner")
    events: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        mem, "_emit_wrapper", lambda *args, **kwargs: events.append((args, kwargs))
    )

    assert mem.forget("collision") is True
    assert events == [(("MEMORY_INVALIDATED", "collision"), {})]
    assert mem.get("collision") is None


def test_batch_forget_falls_back_to_episodic(tmp_path: Path):
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_episodic(beam.conn, "batch-episode", "owner")

    result = apply_beam_batch(
        beam,
        validate_batch_operations(
            [{"action": "forget", "memory_id": "batch-episode"}]
        ),
    )

    assert result["status"] == "ok"
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'batch-episode'"
    ).fetchone()[0] == 0


def test_batch_forget_locks_before_episodic_fallback_reads(tmp_path: Path):
    db_path = tmp_path / "forget.db"
    beam = BeamMemory(session_id="owner", db_path=db_path)
    _seed_episodic(beam.conn, "batch-episode", "owner")
    attempts: list[str] = []
    competitor_thread: list[threading.Thread] = []

    def compete() -> None:
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA busy_timeout = 0")
        try:
            conn.execute(
                "INSERT INTO episodic_memory "
                "(id, content, source, timestamp, session_id, importance, scope) "
                "VALUES ('competitor', 'content', 'test', datetime('now'), "
                "'competitor', 0.6, 'session')"
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            conn.rollback()
            attempts.append(str(exc))
        else:
            attempts.append("committed")
        finally:
            conn.close()

    def compete_before_fallback_read(sql: str) -> None:
        if competitor_thread or "SELECT rowid FROM episodic_memory" not in sql:
            return
        thread = threading.Thread(target=compete)
        competitor_thread.append(thread)
        thread.start()
        thread.join(timeout=2)

    beam.conn.set_trace_callback(compete_before_fallback_read)
    try:
        result = apply_beam_batch(
            beam,
            validate_batch_operations(
                [{"action": "forget", "memory_id": "batch-episode"}]
            ),
        )
    finally:
        beam.conn.set_trace_callback(None)

    assert result["status"] == "ok"
    assert competitor_thread and not competitor_thread[0].is_alive()
    assert attempts and "locked" in attempts[0]


def test_batch_forget_preserves_legacy_beam_without_episodic_method(
    tmp_path: Path,
):
    class LegacyBeam:
        def __init__(self, db_path: Path) -> None:
            self.conn = sqlite3.connect(db_path)

        def forget_working(self, memory_id: str) -> bool:
            assert memory_id == "missing"
            return False

    beam = LegacyBeam(tmp_path / "legacy.db")
    result = apply_beam_batch(
        beam,
        validate_batch_operations(
            [{"action": "forget", "memory_id": "missing"}]
        ),
    )
    beam.conn.close()

    assert result == {
        "status": "error",
        "error": "batch_failed",
        "failed_index": 0,
        "action": "forget",
    }


def test_forget_episodic_failure_rolls_back_full_cascade(tmp_path: Path):
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_episodic(beam.conn, "episode", "owner")
    _seed_children(beam.conn, "episode")
    beam.conn.execute(
        "CREATE TRIGGER fail_annotation_delete BEFORE DELETE ON annotations "
        "BEGIN SELECT RAISE(ABORT, 'forced child failure'); END"
    )
    beam.conn.commit()

    with pytest.raises(Exception, match="forced child failure"):
        beam.forget_episodic("episode")

    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'episode'"
    ).fetchone()[0] == 1
    assert _child_counts(beam.conn, "episode") == (1, 1, 1)


def test_forget_episodic_defers_event_to_outer_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mem = Mnemosyne(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_episodic(mem.conn, "episode", "owner")
    events: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        mem, "_emit_wrapper", lambda *args, **kwargs: events.append((args, kwargs))
    )

    mem.conn.execute("BEGIN")
    assert mem.forget("episode") is True
    assert events == []
    mem.conn.commit()
    assert events == [(("MEMORY_INVALIDATED", "episode"), {})]


def test_forget_episodic_rollback_discards_event_and_restores_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mem = Mnemosyne(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_episodic(mem.conn, "episode", "owner")
    events: list[tuple[tuple, dict]] = []
    invalidations: list[str] = []
    monkeypatch.setattr(
        mem, "_emit_wrapper", lambda *args, **kwargs: events.append((args, kwargs))
    )
    monkeypatch.setattr(
        mem.beam,
        "_invalidate_query_cache",
        lambda: invalidations.append("invalidated"),
    )

    mem.conn.execute("BEGIN")
    assert mem.forget("episode") is True
    mem.conn.rollback()

    assert events == []
    assert invalidations == ["invalidated"]
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = 'episode'"
    ).fetchone()[0] == 1


def test_forget_episodic_invalidates_cache_inside_outer_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    beam = BeamMemory(session_id="owner", db_path=tmp_path / "forget.db")
    _seed_episodic(beam.conn, "episode", "owner")
    invalidations: list[str] = []
    monkeypatch.setattr(
        beam, "_invalidate_query_cache", lambda: invalidations.append("invalidated")
    )

    beam.conn.execute("BEGIN")
    assert beam.forget_episodic("episode") is True
    assert invalidations == ["invalidated"]
    beam.conn.rollback()
