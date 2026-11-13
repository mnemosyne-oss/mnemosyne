"""Working-memory trim removes the trimmed rows' derived vectors (#1164).

``_trim_working_memory()`` runs after every ``remember()`` and
``remember_batch()``. The ``wm_ad`` trigger clears only ``fts_working``, so the
trim itself has to delete the ``vec_working`` rows (keyed by the working
row's rowid) and the ``memory_embeddings`` fallback rows (keyed by id) of
the rows it removes, and leave the vectors of every surviving row alone.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

import mnemosyne.core.beam as bm


def _stamp(**delta) -> str:
    return (datetime.now(timezone.utc) - timedelta(**delta)).replace(tzinfo=None).isoformat()


def _vector(seed: int):
    return [0.1 + seed * 0.01] * bm.EMBEDDING_DIM


def _add_row(beam, memory_id, timestamp, *, session_id="trim", pinned=0, consolidated_at=None, seed=0):
    beam.conn.execute(
        "INSERT INTO working_memory(id, content, timestamp, session_id, pinned, consolidated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (memory_id, f"trim witness {memory_id}", timestamp, session_id, pinned, consolidated_at),
    )
    bm._store_working_embedding(beam.conn, memory_id, _vector(seed))
    beam.conn.commit()


def _working_rowids(beam):
    return dict(beam.conn.execute("SELECT id, rowid FROM working_memory"))


def _embedding_ids(beam):
    return {row[0] for row in beam.conn.execute("SELECT memory_id FROM memory_embeddings")}


def _vec_rowids(beam):
    return {row[0] for row in beam.conn.execute("SELECT rowid FROM vec_working")}


def _require_vec(beam):
    if not bm._wm_vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_working table unavailable")


@pytest.fixture
def beam(tmp_path):
    memory = bm.BeamMemory(session_id="trim", db_path=tmp_path / "trim.db")
    try:
        yield memory
    finally:
        memory.conn.close()


def test_item_cap_trim_removes_vectors_of_trimmed_rows_only(beam, monkeypatch):
    _require_vec(beam)
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "older", _stamp(minutes=30), seed=1)
    _add_row(beam, "old", _stamp(minutes=20), seed=2)
    _add_row(beam, "newest", _stamp(minutes=1), seed=3)
    _add_row(beam, "pinned", _stamp(minutes=40), pinned=1, seed=4)
    _add_row(beam, "consolidated", _stamp(minutes=40), consolidated_at=_stamp(minutes=5), seed=5)
    _add_row(beam, "other-session", _stamp(minutes=40), session_id="other", seed=6)
    assert _vec_rowids(beam) == set(_working_rowids(beam).values())

    beam._trim_working_memory()

    survivors = _working_rowids(beam)
    assert set(survivors) == {"newest", "pinned", "consolidated", "other-session"}
    assert _embedding_ids(beam) == set(survivors)
    assert _vec_rowids(beam) == set(survivors.values())


def test_ttl_trim_removes_vectors_of_expired_rows(beam, monkeypatch):
    _require_vec(beam)
    monkeypatch.setattr(bm, "WORKING_MEMORY_TTL_HOURS", 24)
    _add_row(beam, "expired", _stamp(hours=25), seed=1)
    _add_row(beam, "fresh", _stamp(minutes=1), seed=2)
    assert len(_vec_rowids(beam)) == 2

    beam._trim_working_memory()

    survivors = _working_rowids(beam)
    assert set(survivors) == {"fresh"}
    assert _embedding_ids(beam) == {"fresh"}
    assert _vec_rowids(beam) == {survivors["fresh"]}


def test_fallback_only_trim_removes_embeddings_of_trimmed_rows(tmp_path, monkeypatch):
    """Without sqlite-vec and without a vec_working table, trim still clears
    the memory_embeddings fallback rows of the rows it deletes."""
    monkeypatch.setattr(bm, "_SQLITE_VEC_AVAILABLE", False)
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    memory = bm.BeamMemory(session_id="trim", db_path=tmp_path / "plain.db")
    try:
        assert not bm._vec_table_exists(memory.conn, "vec_working")
        _add_row(memory, "old", _stamp(minutes=30), seed=1)
        _add_row(memory, "newest", _stamp(minutes=1), seed=2)

        memory._trim_working_memory()

        assert set(_working_rowids(memory)) == {"newest"}
        assert _embedding_ids(memory) == {"newest"}
    finally:
        memory.conn.close()


def test_trim_keeps_fallback_vector_while_episodic_row_owns_the_id(beam, monkeypatch):
    """``memory_embeddings`` carries only ``memory_id``; an episodic row with
    the same id still owns it after the working row is trimmed (#1002)."""
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "shared", _stamp(minutes=30), seed=1)
    _add_row(beam, "newest", _stamp(minutes=1), seed=2)
    beam.conn.execute(
        "INSERT INTO episodic_memory(id, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
        ("shared", "episodic copy", _stamp(minutes=30), "trim"),
    )
    beam.conn.commit()
    vec_on = bm._wm_vec_available(beam.conn)

    beam._trim_working_memory()

    survivors = _working_rowids(beam)
    assert set(survivors) == {"newest"}
    assert _embedding_ids(beam) == {"shared", "newest"}
    if vec_on:
        assert _vec_rowids(beam) == {survivors["newest"]}


def test_failed_trim_rolls_back_vector_deletes(beam, monkeypatch):
    _require_vec(beam)
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "old", _stamp(minutes=30), seed=1)
    _add_row(beam, "newest", _stamp(minutes=1), seed=2)
    before_vec = _vec_rowids(beam)
    beam.conn.execute(
        "CREATE TRIGGER fail_trim BEFORE DELETE ON working_memory "
        "BEGIN SELECT RAISE(ABORT, 'trim blocked'); END"
    )
    beam.conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="trim blocked"):
        beam._trim_working_memory()

    assert not beam.conn.in_transaction
    assert set(_working_rowids(beam)) == {"old", "newest"}
    assert _embedding_ids(beam) == {"old", "newest"}
    assert _vec_rowids(beam) == before_vec


def test_failed_commit_rolls_back_trim(beam, monkeypatch):
    """A commit that fails (deferred foreign key) must not leave the trim
    deletes staged in an open transaction."""
    _require_vec(beam)
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "old", _stamp(minutes=30), seed=1)
    _add_row(beam, "newest", _stamp(minutes=1), seed=2)
    before_vec = _vec_rowids(beam)
    beam.conn.execute(
        "CREATE TABLE trim_fk_witness (memory_id TEXT REFERENCES working_memory(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    beam.conn.execute("INSERT INTO trim_fk_witness VALUES ('old')")
    beam.conn.commit()

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        beam._trim_working_memory()

    assert not beam.conn.in_transaction
    assert set(_working_rowids(beam)) == {"old", "newest"}
    assert _embedding_ids(beam) == {"old", "newest"}
    assert _vec_rowids(beam) == before_vec


def test_failed_rollback_keeps_the_original_trim_error(beam, monkeypatch):
    _require_vec(beam)
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "old", _stamp(minutes=30), seed=1)
    _add_row(beam, "newest", _stamp(minutes=1), seed=2)
    beam.conn.execute(
        "CREATE TABLE trim_fk_witness (memory_id TEXT REFERENCES working_memory(id) "
        "DEFERRABLE INITIALLY DEFERRED)"
    )
    beam.conn.execute("INSERT INTO trim_fk_witness VALUES ('old')")
    beam.conn.commit()
    real_rollback = bm._BeamConnection.rollback

    def broken_rollback(conn):
        raise sqlite3.OperationalError("rollback failed")

    monkeypatch.setattr(bm._BeamConnection, "rollback", broken_rollback)

    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        beam._trim_working_memory()

    real_rollback(beam.conn)
    assert set(_working_rowids(beam)) == {"old", "newest"}


@pytest.mark.parametrize("over_cap", [False, True])
def test_trim_inside_caller_transaction_commits_nothing(beam, monkeypatch, over_cap):
    _require_vec(beam)
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1 if over_cap else 10)
    _add_row(beam, "old", _stamp(minutes=30), seed=1)
    _add_row(beam, "newest", _stamp(minutes=1), seed=2)
    before_vec = _vec_rowids(beam)
    beam.conn.execute("BEGIN IMMEDIATE")
    beam.conn.execute(
        "INSERT INTO episodic_memory(id, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
        ("caller-marker", "caller write", _stamp(minutes=1), "trim"),
    )

    beam._trim_working_memory()

    assert beam.conn.in_transaction
    expected = {"newest"} if over_cap else {"old", "newest"}
    assert set(_working_rowids(beam)) == expected
    beam.conn.rollback()
    assert beam.conn.execute("SELECT 1 FROM episodic_memory WHERE id = 'caller-marker'").fetchone() is None
    assert set(_working_rowids(beam)) == {"old", "newest"}
    assert _embedding_ids(beam) == {"old", "newest"}
    assert _vec_rowids(beam) == before_vec


def test_failed_trim_inside_caller_transaction_rolls_back_only_the_trim(beam, monkeypatch):
    _require_vec(beam)
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    _add_row(beam, "old", _stamp(minutes=30), seed=1)
    _add_row(beam, "newest", _stamp(minutes=1), seed=2)
    before_vec = _vec_rowids(beam)
    beam.conn.execute(
        "CREATE TRIGGER fail_trim BEFORE DELETE ON working_memory "
        "BEGIN SELECT RAISE(ABORT, 'trim blocked'); END"
    )
    beam.conn.commit()
    beam.conn.execute("BEGIN IMMEDIATE")
    beam.conn.execute(
        "INSERT INTO episodic_memory(id, content, timestamp, session_id) VALUES (?, ?, ?, ?)",
        ("caller-marker", "caller write", _stamp(minutes=1), "trim"),
    )

    with pytest.raises(sqlite3.IntegrityError, match="trim blocked"):
        beam._trim_working_memory()

    assert beam.conn.in_transaction
    beam.conn.commit()
    assert beam.conn.execute("SELECT 1 FROM episodic_memory WHERE id = 'caller-marker'").fetchone() is not None
    assert set(_working_rowids(beam)) == {"old", "newest"}
    assert _embedding_ids(beam) == {"old", "newest"}
    assert _vec_rowids(beam) == before_vec


def test_trim_keeps_rows_while_vec_working_exists_but_is_unavailable(tmp_path, monkeypatch):
    """A store with vec_working opened without sqlite-vec must not delete
    working rows whose vectors it cannot delete; the next trim with
    sqlite-vec loaded removes both."""
    db_path = tmp_path / "vec.db"
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    memory = bm.BeamMemory(session_id="trim", db_path=db_path)
    try:
        _require_vec(memory)
        _add_row(memory, "old", _stamp(minutes=30), seed=1)
        _add_row(memory, "newest", _stamp(minutes=1), seed=2)
        before_vec = _vec_rowids(memory)
    finally:
        memory.conn.close()

    monkeypatch.setattr(bm, "_SQLITE_VEC_AVAILABLE", False)
    memory = bm.BeamMemory(session_id="trim", db_path=db_path)
    try:
        assert bm._vec_table_exists(memory.conn, "vec_working")
        assert not bm._wm_vec_available(memory.conn)

        memory._trim_working_memory()

        assert not memory.conn.in_transaction
        assert set(_working_rowids(memory)) == {"old", "newest"}
        assert _embedding_ids(memory) == {"old", "newest"}
    finally:
        memory.conn.close()

    monkeypatch.setattr(bm, "_SQLITE_VEC_AVAILABLE", True)
    memory = bm.BeamMemory(session_id="trim", db_path=db_path)
    try:
        assert _vec_rowids(memory) == before_vec

        memory._trim_working_memory()

        survivors = _working_rowids(memory)
        assert set(survivors) == {"newest"}
        assert _embedding_ids(memory) == {"newest"}
        assert _vec_rowids(memory) == {survivors["newest"]}
    finally:
        memory.conn.close()


def test_remember_trim_removes_vectors_of_the_row_it_evicts(beam, monkeypatch):
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 2)
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        bm._embeddings, "embed", lambda values: [_vector(len(text)) for text in values]
    )
    first = beam.remember("first trim witness", source="test")
    second = beam.remember("second trim witness, longer", source="test")
    for memory_id, minutes in ((first, 30), (second, 20)):
        beam.conn.execute(
            "UPDATE working_memory SET timestamp = ? WHERE id = ?", (_stamp(minutes=minutes), memory_id)
        )
    beam.conn.commit()
    assert _embedding_ids(beam) == {first, second}

    third = beam.remember("third trim witness, the longest one", source="test")

    survivors = _working_rowids(beam)
    assert set(survivors) == {second, third}
    assert _embedding_ids(beam) == {second, third}
    if bm._wm_vec_available(beam.conn):
        assert _vec_rowids(beam) == set(survivors.values())
