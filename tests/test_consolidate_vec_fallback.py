"""Episodic consolidation keeps a dense fallback when sqlite-vec writes fail."""

import json
import sqlite3

import numpy as np
import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory


def _embedding() -> np.ndarray:
    vector = np.zeros(beam_module.EMBEDDING_DIM, dtype=np.float32)
    vector[:4] = [0.25, -0.5, 0.75, 1.0]
    return vector


@pytest.fixture
def direct_json_embeddings(monkeypatch):
    """Use a deterministic embedding with sqlite-vec unavailable."""
    vector = _embedding()
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda texts: np.stack([vector for _ in texts]),
    )
    monkeypatch.setattr(beam_module, "_vec_available", lambda conn: False)
    return vector


@pytest.fixture
def real_vec_embeddings(monkeypatch):
    """Use real sqlite-vec tables with a deterministic provider result."""
    pytest.importorskip("sqlite_vec")
    vector = _embedding()
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        beam_module._embeddings,
        "embed",
        lambda texts: np.stack([vector for _ in texts]),
    )
    return vector


def _new_vec_beam(tmp_path, real_vec_embeddings, session_id="s1"):
    beam = BeamMemory(db_path=tmp_path / f"{session_id}.db", session_id=session_id)
    if not beam_module._vec_available(beam.conn):
        pytest.skip("sqlite-vec vec_episodes unavailable in this build")
    return beam


def _install_vec_rollback_failure(beam, monkeypatch):
    """Make _vec_insert execute a real SQLite RAISE(ROLLBACK)."""
    beam.conn.execute("CREATE TABLE vec_insert_rollback_probe (value INTEGER)")
    beam.conn.execute("""
        CREATE TRIGGER rollback_vec_insert
        BEFORE INSERT ON vec_insert_rollback_probe
        BEGIN
            SELECT RAISE(ROLLBACK, 'forced vec transaction rollback');
        END
    """)
    beam.conn.commit()

    def fail_vec_insert(conn, rowid, embedding, *, commit=True):
        conn.execute("INSERT INTO vec_insert_rollback_probe VALUES (1)")

    monkeypatch.setattr(beam_module, "_vec_insert", fail_vec_insert)


def _install_fallback_rollback_failure(beam, monkeypatch):
    """Fail ANN first, then make the JSON fallback roll back SQLite."""
    beam.conn.execute("""
        CREATE TRIGGER rollback_episodic_json_fallback
        BEFORE INSERT ON memory_embeddings
        BEGIN
            SELECT RAISE(ROLLBACK, 'forced fallback transaction rollback');
        END
    """)
    beam.conn.commit()

    def fail_vec_insert(*args, **kwargs):
        raise RuntimeError("controlled ANN failure")

    monkeypatch.setattr(beam_module, "_vec_insert", fail_vec_insert)


def _install_direct_fallback_rollback_failure(beam):
    """Make the direct JSON fallback roll back SQLite."""
    beam.conn.execute("""
        CREATE TRIGGER rollback_direct_episodic_json_fallback
        BEFORE INSERT ON memory_embeddings
        BEGIN
            SELECT RAISE(ROLLBACK, 'forced direct fallback transaction rollback');
        END
    """)
    beam.conn.commit()


def test_vec_insert_failure_commits_json_fallback_for_later_lookup(
    tmp_path, monkeypatch, real_vec_embeddings
):
    beam = _new_vec_beam(tmp_path, real_vec_embeddings)

    def fail_vec_insert(conn, rowid, embedding, *, commit=True):
        assert conn.in_transaction
        assert conn.execute(
            "SELECT 1 FROM episodic_memory WHERE rowid = ?", (rowid,)
        ).fetchone()
        raise RuntimeError("private vec failure detail")

    monkeypatch.setattr(beam_module, "_vec_insert", fail_vec_insert)

    memory_id = beam.consolidate_to_episodic(
        "A summary that remains densely retrievable.", ["wm-1"]
    )
    row = beam.conn.execute(
        "SELECT rowid, binary_vector FROM episodic_memory WHERE id = ?", (memory_id,)
    ).fetchone()
    rowid = row["rowid"]
    stored = beam.conn.execute(
        "SELECT embedding_json FROM memory_embeddings WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()

    assert stored is not None
    np.testing.assert_allclose(json.loads(stored[0]), real_vec_embeddings)
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM vec_episodes WHERE rowid = ?", (rowid,)
    ).fetchone()[0] == 0
    assert row["binary_vector"] is not None
    assert not beam.conn.in_transaction

    fallback_conn = sqlite3.connect(beam.db_path)
    fallback_conn.row_factory = sqlite3.Row
    try:
        with pytest.raises(sqlite3.OperationalError, match="no such module: vec0"):
            fallback_conn.execute("SELECT 1 FROM vec_episodes LIMIT 0")
        results = beam_module._in_memory_vec_search(
            fallback_conn, real_vec_embeddings, k=5
        )
        assert rowid in {result["rowid"] for result in results}
    finally:
        fallback_conn.close()


def test_provider_unavailable_keeps_fts_only_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)
    beam = BeamMemory(db_path=tmp_path / "fts-only.db", session_id="fts-only")

    memory_id = beam.consolidate_to_episodic("Useful without an embedding.", [])

    assert beam.conn.execute(
        "SELECT content FROM episodic_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == "Useful without an embedding."
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 0
    assert not beam.conn.in_transaction


def test_direct_json_fallback_success_remains_json_and_binary_backed(
    tmp_path, direct_json_embeddings
):
    beam = BeamMemory(db_path=tmp_path / "direct-json.db", session_id="direct-json")

    memory_id = beam.consolidate_to_episodic("Stored in the direct fallback.", [])
    row = beam.conn.execute(
        "SELECT binary_vector FROM episodic_memory WHERE id = ?", (memory_id,)
    ).fetchone()
    stored = beam.conn.execute(
        "SELECT embedding_json FROM memory_embeddings WHERE memory_id = ?",
        (memory_id,),
    ).fetchone()

    np.testing.assert_allclose(json.loads(stored[0]), direct_json_embeddings)
    assert row["binary_vector"] is not None
    assert not beam.conn.in_transaction


def test_direct_json_fallback_abort_commits_fts_only_with_precise_warning(
    tmp_path, direct_json_embeddings, caplog
):
    beam = BeamMemory(db_path=tmp_path / "direct-abort.db", session_id="direct-abort")
    beam.conn.execute("""
        CREATE TRIGGER reject_direct_episodic_json_fallback
        BEFORE INSERT ON memory_embeddings
        BEGIN
            SELECT RAISE(ABORT, 'private direct json failure detail');
        END
    """)
    beam.conn.commit()

    memory_id = beam.consolidate_to_episodic("Direct fallback becomes FTS-only.", [])
    row = beam.conn.execute(
        "SELECT rowid, content, binary_vector FROM episodic_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()

    assert row["content"] == "Direct fallback becomes FTS-only."
    assert row["binary_vector"] is None
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 0
    assert not beam.conn.in_transaction
    assert (
        "consolidate_to_episodic: memory_embeddings fallback failed; summary "
        "stored FTS-only (rowid=1, fallback_error=IntegrityError)"
    ) in caplog.text
    assert "private direct json failure detail" not in caplog.text


def test_direct_json_fallback_rollback_preserves_owned_transaction_failure(
    tmp_path, direct_json_embeddings, caplog
):
    beam = BeamMemory(
        db_path=tmp_path / "direct-owned-rollback.db",
        session_id="direct-owned-rollback",
    )
    _install_direct_fallback_rollback_failure(beam)

    with pytest.raises(
        sqlite3.IntegrityError,
        match="forced direct fallback transaction rollback",
    ):
        beam.consolidate_to_episodic("Direct fallback rolled back its transaction.", [])

    assert not beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE content = ?",
        ("Direct fallback rolled back its transaction.",),
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE binary_vector IS NOT NULL"
    ).fetchone()[0] == 0
    assert "stored FTS-only" not in caplog.text


def test_direct_json_fallback_rollback_preserves_caller_transaction_failure(
    tmp_path, direct_json_embeddings, caplog
):
    beam = BeamMemory(
        db_path=tmp_path / "direct-caller-rollback.db",
        session_id="direct-caller-rollback",
    )
    _install_direct_fallback_rollback_failure(beam)
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
        (
            "direct-rollback-marker",
            "caller marker",
            "test",
            "2026-01-01T00:00:00",
            "direct-caller-rollback",
        ),
    )
    assert beam.conn.in_transaction

    with pytest.raises(
        sqlite3.IntegrityError,
        match="forced direct fallback transaction rollback",
    ):
        beam.consolidate_to_episodic(
            "Direct fallback rolled back its caller transaction.",
            [],
            emit_event=False,
        )

    assert not beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = 'direct-rollback-marker'"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE content = ?",
        ("Direct fallback rolled back its caller transaction.",),
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE binary_vector IS NOT NULL"
    ).fetchone()[0] == 0
    assert "stored FTS-only" not in caplog.text


def test_vec_and_json_fallback_failures_commit_fts_only_with_precise_warning(
    tmp_path, monkeypatch, real_vec_embeddings, caplog
):
    beam = _new_vec_beam(tmp_path, real_vec_embeddings, session_id="both-fail")
    beam.conn.execute("""
        CREATE TRIGGER reject_episodic_json_fallback
        BEFORE INSERT ON memory_embeddings
        BEGIN
            SELECT RAISE(ABORT, 'private json failure detail');
        END
    """)
    beam.conn.commit()

    def fail_vec_insert(*args, **kwargs):
        raise RuntimeError("private vec failure detail")

    monkeypatch.setattr(beam_module, "_vec_insert", fail_vec_insert)

    memory_id = beam.consolidate_to_episodic("FTS-only after both writes fail.", [])
    row = beam.conn.execute(
        "SELECT rowid, content, binary_vector FROM episodic_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()

    assert row["content"] == "FTS-only after both writes fail."
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM vec_episodes WHERE rowid = ?", (row["rowid"],)
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 0
    assert row["binary_vector"] is None
    assert not beam.conn.in_transaction
    assert (
        "consolidate_to_episodic: vec_episodes insert and memory_embeddings "
        "fallback failed; summary stored FTS-only "
        "(rowid=1, vec_error=RuntimeError, fallback_error=IntegrityError)"
    ) in caplog.text
    assert "private vec failure detail" not in caplog.text
    assert "private json failure detail" not in caplog.text


def test_successful_vec_insert_remains_ann_backed(
    tmp_path, real_vec_embeddings
):
    beam = _new_vec_beam(tmp_path, real_vec_embeddings, session_id="ann")

    memory_id = beam.consolidate_to_episodic("Stored in the ANN index.", [])
    row = beam.conn.execute(
        "SELECT rowid, binary_vector FROM episodic_memory WHERE id = ?", (memory_id,)
    ).fetchone()
    rowid = row["rowid"]

    assert beam.conn.execute(
        "SELECT COUNT(*) FROM vec_episodes WHERE rowid = ?", (rowid,)
    ).fetchone()[0] == 1
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 0
    assert row["binary_vector"] is not None
    assert not beam.conn.in_transaction


def test_vec_sqlite_rollback_preserves_owned_transaction_failure(
    tmp_path, monkeypatch, real_vec_embeddings
):
    beam = _new_vec_beam(tmp_path, real_vec_embeddings, session_id="owned-rollback")
    _install_vec_rollback_failure(beam, monkeypatch)

    with pytest.raises(
        sqlite3.IntegrityError, match="forced vec transaction rollback"
    ):
        beam.consolidate_to_episodic("Rolled back with its owned transaction.", [])

    assert not beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE content = ?",
        ("Rolled back with its owned transaction.",),
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings"
    ).fetchone()[0] == 0


def test_vec_sqlite_rollback_preserves_caller_transaction_failure(
    tmp_path, monkeypatch, real_vec_embeddings
):
    beam = _new_vec_beam(tmp_path, real_vec_embeddings, session_id="caller-rollback")
    _install_vec_rollback_failure(beam, monkeypatch)
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
        (
            "rollback-marker",
            "caller marker",
            "test",
            "2026-01-01T00:00:00",
            "caller-rollback",
        ),
    )
    assert beam.conn.in_transaction

    with pytest.raises(
        sqlite3.IntegrityError, match="forced vec transaction rollback"
    ):
        beam.consolidate_to_episodic(
            "Rolled back with its caller transaction.", [], emit_event=False
        )

    assert not beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = 'rollback-marker'"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE content = ?",
        ("Rolled back with its caller transaction.",),
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings"
    ).fetchone()[0] == 0


def test_fallback_sqlite_rollback_preserves_owned_transaction_failure(
    tmp_path, monkeypatch, real_vec_embeddings, caplog
):
    beam = _new_vec_beam(
        tmp_path, real_vec_embeddings, session_id="owned-fallback-rollback"
    )
    _install_fallback_rollback_failure(beam, monkeypatch)

    with pytest.raises(
        sqlite3.IntegrityError, match="forced fallback transaction rollback"
    ):
        beam.consolidate_to_episodic("Fallback rolled back its owned transaction.", [])

    assert not beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE content = ?",
        ("Fallback rolled back its owned transaction.",),
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE binary_vector IS NOT NULL"
    ).fetchone()[0] == 0
    assert "stored FTS-only" not in caplog.text


def test_fallback_sqlite_rollback_preserves_caller_transaction_failure(
    tmp_path, monkeypatch, real_vec_embeddings, caplog
):
    beam = _new_vec_beam(
        tmp_path, real_vec_embeddings, session_id="caller-fallback-rollback"
    )
    _install_fallback_rollback_failure(beam, monkeypatch)
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
        (
            "fallback-rollback-marker",
            "caller marker",
            "test",
            "2026-01-01T00:00:00",
            "caller-fallback-rollback",
        ),
    )
    assert beam.conn.in_transaction

    with pytest.raises(
        sqlite3.IntegrityError, match="forced fallback transaction rollback"
    ):
        beam.consolidate_to_episodic(
            "Fallback rolled back its caller transaction.", [], emit_event=False
        )

    assert not beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory "
        "WHERE id = 'fallback-rollback-marker'"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE content = ?",
        ("Fallback rolled back its caller transaction.",),
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE binary_vector IS NOT NULL"
    ).fetchone()[0] == 0
    assert "stored FTS-only" not in caplog.text


def test_vec_failure_fallback_does_not_commit_caller_transaction(
    tmp_path, monkeypatch, real_vec_embeddings
):
    beam = _new_vec_beam(tmp_path, real_vec_embeddings, session_id="caller")
    monkeypatch.setattr(
        beam_module,
        "_vec_insert",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("vec failed")),
    )
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
        ("marker", "caller marker", "test", "2026-01-01T00:00:00", "caller"),
    )
    assert beam.conn.in_transaction

    memory_id = beam.consolidate_to_episodic(
        "Caller still owns this transaction.", [], emit_event=False
    )

    assert beam.conn.in_transaction
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 1
    observer = sqlite3.connect(beam.db_path)
    try:
        assert observer.execute(
            "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0] == 0
        assert observer.execute(
            "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0] == 0
    finally:
        observer.close()

    beam.conn.rollback()
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0] == 0
