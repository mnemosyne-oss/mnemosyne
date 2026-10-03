"""Regression coverage for #782 gist cleanup in authorized forget cascades."""

from pathlib import Path

import pytest

from mnemosyne.batch_tool import apply_beam_batch, validate_batch_operations
from mnemosyne.core.beam import BeamMemory


def _beam(tmp_path: Path, session_id: str = "owner") -> BeamMemory:
    return BeamMemory(session_id=session_id, db_path=tmp_path / "forget-gists.db")


def _seed_gist(beam: BeamMemory, memory_id: str, suffix: str = "target") -> None:
    beam.conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, ?, ?)",
        (f"gist-{suffix}-{memory_id}", f"gist for {suffix}", memory_id),
    )
    beam.conn.commit()


def _gist_count(beam: BeamMemory, memory_id: str) -> int:
    return beam.conn.execute(
        "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (memory_id,)
    ).fetchone()[0]


def test_direct_forget_deletes_only_authorized_target_gists(tmp_path: Path):
    beam = _beam(tmp_path)
    delete_id = beam.remember("#782 delete target")
    keep_id = beam.remember("#782 keep target")
    _seed_gist(beam, delete_id, "delete-first")
    _seed_gist(beam, delete_id, "delete-second")
    _seed_gist(beam, keep_id, "keep")
    expected_keep_gists = _gist_count(beam, keep_id)

    assert beam.forget_working(delete_id) is True

    assert _gist_count(beam, delete_id) == 0
    assert _gist_count(beam, keep_id) == expected_keep_gists
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (delete_id,)
    ).fetchone()[0] == 0


def test_batch_forget_uses_the_same_gist_cascade(tmp_path: Path):
    beam = _beam(tmp_path)
    memory_id = beam.remember("#782 batch target")
    _seed_gist(beam, memory_id, "batch")

    result = apply_beam_batch(
        beam,
        validate_batch_operations([{"action": "forget", "memory_id": memory_id}]),
    )

    assert result["status"] == "ok"
    assert _gist_count(beam, memory_id) == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 0


def test_forget_succeeds_when_optional_gists_table_is_absent(tmp_path: Path):
    beam = _beam(tmp_path)
    memory_id = beam.remember("#782 no gists table")
    beam.conn.execute("DROP TABLE gists")
    beam.conn.commit()

    assert beam.forget_working(memory_id) is True
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 0


def test_foreign_session_forget_keeps_private_memory_and_gist(tmp_path: Path):
    owner = _beam(tmp_path, "owner")
    memory_id = owner.remember("#782 private target")
    _seed_gist(owner, memory_id, "private")
    expected_gists = _gist_count(owner, memory_id)
    foreign = _beam(tmp_path, "foreign")

    assert foreign.forget_working(memory_id) is False

    assert _gist_count(owner, memory_id) == expected_gists
    assert owner.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 1


def test_gist_delete_failure_rolls_back_the_entire_forget_cascade(tmp_path: Path):
    beam = _beam(tmp_path)
    memory_id = beam.remember("#782 rollback target")
    _seed_gist(beam, memory_id, "rollback")
    expected_gists = _gist_count(beam, memory_id)
    beam.conn.execute(
        "CREATE TRIGGER fail_gist_delete BEFORE DELETE ON gists "
        "BEGIN SELECT RAISE(ABORT, 'forced gist failure'); END"
    )
    beam.conn.commit()

    with pytest.raises(Exception, match="forced gist failure"):
        beam.forget_working(memory_id)

    assert _gist_count(beam, memory_id) == expected_gists
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 1
