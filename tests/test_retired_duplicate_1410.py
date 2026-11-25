"""Exact-content deduplication only reuses active working memory."""

from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.memory import Mnemosyne


@pytest.mark.parametrize("retirement", ["invalidate", "supersede", "expire"])
def test_remember_reasserts_retired_content_as_new_active_row(tmp_path, retirement):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    first = memory.remember(content, importance=0.9)
    assert first is not None
    assert any(hit["id"] == first for hit in memory.recall("metric units"))

    replacement = None
    if retirement == "supersede":
        replacement = memory.remember("Prefers replies in imperial units")
        assert replacement is not None
        assert memory.invalidate(first, replacement_id=replacement)
    elif retirement == "invalidate":
        assert memory.invalidate(first)
    else:
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        memory.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?", (past, first)
        )
        memory.conn.commit()

    old = memory.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (first,)
    ).fetchone()
    second = memory.remember(content, importance=0.9)
    rows = memory.conn.execute(
        "SELECT id, valid_until, superseded_by FROM working_memory "
        "WHERE session_id = ? AND content = ? ORDER BY rowid", ("s1", content)
    ).fetchall()
    assert second != first
    assert [row["id"] for row in rows] == [first, second]
    assert (rows[0]["valid_until"], rows[0]["superseded_by"]) == tuple(old)
    assert rows[1]["valid_until"] is None
    assert rows[1]["superseded_by"] is None
    assert any(hit["id"] == second for hit in memory.recall("metric units"))
    assert all(hit["id"] != first for hit in memory.recall("metric units"))

    # A further identical write still deduplicates against the active version.
    assert memory.remember(content) == second
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE session_id = ? AND content = ?",
        ("s1", content),
    ).fetchone()[0] == 2


def test_duplicate_control_and_opt_out(tmp_path):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    first = memory.remember(content)
    assert memory.remember(content, memory_id="caller-provided") == first
    second = memory.remember(content, dedupe=False)
    assert second not in {None, first}
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE session_id = ? AND content = ?",
        ("s1", content),
    ).fetchone()[0] == 2


def test_caller_provided_id_used_for_reassertion(tmp_path):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    old_id = memory.remember(content)
    assert old_id is not None
    assert memory.invalidate(old_id)
    assert memory.remember(content, memory_id="new-external-id") == "new-external-id"
    assert memory.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", ("new-external-id",)
    ).fetchone()[0] is None


def test_future_expiration_remains_eligible_for_deduplication(tmp_path):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    original = memory.remember(content, valid_until=future)
    assert original is not None
    assert memory.remember(content) == original
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?", (content,)
    ).fetchone()[0] == 1


def test_wrapper_reassertion_writes_new_legacy_id(tmp_path):
    memory = Mnemosyne(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    original = memory.remember(content)
    assert original is not None
    assert memory.invalidate(original)
    restored = memory.remember(content)
    assert restored not in {None, original}
    assert memory.conn.execute(
        "SELECT id FROM memories WHERE id = ?", (restored,)
    ).fetchone()[0] == restored
    assert memory.beam.conn.execute(
        "SELECT id FROM working_memory WHERE id = ? AND superseded_by IS NULL",
        (restored,),
    ).fetchone()[0] == restored
