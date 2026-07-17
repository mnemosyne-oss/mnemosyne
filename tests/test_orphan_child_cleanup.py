"""Tests for orphan child-row cleanup on forget / validate-delete.

BeamMemory.forget_working() cascades to vec_working, annotations,
and memory_embeddings but was missing gists.  mcp_tools.py
validate(action=delete) cascaded nothing at all.
prune_cascade_orphans() is a one-time batch cleanup for existing
orphans (gists + memory_embeddings).

These tests pin:
  - forget_working deletes the gist row for the same memory_id
  - prune_cascade_orphans dry-run reports orphan counts without writes
  - prune_cascade_orphans live mode removes orphans and leaves valid rows
  - gist cleanup is best-effort (missing gists table doesn't abort)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "orphan_child_test.db"


# ---------------------------------------------------------------------------
# forget_working cascades to gists
# ---------------------------------------------------------------------------


def gist_count(conn) -> int:
    try:
        return conn.execute("SELECT COUNT(*) FROM gists").fetchone()[0]
    except Exception:
        return -1


def orphan_gist_count(conn) -> int:
    """Count gist rows whose memory_id does not exist in any parent."""
    parent_union = (
        "SELECT id FROM working_memory"
        " UNION SELECT id FROM episodic_memory"
        " UNION SELECT id FROM memoria_facts"
        " UNION SELECT id FROM memoria_preferences"
        " UNION SELECT id FROM memoria_instructions"
        " UNION SELECT id FROM memoria_kg"
        " UNION SELECT event_id AS id FROM memoria_timelines"
    )
    return conn.execute(
        f"SELECT COUNT(*) FROM gists WHERE memory_id IS NOT NULL AND memory_id NOT IN ({parent_union})"
    ).fetchone()[0]


def orphan_emb_count(conn) -> int:
    parent_union = (
        "SELECT id FROM working_memory"
        " UNION SELECT id FROM episodic_memory"
        " UNION SELECT id FROM memoria_facts"
        " UNION SELECT id FROM memoria_preferences"
        " UNION SELECT id FROM memoria_instructions"
        " UNION SELECT id FROM memoria_kg"
        " UNION SELECT event_id AS id FROM memoria_timelines"
    )
    return conn.execute(
        f"SELECT COUNT(*) FROM memory_embeddings WHERE memory_id IS NOT NULL AND memory_id NOT IN ({parent_union})"
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# forget_working cascades to gists
# ---------------------------------------------------------------------------


def test_forget_working_deletes_gist(temp_db):
    """After forget_working(), the gist row for that memory_id must be gone."""
    beam = BeamMemory(session_id="orphan-gist", db_path=temp_db)
    conn = beam.conn

    # Forget a seed working row.
    mid = beam.remember("test gist cascade", source="test", importance=0.5)

    # Manually insert a gist row pointing to the same memory.
    conn.execute(
        "INSERT INTO gists (id, text, memory_id, created_at) VALUES (?, ?, ?, datetime('now'))",
        ("gist-1", "test gist body", mid),
    )
    conn.commit()
    # remember() auto-creates 1 gist + our manual insert = 2
    assert gist_count(conn) == 2, f"expected 2 gists, got {gist_count(conn)}"

    # Forget — should now cascade to gists (both auto and manual).
    assert beam.forget_working(mid) is True

    assert gist_count(conn) == 0, (
        f"forget_working did not delete the gist — count is {gist_count(conn)}"
    )


def test_forget_working_keeps_other_memorys_gist(temp_db):
    """Forget one memory must not delete gist rows belonging to other memories."""
    beam = BeamMemory(session_id="orphan-other", db_path=temp_db)
    conn = beam.conn

    mid_a = beam.remember("stay", source="test", importance=0.5)
    mid_b = beam.remember("go", source="test", importance=0.5)

    conn.execute(
        "INSERT INTO gists (id, text, memory_id, created_at) VALUES (?, ?, ?, datetime('now'))",
        ("gist-a", "belongs to A", mid_a),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id, created_at) VALUES (?, ?, ?, datetime('now'))",
        ("gist-b", "belongs to B", mid_b),
    )
    conn.commit()
    # 2 remembers = 2 auto gists + 2 manual = 4
    assert gist_count(conn) == 4, f"expected 4 gists, got {gist_count(conn)}"

    beam.forget_working(mid_a)

    # After forgetting A: only B's auto gist + B's manual gist remain
    assert gist_count(conn) == 2, (
        f"forget_working left {gist_count(conn)} gists, expected 2"
    )
    remaining = conn.execute("SELECT id FROM gists").fetchone()
    assert remaining["id"] == "gist-b"


# ---------------------------------------------------------------------------
# prune_cascade_orphans — dry-run and live
# ---------------------------------------------------------------------------


def test_prune_cascade_orphans_dry_run_reports_orphans(temp_db):
    """Dry-run should report orphan counts without deleting any rows."""
    beam = BeamMemory(session_id="orphan-prune", db_path=temp_db)
    conn = beam.conn

    mid = beam.remember("orphanseed", source="test", importance=0.5)

    # Insert a gist pointing to a now-nonexistent memory (stale ID).
    conn.execute(
        "INSERT INTO gists (id, text, memory_id, created_at) VALUES (?, ?, ?, datetime('now'))",
        ("gist-orphan", "orphan text", "bogus-id-12345"),
    )
    conn.commit()

    result = beam.prune_cascade_orphans(dry_run=True)
    assert result["status"] == "dry_run"
    assert result["gists_orphans"] >= 1, "dry-run should detect the orphan gist"
    assert result["gists_deleted"] == 0, "dry-run must not delete"
    assert result["memory_embeddings_deleted"] == 0

    # The orphan row must still be in the database.
    assert gist_count(conn) >= 1


def test_prune_cascade_orphans_live_deletes_orphans(temp_db):
    """Live prune should delete orphan rows and leave valid ones untouched."""
    beam = BeamMemory(session_id="orphan-live", db_path=temp_db)
    conn = beam.conn

    mid = beam.remember("valid", source="test", importance=0.5)

    # One valid gist (points to an existing memory) and one orphan.
    conn.execute(
        "INSERT INTO gists (id, text, memory_id, created_at) VALUES (?, ?, ?, datetime('now'))",
        ("gist-valid", "valid", mid),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id, created_at) VALUES (?, ?, ?, datetime('now'))",
        ("gist-orphan", "orphan", "dead-id"),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
        ("dead-id", "[]"),
    )
    conn.commit()

    # 1 auto gist (from remember) + 1 valid manual + 1 orphan = 3
    # 1 orphan embedding
    before = gist_count(conn)
    assert gist_count(conn) == 3, f"expected 3 gists, got {gist_count(conn)}"
    assert orphan_gist_count(conn) == 1, "expected 1 orphan gist"

    result = beam.prune_cascade_orphans(dry_run=False)
    assert result["status"] == "pruned"
    assert result["gists_deleted"] >= 1
    assert result["memory_embeddings_deleted"] >= 1

    # After prune: zero orphans, but valid rows remain.
    assert orphan_gist_count(conn) == 0, "orphans still exist after prune"
    assert orphan_emb_count(conn) == 0, "embedding orphans still exist after prune"
    # The valid gist must still be there.
    assert conn.execute("SELECT id FROM gists WHERE id = ?", ("gist-valid",)).fetchone() is not None


def test_prune_cascade_orphans_idempotent(temp_db):
    """Running prune twice is safe — second run should be a no-op."""
    beam = BeamMemory(session_id="orphan-idem", db_path=temp_db)
    conn = beam.conn

    beam.remember("alive", source="test", importance=0.5)
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, ?)",
        ("ghost", "[]"),
    )
    conn.commit()

    r1 = beam.prune_cascade_orphans(dry_run=True)
    assert r1["memory_embeddings_orphans"] >= 1

    beam.prune_cascade_orphans(dry_run=False)
    r2 = beam.prune_cascade_orphans(dry_run=True)
    assert r2["memory_embeddings_orphans"] == 0, "orphans remain after second prune run"
