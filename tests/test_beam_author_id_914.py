"""Regression tests for issue #914 — per-write author stamps.

Covers the two core bugs fixed on the dev branch:
  1. ``BeamMemory.remember()`` accepts per-write ``author_id`` / ``author_type``
     overrides so integrations (Hermes provider surfaces) can stamp rows
     without setting ``beam.author_id`` (keeping recall/prefetch session-
     scoped).
  2. ``consolidate_to_episodic()`` and ``sleep()`` preserve the row-level
     author stamp instead of copying the beam identity: the source SELECT
     fetches author columns and the summary inherits the unanimous
     source-row author (mixed/absent falls back to beam identity).
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _wm_author(db_path, memory_id):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT author_id, author_type FROM working_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        conn.close()


def _ep_author(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT author_id, author_type FROM episodic_memory ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        conn.close()


def _seed_old_wm(beam, rows, ts_offset_hours=200):
    """Insert old working_memory rows with explicit authors (bypasses
    remember() so the write-side path under test is exercised from the
    other side / consolidation only)."""
    conn = sqlite3.connect(str(beam.db_path))
    ts = (datetime.now() - timedelta(hours=ts_offset_hours)).isoformat()
    data = [
        (rid, content, source, ts, beam.session_id, author, author_type)
        for rid, content, source, author, author_type in rows
    ]
    conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, author_id, author_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        data,
    )
    conn.commit()
    conn.close()


class TestConsolidateToEpisodicAuthor:
    def test_consolidation_stamps_explicit_author(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            "summary", ["wm1"], author_id="alice", author_type="human"
        )
        row = tuple(beam.conn.execute(
            "SELECT author_id, author_type FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone())
        assert row == ("alice", "human")

    def test_consolidation_falls_back_to_beam_identity(self, temp_db):
        beam = BeamMemory(
            session_id="s1", db_path=temp_db,
            author_id="carol", author_type="agent",
        )
        eid = beam.consolidate_to_episodic("summary", ["wm1"])
        row = tuple(beam.conn.execute(
            "SELECT author_id, author_type FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone())
        assert row == ("carol", "agent")

    def test_consolidation_kwarg_overrides_beam_identity(self, temp_db):
        beam = BeamMemory(
            session_id="s1", db_path=temp_db,
            author_id="carol", author_type="agent",
        )
        eid = beam.consolidate_to_episodic(
            "summary", ["wm1"], author_id="dave", author_type="human"
        )
        row = tuple(beam.conn.execute(
            "SELECT author_id, author_type FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone())
        assert row == ("dave", "human")


class TestSleepAuthorPreservation:
    def test_sleep_preserves_unanimous_source_author(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(beam, [
            ("a1", "alpha fact", "conversation", "alice", "human"),
            ("a2", "beta fact", "conversation", "alice", "human"),
        ])

        result = beam.sleep()

        assert result["status"] == "consolidated"
        row = tuple(beam.conn.execute(
            "SELECT author_id, author_type FROM episodic_memory ORDER BY rowid DESC LIMIT 1"
        ).fetchone())
        assert row == ("alice", "human")
        # Beam read identity untouched: prefetch stays session-scoped.
        assert beam.author_id is None

    def test_sleep_mixed_authors_fall_back_to_beam_identity(self, temp_db):
        beam = BeamMemory(
            session_id="s1", db_path=temp_db,
            author_id="carol", author_type="agent",
        )
        _seed_old_wm(beam, [
            ("a1", "alpha fact", "conversation", "alice", "human"),
            ("a2", "beta fact", "conversation", "bob", "agent"),
        ])

        beam.sleep()

        assert _ep_author(temp_db) == ("carol", "agent")

    def test_sleep_absent_authors_fall_back_to_beam_identity(self, temp_db):
        beam = BeamMemory(
            session_id="s1", db_path=temp_db,
            author_id="carol", author_type="agent",
        )
        _seed_old_wm(beam, [
            ("a1", "alpha fact", "conversation", None, None),
            ("a2", "beta fact", "conversation", None, None),
        ])

        beam.sleep()

        assert _ep_author(temp_db) == ("carol", "agent")

    def test_sleep_missing_sibling_author_not_attributed_to_present_author(self, temp_db):
        """Regression: a group where one row has an author and a sibling
        row has none must NOT be treated as unanimous — the summary falls
        back to the beam identity instead of being stamped with the
        present row's author."""
        beam = BeamMemory(
            session_id="s1", db_path=temp_db,
            author_id="carol", author_type="agent",
        )
        _seed_old_wm(beam, [
            ("a1", "alpha fact", "conversation", "alice", "human"),
            ("a2", "beta fact", "conversation", None, None),
        ])

        beam.sleep()

        assert _ep_author(temp_db) == ("carol", "agent")

    def test_sleep_missing_sibling_author_type_not_attributed_to_present_type(self, temp_db):
        """author_id unanimous but a sibling row's author_type is absent:
        the type must fall back to the beam identity, not the present
        row's type."""
        beam = BeamMemory(
            session_id="s1", db_path=temp_db,
            author_id="maintenance", author_type="bot",
        )
        _seed_old_wm(beam, [
            ("a1", "alpha fact", "conversation", "alice", "human"),
            ("a2", "beta fact", "conversation", "alice", None),
        ])

        beam.sleep()

        assert _ep_author(temp_db) == ("alice", "bot")

    def test_sleep_mixed_author_type_falls_back_author_id_only(self, temp_db):
        """author_id unanimous but author_type mixed: id is preserved,
        type falls back to beam identity independently."""
        beam = BeamMemory(
            session_id="s1", db_path=temp_db,
            author_id="maintenance", author_type="bot",
        )
        _seed_old_wm(beam, [
            ("a1", "alpha fact", "conversation", "alice", "human"),
            ("a2", "beta fact", "conversation", "alice", "agent"),
        ])

        beam.sleep()

        assert _ep_author(temp_db) == ("alice", "bot")
