"""
Tests for get_context() consolidated working-memory exclusion.

Covers the default exclusion of consolidated rows and the
MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED compatibility override.
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


class TestGetContextConsolidatedExclusion:
    def test_get_context_includes_unconsolidated_working_row(self, temp_db, monkeypatch):
        """Unconsolidated working rows (consolidated_at IS NULL) appear in get_context()."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        now = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
            "VALUES (?, ?, 'conversation', ?, ?, ?)",
            ("unconsol-1", "hot memory", now, "s1", 0.9),
        )
        beam.conn.commit()

        monkeypatch.delenv("MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED", raising=False)

        results = beam.get_context(limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "unconsol-1"

    def test_get_context_excludes_consolidated_working_row_by_default(self, temp_db, monkeypatch):
        """Consolidated working rows (consolidated_at IS NOT NULL) are excluded from get_context() by default."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        now = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
            "VALUES (?, ?, 'conversation', ?, ?, ?)",
            ("unconsol-1", "hot memory", now, "s1", 0.9),
        )
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, consolidated_at) "
            "VALUES (?, ?, 'conversation', ?, ?, ?, ?)",
            ("consol-1", "old memory", now, "s1", 0.8, now),
        )
        beam.conn.commit()

        monkeypatch.delenv("MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED", raising=False)

        results = beam.get_context(limit=10)
        assert len(results) == 1
        assert results[0]["id"] == "unconsol-1"
        ids = [r["id"] for r in results]
        assert "consol-1" not in ids

    def test_get_context_includes_consolidated_when_env_override_enabled(self, temp_db, monkeypatch):
        """When MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED=1, consolidated rows appear in get_context()."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        now = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
            "VALUES (?, ?, 'conversation', ?, ?, ?)",
            ("unconsol-1", "hot memory", now, "s1", 0.9),
        )
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, consolidated_at) "
            "VALUES (?, ?, 'conversation', ?, ?, ?, ?)",
            ("consol-1", "old memory", now, "s1", 0.8, now),
        )
        beam.conn.commit()

        monkeypatch.setenv("MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED", "1")

        results = beam.get_context(limit=10)
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert "unconsol-1" in ids
        assert "consol-1" in ids

    def test_recall_still_finds_consolidated_working_row(self, temp_db, monkeypatch):
        """recall() can still find consolidated working-memory rows (provenance preserved)."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        now = datetime.now().isoformat()
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, consolidated_at) "
            "VALUES (?, ?, 'conversation', ?, ?, ?, ?)",
            ("consol-recall", "unique marker zorblax recall test", now, "s1", 0.7, now),
        )
        beam.conn.commit()

        monkeypatch.delenv("MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED", raising=False)

        context = beam.get_context(limit=10)
        context_ids = [r["id"] for r in context]
        assert "consol-recall" not in context_ids, "get_context should exclude consolidated rows by default"

        results = beam.recall("zorblax", top_k=10)
        found = [r for r in results if "zorblax" in (r.get("content") or "").lower()]
        assert len(found) >= 1, (
            f"recall('zorblax') should find the consolidated working row. "
            f"Got {len(results)} results: {[r.get('id') for r in results]}"
        )
