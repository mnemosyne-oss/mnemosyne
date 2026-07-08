"""Tests for the memory hygiene module (Layer 2, issue #428).

Covers:
- Noise scoring: terminal output, stack traces, heartbeats, dumps, secrets
- Audit: scanning working_memory + memories tables, ranking candidates
- Cleanup: delete / archive / flag / keep actions, audit log integrity
- Dry-run safety: no modifications without confirm=True
- Reversibility: restore_archived() recovers archived rows
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory, init_beam
from mnemosyne.core.hygiene import (
    AuditReport,
    CleanResult,
    NoiseCandidate,
    audit_noise,
    clean_noise,
    restore_archived,
    _score_noise,
    _suggest_action,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    """Create a temporary Mnemosyne database with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_mnemosyne.db"
        beam = BeamMemory(session_id="test", db_path=db_path)
        init_beam(db_path)

        # Also create the legacy memories table
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                timestamp TEXT,
                session_id TEXT DEFAULT 'default',
                importance REAL DEFAULT 0.5,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

        yield db_path, beam


def _insert_row(beam, table, memory_id, content, source="conversation", importance=0.5, metadata=None):
    """Insert a row directly into a table."""
    conn = beam.conn
    meta_json = json.dumps(metadata or {})
    conn.execute(
        f"INSERT INTO {table} (id, content, source, timestamp, session_id, importance, metadata_json) "
        f"VALUES (?, ?, ?, ?, ?, ?, ?)",
        (memory_id, content, source, "2025-01-01T00:00:00", "test", importance, meta_json),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# _score_noise
# ---------------------------------------------------------------------------

class TestScoreNoise:
    def test_empty_content(self):
        score, reasons = _score_noise("", 0.5, "")
        assert score == 1.0
        assert "empty_content" in reasons

    def test_terminal_output(self):
        score, reasons = _score_noise("$ pip install foo\nCollecting foo", 0.5, "terminal")
        assert score >= 0.7
        assert "terminal_output" in reasons or "noise_pattern_match" in reasons

    def test_stack_trace(self):
        content = "Traceback (most recent call last):\n  File \"test.py\", line 10"
        score, reasons = _score_noise(content, 0.5, "")
        assert score >= 0.8
        assert "stack_trace" in reasons

    def test_heartbeat(self):
        score, reasons = _score_noise("heartbeat", 0.5, "heartbeat")
        assert score >= 0.7
        assert "trivial_keyword" in reasons or "noisy_source" in reasons

    def test_secret(self):
        score, reasons = _score_noise("password = hunter2supersecret", 0.5, "")
        assert score >= 0.9
        assert any("secret" in r for r in reasons)

    def test_valuable_content(self):
        score, reasons = _score_noise("User prefers concise responses in English.", 0.7, "conversation")
        assert score < 0.5

    def test_low_importance_penalty(self):
        score, reasons = _score_noise("some content", 0.1, "")
        assert score >= 0.5
        assert "low_importance" in reasons

    def test_value_keywords_reduce(self):
        content = "The user prefers using pytest. This is a stable project convention."
        score, reasons = _score_noise(content, 0.5, "")
        assert "value_keyword_present" in reasons
        assert score <= 0.3

    def test_large_dump(self):
        # 60 lines of non-sentence content, >1000 chars total
        content = "\n".join(["some random data line that is long enough"] * 60)
        score, reasons = _score_noise(content, 0.5, "")
        assert score >= 0.6
        assert "likely_dump" in reasons


# ---------------------------------------------------------------------------
# _suggest_action
# ---------------------------------------------------------------------------

class TestSuggestAction:
    def test_high_score_suggests_delete(self):
        assert _suggest_action(0.85, []) == "delete"

    def test_medium_score_suggests_archive(self):
        assert _suggest_action(0.6, []) == "archive"

    def test_low_score_keeps(self):
        assert _suggest_action(0.2, []) == "keep"

    def test_secrets_always_flag(self):
        assert _suggest_action(0.95, ["api_key_prefix"]) == "flag"


# ---------------------------------------------------------------------------
# audit_noise
# ---------------------------------------------------------------------------

class TestAuditNoise:
    def test_audit_finds_noise(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "noise1", "$ pip install foo\nCollecting foo", source="terminal")
        _insert_row(beam, "working_memory", "val1", "User prefers concise responses in English.", importance=0.7)
        _insert_row(beam, "working_memory", "noise2", "heartbeat", source="heartbeat")

        report = audit_noise(db_path=db_path, limit=100, min_score=0.3)

        assert report.total_scanned == 3
        assert len(report.candidates) >= 2
        # Highest score first
        assert report.candidates[0].noise_score >= report.candidates[-1].noise_score
        assert "working_memory" in report.tables_scanned

    def test_audit_finds_secrets(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "secret1", "password = hunter2supersecret")

        report = audit_noise(db_path=db_path, min_score=0.0)

        assert len(report.candidates) == 1
        assert len(report.candidates[0].secret_flags) > 0
        assert report.candidates[0].suggested_action == "flag"
        assert report.summary["with_secrets"] == 1

    def test_audit_scans_memories_table(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "memories", "legacy_noise", "ok", source="conversation")

        report = audit_noise(db_path=db_path, min_score=0.0)

        assert len(report.candidates) == 1
        assert report.candidates[0].table_name == "memories"

    def test_audit_min_score_filter(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "val1", "User prefers pytest. This is a project convention.", importance=0.8)
        _insert_row(beam, "working_memory", "noise1", "heartbeat", source="heartbeat")

        report = audit_noise(db_path=db_path, min_score=0.6)

        # Value content should be filtered out by min_score
        assert all(c.noise_score >= 0.6 or c.secret_flags for c in report.candidates)

    def test_audit_nonexistent_table_skipped(self, temp_db):
        db_path, beam = temp_db
        report = audit_noise(db_path=db_path, tables=["nonexistent_table"])
        assert report.total_scanned == 0
        assert report.candidates == []

    def test_audit_report_serializable(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat")
        report = audit_noise(db_path=db_path, min_score=0.0)
        d = report.to_dict()
        assert "candidates" in d
        assert "summary" in d
        json.dumps(d)  # should not raise


# ---------------------------------------------------------------------------
# clean_noise
# ---------------------------------------------------------------------------

class TestCleanNoise:
    def test_dry_run_no_changes(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat", source="heartbeat")

        candidates = [NoiseCandidate(
            memory_id="n1", table_name="working_memory",
            content_preview="heartbeat", noise_score=0.8,
            noise_reasons=["trivial_keyword"], suggested_action="delete",
        )]

        result = clean_noise(db_path, candidates, action="delete", dry_run=True)
        assert result.deleted == 1

        # Verify row still exists
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM working_memory WHERE id = 'n1'")
        assert cursor.fetchone()[0] == 1
        conn.close()

    def test_no_confirm_returns_error(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat")

        candidates = [NoiseCandidate(
            memory_id="n1", table_name="working_memory",
            content_preview="heartbeat", noise_score=0.8,
            noise_reasons=["trivial_keyword"], suggested_action="delete",
        )]

        result = clean_noise(db_path, candidates, action="delete", confirm=False, dry_run=False)
        assert len(result.errors) > 0
        assert "confirm" in result.errors[0].lower()

    def test_delete_with_confirm(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat")

        candidates = [NoiseCandidate(
            memory_id="n1", table_name="working_memory",
            content_preview="heartbeat", noise_score=0.8,
            noise_reasons=["trivial_keyword"], suggested_action="delete",
        )]

        result = clean_noise(db_path, candidates, action="delete", confirm=True, dry_run=False)
        assert result.deleted == 1
        assert result.log_entries == 1

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM working_memory WHERE id = 'n1'")
        assert cursor.fetchone()[0] == 0
        # Audit log written
        cursor = conn.execute("SELECT COUNT(*) FROM hygiene_audit_log WHERE action = 'deleted'")
        assert cursor.fetchone()[0] == 1
        conn.close()

    def test_archive_with_confirm(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat", importance=0.5)

        candidates = [NoiseCandidate(
            memory_id="n1", table_name="working_memory",
            content_preview="heartbeat", noise_score=0.6,
            noise_reasons=["trivial_keyword"], suggested_action="archive",
        )]

        result = clean_noise(db_path, candidates, action="archive", confirm=True, dry_run=False)
        assert result.archived == 1

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT importance, metadata_json FROM working_memory WHERE id = 'n1'")
        row = cursor.fetchone()
        assert row[0] == 0  # importance decayed to 0
        meta = json.loads(row[1])
        assert meta.get("_archived") is True
        conn.close()

    def test_flag_with_confirm(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "s1", "password = hunter2supersecret")

        candidates = [NoiseCandidate(
            memory_id="s1", table_name="working_memory",
            content_preview="password = ...", noise_score=0.9,
            noise_reasons=["secret_detected"], secret_flags=["secret_assignment"],
            suggested_action="flag",
        )]

        result = clean_noise(db_path, candidates, action="flag", confirm=True, dry_run=False)
        assert result.flagged == 1

        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT metadata_json FROM working_memory WHERE id = 's1'")
        meta = json.loads(cursor.fetchone()[0])
        assert meta.get("_hygiene_flagged") is True
        conn.close()

    def test_missing_row_logs_error(self, temp_db):
        db_path, beam = temp_db

        candidates = [NoiseCandidate(
            memory_id="nonexistent", table_name="working_memory",
            content_preview="", noise_score=0.5,
            noise_reasons=["test"], suggested_action="delete",
        )]

        result = clean_noise(db_path, candidates, action="delete", confirm=True, dry_run=False)
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()

    def test_uses_suggested_action_when_action_keep(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat")
        _insert_row(beam, "working_memory", "s1", "password = hunter2supersecret")

        candidates = [
            NoiseCandidate(memory_id="n1", table_name="working_memory",
                           content_preview="heartbeat", noise_score=0.8,
                           noise_reasons=["trivial"], suggested_action="delete"),
            NoiseCandidate(memory_id="s1", table_name="working_memory",
                           content_preview="password", noise_score=0.9,
                           noise_reasons=["secret"], secret_flags=["secret_assignment"],
                           suggested_action="flag"),
        ]

        result = clean_noise(db_path, candidates, action="keep", confirm=True, dry_run=False)
        assert result.deleted == 1
        assert result.flagged == 1


# ---------------------------------------------------------------------------
# restore_archived
# ---------------------------------------------------------------------------

class TestRestoreArchived:
    def test_restore_recovers_archived_row(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat", importance=0.5,
                    metadata={"original": "data"})

        candidates = [NoiseCandidate(
            memory_id="n1", table_name="working_memory",
            content_preview="heartbeat", noise_score=0.6,
            noise_reasons=["trivial"], suggested_action="archive",
        )]

        # Archive it
        clean_noise(db_path, candidates, action="archive", confirm=True, dry_run=False)

        # Verify archived
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT importance FROM working_memory WHERE id = 'n1'")
        assert cursor.fetchone()[0] == 0

        # Restore
        restored = restore_archived(db_path)
        assert restored >= 1

        # Verify restored
        cursor = conn.execute("SELECT importance, metadata_json FROM working_memory WHERE id = 'n1'")
        row = cursor.fetchone()
        assert row[0] == 0.5  # importance restored
        meta = json.loads(row[1])
        assert "_archived" not in meta
        assert meta.get("original") == "data"
        conn.close()
