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

from mnemosyne.cli import cmd_hygiene
from mnemosyne.core.beam import BeamMemory, init_beam
from mnemosyne.core.filters import SECRET_LABELED_PATTERNS
from mnemosyne.core.hygiene import (
    AuditReport,
    CleanResult,
    NoiseCandidate,
    audit_noise,
    clean_noise,
    doctor_hygiene_summary,
    hygiene_status,
    noise_summary,
    restore_archived,
    _score_noise,
    _suggest_action,
)
from mnemosyne.doctor import (
    DoctorReport,
    HygieneSummaryAdapter,
    STATUS_OK,
    build_doctor_report,
    open_readonly_doctor_db,
    safe_preview,
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
        # nosec - test fixture
        score, reasons = _score_noise("password = hunter2supersecret", 0.5, "")
        assert score >= 0.9
        assert any("secret" in r for r in reasons)

    def test_secret_with_value_keyword_not_dampened(self):
        """Secret + value keyword should NOT dampen the score."""
        # nosec - test fixture
        # Use content that triggers both the secret pattern (password = ...)
        # and a value keyword ("prefer") — secret should win.
        content = "User prefers the password = hunter2supersecret for access"
        score, reasons = _score_noise(content, 0.5, "")
        assert score >= 0.9  # secret wins, not dampened
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
    @pytest.mark.parametrize(
        ("label", "secret", "sensitive_prefix"),
        [
            ("api_key_prefix", "sk-" + "a" * 20, "sk-"),
            ("aws_access_key", "AKIA" + "A" * 16, "AKIA"),
            ("github_token", "ghp_" + "a" * 36, "ghp_"),
            ("slack_token", "xoxr-" + "a" * 20, "xoxr-"),
            ("google_api_key", "AIza" + "a" * 35, "AIza"),
            ("jwt_token", "eyJaaa.eyJbbb.ccc", "eyJ"),
            ("secret_assignment", "passwd=supersecretvalue123", "supersecret"),
            ("secret_assignment", "pwd=supersecretvalue123", "supersecret"),
            ("secret_assignment", "access_key=supersecretvalue123", "supersecret"),
            (
                "private_key_block",
                "-----BEGIN RSA PRIVATE KEY-----\nMIIJKQIBAA",
                "MIIJK",
            ),
            (
                "connection_string_with_credentials",
                "postgres://alice:***@localhost/db",
                "postgres://alice:",
            ),
            ("env_secret_assignment", "DB_PASS=supersecretvalue123", "supersecret"),
        ],
    )
    def test_safe_preview_redacts_all_canonical_hygiene_secrets_before_truncating(
        self, label, secret, sensitive_prefix
    ):
        prefix = " " * 110 if label == "env_secret_assignment" else "x" * 110 + " "
        preview = safe_preview(prefix + secret, max_length=120)
        payload = json.dumps(
            DoctorReport(
                bank_name="test",
                hygiene_summary={"candidates": [{"preview": preview}]},
            ).to_dict()
        )

        assert label in {name for name, _pattern in SECRET_LABELED_PATTERNS}
        assert secret not in payload
        assert sensitive_prefix not in payload
        assert "<redact" in payload

    def test_safe_preview_redacts_a_secret_before_truncating(self):
        # If truncation happened first, the secret value would partially survive.
        raw_secret = "redaction-before-truncation-secret"  # nosec - regression fixture
        preview = safe_preview("x" * 90 + f" password={raw_secret}" + " trailing", max_length=120)

        assert len(preview) <= 120
        assert raw_secret not in preview
        assert "password=<redacted>" in preview

    def test_doctor_hygiene_redacts_cross_boundary_values_before_truncating(self, tmp_path):
        db_path = tmp_path / "cross-boundary-hygiene.db"
        email = "crossboundary-email-" + "a" * 85 + "@example.test"
        secret = "crossboundary-secret-" + "b" * 80  # nosec - regression fixture
        content = "x" * 110 + f" {email} password={secret}"
        writable = sqlite3.connect(db_path)
        writable.execute(
            """
            CREATE TABLE working_memory (
                id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT,
                session_id TEXT, importance REAL, metadata_json TEXT
            )
            """
        )
        writable.execute(
            "INSERT INTO working_memory VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("candidate", content, "test", "2026-01-01", "s", 0.5, "{}"),
        )
        writable.commit()
        writable.close()

        readonly = open_readonly_doctor_db(db_path)
        try:
            summary = doctor_hygiene_summary(
                db_path, conn=readonly, min_score=0.0, candidate_limit=1
            )
        finally:
            readonly.close()

        payload = json.dumps(summary)
        assert summary["candidates"][0]["preview"] == "x" * 110 + " <redacte…"
        assert len(summary["candidates"][0]["preview"]) <= 120
        for sensitive in (email, secret, "crossboundary-email", "crossboundary-secret"):
            assert sensitive not in payload

    def test_noise_summary_uses_supplied_readonly_connector_without_db_changes(self, tmp_path):
        db_path = tmp_path / "readonly-hygiene.db"
        fixture_secret = "fixture-hygiene-secret-123456"  # nosec - redaction fixture
        writable = sqlite3.connect(db_path)
        writable.execute(
            """
            CREATE TABLE working_memory (
                id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT,
                session_id TEXT, importance REAL, metadata_json TEXT
            )
            """
        )
        writable.execute(
            "INSERT INTO working_memory VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("candidate", f"password = {fixture_secret}", "test", "2026-01-01", "s", 0.5, "{}"),
        )
        writable.commit()
        writable.close()

        readonly = open_readonly_doctor_db(db_path)
        try:
            summary = doctor_hygiene_summary(
                db_path, conn=readonly, min_score=0.0, candidate_limit=1
            )
            with pytest.raises(sqlite3.OperationalError):
                readonly.execute("INSERT INTO working_memory (id) VALUES ('forbidden')")
        finally:
            readonly.close()

        verify = sqlite3.connect(db_path)
        try:
            assert verify.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 1
            assert verify.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'hygiene_audit_log'"
            ).fetchone()[0] == 0
        finally:
            verify.close()
        assert summary["status"] == "ok"
        assert summary["with_secrets"] == 1
        assert fixture_secret not in json.dumps(summary)
        assert summary["candidates"][0]["preview"] == "password=<redacted>"
        assert "id" not in summary["candidates"][0]

    def test_hygiene_adapter_whitelists_bounded_safe_data(self, monkeypatch, tmp_path):
        db_path = tmp_path / "doctor.db"
        sqlite3.connect(db_path).close()
        raw_secret = "adapter-private-secret"  # nosec - regression fixture

        def unsafe_summary(*_args, **_kwargs):
            return {
                "status": "ok",
                "total_scanned": 1,
                "total_candidates": 1,
                "with_secrets": 1,
                "candidates": [{
                    "id": "candidate",
                    "table": "working_memory",
                    "noise_score": 0.9,
                    "reasons": ["secret_detected"],
                    "secret_flags": ["secret_assignment"],
                    "suggested_action": "flag",
                    "preview": f"password={raw_secret}",
                    "content": raw_secret,
                    "body": raw_secret,
                    "embedding_json": raw_secret,
                    "metadata": {"private": raw_secret},
                }],
            }

        monkeypatch.setattr("mnemosyne.core.hygiene.doctor_hygiene_summary", unsafe_summary)
        conn = open_readonly_doctor_db(db_path)
        try:
            summary = HygieneSummaryAdapter(conn, db_path, candidate_limit=1).inspect().metrics
        finally:
            conn.close()

        payload = json.dumps(summary)
        assert raw_secret not in payload
        assert "password=<redacted>" in payload
        for forbidden in ("id", "content", "body", "embedding_json", "metadata"):
            assert forbidden not in summary["candidates"][0]

    def test_build_doctor_report_never_constructs_mnemosyne(self, tmp_path, monkeypatch):
        db_path = tmp_path / "doctor.db"
        writable = sqlite3.connect(db_path)
        writable.execute(
            "CREATE TABLE working_memory (id TEXT, content TEXT, source TEXT, timestamp TEXT, "
            "session_id TEXT, importance REAL, metadata_json TEXT)"
        )
        writable.commit()
        writable.close()

        class ForbiddenMnemosyne:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("Doctor must not construct Mnemosyne")

        monkeypatch.setattr("mnemosyne.core.memory.Mnemosyne", ForbiddenMnemosyne)
        report = build_doctor_report("work", db_path, scan_limit=1, candidate_limit=1)

        assert report.bank_name == "work"
        assert report.hygiene_summary["status"] == STATUS_OK

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

    def test_audit_scans_episodic_memory_by_default(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "episodic_memory", "ep_noise", "heartbeat", source="heartbeat")

        report = audit_noise(db_path=db_path, min_score=0.3)

        assert "episodic_memory" in report.tables_scanned
        assert report.summary["table_counts"]["episodic_memory"] == 1
        assert any(c.table_name == "episodic_memory" for c in report.candidates)

    def test_audit_offset_and_scan_all(self, temp_db):
        db_path, beam = temp_db
        for idx in range(3):
            _insert_row(
                beam,
                "working_memory",
                f"noise{idx}",
                f"heartbeat page marker {idx}",
                source="heartbeat",
            )

        paged = audit_noise(db_path=db_path, limit=1, offset=1, tables=["working_memory"], min_score=0.3)
        full = audit_noise(
            db_path=db_path,
            limit=1,
            tables=["working_memory"],
            min_score=0.3,
            scan_all=True,
            batch_size=2,
        )

        assert paged.total_scanned == 1
        assert [c.memory_id for c in paged.candidates] == ["noise1"]
        assert full.total_scanned == 3
        assert {c.memory_id for c in full.candidates} == {"noise0", "noise1", "noise2"}

    def test_audit_noise_rejects_invalid_pagination_args(self, temp_db):
        db_path, _beam = temp_db

        with pytest.raises(ValueError, match="limit must be >= 0"):
            audit_noise(db_path=db_path, limit=-1)
        with pytest.raises(ValueError, match="offset must be >= 0"):
            audit_noise(db_path=db_path, offset=-1)
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            audit_noise(db_path=db_path, batch_size=0)

    @pytest.mark.parametrize(
        ("args", "message"),
        [
            (["audit", "--limit"], "--limit requires a value"),
            (["audit", "--limit", "--json"], "--limit requires a value"),
            (["audit", "--offset"], "--offset requires a value"),
            (["audit", "--batch-size"], "--batch-size requires a value"),
            (["audit", "--min-score"], "--min-score requires a value"),
            (["audit", "--bogus"], "Unknown hygiene audit option: --bogus"),
            (["status", "--limit"], "--limit requires a value"),
            (["status", "--limit", "--json"], "--limit requires a value"),
            (["status", "--bogus"], "Unknown hygiene status option: --bogus"),
            (["restore", "--limit"], "--limit requires a value"),
            (["restore", "--limit", "--dry-run"], "--limit requires a value"),
            (["restore", "--bogus"], "Unknown hygiene restore option: --bogus"),
        ],
    )
    def test_cmd_hygiene_fails_fast_on_invalid_options(self, args, message, capsys):
        with pytest.raises(SystemExit):
            cmd_hygiene(args)

        assert message in capsys.readouterr().err

    def test_hygiene_status_without_audit_log(self, temp_db):
        db_path, _beam = temp_db

        status = hygiene_status(db_path=db_path, limit=10)

        assert status["audit_log"]["present"] is False
        assert status["audit_log"]["total_entries"] == 0
        assert status["audit_log"]["by_action"] == {}

    def test_hygiene_status_can_skip_noise_summary(self, temp_db):
        db_path, _beam = temp_db

        status = hygiene_status(db_path=db_path, include_noise_summary=False)

        assert "noise_summary" not in status

    def test_noise_summary_is_pii_safe(self, temp_db):
        db_path, beam = temp_db
        secret_content = "password = hunter2supersecret"  # nosec - test fixture
        _insert_row(beam, "working_memory", "secret1", secret_content)

        summary = noise_summary(db_path=db_path, min_score=0.0)

        assert summary["total_candidates"] == 1
        assert summary["with_secrets"] == 1
        assert secret_content not in json.dumps(summary)
        assert "content_preview" not in json.dumps(summary)

    def test_hygiene_status_reports_audit_log_without_content(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "noise1", "heartbeat", source="heartbeat")
        conn = beam.conn
        conn.execute(
            """
            CREATE TABLE hygiene_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                table_name TEXT NOT NULL,
                action TEXT NOT NULL,
                reason TEXT,
                noise_score REAL,
                secret_flags TEXT,
                original_content_preview TEXT,
                original_metadata TEXT,
                timestamp TEXT NOT NULL,
                session_id TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO hygiene_audit_log (memory_id, table_name, action, timestamp) VALUES (?, ?, ?, ?)",
            ("noise1", "working_memory", "flagged", "2025-01-01T00:00:00"),
        )
        conn.commit()

        status = hygiene_status(db_path=db_path, limit=10)

        assert status["audit_log"]["present"] is True
        assert status["audit_log"]["total_entries"] == 1
        assert status["audit_log"]["by_action"]["flagged"] == 1
        assert "heartbeat" not in json.dumps(status)

    def test_audit_preserves_zero_importance(self, temp_db):
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "zero", "heartbeat", source="heartbeat", importance=0.0)

        report = audit_noise(db_path=db_path, tables=["working_memory"], min_score=0.0)

        assert report.candidates[0].importance == 0.0

    def test_audit_rejects_invalid_table_identifiers(self, temp_db):
        db_path, _beam = temp_db

        with pytest.raises(ValueError, match="Invalid table identifier"):
            audit_noise(db_path=db_path, tables=["working_memory; DROP TABLE memories"])

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
        _insert_row(beam, "working_memory", "n1", "heartbeat", importance=0.8,
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

        # Verify restored to ORIGINAL importance (0.8), not hardcoded 0.5
        cursor = conn.execute("SELECT importance, metadata_json FROM working_memory WHERE id = 'n1'")
        row = cursor.fetchone()
        assert row[0] == 0.8  # original importance preserved and restored
        meta = json.loads(row[1])
        assert "_archived" not in meta
        assert "_original_importance" not in meta  # cleaned up on restore
        assert meta.get("original") == "data"
        conn.close()


def test_hygiene_suite_does_not_leak_config_into_subagent_provider(tmp_path, monkeypatch):
    """A provider after hygiene must use its own safe temporary config.

    Hygiene/Doctor paths initialize the process-wide MnemosyneConfig singleton.
    This regression exercises the subsequent provider path in the same pytest
    process: the autouse cleanup must discard that singleton before a subagent
    provider resolves its temporary data-directory configuration.
    """
    from conftest import _close_cached_connections
    from hermes_memory_provider import MnemosyneMemoryProvider
    from mnemosyne.core.config import MnemosyneConfig

    stale_data_dir = tmp_path / "stale-data"
    test_data_dir = tmp_path / "provider-data"
    stale_data_dir.mkdir()
    test_data_dir.mkdir()
    # The stale config enables subagent initialization. The replacement config
    # must win after the fixture boundary resets the process-global singleton.
    (stale_data_dir / "config.yaml").write_text("skip_contexts: ''\n", encoding="utf-8")
    (test_data_dir / "config.yaml").write_text("skip_contexts: subagent\n", encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(stale_data_dir))
    stale_config = MnemosyneConfig.get_instance()
    assert stale_config.config_path == stale_data_dir / "config.yaml"

    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(test_data_dir))
    _close_cached_connections()
    assert MnemosyneConfig._instance is None
    provider = MnemosyneMemoryProvider()
    provider.initialize("hygiene-followup", agent_context="subagent")

    assert provider._beam is None
