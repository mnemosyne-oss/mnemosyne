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

import mnemosyne.core.hygiene as hygiene_module
from mnemosyne.cli import cmd_hygiene
from mnemosyne.core.beam import BeamMemory, init_beam
from mnemosyne.core.filters import SECRET_LABELED_PATTERNS
from mnemosyne.core.hygiene import (
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

    def test_cjk_secret_flagged(self):
        """CJK-labelled secrets must be flagged by the hygiene scorer (issue #806)."""
        # nosec - test fixture
        score, reasons = _score_noise("数据库密码：s3cr3t_pa55word_x1y2z3w4", 0.5, "user")
        assert score >= 0.9
        assert "secret_detected:cjk_secret_assignment" in reasons
        assert _suggest_action(score, ["cjk_secret_assignment"]) == "flag"

    def test_cjk_secret_with_trailing_prose_flagged(self):
        """Trailing CJK prose must not hide a CJK-labelled secret."""
        # nosec - test fixture
        score, reasons = _score_noise("数据库密码：s3cr3t_pa55word_x1y2z3w4，请勿外传", 0.5, "user")
        assert "secret_detected:cjk_secret_assignment" in reasons

    def test_cjk_policy_prose_not_flagged(self):
        """Ordinary Chinese policy prose after a label must stay clean."""
        score, reasons = _score_noise("密码：建议每90天更换一次", 0.5, "user")
        assert not any("secret" in r for r in reasons)

    @pytest.mark.parametrize(
        "prose",
        [
            "password：建议每90天更换一次",
            "password＝建议每90天更换一次",
        ],
    )
    def test_english_label_fullwidth_separator_policy_prose_not_flagged(self, prose):
        """English labels with fullwidth separators must not flag CJK prose."""
        _score, reasons = _score_noise(prose, 0.5, "user")
        assert "secret_detected:secret_assignment" not in reasons

    def test_cjk_ascii_prefix_then_prose_not_flagged(self):
        """ASCII prefix followed by CJK prose must not be flagged."""
        score, reasons = _score_noise("密码：abc12345我的密码", 0.5, "user")
        assert not any("secret" in r for r in reasons)

    def test_cjk_non_bmp_prefix_then_prose_not_flagged(self):
        """Non-BMP CJK after an ASCII prefix must not be flagged."""
        score, reasons = _score_noise("密码：abc12345\U00020000", 0.5, "user")
        assert not any("secret" in r for r in reasons)

    @pytest.mark.parametrize(
        "character",
        [
            "\u3005",
            "\u3006",
            "\u3007",
            "\u31f0",
            "\U000323b0",
            "\U0003347f",
            "\uff21",
            "\uffa0",
            "\uffbf",
            "\uffc1",
            "\uffc8",
            "\uffc9",
            "\uffd0",
            "\uffd1",
            "\uffd8",
            "\uffd9",
        ],
    )
    def test_cjk_boundary_prefix_then_prose_not_flagged(self, character):
        """CJK or fullwidth prose after an ASCII prefix must not be flagged."""
        score, reasons = _score_noise(f"密码：abc12345{character}", 0.5, "user")
        assert not any("secret" in r for r in reasons)

    @pytest.mark.parametrize("character", ["ſ", "ı", "İ", "K"])
    def test_unicode_casefold_equivalent_not_flagged_as_secret(self, character):
        """Unicode case-fold equivalents are not ASCII credential characters."""
        _score, reasons = _score_noise(f"密码：!!!!!!!!{character}", 0.5, "user")
        assert not any(reason.startswith("secret_detected:") for reason in reasons)

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

    @pytest.mark.parametrize("terminator", ["。", "！", "？", "."])
    def test_multiline_sentences_are_not_flagged_as_dump(self, terminator):
        content = "\n".join(
            [f"This is a complete sentence with useful content{terminator}"] * 60
        )

        _score, reasons = _score_noise(content, 0.5, "")

        assert "likely_dump" not in reasons


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

    def _prepare_clean_db(self, temp_db, monkeypatch):
        """Insert a heartbeat row and point the CLI at a copy of the database."""
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat", source="heartbeat")
        cli_db = db_path.parent / "mnemosyne.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("VACUUM INTO ?", (str(cli_db),))
        finally:
            conn.close()
        monkeypatch.setattr("mnemosyne.cli.DATA_DIR", str(db_path.parent))
        return db_path, beam, cli_db

    def test_cmd_hygiene_clean_unwraps_audit_envelope(self, temp_db, monkeypatch, capsys):
        """Regression test for #606: clean must unwrap the audit JSON envelope."""
        db_path, _beam, cli_db = self._prepare_clean_db(temp_db, monkeypatch)
        report = audit_noise(db_path=db_path, min_score=0.0)

        candidates_file = db_path.parent / "audit.json"
        candidates_file.write_text(json.dumps(report.to_dict()))

        cmd_hygiene(["clean", "--action", "delete", "--confirm", str(candidates_file)])

        captured = capsys.readouterr()
        assert "deleted=1" in captured.out
        conn = sqlite3.connect(str(cli_db))
        try:
            row = conn.execute(
                "SELECT 1 FROM working_memory WHERE id = ?", ("n1",)
            ).fetchone()
            assert row is None
        finally:
            conn.close()

    def test_cmd_hygiene_clean_accepts_raw_candidate_array(self, temp_db, monkeypatch, capsys):
        """clean must also accept the candidates array without an envelope."""
        db_path, _beam, _cli_db = self._prepare_clean_db(temp_db, monkeypatch)

        candidates_file = db_path.parent / "candidates.json"
        candidates_file.write_text(json.dumps([
            {
                "memory_id": "n1",
                "table_name": "working_memory",
                "content_preview": "heartbeat",
                "noise_score": 0.8,
                "noise_reasons": ["trivial_keyword"],
                "suggested_action": "delete",
            }
        ]))

        cmd_hygiene(["clean", "--action", "delete", "--confirm", str(candidates_file)])

        assert "deleted=1" in capsys.readouterr().out

    def test_cmd_hygiene_clean_fails_on_invalid_candidate_container(self, temp_db, monkeypatch, capsys):
        """Non-list, non-envelope containers must route to _fail."""
        db_path, _beam, _cli_db = self._prepare_clean_db(temp_db, monkeypatch)

        candidates_file = db_path.parent / "bad.json"
        candidates_file.write_text(json.dumps("not a list"))

        with pytest.raises(SystemExit):
            cmd_hygiene(["clean", "--action", "delete", "--confirm", str(candidates_file)])

        assert "JSON array" in capsys.readouterr().err

    def test_cmd_hygiene_clean_fails_on_missing_candidates_field(self, temp_db, monkeypatch, capsys):
        """An envelope without 'candidates' must route to _fail."""
        db_path, _beam, _cli_db = self._prepare_clean_db(temp_db, monkeypatch)

        candidates_file = db_path.parent / "bad.json"
        candidates_file.write_text(json.dumps({"total_scanned": 1}))

        with pytest.raises(SystemExit):
            cmd_hygiene(["clean", "--action", "delete", "--confirm", str(candidates_file)])

        assert "'candidates'" in capsys.readouterr().err

    def test_cmd_hygiene_clean_fails_on_non_object_candidate(self, temp_db, monkeypatch, capsys):
        """A candidate entry that is not a JSON object must route to _fail."""
        db_path, _beam, _cli_db = self._prepare_clean_db(temp_db, monkeypatch)

        candidates_file = db_path.parent / "bad.json"
        candidates_file.write_text(json.dumps(["not an object"]))

        with pytest.raises(SystemExit):
            cmd_hygiene(["clean", "--action", "delete", "--confirm", str(candidates_file)])

        assert "JSON object" in capsys.readouterr().err

    def test_cmd_hygiene_clean_fails_on_missing_required_field(self, temp_db, monkeypatch, capsys):
        """A candidate missing a required field must route to _fail."""
        db_path, _beam, _cli_db = self._prepare_clean_db(temp_db, monkeypatch)

        candidates_file = db_path.parent / "bad.json"
        candidates_file.write_text(json.dumps([{"table_name": "working_memory"}]))

        with pytest.raises(SystemExit):
            cmd_hygiene(["clean", "--action", "delete", "--confirm", str(candidates_file)])

        assert "Missing required field" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "malformed_candidate",
        [
            {"noise_score": "high"},
            {"table_name": "unknown_table"},
            {"suggested_action": "destroy"},
            {"noise_score": 1.1},
            {"importance": float("inf")},
            {"importance": float("nan")},
            {"content_length": -1},
            {"noise_reasons": "short"},
        ],
    )
    def test_cmd_hygiene_clean_rejects_malformed_envelope_candidate(
        self, temp_db, monkeypatch, capsys, malformed_candidate
    ):
        """Malformed envelope candidates abort through _fail, preserve the row, and do not write an audit log."""
        db_path, _beam, cli_db = self._prepare_clean_db(temp_db, monkeypatch)
        report = audit_noise(db_path=db_path, min_score=0.0)
        candidate = report.candidates[0].to_dict()
        candidate.update(malformed_candidate)

        candidates_file = db_path.parent / "bad.json"
        candidates_file.write_text(json.dumps({"candidates": [candidate]}))

        with pytest.raises(SystemExit):
            cmd_hygiene(["clean", "--action", "delete", "--confirm", str(candidates_file)])

        assert "Candidate #0" in capsys.readouterr().err
        conn = sqlite3.connect(str(cli_db))
        try:
            row = conn.execute("SELECT 1 FROM working_memory WHERE id = ?", ("n1",)).fetchone()
            assert row is not None
            audit_table = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='hygiene_audit_log'"
            ).fetchone()
            assert audit_table is None
        finally:
            conn.close()

    def test_cmd_hygiene_clean_accepts_audit_output_with_out_of_range_importance(
        self, temp_db, monkeypatch, capsys
    ):
        """Regression for the reported blocker: audit --json → clean --confirm must work
        even when BeamMemory persisted an importance value outside [0, 1]."""
        db_path, beam = temp_db
        _insert_row(beam, "working_memory", "n1", "heartbeat", source="heartbeat", importance=2.0)

        cli_db = db_path.parent / "mnemosyne.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("VACUUM INTO ?", (str(cli_db),))
        finally:
            conn.close()
        monkeypatch.setattr("mnemosyne.cli.DATA_DIR", str(db_path.parent))

        cmd_hygiene(["audit", "--json", "--min-score", "0.0"])
        envelope = json.loads(capsys.readouterr().out)
        assert envelope["candidates"]

        candidates_file = db_path.parent / "audit.json"
        candidates_file.write_text(json.dumps(envelope))

        cmd_hygiene(["clean", "--action", "delete", "--confirm", str(candidates_file)])

        captured = capsys.readouterr()
        assert "deleted=1" in captured.out
        conn = sqlite3.connect(str(cli_db))
        try:
            row = conn.execute("SELECT 1 FROM working_memory WHERE id = ?", ("n1",)).fetchone()
            assert row is None
        finally:
            conn.close()

    def test_hygiene_status_without_audit_log(self, temp_db):
        db_path, _beam = temp_db

        status = hygiene_status(db_path=db_path, limit=10)

        assert status["audit_log"]["present"] is False
        assert status["audit_log"]["total_entries"] == 0
        assert status["audit_log"]["by_action"] == {}

    def test_hygiene_status_can_skip_noise_summary_with_owned_connection(self, temp_db, monkeypatch):
        db_path, _beam = temp_db
        original_connect = sqlite3.connect
        owned_connections = []

        def capture_connection(*args, **kwargs):
            connection = original_connect(*args, **kwargs)
            owned_connections.append(connection)
            return connection

        def unexpected_noise_summary(*_args, **_kwargs):
            raise AssertionError("noise_summary must not be called when disabled")

        monkeypatch.setattr(hygiene_module.sqlite3, "connect", capture_connection)
        monkeypatch.setattr(hygiene_module, "noise_summary", unexpected_noise_summary)

        status = hygiene_status(db_path=db_path, include_noise_summary=False)

        assert status["status"] == "ok"
        assert "noise_summary" not in status
        assert len(owned_connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            owned_connections[0].execute("SELECT 1")

    def test_hygiene_status_closes_owned_connection_when_audit_log_read_fails(self, temp_db, monkeypatch):
        db_path, _beam = temp_db
        original_connect = sqlite3.connect
        owned_connections = []
        read_error = sqlite3.OperationalError("audit log read failed")

        def capture_connection(*args, **kwargs):
            connection = original_connect(*args, **kwargs)
            owned_connections.append(connection)
            return connection

        def fail_audit_log_read(*_args, **_kwargs):
            raise read_error

        monkeypatch.setattr(hygiene_module.sqlite3, "connect", capture_connection)
        monkeypatch.setattr(hygiene_module, "_table_exists", fail_audit_log_read)

        with pytest.raises(sqlite3.OperationalError, match="audit log read failed") as exc_info:
            hygiene_status(db_path=db_path, include_noise_summary=False)

        assert exc_info.value is read_error
        assert len(owned_connections) == 1
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            owned_connections[0].execute("SELECT 1")

    def test_hygiene_status_can_skip_noise_summary_without_closing_supplied_readonly_connection(self, temp_db):
        db_path, _beam = temp_db
        readonly = open_readonly_doctor_db(db_path)
        try:
            status = hygiene_status(
                db_path=db_path,
                include_noise_summary=False,
                conn=readonly,
            )

            assert "noise_summary" not in status
            assert status["status"] == "ok"
            assert readonly.execute("SELECT 1").fetchone()[0] == 1
        finally:
            readonly.close()

    def test_hygiene_status_uses_supplied_readonly_connection(self, temp_db, monkeypatch):
        db_path, _beam = temp_db
        readonly = open_readonly_doctor_db(db_path)
        original_noise_summary = hygiene_module.noise_summary
        received_connection = None

        def capture_noise_summary(*args, **kwargs):
            nonlocal received_connection
            received_connection = kwargs["conn"]
            return original_noise_summary(*args, **kwargs)

        monkeypatch.setattr(hygiene_module, "noise_summary", capture_noise_summary)
        try:
            status = hygiene_status(db_path=db_path, limit=10, conn=readonly)

            assert received_connection is readonly
            assert readonly.execute("PRAGMA query_only").fetchone()[0] == 1
            readonly.execute("PRAGMA query_only = OFF")
            with pytest.raises(sqlite3.OperationalError):
                readonly.execute("CREATE TABLE forbidden_hygiene_write (id INTEGER)")
        finally:
            readonly.close()

        assert status["status"] == "ok"

    def test_hygiene_status_keeps_supplied_readonly_connection_open_when_noise_summary_fails(
        self, temp_db, monkeypatch
    ):
        db_path, _beam = temp_db
        readonly = open_readonly_doctor_db(db_path)
        received_connection = None

        def fail_noise_summary(*_args, **kwargs):
            nonlocal received_connection
            received_connection = kwargs["conn"]
            raise sqlite3.OperationalError("noise summary failed")

        monkeypatch.setattr(hygiene_module, "noise_summary", fail_noise_summary)
        try:
            with pytest.raises(sqlite3.OperationalError, match="noise summary failed"):
                hygiene_status(db_path=db_path, limit=10, conn=readonly)

            assert received_connection is readonly
            assert readonly.execute("SELECT 1").fetchone()[0] == 1
        finally:
            readonly.close()

    @pytest.mark.parametrize(
        ("command_args", "function_name"),
        [
            (["audit", "--json"], "audit_noise"),
            (["status", "--json"], "hygiene_status"),
        ],
    )
    def test_cmd_hygiene_read_commands_use_readonly_connection(
        self, monkeypatch, tmp_path, capsys, command_args, function_name
    ):
        db_path = tmp_path / "mnemosyne.db"
        writable = sqlite3.connect(db_path)
        writable.execute(
            """
            CREATE TABLE working_memory (
                id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT,
                session_id TEXT, importance REAL, metadata_json TEXT
            )
            """
        )
        writable.commit()
        writable.close()
        monkeypatch.setattr("mnemosyne.cli.DATA_DIR", str(tmp_path))

        original = getattr(hygiene_module, function_name)
        injected_connection = None

        def require_readonly_connection(*args, **kwargs):
            nonlocal injected_connection
            conn = kwargs["conn"]
            injected_connection = conn
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            conn.execute("PRAGMA query_only = OFF")
            with pytest.raises(sqlite3.OperationalError):
                conn.execute("CREATE TABLE forbidden_cli_write (id INTEGER)")
            return original(*args, **kwargs)

        monkeypatch.setattr(hygiene_module, function_name, require_readonly_connection)

        cmd_hygiene(command_args)

        assert injected_connection is not None
        with pytest.raises(sqlite3.ProgrammingError):
            injected_connection.execute("SELECT 1")
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize(
        ("command_args", "function_name"),
        [
            (["audit"], "audit_noise"),
            (["status"], "hygiene_status"),
        ],
    )
    def test_cmd_hygiene_read_commands_close_connection_after_hygiene_error(
        self, monkeypatch, tmp_path, capsys, command_args, function_name
    ):
        db_path = tmp_path / "mnemosyne.db"
        sqlite3.connect(db_path).close()
        monkeypatch.setattr("mnemosyne.cli.DATA_DIR", str(tmp_path))

        connection = None

        def capture_readonly_connection(path):
            nonlocal connection
            connection = open_readonly_doctor_db(path)
            return connection

        def fail_after_connection(*_args, **kwargs):
            assert kwargs["conn"] is connection
            raise sqlite3.OperationalError("hygiene read failed")

        monkeypatch.setattr("mnemosyne.doctor.open_readonly_doctor_db", capture_readonly_connection)
        monkeypatch.setattr(hygiene_module, function_name, fail_after_connection)

        with pytest.raises(SystemExit):
            cmd_hygiene(command_args)

        assert connection is not None
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")
        expected_code = (
            "hygiene_audit_failed"
            if function_name == "audit_noise"
            else "hygiene_status_failed"
        )
        assert capsys.readouterr().err == f"Error: {expected_code}\n"

    @pytest.mark.parametrize("command_args", [["audit"], ["status"]])
    def test_cmd_hygiene_read_commands_report_readonly_open_errors(
        self, monkeypatch, tmp_path, capsys, command_args
    ):
        db_path = tmp_path / "mnemosyne.db"
        sqlite3.connect(db_path).close()
        monkeypatch.setattr("mnemosyne.cli.DATA_DIR", str(tmp_path))

        def fail_open(_db_path):
            raise sqlite3.OperationalError("readonly connection failed")

        monkeypatch.setattr("mnemosyne.doctor.open_readonly_doctor_db", fail_open)

        with pytest.raises(SystemExit):
            cmd_hygiene(command_args)

        assert capsys.readouterr().err == f"Error: hygiene_{command_args[0]}_failed\n"

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
