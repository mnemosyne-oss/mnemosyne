"""Tests for doctor bank resolution in the hermes CLI.

Regression coverage for #214: `hermes mnemosyne doctor` ignored
the resolved profile bank and inspected the profile-root metadata DB.

Related: #362, #373.
"""

import json
import os
import sqlite3
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mnemosyne_hermes.cli import _resolve_cli_bank


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args(**kw):
    return types.SimpleNamespace(**kw)


def _write_config(home, isolation, sync_roles=None):
    """Write a minimal Hermes config.yaml with profile_isolation setting."""
    home.mkdir(parents=True, exist_ok=True)
    body = (
        "memory:\n"
        "  provider: mnemosyne\n"
        f"  memory_enabled: {str(isolation).lower()}\n"
        f"  user_profile_enabled: {str(isolation).lower()}\n"
        "  mnemosyne:\n"
        f"    profile_isolation: {str(isolation).lower()}\n"
        f"    sync_roles: {sync_roles or []}\n"
    )
    (home / "config.yaml").write_text(body)


def _create_minimal_schema(conn):
    """Create the working_memory schema that Mnemosyne expects."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS working_memory ("
        "id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT,"
        "session_id TEXT, importance REAL, metadata_json TEXT, veracity TEXT,"
        "created_at TEXT, memory_type TEXT, consolidated_at TEXT,"
        "consolidation_claimed_at TEXT, recall_count INTEGER, last_recalled TEXT,"
        "pinned INTEGER, valid_until TEXT, superseded_by TEXT, scope TEXT,"
        "author_id TEXT, author_type TEXT, channel_id TEXT, trust_tier TEXT,"
        "validator TEXT, validated_at TEXT, validation_count INTEGER,"
        "event_date TEXT, event_date_precision TEXT, temporal_tags TEXT,"
        "corrected_by TEXT, event_date_end TEXT)"
    )


def make_profile_bank(home, bank_name, wm_count=3):
    """Create a minimal named bank with verifiable WM rows.
    Returns (db_path, data_dir).
    """
    data_dir = home / "mnemosyne" / "data"
    bank_dir = data_dir / "banks" / bank_name
    bank_dir.mkdir(parents=True, exist_ok=True)
    db_path = bank_dir / "mnemosyne.db"
    conn = sqlite3.connect(str(db_path))
    _create_minimal_schema(conn)
    for i in range(wm_count):
        conn.execute(
            "INSERT OR IGNORE INTO working_memory (id, content, source, timestamp) VALUES (?, ?, 'user', '2026-01-01')",
            (f"{bank_name}-wm-{i}", f"memory {i} in {bank_name}"),
        )
    conn.commit()
    conn.close()
    return db_path, data_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDoctorBankResolution:
    """Doctor respects resolved profile bank and rejects unknown banks."""

    def test_default_doctor_resolves_default_db(self, tmp_path, monkeypatch, capsys):
        """Default profile -> default DB."""
        home = tmp_path / ".hermes"
        _write_config(home, False)
        data_dir = home / "mnemosyne" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        conn = sqlite3.connect(str(data_dir / "mnemosyne.db"))
        _create_minimal_schema(conn)
        conn.execute("INSERT OR IGNORE INTO working_memory (id) VALUES ('default-1')")
        conn.commit()
        conn.close()

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank=None,
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert "resolved_bank: default" in out

    def test_profile_isolation_implicit_bank(self, tmp_path, monkeypatch, capsys):
        """Named profile with isolation -> its own bank."""
        home = tmp_path / "profiles" / "reverse-engineer"
        _write_config(home, True)
        db_path, data_dir = make_profile_bank(home, "reverse-engineer", wm_count=5)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank=None,
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert "resolved_bank: reverse-engineer" in out

    def test_explicit_bank_flag_overrides_profile(self, tmp_path, monkeypatch, capsys):
        """--bank takes precedence."""
        home = tmp_path / "profiles" / "default"
        _write_config(home, True)
        db_path, data_dir = make_profile_bank(home, "custom-bank", wm_count=2)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank="custom-bank",
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert "resolved_bank: custom-bank" in out

    def test_bank_name_sanitized_via_resolver(self, tmp_path, monkeypatch, capsys):
        """Explicit --bank is sanitized via _resolve_cli_bank."""
        home = tmp_path / "profiles" / "default"
        _write_config(home, True)
        db_path, data_dir = make_profile_bank(home, "my_bank", wm_count=1)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank="My Bank",
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert "resolved_bank: my_bank" in out

    def test_empty_metadata_db_does_not_override_named_bank(self, tmp_path, monkeypatch, capsys):
        """Named bank has data -> use it, even if metadata DB is empty."""
        home = tmp_path / "profiles" / "reverse-engineer"
        _write_config(home, True)
        db_path, data_dir = make_profile_bank(home, "reverse-engineer", wm_count=7)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        # Empty metadata DB at wrong path
        meta_path = home / "profiles" / "reverse-engineer" / "mnemosyne" / "data" / "mnemosyne.db"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(meta_path))
        _create_minimal_schema(conn)
        conn.commit()
        conn.close()

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank=None,
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert "resolved_bank: reverse-engineer" in out

    def test_unknown_bank_returns_error(self, tmp_path, monkeypatch, capsys):
        """Unknown bank -> non-zero exit, clear error."""
        home = tmp_path / "profiles" / "default"
        _write_config(home, True)
        data_dir = home / "mnemosyne" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank="nonexistent-bank-xyz",
            dry_run=False,
            no_fix=True,
        )
        ret = cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert ret == 1
        assert "Bank not found: nonexistent-bank-xyz" in out

    def test_unknown_bank_creates_no_filesystem_artifacts(self, tmp_path, monkeypatch):
        """Unknown bank -> no banks/ dir, named dir, DB, or any other artifact."""
        home = tmp_path / "profiles" / "default"
        _write_config(home, True)
        data_dir = home / "mnemosyne" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        banks_dir = data_dir / "banks"
        named_dir = banks_dir / "nonexistent-bank-xyz"
        named_db = named_dir / "mnemosyne.db"

        # Full data_dir snapshot BEFORE the command.
        before = sorted(
            p.relative_to(data_dir) for p in data_dir.rglob("*")
        )
        banks_dir_existed_before = banks_dir.exists()

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank="nonexistent-bank-xyz",
            dry_run=False,
            no_fix=True,
        )
        ret = cli_mod.mnemosyne_command(args)
        assert ret == 1

        # Full data_dir snapshot AFTER the command.
        after = sorted(
            p.relative_to(data_dir) for p in data_dir.rglob("*")
        )

        # No filesystem materialization of the unknown bank.
        assert banks_dir.exists() == banks_dir_existed_before
        assert not named_dir.exists()
        assert not named_db.exists()
        # And the whole data_dir tree is byte-for-byte unchanged.
        assert before == after

    def test_output_contains_resolved_bank_and_db(self, tmp_path, monkeypatch, capsys):
        """Output shows resolved_bank and resolved_db paths."""
        home = tmp_path / "profiles" / "reverse-engineer"
        _write_config(home, True)
        db_path, data_dir = make_profile_bank(home, "reverse-engineer", wm_count=3)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank=None,
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert "resolved_bank: reverse-engineer" in out
        assert "resolved_db:" in out
        assert str(db_path) in out

    def test_stats_sleep_routing_unaffected(self, tmp_path, monkeypatch):
        """Existing stats/sleep routing not broken."""
        home = tmp_path / "profiles" / "work"
        _write_config(home, True)
        db_path, data_dir = make_profile_bank(home, "work", wm_count=4)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        from mnemosyne_hermes import cli as cli_mod

        stats_args = types.SimpleNamespace(
            mnemosyne_cmd="stats",
            global_=False,
            bank=None,
        )
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            cli_mod.mnemosyne_command(stats_args)
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout
        stats_data = json.loads(output)
        assert stats_data["working"]["total"] == 4

    def test_doctor_no_schema_mutation(self, tmp_path, monkeypatch, capsys):
        """Routing change does not alter existing data."""
        home = tmp_path / "profiles" / "reverse-engineer"
        _write_config(home, True)
        db_path, data_dir = make_profile_bank(home, "reverse-engineer", wm_count=3)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        conn = sqlite3.connect(str(db_path))
        before_count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        conn.close()

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank=None,
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)

        conn = sqlite3.connect(str(db_path))
        after_count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        conn.close()

        assert before_count == after_count == 3


class TestDoctorBankEdgeCases:

    def test_default_bank_passthrough(self, tmp_path, monkeypatch, capsys):
        """bank=None -> default behavior."""
        home = tmp_path / ".hermes"
        _write_config(home, False)
        data_dir = home / "mnemosyne" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank=None,
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert "resolved_bank: default" in out

    def test_empty_string_bank_treated_as_default(self, tmp_path, monkeypatch, capsys):
        """Empty bank -> default (no empty structures created)."""
        home = tmp_path / ".hermes"
        _write_config(home, False)
        data_dir = home / "mnemosyne" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank="",
            dry_run=False,
            no_fix=True,
        )
        cli_mod.mnemosyne_command(args)
        out = capsys.readouterr().out
        assert "resolved_bank: default" in out

    def test_malformed_bank_name_rejected_no_artifacts(self, tmp_path, monkeypatch):
        """Malformed bank name -> non-zero exit, no dir/DB created."""
        home = tmp_path / ".hermes"
        _write_config(home, False)
        data_dir = home / "mnemosyne" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        banks_dir = data_dir / "banks"
        # Malformed names that fail _validate_bank_name.
        for bad in ("bad name!!", "way_too_long_" + "x" * 80):
            named_dir = banks_dir / bad
            named_db = named_dir / "mnemosyne.db"
            from mnemosyne_hermes import cli as cli_mod
            args = types.SimpleNamespace(
                mnemosyne_cmd="doctor",
                bank=bad,
                dry_run=False,
                no_fix=True,
            )
            ret = cli_mod.mnemosyne_command(args)
            assert ret == 1, f"expected non-zero exit for malformed bank {bad!r}"
            assert not named_dir.exists(), f"malformed bank {bad!r} created a directory"
            assert not named_db.exists(), f"malformed bank {bad!r} created a DB"


    def test_profile_isolation_implicit_bank_missing_fails_cleanly(self, tmp_path, monkeypatch):
        """Implicit profile-derived bank that does not yet exist is rejected.

        CodeRabbit suggested limiting the guard to explicit --bank only,
        letting a missing implicit profile bank fall through to lazy creation.
        We intentionally reject it: a profile-derived bank is still the
        resolved diagnostic target, and allowing Mnemosyne(bank=...) to
        materialize an empty bank mid-diagnostic would make `doctor`
        mutate the very state it is meant to inspect. This test pins
        that behavior and proves no filesystem artifact is created.
        """
        home = tmp_path / "profiles" / "reverse-engineer"
        _write_config(home, True)  # profile_isolation enabled
        data_dir = home / "mnemosyne" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        banks_dir = data_dir / "banks"
        named_dir = banks_dir / "reverse-engineer"
        named_db = named_dir / "mnemosyne.db"
        # The implicit profile bank has NOT been created yet.
        assert not named_dir.exists()

        from mnemosyne_hermes import cli as cli_mod
        args = types.SimpleNamespace(
            mnemosyne_cmd="doctor",
            bank=None,  # no explicit --bank; profile basename drives resolution
            dry_run=False,
            no_fix=True,
        )
        ret = cli_mod.mnemosyne_command(args)
        assert ret == 1
        # run_diagnostics() must NOT have been reached / no FS materialization.
        assert not banks_dir.exists() or not named_dir.exists()
        assert not named_db.exists()


    def test_sleep_routing_unaffected_dry_run(self, tmp_path, monkeypatch):
        """Sleep (dry-run) still routes to the resolved named bank beam.

        `doctor` guard only fires for cmd == "doctor"; sleep keeps its
        existing routing. We assert that:
          * the resolved bank beam is bound to the profile/implicit bank
            ("work"), not the default bank,
          * `beam.sleep(dry_run=True)` is actually invoked on that beam
            (not merely that the command returns without raising),
          * the doctor guard is NOT applied to sleep (no rejection / no
            `return 1`),
          * no real consolidation or DB mutation occurs (mocked beam).
        """
        home = tmp_path / "profiles" / "work"
        _write_config(home, True)
        db_path, data_dir = make_profile_bank(home, "work", wm_count=2)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))

        # Build a mock Mnemosyne whose beam records the sleep call.
        mock_beam = MagicMock()
        mock_instance = MagicMock()
        mock_instance.beam = mock_beam
        # `beam.sleep` must return a dry-run no-op dict.
        mock_beam.sleep.return_value = {"status": "no_op"}

        from mnemosyne_hermes import cli as cli_mod

        with patch(
            "mnemosyne.core.memory.Mnemosyne", return_value=mock_instance
        ) as mock_mnemo:
            sleep_args = types.SimpleNamespace(
                mnemosyne_cmd="sleep",
                all_sessions=False,
                dry_run=True,
                bank=None,
            )
            ret = cli_mod.mnemosyne_command(sleep_args)

        # Dispatch must not hit the doctor guard and must succeed.
        assert ret is None or ret == 0, f"sleep unexpectedly rejected: {ret}"

        # Mnemosyne must have been constructed bound to the resolved
        # profile/implicit bank "work", never the default bank.
        assert mock_mnemo.called, "Mnemosyne was never constructed"
        (call_bank,) = [
            kw.get("bank") for (_, kw) in mock_mnemo.call_args_list
        ]
        assert call_bank == "work", (
            f"sleep bound to wrong bank: expected 'work', got {call_bank!r}"
        )

        # The resolved-bank beam must have received the dry-run sleep call.
        mock_beam.sleep.assert_called_once_with(dry_run=True)

