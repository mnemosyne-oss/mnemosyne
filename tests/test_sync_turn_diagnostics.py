"""Tests for sync_turn telemetry and diagnostics."""

import json
import logging
from unittest.mock import MagicMock

import pytest


_REQUIRED_SYNC_TURN_KEYS = {
    "pending_queue_length",
    "max_queue_length",
    "completed",
    "failed",
    "merged",
    "dropped",
    "slow_sync_count",
    "last_error",
}


@pytest.fixture
def provider():
    from hermes_memory_provider import MnemosyneMemoryProvider

    p = MnemosyneMemoryProvider()
    p._agent_context = "test"
    p._skip_contexts = set()
    p._auto_sleep_enabled = False
    p._default_scope = "session"
    p._beam = MagicMock()
    p._beam.db_path = None
    return p


def test_sync_turn_increments_completed_and_records_duration(provider):
    provider.sync_turn("remember this user preference", "ack")

    diag = provider._sync_turn_diagnostics()
    assert diag["completed"] == 1
    assert diag["failed"] == 0
    assert diag["in_flight"] == 0
    assert diag["pending_queue_length"] == 0
    assert diag["max_queue_length"] >= 1
    assert diag["last_duration_ms"] is not None
    assert diag["max_duration_ms"] >= diag["last_duration_ms"]
    assert _REQUIRED_SYNC_TURN_KEYS.issubset(diag)


def test_diagnose_exposes_sync_turn_section(provider, monkeypatch):
    import mnemosyne.diagnose

    monkeypatch.setattr(
        mnemosyne.diagnose,
        "run_diagnostics",
        lambda **kwargs: {"checks_total": 0, "entries": [], "key_findings": []},
    )
    provider.sync_turn("remember this user preference", "ack")

    result = json.loads(provider._handle_diagnose({}))

    assert "sync_turn" in result
    assert result["sync_turn"]["completed"] == 1
    assert _REQUIRED_SYNC_TURN_KEYS.issubset(result["sync_turn"])


def test_slow_sync_turn_warns_without_content(provider, monkeypatch, caplog):
    import hermes_memory_provider

    secret_user = "secret user content should not appear"
    secret_assistant = "secret assistant content should not appear"
    times = iter([100.0, 100.250])
    monkeypatch.setattr(hermes_memory_provider.time, "perf_counter", lambda: next(times))
    provider._SYNC_TURN_SLOW_THRESHOLD_SECONDS = 0.001

    with caplog.at_level(logging.WARNING):
        provider.sync_turn(secret_user, secret_assistant)

    diag = provider._sync_turn_diagnostics()
    warning_text = "\n".join(record.getMessage() for record in caplog.records)
    assert diag["slow_sync_count"] == 1
    assert "Mnemosyne sync_turn slow" in warning_text
    assert secret_user not in warning_text
    assert secret_assistant not in warning_text


def test_sync_turn_failure_is_recorded_and_sanitized(provider):
    secret = "private user message should not leak"
    provider._beam.remember.side_effect = RuntimeError(secret)

    provider.sync_turn("remember this user preference", "ack")

    diag = provider._sync_turn_diagnostics()
    assert diag["completed"] == 0
    assert diag["failed"] == 1
    assert diag["last_error"] == "RuntimeError: <redacted>"
    assert secret not in diag["last_error"]
    assert diag["in_flight"] == 0
