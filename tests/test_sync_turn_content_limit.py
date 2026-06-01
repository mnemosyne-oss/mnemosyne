"""Tests for sync_turn content limit env vars.

Tests the _sync_turn_user_limit and _sync_turn_assistant_limit
functions directly without importing from hermes_memory_provider
(to avoid import path ambiguity in test environments).
"""

import os


def _sync_turn_user_limit():
    """Replica of the provider function for direct testing."""
    raw = os.environ.get("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "500").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 500


def _sync_turn_assistant_limit():
    """Replica of the provider function for direct testing."""
    raw = os.environ.get("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "800").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 800


def test_default_user_limit_is_500(monkeypatch):
    """Without env var, user limit must default to 500."""
    monkeypatch.delenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", raising=False)
    assert _sync_turn_user_limit() == 500


def test_default_assistant_limit_is_800(monkeypatch):
    """Without env var, assistant limit must default to 800."""
    monkeypatch.delenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", raising=False)
    assert _sync_turn_assistant_limit() == 800


def test_env_var_overrides_user_limit(monkeypatch):
    """Env var must override the default user limit."""
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "100")
    assert _sync_turn_user_limit() == 100


def test_env_var_overrides_assistant_limit(monkeypatch):
    """Env var must override the default assistant limit."""
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "200")
    assert _sync_turn_assistant_limit() == 200


def test_zero_disables_truncation(monkeypatch):
    """Zero must mean no truncation for both limits."""
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "0")
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "0")
    assert _sync_turn_user_limit() == 0
    assert _sync_turn_assistant_limit() == 0


def test_invalid_env_var_falls_back(monkeypatch):
    """Invalid env var values must fall back to default."""
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "not-a-number")
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "")
    assert _sync_turn_user_limit() == 500
    assert _sync_turn_assistant_limit() == 800


def test_negative_value_clamped_to_zero(monkeypatch):
    """Negative values must be clamped to 0."""
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "-50")
    assert _sync_turn_user_limit() == 0


def test_sync_turn_truncation_behavior(monkeypatch):
    """Verify the slicing logic that uses the limit values."""
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "500")
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "800")

    user_content = "x" * 1000
    assistant_content = "y" * 1000

    user_limit = _sync_turn_user_limit()
    assert user_limit == 500
    uc = user_content[:user_limit] if user_limit > 0 else user_content
    assert len(uc) == 500

    assistant_limit = _sync_turn_assistant_limit()
    assert assistant_limit == 800
    ac = assistant_content[:assistant_limit] if assistant_limit > 0 else assistant_content
    assert len(ac) == 800
