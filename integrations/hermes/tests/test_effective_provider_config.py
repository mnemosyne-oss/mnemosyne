"""Regression tests for the packaged Hermes provider's effective config."""
from __future__ import annotations

import logging

import pytest

from mnemosyne.core import config as core_config
from mnemosyne_hermes import MnemosyneMemoryProvider


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_SYNC_ROLES", raising=False)
    instance = MnemosyneMemoryProvider()
    monkeypatch.setattr(instance, "_read_config_key", lambda _key: None)
    return instance


def test_serialized_yaml_list_string_no_longer_disables_capture(provider):
    provider._apply_provider_config({"sync_roles": "['user']"})
    assert provider._sync_roles == {"user"}


def test_explicit_empty_list_disables_capture(provider):
    provider._apply_provider_config({"sync_roles": []})
    assert provider._sync_roles == set()


def test_unknown_only_keeps_safe_default_and_logs(provider, caplog):
    with caplog.at_level(logging.WARNING):
        provider._apply_provider_config({"sync_roles": "['users']"})
    assert provider._sync_roles == {"user"}
    assert "no valid" in caplog.text


def test_effective_config_reports_applied_guardrails(provider):
    provider._apply_provider_config({
        "auto_sleep": False,
        "sync_roles": [],
        "default_scope": "session",
        "reflect": {"disabled_for_cron": True, "max_calls_per_session": 0},
    })
    assert provider.effective_config() == {
        "auto_sleep": False,
        "default_scope": "session",
        "profile_isolation": False,
        "reflect": {"disabled_for_cron": True, "max_calls_per_session": 0},
        "skip_contexts": ["background", "cron", "flush", "skill_loop", "subagent"],
        "sync_roles": [],
    }



def test_invalid_profile_isolation_preserves_existing_boundary(provider, caplog):
    provider._apply_provider_config({"profile_isolation": "true"})
    assert provider._profile_isolation_enabled is True

    with caplog.at_level(logging.WARNING):
        provider._apply_provider_config({"profile_isolation": "definitely-not-a-bool"})

    assert provider._profile_isolation_enabled is True
    assert "invalid boolean" in caplog.text


def test_fresh_core_config_preserves_user_only_capture_default(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("MNEMOSYNE_SYNC_ROLES", raising=False)
    monkeypatch.setattr(core_config.MnemosyneConfig, "_instance", None)
    instance = MnemosyneMemoryProvider()
    instance._hermes_home = str(tmp_path)

    instance._apply_provider_config({})

    assert instance.effective_config()["sync_roles"] == ["user"]
