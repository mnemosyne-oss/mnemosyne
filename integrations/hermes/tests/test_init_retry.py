from __future__ import annotations

import sqlite3

import pytest

import mnemosyne_hermes
from mnemosyne_hermes import MnemosyneMemoryProvider


class _OkBeam:
    """Minimal stand-in for BeamMemory: accepts the init kwargs and the
    attributes initialize() sets on success."""

    author_id = None

    def __init__(self, *args, **kwargs):
        pass


def _locked_beam_class(calls):
    """A BeamMemory class whose first `calls['fail']` constructions raise the
    transient SQLite lock error, then succeed."""

    class _Beam(_OkBeam):
        def __init__(self, *args, **kwargs):
            calls["n"] += 1
            if calls["n"] <= calls["fail"]:
                raise sqlite3.OperationalError("database is locked")

    return _Beam


def test_transient_init_failure_stashes_retry_and_says_so(monkeypatch):
    calls = {"n": 0, "fail": 99}
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _locked_beam_class(calls))

    provider = MnemosyneMemoryProvider()
    provider.initialize("sess")

    assert provider._beam is None
    assert provider._init_error is not None
    assert provider._retry_init_args is not None

    block = provider.system_prompt_block()
    assert "UNAVAILABLE" in block
    assert "retried automatically" in block
    # The old "restart Hermes" guidance must not appear while a retry is pending.
    assert "restart Hermes" not in block


def test_non_transient_init_failure_keeps_fail_once(monkeypatch):
    class _CorruptBeam:
        def __init__(self, *args, **kwargs):
            raise sqlite3.DatabaseError("file is not a database")

    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _CorruptBeam)

    provider = MnemosyneMemoryProvider()
    provider.initialize("sess")

    assert provider._beam is None
    assert provider._retry_init_args is None

    block = provider.system_prompt_block()
    assert "UNAVAILABLE" in block
    assert "restart Hermes" in block


def test_retry_recovers_once_the_lock_clears(monkeypatch):
    calls = {"n": 0, "fail": 1}
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _locked_beam_class(calls))

    provider = MnemosyneMemoryProvider()
    provider.initialize("sess")
    assert provider._beam is None

    # Interval elapsed -> the next per-turn surface call re-runs initialize().
    provider._retry_init_at = 0.0
    block = provider.system_prompt_block()

    assert provider._beam is not None
    assert provider._init_error is None
    assert provider._retry_init_args is None
    assert "UNAVAILABLE" not in block


def test_retry_waits_for_the_interval(monkeypatch):
    calls = {"n": 0, "fail": 99}
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _locked_beam_class(calls))

    provider = MnemosyneMemoryProvider()
    provider.initialize("sess")
    attempts_after_init = calls["n"]

    # _retry_init_at is ~60s in the future; surface calls must not retry yet.
    provider.system_prompt_block()
    provider.prefetch("query")
    assert calls["n"] == attempts_after_init


def test_fresh_initialize_supersedes_pending_retry(monkeypatch):
    calls = {"n": 0, "fail": 1}
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _locked_beam_class(calls))

    provider = MnemosyneMemoryProvider()
    provider.initialize("sess")
    assert provider._retry_init_args is not None

    # A real re-init (e.g. session reset) that succeeds clears the stash.
    provider.initialize("sess2")
    assert provider._beam is not None
    assert provider._retry_init_args is None


def test_retry_validation_failure_is_reported_not_raised(monkeypatch):
    """A config edit landing during a pending transient retry must not let
    _configured_tool_schemas()'s ValueError escape a per-turn surface."""
    calls = {"n": 0, "fail": 99}
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _locked_beam_class(calls))

    provider = MnemosyneMemoryProvider()
    provider.initialize("sess")
    assert provider._beam is None
    assert provider._retry_init_args is not None

    def _bad_tools_config(self, key):
        return ["not_a_real_tool"] if key == "tools" else None

    monkeypatch.setattr(MnemosyneMemoryProvider, "_read_config_key", _bad_tools_config)
    provider._retry_init_at = 0.0

    block = provider.system_prompt_block()

    assert provider._beam is None
    assert isinstance(provider._init_error, ValueError)
    assert "Unknown Mnemosyne tool" in str(provider._init_error)
    # Fail-once: a validation error must not re-arm the retry.
    assert provider._retry_init_args is None
    assert "UNAVAILABLE" in block
    assert "restart Hermes" in block


def test_explicit_initialize_still_raises_validation_error(monkeypatch):
    """The retry path's new except must not swallow #1063 for a direct call."""
    def _bad_tools_config(self, key):
        return ["not_a_real_tool"] if key == "tools" else None

    monkeypatch.setattr(MnemosyneMemoryProvider, "_read_config_key", _bad_tools_config)

    provider = MnemosyneMemoryProvider()
    with pytest.raises(ValueError, match="Unknown Mnemosyne tool"):
        provider.initialize("sess")


def test_unrelated_valueerror_before_reset_propagates(monkeypatch):
    """dplush's first boundary probe (#1091 comment): a ValueError that has
    nothing to do with tool validation, raised before _initialize_locked()
    clears the pending-retry stash, must come out of the per-turn surface
    as itself, not get relabeled as a validation failure."""
    calls = {"n": 0, "fail": 99}
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _locked_beam_class(calls))

    provider = MnemosyneMemoryProvider()
    provider.initialize("sess")
    assert provider._beam is None
    assert provider._retry_init_args is not None

    def _boom(self):
        raise ValueError("boundary probe: unrelated failure before reset")

    monkeypatch.setattr(MnemosyneMemoryProvider, "_invalidate_surface_locked", _boom)
    provider._retry_init_at = 0.0

    with pytest.raises(ValueError, match="boundary probe: unrelated failure before reset"):
        provider.system_prompt_block()

    # Not relabeled as a tool-validation failure.
    assert provider._init_error is None or "boundary probe" not in str(provider._init_error)


def test_unrelated_valueerror_after_beam_creation_propagates(monkeypatch):
    """dplush's second boundary probe (#1091 comment): a ValueError raised
    after the Beam object already exists must still come out of the
    per-turn surface as itself, not get relabeled as a validation failure."""
    calls = {"n": 0, "fail": 1}
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _locked_beam_class(calls))

    provider = MnemosyneMemoryProvider()
    provider.initialize("sess")
    assert provider._beam is None
    assert provider._retry_init_args is not None

    def _boom(self):
        raise ValueError("boundary probe: unrelated failure after Beam creation")

    monkeypatch.setattr(MnemosyneMemoryProvider, "_init_audit_log", _boom)
    provider._retry_init_at = 0.0

    with pytest.raises(ValueError, match="boundary probe: unrelated failure after Beam creation"):
        provider.system_prompt_block()

    # The Beam was already live when the unrelated error hit.
    assert provider._beam is not None
    assert provider._init_error is None or "boundary probe" not in str(provider._init_error)
