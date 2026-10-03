from __future__ import annotations

import json
import logging

import mnemosyne_hermes
from mnemosyne_hermes import MnemosyneMemoryProvider


def test_skip_context_reinit_reports_destroyed_live_beam(tmp_path, caplog):
    """A primary -> skip-context re-init must explain the intentional reset."""
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        "primary",
        hermes_home=str(tmp_path),
        profile_isolation=False,
        agent_context="primary",
    )
    assert provider._beam is not None

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        provider.initialize(
            "subagent",
            hermes_home=str(tmp_path),
            profile_isolation=False,
            agent_context="subagent",
        )

    payload = json.loads(
        provider.handle_tool_call("mnemosyne_remember", {"content": "x"})
    )
    assert provider._beam is None
    assert payload["status"] == "memory_unavailable"
    assert payload["reason_code"] == "reset_by_reinit"
    assert "subagent" in payload["reason"]
    assert "UNAVAILABLE" in provider.system_prompt_block()
    assert "dropped a live beam" in caplog.text

    provider.initialize(
        "recovered-primary",
        hermes_home=str(tmp_path),
        profile_isolation=False,
        agent_context="primary",
    )
    recovered = json.loads(
        provider.handle_tool_call("mnemosyne_remember", {"content": "recovered"})
    )
    assert provider._beam is not None
    assert recovered.get("status") != "memory_unavailable"
    assert "UNAVAILABLE" not in provider.system_prompt_block()


def test_first_skip_context_init_is_distinct_from_reset(tmp_path, caplog):
    """A provider that never held a Beam remains a silent skip-context session."""
    provider = MnemosyneMemoryProvider()

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        provider.initialize(
            "subagent",
            hermes_home=str(tmp_path),
            profile_isolation=False,
            agent_context="subagent",
        )

    payload = json.loads(
        provider.handle_tool_call("mnemosyne_remember", {"content": "x"})
    )
    assert payload["status"] == "memory_unavailable"
    assert payload["reason_code"] == "skipped_context"
    assert provider.system_prompt_block() == ""
    assert "dropped a live beam" not in caplog.text


def test_never_initialized_reason_code_is_distinct():
    provider = MnemosyneMemoryProvider()
    provider._agent_context = "subagent"

    payload = json.loads(
        provider.handle_tool_call("mnemosyne_remember", {"content": "x"})
    )

    assert payload["reason_code"] == "never_initialized"
    assert payload["reason"] == "Mnemosyne not initialized"


def test_initialization_failure_has_its_own_reason_code(monkeypatch, tmp_path):
    class _CorruptBeam:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("corrupt test database")

    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _CorruptBeam)
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        "primary",
        hermes_home=str(tmp_path),
        profile_isolation=False,
        agent_context="primary",
    )

    payload = json.loads(
        provider.handle_tool_call("mnemosyne_remember", {"content": "x"})
    )
    assert payload["reason_code"] == "init_failed"
    assert payload["reason"].startswith("RuntimeError:")
