"""Reflection/sleep guardrail tests for the Hermes memory provider."""

from __future__ import annotations

import json
from typing import Optional, cast

from hermes_memory_provider import MnemosyneMemoryProvider


class FakeBeam:
    def __init__(self):
        self.sleep_calls = 0
        self.sleep_all_sessions_calls = 0

    def sleep(self, *args, **kwargs):
        self.sleep_calls += 1
        return {"status": "consolidated"}

    def sleep_all_sessions(self, *args, **kwargs):
        self.sleep_all_sessions_calls += 1
        return {"status": "consolidated"}

    def get_working_stats(self):
        return {"total": 99}

    def get_episodic_stats(self):
        return {"total": 1}

    def _count_unconsolidated_before(self, cutoff):
        return 1


def _provider(*, context="primary", max_calls: Optional[int] = 3, disabled_for_cron=True):
    provider = MnemosyneMemoryProvider()
    provider._beam = FakeBeam()
    provider._agent_context = context
    provider._reflect_max_calls_per_session = max_calls
    provider._reflect_disabled_for_cron = disabled_for_cron
    return provider


def _fake_beam(provider: MnemosyneMemoryProvider) -> FakeBeam:
    return cast(FakeBeam, provider._beam)


def _call(provider: MnemosyneMemoryProvider, name: str, args: dict | None = None) -> dict:
    return json.loads(provider.handle_tool_call(name, args or {}))


def test_sleep_skips_in_cron_context_even_before_beam_init():
    provider = MnemosyneMemoryProvider()
    provider._beam = None
    provider._agent_context = "cron"
    provider._reflect_disabled_for_cron = True

    result = _call(provider, "mnemosyne_sleep")

    assert result["status"] == "skipped"
    assert result["reason"] == "reflect_disabled_for_cron"
    assert result["trigger"] == "tool"


def test_sleep_allows_cron_when_guard_disabled():
    provider = _provider(context="cron", disabled_for_cron=False)

    result = _call(provider, "mnemosyne_sleep")

    assert result["status"] == "consolidated"
    assert _fake_beam(provider).sleep_calls == 1


def test_sleep_budget_skips_after_max_calls_per_session():
    provider = _provider(max_calls=1)

    first = _call(provider, "mnemosyne_sleep")
    second = _call(provider, "mnemosyne_sleep")

    assert first["status"] == "consolidated"
    assert second["status"] == "skipped"
    assert second["reason"] == "reflect_budget_exhausted"
    assert _fake_beam(provider).sleep_calls == 1


def test_negative_budget_disables_cap():
    provider = _provider(max_calls=None)

    _call(provider, "mnemosyne_sleep")
    _call(provider, "mnemosyne_sleep")

    assert _fake_beam(provider).sleep_calls == 2


def test_auto_sleep_respects_budget_without_calling_sleep():
    provider = _provider(max_calls=0)
    provider._auto_sleep_threshold = 1

    provider._maybe_auto_sleep()

    assert _fake_beam(provider).sleep_calls == 0
    assert _fake_beam(provider).sleep_all_sessions_calls == 0


def test_session_end_respects_budget_without_calling_sleep():
    provider = _provider(max_calls=0)

    provider.on_session_end([])

    assert _fake_beam(provider).sleep_calls == 0
