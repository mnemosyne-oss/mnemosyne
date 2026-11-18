"""Plugin-registered tools must use the initialized memory provider."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import mnemosyne_hermes as plugin
from mnemosyne_hermes import MnemosyneMemoryProvider


class _Context:
    def __init__(self):
        self.tools = {}
        self.provider = None

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_cli_command(self, **_kwargs):
        pass

    def register_tool(self, *, name, handler, **_kwargs):
        self.tools[name] = handler


class _BindingProvider:
    def __init__(self):
        self.identity = None
        self.beam = None

    def initialize(self, _session_id, *, agent_identity, **_kwargs):
        self.identity = agent_identity
        self.beam = object()

    def get_tool_schemas(self):
        return [
            {"name": "mnemosyne_remember"},
            {"name": "mnemosyne_sync_status"},
            {"name": "mnemosyne_persona_list"},
        ]

    def handle_tool_call(self, tool_name, _arguments):
        return f"memory:{self.identity}:{tool_name}"

    def _handle_sync_tool(self, tool_name, _arguments):
        return f"sync:{self.identity}:{tool_name}"

    def _handle_persona_tool(self, tool_name, _arguments):
        return f"persona:{self.identity}:{tool_name}"


def test_register_uses_one_provider_per_call_for_manager_and_tools(monkeypatch):
    created = []

    class CountingProvider(MnemosyneMemoryProvider):
        def __init__(self):
            super().__init__()
            created.append(self)

    monkeypatch.setattr(plugin, "MnemosyneMemoryProvider", CountingProvider)
    monkeypatch.setattr(plugin, "_provider", None)

    context = _Context()
    plugin.register(context)

    assert len(created) == 1
    assert context.provider is created[0]
    assert plugin._provider is created[0]
    for tool_name in (
        "mnemosyne_remember",
        "mnemosyne_sync_status",
        "mnemosyne_persona_list",
    ):
        handler = context.tools[tool_name]
        bound = getattr(handler, "func", handler)
        assert getattr(bound, "__self__", None) is created[0]


def test_memory_provider_registration_returns_distinct_profile_instances(monkeypatch):
    monkeypatch.setattr(plugin, "MnemosyneMemoryProvider", _BindingProvider)
    monkeypatch.setattr(plugin, "_provider", None)

    profile_a = _Context()
    profile_b = _Context()
    plugin.register_memory_provider(profile_a)
    profile_a.provider.initialize("session-a", agent_identity="profile-a")
    beam_a = profile_a.provider.beam

    plugin.register_memory_provider(profile_b)
    profile_b.provider.initialize("session-b", agent_identity="profile-b")

    assert profile_a.provider is not profile_b.provider
    assert profile_a.provider.identity == "profile-a"
    assert profile_a.provider.beam is beam_a
    assert profile_b.provider.identity == "profile-b"
    assert plugin._provider is profile_a.provider


def test_repeated_and_concurrent_memory_provider_registrations_are_distinct(monkeypatch):
    monkeypatch.setattr(plugin, "MnemosyneMemoryProvider", _BindingProvider)
    monkeypatch.setattr(plugin, "_provider", None)

    contexts = [_Context() for _ in range(16)]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(plugin.register_memory_provider, contexts))

    providers = [context.provider for context in contexts]
    assert len({id(provider) for provider in providers}) == len(contexts)
    assert plugin._provider in providers


def test_register_handlers_stay_bound_after_later_profile_registration(monkeypatch):
    monkeypatch.setattr(plugin, "MnemosyneMemoryProvider", _BindingProvider)
    monkeypatch.setattr(plugin, "_provider", None)

    profile_a = _Context()
    plugin.register(profile_a)
    profile_a.provider.initialize("session-a", agent_identity="profile-a")

    profile_b = _Context()
    plugin.register(profile_b)
    profile_b.provider.initialize("session-b", agent_identity="profile-b")

    assert profile_a.provider is not profile_b.provider
    assert profile_a.tools["mnemosyne_remember"]({}) == (
        "memory:profile-a:mnemosyne_remember"
    )
    assert profile_a.tools["mnemosyne_sync_status"]({}) == (
        "sync:profile-a:mnemosyne_sync_status"
    )
    assert profile_a.tools["mnemosyne_persona_list"]({}) == (
        "persona:profile-a:mnemosyne_persona_list"
    )
    assert profile_b.tools["mnemosyne_remember"]({}) == (
        "memory:profile-b:mnemosyne_remember"
    )


def test_plugin_remember_without_prior_initialize_stores(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugin, "_provider", None)

    context = _Context()
    plugin.register(context)
    try:
        raw = context.tools["mnemosyne_remember"](
            {"content": "user prefers tea", "importance": 0.9, "scope": "global"}
        )
        data = json.loads(raw)

        assert data.get("status") != "memory_unavailable", data
        assert "not initialized" not in str(data).lower()
        assert data.get("status") == "stored"
        assert data.get("memory_id")
    finally:
        if context.provider is not None:
            context.provider.shutdown()
