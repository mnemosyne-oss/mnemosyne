"""Plugin-registered tools must use the initialized memory provider."""

from __future__ import annotations

import json

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


def test_register_uses_one_provider_for_manager_and_tools(monkeypatch):
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
    remember = context.tools["mnemosyne_remember"]
    bound = getattr(remember, "func", remember)
    assert getattr(bound, "__self__", None) is created[0]


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
