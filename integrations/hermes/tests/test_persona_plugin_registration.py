"""Regression tests for standalone-plugin persona tool registration."""

import mnemosyne_hermes as plugin
from mnemosyne_hermes import persona_adapter


class _Context:
    def __init__(self):
        self.tools = {}

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_cli_command(self, **_kwargs):
        pass

    def register_tool(self, *, name, handler, **_kwargs):
        self.tools[name] = handler


class _Provider:
    def __init__(self):
        self._beam = object()

    def handle_tool_call(self, _tool_name, _arguments):
        return '{"status": "ok"}'

    def _handle_persona_tool(self, tool_name, arguments):
        adapter = persona_adapter.PersonaAdapter(beam_instance=self._beam)
        return adapter.handle_tool_call(tool_name, arguments)

    def get_tool_schemas(self):
        return [{"name": "mnemosyne_persona_promote", "description": ""}]


class _PersonaAdapter:
    received_beam = None

    def __init__(self, *, beam_instance):
        type(self).received_beam = beam_instance

    def handle_tool_call(self, _tool_name, _arguments):
        return '{"status": "ok"}'


def test_persona_tool_uses_plugin_registered_provider_beam(monkeypatch):
    """Persona tools share the standalone plugin provider's BeamMemory instance."""
    provider = _Provider()
    context = _Context()

    monkeypatch.setattr(plugin, "MnemosyneMemoryProvider", lambda: provider)
    monkeypatch.setattr(persona_adapter, "PersonaAdapter", _PersonaAdapter)
    monkeypatch.delattr(plugin, "_provider", raising=False)
    monkeypatch.setattr(plugin, "_persona_adapter", None)
    _PersonaAdapter.received_beam = None

    plugin.register(context)
    context.tools["mnemosyne_persona_promote"]({"memory_id": "memory-1"})

    assert _PersonaAdapter.received_beam is provider._beam


def test_plugin_registration_honors_tool_allowlist_before_initialize(tmp_path, monkeypatch):
    """Standalone PluginManager registration must not advertise excluded tools."""
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  mnemosyne:\n"
        "    tools:\n"
        "      - mnemosyne_remember\n"
        "      - mnemosyne_recall\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(plugin, "_provider", None)

    context = _Context()
    plugin.register(context)

    assert list(context.tools) == ["mnemosyne_remember", "mnemosyne_recall"]


def test_plugin_registration_without_allowlist_exposes_full_surface(monkeypatch):
    """Standalone PluginManager preserves the complete historical tool surface."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(plugin, "_provider", None)

    context = _Context()
    plugin.register(context)

    assert list(context.tools) == [schema["name"] for schema in plugin.ALL_TOOL_SCHEMAS]
