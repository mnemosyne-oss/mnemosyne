"""Parity checks for the two Hermes Mnemosyne provider implementations."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"


def _drop_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]


def _import_module(package: str, import_root: Path):
    _drop_modules(package)
    sys.path.insert(0, str(import_root))
    try:
        return importlib.import_module(package)
    finally:
        try:
            sys.path.remove(str(import_root))
        except ValueError:
            pass


@pytest.fixture(scope="module")
def provider_modules():
    return {
        "hermes_memory_provider": _import_module("hermes_memory_provider", PROJECT_ROOT),
        "mnemosyne_hermes": _import_module("mnemosyne_hermes", INTEGRATION_SRC),
    }


def _tool_schemas(module):
    return {schema["name"]: schema for schema in module.ALL_TOOL_SCHEMAS}


def _config_schema(module):
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    return {entry["key"]: entry for entry in provider.get_config_schema()}


def _json_stable(value):
    return json.loads(json.dumps(value, sort_keys=True))


def test_provider_tool_sets_match(provider_modules):
    tool_sets = {name: set(_tool_schemas(module)) for name, module in provider_modules.items()}

    assert tool_sets["hermes_memory_provider"] == tool_sets["mnemosyne_hermes"]
    assert "mnemosyne_sync_push" in tool_sets["hermes_memory_provider"]
    assert "mnemosyne_persona_list" in tool_sets["hermes_memory_provider"]
    assert "mnemosyne_triple_end" in tool_sets["hermes_memory_provider"]


def test_provider_tool_schemas_match(provider_modules):
    root_tools = _tool_schemas(provider_modules["hermes_memory_provider"])
    integration_tools = _tool_schemas(provider_modules["mnemosyne_hermes"])

    assert _json_stable(root_tools) == _json_stable(integration_tools)


def test_provider_config_defaults_match(provider_modules):
    root_config = _config_schema(provider_modules["hermes_memory_provider"])
    integration_config = _config_schema(provider_modules["mnemosyne_hermes"])

    assert _json_stable(root_config) == _json_stable(integration_config)
    assert root_config["sync_roles"]["default"] == ["user", "assistant"]
    assert root_config["default_scope"]["choices"] == ["session", "global"]
    assert root_config["default_scope"]["default"] == "session"


@pytest.mark.parametrize(
    ("env_name", "helper_name", "default", "custom"),
    [
        ("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "_sync_turn_user_limit", 500, 123),
        ("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "_sync_turn_assistant_limit", 800, 234),
    ],
)
def test_provider_sync_limit_helpers_match(monkeypatch, provider_modules, env_name, helper_name, default, custom):
    monkeypatch.delenv(env_name, raising=False)
    assert {name: getattr(module, helper_name)() for name, module in provider_modules.items()} == {
        "hermes_memory_provider": default,
        "mnemosyne_hermes": default,
    }

    monkeypatch.setenv(env_name, str(custom))
    assert {name: getattr(module, helper_name)() for name, module in provider_modules.items()} == {
        "hermes_memory_provider": custom,
        "mnemosyne_hermes": custom,
    }

    monkeypatch.setenv(env_name, "-10")
    assert {name: getattr(module, helper_name)() for name, module in provider_modules.items()} == {
        "hermes_memory_provider": 0,
        "mnemosyne_hermes": 0,
    }

    monkeypatch.setenv(env_name, "not-an-int")
    assert {name: getattr(module, helper_name)() for name, module in provider_modules.items()} == {
        "hermes_memory_provider": default,
        "mnemosyne_hermes": default,
    }


class _FakeBeam:
    def __init__(self):
        self.calls = []

    def remember(self, **kwargs):
        self.calls.append(kwargs)


def _new_provider(module, *, scope="session", roles=("user", "assistant")):
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    provider._beam = _FakeBeam()
    provider._agent_context = ""
    provider._skip_contexts = set()
    provider._sync_roles = set(roles)
    provider._default_scope = scope
    provider._should_filter = lambda _content: False
    provider._capture_identity_signals = lambda _content: None
    provider._turn_count = 0
    provider._auto_sleep_enabled = False
    return provider


@pytest.mark.parametrize("scope", ["session", "global"])
def test_provider_sync_turn_scope_and_truncation_match(monkeypatch, provider_modules, scope):
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "7")
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "9")

    observed = {}
    for name, module in provider_modules.items():
        provider = _new_provider(module, scope=scope)
        provider.sync_turn("user-content", "assistant-content")
        observed[name] = provider._beam.calls

    assert observed["hermes_memory_provider"] == observed["mnemosyne_hermes"]
    assert [call["scope"] for call in observed["hermes_memory_provider"]] == [scope, scope]
    assert [call["content"] for call in observed["hermes_memory_provider"]] == [
        "[USER] user-co",
        "[ASSISTANT] assistant",
    ]


def test_provider_sync_turn_zero_limit_means_untruncated(monkeypatch, provider_modules):
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_USER_LIMIT", "0")
    monkeypatch.setenv("MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT", "0")

    observed = {}
    for name, module in provider_modules.items():
        provider = _new_provider(module)
        provider.sync_turn("user-content", "assistant-content")
        observed[name] = [call["content"] for call in provider._beam.calls]

    assert observed["hermes_memory_provider"] == observed["mnemosyne_hermes"]
    assert observed["hermes_memory_provider"] == [
        "[USER] user-content",
        "[ASSISTANT] assistant-content",
    ]


def test_provider_persona_tool_dispatch_matches(tmp_path, provider_modules):
    from mnemosyne.core.beam import BeamMemory

    observed = {}
    for name, module in provider_modules.items():
        db_path = tmp_path / f"{name}.db"
        beam = BeamMemory(session_id=f"persona-{name}", db_path=str(db_path))
        beam.conn.execute(
            "INSERT INTO memoria_persona (tier, topic, content, confidence) "
            "VALUES (?, ?, ?, ?)",
            ("long_term", "test", f"persona rule for {name}", 0.9),
        )
        beam.conn.commit()

        provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
        provider._beam = beam
        result = json.loads(provider.handle_tool_call("mnemosyne_persona_list", {}))
        observed[name] = {
            "status": result.get("status"),
            "count": result.get("count"),
            "topics": [row.get("topic") for row in result.get("personas", [])],
        }

    assert observed["hermes_memory_provider"] == observed["mnemosyne_hermes"]
    assert observed["hermes_memory_provider"] == {
        "status": "ok",
        "count": 1,
        "topics": ["test"],
    }
