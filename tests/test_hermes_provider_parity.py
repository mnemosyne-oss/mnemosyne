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
    saved_mnemosyne_modules = {
        name: module for name, module in sys.modules.items()
        if name == "mnemosyne" or name.startswith("mnemosyne.")
    }
    _drop_modules("mnemosyne")
    inserted = [str(import_root)]
    if import_root != PROJECT_ROOT:
        inserted.append(str(PROJECT_ROOT))
    for path in reversed(inserted):
        sys.path.insert(0, path)
    try:
        return importlib.import_module(package)
    finally:
        for path in inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass
        for name in list(sys.modules):
            if name == "mnemosyne" or name.startswith("mnemosyne."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_mnemosyne_modules)


@pytest.fixture(scope="module")
def provider_modules():
    return {
        "hermes_memory_provider": _import_module("hermes_memory_provider", PROJECT_ROOT),
        "mnemosyne_hermes": _import_module("mnemosyne_hermes", INTEGRATION_SRC),
    }


@pytest.fixture(scope="module")
def sync_modules():
    return {
        "hermes_memory_provider": _import_module("hermes_memory_provider.sync_adapter", PROJECT_ROOT),
        "mnemosyne_hermes": _import_module("mnemosyne_hermes.sync_adapter", INTEGRATION_SRC),
    }


def _tool_schemas(module):
    return {schema["name"]: schema for schema in module.ALL_TOOL_SCHEMAS}


def _config_schema(module):
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    return {entry["key"]: entry for entry in provider.get_config_schema()}


def _write_mnemosyne_config(hermes_home: Path, tools) -> None:
    if tools is None:
        body = "memory:\n  provider: mnemosyne\n  mnemosyne: {}\n"
    else:
        rendered_tools = "\n".join(f"      - {tool}" for tool in tools)
        body = (
            "memory:\n"
            "  provider: mnemosyne\n"
            "  mnemosyne:\n"
            "    tools:\n"
            f"{rendered_tools}\n"
        )
    (hermes_home / "config.yaml").write_text(body)


def _schema_names(provider) -> list[str]:
    return [schema["name"] for schema in provider.get_tool_schemas()]


def _provider_for_config(module, hermes_home: Path):
    provider = module.MnemosyneMemoryProvider()
    provider._hermes_home = str(hermes_home)
    return provider


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
    assert root_config["sync_roles"]["default"] == ["user"]
    assert root_config["default_scope"]["choices"] == ["session", "global"]
    assert root_config["default_scope"]["default"] == "session"
    assert root_config["tools"]["default"] is None


def test_tool_whitelist_omitted_exposes_all_tools(tmp_path, provider_modules):
    _write_mnemosyne_config(tmp_path, None)

    observed = {}
    for name, module in provider_modules.items():
        provider = _provider_for_config(module, tmp_path)
        observed[name] = _schema_names(provider)

    all_tools = list(_tool_schemas(provider_modules["hermes_memory_provider"]))
    assert observed["hermes_memory_provider"] == all_tools
    assert observed["mnemosyne_hermes"] == all_tools


def test_tool_whitelist_filters_schemas_before_routing(tmp_path, provider_modules):
    allowed = ["mnemosyne_remember", "mnemosyne_recall", "mnemosyne_sleep"]
    _write_mnemosyne_config(tmp_path, allowed)

    observed = {}
    for name, module in provider_modules.items():
        provider = _provider_for_config(module, tmp_path)
        observed[name] = _schema_names(provider)
        assert provider.has_tool("mnemosyne_remember") is True
        assert provider.has_tool("mnemosyne_forget") is False
        rejected = json.loads(provider.handle_tool_call("mnemosyne_forget", {"memory_id": "x"}))
        assert rejected == {"error": "Unknown Mnemosyne tool: mnemosyne_forget"}

    assert observed["hermes_memory_provider"] == allowed
    assert observed["mnemosyne_hermes"] == allowed
    assert "mnemosyne_forget" not in observed["hermes_memory_provider"]
    # Hermes builds its tool routing map from exposed schemas; filtered-out
    # names must therefore be absent from that registration surface.
    assert "mnemosyne_forget" not in set(observed["mnemosyne_hermes"])


def test_tool_whitelist_empty_list_exposes_no_tools(tmp_path, provider_modules):
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  provider: mnemosyne\n"
        "  mnemosyne:\n"
        "    tools: []\n"
    )

    for module in provider_modules.values():
        provider = _provider_for_config(module, tmp_path)
        assert provider.get_tool_schemas() == []


def test_tool_whitelist_unknown_name_fails_loudly(tmp_path, provider_modules):
    _write_mnemosyne_config(tmp_path, ["mnemosyne_remember", "mnemosyne_not_real"])

    for module in provider_modules.values():
        provider = _provider_for_config(module, tmp_path)
        with pytest.raises(ValueError, match="Unknown Mnemosyne tool.*mnemosyne_not_real"):
            provider.get_tool_schemas()


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


def test_sync_adapter_schema_and_lifecycle_surface_match(sync_modules):
    root_sync = sync_modules["hermes_memory_provider"]
    integration_sync = sync_modules["mnemosyne_hermes"]

    assert _json_stable(integration_sync.ALL_SYNC_TOOL_SCHEMAS) == _json_stable(root_sync.ALL_SYNC_TOOL_SCHEMAS)

    for module in sync_modules.values():
        adapter = module.SyncAdapter.__new__(module.SyncAdapter)
        adapter._engine = object()
        assert adapter.start() is True
        assert _json_stable(adapter.tool_schemas) == _json_stable(root_sync.ALL_SYNC_TOOL_SCHEMAS)
        adapter.shutdown()
        assert adapter.tool_schemas == []


class _FakeSyncEngine:
    def __init__(self, beam_instance, encryption=None):
        self.beam_instance = beam_instance
        self.encryption = encryption
        self.device_id = "fake-device"


class _FakeSyncEncryption:
    def __init__(self, key_source):
        self.key_source = key_source

    @classmethod
    def from_config(cls, key_source=None, **_kwargs):
        return cls(key_source)


class _UnexpectedBeam:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def _install_fake_sync_modules(monkeypatch):
    import types

    fake_sync = types.ModuleType("mnemosyne.core.sync")
    fake_sync.SyncEngine = _FakeSyncEngine
    fake_sync.SyncEncryption = _FakeSyncEncryption
    fake_beam = types.ModuleType("mnemosyne.core.beam")
    fake_beam.BeamMemory = _UnexpectedBeam
    monkeypatch.setitem(sys.modules, "mnemosyne.core.sync", fake_sync)
    monkeypatch.setitem(sys.modules, "mnemosyne.core.beam", fake_beam)


def test_sync_adapter_uses_provider_beam_for_both_surfaces(monkeypatch, sync_modules):
    _install_fake_sync_modules(monkeypatch)

    provider_beam = object()
    for module in sync_modules.values():
        adapter = module.SyncAdapter(provider_beam, {})
        assert adapter.is_ready is True
        assert adapter._engine.beam_instance is provider_beam


def test_sync_adapter_config_resolution_matches(monkeypatch, sync_modules):
    _install_fake_sync_modules(monkeypatch)
    monkeypatch.delenv("MNEMOSYNE_SYNC_REMOTE", raising=False)
    monkeypatch.setenv("MNEMOSYNE_SYNC_HOST", "sync.example")
    monkeypatch.setenv("MNEMOSYNE_SYNC_PORT", "443")

    observed = {}
    for name, module in sync_modules.items():
        adapter = module.SyncAdapter(object(), {"encrypt": True, "key": "encoded-key"})
        observed[name] = {
            "remote": adapter.remote,
            "encryption_key_source": adapter._engine.encryption.key_source,
        }

    assert observed["mnemosyne_hermes"] == observed["hermes_memory_provider"]
    assert observed["hermes_memory_provider"] == {
        "remote": "https://sync.example:443",
        "encryption_key_source": "encoded-key",
    }


class _ToolEngine:
    device_id = "device-1"

    def __init__(self):
        self.meta = {"last_sync_cursor": "cursor-previous"}
        self.conn = self

    def _meta_get(self, key):
        return self.meta.get(key)

    def _meta_set(self, key, value):
        self.meta[key] = value

    def pull_changes(self, since_cursor=None, limit=500):
        return {"events": [{"id": "e1"}], "next_cursor": "local-cursor"}

    def push_changes(self, events):
        self.pushed_events = events
        return {"accepted": 2, "duplicates": 1, "conflicts": 1}

    def execute(self, _sql):
        return self

    def fetchone(self):
        return (3,)


def _adapter_with_tool_engine(module, *, next_cursor: str | None = "remote-cursor"):
    adapter = module.SyncAdapter.__new__(module.SyncAdapter)
    adapter._engine = _ToolEngine()
    adapter._error = None
    adapter.remote = "https://sync.example"
    adapter.encrypt_enabled = False
    adapter.mode = "bidirectional"
    adapter.auth_token = ""

    def fake_post(_path, _payload):
        return {
            "status": "ok",
            "accepted": 2,
            "duplicates": 1,
            "conflicts": 1,
            "events": [{"id": "remote-1"}, {"id": "remote-2"}],
            "next_cursor": next_cursor,
        }

    adapter._http_post = fake_post
    adapter._post = fake_post
    return adapter


def test_sync_adapter_tool_results_match(sync_modules):
    observed = {}
    for name, module in sync_modules.items():
        adapter = _adapter_with_tool_engine(module)
        observed[name] = {
            "push": json.loads(adapter.handle_tool_call("mnemosyne_sync_push", {})),
            "pull": json.loads(adapter.handle_tool_call("mnemosyne_sync_pull", {})),
            "status": json.loads(adapter.handle_tool_call("mnemosyne_sync_status", {})),
            "unknown": json.loads(adapter.handle_tool_call("mnemosyne_sync_unknown", {})),
        }

    assert observed["mnemosyne_hermes"] == observed["hermes_memory_provider"]
    assert observed["hermes_memory_provider"]["push"] == {
        "status": "ok",
        "pushed": 2,
        "duplicates": 1,
        "conflicts": 1,
        "next_cursor": "remote-cursor",
    }
    assert observed["hermes_memory_provider"]["pull"] == {
        "status": "ok",
        "pulled": 2,
        "duplicates": 1,
        "conflicts": 1,
        "next_cursor": "remote-cursor",
    }


def test_sync_adapter_pull_tolerates_null_next_cursor(sync_modules):
    observed = {}
    for name, module in sync_modules.items():
        adapter = _adapter_with_tool_engine(module, next_cursor=None)
        observed[name] = json.loads(adapter.handle_tool_call("mnemosyne_sync_pull", {}))

    assert observed["mnemosyne_hermes"] == observed["hermes_memory_provider"]
    assert observed["hermes_memory_provider"] == {
        "status": "ok",
        "pulled": 2,
        "duplicates": 1,
        "conflicts": 1,
        "next_cursor": "",
    }



def _save_mnemosyne_modules():
    return {
        name: module for name, module in sys.modules.items()
        if name == "mnemosyne" or name.startswith("mnemosyne.")
    }


def _restore_mnemosyne_modules(saved_modules):
    for name in list(sys.modules):
        if name == "mnemosyne" or name.startswith("mnemosyne."):
            sys.modules.pop(name, None)
    sys.modules.update(saved_modules)


def test_provider_persona_tool_dispatch_matches(tmp_path, provider_modules):
    saved_mnemosyne_modules = _save_mnemosyne_modules()
    _drop_modules("mnemosyne")
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
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
    finally:
        try:
            sys.path.remove(str(PROJECT_ROOT))
        except ValueError:
            pass
        _restore_mnemosyne_modules(saved_mnemosyne_modules)

    assert observed["hermes_memory_provider"] == observed["mnemosyne_hermes"]
    assert observed["hermes_memory_provider"] == {
        "status": "ok",
        "count": 1,
        "topics": ["test"],
    }
