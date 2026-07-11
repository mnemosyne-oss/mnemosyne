"""Parity checks for the two Hermes Mnemosyne provider implementations."""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
import yaml


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
    assert root_config["auto_sleep"]["default"] is True
    assert root_config["sync_roles"]["default"] == ["user"]
    assert root_config["default_scope"]["choices"] == ["session", "global"]
    assert root_config["default_scope"]["default"] == "session"
    assert root_config["tools"]["default"] is None


def test_auto_sleep_runtime_default_enabled(monkeypatch, provider_modules):
    monkeypatch.delenv("MNEMOSYNE_AUTO_SLEEP_ENABLED", raising=False)

    for module in provider_modules.values():
        provider = module.MnemosyneMemoryProvider()
        assert provider._auto_sleep_enabled is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off"])
def test_auto_sleep_env_can_disable_default(monkeypatch, provider_modules, value):
    monkeypatch.setenv("MNEMOSYNE_AUTO_SLEEP_ENABLED", value)

    for module in provider_modules.values():
        provider = module.MnemosyneMemoryProvider()
        assert provider._auto_sleep_enabled is False


@pytest.mark.parametrize("configured", [False, "false", 0])
def test_auto_sleep_config_can_disable_default(tmp_path, monkeypatch, provider_modules, configured):
    monkeypatch.delenv("MNEMOSYNE_AUTO_SLEEP_ENABLED", raising=False)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"memory": {"provider": "mnemosyne", "mnemosyne": {"auto_sleep": configured}}})
    )

    for module in provider_modules.values():
        provider = _provider_for_config(module, tmp_path)
        provider._apply_provider_config({})
        assert provider._auto_sleep_enabled is False


@pytest.mark.parametrize(
    ("env_value", "config_value", "kwarg_value", "expected"),
    [
        ("0", False, True, True),
        ("1", True, False, False),
        ("0", False, "true", True),
        ("1", True, "false", False),
    ],
)
def test_auto_sleep_kwargs_have_highest_precedence(
    tmp_path, monkeypatch, provider_modules, env_value, config_value, kwarg_value, expected
):
    monkeypatch.setenv("MNEMOSYNE_AUTO_SLEEP_ENABLED", env_value)
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"memory": {"provider": "mnemosyne", "mnemosyne": {"auto_sleep": config_value}}})
    )

    for module in provider_modules.values():
        provider = _provider_for_config(module, tmp_path)
        provider._apply_provider_config({"auto_sleep": kwarg_value})
        assert provider._auto_sleep_enabled is expected


def test_save_config_persists_auto_sleep_default_when_missing(tmp_path, provider_modules):
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  provider: mnemosyne\n"
        "  mnemosyne:\n"
        "    sleep_threshold: 75\n"
    )

    for name, module in provider_modules.items():
        hermes_home = tmp_path / name
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text((tmp_path / "config.yaml").read_text())

        provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
        provider.save_config({}, str(hermes_home))

        cfg = yaml.safe_load((hermes_home / "config.yaml").read_text())
        mnemosyne_cfg = cfg["memory"]["mnemosyne"]
        assert mnemosyne_cfg["auto_sleep"] is True
        assert mnemosyne_cfg["sleep_threshold"] == 75


def test_save_config_respects_auto_sleep_env_opt_out(tmp_path, monkeypatch, provider_modules):
    monkeypatch.setenv("MNEMOSYNE_AUTO_SLEEP_ENABLED", "0")

    for name, module in provider_modules.items():
        hermes_home = tmp_path / name
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "memory:\n"
            "  provider: mnemosyne\n"
            "  mnemosyne:\n"
            "    sleep_threshold: 75\n"
        )

        provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
        provider.save_config({}, str(hermes_home))

        cfg = yaml.safe_load((hermes_home / "config.yaml").read_text())
        mnemosyne_cfg = cfg["memory"]["mnemosyne"]
        assert mnemosyne_cfg["auto_sleep"] is False
        assert mnemosyne_cfg["sleep_threshold"] == 75


def test_save_config_preserves_explicit_auto_sleep_false(tmp_path, provider_modules):
    for name, module in provider_modules.items():
        hermes_home = tmp_path / name
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(
            "memory:\n"
            "  provider: mnemosyne\n"
            "  mnemosyne:\n"
            "    auto_sleep: false\n"
        )

        provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
        provider.save_config({}, str(hermes_home))

        cfg = yaml.safe_load((hermes_home / "config.yaml").read_text())
        assert cfg["memory"]["mnemosyne"]["auto_sleep"] is False


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
        assert provider.has_tool("mnemosyne_batch") is False
        rejected = json.loads(provider.handle_tool_call("mnemosyne_forget", {"memory_id": "x"}))
        assert rejected == {"error": "Unknown Mnemosyne tool: mnemosyne_forget"}
        rejected_batch = json.loads(provider.handle_tool_call("mnemosyne_batch", {"operations": []}))
        assert rejected_batch == {"error": "Unknown Mnemosyne tool: mnemosyne_batch"}

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


def test_config_reader_tolerates_null_and_non_mapping_levels(tmp_path):
    from mnemosyne.hermes_config import read_hermes_config_key

    cases = [
        "memory:\n",
        "memory: []\n",
        "memory:\n  mnemosyne:\n",
        "memory:\n  mnemosyne: []\n",
        "[]\n",
    ]
    for index, body in enumerate(cases):
        hermes_home = tmp_path / f"case-{index}"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(body)
        assert read_hermes_config_key(str(hermes_home), "tools") is None


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
    provider._audit_event = lambda *args, **kwargs: None
    return provider


def test_provider_remember_extract_uses_default_scope(provider_modules):
    observed = {}
    for name, module in provider_modules.items():
        provider = _new_provider(module, scope="session")
        result = json.loads(provider._handle_remember({
            "content": f"extract scope {name}",
            "extract": True,
        }))
        observed[name] = {
            "status": result.get("status"),
            "scope": provider._beam.calls[0]["scope"],
        }

    assert observed["hermes_memory_provider"] == observed["mnemosyne_hermes"]
    assert observed["hermes_memory_provider"] == {"status": "stored", "scope": "session"}


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
    def __init__(
        self,
        beam_instance,
        encryption=None,
        require_encryption=False,
        surface_only=False,
        initialize_surface=False,
    ):
        self.beam_instance = beam_instance
        self.encryption = encryption
        self.require_encryption = require_encryption
        self.surface_only = surface_only
        self.surface_id = "shared-surface-v1" if surface_only else None
        self.initialize_surface = initialize_surface
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


def test_provider_sync_tools_are_bound_to_shared_surface(
    monkeypatch, provider_modules, sync_modules
):
    captured = {}

    class CaptureAdapter:
        def __init__(self, beam, _config):
            captured["beam"] = beam

        def handle_tool_call(self, tool_name, _args):
            return json.dumps({"tool": tool_name})

    for name, provider_module in provider_modules.items():
        sync_module = sync_modules[name]
        monkeypatch.setattr(sync_module, "SyncAdapter", CaptureAdapter)
        provider = provider_module.MnemosyneMemoryProvider.__new__(
            provider_module.MnemosyneMemoryProvider
        )
        provider._beam = object()
        surface_beam = object()
        provider._surface_beam = surface_beam
        provider._require_surface_beam = lambda: None
        if name == "hermes_memory_provider":
            provider._sync_adapter = None
        else:
            provider._provider_sync_adapter = None

        result = json.loads(
            provider._handle_sync_tool("mnemosyne_sync_status", {})
        )
        assert result["tool"] == "mnemosyne_sync_status"
        assert captured["beam"] is surface_beam
        assert captured["beam"] is not provider._beam


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
            "require_encryption": adapter._engine.require_encryption,
        }

    assert observed["mnemosyne_hermes"] == observed["hermes_memory_provider"]
    assert observed["hermes_memory_provider"] == {
        "remote": "https://sync.example:443",
        "encryption_key_source": "encoded-key",
        "require_encryption": True,
    }


def test_sync_adapter_fails_closed_when_encryption_key_is_missing(
    monkeypatch, sync_modules
):
    _install_fake_sync_modules(monkeypatch)
    monkeypatch.delenv("MNEMOSYNE_SYNC_KEY", raising=False)
    for module in sync_modules.values():
        adapter = module.SyncAdapter(object(), {"encrypt": True})
        assert adapter.is_ready is False
        assert adapter._engine is None
        assert "no key" in adapter._error.lower()


def test_sync_adapter_key_source_file_preserves_path_case(tmp_path, sync_modules):
    key_file = tmp_path / "MixedCaseSync.key"
    key_file.write_text("file-key")
    if os.name != "nt":
        key_file.chmod(0o600)

    observed = {}
    for name, module in sync_modules.items():
        adapter = module.SyncAdapter.__new__(module.SyncAdapter)
        adapter._config = {"key_source": f"FILE:{key_file}"}
        observed[name] = adapter._resolve_key()

    assert observed["mnemosyne_hermes"] == observed["hermes_memory_provider"]
    assert observed["hermes_memory_provider"] == "file-key"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits only")
def test_sync_adapter_rejects_insecure_key_file_permissions(tmp_path, sync_modules):
    key_file = tmp_path / "insecure-sync.key"
    key_file.write_text("file-key")
    key_file.chmod(0o644)

    for module in sync_modules.values():
        adapter = module.SyncAdapter.__new__(module.SyncAdapter)
        adapter._config = {"key_source": f"file:{key_file}"}
        assert adapter._resolve_key() == ""


class _ToolEngine:
    device_id = "device-1"

    def __init__(self, *, local_next_cursor: str | None = "local-cursor"):
        self.meta = {
            "last_pull_cursor_https://sync.example": local_next_cursor or ""
        }
        self.conn = self

    def _meta_get(self, key):
        return self.meta.get(key)

    def _meta_set(self, key, value):
        self.meta[key] = value

    def sync_with(self, remote, mode="bidirectional", api_key=None):
        assert remote == "https://sync.example"
        assert api_key is None
        if mode == "push":
            return {
                "errors": [],
                "push": {
                    "accepted": 2,
                    "duplicates": 1,
                    "conflicts": 1,
                    "batches": 1,
                    "discovered": {"created": 1, "updated": 1, "deleted": 0},
                },
            }
        return {
            "errors": [],
            "pull": {
                "accepted": 2,
                "events_fetched": 2,
                "duplicates": 1,
                "conflicts": 1,
                "batches": 1,
            },
        }

    def execute(self, _sql, _params=None):
        return self

    def fetchone(self):
        return (3,)


def _adapter_with_tool_engine(
    module,
    *,
    next_cursor: str | None = "remote-cursor",
    local_next_cursor: str | None = "local-cursor",
):
    del next_cursor
    adapter = module.SyncAdapter.__new__(module.SyncAdapter)
    adapter._engine = _ToolEngine(local_next_cursor=local_next_cursor)
    adapter._error = None
    adapter.remote = "https://sync.example"
    adapter.encrypt_enabled = False
    adapter.mode = "bidirectional"
    adapter.auth_token = ""
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
        "discovered": {"created": 1, "updated": 1, "deleted": 0},
        "batches": 1,
        "errors": [],
    }
    assert observed["hermes_memory_provider"]["pull"] == {
        "status": "ok",
        "pulled": 2,
        "events_fetched": 2,
        "duplicates": 1,
        "conflicts": 1,
        "batches": 1,
        "errors": [],
    }


def test_sync_adapter_push_does_not_require_transport_cursor(sync_modules):
    observed = {}
    for name, module in sync_modules.items():
        adapter = _adapter_with_tool_engine(module, next_cursor=None, local_next_cursor=None)
        observed[name] = json.loads(adapter.handle_tool_call("mnemosyne_sync_push", {}))

    assert observed["mnemosyne_hermes"] == observed["hermes_memory_provider"]
    assert observed["hermes_memory_provider"]["status"] == "ok"
    assert observed["hermes_memory_provider"]["pushed"] == 2


def test_sync_adapter_pull_does_not_require_transport_cursor(sync_modules):
    observed = {}
    for name, module in sync_modules.items():
        adapter = _adapter_with_tool_engine(module, next_cursor=None)
        observed[name] = json.loads(adapter.handle_tool_call("mnemosyne_sync_pull", {}))

    assert observed["mnemosyne_hermes"] == observed["hermes_memory_provider"]
    assert observed["hermes_memory_provider"]["status"] == "ok"
    assert observed["hermes_memory_provider"]["pulled"] == 2

def _prompt_provider(module):
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    provider._beam = object()
    provider._init_error = None
    if hasattr(provider, "_persona_cache"):
        provider._persona_cache = {"mtime": None, "content": None}
    return provider


def test_provider_persona_prompt_injection_matches(tmp_path, provider_modules):
    persona_file = tmp_path / "persona.md"
    persona_file.write_text(
        "# Persona\n\n"
        "## privacy\n"
        "- expected persona/privacy rule [importance: 0.90]\n"
    )

    observed = {}
    for name, module in provider_modules.items():
        provider = _prompt_provider(module)
        # Class-level env defaults are read at import time; set the attrs
        # directly so both already-imported provider surfaces see this file.
        provider.PERSONA_ENABLED = True
        provider.PERSONA_FILE = persona_file
        observed[name] = provider.system_prompt_block()

    for block in observed.values():
        assert "# Mnemosyne Memory" in block
        assert "# L3 Persona (Active Behavioral Rules)" in block
        assert "expected persona/privacy rule" in block


def test_provider_persona_prompt_silent_when_disabled_or_missing(tmp_path, provider_modules):
    persona_file = tmp_path / "persona.md"
    persona_file.write_text("# Persona\n\n- should stay hidden when disabled\n")
    missing_file = tmp_path / "missing-persona.md"

    for module in provider_modules.values():
        provider = _prompt_provider(module)
        provider.PERSONA_ENABLED = False
        provider.PERSONA_FILE = persona_file
        block = provider.system_prompt_block()
        assert "# L3 Persona" not in block
        assert "should stay hidden when disabled" not in block

        provider = _prompt_provider(module)
        provider.PERSONA_ENABLED = True
        provider.PERSONA_FILE = missing_file
        assert "# L3 Persona" not in provider.system_prompt_block()


def test_provider_persona_negative_token_cap_does_not_slice_from_end(tmp_path, provider_modules):
    persona_file = tmp_path / "persona.md"
    persona_file.write_text("# Persona\n\n## privacy\n- secret tail should not leak\n")

    for module in provider_modules.values():
        provider = _prompt_provider(module)
        provider.PERSONA_ENABLED = True
        provider.PERSONA_FILE = persona_file
        provider.PERSONA_TOKEN_CAP = -10
        block = provider.system_prompt_block()
        assert "secret tail should not leak" not in block
        assert "truncated" in block


@pytest.mark.parametrize("bad_token_cap", ["", "not-an-int"])
def test_provider_persona_token_cap_invalid_env_falls_back(monkeypatch, bad_token_cap):
    monkeypatch.setenv("MNEMOSYNE_PERSONA_TOKEN_CAP", bad_token_cap)

    modules = {
        "hermes_memory_provider": _import_module("hermes_memory_provider", PROJECT_ROOT),
        "mnemosyne_hermes": _import_module("mnemosyne_hermes", INTEGRATION_SRC),
    }

    assert {name: module.MnemosyneMemoryProvider.PERSONA_TOKEN_CAP for name, module in modules.items()} == {
        "hermes_memory_provider": 1500,
        "mnemosyne_hermes": 1500,
    }


def test_packaged_provider_import_survives_missing_core_helpers():
    """Installer/status diagnostics must import even with a broken core install."""

    import importlib.abc

    blocked = {
        "mnemosyne.batch_tool",
        "mnemosyne.hermes_config",
        "mnemosyne.integrations.hermes_persona_prompt",
    }

    class _BlockCoreHelperImports(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in blocked:
                raise ModuleNotFoundError(f"blocked test import: {fullname}")
            return None

    finder = _BlockCoreHelperImports()
    saved = {name: module for name, module in sys.modules.items() if name in blocked}
    for name in blocked:
        sys.modules.pop(name, None)
    _drop_modules("mnemosyne_hermes")
    sys.path.insert(0, str(INTEGRATION_SRC))
    sys.meta_path.insert(0, finder)
    try:
        module = importlib.import_module("mnemosyne_hermes")
    finally:
        sys.meta_path.remove(finder)
        try:
            sys.path.remove(str(INTEGRATION_SRC))
        except ValueError:
            pass
        for name in blocked:
            sys.modules.pop(name, None)
        sys.modules.update(saved)

    try:
        assert module.read_hermes_config_key(None, "tools") is None
        with pytest.raises(module.BatchValidationError):
            module.validate_batch_operations([])
        provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
        assert provider._with_persona_block("base") == "base"
    finally:
        _drop_modules("mnemosyne_hermes")
        _import_module("mnemosyne_hermes", INTEGRATION_SRC)


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


def test_provider_batch_dispatch_matches(tmp_path, provider_modules):
    saved_mnemosyne_modules = _save_mnemosyne_modules()
    _drop_modules("mnemosyne")
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        from mnemosyne.core.beam import BeamMemory

        observed = {}
        for name, module in provider_modules.items():
            db_path = tmp_path / f"{name}-batch.db"
            beam = BeamMemory(session_id=f"batch-{name}", db_path=str(db_path))
            provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
            provider._beam = beam
            provider._hermes_home = str(tmp_path)
            provider._default_scope = "session"
            provider._audit_event = lambda *args, **kwargs: None

            result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
                "operations": [
                    {"action": "remember", "content": f"batch parity {name}"},
                ],
            }))
            observed[name] = {
                "status": result.get("status"),
                "operations_count": result.get("operations_count"),
                "result_statuses": [row.get("status") for row in result.get("results", [])],
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
        "operations_count": 1,
        "result_statuses": ["stored"],
    }


def test_sync_status_uses_per_remote_ack_table(sync_modules):
    import sqlite3

    class Engine:
        device_id = "device-a"
        surface_id = "shared-surface-v1"

        def __init__(self):
            self.conn = sqlite3.connect(":memory:")
            self.conn.execute(
                """CREATE TABLE memory_events (
                    event_id TEXT PRIMARY KEY, device_id TEXT, surface_id TEXT
                )"""
            )
            self.conn.execute(
                """CREATE TABLE sync_outbox_ack (
                    remote_url TEXT, event_id TEXT, acked_at TEXT,
                    PRIMARY KEY (remote_url, event_id)
                )"""
            )
            self.conn.execute(
                "INSERT INTO memory_events VALUES ('event-a', 'device-a', 'shared-surface-v1')"
            )
            self.conn.execute(
                "INSERT INTO sync_outbox_ack VALUES ('https://relay.example', 'event-a', 'now')"
            )

        @staticmethod
        def _meta_get(_key):
            return None

    observed = {}
    for name, module in sync_modules.items():
        adapter = module.SyncAdapter.__new__(module.SyncAdapter)
        adapter._engine = Engine()
        adapter.remote = "https://relay.example"
        adapter.encrypt_enabled = True
        adapter.mode = "bidirectional"
        observed[name] = json.loads(adapter._handle_status())["pending_push"]

    assert observed == {"hermes_memory_provider": 0, "mnemosyne_hermes": 0}
