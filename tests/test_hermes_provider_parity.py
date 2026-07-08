"""Parity checks for the two Hermes Mnemosyne provider implementations."""

from __future__ import annotations

import importlib
import json
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
