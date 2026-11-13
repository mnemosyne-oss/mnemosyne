"""Tests for default_scope configuration.

Covers the default_scope feature that allows users to configure the default
scope for remember() calls from "session" to "global". Two provider paths
are tested:

1. hermes_memory_provider — legacy plugin (imported by existing test suite)
2. mnemosyne_hermes — new pip-installable integration provider
"""

from __future__ import annotations

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Legacy plugin provider tests
# ---------------------------------------------------------------------------

class TestLegacyDefaultScope:
    """Tests for default_scope in hermes_memory_provider (legacy plugin)."""

    def test_default_scope_defaults_to_session(self):
        """Provider defaults _default_scope to 'session'."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        assert provider._default_scope == "session"

    def test_apply_provider_config_sets_global(self):
        """_apply_provider_config accepts default_scope='global' from kwargs."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        provider._apply_provider_config({"default_scope": "global"})
        assert provider._default_scope == "global"

    def test_apply_provider_config_sets_session(self):
        """_apply_provider_config accepts default_scope='session' from kwargs."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        provider._default_scope = "global"
        provider._apply_provider_config({"default_scope": "session"})
        assert provider._default_scope == "session"

    def test_apply_provider_config_rejects_invalid_scope(self, caplog):
        """_apply_provider_config rejects values other than 'session'/'global'."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        with caplog.at_level("WARNING", logger="hermes_memory_provider"):
            provider._apply_provider_config({"default_scope": "invalid"})
        assert provider._default_scope == "session"  # unchanged
        assert any("invalid default_scope" in r.getMessage() for r in caplog.records)

    def test_apply_provider_config_case_insensitive(self):
        """_apply_provider_config normalizes scope to lowercase."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        provider._apply_provider_config({"default_scope": "GLOBAL"})
        assert provider._default_scope == "global"

    def test_get_config_schema_includes_default_scope(self):
        """get_config_schema advertises the default_scope key."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        schema = provider.get_config_schema()
        scope_entries = [e for e in schema if e["key"] == "default_scope"]
        assert len(scope_entries) == 1
        entry = scope_entries[0]
        assert entry["choices"] == ["session", "global"]
        assert entry["default"] == "session"

    def test_handle_remember_uses_default_scope_when_not_passed(self, monkeypatch):
        """_handle_remember uses _default_scope when caller doesn't pass scope."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        provider._default_scope = "global"
        beam = MagicMock()
        beam.remember.return_value = "mem-123"
        provider._beam = beam

        provider._handle_remember({"content": "test fact"})

        kwargs = beam.remember.call_args.kwargs
        assert kwargs.get("scope") == "global", (
            f"_handle_remember should use _default_scope; got scope={kwargs.get('scope')!r}"
        )

    def test_handle_remember_respects_explicit_scope(self, monkeypatch):
        """_handle_remember respects explicit scope arg over _default_scope."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        provider._default_scope = "global"
        beam = MagicMock()
        beam.remember.return_value = "mem-456"
        provider._beam = beam

        provider._handle_remember({"content": "test", "scope": "session"})

        kwargs = beam.remember.call_args.kwargs
        assert kwargs.get("scope") == "session", (
            "Explicit scope arg should override _default_scope"
        )

    def test_sync_turn_passes_default_scope_user(self, monkeypatch):
        """sync_turn passes _default_scope to beam.remember for user content."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        provider._default_scope = "global"
        beam = MagicMock()
        provider._beam = beam
        provider._sync_roles = {"user", "assistant"}

        provider.sync_turn(
            user_content="This is a test message",
            assistant_content="This is a test response",
        )

        # First call is for user content
        user_call = beam.remember.call_args_list[0]
        assert user_call.kwargs.get("scope") == "global", (
            f"sync_turn should pass _default_scope for user content; "
            f"got scope={user_call.kwargs.get('scope')!r}"
        )

    def test_sync_turn_passes_default_scope_assistant(self, monkeypatch):
        """sync_turn passes _default_scope to beam.remember for assistant content."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        provider._default_scope = "global"
        beam = MagicMock()
        provider._beam = beam
        provider._sync_roles = {"user", "assistant"}

        provider.sync_turn(
            user_content="This is a test message",
            assistant_content="This is a test response",
        )

        # Second call is for assistant content
        assistant_call = beam.remember.call_args_list[1]
        assert assistant_call.kwargs.get("scope") == "global", (
            f"sync_turn should pass _default_scope for assistant content; "
            f"got scope={assistant_call.kwargs.get('scope')!r}"
        )

    def test_sync_turn_uses_session_by_default(self, monkeypatch):
        """sync_turn uses 'session' scope when _default_scope is not changed."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        provider = MnemosyneMemoryProvider()
        # Don't change _default_scope — should be "session"
        beam = MagicMock()
        provider._beam = beam
        provider._sync_roles = {"user"}

        provider.sync_turn(
            user_content="This is a test message",
            assistant_content="",
        )

        kwargs = beam.remember.call_args.kwargs
        assert kwargs.get("scope") == "session"


# ---------------------------------------------------------------------------
# MCP server tests
# ---------------------------------------------------------------------------

class TestMCPDefaultScope:
    """Tests for MNEMOSYNE_DEFAULT_SCOPE in mnemosyne.mcp_tools (#653)."""

    def test_resolve_default_scope_defaults_to_global(self, monkeypatch):
        """_resolve_default_scope defaults to 'global' when env var is unset (#653)."""
        monkeypatch.delenv("MNEMOSYNE_DEFAULT_SCOPE", raising=False)

        from mnemosyne.mcp_tools import _resolve_default_scope
        assert _resolve_default_scope() == "global"

    def test_resolve_default_scope_respects_global(self, monkeypatch):
        """_resolve_default_scope returns 'global' when env var is set."""
        monkeypatch.setenv("MNEMOSYNE_DEFAULT_SCOPE", "global")

        # Force re-import
        import importlib
        import mnemosyne.mcp_tools as mcp
        importlib.reload(mcp)
        assert mcp._resolve_default_scope() == "global"

    def test_resolve_default_scope_respects_session(self, monkeypatch):
        """_resolve_default_scope returns 'session' when env var is set to session."""
        monkeypatch.setenv("MNEMOSYNE_DEFAULT_SCOPE", "session")

        import importlib
        import mnemosyne.mcp_tools as mcp
        importlib.reload(mcp)
        assert mcp._resolve_default_scope() == "session"

    def test_resolve_default_scope_rejects_invalid(self, monkeypatch):
        """_resolve_default_scope falls back to 'global' for invalid values (#653)."""
        monkeypatch.setenv("MNEMOSYNE_DEFAULT_SCOPE", "invalid")

        import importlib
        import mnemosyne.mcp_tools as mcp
        importlib.reload(mcp)
        assert mcp._resolve_default_scope() == "global"

    def test_resolve_default_scope_case_insensitive(self, monkeypatch):
        """_resolve_default_scope normalizes to lowercase."""
        monkeypatch.setenv("MNEMOSYNE_DEFAULT_SCOPE", "GLOBAL")

        import importlib
        import mnemosyne.mcp_tools as mcp
        importlib.reload(mcp)
        assert mcp._resolve_default_scope() == "global"

    def test_mcp_remember_omitted_scope_cross_session_visible(self, tmp_path, monkeypatch):
        """[#653] An MCP-authored memory with omitted scope is visible to CLI/SDK readers."""
        monkeypatch.delenv("MNEMOSYNE_DEFAULT_SCOPE", raising=False)
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.mcp_tools import _handle_remember

        # 1. MCP write path defaults to session_id="mcp_default"
        res = _handle_remember({"content": "cross session probe content 653"})
        assert res.get("status") == "stored"
        mid = res.get("memory_id")

        # 2. Fresh CLI/SDK reader with different session_id (e.g. "default")
        reader = Mnemosyne(session_id="default")
        hits = reader.recall("cross session probe content 653", top_k=5)
        hit_ids = [h.get("id") for h in hits]
        assert mid in hit_ids

    def test_mcp_remember_explicit_session_scope_retains_isolation(self, tmp_path, monkeypatch):
        """[#653] An MCP-authored memory with explicit scope='session' retains session isolation."""
        monkeypatch.delenv("MNEMOSYNE_DEFAULT_SCOPE", raising=False)
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.mcp_tools import _handle_remember

        res = _handle_remember({
            "content": "isolated probe content 653",
            "scope": "session",
        })
        assert res.get("status") == "stored"
        mid = res.get("memory_id")

        # Same session (mcp_default) sees it
        mcp_reader = Mnemosyne(session_id="mcp_default")
        hits_mcp = mcp_reader.recall("isolated probe content 653", top_k=5)
        assert mid in [h.get("id") for h in hits_mcp]

        # Different session does NOT see it when cross_session is False (default)
        cli_reader = Mnemosyne(session_id="default")
        hits_cli = cli_reader.recall("isolated probe content 653", top_k=5)
        assert mid not in [h.get("id") for h in hits_cli]

    def test_mcp_remember_schema_omits_static_scope_default(self):
        """[#653] MCP remember schema omits static scope default while Core/Hermes retains it."""
        from mnemosyne.mcp_tools import TOOLS, get_tool_definitions
        from mnemosyne.tool_schemas import REMEMBER_SCHEMA, ALL_TOOL_SCHEMAS

        # 1. MCP TOOLS and advertised definitions have no static default on scope
        mcp_remember = next(t for t in TOOLS if t["name"] == "mnemosyne_remember")
        scope_prop = mcp_remember["input_schema"]["properties"]["scope"]
        assert "default" not in scope_prop
        assert "MNEMOSYNE_DEFAULT_SCOPE" in scope_prop["description"]
        assert "global" in scope_prop["description"]

        advertised_remember = next(t for t in get_tool_definitions() if t["name"] == "mnemosyne_remember")
        assert "default" not in advertised_remember["input_schema"]["properties"]["scope"]

        # 2. Shared Core/Hermes schemas retain unchanged static 'session' default
        assert REMEMBER_SCHEMA["parameters"]["properties"]["scope"]["default"] == "session"
        raw_core_remember = next(s for s in ALL_TOOL_SCHEMAS if s["name"] == "mnemosyne_remember")
        assert raw_core_remember["parameters"]["properties"]["scope"]["default"] == "session"

    def test_mcp_remember_respects_session_env_default(self, tmp_path, monkeypatch):
        """[#653] Setting MNEMOSYNE_DEFAULT_SCOPE=session preserves session isolation on omitted scope."""
        monkeypatch.setenv("MNEMOSYNE_DEFAULT_SCOPE", "session")
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.mcp_tools import _handle_remember

        res = _handle_remember({"content": "env session content 653"})
        assert res.get("status") == "stored"
        mid = res.get("memory_id")

        # Same session (mcp_default) sees it
        mcp_reader = Mnemosyne(session_id="mcp_default")
        hits_mcp = mcp_reader.recall("env session content 653", top_k=5)
        assert mid in [h.get("id") for h in hits_mcp]

        # Distinct session cannot recall it
        cli_reader = Mnemosyne(session_id="default")
        hits_cli = cli_reader.recall("env session content 653", top_k=5)
        assert mid not in [h.get("id") for h in hits_cli]

    def test_mcp_batch_omitted_scope_cross_session_visible(self, tmp_path, monkeypatch):
        """[#653] An item remembered via mnemosyne_batch with omitted scope defaults to global."""
        monkeypatch.delenv("MNEMOSYNE_DEFAULT_SCOPE", raising=False)
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.mcp_tools import _handle_batch

        res = _handle_batch({
            "operations": [
                {"action": "remember", "content": "batch cross session probe 653"},
            ]
        })
        assert res.get("status") == "ok"
        mid = res["results"][0]["memory_id"]

        # Different session reader can retrieve it
        reader = Mnemosyne(session_id="other_session")
        hits = reader.recall("batch cross session probe 653", top_k=5)
        assert mid in [h.get("id") for h in hits]

    def test_mcp_batch_explicit_session_scope_retains_isolation(self, tmp_path, monkeypatch):
        """[#653] An item remembered via mnemosyne_batch with explicit scope='session' retains isolation."""
        monkeypatch.delenv("MNEMOSYNE_DEFAULT_SCOPE", raising=False)
        monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

        from mnemosyne.core.memory import Mnemosyne
        from mnemosyne.mcp_tools import _handle_batch

        res = _handle_batch({
            "operations": [
                {"action": "remember", "content": "batch isolated probe 653", "scope": "session"},
            ]
        })
        assert res.get("status") == "ok"
        mid = res["results"][0]["memory_id"]

        # Same session (mcp_default) sees it
        mcp_reader = Mnemosyne(session_id="mcp_default")
        hits_mcp = mcp_reader.recall("batch isolated probe 653", top_k=5)
        assert mid in [h.get("id") for h in hits_mcp]

        # Different session cannot retrieve it
        cli_reader = Mnemosyne(session_id="other_session")
        hits_cli = cli_reader.recall("batch isolated probe 653", top_k=5)
        assert mid not in [h.get("id") for h in hits_cli]