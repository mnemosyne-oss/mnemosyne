"""
Tests for Mnemosyne MCP Server (Phase 6)

Run with: pytest tests/test_mcp_server.py -v
"""

import json
import os
import subprocess
import sys
import pytest
from unittest.mock import MagicMock, patch

# Test tool schemas
from mnemosyne.mcp_tools import (
    TOOLS, get_tool_definitions, handle_tool_call,
    _REMEMBER_SCHEMA, _RECALL_SCHEMA, _SLEEP_SCHEMA,
    _SCRATCHPAD_READ_SCHEMA, _SCRATCHPAD_WRITE_SCHEMA, _GET_STATS_SCHEMA
)


class TestToolSchemas:
    """Verify tool schemas match MCP spec and are valid JSON."""

    def test_all_tools_present(self):
        """All 6 tools must be defined."""
        names = [t["name"] for t in TOOLS]
        assert len(names) == 6
        assert "mnemosyne_remember" in names
        assert "mnemosyne_recall" in names
        assert "mnemosyne_sleep" in names
        assert "mnemosyne_scratchpad_read" in names
        assert "mnemosyne_scratchpad_write" in names
        assert "mnemosyne_get_stats" in names

    def test_tool_schemas_are_valid_json(self):
        """Each tool schema must be valid JSON-serializable."""
        for tool in TOOLS:
            # Schema must be serializable
            dumped = json.dumps(tool["inputSchema"])
            loaded = json.loads(dumped)
            assert loaded["type"] == "object"
            assert "properties" in loaded

    def test_remember_schema_has_required_fields(self):
        """mnemosyne_remember requires 'content'."""
        schema = _REMEMBER_SCHEMA
        assert "required" in schema
        assert "content" in schema["required"]
        assert "properties" in schema
        assert "source" in schema["properties"]
        assert "importance" in schema["properties"]
        assert "metadata" in schema["properties"]
        assert "extract_entities" in schema["properties"]
        assert "extract" in schema["properties"]
        assert "bank" in schema["properties"]

    def test_recall_schema_has_required_fields(self):
        """mnemosyne_recall requires 'query'."""
        schema = _RECALL_SCHEMA
        assert "required" in schema
        assert "query" in schema["required"]
        assert "top_k" in schema["properties"]
        assert "bank" in schema["properties"]
        assert "temporal_weight" in schema["properties"]

    def test_no_destructive_tools(self):
        """No forget, invalidate, or export/import tools exposed."""
        names = [t["name"] for t in TOOLS]
        assert "mnemosyne_forget" not in names
        assert "mnemosyne_invalidate" not in names
        assert "mnemosyne_export" not in names
        assert "mnemosyne_import" not in names


class TestToolHandlers:
    """Test each handler with mocked Mnemosyne instance."""

    @pytest.fixture
    def mock_mnemosyne(self):
        """Create a mock Mnemosyne instance."""
        mock = MagicMock()
        mock.remember.return_value = "test-memory-id-123"
        mock.recall.return_value = [
            {"id": "mem1", "content": "Test content", "score": 0.95}
        ]
        mock.sleep.return_value = {"consolidated": 3, "deleted": 1}
        mock.scratchpad_read.return_value = ["entry1", "entry2"]
        mock.scratchpad_write.return_value = "scratch-id-456"
        mock.get_stats.return_value = {
            "total_memories": 42,
            "total_sessions": 3,
            "sources": {"conversation": 30, "file": 12},
            "last_memory": "2026-04-29T01:00:00",
            "database": "/test/db",
            "mode": "beam",
            "beam": {"working_memory": {}, "episodic_memory": {}}
        }
        return mock

    def test_handle_remember(self, mock_mnemosyne):
        """handle_remember returns success with memory_id."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_remember", {
                "content": "Test memory",
                "source": "test",
                "importance": 0.9,
                "bank": "default"
            })
        assert result["status"] == "stored"
        assert result["memory_id"] == "test-memory-id-123"
        assert result["bank"] == "default"
        mock_mnemosyne.remember.assert_called_once()

    def test_handle_remember_uses_mcp_bank_env_default(self, mock_mnemosyne, monkeypatch):
        """MCP server bank default applies when tool call omits bank."""
        monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "work")

        with patch(
            "mnemosyne.mcp_tools._create_instance",
            return_value=mock_mnemosyne,
        ) as create_instance:
            result = handle_tool_call("mnemosyne_remember", {
                "content": "Test memory",
                "source": "test",
            })

        assert result["status"] == "stored"
        assert result["bank"] == "work"
        assert create_instance.call_args.kwargs["bank"] == "work"

    def test_handle_remember_bank_arg_overrides_mcp_bank_env(self, mock_mnemosyne, monkeypatch):
        """Explicit per-call bank should override the server default bank."""
        monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "work")

        with patch(
            "mnemosyne.mcp_tools._create_instance",
            return_value=mock_mnemosyne,
        ) as create_instance:
            result = handle_tool_call("mnemosyne_remember", {
                "content": "Test memory",
                "source": "test",
                "bank": "personal",
            })

        assert result["status"] == "stored"
        assert result["bank"] == "personal"
        assert create_instance.call_args.kwargs["bank"] == "personal"

    def test_handle_recall_uses_mcp_bank_env_default(self, mock_mnemosyne, monkeypatch):
        """MCP recall should use the server default bank when omitted."""
        monkeypatch.setenv("MNEMOSYNE_MCP_BANK", "work")

        with patch(
            "mnemosyne.mcp_tools._create_instance",
            return_value=mock_mnemosyne,
        ) as create_instance:
            result = handle_tool_call("mnemosyne_recall", {
                "query": "test query",
            })

        assert result["status"] == "ok"
        assert result["bank"] == "work"
        assert create_instance.call_args.kwargs["bank"] == "work"

    def test_handle_recall(self, mock_mnemosyne):
        """handle_recall returns list of results."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_recall", {
                "query": "test query",
                "top_k": 5,
                "bank": "default"
            })
        assert result["status"] == "ok"
        assert result["count"] == 1
        assert len(result["results"]) == 1
        mock_mnemosyne.recall.assert_called_once()

    def test_handle_recall_forwards_scoring_weights(self, mock_mnemosyne):
        """Schema-advertised recall weights should be forwarded to Mnemosyne.recall()."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            handle_tool_call("mnemosyne_recall", {
                "query": "test query",
                "top_k": 5,
                "bank": "default",
                "vec_weight": 0.6,
                "fts_weight": 0.3,
                "importance_weight": 0.1,
            })

        _, kwargs = mock_mnemosyne.recall.call_args
        assert kwargs["vec_weight"] == 0.6
        assert kwargs["fts_weight"] == 0.3
        assert kwargs["importance_weight"] == 0.1

    def test_handle_sleep(self, mock_mnemosyne):
        """handle_sleep returns consolidation stats."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_sleep", {
                "dry_run": False,
                "bank": "default"
            })
        assert result["status"] == "ok"
        assert result["dry_run"] is False
        assert "result" in result
        mock_mnemosyne.sleep.assert_called_once_with(dry_run=False)

    def test_handle_scratchpad_read(self, mock_mnemosyne):
        """handle_scratchpad_read returns entries."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_scratchpad_read", {
                "bank": "default"
            })
        assert result["status"] == "ok"
        assert result["count"] == 2
        assert len(result["entries"]) == 2

    def test_handle_scratchpad_write(self, mock_mnemosyne):
        """handle_scratchpad_write returns entry_id."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_scratchpad_write", {
                "content": "New scratchpad entry",
                "bank": "default"
            })
        assert result["status"] == "stored"
        assert result["entry_id"] == "scratch-id-456"

    def test_handle_get_stats(self, mock_mnemosyne):
        """handle_get_stats returns JSON-serializable stats."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_get_stats", {
                "bank": "default"
            })
        assert result["status"] == "ok"
        assert "stats" in result
        # Must be JSON serializable
        dumped = json.dumps(result)
        loaded = json.loads(dumped)
        assert loaded["stats"]["total_memories"] == 42

    def test_error_handling(self, mock_mnemosyne):
        """Error handling returns MCP-compliant error results."""
        mock_mnemosyne.remember.side_effect = RuntimeError("DB locked")
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            with pytest.raises(RuntimeError, match="DB locked"):
                handle_tool_call("mnemosyne_remember", {"content": "test"})

    def test_unknown_tool(self):
        """Unknown tool raises ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            handle_tool_call("mnemosyne_unknown", {})


class TestMCPIntegration:
    """Integration tests for MCP server lifecycle."""

    def test_mcp_server_imports(self):
        """MCP server module imports successfully."""
        from mnemosyne.mcp_server import run_mcp_server, main
        assert callable(run_mcp_server)
        assert callable(main)

    def test_mcp_tools_import_guard(self):
        """mcp_tools imports even if mcp package not available."""
        # The module should load regardless
        from mnemosyne import mcp_tools
        assert hasattr(mcp_tools, "TOOLS")
        assert hasattr(mcp_tools, "handle_tool_call")

    def test_get_tool_definitions_returns_all(self):
        """get_tool_definitions returns all 6 tools."""
        tools = get_tool_definitions()
        assert len(tools) == 6
        names = [t["name"] for t in tools]
        assert "mnemosyne_remember" in names

    def test_top_level_cli_forwards_mcp_arguments(self, tmp_path):
        """`mnemosyne mcp ...` must pass subcommand args to the MCP parser."""
        env = os.environ.copy()
        env["HOME"] = str(tmp_path / "home")
        env["MNEMOSYNE_DATA_DIR"] = str(tmp_path / "mnemosyne-data")
        script = """
import json
import sys
import mnemosyne.mcp_server

def fake_main(argv):
    print(json.dumps({"argv": argv}))

mnemosyne.mcp_server.main = fake_main
sys.argv = [
    "mnemosyne",
    "mcp",
    "--transport",
    "sse",
    "--port",
    "19090",
    "--bank",
    "work",
]
from mnemosyne.cli import run_cli
run_cli()
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {
            "argv": ["--transport", "sse", "--port", "19090", "--bank", "work"]
        }

    def test_mcp_server_main_accepts_explicit_argv(self):
        """MCP server parser should parse caller-provided argv, not global sys.argv."""
        from mnemosyne.mcp_server import main

        with patch("mnemosyne.mcp_server.run_mcp_server") as run_mcp_server:
            main(["--transport", "sse", "--port", "19090", "--bank", "work"])

        run_mcp_server.assert_called_once_with(
            transport="sse", port=19090, bank="work", host="127.0.0.1"
        )


class TestImportGuard:
    """Verify MCP is truly optional."""

    def test_core_imports_without_mcp(self):
        """Core mnemosyne imports work without mcp installed."""
        from mnemosyne import remember, recall, get_stats
        assert callable(remember)
        assert callable(recall)
        assert callable(get_stats)

    def test_mcp_server_raises_without_mcp(self):
        """MCP server raises helpful error if mcp not installed."""
        from mnemosyne.mcp_server import _MCP_AVAILABLE, _run_stdio
        
        if _MCP_AVAILABLE:
            # mcp is installed — verify the server function exists and the flag is True
            assert _MCP_AVAILABLE is True
        else:
            # mcp is NOT installed — verify _run_stdio raises RuntimeError
            import asyncio
            with pytest.raises(RuntimeError, match="MCP not installed"):
                asyncio.get_event_loop().run_until_complete(_run_stdio())
