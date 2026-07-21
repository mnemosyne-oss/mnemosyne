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

import mnemosyne.mcp_tools as mcp_tools

# Test tool schemas
from mnemosyne.mcp_tools import (
    TOOLS, get_tool_definitions, handle_tool_call, _create_instance,
)


class TestToolSchemas:
    """Verify tool schemas match MCP spec and are valid JSON."""

    def test_all_tools_present(self):
        """Core MCP tools must be defined without duplicate names."""
        names = [t["name"] for t in TOOLS]
        assert len(names) == len(set(names))
        assert len(names) >= 25
        assert "mnemosyne_remember_canonical" in names
        assert "mnemosyne_recall_canonical" in names
        assert "mnemosyne_remember" in names
        assert "mnemosyne_batch" in names
        assert "mnemosyne_recall" in names
        assert "mnemosyne_sleep" in names
        assert "mnemosyne_scratchpad_read" in names
        assert "mnemosyne_scratchpad_write" in names
        assert "mnemosyne_stats" in names  # renamed from get_stats
        # New shared tools
        assert "mnemosyne_shared_remember" in names
        assert "mnemosyne_shared_recall" in names
        assert "mnemosyne_shared_forget" in names
        assert "mnemosyne_shared_stats" in names
        # New validation/memory tools
        assert "mnemosyne_invalidate" in names
        assert "mnemosyne_validate" in names
        assert "mnemosyne_get" in names
        assert "mnemosyne_forget" in names
        assert "mnemosyne_export" in names
        assert "mnemosyne_import" in names
        # New graph/triple tools
        assert "mnemosyne_triple_add" in names
        assert "mnemosyne_triple_query" in names
        assert "mnemosyne_graph_query" in names
        assert "mnemosyne_graph_link" in names
        assert "mnemosyne_scratchpad_clear" in names
        assert "mnemosyne_update" in names
        assert "mnemosyne_diagnose" in names

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
        remember_tool = next(t for t in TOOLS if t["name"] == "mnemosyne_remember")
        schema = remember_tool["inputSchema"]
        assert "required" in schema
        assert "content" in schema["required"]
        assert "properties" in schema
        assert "source" in schema["properties"]
        assert "importance" in schema["properties"]
        assert "metadata" in schema["properties"]
        # bank is not in the schema - handled via MCP server env var MNEMOSYNE_MCP_BANK
        assert "extract_entities" in schema["properties"]
        assert "extract" in schema["properties"]
        assert "veracity" in schema["properties"]

    def test_recall_schema_has_required_fields(self):
        """mnemosyne_recall requires 'query'."""
        recall_tool = next(t for t in TOOLS if t["name"] == "mnemosyne_recall")
        schema = recall_tool["inputSchema"]
        assert "required" in schema
        assert "query" in schema["required"]
        assert "limit" in schema["properties"]
        # bank is not in the schema - handled via MCP server env var MNEMOSYNE_MCP_BANK
        assert "temporal_weight" in schema["properties"]
        assert schema["properties"]["explain"]["type"] == "boolean"

    def test_destructive_tools_exist(self):
        """Destructive tools are now exposed (Phase 7+)."""
        names = [t["name"] for t in TOOLS]
        # These tools now exist in the 23-tool set
        assert "mnemosyne_forget" in names
        assert "mnemosyne_invalidate" in names
        assert "mnemosyne_export" in names
        assert "mnemosyne_import" in names

    def test_batch_schema_has_operations(self):
        batch_tool = next(t for t in TOOLS if t["name"] == "mnemosyne_batch")
        schema = batch_tool["inputSchema"]
        assert "operations" in schema["required"]
        assert schema["properties"]["operations"]["type"] == "array"
        assert schema["properties"]["operations"]["maxItems"] == 50
        for context_field in ("bank", "author_id", "author_type", "channel_id"):
            assert context_field in schema["properties"]


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

    def test_handle_remember_forwards_veracity(self, tmp_path, monkeypatch):
        """Regression: MCP remember must forward veracity to Mnemosyne.remember.

        #386 wired veracity into _handle_remember() but Mnemosyne.remember()
        never got the parameter, so every real MCP remember raised
        `TypeError: remember() got an unexpected keyword argument 'veracity'`.
        The mocked handler tests above miss it because a MagicMock swallows any
        kwarg -- this test drives a real instance so the signature is exercised.
        """
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        result = handle_tool_call("mnemosyne_remember", {
            "content": "veracity plumbing regression",
            "source": "test",
            "scope": "global",
            "veracity": "tool",
        })
        assert result["status"] == "stored"

        # veracity must persist to the beam working_memory row, not be dropped.
        mem = _create_instance(bank="default")
        row = mem.beam.conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?",
            (result["memory_id"],),
        ).fetchone()
        assert row is not None
        assert row[0] == "tool"

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

    def test_handle_batch_multiple_remember(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        result = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "remember", "content": "mcp batch one"},
                {"action": "remember", "content": "mcp batch two"},
            ],
        })

        assert result["status"] == "ok"
        assert [item["status"] for item in result["results"]] == ["stored", "stored"]
        assert [event["event"] for event in result["audit_events"]] == ["remember", "remember"]
        mem = _create_instance(bank="default")
        legacy_count = mem.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE content IN (?, ?)",
            ("mcp batch one", "mcp batch two"),
        ).fetchone()[0]
        assert legacy_count == 2

    def test_handle_batch_updates_beam_only_memory(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        mem = _create_instance(bank="default")
        memory_id = mem.beam.remember("beam only before", importance=0.2)

        result = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "update", "memory_id": memory_id, "content": "beam only after", "importance": "0.9"},
            ],
        })

        assert result["status"] == "ok"
        assert result["results"][0]["status"] == "updated"
        updated = _create_instance(bank="default").beam.get(memory_id)
        assert updated["content"] == "beam only after"
        assert updated["importance"] == 0.9


    def test_handle_batch_wrapper_update_forget_invalidate_and_scope(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        stored = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "remember", "content": "mcp update target"},
                {"action": "remember", "content": "mcp forget target"},
                {"action": "remember", "content": "mcp invalidate target"},
                {"action": "remember", "content": "mcp extract target", "extract": True},
            ],
        })
        ids = [row["memory_id"] for row in stored["results"]]

        result = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "update", "memory_id": ids[0], "content": "mcp updated", "importance": "0.7"},
                {"action": "forget", "memory_id": ids[1]},
                {"action": "invalidate", "memory_id": ids[2]},
            ],
        })

        assert result["status"] == "ok"
        assert [item["status"] for item in result["results"]] == ["updated", "deleted", "invalidated"]
        assert [event["event"] for event in result["audit_events"]] == ["update", "forget", "invalidate"]
        mem = _create_instance(bank="default")
        updated = mem.beam.get(ids[0])
        assert updated["content"] == "mcp updated"
        assert updated["importance"] == 0.7
        assert mem.beam.get(ids[1]) is None
        invalidated = mem.beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = ?",
            (ids[2],),
        ).fetchone()
        assert invalidated[0]
        extract_scope = mem.beam.conn.execute(
            "SELECT scope FROM working_memory WHERE id = ?",
            (ids[3],),
        ).fetchone()
        assert extract_scope[0] == "session"


    def test_handle_batch_failure_rolls_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("HOME", str(tmp_path))

        result = handle_tool_call("mnemosyne_batch", {
            "operations": [
                {"action": "remember", "content": "mcp rollback"},
                {"action": "update", "memory_id": "missing", "content": "x"},
            ],
        })

        assert result["status"] == "error"
        assert result["failed_index"] == 1
        mem = _create_instance(bank="default")
        row = mem.beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE content = ?",
            ("mcp rollback",),
        ).fetchone()
        assert row[0] == 0

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
        assert mock_mnemosyne.recall.call_args.kwargs["explain"] is False

    def test_handle_recall_explain_unwraps_payload(self, mock_mnemosyne):
        mock_mnemosyne.recall.return_value = {
            "query": "test query",
            "top_k": 5,
            "results": [{"id": "mem1", "content": "Test content", "score": 0.95}],
            "explain": {"stages": [], "candidates": []},
        }
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_recall", {
                "query": "test query",
                "top_k": 5,
                "explain": True,
                "bank": "default",
            })

        assert result["status"] == "ok"
        assert result["query"] == "test query"
        assert result["top_k"] == 5
        assert result["count"] == 1
        assert "explain" in result
        assert mock_mnemosyne.recall.call_args.kwargs["explain"] is True

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
        assert result["status"] == "consolidated"
        assert "result" in result
        assert "working" in result
        assert "episodic" in result
        assert result["bank"] == "default"
        mock_mnemosyne.sleep.assert_called_once_with(dry_run=False, force=False)

    def test_handle_scratchpad_read(self, mock_mnemosyne):
        """handle_scratchpad_read returns entries."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_scratchpad_read", {
                "bank": "default"
            })
        assert result["entries_count"] == 2
        assert len(result["entries"]) == 2

    def test_handle_scratchpad_write(self, mock_mnemosyne):
        """handle_scratchpad_write returns entry_id."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_scratchpad_write", {
                "content": "New scratchpad entry",
                "bank": "default"
            })
        assert result["status"] == "written"
        assert result["id"] == "scratch-id-456"

    def test_handle_get_stats(self, mock_mnemosyne):
        """handle_get_stats returns JSON-serializable stats."""
        mock_mnemosyne.get_stats.return_value = {
            "total_memories": 42,
            "total_sessions": 3,
            "sources": {"conversation": 30, "file": 12},
            "last_memory": "2026-04-29T01:00:00",
            "database": "/test/db",
            "mode": "beam",
            "beam": {"working_memory": {}, "episodic_memory": {}}
        }
        mock_mnemosyne._session_id = "test-session-123"
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            result = handle_tool_call("mnemosyne_stats", {
                "bank": "default"
            })
        assert "provider" in result
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

    def test_hygiene_clean_rejects_invalid_candidates_json(self):
        """Invalid hygiene payloads return the documented MCP error instead of raising."""
        result = handle_tool_call(
            "mnemosyne_hygiene_clean", {"candidates_json": "not-json"},
        )

        assert result == {"error": "candidates_json is not valid JSON"}

    @pytest.mark.parametrize(
        "candidates_json",
        [
            "null",
            "{}",
            "[{}]",
            json.dumps([{"memory_id": "memory-1", "table_name": "unknown"}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "noise_score": float("nan")}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "noise_score": 1.1}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "noise_score": True}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "noise_score": 10**1000}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "importance": float("inf")}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "importance": 1.1}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "importance": True}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "importance": 10**1000}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "content_length": -1}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "content_length": 1.5}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "content_length": True}]),
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "suggested_action": "destroy"}]),
        ],
    )
    def test_hygiene_clean_rejects_malformed_candidates(self, candidates_json, monkeypatch):
        """Malformed candidate payloads return MCP errors before opening a memory instance."""
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("malformed payload must not initialize memory"),
        )
        result = handle_tool_call(
            "mnemosyne_hygiene_clean", {"candidates_json": candidates_json},
        )

        assert result == {"error": "candidates_json must be a list of valid hygiene candidates"}

    def test_hygiene_clean_parses_valid_candidates_json(self, monkeypatch):
        """Valid JSON reaches the hygiene cleaner with mapped candidate fields."""
        class _Memory:
            class beam:
                db_path = "test.db"

        class _Result:
            def to_dict(self):
                return {"cleaned": 1}

        captured = {}

        def _clean_noise(**kwargs):
            captured.update(kwargs)
            return _Result()

        from mnemosyne.core import hygiene

        monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: _Memory())
        monkeypatch.setattr(hygiene, "clean_noise", _clean_noise)

        candidates_json = json.dumps([{
            "memory_id": "memory-1",
            "table_name": "working_memory",
            "content_preview": "done",
            "noise_score": 0.9,
            "noise_reasons": ["short acknowledgement"],
            "secret_flags": [],
            "importance": 0.2,
            "source": "test",
            "timestamp": "2026-07-21T00:00:00Z",
            "suggested_action": "archive",
            "content_length": 4,
        }])
        result = handle_tool_call(
            "mnemosyne_hygiene_clean",
            {"candidates_json": candidates_json, "action": "archive", "confirm": True},
        )

        assert result == {"status": "applied", "result": {"cleaned": 1}, "bank": "default"}
        assert captured["db_path"] == "test.db"
        assert captured["action"] == "archive"
        assert captured["confirm"] is True
        assert captured["dry_run"] is False
        candidate = captured["candidates"][0]
        assert candidate.memory_id == "memory-1"
        assert candidate.table_name == "working_memory"
        assert candidate.content_preview == "done"
        assert candidate.noise_score == 0.9
        assert candidate.noise_reasons == ["short acknowledgement"]
        assert candidate.secret_flags == []
        assert candidate.importance == 0.2
        assert candidate.source == "test"
        assert candidate.timestamp == "2026-07-21T00:00:00Z"
        assert candidate.suggested_action == "archive"
        assert candidate.content_length == 4

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
        """get_tool_definitions returns all registered tools."""
        tools = get_tool_definitions()
        names = [t["name"] for t in tools]
        assert len(tools) == len(TOOLS)
        assert len(names) == len(set(names))
        assert "mnemosyne_remember" in names

    def test_tool_definitions_convertible_to_tool_pydantic(self):
        """Tool dict definitions must be compatible with mcp SDK 1.x Tool Pydantic model.

        The SDK 1.x list_tools handler expects Tool() instances with typed fields.
        If get_tool_definitions() returns dicts with unexpected keys or missing
        required fields, Tool(**t) will raise a ValidationError.
        """
        try:
            from mcp.types import Tool
        except ImportError:
            pytest.skip("mcp SDK not installed")

        tools = get_tool_definitions()
        for t in tools:
            tool = Tool(**t)
            assert isinstance(tool, Tool)
            assert tool.name == t["name"]
            assert tool.description == t.get("description")
            assert tool.inputSchema == t["inputSchema"]

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
