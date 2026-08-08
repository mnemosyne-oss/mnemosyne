"""
Tests for Mnemosyne MCP Server (Phase 6)

Run with: pytest tests/test_mcp_server.py -v
"""

import json
import os
import sqlite3
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
            # Schema must be serializable. ``mcp`` SDK 2.x renamed the wire
            # field to ``input_schema`` (was ``inputSchema`` in 1.x); the
            # schema dict in ``TOOLS`` uses the new key.
            dumped = json.dumps(tool["input_schema"])
            loaded = json.loads(dumped)
            assert loaded["type"] == "object"
            assert "properties" in loaded

    def test_remember_schema_has_required_fields(self):
        """mnemosyne_remember requires 'content'."""
        remember_tool = next(t for t in TOOLS if t["name"] == "mnemosyne_remember")
        schema = remember_tool["input_schema"]
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
        schema = recall_tool["input_schema"]
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
        schema = batch_tool["input_schema"]
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

    def test_handle_recall_normalizes_blank_query_time(self, mock_mnemosyne):
        """Blank query_time reaches Mnemosyne.recall() as unset, not as "" (#555).

        Harnesses built against the older schema sent the declared default "",
        which used to reach _parse_query_time and raise.
        """
        for blank in ("", "   ", "\t"):
            mock_mnemosyne.recall.reset_mock()
            with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
                handle_tool_call("mnemosyne_recall", {
                    "query": "test query",
                    "bank": "default",
                    "query_time": blank,
                })
            _, kwargs = mock_mnemosyne.recall.call_args
            assert kwargs["query_time"] is None, f"blank {blank!r} not normalized"

    def test_handle_recall_preserves_explicit_query_time(self, mock_mnemosyne):
        """A real ISO timestamp is forwarded untouched."""
        with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
            handle_tool_call("mnemosyne_recall", {
                "query": "test query",
                "bank": "default",
                "query_time": "2026-04-29T12:00:00",
            })
        _, kwargs = mock_mnemosyne.recall.call_args
        assert kwargs["query_time"] == "2026-04-29T12:00:00"

    def test_handle_recall_does_not_swallow_falsey_non_strings(self, mock_mnemosyne):
        """0/False/[] are type errors, not "unset" — they must not become None.

        Guards against normalizing with `or None`, which would silently accept
        them and suppress _parse_query_time's TypeError.
        """
        for bad in (0, False, []):
            mock_mnemosyne.recall.reset_mock()
            with patch("mnemosyne.mcp_tools._create_instance", return_value=mock_mnemosyne):
                handle_tool_call("mnemosyne_recall", {
                    "query": "test query",
                    "bank": "default",
                    "query_time": bad,
                })
            _, kwargs = mock_mnemosyne.recall.call_args
            # Identity, not equality: 0 == False in Python, so `==` would let a
            # bool/int mix-up pass. The exact object must be forwarded.
            assert kwargs["query_time"] is bad, f"{bad!r} not forwarded unchanged"

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

    def test_hygiene_audit_default_uses_and_closes_exact_readonly_connection(
        self, tmp_path, monkeypatch
    ):
        """Default-bank audits use one real query-only connection, never Mnemosyne."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "mnemosyne.db"
        with sqlite3.connect(db_path) as writable:
            writable.execute("CREATE TABLE audit_marker (id INTEGER)")

        class _Report:
            def to_dict(self):
                return {"total_candidates": 0}

        from mnemosyne.core import hygiene

        captured = {}

        def _audit_noise(**kwargs):
            conn = kwargs["conn"]
            captured.update(kwargs)
            assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
            with pytest.raises(
                sqlite3.OperationalError, match="attempt to write a readonly database"
            ):
                conn.execute("CREATE TABLE forbidden_write (id INTEGER)")
            return _Report()

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("hygiene audit must not initialize Mnemosyne"),
        )
        monkeypatch.setattr(hygiene, "audit_noise", _audit_noise)

        arguments = {
            "limit": 17,
            "tables": ["audit_marker"],
            "min_score": 0.8,
            "offset": 3,
            "scan_all": True,
            "batch_size": 11,
        }
        result = handle_tool_call("mnemosyne_hygiene_audit", arguments)

        assert result == {
            "status": "audited",
            "report": {"total_candidates": 0},
            "bank": "default",
        }
        assert captured == {
            "db_path": db_path,
            "limit": 17,
            "tables": ["audit_marker"],
            "min_score": 0.8,
            "offset": 3,
            "scan_all": True,
            "batch_size": 11,
            "conn": captured["conn"],
        }
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            captured["conn"].execute("SELECT 1")

    def test_hygiene_audit_closes_connection_when_audit_raises(self, tmp_path, monkeypatch):
        """The read-only connection closes even if audit_noise raises."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        db_path = data_dir / "mnemosyne.db"
        sqlite3.connect(db_path).close()

        from mnemosyne.core import hygiene
        from mnemosyne import doctor

        captured = {}
        real_open = doctor.open_readonly_doctor_db

        def _open_readonly(path):
            captured["conn"] = real_open(path)
            return captured["conn"]

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("hygiene audit must not initialize Mnemosyne"),
        )
        monkeypatch.setattr(doctor, "open_readonly_doctor_db", _open_readonly)
        monkeypatch.setattr(
            hygiene,
            "audit_noise",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("audit failed")),
        )

        with pytest.raises(RuntimeError, match="audit failed"):
            handle_tool_call("mnemosyne_hygiene_audit", {})

        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            captured["conn"].execute("SELECT 1")

    def test_hygiene_audit_readonly_open_failure_has_no_writable_fallback(
        self, tmp_path, monkeypatch
    ):
        """A missing default DB fails at the read-only open without running audit_noise."""
        data_dir = tmp_path / "missing-data"
        from mnemosyne.core import hygiene

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("hygiene audit must not initialize Mnemosyne"),
        )
        monkeypatch.setattr(
            hygiene,
            "audit_noise",
            lambda **_kwargs: pytest.fail("audit must not run after readonly open failure"),
        )

        with pytest.raises(sqlite3.OperationalError):
            handle_tool_call("mnemosyne_hygiene_audit", {})

        assert not data_dir.exists()

    def test_hygiene_audit_unknown_bank_never_initializes_or_creates_state(
        self, tmp_path, monkeypatch
    ):
        """Unknown named banks fail before any manager/instance can create state."""
        data_dir = tmp_path / "fresh-data"

        def _snapshot(root):
            return (
                sorted(path.relative_to(root) for path in root.rglob("*"))
                if root.exists()
                else []
            )

        before = _snapshot(data_dir)
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("unknown bank must not initialize Mnemosyne"),
        )

        with pytest.raises(ValueError, match="Bank 'unknown' does not exist"):
            handle_tool_call("mnemosyne_hygiene_audit", {"bank": "unknown"})

        assert _snapshot(data_dir) == before
        assert not (data_dir / "banks").exists()
        assert not (data_dir / "config").exists()

    def test_hygiene_audit_named_bank_requires_an_existing_database(
        self, tmp_path, monkeypatch
    ):
        """A named bank directory without its DB is rejected without mutations."""
        data_dir = tmp_path / "data"
        bank_dir = data_dir / "banks" / "team"
        bank_dir.mkdir(parents=True)
        before = sorted(path.relative_to(data_dir) for path in data_dir.rglob("*"))

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir))
        monkeypatch.setattr(
            mcp_tools,
            "_create_instance",
            lambda **_kwargs: pytest.fail("named-bank audit must not initialize Mnemosyne"),
        )

        with pytest.raises(FileNotFoundError, match="Database for bank 'team' does not exist"):
            handle_tool_call("mnemosyne_hygiene_audit", {"bank": "team"})

        assert sorted(path.relative_to(data_dir) for path in data_dir.rglob("*")) == before

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
            json.dumps([{"memory_id": "memory-1", "table_name": "working_memory", "importance": True}]),
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

    def test_mcp_tools_fresh_import_does_not_materialize_default_state(self, tmp_path):
        """A fresh MCP import must not initialize the legacy default database."""
        data_dir = tmp_path / "data"
        mnemosyne_home = tmp_path / "mnemosyne-home"
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(mnemosyne_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
        })
        script = """
import json
import os
from pathlib import Path

import mnemosyne.mcp_tools

root = Path(os.environ["MNEMOSYNE_DATA_DIR"])
print(json.dumps(sorted(str(path.relative_to(root)) for path in root.rglob("*")) if root.exists() else []))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []
        assert not data_dir.exists()
        assert not (data_dir / "mnemosyne.db").exists()
        assert not (data_dir / "mnemosyne.db-wal").exists()
        assert not (data_dir / "mnemosyne.db-shm").exists()
        assert not (data_dir / "config").exists()
        assert not mnemosyne_home.exists()

    def test_mcp_tools_runtime_type_hints_are_resolvable_without_default_state(self, tmp_path):
        """Lazy imports must not break runtime type-hint consumers or create state."""
        data_dir = tmp_path / "data"
        mnemosyne_home = tmp_path / "mnemosyne-home"
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(mnemosyne_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
        })
        script = """
import json
import os
from pathlib import Path
import sys
import typing

import mnemosyne.mcp_tools as m

assert typing.get_type_hints(m._create_instance)["return"] is typing.Any
assert typing.get_type_hints(m._WrapperBatchAdapter.__init__)["mem"] is typing.Any
assert "mnemosyne.core.memory" not in sys.modules
root = Path(os.environ["MNEMOSYNE_DATA_DIR"])
print(json.dumps(sorted(str(path.relative_to(root)) for path in root.rglob("*")) if root.exists() else []))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == []
        assert not data_dir.exists()
        assert not mnemosyne_home.exists()

    def test_mcp_tools_public_mnemosyne_export_is_lazy_and_canonical(self, tmp_path):
        """The legacy public export resolves to the canonical class on demand."""
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(tmp_path / "mnemosyne-home"),
            "MNEMOSYNE_DATA_DIR": str(tmp_path / "data"),
        })
        script = """
import json

from mnemosyne.mcp_tools import Mnemosyne
from mnemosyne.core.memory import Mnemosyne as CoreMnemosyne

print(json.dumps({"is_canonical": Mnemosyne is CoreMnemosyne}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == {"is_canonical": True}

    def test_mcp_tools_star_import_preserves_legacy_export_surface(self, tmp_path):
        """Star imports retain every old public name and resolve Mnemosyne canonically."""
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(tmp_path / "mnemosyne-home"),
            "MNEMOSYNE_DATA_DIR": str(tmp_path / "data"),
        })
        expected = [
            "ALL_TOOL_SCHEMAS",
            "Any",
            "BatchValidationError",
            "BeamMemory",
            "CallToolResult",
            "Dict",
            "ErrorData",
            "List",
            "Mnemosyne",
            "Path",
            "TOOLS",
            "TextContent",
            "Tool",
            "apply_beam_batch",
            "batch_validation_error_payload",
            "dry_run_batch",
            "get_tool_definitions",
            "handle_tool_call",
            "json",
            "math",
            "os",
            "sqlite3",
            "validate_batch_operations",
        ]
        script = f"""
import json

ns = {{}}
exec("from mnemosyne.mcp_tools import *", ns)
from mnemosyne.core.memory import Mnemosyne as CoreMnemosyne

names = sorted(name for name in ns if name != "__builtins__")
assert names == {expected!r}
assert ns["Mnemosyne"] is CoreMnemosyne
assert not {{"TYPE_CHECKING", "TypeAlias", "MnemosyneInstance"}} & ns.keys()
print(json.dumps({{"names": names, "count": len(names)}}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        assert json.loads(completed.stdout) == {"names": expected, "count": len(expected)}

    def test_named_bank_hygiene_audit_in_fresh_process_creates_no_default_state(self, tmp_path):
        """A real named-bank audit never initializes default state in a fresh process."""
        data_dir = tmp_path / "data"
        named_db = data_dir / "banks" / "team" / "mnemosyne.db"
        named_db.parent.mkdir(parents=True)
        sqlite3.connect(named_db).close()
        before = sorted(str(path.relative_to(data_dir)) for path in data_dir.rglob("*"))
        mnemosyne_home = tmp_path / "mnemosyne-home"
        env = os.environ.copy()
        env.update({
            "HOME": str(tmp_path / "home"),
            "MNEMOSYNE_HOME": str(mnemosyne_home),
            "MNEMOSYNE_DATA_DIR": str(data_dir),
        })
        script = """
import json
import os
from pathlib import Path

from mnemosyne.mcp_tools import handle_tool_call

root = Path(os.environ["MNEMOSYNE_DATA_DIR"])
result = handle_tool_call("mnemosyne_hygiene_audit", {"bank": "team"})
after = sorted(str(path.relative_to(root)) for path in root.rglob("*"))
print(json.dumps({"result": result, "after": after}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script],
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )

        assert completed.returncode == 0, completed.stderr
        payload = json.loads(completed.stdout)
        assert payload["result"]["status"] == "audited"
        assert payload["result"]["bank"] == "team"
        assert payload["after"] == before
        assert not (data_dir / "mnemosyne.db").exists()
        assert not (data_dir / "mnemosyne.db-wal").exists()
        assert not (data_dir / "mnemosyne.db-shm").exists()
        assert not (data_dir / "config").exists()
        assert not mnemosyne_home.exists()

    def test_get_tool_definitions_returns_all(self):
        """get_tool_definitions returns all registered tools."""
        tools = get_tool_definitions()
        names = [t["name"] for t in tools]
        assert len(tools) == len(TOOLS)
        assert len(names) == len(set(names))
        assert "mnemosyne_remember" in names

    def test_tool_definitions_convertible_to_tool_pydantic(self):
        """Tool dict definitions must be compatible with the ``mcp`` SDK 2.x Tool Pydantic model.

        The SDK 2.x ``Tool`` model exposes ``input_schema`` (snake_case) as
        its canonical Python field; ``inputSchema`` is still accepted as a
        Pydantic alias on construction. This test asserts both paths:
        (a) ``Tool(**t)`` constructs without raising; (b) the constructed
        ``Tool`` carries the same schema as the source dict regardless of
        which key the dict used.
        """
        from mcp.types import Tool

        tools = get_tool_definitions()
        for t in tools:
            tool = Tool(**t)
            assert isinstance(tool, Tool)
            assert tool.name == t["name"]
            assert tool.description == t.get("description")
            # Pydantic normalizes both ``inputSchema`` (1.x alias) and
            # ``input_schema`` (2.x canonical) to ``tool.input_schema``.
            expected = t.get("input_schema", t.get("inputSchema"))
            assert tool.input_schema == expected

        # Keep the legacy-only wire shape covered even though the current
        # normalizer emits the SDK 2.x canonical key.
        legacy = dict(tools[0])
        legacy["inputSchema"] = legacy.pop("input_schema")
        legacy_tool = Tool(**legacy)
        assert legacy_tool.input_schema == legacy["inputSchema"]

    def test_mcp_client_request_surface_returns_typed_results(self):
        """Real SDK client requests exercise list/call registration and dispatch."""
        from mcp import ClientSession
        from mcp.shared.memory import create_client_server_memory_streams
        from mcp.types import CallToolResult, ListToolsResult
        from mnemosyne.mcp_server import _build_mcp_server

        import asyncio
        import contextlib
        from unittest.mock import patch

        async def exercise():
            server = _build_mcp_server()
            async with create_client_server_memory_streams() as (client_streams, server_streams):
                server_task = asyncio.create_task(
                    server.run(*server_streams, server.create_initialization_options())
                )
                try:
                    async with ClientSession(*client_streams) as client:
                        await client.initialize()
                        listed = await client.list_tools()
                        assert isinstance(listed, ListToolsResult)
                        assert len(listed.tools) >= 25
                        with patch(
                            "mnemosyne.mcp_server.handle_tool_call",
                            return_value={"status": "ok"},
                        ) as handle_call:
                            success = await client.call_tool("mnemosyne_stats", None)
                        assert isinstance(success, CallToolResult)
                        assert success.is_error is False
                        handle_call.assert_called_once_with("mnemosyne_stats", {})
                finally:
                    server_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await server_task

        asyncio.run(exercise())

    def test_transport_builders_share_mcp_server(self):
        """stdio and SSE transport setup both obtain the shared server builder."""
        from mnemosyne import mcp_server

        import asyncio
        from unittest.mock import AsyncMock, patch

        async def exercise_stdio():
            fake_server = type(
                "FakeServer",
                (),
                {
                    "run": AsyncMock(),
                    "create_initialization_options": lambda self: object(),
                },
            )()

            class _StdioContext:
                async def __aenter__(self):
                    return (object(), object())

                async def __aexit__(self, *args):
                    return False

            with (
                patch.object(mcp_server, "_build_mcp_server", return_value=fake_server) as build,
                patch.object(mcp_server, "stdio_server", return_value=_StdioContext()),
            ):
                await mcp_server._run_stdio()
            build.assert_called_once_with()
            fake_server.run.assert_awaited_once()

        asyncio.run(exercise_stdio())

        with patch.object(mcp_server, "_build_mcp_server", return_value=object()) as build:
            app = mcp_server._build_sse_app(host="127.0.0.1")
        build.assert_called_once_with()
        sse_route = next(route for route in app.routes if getattr(route, "path", None) == "/sse")
        assert any(cell.cell_contents is build.return_value for cell in (sse_route.endpoint.__closure__ or ()))

    def test_sse_messages_transport_uses_matching_mount_path(self):
        """The advertised POST URI and mounted raw ASGI app must agree."""
        from mnemosyne import mcp_server
        from starlette.routing import Mount
        from unittest.mock import AsyncMock, patch

        handle_post_message = AsyncMock()
        transport = type("FakeTransport", (), {"handle_post_message": handle_post_message})()
        transport_factory = patch("mcp.server.sse.SseServerTransport")
        with (
            patch.object(mcp_server, "_build_mcp_server", return_value=object()),
            transport_factory as build_transport,
        ):
            build_transport.return_value = transport
            app = mcp_server._build_sse_app(host="127.0.0.1")

        build_transport.assert_called_once_with("/messages/")
        message_mount = next(
            route
            for route in app.routes
            if isinstance(route, Mount) and route.path == "/messages"
        )
        assert message_mount.app is handle_post_message

    def test_sse_messages_invalid_session_returns_clean_error(self):
        """An invalid session POST must return an error without a second ASGI response."""
        from mnemosyne import mcp_server
        from starlette.testclient import TestClient
        from unittest.mock import patch

        with patch.object(mcp_server, "_build_mcp_server", return_value=object()):
            app = mcp_server._build_sse_app(host="127.0.0.1")

        response = TestClient(app).post(
            "/messages/?session_id=missing",
            json={"jsonrpc": "2.0", "method": "ping"},
        )

        assert response.status_code == 400
        assert response.text == "Invalid session ID"

    def test_sse_messages_missing_session_returns_clean_error(self):
        """A valid-but-missing session must return 404 without an ASGI exception."""
        from mnemosyne import mcp_server
        from starlette.testclient import TestClient
        from unittest.mock import patch

        with patch.object(mcp_server, "_build_mcp_server", return_value=object()):
            app = mcp_server._build_sse_app(host="127.0.0.1")

        response = TestClient(app).post(
            "/messages/?session_id=00000000-0000-0000-0000-000000000000",
            json={"jsonrpc": "2.0", "method": "ping"},
        )

        assert response.status_code == 404
        assert response.text == "Could not find session"

    def test_build_mcp_server_list_tools_returns_listtoolsresult(self):
        """SDK 2.x contract: the tools/list callback must return a ListToolsResult.

        Regression test for the CodeRabbit finding on PR #571 (round 2):
        ``_on_list_tools`` must wrap the Tool list in ``ListToolsResult(tools=...)``
        rather than returning a bare list. The SDK 2.x low-level server expects
        a ``ListToolsResult`` object — a bare list breaks the ``tools/list``
        response contract for both stdio and SSE transports.
        """
        from mcp.types import ListToolsResult, Tool
        from mnemosyne.mcp_server import _build_mcp_server

        server = _build_mcp_server()
        entry = server.get_request_handler("tools/list")
        assert entry is not None
        assert callable(entry.handler)
        on_list_tools = entry.handler

        import asyncio
        result = asyncio.run(on_list_tools(ctx=None, params=None))
        assert isinstance(result, ListToolsResult), (
            f"tools/list callback must return ListToolsResult, got {type(result).__name__}"
        )
        assert isinstance(result.tools, list)
        assert all(isinstance(t, Tool) for t in result.tools)
        assert len(result.tools) >= 25  # matches test_all_tools_present

    def test_build_mcp_server_call_tool_returns_is_error_on_failure(self):
        """SDK 2.x contract: tools/call must return CallToolResult with is_error=True on failure.

        Regression test for the CodeRabbit finding on PR #571 (round 2):
        the ``_on_call_tool`` exception path must set ``is_error=True`` so MCP
        clients can distinguish implementation failures from successful calls.
        Preserves the existing error payload shape for backward compatibility.
        """
        from mcp.types import CallToolResult
        from mnemosyne.mcp_server import _build_mcp_server

        server = _build_mcp_server()
        entry = server.get_request_handler("tools/call")
        assert entry is not None
        assert callable(entry.handler)
        on_call_tool = entry.handler

        # Construct a params-shaped object with empty name → handle_tool_call
        # raises before returning a valid result. This is the path that was
        # previously returning a successful-looking CallToolResult with error
        # content instead of a flagged failure.
        class _Params:
            name = ""
            arguments = {}

        import asyncio
        result = asyncio.run(on_call_tool(ctx=None, params=_Params()))
        assert isinstance(result, CallToolResult)
        assert result.is_error is True, (
            "tools/call failure path must set is_error=True for SDK 2.x contract"
        )
        # Error payload preserved for backward compatibility
        import json as _json
        assert len(result.content) >= 1
        payload = _json.loads(result.content[0].text)
        assert payload.get("status") == "error"

    def test_build_mcp_server_call_tool_success_returns_is_error_false(self):
        """SDK 2.x contract: successful tools/call must return is_error=False (or unset)."""
        from mcp.types import CallToolResult
        from mnemosyne.mcp_server import _build_mcp_server

        server = _build_mcp_server()
        entry = server.get_request_handler("tools/call")
        assert entry is not None
        assert callable(entry.handler)
        on_call_tool = entry.handler

        # Use a real tool definition and call it through the path. We mock
        # handle_tool_call to return a minimal valid result so we don't need
        # the full DB-backed stack here.
        import asyncio
        from unittest.mock import patch

        class _Params:
            name = "mnemosyne_stats"
            arguments = None

        with patch("mnemosyne.mcp_server.handle_tool_call", return_value={"status": "ok"}) as handle_call:
            result = asyncio.run(on_call_tool(ctx=None, params=_Params()))
        handle_call.assert_called_once_with("mnemosyne_stats", {})
        assert isinstance(result, CallToolResult)
        assert not result.is_error, (
            "successful tools/call must have is_error=False (or None)"
        )
        assert json.loads(result.content[0].text) == {"status": "ok"}

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
