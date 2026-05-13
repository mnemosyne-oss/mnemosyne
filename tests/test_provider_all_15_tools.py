"""
Tests for MnemosyneMemoryProvider — all 15 tools wired in provider mode.

Verifies schema registration, dispatch routing, and handler execution
for each of the 8 tools ported from hermes_plugin (scratchpad, export,
update, forget, import, diagnose) plus the 7 already-existing tools.
"""

import json
import pytest
from pathlib import Path

from mnemosyne.core.beam import BeamMemory
from hermes_memory_provider import MnemosyneMemoryProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_beam(tmp_path):
    """Create a BeamMemory backed by a temporary database."""
    db_path = Path(tmp_path) / "test.db"
    return BeamMemory(session_id="test_provider", db_path=db_path)


def _build_provider(beam) -> MnemosyneMemoryProvider:
    """Return a ready-to-test MnemosyneMemoryProvider with beam injected."""
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider._session_id = "test_provider"
    provider._agent_context = "primary"
    return provider


def _provider(tmp_path) -> MnemosyneMemoryProvider:
    """One-liner to get a fully-wired provider backed by a tmp DB."""
    return _build_provider(_make_beam(tmp_path))


def _tool_names(provider) -> set[str]:
    return {s["name"] for s in provider.get_tool_schemas()}


# ---------------------------------------------------------------------------
# Tool schema & dispatch registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    """Verify the provider registers and dispatches all 15 tools."""

    def test_all_15_tools_registered(self, tmp_path):
        provider = _provider(tmp_path)
        names = _tool_names(provider)
        assert len(names) == 15, f"Expected 15 tools, got {len(names)}"

    def test_new_8_tools_present(self, tmp_path):
        provider = _provider(tmp_path)
        names = _tool_names(provider)
        for tool in ("scratchpad_write", "scratchpad_read", "scratchpad_clear",
                     "export", "update", "forget", "import", "diagnose"):
            assert f"mnemosyne_{tool}" in names

    def test_existing_7_tools_present(self, tmp_path):
        provider = _provider(tmp_path)
        names = _tool_names(provider)
        for tool in ("remember", "recall", "sleep", "stats",
                     "invalidate", "triple_add", "triple_query"):
            assert f"mnemosyne_{tool}" in names

    def test_unknown_tool_returns_error(self, tmp_path):
        provider = _provider(tmp_path)
        result = json.loads(provider.handle_tool_call("mnemosyne_nonexistent", {}))
        assert "error" in result
        assert "Unknown" in result["error"]


# ---------------------------------------------------------------------------
# Scratchpad tools
# ---------------------------------------------------------------------------

class TestScratchpad:
    def test_write_and_read(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_scratchpad_write({"content": "hello"}))
        assert r["status"] == "written"
        r2 = json.loads(provider._handle_scratchpad_read({}))
        assert r2["entries_count"] >= 1

    def test_write_empty(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_scratchpad_write({"content": ""}))
        assert "error" in r

    def test_clear(self, tmp_path):
        provider = _provider(tmp_path)
        provider._handle_scratchpad_write({"content": "temp"})
        r = json.loads(provider._handle_scratchpad_clear({}))
        assert r["status"] == "cleared"
        r2 = json.loads(provider._handle_scratchpad_read({}))
        assert r2["entries_count"] == 0

    def test_dispatch_scratchpad_write(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider.handle_tool_call("mnemosyne_scratchpad_write",
                                                  {"content": "dispatch"}))
        assert r["status"] == "written"

    def test_dispatch_scratchpad_read(self, tmp_path):
        provider = _provider(tmp_path)
        provider._handle_scratchpad_write({"content": "ping"})
        r = json.loads(provider.handle_tool_call("mnemosyne_scratchpad_read", {}))
        assert r["entries_count"] >= 1

    def test_dispatch_scratchpad_clear(self, tmp_path):
        provider = _provider(tmp_path)
        provider._handle_scratchpad_write({"content": "del"})
        r = json.loads(provider.handle_tool_call("mnemosyne_scratchpad_clear", {}))
        assert r["status"] == "cleared"


# ---------------------------------------------------------------------------
# Update tool
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_content(self, tmp_path):
        beam = _make_beam(tmp_path)
        provider = _build_provider(beam)
        mid = beam.remember("original", importance=0.5)
        r = json.loads(provider._handle_update({"memory_id": mid,
                                                 "content": "updated"}))
        assert r["status"] == "updated"
        found = beam.recall("updated", top_k=5)
        assert any("updated" in x.get("content", "") for x in found)

    def test_update_not_found(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_update({"memory_id": "noid",
                                                 "content": "x"}))
        assert r["status"] == "not_found"

    def test_update_missing_id(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_update({}))
        assert "error" in r

    def test_dispatch_update(self, tmp_path):
        beam = _make_beam(tmp_path)
        provider = _build_provider(beam)
        mid = beam.remember("old", importance=0.5)
        r = json.loads(provider.handle_tool_call("mnemosyne_update",
                                                  {"memory_id": mid,
                                                   "content": "new"}))
        assert r["status"] == "updated"


# ---------------------------------------------------------------------------
# Forget tool
# ---------------------------------------------------------------------------

class TestForget:
    def test_forget_existing(self, tmp_path):
        beam = _make_beam(tmp_path)
        provider = _build_provider(beam)
        mid = beam.remember("to delete", importance=0.5)
        r = json.loads(provider._handle_forget({"memory_id": mid}))
        assert r["status"] == "deleted"
        found = beam.recall("delete", top_k=5)
        assert not any(x.get("id") == mid for x in found)

    def test_forget_not_found(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_forget({"memory_id": "noid"}))
        assert r["status"] == "not_found"

    def test_forget_missing_id(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_forget({}))
        assert "error" in r

    def test_dispatch_forget(self, tmp_path):
        beam = _make_beam(tmp_path)
        provider = _build_provider(beam)
        mid = beam.remember("del", importance=0.5)
        r = json.loads(provider.handle_tool_call("mnemosyne_forget",
                                                  {"memory_id": mid}))
        assert r["status"] == "deleted"


# ---------------------------------------------------------------------------
# Export tool
# ---------------------------------------------------------------------------

class TestExport:
    def test_export_to_file(self, tmp_path):
        beam = _make_beam(tmp_path)
        provider = _build_provider(beam)
        beam.remember("test export", importance=0.8)
        beam.scratchpad_write("note")
        out = tmp_path / "export.json"
        r = json.loads(provider._handle_export({"output_path": str(out)}))
        assert r["status"] == "exported"
        assert out.exists()
        data = json.loads(out.read_text())
        assert "working_memory" in data

    def test_export_missing_path(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_export({}))
        assert "error" in r

    def test_dispatch_export(self, tmp_path):
        beam = _make_beam(tmp_path)
        provider = _build_provider(beam)
        beam.remember("dispatch export", importance=0.5)
        out = tmp_path / "dispatch.json"
        r = json.loads(provider.handle_tool_call("mnemosyne_export",
                                                  {"output_path": str(out)}))
        assert r["status"] == "exported"
        assert out.exists()


# ---------------------------------------------------------------------------
# Import tool (file path)
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_from_file(self, tmp_path):
        # Export first
        beam1 = _make_beam(tmp_path / "src")
        p1 = _build_provider(beam1)
        beam1.remember("source memory", importance=0.5)
        export_path = tmp_path / "data.json"
        p1._handle_export({"output_path": str(export_path)})

        # Import into a fresh DB
        beam2 = _make_beam(tmp_path / "dst")
        p2 = _build_provider(beam2)
        r = json.loads(p2._handle_import({"input_path": str(export_path)}))
        assert r["status"] == "imported"
        assert r["stats"]["beam"]["working_memory"]["inserted"] >= 1

    def test_import_missing_args(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_import({}))
        assert "error" in r


# ---------------------------------------------------------------------------
# Diagnose tool
# ---------------------------------------------------------------------------

class TestDiagnose:
    def test_diagnose_returns_valid_json(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider._handle_diagnose({}))
        assert isinstance(r, dict)
        assert len(r) > 0

    def test_dispatch_diagnose(self, tmp_path):
        provider = _provider(tmp_path)
        r = json.loads(provider.handle_tool_call("mnemosyne_diagnose", {}))
        assert isinstance(r, dict)


# ---------------------------------------------------------------------------
# Provider not-initialized guard
# ---------------------------------------------------------------------------

class TestUnavailableGuard:
    def test_tool_call_returns_reason_when_not_initialized(self):
        provider = MnemosyneMemoryProvider()
        provider._beam = None
        provider._hermes_home = ""
        r = json.loads(provider.handle_tool_call("mnemosyne_remember",
                                                  {"content": "test"}))
        assert r["status"] == "memory_unavailable"

    def test_system_prompt_empty_when_not_initialized(self):
        provider = MnemosyneMemoryProvider()
        provider._beam = None
        assert provider.system_prompt_block() == ""
