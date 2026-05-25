"""Tests for memory audit log integration."""

from __future__ import annotations

import json
from pathlib import Path

from mnemosyne.core.beam import BeamMemory
from hermes_memory_provider import MnemosyneMemoryProvider
from hermes_memory_provider.audit import AuditLog


def _provider(tmp_path: Path) -> MnemosyneMemoryProvider:
    db_path = tmp_path / "banks" / "test" / "mnemosyne.db"
    beam = BeamMemory(session_id="audit-test", db_path=db_path)
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider._session_id = "audit-test"
    provider._agent_context = "primary"
    provider._profile_isolation_enabled = True
    provider._init_audit_log()
    return provider


def _call(provider: MnemosyneMemoryProvider, name: str, args: dict) -> dict:
    return json.loads(provider.handle_tool_call(name, args))


class TestAuditLogModule:
    def test_creates_table(self, tmp_path):
        db_path = tmp_path / "audit.db"
        log = AuditLog(db_path)
        assert log.count() == 0
        log.close()

    def test_record_and_query(self, tmp_path):
        db_path = tmp_path / "audit.db"
        log = AuditLog(db_path)
        log.record("remember", memory_id="m1", bank="private", scope="global")
        log.record("forget", memory_id="m2", bank="private")
        assert log.count() == 2
        events = log.query(limit=10)
        assert events[0]["action"] == "forget"
        assert events[1]["action"] == "remember"
        log.close()

    def test_never_raises_on_bad_path(self, tmp_path):
        log = AuditLog(Path("/nonexistent/dir/audit.db"))
        log.record("remember", memory_id="x")
        assert log.count() == 0


class TestAuditIntegration:
    def test_remember_creates_audit_event(self, tmp_path):
        provider = _provider(tmp_path)
        result = _call(provider, "mnemosyne_remember", {
            "content": "audit test fact",
            "source": "fact",
            "importance": 0.7,
        })
        assert result["status"] == "stored"
        events = provider._audit.query(limit=10)
        assert len(events) == 1
        assert events[0]["action"] == "remember"
        assert events[0]["memory_id"] == result["memory_id"]
        assert events[0]["bank"] == "private"
        assert events[0]["source_tool"] == "mnemosyne_remember"

    def test_forget_creates_audit_event(self, tmp_path):
        provider = _provider(tmp_path)
        stored = _call(provider, "mnemosyne_remember", {
            "content": "to be forgotten",
            "source": "fact",
        })
        _call(provider, "mnemosyne_forget", {"memory_id": stored["memory_id"]})
        events = provider._audit.query(limit=10)
        assert len(events) == 2
        assert events[0]["action"] == "forget"
        assert events[0]["memory_id"] == stored["memory_id"]

    def test_forget_not_found_no_audit(self, tmp_path):
        provider = _provider(tmp_path)
        _call(provider, "mnemosyne_forget", {"memory_id": "nonexistent"})
        events = provider._audit.query(limit=10)
        assert len(events) == 0

    def test_invalidate_creates_audit_event(self, tmp_path):
        provider = _provider(tmp_path)
        stored = _call(provider, "mnemosyne_remember", {
            "content": "will be invalidated",
            "source": "fact",
        })
        _call(provider, "mnemosyne_invalidate", {"memory_id": stored["memory_id"]})
        events = provider._audit.query(limit=10)
        assert len(events) == 2
        assert events[0]["action"] == "invalidate"
        assert events[0]["memory_id"] == stored["memory_id"]

    def test_sleep_creates_audit_event(self, tmp_path):
        provider = _provider(tmp_path)
        _call(provider, "mnemosyne_remember", {
            "content": "sleep test",
            "source": "fact",
        })
        _call(provider, "mnemosyne_sleep", {})
        events = provider._audit.query(limit=10)
        sleep_events = [e for e in events if e["action"] == "sleep"]
        assert len(sleep_events) == 1
        assert sleep_events[0]["source_tool"] == "mnemosyne_sleep"

    def test_sleep_dry_run_no_audit(self, tmp_path):
        provider = _provider(tmp_path)
        _call(provider, "mnemosyne_sleep", {"dry_run": True})
        events = provider._audit.query(limit=10)
        sleep_events = [e for e in events if e["action"] == "sleep"]
        assert len(sleep_events) == 0

    def test_shared_remember_creates_audit_event(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
        provider = MnemosyneMemoryProvider()
        provider.initialize(
            session_id="audit-shared-test",
            hermes_home=str(tmp_path / "profiles" / "test"),
            agent_identity="test",
            shared_surface_path=str(tmp_path / "shared" / "mnemosyne.db"),
        )
        provider._init_audit_log()
        result = _call(provider, "mnemosyne_shared_remember", {
            "content": "shared audit fact",
            "kind": "meta",
        })
        assert result["status"] == "stored_shared"
        events = provider._audit.query(limit=10)
        shared_events = [e for e in events if e["action"] == "shared_remember"]
        assert len(shared_events) == 1
        assert shared_events[0]["bank"] == "surface"

    def test_shared_forget_creates_audit_event(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
        provider = MnemosyneMemoryProvider()
        provider.initialize(
            session_id="audit-shared-test",
            hermes_home=str(tmp_path / "profiles" / "test"),
            agent_identity="test",
            shared_surface_path=str(tmp_path / "shared" / "mnemosyne.db"),
        )
        provider._init_audit_log()
        stored = _call(provider, "mnemosyne_shared_remember", {
            "content": "shared to delete",
            "kind": "meta",
        })
        _call(provider, "mnemosyne_shared_forget", {"memory_id": stored["memory_id"]})
        events = provider._audit.query(limit=10)
        forget_events = [e for e in events if e["action"] == "shared_forget"]
        assert len(forget_events) == 1
        assert forget_events[0]["memory_id"] == stored["memory_id"]

    def test_no_audit_when_beam_missing(self, tmp_path):
        provider = MnemosyneMemoryProvider()
        provider._session_id = "no-beam"
        provider._agent_context = "primary"
        # _audit is None, should not crash
        provider._audit_event("remember", memory_id="x")
