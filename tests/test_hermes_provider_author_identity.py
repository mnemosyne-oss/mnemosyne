"""
Regression tests for issue #914: Hermes provider writes never carry author_id.

Expected fix (per maintainer design note on #914): a NARROW PER-WRITE identity
path. Identity resolution precedence: agent_identity kwarg > MNEMOSYNE_AUTHOR_ID
env > no identity. The identity is attached to WRITE calls via remember()'s
per-write author_id/author_type params — the Beam constructor's read identity
is deliberately NOT set, because recall author-scoping keys off it and setting
it would bypass session/channel scoping in prefetch (the exact regression the
thread-isolation suite guards against).

These tests use a REAL BeamMemory in a temp dir (matching the repo's own
shared-crud test conventions) rather than mocking the beam, so they exercise
the actual write path end to end.
"""

import json
import os
from pathlib import Path

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider


def _provider(tmp_path: Path, monkeypatch, agent_identity="sphinx"):
    data_dir = tmp_path / "mnemosyne-data"
    hermes_home = tmp_path / "profiles" / (agent_identity or "main")
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data_dir / "private"))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.delenv("MNEMOSYNE_AUTHOR_ID", raising=False)
    monkeypatch.delenv("MNEMOSYNE_AUTHOR_TYPE", raising=False)
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id="s1",
        hermes_home=str(hermes_home),
        agent_identity=agent_identity,
    )
    assert provider._beam is not None
    return provider


def _call(provider, name, args):
    return json.loads(provider.handle_tool_call(name, args))


class TestWriteIdentityResolution:
    """_resolve_author_identity precedence contract."""

    def test_kwarg_identity_wins(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", "env-agent")
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = "sphinx"
        assert provider._resolve_author_identity() == {"author_id": "sphinx"}

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", "env-agent")
        monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", "agent")
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = ""
        assert provider._resolve_author_identity() == {
            "author_id": "env-agent", "author_type": "agent",
        }

    def test_primary_is_generic(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_AUTHOR_ID", raising=False)
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = "primary"
        assert provider._resolve_author_identity() == {}

    def test_no_identity_empty(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_AUTHOR_ID", raising=False)
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = ""
        assert provider._resolve_author_identity() == {}


class TestConstructorsUnchanged:
    """Read identity must stay untouched: the beam's author_id stays None."""

    def test_beam_read_identity_is_none(self, tmp_path, monkeypatch):
        provider = _provider(tmp_path, monkeypatch, agent_identity="sphinx")
        assert provider._beam.author_id is None
        assert provider._beam.author_type is None


class TestRememberCarriesIdentity:
    """mnemosyne_remember stores rows with the resolved author, per-write."""

    def test_remember_row_has_author(self, tmp_path, monkeypatch):
        provider = _provider(tmp_path, monkeypatch, agent_identity="sphinx")
        result = _call(provider, "mnemosyne_remember", {
            "content": "james prefers tea", "scope": "session",
        })
        assert result["status"] == "stored"
        row = provider._beam.conn.execute(
            "SELECT author_id FROM working_memory WHERE content = 'james prefers tea'"
        ).fetchone()
        assert row is not None
        assert row[0] == "sphinx"
        # Read identity STILL untouched after a write
        assert provider._beam.author_id is None

    def test_remember_without_identity_stores_null(self, tmp_path, monkeypatch):
        provider = _provider(tmp_path, monkeypatch, agent_identity="")
        result = _call(provider, "mnemosyne_remember", {
            "content": "plain note", "scope": "session",
        })
        assert result["status"] == "stored"
        row = provider._beam.conn.execute(
            "SELECT author_id FROM working_memory WHERE content = 'plain note'"
        ).fetchone()
        assert row is not None
        assert row[0] is None

    def test_env_author_used_when_no_kwarg(self, tmp_path, monkeypatch):
        provider = _provider(tmp_path, monkeypatch, agent_identity="")
        monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", "env-agent")
        result = _call(provider, "mnemosyne_remember", {
            "content": "env authored", "scope": "session",
        })
        assert result["status"] == "stored"
        row = provider._beam.conn.execute(
            "SELECT author_id FROM working_memory WHERE content = 'env authored'"
        ).fetchone()
        assert row is not None
        assert row[0] == "env-agent"
