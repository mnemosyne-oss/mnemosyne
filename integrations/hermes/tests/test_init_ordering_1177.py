"""Regression test for #1177: _initialize_locked isolation ordering defect.

Ensures that when HERMES_HOME is absent from os.environ and profile home is delivered
via ContextVar override (set_hermes_home_override), the provider resolves profile
isolation using the configured hermes_home and settles canonical owner to the profile
name ('markets') instead of 'default'.
"""
from __future__ import annotations

import contextvars
import sys
import types
from pathlib import Path

import pytest

import mnemosyne_hermes as m

try:
    import hermes_constants as hc
except ImportError:
    hc = None


@pytest.fixture
def host_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    home = tmp_path / ".hermes" / "profiles" / "markets"
    home.mkdir(parents=True)

    if hc is not None and getattr(hc, "__file__", None) is not None:
        c_mod = hc
        tok = c_mod.set_hermes_home_override(str(home))

        def reset_fn():
            c_mod._HERMES_HOME_OVERRIDE.reset(tok)
    else:
        # Standalone CI double when hermes-agent is not installed
        override_var = contextvars.ContextVar("test_hermes_home_override", default=str(home))
        c_mod = types.ModuleType("hermes_constants")
        c_mod._HERMES_HOME_OVERRIDE = override_var

        def _set_override(p):
            override_var.set(str(p) if p is not None else None)

        def _get_home():
            val = override_var.get()
            return Path(val) if val is not None else tmp_path / ".hermes"

        c_mod.set_hermes_home_override = _set_override
        c_mod.get_hermes_home = _get_home
        c_mod.hermes_home_key = lambda h: str(h)
        monkeypatch.setitem(sys.modules, "hermes_constants", c_mod)

        def reset_fn():
            pass

    if "hermes_cli" not in sys.modules or "hermes_cli.profiles" not in sys.modules:
        cli = types.ModuleType("hermes_cli")
        prof = types.ModuleType("hermes_cli.profiles")
        prof.get_active_profile_name = lambda: "markets"
        cli.profiles = prof
        monkeypatch.setitem(sys.modules, "hermes_cli", cli)
        monkeypatch.setitem(sys.modules, "hermes_cli.profiles", prof)

    return home, reset_fn


@pytest.fixture
def agent_init(monkeypatch):
    if hc is not None and getattr(hc, "__file__", None) is not None:
        try:
            agent_root = str(Path(hc.__file__).resolve().parent)
            monkeypatch.syspath_prepend(agent_root)
            import agent.agent_init as ai
            return ai
        except Exception:
            pass

    ai = types.ModuleType("agent.agent_init")
    ai._GATEWAY_IDENTITY_PARAMS = (
        "gateway_user_id", "gateway_chat_id", "gateway_channel_id",
        "gateway_session_key", "gateway_app_id", "gateway_account_id",
    )

    def _memory_provider_init_kwargs(agent, platform):
        kw = {
            "session_id": agent.session_id,
            "platform": platform or "cli",
            "hermes_home": str(sys.modules["hermes_constants"].get_hermes_home()),
            "agent_context": "primary",
            "agent_identity": "markets",
            "agent_workspace": "hermes",
        }
        return kw

    ai._memory_provider_init_kwargs = _memory_provider_init_kwargs
    return ai


@pytest.mark.parametrize("isolation", ["true"])
def test_profile_home_identity_must_settle_on_profile(tmp_path, monkeypatch, host_environment, agent_init, isolation):
    home, reset_fn = host_environment
    (home / "config.yaml").write_text(f"memory:\n  mnemosyne:\n    profile_isolation: {isolation}\n")

    ids = {f"_{i}": None for i in agent_init._GATEWAY_IDENTITY_PARAMS}
    agent = types.SimpleNamespace(
        **ids,
        session_id="s",
        _session_db=None,
        session_cwd=None,
        _emit_warning=lambda *a, **k: None,
        _emit_status=lambda *a, **k: None,
    )

    try:
        kw = agent_init._memory_provider_init_kwargs(agent, "desktop")
        assert kw.get("agent_identity") == "markets"
        provider = m.MnemosyneMemoryProvider()
        try:
            provider.initialize(**kw)
            owner = provider._canonical_owner()
        finally:
            provider.shutdown()
    finally:
        reset_fn()

    assert owner == "markets", f"owner settled to {owner!r}; identity lost at init"
