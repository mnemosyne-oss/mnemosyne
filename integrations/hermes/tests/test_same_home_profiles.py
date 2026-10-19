"""Same-home multi-profile isolation (#1188).

Hermes multiplexes profiles through one provider instance. Under
``profile_isolation`` those profiles can share a Hermes home while using
separate banks, so binding state must be keyed by home plus active profile.
Without isolation the home remains the unit of isolation, and profiles
sharing it intentionally share one store.
"""
from __future__ import annotations

import contextvars
import sys
import types

import pytest

import mnemosyne_hermes
from mnemosyne_hermes import MnemosyneMemoryProvider

_HOME = contextvars.ContextVar("test_same_home", default=None)
_PROFILE = contextvars.ContextVar("test_same_profile", default=None)


@pytest.fixture(autouse=True)
def _fake_host(monkeypatch):
    monkeypatch.setattr(
        mnemosyne_hermes,
        "_p1b_home_key",
        lambda home=None: str(home if home is not None else _HOME.get() or "default"),
    )

    def current_key():
        home = _HOME.get()
        return None if home is None else str(home)

    monkeypatch.setattr(mnemosyne_hermes, "_p1b_current_key", current_key)
    profiles = types.ModuleType("hermes_cli.profiles")
    profiles.get_active_profile_name = lambda: _PROFILE.get()
    cli = types.ModuleType("hermes_cli")
    cli.profiles = profiles
    monkeypatch.setitem(sys.modules, "hermes_cli", cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", profiles)
    _HOME.set(None)
    _PROFILE.set(None)


class _Beam:
    author_id = None

    def __init__(self, *args, db_path=None, session_id=None, **kwargs):
        self.db_path = str(db_path or f":memory:{id(self)}")
        self.conn = None
        self.canonical = None
        self.session_id = session_id

    def invalidate(self, *args, **kwargs):
        return True

    def get(self, *args, **kwargs):
        return None


class _Wrapper:
    """Recording stand-in for the Mnemosyne facade used by profile isolation."""

    def __init__(self, *args, **kwargs):
        self.beam = _Beam(db_path=kwargs.get("db_path"), session_id=kwargs.get("session_id"))
        self.db_path = self.beam.db_path

    def close(self):
        pass


def test_binding_key_includes_profile_only_when_isolated():
    assert mnemosyne_hermes._p1b_binding_key("home", "agent-a", False) == "home"
    assert mnemosyne_hermes._p1b_binding_key("home", None, True) == "home"
    assert mnemosyne_hermes._p1b_binding_key("home", "agent-a", True) == "home\x1fagent-a"


def test_same_home_profiles_get_separate_bindings(tmp_path, monkeypatch):
    import mnemosyne.core.memory as memory_module

    monkeypatch.setattr(memory_module, "Mnemosyne", _Wrapper)
    provider = MnemosyneMemoryProvider()
    home = str(tmp_path / "shared-home")
    _HOME.set(home)

    _PROFILE.set("agent-a")
    provider.initialize("sess-a", hermes_home=home, profile_isolation=True)
    beam_a = provider._beam
    _PROFILE.set("agent-b")
    provider.initialize("sess-b", hermes_home=home, profile_isolation=True)
    beam_b = provider._beam

    assert beam_a is not None
    assert beam_b is not None
    assert beam_a is not beam_b
    assert beam_a.db_path != beam_b.db_path

    _PROFILE.set("agent-a")
    assert provider._beam is beam_a
    _PROFILE.set("agent-b")
    assert provider._beam is beam_b

    # A profile that never initialized must not read or write another
    # profile's binding in the same home.
    _PROFILE.set("agent-c")
    assert provider._beam is None
    with pytest.raises(RuntimeError):
        provider._beam = object()


def test_same_home_without_isolation_uses_the_last_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _Beam)
    provider = MnemosyneMemoryProvider()
    home = str(tmp_path / "shared-home")
    _HOME.set(home)

    _PROFILE.set("agent-a")
    provider.initialize("sess-a", hermes_home=home, profile_isolation=False)
    beam_a = provider._beam
    _PROFILE.set("agent-b")
    provider.initialize("sess-b", hermes_home=home, profile_isolation=False)
    beam_b = provider._beam

    assert beam_a is not beam_b
    # Isolation off means the home is the unit: both profiles resolve to the
    # last initialized binding, which is the documented opt-in behavior.
    assert provider._beam is beam_b
    _PROFILE.set("agent-a")
    assert provider._beam is beam_b


def test_missing_profile_getter_keeps_the_home_key(tmp_path, monkeypatch):
    import mnemosyne.core.memory as memory_module

    monkeypatch.setattr(memory_module, "Mnemosyne", _Wrapper)
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_active_profile", lambda: None)
    provider = MnemosyneMemoryProvider()
    home = str(tmp_path / "shared-home")
    _HOME.set(home)

    provider.initialize(
        "sess", hermes_home=home, profile_isolation=True, agent_identity="profile-b"
    )

    # Without a call-time profile getter the home is the only key, so init and
    # call time agree. The explicit identity still selects the bank path.
    assert provider._beam is not None
    assert "banks" in provider._beam.db_path
