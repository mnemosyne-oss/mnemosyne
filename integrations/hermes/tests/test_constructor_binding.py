"""Construction must not write through the call-time binding property (#1050 follow-up).

The 0.7.4 constructor (shipped in c74c17e, PR #1050) wrote ``self._beam =
None`` through the ``_beam`` property setter, which routes into
``_write_slot()`` -> ``_p1b_current_key()``. When the provider is constructed
inside a turn that already carries a home key with no binding yet (a
multiplexed gateway serving several Hermes homes), that write raised

    RuntimeError: no binding for the current home ...; refusing to write
    through another home's binding

INSIDE THE CONSTRUCTOR -- before ``initialize()`` could create the binding.
The provider could never initialise for that profile, every memory write was
lost, and the agent still reported "Stored" to the user: a silent write loss
dressed as success.

The assignment is redundant anyway: the binding slot's ``beam`` default is
already ``None``, so dropping it changes no read behaviour. These tests pin
both halves of the contract:

  1. constructing under an unbound turn home must NOT raise;
  2. the fail-closed guard for genuine cross-home writes stays intact.
"""

from __future__ import annotations

import contextvars
import json

import pytest

import mnemosyne_hermes
from mnemosyne_hermes import MnemosyneMemoryProvider

_HOME = contextvars.ContextVar("test_hermes_home", default=None)


@pytest.fixture(autouse=True)
def _fake_home_keying(monkeypatch):
    def fake_home_key(home=None):
        if home is not None:
            return str(home)
        return str(_HOME.get() or "default")

    def fake_current_key():
        # Mirrors _p1b_current_key: None means out-of-turn (ambient answers).
        home = _HOME.get()
        return None if home is None else str(home)

    monkeypatch.setattr(mnemosyne_hermes, "_p1b_home_key", fake_home_key)
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_current_key", fake_current_key)
    _HOME.set(None)


class _RecordingBeam:
    """Minimal BeamMemory stand-in (same as test_calltime_bindings): real
    file-backed db_path so helpers that build CanonicalStore/AuditLog from the
    beam stay hermetic under tmp_path."""

    author_id = None

    def __init__(self, *args, db_path=None, session_id=None, **kwargs):
        self.db_path = str(db_path or f":memory:{id(self)}")
        self.conn = None
        self.canonical = None
        self.session_id = session_id
        self.invalidated: list[tuple] = []

    def invalidate(self, memory_id, replacement_id=None):
        self.invalidated.append((memory_id, replacement_id))
        return True


def test_constructor_under_unbound_turn_home_does_not_raise(tmp_path, monkeypatch):
    # Regression: released 0.7.4 raised RuntimeError here, before
    # initialize() ever ran, because __init__ wrote `self._beam = None`
    # through the property into an unbound home key.
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    _HOME.set("markets")  # turn scoped to a home that has not initialized yet

    p = MnemosyneMemoryProvider()  # must construct cleanly

    # Unbound home reads as an empty slot -- the same state the dropped
    # assignment tried (and failed) to write, so read behaviour is unchanged.
    assert p._beam is None

    # The provider is fully usable once that home initializes its binding.
    p.initialize("sess", hermes_home="markets")
    assert p._beam is not None


def test_cross_home_write_still_fails_closed_after_constructor_fix(tmp_path, monkeypatch):
    # The fix must not weaken the guard: a write from a home with no binding
    # still refuses loudly and never reroutes into another home's store.
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    p = MnemosyneMemoryProvider()
    _HOME.set("home-a")
    p.initialize("sess-a", hermes_home="home-a")
    beam_a = p._beam
    assert beam_a is not None

    _HOME.set("never-initialized-home")
    out = json.loads(p.handle_tool_call("mnemosyne_remember", {"content": "x"}))
    assert out.get("status") == "memory_unavailable", out
    assert out.get("reason_code") == "never_initialized"
    holders = [k for k, b in p.__dict__["_bindings"].items() if b.get("beam") is not None]
    assert holders == ["home-a"], holders
