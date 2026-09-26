"""Call-time binding and writer-provenance regression tests (P1b).

Covers the multiplex incident: when one gateway process serves several
Hermes homes, provider state used to be captured once at initialization
and the last home to initialize won every later call — writes from one
home could land in another home's database. These tests drive a provider
with a fake beam and a context-var "home" so they run anywhere, with no
live stores and no hermes install.

Also pins the two supporting behaviors that make the incident
detectable and preventable:
  - canonical rows carry writer provenance (writer_id / writer_home);
  - a canonical write through an instance bound to another profile fails
    closed instead of silently rerouting;
  - mnemosyne_invalidate routes to the shared surface when (and only
    when) the id namespace or an explicit bank says so.
"""

from __future__ import annotations

import contextvars
import json
import sqlite3
import sys
import types

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

    monkeypatch.setattr(mnemosyne_hermes, "_p1b_home_key", fake_home_key)
    _HOME.set(None)


class _RecordingBeam:
    """Minimal BeamMemory stand-in: real file-backed db_path so helpers that
    build CanonicalStore/AuditLog from the beam stay hermetic under tmp_path."""

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


def _provider(tmp_path, monkeypatch, **init_kwargs):
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    p = MnemosyneMemoryProvider()
    home = str(tmp_path / "h")
    _HOME.set(home)  # this turn carries its own home, as a real multiplexed turn does
    p.initialize("sess", hermes_home=home, **init_kwargs)
    return p


# --------------------------------------------------------------------------
# The incident itself: last-init-must-not-win at call time
# --------------------------------------------------------------------------

def test_last_initialized_home_does_not_win_at_call_time(tmp_path, monkeypatch):
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    p = MnemosyneMemoryProvider()

    _HOME.set("home-a")
    p.initialize("sess-a", hermes_home="home-a")
    beam_a = p._beam
    assert beam_a is not None

    _HOME.set("home-b")
    p.initialize("sess-b", hermes_home="home-b")
    beam_b = p._beam
    assert beam_b is not None and beam_b is not beam_a

    # A turn scoped to home-a, arriving after home-b initialized, must still
    # dispatch against home-a's beam. Under the old ambient slot it got
    # home-b's — that was the misfile.
    _HOME.set("home-a")
    assert p._beam is beam_a
    _HOME.set("home-b")
    assert p._beam is beam_b


def test_unknown_home_at_call_time_falls_back_to_ambient(tmp_path, monkeypatch):
    # Out-of-turn callers (cron/teardown) keep last-init behavior: no binding
    # for the ambient home means the ambient slot answers, never None.
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    p = MnemosyneMemoryProvider()
    _HOME.set("home-a")
    p.initialize("sess-a", hermes_home="home-a")
    beam_a = p._beam
    _HOME.set("never-initialized-home")
    assert p._beam is beam_a


# --------------------------------------------------------------------------
# Writer provenance on canonical facts
# --------------------------------------------------------------------------

def test_canonical_supersede_records_writer_per_version(tmp_path):
    from mnemosyne.core.canonical import CanonicalStore

    db = str(tmp_path / "canonical.db")
    store = CanonicalStore(db_path=db)
    store.remember("owner1", "identity", "role", "engineer",
                   writer_id="writer-one", writer_home="home-a")
    store.remember("owner1", "identity", "role", "systems engineer",
                   writer_id="writer-two", writer_home="home-b")

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = {
        r["version"]: dict(r)
        for r in con.execute(
            "SELECT version, writer_id, writer_home, valid_until "
            "FROM canonical_facts WHERE owner_id='owner1' AND category='identity'"
        )
    }
    con.close()
    assert rows[2]["writer_id"] == "writer-two"
    assert rows[2]["writer_home"] == "home-b"
    assert rows[2]["valid_until"] is None  # current
    assert rows[1]["writer_id"] == "writer-one"  # history keeps its writer
    assert rows[1]["valid_until"] is not None


def _install_fake_hermes_modules(monkeypatch, tmp_path, profile_name):
    profiles = types.ModuleType("hermes_cli.profiles")
    profiles.get_active_profile_name = lambda: profile_name
    cli = types.ModuleType("hermes_cli")
    cli.profiles = profiles
    consts = types.ModuleType("hermes_constants")
    consts.get_hermes_home = lambda: tmp_path / "active-home"
    consts.hermes_home_key = lambda h: str(h)
    monkeypatch.setitem(sys.modules, "hermes_cli", cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", profiles)
    monkeypatch.setitem(sys.modules, "hermes_constants", consts)
    return profiles


def test_provider_canonical_write_stamps_writer(tmp_path, monkeypatch):
    _install_fake_hermes_modules(monkeypatch, tmp_path, "alice")
    p = _provider(tmp_path, monkeypatch)
    p._agent_identity = "alice"

    out = json.loads(p._handle_remember_canonical(
        {"category": "identity", "name": "role", "body": "engineer"}
    ))
    assert out["status"] in ("created", "updated")

    row = p._beam.canonical.recall("alice", "identity", "role")
    assert row is not None
    assert row["writer_id"] == "alice"
    assert row["writer_home"] == str(tmp_path / "active-home")


def test_canonical_write_guard_fails_closed_on_mismatch(tmp_path, monkeypatch):
    profiles = _install_fake_hermes_modules(monkeypatch, tmp_path, "alice")
    p = _provider(tmp_path, monkeypatch)
    p._agent_identity = "alice"

    # Turn owned by the bound profile: guard allows the write.
    assert p._canonical_write_guard("mnemosyne_remember_canonical") is None

    # Turn owned by a different profile through this instance: loud,
    # structured refusal — never a silent reroute.
    profiles.get_active_profile_name = lambda: "bob"
    err = json.loads(p._canonical_write_guard("mnemosyne_remember_canonical"))
    assert err["status"] == "canonical_owner_mismatch"
    assert err["bound_owner"] == "alice"
    assert err["active_profile"] == "bob"


# --------------------------------------------------------------------------
# Invalidate bank routing (surface branch)
# --------------------------------------------------------------------------

def test_invalidate_routes_by_id_namespace_and_explicit_bank(tmp_path, monkeypatch):
    p = _provider(tmp_path, monkeypatch)
    surface = _RecordingBeam(db_path=str(tmp_path / "surface.db"))
    p._surface_beam = surface
    monkeypatch.setattr(p, "_require_surface_beam", lambda: None)

    # Bare hex id: private namespace.
    out = json.loads(p._handle_invalidate({"memory_id": "abc123"}))
    assert out["bank"] == "private" and out["status"] == "invalidated"
    assert p._beam.invalidated == [("abc123", None)]
    assert surface.invalidated == []

    # sf_ prefix: surface namespace minted by shared_remember.
    out = json.loads(p._handle_invalidate(
        {"memory_id": "sf_deadbeef", "replacement_id": "sf_cafe"}))
    assert out["bank"] == "surface"
    assert surface.invalidated == [("sf_deadbeef", "sf_cafe")]

    # Explicit bank wins over the prefix inference.
    out = json.loads(p._handle_invalidate({"memory_id": "sf_other", "bank": "private"}))
    assert out["bank"] == "private"
    assert p._beam.invalidated[-1] == ("sf_other", None)

    # Unknown bank is refused, not defaulted.
    out = json.loads(p._handle_invalidate({"memory_id": "x1", "bank": "public"}))
    assert "unknown bank" in out["error"]
