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

    def fake_current_key():
        # Mirrors _p1b_current_key: None means out-of-turn (ambient answers).
        home = _HOME.get()
        return None if home is None else str(home)

    monkeypatch.setattr(mnemosyne_hermes, "_p1b_home_key", fake_home_key)
    monkeypatch.setattr(mnemosyne_hermes, "_p1b_current_key", fake_current_key)
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


def test_unknown_home_in_turn_fails_closed(tmp_path, monkeypatch):
    # #1050 review point 1: an in-turn call from a home that never
    # initialized must NOT fall back to another home's binding.
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    p = MnemosyneMemoryProvider()
    _HOME.set("home-a")
    p.initialize("sess-a", hermes_home="home-a")
    beam_a = p._beam
    assert beam_a is not None

    _HOME.set("never-initialized-home")
    # Reads resolve to an empty slot, never another home's beam.
    assert p._beam is not beam_a and p._beam is None
    assert p._session_id is None and p._agent_identity == ""
    # A read degrades to the existing unavailable surface, it does not raise.
    assert p.prefetch("anything", session_id="sess-x") == ""
    # A write fails closed instead of persisting into home-a's store.
    out = json.loads(p.handle_tool_call("mnemosyne_remember", {"content": "x"}))
    assert out.get("status") == "memory_unavailable", (
        f"unknown-home write did not fail closed: {out}"
    )
    assert out.get("reason_code") == "never_initialized"
    # The refused write created no binding for the unknown home and did not
    # reroute: home-a's slot still holds exactly its own beam.
    assert "never-initialized-home" not in p.__dict__["_bindings"]
    holders = [k for k, b in p.__dict__["_bindings"].items() if b.get("beam") is not None]
    assert holders == ["home-a"], holders
    assert p.__dict__["_bindings"]["home-a"]["beam"] is beam_a
    _HOME.set("home-a")
    assert p._beam is beam_a


def test_out_of_turn_still_uses_ambient_binding(tmp_path, monkeypatch):
    # Ambient fallback is reserved for genuinely out-of-turn callers
    # (cron/teardown/workers): with no turn home, the ambient slot answers.
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    p = MnemosyneMemoryProvider()
    _HOME.set("home-a")
    p.initialize("sess-a", hermes_home="home-a")
    beam_a = p._beam
    _HOME.set(None)
    assert p._beam is beam_a


def test_failed_second_home_init_restores_ambient_reads(tmp_path, monkeypatch):
    """A failed initialize(home-b) must not black out home-a's service.

    CodeRabbit point on #1050 (head 9ae0531): _initialize_locked mirrors the
    target home into _ambient_key BEFORE construction; when construction
    raises, B's slot is left empty but the ambient key stayed on B — so
    out-of-turn readers (cron prefetch, teardown flush) resolved to the dead
    B slot instead of the still-live home-a binding.
    """
    class _BeamThatDiesForB(_RecordingBeam):
        def __init__(self, *args, db_path=None, **kwargs):
            if db_path is not None and "home-b" in str(db_path):
                raise sqlite3.OperationalError("simulated corrupt store")
            super().__init__(*args, db_path=db_path, **kwargs)

    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _BeamThatDiesForB)
    p = MnemosyneMemoryProvider()

    _HOME.set("home-a")
    p.initialize("sess-a", hermes_home="home-a")
    beam_a = p._beam
    assert beam_a is not None

    _HOME.set("home-b")
    p.initialize("sess-b", hermes_home="home-b")
    assert p._init_error is not None, "B's construction failure must be recorded"
    assert p.__dict__.get("_retry_init_args") is None, (
        "corrupt-store failure is non-transient: no retry may be pending, "
        "or the ambient restore would (correctly) not fire and this test "
        "would pin the wrong arm"
    )

    # In-turn reads still address B's own (empty) slot — never home-a's beam:
    # a turn scoped to the home that failed must not silently reroute.
    _HOME.set("home-b")
    assert p._beam is None, "dead B turn must read empty, not reroute to A"
    _HOME.set("home-a")
    assert p._beam is beam_a, "A's own turn must still reach A's beam"

    # THE FIX: out-of-turn reads fall back to the last LIVE ambient (home-a),
    # not to the dead ambient that the failed init left behind.
    _HOME.set(None)
    assert p._beam is beam_a, (
        "failed init stranded the ambient key on the dead home; out-of-turn "
        "service (cron/teardown) was blacked out by an unrelated home's failure"
    )


def test_skip_context_init_does_not_restore_previous_ambient(tmp_path, monkeypatch):
    """Deliberate skip contexts stay unavailable — the ambient key must
    remain on the skip slot so system_prompt_block() reports the skip, not a
    stale 'Active' from the previous home (C13/C27 contract). The restore
    added for failed inits must NOT widen to this path."""
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _RecordingBeam)
    p = MnemosyneMemoryProvider()

    _HOME.set("home-a")
    p.initialize("sess-a", hermes_home="home-a")
    beam_a = p._beam

    _HOME.set("home-b")
    p.initialize("sess-b", hermes_home="home-b", agent_context="subagent")
    assert p._unavailable_reason_code == "skipped_context"

    _HOME.set(None)
    assert p._beam is not beam_a, (
        "skip-context init must keep the provider reading as its own empty "
        "slot, not reclaim the previous home's live binding"
    )


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


def test_task_progress_write_carries_writer_stamp(tmp_path, monkeypatch):
    """task:progress rows are canonical writes — the attribution lane was
    storing them with EMPTY writer fields (CodeRabbit on 9ae0531: the
    remember() call in _handle_task_progress predated the writer args)."""
    _install_fake_hermes_modules(monkeypatch, tmp_path, "alice")
    p = _provider(tmp_path, monkeypatch)
    p._agent_identity = "alice"

    out = json.loads(p._handle_task_progress(
        {"action": "set", "task": "t1", "state": "halfway"}
    ))
    assert out["status"] == "set", out

    row = p._beam.canonical.recall("alice", "task:progress", "t1")
    assert row is not None
    assert row["writer_id"] == "alice", (
        "task:progress landed without the active profile's writer stamp"
    )
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
