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
from datetime import datetime, timedelta

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
        self.rows: dict = {}
        # None keeps invalidate() reporting success (the common path);
        # set False to exercise the failed-invalidation reporting below.
        self.invalidate_result = None

    def invalidate(self, memory_id, replacement_id=None):
        self.invalidated.append((memory_id, replacement_id))
        return True if self.invalidate_result is None else self.invalidate_result

    def get(self, memory_id):
        return self.rows.get(memory_id)


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


def test_canonical_write_guard_fails_closed_on_unresolvable_profile(tmp_path, monkeypatch):
    # An unresolvable turn profile proves nothing about ownership, so a
    # canonical write through it is refused — the fail-closed intent the
    # dispatch site documents. Both anomaly shapes count: an empty name
    # and a lookup that raises.
    profiles = _install_fake_hermes_modules(monkeypatch, tmp_path, "alice")
    p = _provider(tmp_path, monkeypatch)
    p._agent_identity = "alice"

    profiles.get_active_profile_name = lambda: ""
    err = json.loads(p._canonical_write_guard("mnemosyne_remember_canonical"))
    assert err["status"] == "canonical_profile_unavailable"
    assert err["tool"] == "mnemosyne_remember_canonical"

    def _boom():
        raise RuntimeError("profiles module down")

    profiles.get_active_profile_name = _boom
    err = json.loads(p._canonical_write_guard("mnemosyne_forget_canonical"))
    assert err["status"] == "canonical_profile_unavailable"
    assert err["tool"] == "mnemosyne_forget_canonical"


def test_task_progress_mutations_fail_closed_on_mismatch(tmp_path, monkeypatch):
    """Mutating task_progress (set/clear) must fail closed when the turn profile
    does not match the bound owner (#1177). Reads (get/list) remain ungated."""
    profiles = _install_fake_hermes_modules(monkeypatch, tmp_path, "alice")
    p = _provider(tmp_path, monkeypatch)
    p._agent_identity = "alice"

    # Alice sets her task progress normally.
    init_res = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "set", "task": "deploy", "state": "staging"},
    ))
    assert init_res["status"] == "set"

    # Seed a Bob task-progress row before switching active profile to Bob.
    profiles.get_active_profile_name = lambda: "bob"
    p._agent_identity = "bob"
    bob_init = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "set", "task": "deploy", "state": "idle"},
    ))
    assert bob_init["status"] == "set"

    # Switch provider back to Alice; turn profile switches to Bob.
    p._agent_identity = "alice"
    profiles.get_active_profile_name = lambda: "bob"

    set_err = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "set", "task": "deploy", "state": "production"},
    ))
    assert set_err["status"] == "canonical_owner_mismatch"
    assert set_err["error"] == "canonical_owner_mismatch"
    assert set_err["tool"] == "mnemosyne_task_progress"
    assert set_err["bound_owner"] == "alice"
    assert set_err["active_profile"] == "bob"

    # Alice's task progress and Bob's pre-seeded task progress are untouched.
    row = p._beam.canonical.recall("alice", "task:progress", "deploy")
    assert row is not None
    assert row["body"] == "staging"
    bob_row = p._beam.canonical.recall("bob", "task:progress", "deploy")
    assert bob_row is not None
    assert bob_row["body"] == "idle"

    clear_err = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "clear", "task": "deploy"},
    ))
    assert clear_err["status"] == "canonical_owner_mismatch"
    assert clear_err["tool"] == "mnemosyne_task_progress"

    # Tasks were not cleared — both Alice's and Bob's rows remain unchanged.
    row = p._beam.canonical.recall("alice", "task:progress", "deploy")
    assert row is not None
    assert row["body"] == "staging"
    bob_row = p._beam.canonical.recall("bob", "task:progress", "deploy")
    assert bob_row is not None
    assert bob_row["body"] == "idle"

    # Reads (get/list) remain accessible.
    get_res = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "get", "task": "deploy"},
    ))
    assert get_res["status"] == "found"
    assert get_res["state"] == "staging"

    list_res = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "list"},
    ))
    assert list_res["count"] == 1


def test_task_progress_mutations_fail_closed_on_unresolvable_profile(tmp_path, monkeypatch):
    """Mutating task_progress fails closed when active profile cannot be resolved (#1177)."""
    profiles = _install_fake_hermes_modules(monkeypatch, tmp_path, "alice")
    p = _provider(tmp_path, monkeypatch)
    p._agent_identity = "alice"

    # Seed an Alice task-progress row before disabling profile resolution.
    init_res = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "set", "task": "build", "state": "queued"},
    ))
    assert init_res["status"] == "set"

    profiles.get_active_profile_name = lambda: ""
    err_set = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "set", "task": "build", "state": "in_progress"},
    ))
    assert err_set["status"] == "canonical_profile_unavailable"
    assert err_set["error"] == "canonical_profile_unavailable"
    assert err_set["tool"] == "mnemosyne_task_progress"

    # Alice's seeded row remains present and unchanged after refused set.
    row = p._beam.canonical.recall("alice", "task:progress", "build")
    assert row is not None
    assert row["body"] == "queued"

    def _boom():
        raise RuntimeError("profiles lookup failed")

    profiles.get_active_profile_name = _boom
    err_clear = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "clear", "task": "build"},
    ))
    assert err_clear["status"] == "canonical_profile_unavailable"
    assert err_clear["tool"] == "mnemosyne_task_progress"

    # Alice's seeded row remains present and unchanged after refused clear.
    row = p._beam.canonical.recall("alice", "task:progress", "build")
    assert row is not None
    assert row["body"] == "queued"

    # Direct handler call also fails closed on unresolvable profile
    direct_err = json.loads(p._handle_task_progress(
        {"action": "set", "task": "build", "state": "in_progress"}
    ))
    assert direct_err["status"] == "canonical_profile_unavailable"

    # Seeded row is still present and unchanged after direct handler refusal.
    row = p._beam.canonical.recall("alice", "task:progress", "build")
    assert row is not None
    assert row["body"] == "queued"

    # Reads remain ungated even with unresolvable profile
    get_res = json.loads(p.handle_tool_call(
        "mnemosyne_task_progress",
        {"action": "get", "task": "build"},
    ))
    assert get_res["status"] == "found"
    assert get_res["state"] == "queued"


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


def test_invalidate_self_replacement_rejected_before_lookup(tmp_path, monkeypatch):
    # Parity with the root provider: a row cannot supersede itself, and the
    # caller must not have to read that off a memory_not_found.
    p = _provider(tmp_path, monkeypatch)
    out = json.loads(p._handle_invalidate(
        {"memory_id": "abc123", "replacement_id": "abc123"}))
    assert "replacement_id must differ" in out["error"]
    assert p._beam.invalidated == []  # refused before any beam call


def test_invalidate_reports_missing_replacement(tmp_path, monkeypatch):
    # Parity with the root provider: when the target is visible but the
    # invalidation still fails, name the replacement id as the bad input —
    # never a bare memory_not_found that sends the caller hunting for the
    # wrong id. The fallback path only counts a row as active when it
    # POSITIVELY declares its status fields (see the real-get-shape
    # regression below), so this stand-in row declares both unset.
    p = _provider(tmp_path, monkeypatch)
    p._beam.invalidate_result = False
    p._beam.rows["abc123"] = {"id": "abc123", "superseded_by": None, "valid_until": None}

    out = json.loads(p._handle_invalidate(
        {"memory_id": "abc123", "replacement_id": "gone"}))
    assert out["status"] == "replacement_not_found"
    assert out["replacement_id"] == "gone"
    assert out["memory_id"] == "abc123"

    # Target itself invisible: still memory_not_found.
    out = json.loads(p._handle_invalidate(
        {"memory_id": "absent", "replacement_id": "gone"}))
    assert out["status"] == "memory_not_found"


def test_invalidate_inactive_target_stays_memory_not_found(tmp_path, monkeypatch):
    # get() returns superseded and expired rows too, so a bare existence
    # check would let an already-dead target wrongly blame a healthy
    # replacement (review on #1113). Only an ACTIVE target makes the
    # replacement the suspect. The stand-in beam carries no live connection,
    # so the active-state helper falls back to the metadata get() reports.
    p = _provider(tmp_path, monkeypatch)
    p._beam.invalidate_result = False
    p._beam.rows["abc123"] = {"id": "abc123", "superseded_by": "earlier"}
    p._beam.rows["gone"] = {"id": "gone"}

    out = json.loads(p._handle_invalidate(
        {"memory_id": "abc123", "replacement_id": "gone"}))
    assert out["status"] == "memory_not_found"


def test_invalidate_active_helper_uses_sql_when_beam_has_connection(tmp_path, monkeypatch):
    # The metadata fallback above is only half the helper. With a live
    # connection the decision goes through core's own active-row predicate
    # (review on #1113, second round): an EXPIRED target (valid_until in the
    # past) must report the target even though get() still sees the row,
    # while an ACTIVE target keeps letting the replacement take the blame.
    p = _provider(tmp_path, monkeypatch)
    p._beam.invalidate_result = False
    conn = sqlite3.connect(str(tmp_path / "beam.db"))
    for table in ("working_memory", "episodic_memory"):
        conn.execute(
            f"CREATE TABLE {table} (id TEXT, session_id TEXT, scope TEXT,"
            " superseded_by TEXT, valid_until TEXT)"
        )
    conn.execute(
        "INSERT INTO working_memory VALUES ('dead', 'sess', 'global', NULL, ?)",
        ("2000-01-01T00:00:00",),
    )
    conn.execute(
        "INSERT INTO working_memory VALUES ('alive', 'sess', 'global', NULL, NULL)"
    )
    conn.commit()
    p._beam.conn = conn
    # Metadata says both rows are alive; only the SQL predicate knows 'dead'
    # expired. If the test ever silently falls back, this is the line that
    # would flip the expired case and fail the assertion below.
    p._beam.rows.update({"dead": {"id": "dead"}, "alive": {"id": "alive"}})

    out = json.loads(p._handle_invalidate(
        {"memory_id": "dead", "replacement_id": "alive"}))
    assert out["status"] == "memory_not_found"

    out = json.loads(p._handle_invalidate(
        {"memory_id": "alive", "replacement_id": "gone"}))
    assert out["status"] == "replacement_not_found"
    conn.close()


def test_invalidate_failed_activity_probe_with_real_get_shape_falls_closed(tmp_path, monkeypatch):
    # Review on #1113 (reproduced by the maintainer): with a REAL expired
    # target, a healthy replacement, and a failing activity SELECT, the
    # degraded fallback consulted `beam.get()` — whose real row shape does
    # NOT include superseded_by/valid_until at all. Absent fields used to
    # read as "active", so the answer blamed the healthy replacement
    # (replacement_not_found) instead of the target. A get() row that does
    # not positively declare its status means UNKNOWN, and unknown fails
    # closed to memory_not_found: the replacement is only blamed when the
    # target's activity is established.
    from mnemosyne.core.beam import BeamMemory

    class _ProbeDownBeam:
        """Same store, but the helper's activity query cannot run; get()
        stays the real BeamMemory method, so the row shape is real."""
        def __init__(self, real):
            self._real = real

        @property
        def conn(self):
            raise sqlite3.OperationalError("activity probe unavailable")

        def get(self, memory_id):
            return self._real.get(memory_id)

    p = _provider(tmp_path, monkeypatch)
    monkeypatch.setattr(
        mnemosyne_hermes, "_get_beam_class", lambda: BeamMemory)
    real_beam = BeamMemory(session_id="probe-down", db_path=tmp_path / "beam.db")
    p._beam = real_beam
    target = real_beam.remember("expired target row", source="fact")
    replacement = real_beam.remember("healthy replacement row", source="fact")
    past = (datetime.now() - timedelta(days=1)).isoformat()
    real_beam.conn.execute(
        "UPDATE working_memory SET valid_until = ? WHERE id = ?",
        (past, target),
    )
    real_beam.conn.commit()

    orig = p._invalidate_target_active
    monkeypatch.setattr(
        p, "_invalidate_target_active",
        lambda beam, memory_id: orig(_ProbeDownBeam(beam), memory_id))

    out = json.loads(p._handle_invalidate({
        "memory_id": target, "replacement_id": replacement,
    }))
    assert out["status"] == "memory_not_found", (
        "an unknown target state must never blame the replacement")
    row = real_beam.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?",
        (target,),
    ).fetchone()
    assert row[1] is None  # no store mutation
    real_beam.conn.close()


def test_activity_probe_judges_offset_bearing_expiry(tmp_path, monkeypatch):
    # Review on #1113 (fifth round): the probe mirrors core's active-row
    # predicate and must read stored offset-bearing expiries the way every
    # julianday-based surface does. A legacy/imported row carrying
    # ...T18:30:00+07:00 (11:30Z, already past) sorts lexically AFTER an
    # aware-UTC now string; the old text comparison called that dead row
    # active and let a failed invalidation wrongly blame a healthy
    # replacement. Same hand-built-connection pattern as the SQL-predicate
    # test above; the probe is exercised directly because the end-to-end
    # status also depends on core's own predicate (fixed in the companion
    # UTC/expiry PR).
    from datetime import timezone

    p = _provider(tmp_path, monkeypatch)
    conn = sqlite3.connect(str(tmp_path / "beam.db"))
    for table in ("working_memory", "episodic_memory"):
        conn.execute(
            f"CREATE TABLE {table} (id TEXT, session_id TEXT, scope TEXT,"
            " superseded_by TEXT, valid_until TEXT)"
        )
    now = datetime.now(timezone.utc)
    dead = (now - timedelta(minutes=30)).astimezone(
        timezone(timedelta(hours=7))).isoformat()
    alive = (now + timedelta(hours=1)).astimezone(
        timezone(timedelta(hours=-5))).isoformat()
    # Fixture sanity: both values genuinely mislead a lexical comparison,
    # in opposite directions.
    assert dead > now.isoformat()
    assert alive < now.isoformat()
    conn.execute(
        "INSERT INTO working_memory VALUES ('dead', 'sess', 'global', NULL, ?)",
        (dead,),
    )
    conn.execute(
        "INSERT INTO working_memory VALUES ('alive', 'sess', 'global', NULL, ?)",
        (alive,),
    )
    conn.commit()
    p._beam.conn = conn

    assert p._invalidate_target_active(p._beam, "dead") is False, (
        "an expiry already past in UTC must not read as active")
    assert p._invalidate_target_active(p._beam, "alive") is True, (
        "a chronologically future expiry must read as active")
    conn.close()


def test_invalidate_parity_on_west_of_utc_host(tmp_path, monkeypatch):
    # Review on #1113 (fourth round): the activity check in this provider
    # and core's replacement-path now must compare valid_until against UTC,
    # not host-local wall time. On a west-of-UTC host the naive-local clock
    # trails UTC stamps by the host offset, so an expiry inside that window
    # read as active: an expired target wrongly blamed a healthy
    # replacement. Pin the whole contract under TZ=America/Los_Angeles.
    import os
    import time
    from datetime import timezone

    from mnemosyne.core.beam import BeamMemory

    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset() unavailable on this platform")
    original_tz = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    try:
        p = _provider(tmp_path, monkeypatch)
        monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: BeamMemory)
        real_beam = BeamMemory(session_id="west-skew", db_path=tmp_path / "beam.db")
        p._beam = real_beam

        target = real_beam.remember("expired target west skew", source="fact")
        replacement = real_beam.remember("healthy replacement west skew", source="fact")
        # Expired ~1h ago in UTC terms; yesterday on the local wall clock —
        # inside the skew window a local-naive comparison calls "active".
        recent_past = (
            datetime.now(timezone.utc) - timedelta(hours=1)
        ).replace(tzinfo=None).isoformat()
        real_beam.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?",
            (recent_past, target),
        )
        real_beam.conn.commit()

        out = json.loads(p._handle_invalidate({
            "memory_id": target, "replacement_id": replacement,
        }))
        assert out["status"] == "memory_not_found", (
            "an expired target must not be read as active on a "
            "west-of-UTC host and blame the replacement")

        # Happy path on the same clock: active target + active replacement
        # must invalidate, and core must stamp the expiry aware-UTC (the
        # no-replacement path's shape).
        fresh_a = real_beam.remember("active target west skew", source="fact")
        fresh_b = real_beam.remember("active replacement west skew", source="fact")
        out = json.loads(p._handle_invalidate({
            "memory_id": fresh_a, "replacement_id": fresh_b,
        }))
        assert out["status"] == "invalidated"
        row = real_beam.conn.execute(
            "SELECT valid_until FROM working_memory WHERE id = ?", (fresh_a,),
        ).fetchone()
        assert datetime.fromisoformat(row[0]).utcoffset() == timedelta(0)
        real_beam.conn.close()
    finally:
        if original_tz is None:
            monkeypatch.delenv("TZ", raising=False)
        else:
            monkeypatch.setenv("TZ", original_tz)
        time.tzset()
