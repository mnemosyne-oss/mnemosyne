"""Reproductions of the second review round (independent review, 31.10.2026).

The first two rounds are addressed in ``983c7d8``/``d221140`` and
``46c23bc``/``39967f2``/``72d9e4b``. This module pins the findings that
survived the second round as **tests**: each one fails on the reviewed head
(``72d9e4b``), and a fix counts only when the whole module is green.

  R1 (P1)  a blank mapped bank must not fall back to the private profile
  R2 (P1)  the route must be resolved under the binding lock
  R3 (P2)  the shutdown must release the bank connections, not only the cache
  R4 (P2)  an explicitly configured channel must survive a session switch
  R5 (P2)  a documented rollback must retarget the lifecycle work
  R6 (P1)  an unreadable map must not be read as "routing is off"
  R7 (P2)  a routed Beam must carry the canonical owner identity

R1 and R6 are the same defect class: a *failure* was translated into
"the feature is off", and "off" means the private profile storage. R2, R4
and R5 are the same defect class: the routing decision and its application
drift apart. R3 is a resource-ownership gap.

The reproductions use synthetic temporary databases with the host LLM
disabled; no production data is read or written.
"""

import json
import sqlite3
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INTEGRATIONS_SRC = REPO_ROOT / "integrations" / "hermes" / "src"

ALICE_KEY = "agent:main:telegram:dm:111"
BOB_KEY = "agent:main:telegram:dm:222"
ALICE_THREAD_KEY = "agent:main:telegram:dm:111:thread:7"


def _load_bank_router():
    if str(INTEGRATIONS_SRC) not in sys.path:
        sys.path.insert(0, str(INTEGRATIONS_SRC))
    from mnemosyne_hermes import bank_router  # noqa: WPS433

    return bank_router


def _write_map(monkeypatch, data_dir: Path, mapping: dict) -> Path:
    """Write the map and pin the env to it.

    An ambient ``MNEMOSYNE_BANKS_MAP`` (the production deployment exports one)
    would otherwise decide the routing of a unit test.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "banks_map.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_BANKS_MAP", str(path))
    return path


def _forget_cached_map(router) -> None:
    router._cache.clear()
    router._meta_cache.clear()


def _provider(
    tmp_path,
    monkeypatch,
    *,
    session_id,
    gateway_key,
    identity,
    agent_context="primary",
    channel_id="",
):
    """Build a provider whose banks live under ``tmp_path/data``."""
    from mnemosyne_hermes import MnemosyneMemoryProvider

    data = tmp_path / "data"
    (data / "private").mkdir(parents=True, exist_ok=True)
    hermes_home = tmp_path / "homes" / identity
    hermes_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    # The provider binds its beam per home; initialize() and the call-time
    # lookup must resolve the same home or the write guard rightly refuses.
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setenv("HOME", str(hermes_home))
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        session_id=session_id,
        hermes_home=str(hermes_home),
        agent_identity=identity,
        gateway_session_key=gateway_key,
        agent_context=agent_context,
        channel_id=channel_id,
    )
    return provider


def _call(provider, tool: str, args: dict | None = None) -> dict:
    return json.loads(provider.handle_tool_call(tool, args or {}))


def _remember(provider, content: str, scope: str = "global") -> dict:
    return _call(provider, "mnemosyne_remember", {"content": content, "scope": scope})


def _recall_text(provider, query: str) -> str:
    payload = _call(provider, "mnemosyne_recall", {"query": query})
    return " ".join(str(r.get("content", "")) for r in payload.get("results", []))


def _session_ids(db_path: Path) -> dict:
    """``{content: session_id}`` of the working memory in a bank database."""
    if not db_path.exists():
        return {}
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cols = [row[1] for row in con.execute("PRAGMA table_info(working_memory)")]
        session_col = next((c for c in ("session_id", "session") if c in cols), None)
        if session_col is None:
            return {}
        rows = con.execute(
            f"SELECT content, {session_col} FROM working_memory"  # noqa: S608
        ).fetchall()
        return {str(content): str(session) for row in rows for content, session in [row]}
    finally:
        con.close()


# --- R1 (P1): a blank mapped bank must not open the private profile ---------


def test_blank_mapped_bank_must_not_open_the_private_profile(tmp_path, monkeypatch):
    """A map entry without a bank name is a broken configuration.

    The router returned an empty bank for it. The provider reads an empty
    bank as "routing is off" and falls back to the private profile storage —
    so the person the entry names gets the profile owner's memories.
    """
    data = tmp_path / "data"
    secret = "Privatnotiz des Profilinhabers: Projektcode Vulkan"

    # (1) Feature off: the memory lands in the original profile storage.
    monkeypatch.delenv("MNEMOSYNE_BANKS_MAP", raising=False)
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=BOB_KEY, identity="bob",
    )
    assert _remember(provider, secret).get("status") == "stored"

    # (2) A map whose entry names no bank: the entry is broken, not "off".
    router = _load_bank_router()
    _write_map(monkeypatch, data, {"_default": "vela", "_unknown": "quarantine", BOB_KEY: ""})
    _forget_cached_map(router)
    broken = _provider(
        tmp_path, monkeypatch,
        session_id="sess-b", gateway_key=BOB_KEY, identity="bob",
    )

    assert "Vulkan" not in _recall_text(broken, "Vulkan"), (
        "a blank mapped bank fell back to the private profile storage"
    )
    written = _remember(broken, "Ein satz von Bob")
    assert written.get("status") == "stored", written
    profile_rows = _session_ids(
        tmp_path / "homes" / "bob" / "mnemosyne" / "data" / "mnemosyne.db"
    )
    assert "Ein satz von Bob" not in profile_rows, (
        "the write landed in the private profile storage"
    )


# --- R2 (P1): the route must be resolved under the binding lock ------------


def test_route_must_be_resolved_under_the_binding_lock(tmp_path, monkeypatch):
    """A concurrent session switch must not split bank and session.

    The route was resolved *before* the binding lock was taken. A gateway
    switch between the two steps left the provider with Alice's database and
    Bob's durable session — the exact mix the feature exists to prevent.

    The switch runs in a second thread and *attempts* to take effect while
    the route is being resolved. With the resolution inside the binding
    lock, the switch cannot complete before the binding: it waits for the
    scope to finish. Resolved outside the lock, it completes first and the
    provider ends up with Alice's database and Bob's durable session.
    """
    data = tmp_path / "data"
    _write_map(
        monkeypatch, data,
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice", BOB_KEY: "bob"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )

    route_resolved = threading.Event()
    switch_attempted = threading.Event()
    recorded: list = []

    original_route = provider._route_session_key
    original_bind = provider._beam_for_bank

    def racing_route(session_id):
        """Resolve the key, then let a switch attempt to land."""
        key = original_route(session_id)
        route_resolved.set()
        switch_attempted.wait(5)
        return key

    def recording_bind(bank, reason=""):
        recorded.append({
            "bank": bank,
            "reason": reason,
            "gateway_key": getattr(provider, "_gateway_session_key", ""),
            "session_id": getattr(provider, "_session_id", ""),
        })
        return original_bind(bank, reason)

    provider._route_session_key = racing_route
    provider._beam_for_bank = recording_bind

    def _switch() -> None:
        route_resolved.wait(5)
        switch_attempted.set()
        provider.on_session_switch("sess-b", gateway_session_key=BOB_KEY)

    worker = threading.Thread(target=_switch, daemon=True)
    worker.start()
    with provider._beam_session_scope("") as beam:
        assert beam is not None
    worker.join(10)

    assert recorded, "the bank was never bound"
    binding = recorded[0]
    assert binding["gateway_key"] == ALICE_KEY, (
        "the bank was bound under a foreign session key: "
        f"{binding['gateway_key']!r} != {ALICE_KEY!r}"
    )
    assert ALICE_KEY in binding["session_id"], (
        "the bound bank carries a foreign durable session: "
        f"{binding['session_id']!r}"
    )


def test_route_resolution_must_hold_the_binding_lock(tmp_path, monkeypatch):
    """The route decision itself belongs inside the binding lock.

    This is the structural half of R2: a parallel turn must not be able to
    observe the provider between "route decided" and "bank bound".
    """
    data = tmp_path / "data"
    _write_map(
        monkeypatch, data,
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )

    lock = provider._ensure_beam_access_lock()
    observed: list = []
    original_route = provider._route_session_key

    def probing_route(session_id):
        observed.append(lock._is_owned())
        return original_route(session_id)

    provider._route_session_key = probing_route
    with provider._beam_session_scope("") as beam:
        assert beam is not None

    assert observed == [True], (
        "the route was resolved outside the binding lock: "
        f"lock held = {observed}"
    )


# --- R3 (P2): the shutdown must release the bank connections ---------------


def test_shutdown_must_release_the_bank_connections(tmp_path, monkeypatch):
    """A cached bank holds a live SQLite handle — the shutdown must close it.

    Clearing the cache dictionary is not a release: the connection objects
    stay open and keep answering SQL after the provider is gone.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice", BOB_KEY: "bob"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    assert _remember(provider, "Ein satz fuer Alice").get("status") == "stored"
    provider.on_session_switch("sess-b", gateway_session_key=BOB_KEY)
    assert _remember(provider, "Ein satz fuer Bob").get("status") == "stored"

    cached = getattr(provider, "_bank_cache", None)
    assert cached and len(cached) >= 2, f"expected two cached banks, got {cached}"

    handles = []
    for _beam, mem in list(cached.values()):
        for handle in (getattr(mem, "conn", None), getattr(getattr(mem, "beam", None), "conn", None)):
            if handle is not None and not any(handle is h for h in handles):
                handles.append(handle)
    assert handles, "the cached banks hold no connection to release"

    provider.shutdown()

    still_alive = []
    for handle in handles:
        try:
            handle.execute("SELECT 1")
        except sqlite3.ProgrammingError:
            continue
        still_alive.append(handle)
    assert not still_alive, (
        f"{len(still_alive)} bank connections still answer SQL after the shutdown"
    )


# --- R4 (P2): an explicit channel must survive a session switch ------------


def test_explicit_channel_must_survive_a_session_switch(tmp_path, monkeypatch):
    """An explicitly configured channel is a decision, not a coincidence.

    The rebind inferred "implicit" from the *value* of the channel: a
    channel that happened to equal the session id was rewritten to the new
    session id on the next switch. The provider already knows whether the
    channel was explicit — that is the flag to read.
    """
    data = tmp_path / "data"
    _write_map(
        monkeypatch, data,
        {
            "_default": "vela",
            "_unknown": "quarantine",
            ALICE_KEY: "alice",
            ALICE_THREAD_KEY: "alice",
        },
    )
    # The explicit channel equals the session id — exactly the collision the
    # value-based inference cannot tell apart from an implicit channel.
    provider = _provider(
        tmp_path, monkeypatch,
        session_id=ALICE_KEY, gateway_key=ALICE_KEY, identity="alice",
        channel_id="hermes_" + ALICE_KEY,
    )
    assert _remember(provider, "Ein satz fuer Alice").get("status") == "stored"

    provider.on_session_switch("sess-b", gateway_session_key=ALICE_THREAD_KEY)
    assert _remember(provider, "Ein zweiter satz").get("status") == "stored"

    cached = getattr(provider, "_bank_cache", None)
    assert cached, "the bank was not cached"
    channels = set()
    for _beam, mem in cached.values():
        for holder in (mem, getattr(mem, "beam", None)):
            channel = getattr(holder, "channel_id", None)
            if channel:
                channels.add(channel)
    assert channels == {"hermes_" + ALICE_KEY}, (
        f"the explicitly configured channel was overwritten: {channels}"
    )


# --- R5 (P2): a documented rollback must retarget the lifecycle work -------


def test_rollback_must_retarget_the_session_end_consolidation(tmp_path, monkeypatch):
    """After the documented rollback the lifecycle must follow the writes.

    "Turning it off" is a documented operation: delete the map, and the next
    turn is written into the profile storage again. The consolidation still
    selected the bank from before the rollback — a stale target that outlived
    the decision that produced it.
    """
    import mnemosyne_hermes as provider_module

    data = tmp_path / "data"
    map_file = _write_map(
        monkeypatch, data,
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    assert _remember(provider, "Ein satz fuer Alice").get("status") == "stored"

    captured: dict = {}

    class _FakeBeam:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.canonical_owner_id = "default"
            self.agent_context = "primary"

        def sleep(self):
            return None

    monkeypatch.setattr(provider_module, "_get_beam_class", lambda: _FakeBeam)
    monkeypatch.setattr(provider, "_reserve_reflection_budget_locked", lambda *a, **k: None)

    # The documented rollback: remove the map.
    map_file.unlink()
    _forget_cached_map(_load_bank_router())

    assert _remember(provider, "Ein satz nach dem Rollback").get("status") == "stored"

    provider.on_session_end([{"role": "user", "content": "bis gleich"}])

    assert captured.get("db_path"), "the session-end sleep never ran"
    assert "banks" not in str(captured["db_path"]), (
        "the consolidation still targets the bank from before the rollback: "
        f"{captured['db_path']}"
    )


# --- R6 (P1): an unreadable map must not be read as "routing is off" -------


def test_unreadable_map_must_not_be_read_as_routing_off(tmp_path, monkeypatch):
    """A map that cannot even be inspected is a failure, not an absence.

    ``stat()`` failing for any reason other than "the file is not there" was
    reported as "no map" — which switches the routing off and exposes the
    private profile storage.
    """
    router = _load_bank_router()
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    path = data / "banks_map.json"
    path.write_text("{}", encoding="utf-8")
    # Pin the map to this test's file: the ambient MNEMOSYNE_BANKS_MAP of a
    # production deployment would otherwise decide what is read here.
    monkeypatch.setenv("MNEMOSYNE_BANKS_MAP", str(path))
    monkeypatch.delenv("MNEMOSYNE_DATA_DIR", raising=False)

    real_stat = Path.stat

    def exploding_stat(self, *args, **kwargs):
        if self == path:
            raise PermissionError("stat denied")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", exploding_stat)
    _forget_cached_map(router)

    status, entries = router.load_state(str(tmp_path / "home"))
    assert status != "absent", (
        "a map that cannot be inspected was reported as absent — "
        "that switches the routing off"
    )
    assert entries == {}


# --- R7 (P2): a routed Beam must carry the canonical owner identity -------


def test_routed_bank_must_carry_the_canonical_owner(tmp_path, monkeypatch):
    """A routed Beam is a first-class Beam.

    The profile Beam gets its canonical owner and agent context at init. A
    routed Beam was created without them, so the lifecycle work fell back to
    the default owner and the model-refresh writes landed in the wrong
    namespace.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    assert _remember(provider, "Ein satz fuer Alice").get("status") == "stored"

    cached = getattr(provider, "_bank_cache", None)
    assert cached, "the bank was not cached"
    expected_owner = provider._canonical_owner()
    for _beam, mem in cached.values():
        beam = getattr(mem, "beam", None)
        owner = getattr(beam, "canonical_owner_id", None)
        assert owner == expected_owner, (
            f"the routed Beam carries owner {owner!r}, expected {expected_owner!r}"
        )
        assert getattr(beam, "agent_context", None) == provider._agent_context, (
            "the routed Beam carries no agent context"
        )
