"""Supported-path A→B→A regression (#1050 review, 2026-09-26).

The harness tests in ``test_calltime_bindings.py`` drive the call-time
binding with a recording beam stand-in. This test removes that shortcut and
exercises the SUPPORTED path end to end: the provider is obtained through
``register_memory_provider`` — the #1008 provider-discovery entry the real
gateway uses — while every write lands in a real, file-backed store under
the home that owns the turn.

Scenario (the 2026-09-20 multiplex incident): one process serves homes A and
B. Both initialize; then a turn arrives for A again — WITHOUT re-init — and
writes. Last-initialized-home-wins routing misfiles that write into B's
database while the audit says A. This test must fail against the pre-fix
provider for exactly that reason, and passes when routing resolves the
turn's home at call time.

The only fake is the host itself: ``hermes_constants`` provides the
per-turn home the multiplex host would supply (it is a host module and is
absent under bare pytest). Registration, initialization, dispatch, stores,
and rows are all the real thing.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import types

import pytest

from mnemosyne_hermes import register_memory_provider


# ---------------------------------------------------------------------------
# Fake host: per-turn home, as the multiplex gateway would supply it
# ---------------------------------------------------------------------------

_TURN = {"home": None, "profile": "profile_a"}


@pytest.fixture()
def fake_host(monkeypatch):
    consts = types.ModuleType("hermes_constants")
    consts.get_hermes_home = lambda: _TURN["home"]
    consts.hermes_home_key = lambda home=None: str(home if home is not None else _TURN["home"])
    cli = types.ModuleType("hermes_cli")
    profiles = types.ModuleType("hermes_cli.profiles")
    profiles.get_active_profile_name = lambda: _TURN["profile"]
    cli.profiles = profiles
    monkeypatch.setitem(sys.modules, "hermes_constants", consts)
    monkeypatch.setitem(sys.modules, "hermes_cli", cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", profiles)
    _TURN["home"] = None
    _TURN["profile"] = "profile_a"
    yield _TURN
    _TURN["home"] = None
    _TURN["profile"] = "profile_a"


class _RegisteringCtx:
    """Records what the supported discovery entry registers, like the host does."""

    def __init__(self):
        self.providers = []

    def register_memory_provider(self, provider):
        self.providers.append(provider)


def _seed_home(tmp_path, name):  # noqa: D401 — layout mirrors what the installer produces
    home = tmp_path / name
    (home / ".hermes").mkdir(parents=True)
    return home


def _store(home):
    """The private store the provider builds for a home (profile banks are
    config-gated and default off — the default path is data/mnemosyne.db)."""
    return home / "mnemosyne" / "data" / "mnemosyne.db"


def _canonical_rows(home):
    db = _store(home)
    if not db.exists():
        return {}
    conn = sqlite3.connect(str(db))
    try:
        return {
            (cat, nm): body
            for cat, nm, body in conn.execute(
                "SELECT category, name, body FROM canonical_facts"
            )
        }
    finally:
        conn.close()


def _write_canonical(provider, name, body):
    resp = json.loads(
        provider.handle_tool_call(
            "mnemosyne_remember_canonical",
            {"category": "regression", "name": name, "body": body},
        )
    )
    assert resp.get("status") in ("created", "unchanged"), resp
    return resp


def test_a_b_a_turns_route_every_write_to_the_turn_home(tmp_path, fake_host):
    home_a = _seed_home(tmp_path, "turn-home-a")
    home_b = _seed_home(tmp_path, "turn-home-b")

    # --- supported registration entry, exactly as provider discovery calls it
    ctx = _RegisteringCtx()
    provider = register_memory_provider(ctx)
    assert ctx.providers == [provider], "provider never reached the host's ctx"

    # --- home A initializes first (turn A)
    fake_host["home"] = str(home_a)
    fake_host["profile"] = "profile_a"
    provider.initialize(
        "sess-a", hermes_home=str(home_a), agent_context="primary",
        agent_identity="profile_a",
    )
    assert provider._beam is not None
    _write_canonical(provider, "a1", "written by the first A turn")

    # --- home B initializes second; under the incident code B wins everything
    fake_host["home"] = str(home_b)
    fake_host["profile"] = "profile_b"
    provider.initialize(
        "sess-b", hermes_home=str(home_b), agent_context="primary",
        agent_identity="profile_b",
    )
    assert provider._beam is not None
    _write_canonical(provider, "b1", "written by the B turn")

    # --- THE A→B→A STEP: a turn arrives for A with no re-init. Pre-fix, this
    # write lands in B's database (last-init-wins ambient slot).
    fake_host["home"] = str(home_a)
    fake_host["profile"] = "profile_a"
    _write_canonical(provider, "a2", "written by the RETURNING A turn")

    rows_a = _canonical_rows(home_a)
    rows_b = _canonical_rows(home_b)

    assert ("regression", "a1") in rows_a, "A lost its first-turn row"
    assert ("regression", "a2") in rows_a, (
        "A→B→A misfile: the returning-A write never reached A's own store "
        "(call-time home routing is broken)"
    )
    assert ("regression", "a2") not in rows_b, (
        "A→B→A misfile: the returning-A write landed in B's database"
    )
    assert ("regression", "b1") in rows_b, "B lost its own row"
    assert ("regression", "b1") not in rows_a, "B's write leaked into A's store"


def test_canonical_writes_carry_writer_provenance_at_rest(tmp_path, fake_host):
    home_a = _seed_home(tmp_path, "turn-home-a")
    home_b = _seed_home(tmp_path, "turn-home-b")

    ctx = _RegisteringCtx()
    provider = register_memory_provider(ctx)

    fake_host["home"] = str(home_a)
    fake_host["profile"] = "profile_a"
    provider.initialize(
        "sess-a", hermes_home=str(home_a), agent_context="primary",
        agent_identity="profile_a",
    )
    _write_canonical(provider, "a1", "stamp check A")

    fake_host["home"] = str(home_b)
    fake_host["profile"] = "profile_b"
    provider.initialize(
        "sess-b", hermes_home=str(home_b), agent_context="primary",
        agent_identity="profile_b",
    )
    _write_canonical(provider, "b1", "stamp check B")

    # THE A→B→A STEP, provenance arm (CodeRabbit on 9ae0531): placement of
    # a2 (test above) does not pin its STAMP — a returning-A write could
    # land in A's store wearing B's provenance while both original arms
    # stayed green. Each row is now selected by name, not "latest row".
    fake_host["home"] = str(home_a)
    fake_host["profile"] = "profile_a"
    _write_canonical(provider, "a2", "stamp check returning A")

    expect = {
        home_a: (("a1", "profile_a"), ("a2", "profile_a")),
        home_b: (("b1", "profile_b"),),
    }
    for home, rows in expect.items():
        db = _store(home)
        assert db.exists()
        conn = sqlite3.connect(str(db))
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(canonical_facts)")}
            assert "writer_id" in cols and "writer_home" in cols, (
                "canonical rows carry no writer-provenance columns"
            )
            for name, want_writer in rows:
                row = conn.execute(
                    "SELECT writer_id, writer_home FROM canonical_facts "
                    "WHERE category='regression' AND name=? "
                    "ORDER BY created_at DESC LIMIT 1",
                    (name,),
                ).fetchone()
                assert row is not None, f"{home.name}: {name!r} never landed"
                writer_id, writer_home = row
                assert writer_id == want_writer, (
                    f"{home.name}: row {name!r} written under a {want_writer} "
                    f"turn carries writer_id={writer_id!r} — provenance would "
                    "misattribute the write"
                )
                assert writer_home and str(writer_home) == str(home), (
                    f"{home.name}: writer_home {writer_home!r} does not name "
                    "the writing turn's home"
                )
        finally:
            conn.close()


def test_initialize_reads_and_writes_share_the_new_home_slot(tmp_path, fake_host):
    """#1050 review point 2: mid-init, getters must not read the PREVIOUS
    home's ambient slot while setters target the home under construction.
    Pre-fix, B's bank path inherited A's identity through that skew."""
    home_a = _seed_home(tmp_path, "align-home-a")
    home_b = _seed_home(tmp_path, "align-home-b")

    ctx = _RegisteringCtx()
    provider = register_memory_provider(ctx)

    fake_host["home"] = str(home_a)
    fake_host["profile"] = "profile_a"
    provider.initialize(
        "sess-a", hermes_home=str(home_a), agent_context="primary",
        agent_identity="profile_a",
    )

    # Initialize B WHILE A's turn scope is still current: the exact skew the
    # reviewer probed. Every B binding write must resolve to B's slot.
    fake_host["home"] = str(home_a)
    fake_host["profile"] = "profile_a"
    provider.initialize(
        "sess-b", hermes_home=str(home_b), agent_context="primary",
        agent_identity="profile_b",
    )

    bindings = provider.__dict__["_bindings"]
    b_key = str(home_b)
    assert b_key in bindings, "B's slot was never created — writes went elsewhere"
    assert bindings[b_key]["agent_identity"] == "profile_b", (
        "B's identity landed outside B's slot (read/write skew during init)"
    )
    # Under B's turn, the getter resolves to the very slot the init wrote.
    # Pre-fix this read A's slot entirely: B's bank path inherited A's
    # session/identity through the getter/setter skew.
    fake_host["home"] = str(home_b)
    assert provider._session_id == "hermes_sess-b", (
        "B's reads did not resolve to B's freshly written slot"
    )
    assert bindings[b_key]["session_id"] == "hermes_sess-b"
    # A's slot keeps its own identity untouched by B's init.
    fake_host["home"] = str(home_a)
    assert provider._session_id == "hermes_sess-a"
    # The init routing key must not outlive the init that set it.
    assert provider.__dict__.get("_init_home") is None, (
        "_init_home survived initialization; later writes would be steered "
        "into the last-initialized home's slot"
    )


def test_session_switch_after_later_init_targets_the_turn_home(tmp_path, fake_host):
    """#1050 review point 2 residue probe: after A→B init, an A session
    switch must update A's binding, not the stored B slot."""
    home_a = _seed_home(tmp_path, "switch-home-a")
    home_b = _seed_home(tmp_path, "switch-home-b")

    ctx = _RegisteringCtx()
    provider = register_memory_provider(ctx)

    fake_host["home"] = str(home_a)
    provider.initialize(
        "sess-a", hermes_home=str(home_a), agent_context="primary",
        agent_identity="profile_a",
    )
    fake_host["home"] = str(home_b)
    provider.initialize(
        "sess-b", hermes_home=str(home_b), agent_context="primary",
        agent_identity="profile_b",
    )

    fake_host["home"] = str(home_a)
    provider.on_session_switch("sess-a2")

    bindings = provider.__dict__["_bindings"]
    assert bindings[str(home_a)]["session_id"].endswith("sess-a2"), (
        "the A switch never reached A's slot"
    )
    assert not bindings[str(home_b)]["session_id"].endswith("sess-a2"), (
        "residue steering: the A session switch updated B's stored slot"
    )
