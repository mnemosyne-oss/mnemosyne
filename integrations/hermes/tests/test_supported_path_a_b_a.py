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

import mnemosyne_hermes
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

    for home, want_writer in ((home_a, "profile_a"), (home_b, "profile_b")):
        db = _store(home)
        assert db.exists()
        conn = sqlite3.connect(str(db))
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(canonical_facts)")}
            assert "writer_id" in cols and "writer_home" in cols, (
                "canonical rows carry no writer-provenance columns"
            )
            row = conn.execute(
                "SELECT writer_id, writer_home FROM canonical_facts "
                "WHERE category='regression' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        writer_id, writer_home = row
        assert writer_id == want_writer, (
            f"{home.name}: row written under a {want_writer} turn carries "
            f"writer_id={writer_id!r} — provenance would misattribute the write"
        )
        assert writer_home and str(writer_home) == str(home), (
            f"{home.name}: writer_home {writer_home!r} does not name the "
            "writing turn's home"
        )
