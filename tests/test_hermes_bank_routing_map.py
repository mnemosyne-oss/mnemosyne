"""Tests for per-chat-partner bank routing.

One Hermes instance can talk to several people over different chats. The
upstream adapter binds one bank per profile; this module resolves the bank per
conversation instead, from the authenticated session key, never from message
content.

Fail-closed contract:
  * longest matching prefix in the map wins
  * a chat key with no match resolves to ``_unknown`` (quarantine), never to a
    private bank
  * a run without chat identity (cron, background, CLI) resolves to ``_default``
"""

import json
import sys
from pathlib import Path

import pytest


def _load_bank_router():
    root = Path(__file__).resolve().parents[1] / "integrations" / "hermes" / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from mnemosyne_hermes import bank_router  # noqa: WPS433

    return bank_router


@pytest.fixture()
def router(tmp_path, monkeypatch):
    """Point the router at a throwaway data dir with a known map."""
    banks = tmp_path / "banks"
    banks.mkdir()
    mapping = {
        "_default": "vela",
        "_unknown": "quarantine",
        "agent:main:telegram:dm:111": "alice",
        "agent:main:telegram:dm:222":
            "bob",
        "agent:main:telegram:group:": "shared",
    }
    (tmp_path / "bank_map.json").write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_BANKS_MAP", str(tmp_path / "bank_map.json"))
    mod = _load_bank_router()
    # the module caches by mtime; clear it so each test sees its own map
    mod._cache.clear()
    return mod


def test_known_dm_resolves_to_owner_bank(router):
    assert router.resolve_bank("agent:main:telegram:dm:111") == (
        "alice",
        "map:agent:main:telegram:dm:111",
    )


def test_second_person_gets_their_own_bank(router):
    assert router.resolve_bank("agent:main:telegram:dm:222")[0] == "bob"


def test_group_resolves_to_shared_bank(router):
    assert router.resolve_bank("agent:main:telegram:group:-100")[0] == "shared"


def test_unknown_chat_is_quarantined_not_private(router):
    bank, reason = router.resolve_bank("agent:main:telegram:dm:999")
    assert bank == "quarantine"
    assert reason == "unknown:fail-closed"


def test_run_without_chat_key_uses_default(router):
    assert router.resolve_bank("cron:nightly")[0] == "vela"
    assert router.resolve_bank("")[0] == "vela"


def test_longest_prefix_wins(router, tmp_path, monkeypatch):
    mapping = json.loads((tmp_path / "bank_map.json").read_text())
    mapping["agent:main:telegram:dm:222:thread"] = "carol"
    (tmp_path / "bank_map.json").write_text(json.dumps(mapping))
    mod = _load_bank_router()
    mod._cache.clear()
    assert mod.resolve_bank("agent:main:telegram:dm:222:thread:7")[0] == "carol"
    assert mod.resolve_bank("agent:main:telegram:dm:222")[0] == "bob"


def test_map_is_data_not_code(router, tmp_path, monkeypatch):
    """A new person is an entry, not a code change."""
    mapping = json.loads((tmp_path / "bank_map.json").read_text())
    mapping["agent:main:telegram:dm:333"] = "dave"
    (tmp_path / "bank_map.json").write_text(json.dumps(mapping))
    mod = _load_bank_router()
    mod._cache.clear()
    assert mod.resolve_bank("agent:main:telegram:dm:333")[0] == "dave"


def test_bank_db_path_stays_inside_data_dir(router):
    path = router.bank_db_path("alice")
    assert path.name == "mnemosyne.db"
    assert path.parent.name == "alice"
    assert path.parent.parent.name == "banks"


def test_missing_map_disables_routing(tmp_path, monkeypatch):
    """No map file means the feature is off, not "quarantine everybody".

    Earlier revisions resolved an unmapped chat to ``quarantine`` even when no
    map existed, so a deployment that never enabled per-chat routing silently
    began writing into a fresh quarantine database. Routing is a switch:
    without entries the provider keeps its existing bank and creates nothing
    (review P1-2).
    """
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_BANKS_MAP", str(tmp_path / "does-not-exist.json"))
    mod = _load_bank_router()
    mod._cache.clear()
    bank, reason = mod.resolve_bank("agent:main:telegram:dm:111")
    assert bank == ""
    assert reason == "disabled:no-map"
