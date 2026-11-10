"""Isolation controls for per-chat-partner bank routing (review #6506222609).

Each test here corresponds to one finding of the independent review:

  P1-1  read/write routing divergence  -> provider tests
  P1-2  routing runs while disabled    -> feature-off tests below
  P1-3  binding changed outside the lock -> provider tests
  P1-4  cache identity crosses homes   -> cache-key tests below
  P1-5  chat-ID prefix collision       -> boundary tests below
  P2    map fallback / path validation  -> _meta and containment tests below

The router-level tests are the cheap half; they fail on the reviewed head and
pass only when the boundary rules are explicit.
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
    banks = tmp_path / "banks"
    banks.mkdir()
    mapping = {
        "_default": "vela",
        "_unknown": "quarantine",
        "agent:main:telegram:dm:111": "alice",
        "agent:main:telegram:dm:222": "bob",
        "agent:main:telegram:group:": "shared",
    }
    (tmp_path / "banks_map.json").write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_BANKS_MAP", str(tmp_path / "banks_map.json"))
    mod = _load_bank_router()
    mod._cache.clear()
    return mod


# --- P1-2: routing must not run while the feature is off ---------------------


def test_feature_off_resolves_disabled_not_quarantine(tmp_path, monkeypatch):
    """No map at all: the provider keeps its existing bank.

    Writing into a fresh ``quarantine`` database is a routing effect; a
    feature that was never switched on must not create one.
    """
    (tmp_path / "banks").mkdir()
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("MNEMOSYNE_BANKS_MAP", raising=False)
    mod = _load_bank_router()
    mod._cache.clear()

    bank, reason = mod.resolve_bank("agent:main:telegram:dm:111")
    assert bank == "", f"feature off must not bind a bank, got {bank!r}"
    assert reason.startswith("disabled"), reason


def test_feature_off_with_missing_map_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("MNEMOSYNE_BANKS_MAP", str(tmp_path / "absent.json"))
    mod = _load_bank_router()
    mod._cache.clear()
    assert mod.resolve_bank("agent:main:telegram:dm:111")[0] == ""


def test_routing_enabled_matches_map_presence(router, tmp_path, monkeypatch):
    assert router.routing_enabled() is True
    monkeypatch.delenv("MNEMOSYNE_BANKS_MAP", raising=False)
    (tmp_path / "banks_map.json").unlink()
    router._cache.clear()
    assert router.routing_enabled() is False


# --- P1-5: prefix matching must respect component boundaries ----------------


def test_chat_id_prefix_collision_is_not_a_match(router):
    """``dm:111`` must not capture ``dm:1112``."""
    bank, reason = router.resolve_bank("agent:main:telegram:dm:1112")
    assert bank == "quarantine", f"dm:1112 leaked into {bank!r}"
    assert reason == "unknown:fail-closed"


def test_group_prefix_still_matches_on_boundary(router):
    assert router.resolve_bank("agent:main:telegram:group:-1004301699476")[0] == "shared"


def test_longest_component_prefix_wins(router, tmp_path):
    mapping = json.loads((tmp_path / "banks_map.json").read_text())
    mapping["agent:main:telegram:dm:222:thread"] = "carol"
    (tmp_path / "banks_map.json").write_text(json.dumps(mapping))
    router._cache.clear()
    assert router.resolve_bank("agent:main:telegram:dm:222:thread:7")[0] == "carol"
    assert router.resolve_bank("agent:main:telegram:dm:222")[0] == "bob"
    # a longer key that merely starts with the same characters is not a match
    assert router.resolve_bank("agent:main:telegram:dm:2220")[0] == "quarantine"


# --- P2: map fallback must use the caller's home ---------------------------


def test_meta_reads_the_explicit_hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    data = home / "mnemosyne" / "data"
    data.mkdir(parents=True)
    (data / "banks_map.json").write_text(
        json.dumps(
            {
                "_default": "carol",
                "_unknown": "quarantine",
                "agent:main:telegram:dm:1": "alice",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("MNEMOSYNE_BANKS_MAP", raising=False)
    monkeypatch.delenv("MNEMOSYNE_DATA_DIR", raising=False)
    mod = _load_bank_router()
    mod._cache.clear()

    # a run without chat identity must get the home's _default, not quarantine
    assert mod.resolve_bank("cron:nightly", hermes_home=str(home))[0] == "carol"


# --- P2: bank names are a configuration boundary ---------------------------


@pytest.mark.parametrize(
    "bad",
    ["../../outside", "..", "a/b", "a\\b", "", "   ", "/etc", "."],
)
def test_bank_name_outside_the_banks_root_is_rejected(router, bad):
    with pytest.raises(ValueError):
        router.bank_db_path(bad)


def test_valid_bank_name_stays_inside_root(router):
    path = router.bank_db_path("alice")
    assert path.parent.name == "alice"
    assert path.parent.parent.name == "banks"


# --- P1-4: cache identity is the storage, not the bank name ----------------


def test_cache_key_is_storage_identity(tmp_path, monkeypatch):
    """Two homes with the same bank name are two different databases."""
    a = tmp_path / "home-a"
    b = tmp_path / "home-b"
    monkeypatch.delenv("MNEMOSYNE_DATA_DIR", raising=False)
    monkeypatch.delenv("MNEMOSYNE_BANKS_MAP", raising=False)
    mod = _load_bank_router()
    key_a = mod.cache_key("samebank", hermes_home=str(a))
    key_b = mod.cache_key("samebank", hermes_home=str(b))
    assert key_a != key_b, "same bank name in two homes must not share a cache slot"
    assert key_a == mod.cache_key("samebank", hermes_home=str(a))


# --- P1-1 / P1-2: the provider paths, not just the router -------------------


def _provider(tmp_path, monkeypatch, *, session_id, gateway_key, identity, home=None):
    """Build a provider whose banks live under ``tmp_path/data``."""
    from mnemosyne_hermes import MnemosyneMemoryProvider

    data = tmp_path / "data"
    (data / "private").mkdir(parents=True, exist_ok=True)
    hermes_home = home or (tmp_path / "homes" / identity)
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
    )
    return provider


def _write_map(monkeypatch, data_dir: Path, mapping: dict) -> Path:
    """Write the map **and** pin the env to it.

    An ambient ``MNEMOSYNE_BANKS_MAP`` (the production deployment exports one)
    would otherwise decide the routing of a unit test.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "banks_map.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")
    monkeypatch.setenv("MNEMOSYNE_BANKS_MAP", str(path))
    return path


def _remember(provider, content, scope="global"):
    import json as _json

    return _json.loads(
        provider.handle_tool_call(
            "mnemosyne_remember", {"content": content, "scope": scope}
        )
    )


def test_two_chats_do_not_share_memories(tmp_path, monkeypatch):
    """Alice's fact must not come back to Bob — read paths included.

    The reviewed head routed ``sync_turn`` per bank but left recall, prefetch
    and tool dispatch on the original Beam, so a fact written into Alice's bank
    was returned to Bob (review P1-1).
    """
    _write_map(
        monkeypatch,
        tmp_path / "data",
        {
            "_default": "vela",
            "_unknown": "quarantine",
            "agent:main:telegram:dm:111": "alice",
            "agent:main:telegram:dm:222": "bob",
        },
    )
    alice = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key="agent:main:telegram:dm:111", identity="alice",
    )
    stored = _remember(alice, "Alice wohnt in Kiel und mag Hunde", scope="global")
    assert stored.get("status") == "stored", stored

    bob = _provider(
        tmp_path, monkeypatch,
        session_id="sess-b", gateway_key="agent:main:telegram:dm:222", identity="bob",
    )
    recalled = json.loads(bob.handle_tool_call("mnemosyne_recall", {"query": "wohnt Kiel"}))
    contents = " ".join(str(r.get("content", "")) for r in recalled.get("results", []))
    assert "Kiel" not in contents, f"Bob saw Alice's memory: {contents[:300]}"

    prefetched = bob.prefetch("wohnt Kiel")
    assert "Kiel" not in prefetched, f"Bob's prefetch leaked Alice: {prefetched[:300]}"


def test_feature_off_creates_no_bank_directory(tmp_path, monkeypatch):
    """Without a map nothing is rebound and no quarantine bank is created."""
    monkeypatch.delenv("MNEMOSYNE_BANKS_MAP", raising=False)
    alice = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key="agent:main:telegram:dm:111", identity="alice",
    )
    _remember(alice, "irgendein Satz ohne Bedeutung", scope="global")
    assert not (tmp_path / "data" / "banks" / "quarantine").exists(), (
        "routing ran while the feature was off"
    )


def test_cache_keys_are_per_home(tmp_path, monkeypatch):
    """Two homes with the same bank name must not share one cached instance."""
    from mnemosyne_hermes import bank_router

    monkeypatch.delenv("MNEMOSYNE_DATA_DIR", raising=False)
    monkeypatch.delenv("MNEMOSYNE_BANKS_MAP", raising=False)
    key_a = bank_router.cache_key("samebank", hermes_home=str(tmp_path / "home-a"))
    key_b = bank_router.cache_key("samebank", hermes_home=str(tmp_path / "home-b"))
    assert key_a != key_b


def test_binding_window_is_inside_the_lock(tmp_path, monkeypatch):
    """A parallel turn must not observe the binding being swapped (P1-3).

    The reviewed head replaced ``self._beam`` *before* the access lock was
    acquired. This test probes the lock from a second thread at the exact
    moment the bank is bound: if the binding sits outside the lock, the probe
    acquires it — and a parallel turn could have read the foreign bank.
    """
    import threading
    import time

    _write_map(
        monkeypatch,
        tmp_path / "data",
        {
            "_default": "vela",
            "_unknown": "quarantine",
            "agent:main:telegram:dm:111": "alice",
        },
    )
    alice = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key="agent:main:telegram:dm:111", identity="alice",
    )

    original = alice._beam_for_bank
    binding_started = threading.Event()
    probe: dict = {}

    def probing_bind(bank, reason=""):
        binding_started.set()
        time.sleep(0.25)  # window for the parallel probe
        return original(bank, reason)

    def _probe() -> None:
        binding_started.wait(2)
        lock = alice._ensure_beam_access_lock()
        got = lock.acquire(blocking=False)
        probe["acquired"] = got
        if got:
            lock.release()

    alice._beam_for_bank = probing_bind
    worker = threading.Thread(target=_probe, daemon=True)
    worker.start()
    with alice._beam_session_scope("") as beam:
        assert beam is not None
    worker.join(5)

    assert probe.get("acquired") is False, (
        "the lock was free while the bank was being bound — a parallel turn "
        "could have read the foreign bank"
    )
