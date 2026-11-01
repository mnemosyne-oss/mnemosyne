"""Reproductions of the follow-up findings (independent review, 30.10.2026).

The first review round was addressed in ``983c7d8``/``d221140``. This module
pins the five findings that survived that round as **tests**: each one fails on
the reviewed head, and a fix counts only when the whole module is green.

  F1 (P1)  a malformed map must not fall back to the private profile storage
  F2 (P1)  the cached persona adapter must follow the active session
  F3 (P1)  routing must not revive a skipped (subagent) context
  F4 (P2)  a cached bank must not keep the previous session id
  F5 (P2)  session-end consolidation must use the routed bank, not the profile

The reproductions use synthetic temporary databases with the host LLM
disabled; no production data is read or written.
"""

import json
import sqlite3
import sys
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
        return {str(content): str(session) for content, session in rows}
    finally:
        con.close()


# --- F1 (P1): a broken configuration must not expose the private profile ----


def test_malformed_map_must_not_fall_back_to_private_profile(tmp_path, monkeypatch):
    """A map that cannot be read is a failure, not a switch that is off.

    With a valid map Bob is routed into his own bank and the profile memory
    stays hidden. Replacing the map file with malformed JSON must not drop the
    provider back into the private profile storage — that would hand a private
    memory to whoever asks next.
    """
    router = _load_bank_router()
    data = tmp_path / "data"
    secret = "Privatnotiz des Profilinhabers: Projektcode Vulkan"

    # (1) Feature off: the memory lands in the original profile storage.
    monkeypatch.delenv("MNEMOSYNE_BANKS_MAP", raising=False)
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=BOB_KEY, identity="bob",
    )
    assert _remember(provider, secret).get("status") == "stored"

    # (2) Valid map: Bob is routed into his own bank, the profile stays hidden.
    _write_map(monkeypatch, data, {"_default": "vela", "_unknown": "quarantine", BOB_KEY: "bob"})
    _forget_cached_map(router)
    routed = _provider(
        tmp_path, monkeypatch,
        session_id="sess-b", gateway_key=BOB_KEY, identity="bob",
    )
    assert "Vulkan" not in _recall_text(routed, "Vulkan"), (
        "with a valid map the profile memory must stay hidden"
    )

    # (3) Broken map: the failure must fail closed, not fall back.
    map_file = data / "banks_map.json"
    map_file.write_text("{ kaputt: json,", encoding="utf-8")
    _forget_cached_map(router)
    broken = _provider(
        tmp_path, monkeypatch,
        session_id="sess-c", gateway_key=BOB_KEY, identity="bob",
    )
    assert "Vulkan" not in _recall_text(broken, "Vulkan"), (
        "a malformed map fell back to the private profile storage"
    )


# --- F2 (P1): the persona adapter must follow the active session ------------


def test_persona_adapter_must_follow_the_active_session(tmp_path, monkeypatch):
    """A cached persona adapter must not carry Alice's bank into Bob's chat.

    The adapter is built once from the Beam that is active at that moment and
    then kept on the provider. After a session switch it still points at the
    previous bank's database.
    """
    _write_map(
        monkeypatch, tmp_path / "data",
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice", BOB_KEY: "bob"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    stored = _remember(provider, "Alices Projekt heisst Vulkan")
    assert stored.get("status") == "stored", stored
    promoted = _call(
        provider, "mnemosyne_persona_promote",
        {"memory_id": stored["memory_id"], "tier": "long_term"},
    )
    assert promoted.get("status") == "ok", promoted
    listed = _call(provider, "mnemosyne_persona_list", {})
    assert any("Vulkan" in str(p.get("content", "")) for p in listed.get("personas", [])), listed

    # The very same provider now serves Bob's chat.
    provider.on_session_switch("sess-b", gateway_session_key=BOB_KEY)
    after = _call(provider, "mnemosyne_persona_list", {})
    assert not any("Vulkan" in str(p.get("content", "")) for p in after.get("personas", [])), (
        "Bob saw Alice's persona — the cached adapter still holds Alice's bank"
    )


# --- F3 (P1): routing must not revive a skipped context ---------------------


def test_routing_must_not_revive_a_skipped_context(tmp_path, monkeypatch):
    """A provider that was skipped must stay skipped.

    A subagent context has no Beam. Routing must not create a mapped bank for
    it and report a successful write; the call has to stay unavailable.
    """
    data = tmp_path / "data"
    _write_map(monkeypatch, data, {"_default": "vela", "_unknown": "quarantine", BOB_KEY: "bob"})
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=BOB_KEY, identity="bob",
        agent_context="subagent",
    )
    payload = _remember(provider, "Ich will nicht gespeichert werden")
    assert payload.get("status") == "memory_unavailable", payload
    assert payload.get("reason_code") == "skipped_context", payload
    assert not (data / "banks" / "bob").exists(), (
        "the skipped context created a bank"
    )


# --- F4 (P2): a cached bank must not keep the previous session id ----------


def test_cached_bank_must_not_keep_the_previous_session_id(tmp_path, monkeypatch):
    """Coming back to the same bank under a new thread must use the new session.

    The bank instance is cached per storage identity and carries the session id
    it was built with. After a session switch the write still lands under the
    session id of the first visit.
    """
    data = tmp_path / "data"
    _write_map(
        monkeypatch, data,
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice", ALICE_THREAD_KEY: "alice"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    assert _remember(provider, "Erster satz", scope="session").get("status") == "stored"

    provider.on_session_switch("sess-b", gateway_session_key=ALICE_THREAD_KEY)
    assert _remember(provider, "Zweiter satz", scope="session").get("status") == "stored"

    rows = _session_ids(data / "banks" / "alice" / "mnemosyne.db")
    assert "Erster satz" in rows and "Zweiter satz" in rows, rows
    assert rows["Zweiter satz"] != rows["Erster satz"], (
        "the second write kept the session id of the first visit: "
        f"{rows['Zweiter satz']!r}"
    )


# --- F5 (P2): session-end consolidation must use the routed bank ------------


def test_session_end_consolidation_must_use_the_routed_bank(tmp_path, monkeypatch):
    """After a routed write the consolidation must run on that bank.

    ``on_session_end`` builds its worker from ``self._beam``. The routed scope
    restores the original Beam when it ends, so the consolidation would run
    against the profile database instead of the bank that was written.
    """
    import mnemosyne_hermes as provider_module

    data = tmp_path / "data"
    _write_map(
        monkeypatch, data,
        {"_default": "vela", "_unknown": "quarantine", ALICE_KEY: "alice"},
    )
    provider = _provider(
        tmp_path, monkeypatch,
        session_id="sess-a", gateway_key=ALICE_KEY, identity="alice",
    )
    assert _remember(provider, "Ein satz fuer die konsolidierung").get("status") == "stored"

    captured: dict = {}

    class _FakeBeam:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.canonical_owner_id = "default"
            self.agent_context = "primary"

        def sleep(self):
            return None

    monkeypatch.setattr(provider_module, "_get_beam_class", lambda: _FakeBeam)
    # The reflection budget must not decide this test.
    monkeypatch.setattr(provider, "_reserve_reflection_budget_locked", lambda *a, **k: None)

    provider.on_session_end([{"role": "user", "content": "bis gleich"}])

    assert captured.get("db_path"), "the session-end sleep never ran"
    assert "banks" in str(captured["db_path"]), (
        f"consolidation ran against {captured['db_path']} instead of the routed bank"
    )


# --- F6: the cached banks must not survive the shutdown ---------------------


def test_shutdown_releases_the_cached_banks(tmp_path, monkeypatch):
    """A cached bank holds its own database connection — it must be released.

    The cache is keyed by storage identity, so every bank the session touched
    stays in the provider after the shutdown, with its connection open.
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

    provider.shutdown()

    assert not cached, f"cached banks survived the shutdown: {len(cached)}"
