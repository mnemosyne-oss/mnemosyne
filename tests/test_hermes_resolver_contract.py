"""Resolver schema -> public dispatch -> real core/cost/budget contract.

Both concrete Hermes copies are supported for this migration boundary. Only
candidate detection and the LLM transport are synthetic; no sockets are allowed.
These tests do not claim successful construction inside a real Hermes turn.
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import socket
from types import SimpleNamespace

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.query_cache import QueryCache

TOOL = "mnemosyne_resolve_conflicts"


@pytest.fixture(params=["hermes_memory_provider", "mnemosyne_hermes"])
def provider(request, tmp_path, monkeypatch):
    """Use real providers outside host turn scope with isolated synthetic stores."""
    def no_network(*args, **kwargs):
        raise AssertionError("resolver contract tests prohibit network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION", "1")
    module = importlib.import_module(request.param)
    p = module.MnemosyneMemoryProvider()
    p._hermes_home = str(tmp_path / "home")
    p._reflect_max_calls_per_session = 1
    p._reflect_calls_this_session = 0
    p._reflect_disabled_for_cron = True
    p._agent_context = "primary"
    b = BeamMemory(db_path=tmp_path / "synthetic.db", session_id="resolver-fixture")
    ids = []
    for bank in ("working_memory", "episodic_memory"):
        for i in range(2):
            content = f"Synthetic {bank} fixture fact {i}"
            if bank == "working_memory":
                mid = b.remember(content, source="fact", scope="global")
            else:
                mid = b.consolidate_to_episodic(content, source_wm_ids=[], source="fact", scope="global")
            ids.append(mid)
    # Three independent pairs prove reflection quota is not a per-pair quota.
    extra = [b.remember(f"Synthetic extra fixture {i}", source="fact", scope="global") for i in range(2)]
    pairs = list(zip((ids + extra)[::2], (ids + extra)[1::2]))
    monkeypatch.setattr(b, "_detect_conflicts", lambda rows, **kwargs: pairs)
    cache = QueryCache(db_path=tmp_path / "query-cache.db")
    cache.put("synthetic cached query", [{"id": ids[0], "content": "synthetic"}])
    b._query_cache = cache
    audit_module = importlib.import_module(request.param + ".audit")
    audit = audit_module.AuditLog(b.db_path)
    audit.record("fixture_seed", session_id=b.session_id)
    assert audit.count() == 1
    p._audit = audit
    p._beam = b
    p._session_id = b.session_id
    calls = []
    original = b.resolve_cross_session_conflicts

    def record_core(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(b, "resolve_cross_session_conflicts", record_core)
    transport_calls = []
    lcd = importlib.import_module("mnemosyne.core.llm_conflict_detector")
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)

    def transport(prompt, **kwargs):
        transport_calls.append((prompt, kwargs))
        return json.dumps({"is_conflict": True, "confidence": 0.9, "correct_fact": "synthetic"}), 100, 10

    monkeypatch.setattr(lcd, "_call_conflict_llm_with_retry", transport)
    state = SimpleNamespace(p=p, b=b, audit=audit, cache=cache, calls=calls, transport_calls=transport_calls, lcd=lcd)
    yield state
    p._beam = None
    audit.close()
    cache._conn.close()
    b.conn.close()


def snapshot(state):
    """Compare full memory rows, persisted/active cache, and actual apply audits."""
    return {
        "memories": {table: [tuple(r) for r in state.b.conn.execute(f"SELECT * FROM {table} ORDER BY id")]
                     for table in ("working_memory", "episodic_memory")},
        "audit": state.audit.query(),
        "cache_rows": [tuple(r) for r in state.cache._conn.execute("SELECT * FROM query_cache ORDER BY normalized")],
        "cache_memory": copy.deepcopy((state.cache._tier1, state.cache._tier4, state.cache._cache_version)),
    }


def costs(state):
    """Read the actual core cost writer's table, which is created on demand."""
    if not state.b.conn.execute("SELECT 1 FROM sqlite_master WHERE name='cost_entries'").fetchone():
        return []
    return [dict(r) for r in state.b.conn.execute("SELECT * FROM cost_entries")]


def execute(state, args):
    """Exercise configured discoverability and the provider's public dispatcher."""
    assert state.p.has_tool(TOOL)
    return json.loads(state.p.handle_tool_call(TOOL, args))


def test_resolver_advertised_arguments(provider):
    """Both migration surfaces advertise just the supported opt-in arguments."""
    schema = next(s for s in provider.p.get_tool_schemas() if s["name"] == TOOL)
    props = schema["parameters"]["properties"]
    assert set(props) == {"dry_run", "llm_eval"}
    for name in props:
        assert props[name]["type"] == "boolean"
        assert props[name]["default"] is False
    core = inspect.signature(BeamMemory.resolve_cross_session_conflicts)
    assert core.parameters["dry_run"].default is False
    assert core.parameters["llm_eval"].default is False


@pytest.mark.parametrize("args", [{}, {"dry_run": False}, {"dry_run": False, "llm_eval": False},
                                  {"dry_run": False, "llm_eval": True}, {"dry_run": True},
                                  {"dry_run": True, "llm_eval": False}, {"dry_run": True, "llm_eval": True}])
def test_resolver_real_core_kwargs(provider, args, monkeypatch):
    """Missing/false/true values reach the existing core once, without retries."""
    monkeypatch.delenv("MNEMOSYNE_CROSS_SESSION_CONFLICT_RESOLUTION")
    before = snapshot(provider)
    assert execute(provider, args)["status"] == "disabled"
    assert provider.calls == [{"dry_run": args.get("dry_run", False), "llm_eval": args.get("llm_eval", False)}]
    assert provider.p._reflect_calls_this_session == int(not args.get("dry_run", False) or args.get("llm_eval", False))
    assert snapshot(provider) == before
    assert not provider.transport_calls and not costs(provider)


@pytest.mark.parametrize("llm_eval", [None, False])
@pytest.mark.parametrize("guard", ["positive", "exhausted", "cron"])
def test_deterministic_preview_needs_no_budget(provider, llm_eval, guard):
    """Deterministic preview scans even when guarded, with no cost or reserve."""
    if guard == "exhausted":
        provider.p._reflect_calls_this_session = 1
    if guard == "cron":
        provider.p._agent_context = "cron"
    used = provider.p._reflect_calls_this_session
    args = {"dry_run": True}
    if llm_eval is not None:
        args["llm_eval"] = llm_eval
    before = snapshot(provider)
    result = execute(provider, args)
    assert result["status"] == "dry_run" and result["conflicts_resolved"] == 3
    assert result["llm_validations"] == result["invalidated"] == 0
    assert provider.p._reflect_calls_this_session == used
    assert not provider.transport_calls and not costs(provider)
    assert snapshot(provider) == before


@pytest.mark.parametrize("llm_flag", [False, True])
@pytest.mark.parametrize("verdict", ["confirm", "veto", "invalid", "transport_failure"])
def test_explicit_preview_costs_without_memory_mutation(provider, monkeypatch, llm_flag, verdict):
    """Explicit preview overrides the LLM flag; real logging does not supersede."""
    monkeypatch.setattr(provider.lcd, "LLM_CONFLICT_DETECTION_ENABLED", llm_flag)

    def transport(prompt, **kwargs):
        provider.transport_calls.append((prompt, kwargs))
        if verdict == "transport_failure":
            return None  # Real transport's documented exhausted-retry result.
        if verdict == "invalid":
            return json.dumps({"is_conflict": "yes", "confidence": 0.9}), 100, 10
        return json.dumps({"is_conflict": verdict == "confirm", "confidence": 0.9, "correct_fact": "synthetic"}), 100, 10

    monkeypatch.setattr(provider.lcd, "_call_conflict_llm_with_retry", transport)
    before = snapshot(provider)
    result = execute(provider, {"dry_run": True, "llm_eval": True})
    assert result["status"] == "dry_run"
    assert result["llm_validations"] == len(provider.transport_calls) == 3
    assert result["invalidated"] == 0
    assert result["conflicts_resolved"] == (3 if verdict == "confirm" else 0)
    assert provider.p._reflect_calls_this_session == 1
    logged = costs(provider)
    assert len(logged) == (3 if verdict in ("confirm", "veto") else 0)
    assert all(row["token_count"] == 110 and row["session_id"] == provider.b.session_id for row in logged)
    assert snapshot(provider) == before
    # No opportunistic refund for veto/invalid/transport error; next call skips.
    second = execute(provider, {"dry_run": True, "llm_eval": True})
    assert second["reason"] == "reflect_budget_exhausted"
    assert len(provider.calls) == 1 and len(provider.transport_calls) == 3
    assert provider.p._reflect_calls_this_session == 1
    assert costs(provider) == logged and snapshot(provider) == before


@pytest.mark.parametrize("guard", ["exhausted", "cron"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_cost_capable_calls_stop_before_core(provider, guard, dry_run):
    """Apply and opt-in evaluated previews obey exhausted/cron guardrails."""
    provider.p._reflect_calls_this_session = int(guard == "exhausted")
    provider.p._agent_context = "cron" if guard == "cron" else "primary"
    before = snapshot(provider)
    result = execute(provider, {"dry_run": dry_run, "llm_eval": True})
    assert result["status"] == "skipped"
    assert result["reason"] == ("reflect_budget_exhausted" if guard == "exhausted" else "reflect_disabled_for_cron")
    assert provider.p._reflect_calls_this_session == int(guard == "exhausted")
    assert not provider.calls and not provider.transport_calls and not costs(provider)
    assert snapshot(provider) == before


@pytest.mark.parametrize("guard", ["exhausted", "cron"])
@pytest.mark.parametrize("args", [{}, {"dry_run": False}, {"dry_run": False, "llm_eval": False}],
                         ids=["defaults", "apply", "apply-no-eval"])
def test_guarded_default_apply_stops_before_core(provider, guard, args):
    """Missing/false evaluation must not bypass apply admission or mutate state."""
    provider.p._reflect_calls_this_session = int(guard == "exhausted")
    provider.p._agent_context = "cron" if guard == "cron" else "primary"
    used = provider.p._reflect_calls_this_session
    before = snapshot(provider)
    result = execute(provider, args)
    assert result["status"] == "skipped"
    assert result["reason"] == ("reflect_budget_exhausted" if guard == "exhausted" else "reflect_disabled_for_cron")
    assert provider.p._reflect_calls_this_session == used
    assert not provider.calls and not provider.transport_calls and not costs(provider)
    assert snapshot(provider) == before


def test_missing_resolver_is_unavailable_without_reservation(provider):
    """Older/missing core method is a structured unavailable response, not work."""
    provider.p._beam = SimpleNamespace(session_id=provider.b.session_id)
    before = snapshot(provider)
    result = execute(provider, {"dry_run": True, "llm_eval": True})
    assert result["status"] == "unavailable"
    assert provider.p._reflect_calls_this_session == 0
    assert not provider.calls and not provider.transport_calls and not costs(provider)
    assert snapshot(provider) == before


def test_no_type_error_retry_or_refund(provider, monkeypatch):
    """A core TypeError is surfaced by the dispatcher once, never retried."""
    def fail(**kwargs):
        provider.calls.append(kwargs)
        raise TypeError("synthetic core failure")

    monkeypatch.setattr(provider.b, "resolve_cross_session_conflicts", fail)
    before = snapshot(provider)
    result = execute(provider, {"dry_run": True, "llm_eval": True})
    assert "synthetic core failure" in result["error"]
    assert provider.calls == [{"dry_run": True, "llm_eval": True}]
    assert provider.p._reflect_calls_this_session == 1
    assert snapshot(provider) == before


def test_default_apply_retains_heuristic_and_apply_audit(provider, monkeypatch):
    """Missing dry_run still applies, gated as before, even with LLM flag off."""
    monkeypatch.setattr(provider.lcd, "LLM_CONFLICT_DETECTION_ENABLED", False)
    before = snapshot(provider)
    result = execute(provider, {})
    assert result["status"] == "resolved" and result["invalidated"] == 3
    assert provider.p._reflect_calls_this_session == 1
    assert not provider.transport_calls and not costs(provider)
    after = snapshot(provider)
    assert after["memories"] != before["memories"]
    assert provider.cache._cache_version > before["cache_memory"][2]
    events = provider.audit.query()
    assert events[0]["action"] == "resolve_conflicts"
    assert json.loads(events[0]["metadata_json"])["invalidated"] == 3


def test_restricted_and_unknown_tools_do_not_execute(provider, tmp_path):
    """Configured allowlists do not re-expand at execution for this new option."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text("memory:\n  mnemosyne:\n    tools: [mnemosyne_recall]\n")
    assert [s["name"] for s in provider.p.get_tool_schemas()] == ["mnemosyne_recall"]
    before = snapshot(provider)
    for tool in (TOOL, "synthetic_unknown_tool"):
        assert not provider.p.has_tool(tool)
        assert "Unknown Mnemosyne tool" in json.loads(provider.p.handle_tool_call(tool, {"dry_run": True, "llm_eval": True}))["error"]
    assert provider.p._reflect_calls_this_session == 0 and not provider.calls
    assert snapshot(provider) == before
