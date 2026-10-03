"""#936 review round 2: scope binding, response parity, and replay audit parity.

dplush's 2026-09-16 CHANGES_REQUESTED review on PR #936 asked for three things:

1. BLOCKING — staged pending writes were not bound to their originating Hermes
   scope. Both staging paths omitted the effective Beam session/channel while
   replay called the current ``self._beam``; after ``on_session_switch()``
   rebinds the integration Beam (or ``initialize()`` rebuilds the root Beam), a
   session-scoped write approved from session A could be stored/applied under
   session B — including update/forget/invalidate. Required: persist the
   effective Beam session_id + channel_id in each pending payload on BOTH
   provider surfaces; during replay require the current scope to match or
   restore it under the Beam access lock; reject a mismatch without deleting the
   pending record; keep private targets bound to their originating session; add
   a session-switch regression test.
2. BLOCKING — provider response parity. The legacy surface returned the
   ``pending_ids``/``count`` aliases while the standalone surface returned only
   ``staged``/``staged_actions``/``staged_count``, so a client forwarding the
   documented compatibility key raised KeyError on one surface. Required:
   identical keys/values on both surfaces, asserted WITHOUT conditionally
   skipping a missing key.
3. NON-BLOCKING BUT REQUIRED — successful staged ``forget``/``invalidate``
   mutate memory without calling ``_audit_event``, unlike the direct handlers.
   Required: emit the corresponding audit events on both surfaces with
   ``source_tool="mnemosyne_apply_pending"`` and ``replacement_id`` metadata for
   invalidation.

Every test runs against BOTH provider surfaces unless a test is explicitly
surface-specific (the legacy surface's redirect-reporting fields).
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"

PROVIDER_MODULES = ["hermes_memory_provider", "mnemosyne_hermes"]


# ---------------------------------------------------------------------------
# Harness (mirrors tests/test_apply_pending_replay_926.py)
# ---------------------------------------------------------------------------


def _import_provider(package: str):
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name == "mnemosyne" or name.startswith("mnemosyne."):
            del sys.modules[name]
    import_root = INTEGRATION_SRC if package == "mnemosyne_hermes" else PROJECT_ROOT
    inserted = [str(import_root)]
    if import_root != PROJECT_ROOT:
        inserted.append(str(PROJECT_ROOT))
    for path in reversed(inserted):
        sys.path.insert(0, path)
    try:
        module = importlib.import_module(package)
        import mnemosyne.core.beam as _beam_mod

        module._BEAM_CLS = _beam_mod.BeamMemory
        return module
    finally:
        for path in inserted:
            try:
                sys.path.remove(path)
            except ValueError:
                pass


@pytest.fixture(autouse=True)
def _restore_provider_modules():
    saved_providers = {
        name: sys.modules.get(name) for name in PROVIDER_MODULES
    }
    saved_mnemosyne = {
        name: sys.modules.get(name)
        for name in list(sys.modules)
        if name == "mnemosyne" or name.startswith("mnemosyne.")
    }
    yield
    for name in list(sys.modules):
        if name == "mnemosyne" or name.startswith("mnemosyne."):
            sys.modules.pop(name, None)
    for name, module in saved_mnemosyne.items():
        if module is not None:
            sys.modules[name] = module
    for name, module in saved_providers.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    data = tmp_path / "mnemosyne-data" / "private"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield


@contextmanager
def _pending_home(monkeypatch, tmp_path):
    """Point the pending store at tmp_path for the duration of a test."""
    stub = types.ModuleType("hermes_constants")
    stub.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", stub)
    yield tmp_path / "pending" / "memory"


@contextmanager
def _provider(module, home: Path, session_id: str, *, channel_id: str = ""):
    """A real provider on a real (per-session) Beam in the tmp data dir."""
    p = module.MnemosyneMemoryProvider()
    kwargs = {
        "hermes_home": str(home),
        "agent_identity": "main",
        "profile_isolation": False,
        "agent_context": "primary",
    }
    if channel_id:
        kwargs["channel_id"] = channel_id
    p.initialize(session_id=session_id, **kwargs)
    assert p._beam is not None, "provider beam must initialize in the test home"
    try:
        yield p
    finally:
        try:
            p._beam.conn.close()
        except Exception:
            pass


def _force_approval_gate(module, monkeypatch):
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: True, raising=True)


def _staged_ids(resp):
    """Pending IDs across both staging response shapes."""
    if resp.get("pending_ids"):
        return list(resp["pending_ids"])
    if resp.get("staged"):
        return list(resp["staged"])
    if resp.get("pending_id"):
        return [resp["pending_id"]]
    return []


def _wm_rows(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT id, content, session_id, channel_id FROM working_memory"
        ).fetchall()
    finally:
        conn.close()


def _db_path(provider) -> Path:
    return Path(provider._beam.db_path)


def _record(pending_dir: Path, pid: str) -> dict:
    return json.loads((pending_dir / f"{pid}.json").read_text())


def _read_pending(module, tmp_path: Path, pid: str) -> dict:
    return json.loads((tmp_path / "pending" / "memory" / f"{pid}.json").read_text())


# ---------------------------------------------------------------------------
# 1. Staging records the effective Beam scope on BOTH surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_single_remember_staging_records_session_and_channel(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        with _provider(module, tmp_path, "sess-a") as prov:
            resp = json.loads(prov.handle_tool_call(
                "mnemosyne_remember", {"content": "scoped write", "scope": "session"}
            ))
            assert resp["status"] == "staged", resp
            pid = _staged_ids(resp)[0]
            record = _record(pending_dir, pid)

    assert record["session_scope"] == "hermes_sess-a", record
    # The effective Beam channel is recorded too (defaults to the session).
    assert record["channel_scope"] == str(prov._beam.channel_id)
    assert record["channel_scope"], "channel scope must be recorded, not empty"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_batch_staging_records_scope_for_every_action(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        with _provider(module, tmp_path, "sess-a") as prov:
            # Seed while the gate is OFF so it commits directly.
            seed = json.loads(prov.handle_tool_call(
                "mnemosyne_remember", {"content": "seed"}
            ))
            seed_id = seed["memory_id"]
            _force_approval_gate(module, monkeypatch)
            resp = json.loads(prov.handle_tool_call("mnemosyne_batch", {
                "operations": [
                    {"action": "remember", "content": "b-remember"},
                    {"action": "update", "memory_id": seed_id, "content": "b-update"},
                    {"action": "forget", "memory_id": seed_id},
                    {"action": "invalidate", "memory_id": seed_id},
                ],
            }))
            assert resp["status"] == "staged", resp
            pids = _staged_ids(resp)
            assert len(pids) == 4

            for pid in pids:
                record = _record(pending_dir, pid)
                assert record["session_scope"] == "hermes_sess-a"
                assert record["channel_scope"], "channel scope must be recorded"


# ---------------------------------------------------------------------------
# 2. Redirected replay: the write lands in the STAGING session, not the
#    approval session — for every action, on both surfaces.
# ---------------------------------------------------------------------------


def _stage_ops_in_session_a(prov, ops):
    """Stage a batch in the given provider's session; return the pending IDs."""
    resp = json.loads(prov.handle_tool_call("mnemosyne_batch", {"operations": ops}))
    assert resp["status"] == "staged", resp
    return _staged_ids(resp)


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_redirected_update_targets_the_staging_session(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        # Seed in session A while the gate is off, then stage the update.
        with _provider(module, tmp_path, "sess-a") as prov_a:
            seed = json.loads(prov_a.handle_tool_call(
                "mnemosyne_remember", {"content": "update me", "scope": "session"}
            ))
            seed_id = seed["memory_id"]
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "update", "memory_id": seed_id, "content": "updated by A"},
            ])

        # Approval arrives in session B (a different provider instance on the
        # same store, exactly like a session switch).
        with _provider(module, tmp_path, "sess-b") as prov_b:
            applied = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            assert applied["failed_count"] == 0, applied

            rows = _wm_rows(_db_path(prov_b))
            updated = [r for r in rows if r[0] == seed_id]
            assert len(updated) == 1
            assert updated[0][1] == "updated by A"
            # The row still belongs to session A: the redirect restored the
            # staging scope instead of mutating as session B.
            assert updated[0][2] == "hermes_sess-a", updated
            conn = sqlite3.connect(str(_db_path(prov_b)))
            try:
                owner = conn.execute(
                    "SELECT session_id FROM working_memory WHERE id = ?", (seed_id,)
                ).fetchone()[0]
            finally:
                conn.close()
            assert owner == "hermes_sess-a"

            entry = applied["applied"][0]
            assert entry["session_replayed_into"] == "hermes_sess-a"
            assert entry["session_redirected_from"] == "hermes_sess-b"
            assert applied["session_redirected_count"] == 1

            updates = [
                event for event in _audit_events(prov_b)
                if event["action"] == "update"
                and event["source_tool"] == "mnemosyne_apply_pending"
            ]
            assert len(updates) == 1
            assert updates[0]["memory_id"] == seed_id
            assert updates[0]["bank"] == "private"
            assert updates[0]["session_id"] == "hermes_sess-a"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_redirected_replay_failure_restores_live_scope_and_keeps_pending(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        with _provider(module, tmp_path, "sess-a") as prov_a:
            seed = json.loads(prov_a.handle_tool_call(
                "mnemosyne_remember", {"content": "update me", "scope": "session"}
            ))
            seed_id = seed["memory_id"]
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "update", "memory_id": seed_id, "content": "updated by A"},
            ])

        with _provider(module, tmp_path, "sess-b") as prov_b:
            before_session = prov_b._beam.session_id
            before_channel = prov_b._beam.channel_id
            real_update = prov_b._beam.update_working

            def fail_update(*args, **kwargs):
                raise RuntimeError("simulated redirected replay failure")

            monkeypatch.setattr(prov_b._beam, "update_working", fail_update)
            failed = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert failed["applied_count"] == 0, failed
            assert failed["failed_count"] == 1, failed
            assert (pending_dir / f"{pids[0]}.json").exists()
            assert prov_b._beam.session_id == before_session
            assert prov_b._beam.channel_id == before_channel

            monkeypatch.setattr(prov_b._beam, "update_working", real_update)
            monkeypatch.setattr(module, "_write_approval_enabled", lambda: False)
            follow_up = json.loads(prov_b.handle_tool_call(
                "mnemosyne_remember", {"content": "after failed replay"}
            ))
            assert follow_up["status"] == "stored"
            assert _wm_rows(_db_path(prov_b))[-1][2] == before_session


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_redirected_forget_targets_the_staging_session(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov_a:
            seed = json.loads(prov_a.handle_tool_call(
                "mnemosyne_remember", {"content": "forget me", "scope": "session"}
            ))
            seed_id = seed["memory_id"]
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "forget", "memory_id": seed_id},
            ])

        with _provider(module, tmp_path, "sess-b") as prov_b:
            # Sanity: session B cannot see session A's row (private target).
            conn = sqlite3.connect(str(_db_path(prov_b)))
            try:
                assert conn.execute(
                    "SELECT COUNT(*) FROM working_memory WHERE id = ?", (seed_id,)
                ).fetchone()[0] == 1
            finally:
                conn.close()

            applied = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            assert applied["failed_count"] == 0, applied

            # The row was deleted — through the staging session's scope, which
            # is the only scope that could authorize it.
            conn = sqlite3.connect(str(_db_path(prov_b)))
            try:
                remaining = conn.execute(
                    "SELECT COUNT(*) FROM working_memory WHERE id = ?", (seed_id,)
                ).fetchone()[0]
            finally:
                conn.close()
            assert remaining == 0, "redirected forget must delete the staging session's row"

            entry = applied["applied"][0]
            assert entry["session_replayed_into"] == "hermes_sess-a"
            assert applied["session_redirected_count"] == 1


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_redirected_invalidate_targets_the_staging_session(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov_a:
            old = json.loads(prov_a.handle_tool_call(
                "mnemosyne_remember", {"content": "invalidate me", "scope": "session"}
            ))
            repl = json.loads(prov_a.handle_tool_call(
                "mnemosyne_remember", {"content": "replacement", "scope": "session"}
            ))
            old_id, repl_id = old["memory_id"], repl["memory_id"]
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "invalidate", "memory_id": old_id, "replacement_id": repl_id},
            ])

        with _provider(module, tmp_path, "sess-b") as prov_b:
            applied = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            assert applied["failed_count"] == 0, applied

            conn = sqlite3.connect(str(_db_path(prov_b)))
            try:
                row = conn.execute(
                    "SELECT superseded_by, valid_until FROM working_memory WHERE id = ?",
                    (old_id,),
                ).fetchone()
            finally:
                conn.close()
            assert row is not None
            assert row[0] == repl_id, "replacement chaining must survive the redirect"
            assert row[1] is not None, "the old row must be expired"

            entry = applied["applied"][0]
            assert entry["session_replayed_into"] == "hermes_sess-a"
            assert applied["session_redirected_count"] == 1


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_redirected_remember_reports_scope_metadata(
    provider_module_name, monkeypatch, tmp_path
):
    """A remember staged in A and approved in B lands in A and says so."""
    module = _import_provider(provider_module_name)
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov_a:
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "remember", "content": "staged from A", "scope": "session"},
            ])

        with _provider(module, tmp_path, "sess-b") as prov_b:
            applied = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            rows = _wm_rows(_db_path(prov_b))
            by_content = {r[1]: r for r in rows}
            assert "staged from A" in by_content
            assert by_content["staged from A"][2] == "hermes_sess-a"
            assert applied["applied"][0]["session_replayed_into"] == "hermes_sess-a"
            assert applied["session_redirected_count"] == 1


@pytest.mark.parametrize("provider_module_name", ["mnemosyne_hermes"])
def test_explicit_channel_equal_to_session_is_not_rebound(
    provider_module_name, monkeypatch, tmp_path
):
    """An explicitly pinned channel must survive legacy replay unchanged."""
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        pending_dir.mkdir(parents=True, exist_ok=True)
        pid = "legacy01"
        (pending_dir / f"{pid}.json").write_text(json.dumps({
            "id": pid,
            "subsystem": "memory",
            "provider": "mnemosyne",
            "payload": {"action": "remember", "content": "pinned channel"},
            "session_scope": "hermes_sess-a",
        }))

        with _provider(
            module, tmp_path, "sess-b", channel_id="hermes_sess-b"
        ) as prov_b:
            applied = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": [pid]}
            ))
            assert applied["applied_count"] == 1, applied
            rows = _wm_rows(_db_path(prov_b))
            row = next(r for r in rows if r[1] == "pinned channel")
            assert row[2] == "hermes_sess-a"
            assert row[3] == "hermes_sess-b"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_same_session_replay_is_not_reported_as_redirected(
    provider_module_name, monkeypatch, tmp_path
):
    """The common same-session approval must not claim a redirect."""
    module = _import_provider(provider_module_name)
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov_a:
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "remember", "content": "same session", "scope": "session"},
            ])
            applied = json.loads(prov_a.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            assert applied["session_redirected_count"] == 0
            assert "session_replayed_into" not in applied["applied"][0]
            rows = _wm_rows(_db_path(prov_a))
            assert rows[0][2] == "hermes_sess-a"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_channel_binding_is_restored_when_it_differs_under_one_session(
    provider_module_name, monkeypatch, tmp_path
):
    """The channel is its own axis: it is restored even without a session switch.

    A record staged while the Beam carried an explicit channel_id must replay
    under THAT channel (a channel-only difference is corrected silently; the
    session-redirect fields stay clear because no redirect happened).
    """
    module = _import_provider(provider_module_name)
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        with _provider(module, tmp_path, "sess-a") as prov:
            # Re-bind the live beam to a distinct channel, then stage.
            prov._beam.channel_id = "channel-one"
            pids = _stage_ops_in_session_a(prov, [
                {"action": "remember", "content": "channel bound", "scope": "session"},
            ])
            record = _record(pending_dir, pids[0])
            assert record["channel_scope"] == "channel-one"

            # Same session, different channel at approval time.
            prov._beam.channel_id = "channel-two"
            applied = json.loads(prov.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            assert applied["failed_count"] == 0, applied
            # No session redirect happened, so none is reported.
            assert applied["session_redirected_count"] == 0
            assert "session_replayed_into" not in applied["applied"][0]

            conn = sqlite3.connect(str(_db_path(prov)))
            try:
                row = conn.execute(
                    "SELECT session_id, channel_id FROM working_memory WHERE content = ?",
                    ("channel bound",),
                ).fetchone()
            finally:
                conn.close()
            assert row is not None
            assert row[0] == "hermes_sess-a"
            # The staged channel is restored instead of the approval-time one.
            assert row[1] == "channel-one", row
            # The live beam keeps its own channel afterwards.
            assert prov._beam.channel_id == "channel-two"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_legacy_record_without_recorded_scope_replays_through_active_beam(
    provider_module_name, monkeypatch, tmp_path
):
    """A record staged before session_scope existed stays backward compatible."""
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        pending_dir.mkdir(parents=True, exist_ok=True)
        (pending_dir / "legacy01.json").write_text(json.dumps({
            "id": "legacy01", "subsystem": "memory", "provider": "mnemosyne",
            "tool": "mnemosyne_remember",
            "payload": {"tool": "mnemosyne_remember", "content": "legacy staged"},
            "summary": "legacy staged", "created_at": 0,
        }))
        with _provider(module, tmp_path, "sess-a") as prov:
            applied = json.loads(prov.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": ["legacy01"]}
            ))
            assert applied["applied_count"] == 1, applied
            assert applied["failed_count"] == 0, applied
            assert applied["session_redirected_count"] == 0
            rows = _wm_rows(_db_path(prov))
            assert [r[1] for r in rows] == ["legacy staged"]
            assert rows[0][2] == "hermes_sess-a"


# ---------------------------------------------------------------------------
# 3. Response parity: the staged batch response is identical on both surfaces,
#    asserted WITHOUT a conditional key guard (dplush finding 2).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_staged_batch_response_exposes_the_compat_aliases(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov:
            resp = json.loads(prov.handle_tool_call("mnemosyne_batch", {
                "operations": [
                    {"action": "remember", "content": "one"},
                    {"action": "remember", "content": "two"},
                ],
            }))
    assert resp["status"] == "staged"
    # Unconditional: a client forwarding the documented compatibility key must
    # not have to guard on its presence (that guard is what hid the divergence).
    assert "pending_ids" in resp
    assert "count" in resp
    assert "staged" in resp
    assert "staged_count" in resp
    assert resp["pending_ids"] == resp["staged"] == _staged_ids(resp)
    assert resp["count"] == resp["staged_count"] == len(resp["staged"]) == 2


def test_staged_batch_response_keys_match_across_surfaces(monkeypatch, tmp_path):
    """The two surfaces must return the SAME keys, value-for-value."""
    responses = {}
    for name in PROVIDER_MODULES:
        module = _import_provider(name)
        _force_approval_gate(module, monkeypatch)
        with _pending_home(monkeypatch, tmp_path):
            with _provider(module, tmp_path, "sess-parity") as prov:
                responses[name] = json.loads(prov.handle_tool_call("mnemosyne_batch", {
                    "operations": [
                        {"action": "remember", "content": "parity one"},
                        {"action": "forget", "memory_id": "00000000-dead-beef"},
                    ],
                }))

    legacy, standalone = responses["hermes_memory_provider"], responses["mnemosyne_hermes"]
    assert set(legacy) == set(standalone), (
        f"response keys diverge: {sorted(set(legacy) ^ set(standalone))}"
    )
    # Values that are surface-independent must be equal, not merely present.
    assert legacy["status"] == standalone["status"] == "staged"
    assert legacy["staged_count"] == standalone["staged_count"] == 2
    assert legacy["count"] == standalone["count"] == 2
    assert legacy["message"] == standalone["message"]
    assert [e["action"] for e in legacy["staged_actions"]] == \
        [e["action"] for e in standalone["staged_actions"]] == ["remember", "forget"]
    # Pending IDs are per-record, so compare shape + count rather than identity.
    assert len(legacy["staged"]) == len(standalone["staged"]) == 2
    assert all(isinstance(pid, str) and pid for pid in legacy["staged"])
    assert all(isinstance(pid, str) and pid for pid in standalone["staged"])


# ---------------------------------------------------------------------------
# 4. Audit parity for replayed forget/invalidate (dplush finding 3)
# ---------------------------------------------------------------------------


def _audit_events(prov):
    assert prov._audit is not None, "audit log must initialize with the beam"
    return prov._audit.query(limit=50)


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_direct_and_replayed_update_audit_parity(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov:
            direct_target = json.loads(prov.handle_tool_call(
                "mnemosyne_remember", {"content": "direct audit update target"}
            ))["memory_id"]
            replay_target = json.loads(prov.handle_tool_call(
                "mnemosyne_remember", {"content": "replay audit update target"}
            ))["memory_id"]

            direct = json.loads(prov.handle_tool_call("mnemosyne_update", {
                "memory_id": direct_target,
                "content": "direct audited update",
            }))
            assert direct["status"] == "updated", direct

            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov, [{
                "action": "update",
                "memory_id": replay_target,
                "content": "replayed audited update",
            }])
            applied = json.loads(prov.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            events = _audit_events(prov)

    updates = [event for event in events if event["action"] == "update"]
    assert len(updates) == 2, events
    by_source = {event["source_tool"]: event for event in updates}
    assert set(by_source) == {"mnemosyne_update", "mnemosyne_apply_pending"}
    assert by_source["mnemosyne_update"]["memory_id"] == direct_target
    assert by_source["mnemosyne_apply_pending"]["memory_id"] == replay_target
    for event in updates:
        assert event["bank"] == "private"
        assert event["session_id"] == "hermes_sess-a"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
@pytest.mark.parametrize("action", ["remember", "update"])
def test_replayed_remember_and_update_emit_audit_events(
    provider_module_name, action, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov:
            operation = {"action": "remember", "content": "audited staged remember"}
            if action == "update":
                seed = json.loads(prov.handle_tool_call(
                    "mnemosyne_remember", {"content": "audit update target"}
                ))
                operation = {
                    "action": "update",
                    "memory_id": seed["memory_id"],
                    "content": "audited staged update",
                }
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov, [operation])
            applied = json.loads(prov.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            events = _audit_events(prov)

    replay_events = [
        event for event in events
        if event["action"] == action
        and event["source_tool"] == "mnemosyne_apply_pending"
    ]
    assert len(replay_events) == 1, events
    assert replay_events[0]["session_id"] == "hermes_sess-a"


def test_root_provider_session_switch_rebinds_before_staging(monkeypatch, tmp_path):
    module = _import_provider("hermes_memory_provider")
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        with _provider(module, tmp_path, "sess-a") as prov:
            old_session = prov._session_id
            assert old_session == "hermes_sess-a"
            prov.on_session_switch("sess-b")
            assert prov._session_id == "hermes_sess-b"
            assert prov._beam.session_id == "hermes_sess-b"
            assert prov._beam.channel_id == "hermes_sess-b"

            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov, [
                {"action": "remember", "content": "staged after switch"},
            ])
            record = _record(pending_dir, pids[0])
            assert record["session_scope"] == "hermes_sess-b"
            assert record["channel_scope"] == "hermes_sess-b"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_replayed_forget_emits_audit_event(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov:
            seed = json.loads(prov.handle_tool_call(
                "mnemosyne_remember", {"content": "audit forget target"}
            ))
            seed_id = seed["memory_id"]
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov, [
                {"action": "forget", "memory_id": seed_id},
            ])
            applied = json.loads(prov.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            events = _audit_events(prov)

    forgets = [
        e for e in events
        if e["action"] == "forget" and e["source_tool"] == "mnemosyne_apply_pending"
    ]
    assert len(forgets) == 1, events
    assert forgets[0]["memory_id"] == seed_id
    assert forgets[0]["bank"] == "private"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_replayed_invalidate_emits_audit_event_with_replacement_metadata(
    provider_module_name, monkeypatch, tmp_path
):
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov:
            old = json.loads(prov.handle_tool_call(
                "mnemosyne_remember", {"content": "audit invalidate target"}
            ))
            repl = json.loads(prov.handle_tool_call(
                "mnemosyne_remember", {"content": "audit replacement"}
            ))
            old_id, repl_id = old["memory_id"], repl["memory_id"]
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov, [
                {"action": "invalidate", "memory_id": old_id, "replacement_id": repl_id},
            ])
            applied = json.loads(prov.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            events = _audit_events(prov)

    invalidations = [
        e for e in events
        if e["action"] == "invalidate" and e["source_tool"] == "mnemosyne_apply_pending"
    ]
    assert len(invalidations) == 1, events
    assert invalidations[0]["memory_id"] == old_id
    metadata = json.loads(invalidations[0]["metadata_json"] or "{}")
    assert metadata.get("replacement_id") == repl_id, metadata
    assert metadata.get("invalidated") is True


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_replayed_forget_and_invalidate_audit_the_staging_session(
    provider_module_name, monkeypatch, tmp_path
):
    """The audit row names the RECORDED scope, like the mutation it describes.

    A forget/invalidate staged in session A but approved in session B lands in
    A; the audit trail must not claim it happened under the approving session
    (CodeRabbit review 5241469678, Finding A).
    """
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov_a:
            forget_target = json.loads(prov_a.handle_tool_call(
                "mnemosyne_remember", {"content": "redirect audit forget target"}
            ))
            invalidate_target = json.loads(prov_a.handle_tool_call(
                "mnemosyne_remember", {"content": "redirect audit invalidate target"}
            ))
            replacement = json.loads(prov_a.handle_tool_call(
                "mnemosyne_remember", {"content": "redirect audit replacement"}
            ))
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "forget", "memory_id": forget_target["memory_id"]},
                {
                    "action": "invalidate",
                    "memory_id": invalidate_target["memory_id"],
                    "replacement_id": replacement["memory_id"],
                },
            ])

        with _provider(module, tmp_path, "sess-b") as prov_b:
            applied = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 2, applied
            assert applied["failed_count"] == 0, applied
            assert applied["session_redirected_count"] == 2, applied
            # The applied entries keep reporting the redirect...
            for entry in applied["applied"]:
                assert entry["session_replayed_into"] == "hermes_sess-a", entry
                assert entry["session_redirected_from"] == "hermes_sess-b", entry
            events = _audit_events(prov_b)

    replay_events = [
        e for e in events
        if e["source_tool"] == "mnemosyne_apply_pending"
        and e["action"] in ("forget", "invalidate")
    ]
    assert len(replay_events) == 2, events
    assert {e["action"] for e in replay_events} == {"forget", "invalidate"}
    for event in replay_events:
        # ...and the audit trail names the staging scope the mutation landed
        # in, not the approving session it was replayed from.
        assert event["session_id"] == "hermes_sess-a", event


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_failed_replay_emits_no_audit_event(
    provider_module_name, monkeypatch, tmp_path
):
    """A rejected (not-found) mutation is not a mutation; no audit event."""
    module = _import_provider(provider_module_name)
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov:
            pids = _stage_ops_in_session_a(prov, [
                {"action": "forget", "memory_id": "00000000-dead-beef"},
            ])
            applied = json.loads(prov.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["failed_count"] == 1, applied
            events = _audit_events(prov)

    assert [
        e for e in events
        if e["source_tool"] == "mnemosyne_apply_pending"
    ] == [], events


# ---------------------------------------------------------------------------
# 5. Scope binding does not leak into the live provider after replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_replay_restores_the_live_beam_scope(
    provider_module_name, monkeypatch, tmp_path
):
    """After a redirected replay the approving provider is untouched."""
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov_a:
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "remember", "content": "scope restore", "scope": "session"},
            ])

        with _provider(module, tmp_path, "sess-b") as prov_b:
            before_session = prov_b._beam.session_id
            before_channel = prov_b._beam.channel_id
            applied = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied
            # The temporary swap is restored: later same-session writes keep
            # landing under the approving session.
            assert prov_b._beam.session_id == before_session
            assert prov_b._beam.channel_id == before_channel
            assert prov_b._session_id == "hermes_sess-b"

            # Gate back OFF: the follow-up write must commit directly, and it
            # must land in the APPROVING session, not the replayed one.
            monkeypatch.setattr(
                module, "_write_approval_enabled", lambda: False, raising=True
            )
            follow_up = json.loads(prov_b.handle_tool_call(
                "mnemosyne_remember", {"content": "after replay", "scope": "session"}
            ))
            assert follow_up["status"] == "stored", follow_up
            rows = {r[1]: r for r in _wm_rows(_db_path(prov_b))}
            assert rows["after replay"][2] == "hermes_sess-b"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_replay_keeps_the_live_beam_identity(
    provider_module_name, monkeypatch, tmp_path
):
    """The redirected replay reuses the live Beam, so author identity survives."""
    module = _import_provider(provider_module_name)
    with _pending_home(monkeypatch, tmp_path):
        with _provider(module, tmp_path, "sess-a") as prov_a:
            prov_a._beam.author_id = "alice"
            prov_a._beam.author_type = "human"
            _force_approval_gate(module, monkeypatch)
            pids = _stage_ops_in_session_a(prov_a, [
                {"action": "remember", "content": "authored from A", "scope": "session"},
            ])

        with _provider(module, tmp_path, "sess-b") as prov_b:
            prov_b._beam.author_id = "bob"
            prov_b._beam.author_type = "human"
            applied = json.loads(prov_b.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["applied_count"] == 1, applied

            conn = sqlite3.connect(str(_db_path(prov_b)))
            try:
                row = conn.execute(
                    "SELECT session_id, author_id FROM working_memory "
                    "WHERE content = ?", ("authored from A",),
                ).fetchone()
            finally:
                conn.close()
            assert row is not None
            assert row[0] == "hermes_sess-a"
            # A second BeamMemory would have written (None, None) here.
            assert row[1] == "bob", "replay must not strip the live beam's author identity"


@pytest.mark.parametrize("provider_module_name", PROVIDER_MODULES)
def test_invalid_record_still_retains_on_mismatch(
    provider_module_name, monkeypatch, tmp_path
):
    """A rejected record's file survives — a mismatch must never unlink it."""
    module = _import_provider(provider_module_name)
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        with _provider(module, tmp_path, "sess-a") as prov:
            pids = _stage_ops_in_session_a(prov, [
                {"action": "update", "memory_id": "00000000-dead-beef", "content": "x"},
            ])
            applied = json.loads(prov.handle_tool_call(
                "mnemosyne_apply_pending", {"pending_ids": pids}
            ))
            assert applied["failed_count"] == 1, applied
            assert (pending_dir / f"{pids[0]}.json").is_file()
