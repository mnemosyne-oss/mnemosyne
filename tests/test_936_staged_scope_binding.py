"""#936 review: staged pending writes are not bound to their originating session.

dplush's 2026-09-16 review on PR #936:

  "Before merge, bind staged records to their originating Hermes scope and
   restore/validate that scope during replay; otherwise an approval after a
   session switch can write to or mutate the wrong session."

Mechanism: `_stage_pending_write` stores the tool payload but records no session.
`_handle_apply_pending` replays through `self._beam` — the beam of whatever
session is active when the approval arrives. A record staged in session A and
approved while session B is active is therefore written with B's scope.

Harness mirrors tests/test_apply_pending_replay_926.py: a `hermes_constants`
stub points the pending store at tmp_path, and `_import_provider` loads the
provider from its own source root so the repo-root stub cannot shadow the core.
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

STAGED_CONTENT = "belongs to session A"

# Provider packages whose module identities _import_provider() swaps.
PROVIDER_MODULES = ("hermes_memory_provider", "mnemosyne_hermes")


@pytest.fixture(autouse=True)
def _restore_provider_modules():
    """Restore provider + core module identities after each test.

    ``_import_provider`` drops the ``mnemosyne.*`` namespace and the provider
    packages from ``sys.modules`` and re-imports them off a different sys.path
    entry. Without this restore, any test module imported afterwards binds class
    objects from the superseded module, and isinstance checks, string-form
    monkeypatch targets and @patch counters break in that module.

    Same failure class as incident t_c200045e; the fixture matches the one in
    tests/test_936_scope_parity_audit.py and tests/test_apply_pending_replay_926.py.
    """
    saved_providers = {name: sys.modules.get(name) for name in PROVIDER_MODULES}
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


def _import_provider(package: str):
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            del sys.modules[name]
    for name in list(sys.modules):
        if name == "mnemosyne" or name.startswith("mnemosyne."):
            del sys.modules[name]
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        module = importlib.import_module(package)
        import mnemosyne.core.beam as _beam_mod

        module._BEAM_CLS = _beam_mod.BeamMemory
        return module
    finally:
        try:
            sys.path.remove(str(PROJECT_ROOT))
        except ValueError:
            pass


@contextmanager
def _pending_home(monkeypatch, tmp_path):
    stub = types.ModuleType("hermes_constants")
    stub.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", stub)
    yield tmp_path / "pending" / "memory"


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    data = tmp_path / "mnemosyne-data" / "private"
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(data))
    monkeypatch.setenv("MNEMOSYNE_HOST_LLM_ENABLED", "0")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield


def _provider(module, home: Path, session_id: str):
    p = module.MnemosyneMemoryProvider()
    p.initialize(session_id=session_id, hermes_home=str(home), agent_identity="main")
    assert p._beam is not None
    return p


def _staged_ids(resp):
    """Pending IDs across provider surfaces.

    Legacy `hermes_memory_provider` returns a single `pending_id`; the batch
    path returns `pending_ids`; the standalone surface returns a `staged` list.
    """
    if isinstance(resp.get("staged"), list):
        return [s["pending_id"] if isinstance(s, dict) else s for s in resp["staged"]]
    if resp.get("pending_ids"):
        return list(resp["pending_ids"])
    if resp.get("pending_id"):
        return [resp["pending_id"]]
    return []


def _force_approval_gate(module, monkeypatch):
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: True, raising=True)


def test_staged_payload_records_the_originating_session(monkeypatch, tmp_path):
    """The staged record must identify the session it was staged from."""
    module = _import_provider("hermes_memory_provider")
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        prov = _provider(module, tmp_path, "sess-a")
        resp = json.loads(prov.handle_tool_call(
            "mnemosyne_remember", {"content": STAGED_CONTENT, "scope": "session"}
        ))
        assert resp.get("status") == "staged", resp
        pid = _staged_ids(resp)[0]
        record = json.loads((pending_dir / f"{pid}.json").read_text())

    print("record keys:", sorted(record.keys()))
    print("payload keys:", sorted(record.get("payload", {}).keys()))

    blob = {**record, **record.get("payload", {})}
    session_ish = {k: v for k, v in blob.items() if "session" in k.lower()}
    assert session_ish, (
        "the staged record must carry the originating session so replay can "
        "restore it"
    )
    assert record.get("session_scope") == "hermes_sess-a", (
        f"unexpected recorded scope: {record.get('session_scope')!r}"
    )


def test_replay_restores_the_staging_session_not_the_active_one(monkeypatch, tmp_path):
    """A record staged in A, approved while B is active, must land in A."""
    module = _import_provider("hermes_memory_provider")
    _force_approval_gate(module, monkeypatch)
    with _pending_home(monkeypatch, tmp_path):
        a = _provider(module, tmp_path, "sess-a")
        resp = json.loads(a.handle_tool_call(
            "mnemosyne_remember",
            {"content": STAGED_CONTENT, "scope": "session", "importance": 0.5},
        ))
        pid = _staged_ids(resp)[0]

        # Session switch before the approval arrives.
        b = _provider(module, tmp_path, "sess-b")
        applied = json.loads(b.handle_tool_call(
            "mnemosyne_apply_pending", {"pending_ids": [pid]}
        ))
    print("apply result:", applied)

    db = tmp_path / "mnemosyne-data" / "private" / "mnemosyne.db"
    conn = sqlite3.connect(str(db))
    try:
        rows = conn.execute(
            "SELECT session_id FROM working_memory WHERE content LIKE ?",
            (f"%{STAGED_CONTENT}%",),
        ).fetchall()
    finally:
        conn.close()
    sessions = {r[0] for r in rows}
    print("row session(s):", sessions)
    assert sessions, "the replayed row should exist"
    assert sessions == {"hermes_sess-a"}, (
        "the record staged from sess-a must be replayed into sess-a, not the "
        f"session active at approval time; got {sessions}"
    )
    # The switch must be reported, not silently absorbed.
    entry = (applied.get("applied") or [{}])[0]
    print("applied entry:", entry)
    assert entry.get("session_replayed_into") == "hermes_sess-a"
    assert entry.get("session_redirected_from") == "hermes_sess-b"
    assert applied.get("session_redirected_count") == 1


# ---------------------------------------------------------------------------
# CodeRabbit follow-up on the same head: the redirected-session coverage
# above exercised only `remember`. update/forget/invalidate take the same
# scope-binding path and must be covered too, or a regression in the
# non-remember branch would pass unnoticed.
# ---------------------------------------------------------------------------


def _stage_batch(prov, ops):
    resp = json.loads(prov.handle_tool_call("mnemosyne_batch", {"operations": ops}))
    assert resp["status"] == "staged", resp
    return _staged_ids(resp)


def _seed(prov, content):
    resp = json.loads(prov.handle_tool_call(
        "mnemosyne_remember", {"content": content, "scope": "session"}
    ))
    assert resp["status"] == "stored", resp
    return resp["memory_id"]


def _working_row(db: Path, memory_id: str):
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT session_id, content, superseded_by, valid_until "
            "FROM working_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
    finally:
        conn.close()


@pytest.mark.parametrize("action", ["update", "forget", "invalidate"])
def test_redirected_replay_binds_every_action_to_the_staging_session(
    action, monkeypatch, tmp_path
):
    """Each non-remember action must target the STAGING session after a switch.

    Staged in sess-a, approved while sess-b is active: the operation must apply
    to sess-a's row and report the redirect. Without the scope binding, the
    replayed mutation would run through sess-b's beam and miss the row
    entirely (update/forget are session-scoped SQL) or leave sess-a untouched.
    """
    module = _import_provider("hermes_memory_provider")
    with _pending_home(monkeypatch, tmp_path) as pending_dir:
        a = _provider(module, tmp_path, "sess-a")
        # Seed while the gate is OFF so writes commit directly, then turn the
        # approval gate on for the staged operation.
        target = _seed(a, f"target for {action}")
        op = {"action": action, "memory_id": target}
        if action == "update":
            op["content"] = "updated under sess-a"
        if action == "invalidate":
            op["replacement_id"] = _seed(a, "replacement under sess-a")
        _force_approval_gate(module, monkeypatch)
        pids = _stage_batch(a, [op])
        assert (pending_dir / f"{pids[0]}.json").is_file()

        # Session switch before the approval arrives.
        b = _provider(module, tmp_path, "sess-b")
        applied = json.loads(b.handle_tool_call(
            "mnemosyne_apply_pending", {"pending_ids": pids}
        ))
        print(f"{action} apply result:", applied)

    assert applied["applied_count"] == 1, applied
    assert applied["failed_count"] == 0, applied
    entry = applied["applied"][0]
    assert entry["action"] == action
    assert entry["memory_id"] == target
    assert entry["session_replayed_into"] == "hermes_sess-a"
    assert entry["session_redirected_from"] == "hermes_sess-b"
    assert applied["session_redirected_count"] == 1
    # The record is consumed only by a successful replay.
    assert not (pending_dir / f"{pids[0]}.json").exists()

    db = tmp_path / "mnemosyne-data" / "private" / "mnemosyne.db"
    row = _working_row(db, target)
    if action == "forget":
        assert row is None, "the staging session's row must be the one deleted"
    elif action == "update":
        assert row is not None
        assert row[0] == "hermes_sess-a", row
        assert row[1] == "updated under sess-a", row
    else:  # invalidate
        assert row is not None
        assert row[0] == "hermes_sess-a", row
        assert row[2] is not None, "superseded_by must chain to the replacement"
        assert row[3] is not None, "valid_until must be set"
