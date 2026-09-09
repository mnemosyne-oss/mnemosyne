"""RED-GREEN regression tests for CodeRabbit finding 4 (PR #926) — replay
approved batch operations by action.

The write-approval staging path must preserve the COMPLETE normalized
operation payload (memory_id, replacement_id, action-specific fields), and
``_handle_apply_pending`` must dispatch each approved record by its
captured ``action`` the same way ``apply_beam_batch``/``_apply_one`` does:

1. approved UPDATE modifies the existing memory identified by memory_id —
   no duplicate/new memory is created and the fields land on the existing
   row;
2. approved FORGET executes as a content-less operation — the target
   memory is forgotten and no empty-content record is created or left
   behind;
3. approved INVALIDATE is content-less too;
4. approved operations with replacement_id chain correctly — the linkage
   matches apply_beam_batch behavior;
5. the staged pending record is removed only after successful replay and
   survives a simulated replay failure.

Finding 1 (partial updates): staging must NOT fabricate defaults for an
update that never supplied them. Staging every op with
``content=get("content", "")`` and ``importance=get("importance", 0.5)``
made an importance-only update blank the row content, and a content-only
update reset the row importance to 0.5, because replay forwards both
staged values to ``update_working`` which applies every non-None arg.
These tests pin that an approved update touches ONLY the supplied fields,
and that the approval/replay path produces the identical row as the
direct ``apply_beam_batch`` path for the same partial update.

Before the fix, staging dropped memory_id (legacy surface) and
replacement_id (both surfaces), and replay called beam.remember()
unconditionally with a non-empty-content requirement, so approved updates
created new memories, forgets/invalidates were dropped as "empty content",
and replacement chaining never happened.

Run on both provider surfaces (legacy ``hermes_memory_provider`` and
standalone ``mnemosyne_hermes``).
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import tempfile
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"


@pytest.fixture(autouse=True)
def _restore_provider_modules():
    """Restore the pre-test module identities after each test.

    ``_import_provider`` re-imports the provider packages and the real
    ``mnemosyne`` core (the repo-root ``__init__.py`` stub shadows the core
    under pytest importlib mode). Other test modules bind their imports
    (e.g. ``from mnemosyne.core.llm_backends import get_host_llm_backend``)
    at collection time, so swapping the ``mnemosyne.*`` namespace would
    orphan those references. Snapshot every touched module and restore the
    whole namespace afterwards.
    """
    saved_providers = {
        name: sys.modules.get(name)
        for name in ("hermes_memory_provider", "mnemosyne_hermes")
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


@contextmanager
def _make_provider(module):
    """Construct a minimal provider + real BeamMemory in a temp DB.

    ``beam_cls`` is the REAL core BeamMemory captured inside the import
    swap window (the repo-root __init__.py stub can shadow the core
    package under pytest importlib mode; see _import_provider).
    """
    beam_cls = getattr(module, "_BEAM_CLS")  # set by _import_provider
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "mnemosyne.db"
        provider = module.MnemosyneMemoryProvider()
        provider._beam = beam_cls(session_id="hermes_replay-test", db_path=db_path)
        provider._default_scope = "session"
        provider._agent_context = "primary"
        provider._skip_contexts = set()
        try:
            yield provider, db_path
        finally:
            provider._beam.conn.close()


def _import_provider(package: str):
    """Import a provider package from its own source root.

    Under pytest's importlib mode the repo-root ``__init__.py`` stub can
    shadow the inner ``mnemosyne`` core package, so the core package is
    dropped from sys.modules for the duration of the import (sys.path[0]
    is the repo root, where the real core directory wins), then re-imported
    and LEFT CACHED as the real core: the tests construct real BeamMemory
    instances whose bodies import ``mnemosyne.core.*``, which would fail
    against the stub. Leaving the real core cached matches the installed
    (pip -e) environment CI runs in.
    """
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


# Non-default values for the finding-1 partial-update scenarios: the seed
# carries importance 0.9 (≠ staging's 0.5 default) so a content-only update
# that leaks a fabricated importance is immediately visible.
SEED_CONTENT = "partial update seed"
SEED_IMPORTANCE = 0.9
UPDATED_CONTENT = "content updated by approved op"
UPDATED_IMPORTANCE = 0.25


def _wm_rows(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        # Column order: id, content, superseded_by, valid_until, importance.
        return conn.execute(
            "SELECT id, content, superseded_by, valid_until, importance "
            "FROM working_memory"
        ).fetchall()
    finally:
        conn.close()


def _staged_ids(resp):
    """Legacy surface returns pending_ids; standalone returns staged list."""
    staged = resp.get("staged") or [
        {"pending_id": pid} for pid in resp.get("pending_ids", [])
    ]
    return [s["pending_id"] if isinstance(s, dict) else s for s in staged]


def _record_payload(pending_dir, pid):
    return json.loads((pending_dir / f"{pid}.json").read_text())["payload"]


@contextmanager
def _approval_setup(monkeypatch, tmp_path):
    """Point the pending store at tmp_path for the duration of a test.

    monkeypatch.setitem already restores sys.modules, so no explicit
    teardown is needed here.
    """
    stub = types.ModuleType("hermes_constants")
    stub.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", stub)
    yield tmp_path / "pending" / "memory"


# ---------------------------------------------------------------------------
# 1. Approved UPDATE must modify the existing memory (no new record)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_apply_pending_update_targets_existing_row(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        with _approval_setup(monkeypatch, tmp_path) as pending_dir:
            # Seed a memory while the approval gate is OFF so it commits
            # directly.
            seed = json.loads(provider._handle_remember({"content": "seed content"}))
            seed_id = seed["memory_id"]
            rows = _wm_rows(db_path)
            assert len(rows) == 1
            assert rows[0][0] == seed_id

            # Stage an approved UPDATE through the write-approval gate.
            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "update", "memory_id": seed_id, "content": "updated content"},
            ]}))
            assert resp["status"] == "staged"
            pids = _staged_ids(resp)
            assert len(pids) == 1
            # The staged payload must retain the target identity.
            payload = _record_payload(pending_dir, pids[0])
            assert payload["memory_id"] == seed_id
            assert payload["action"] == "update"

            applied = json.loads(provider._handle_apply_pending({"pending_ids": pids}))
            assert applied["applied_count"] == 1
            assert applied["failed_count"] == 0

            rows = _wm_rows(db_path)
            # No duplicate/new memory: exactly the seed row remains, updated.
            assert len(rows) == 1
            assert rows[0][0] == seed_id
            assert rows[0][1] == "updated content"


# ---------------------------------------------------------------------------
# 2/3. Approved FORGET / INVALIDATE are content-less operations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_apply_pending_forget_is_content_less(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        with _approval_setup(monkeypatch, tmp_path) as pending_dir:
            seed = json.loads(provider._handle_remember({"content": "target to forget"}))
            seed_id = seed["memory_id"]

            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "forget", "memory_id": seed_id},
            ]}))
            assert resp["status"] == "staged"
            pids = _staged_ids(resp)
            assert len(pids) == 1
            payload = _record_payload(pending_dir, pids[0])
            assert payload["action"] == "forget"
            # Content-less op must be staged without fabricated content
            # (finding 1: absent fields stage as None, not "").
            assert not payload.get("content")
            assert payload["memory_id"] == seed_id

            applied = json.loads(provider._handle_apply_pending({"pending_ids": pids}))
            # Content-less op must replay successfully, not fail as
            # "empty content".
            assert applied["applied_count"] == 1
            assert applied["failed_count"] == 0

            rows = _wm_rows(db_path)
            # Target forgotten; no empty-content record is left behind.
            assert len(rows) == 0
            # Pending record consumed after successful replay.
            assert not (pending_dir / f"{pids[0]}.json").exists()


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_apply_pending_invalidate_is_content_less(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        with _approval_setup(monkeypatch, tmp_path) as pending_dir:
            seed = json.loads(provider._handle_remember({"content": "target to invalidate"}))
            seed_id = seed["memory_id"]

            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "invalidate", "memory_id": seed_id},
            ]}))
            assert resp["status"] == "staged"
            pids = _staged_ids(resp)
            payload = _record_payload(pending_dir, pids[0])
            assert payload["action"] == "invalidate"
            # Content-less op must be staged without fabricated content
            # (finding 1: absent fields stage as None, not "").
            assert not payload.get("content")

            applied = json.loads(provider._handle_apply_pending({"pending_ids": pids}))
            assert applied["applied_count"] == 1
            assert applied["failed_count"] == 0

            rows = _wm_rows(db_path)
            # Row still exists (invalidate marks, does not delete), but the
            # memory is expired: valid_until set, superseded_by untouched.
            assert len(rows) == 1
            assert rows[0][0] == seed_id
            assert rows[0][2] is None  # superseded_by
            assert rows[0][3] is not None  # valid_until
            assert not (pending_dir / f"{pids[0]}.json").exists()


# ---------------------------------------------------------------------------
# 4. replacement_id chaining matches apply_beam_batch behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_apply_pending_invalidate_replacement_chains(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        with _approval_setup(monkeypatch, tmp_path) as pending_dir:
            old = json.loads(provider._handle_remember({"content": "superseded memory"}))
            repl = json.loads(provider._handle_remember({"content": "replacement memory"}))
            old_id, repl_id = old["memory_id"], repl["memory_id"]

            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "invalidate", "memory_id": old_id,
                 "replacement_id": repl_id},
            ]}))
            assert resp["status"] == "staged"
            pids = _staged_ids(resp)
            payload = _record_payload(pending_dir, pids[0])
            assert payload["action"] == "invalidate"
            assert payload["replacement_id"] == repl_id

            applied = json.loads(provider._handle_apply_pending({"pending_ids": pids}))
            assert applied["applied_count"] == 1
            assert applied["failed_count"] == 0

            rows = {r[0]: r for r in _wm_rows(db_path)}
            # Chained: old row points at the replacement, is expired.
            assert rows[old_id][2] == repl_id  # superseded_by
            assert rows[old_id][3] is not None  # valid_until
            # Replacement itself untouched.
            assert rows[repl_id][2] is None
            assert rows[repl_id][3] is None

    # Direct apply_beam_batch path (gate OFF) must produce identical
    # linkage for the same invalidate-with-replacement operation.
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: False)
    with _make_provider(module) as (provider2, db_path2):
        old2 = json.loads(provider2._handle_remember({"content": "superseded v2"}))
        repl2 = json.loads(provider2._handle_remember({"content": "replacement v2"}))
        direct = json.loads(provider2._handle_batch({"operations": [
            {"action": "invalidate", "memory_id": old2["memory_id"],
             "replacement_id": repl2["memory_id"]},
        ]}))
        assert direct["status"] == "ok"
        rows2 = {r[0]: r for r in _wm_rows(db_path2)}
        assert rows2[old2["memory_id"]][2] == repl2["memory_id"]
        assert rows2[old2["memory_id"]][3] is not None
        assert rows2[repl2["memory_id"]][2] is None


# ---------------------------------------------------------------------------
# 5. Pending record removed only after successful replay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_apply_pending_record_deleted_only_after_success(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        with _approval_setup(monkeypatch, tmp_path) as pending_dir:
            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)

            # --- success: staged remember is applied and the record goes away
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "remember", "content": "approved remember"},
            ]}))
            pids = _staged_ids(resp)
            assert len(pids) == 1
            rec = pending_dir / f"{pids[0]}.json"
            assert rec.exists()
            applied = json.loads(provider._handle_apply_pending({"pending_ids": pids}))
            assert applied["applied_count"] == 1
            assert applied["failed_count"] == 0
            assert not rec.exists()

            # --- replay failure (exception): record survives, not lost
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "remember", "content": "doomed remember"},
            ]}))
            pids = _staged_ids(resp)
            rec = pending_dir / f"{pids[0]}.json"
            assert rec.exists()

            def _boom(**kwargs):
                raise RuntimeError("simulated replay failure")

            monkeypatch.setattr(provider._beam, "remember", _boom)
            applied = json.loads(provider._handle_apply_pending({"pending_ids": pids}))
            assert applied["applied_count"] == 0
            assert applied["failed_count"] == 1
            # Approved operation must not be silently lost.
            assert rec.exists()

            # --- replay failure (content-less op, target missing): record
            # survives too. forget_working returns False for a memory that
            # does not exist, and the replay must leave the record staged.
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "forget", "memory_id": "00000000-dead-beef"},
            ]}))
            pids = _staged_ids(resp)
            rec = pending_dir / f"{pids[0]}.json"
            assert rec.exists()
            applied = json.loads(provider._handle_apply_pending({"pending_ids": pids}))
            assert applied["applied_count"] == 0
            assert applied["failed_count"] == 1
            assert rec.exists(), "record must survive a failed content-less replay"


# ---------------------------------------------------------------------------
# 6. Finding 1 regression: approved UPDATE touches ONLY the supplied field
#    (importance-only update preserves content, content-only update
#    preserves importance), and matches the direct apply_beam_batch path.
# ---------------------------------------------------------------------------


def _stage_and_replay_update(provider, db_path, seed_id, op_kwargs):
    """Stage an approved partial update for an existing memory and replay
    it through _handle_apply_pending; returns the row after replay."""
    resp = json.loads(provider._handle_batch({"operations": [
        {"action": "update", "memory_id": seed_id, **op_kwargs},
    ]}))
    assert resp["status"] == "staged"
    pids = _staged_ids(resp)
    assert len(pids) == 1

    applied = json.loads(provider._handle_apply_pending({"pending_ids": pids}))
    assert applied["applied_count"] == 1
    assert applied["failed_count"] == 0

    rows = _wm_rows(db_path)
    assert len(rows) == 1
    assert rows[0][0] == seed_id
    return rows[0]


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_apply_pending_update_importance_only_preserves_content(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        # Seed a memory with non-default importance while the approval
        # gate is OFF so it commits directly.
        seed = json.loads(provider._handle_remember(
            {"content": SEED_CONTENT, "importance": SEED_IMPORTANCE}
        ))
        seed_id = seed["memory_id"]
        rows = _wm_rows(db_path)
        assert len(rows) == 1
        assert rows[0][1] == SEED_CONTENT
        assert rows[0][4] == pytest.approx(SEED_IMPORTANCE)
        with _approval_setup(monkeypatch, tmp_path):
            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
            row = _stage_and_replay_update(
                provider, db_path, seed_id, {"importance": UPDATED_IMPORTANCE}
            )
            # Only importance was targeted: content must be untouched
            # (before the fix, staging injected content="" and the replay
            # blanked the row).
            assert row[0] == seed_id
            assert row[1] == SEED_CONTENT
            assert row[4] == pytest.approx(UPDATED_IMPORTANCE)
    # Direct apply_beam_batch (gate OFF) must leave the identical row for
    # the same partial update.
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: False)
    with _make_provider(module) as (provider2, db_path2):
        seed2 = json.loads(provider2._handle_remember(
            {"content": SEED_CONTENT, "importance": SEED_IMPORTANCE}
        ))
        direct = json.loads(provider2._handle_batch({"operations": [
            {"action": "update", "memory_id": seed2["memory_id"],
             "importance": UPDATED_IMPORTANCE},
        ]}))
        assert direct["status"] == "ok"
        rows2 = _wm_rows(db_path2)
        assert len(rows2) == 1
        # Approval/replay path and direct path produce the same final row.
        assert rows2[0][1] == row[1] == SEED_CONTENT
        assert rows2[0][4] == pytest.approx(row[4]) == pytest.approx(UPDATED_IMPORTANCE)


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_apply_pending_update_content_only_preserves_importance(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        # Seed a memory with non-default importance while the approval
        # gate is OFF so it commits directly.
        seed = json.loads(provider._handle_remember(
            {"content": SEED_CONTENT, "importance": SEED_IMPORTANCE}
        ))
        seed_id = seed["memory_id"]
        rows = _wm_rows(db_path)
        assert len(rows) == 1
        assert rows[0][1] == SEED_CONTENT
        assert rows[0][4] == pytest.approx(SEED_IMPORTANCE)
        with _approval_setup(monkeypatch, tmp_path):
            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
            row = _stage_and_replay_update(
                provider, db_path, seed_id, {"content": UPDATED_CONTENT}
            )
            # Only content was targeted: importance must be untouched
            # (before the fix, staging injected importance=0.5 and the
            # replay reset the row's non-default importance).
            assert row[0] == seed_id
            assert row[1] == UPDATED_CONTENT
            assert row[4] == pytest.approx(SEED_IMPORTANCE)
    # Direct apply_beam_batch (gate OFF) must leave the identical row for
    # the same partial update.
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: False)
    with _make_provider(module) as (provider2, db_path2):
        seed2 = json.loads(provider2._handle_remember(
            {"content": SEED_CONTENT, "importance": SEED_IMPORTANCE}
        ))
        direct = json.loads(provider2._handle_batch({"operations": [
            {"action": "update", "memory_id": seed2["memory_id"],
             "content": UPDATED_CONTENT},
        ]}))
        assert direct["status"] == "ok"
        rows2 = _wm_rows(db_path2)
        assert len(rows2) == 1
        # Approval/replay path and direct path produce the same final row.
        assert rows2[0][1] == row[1] == UPDATED_CONTENT
        assert rows2[0][4] == pytest.approx(row[4]) == pytest.approx(SEED_IMPORTANCE)


# ---------------------------------------------------------------------------
# 7. Finding 7 regression (PR #926 CodeRabbit comment 3962350003): the
#    'staged' key must carry RAW pending IDs (strings) on BOTH provider
#    surfaces, so a client can forward response['staged'] verbatim to
#    mnemosyne_apply_pending. Before the fix the standalone surface
#    returned [{"action": ..., "pending_id": ...}] dicts under 'staged',
#    which _handle_apply_pending rejects as "invalid: empty", while the
#    legacy surface had no 'staged' key at all. Action metadata lives in
#    the additive 'staged_actions' field, present on both surfaces.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_staged_contains_raw_pending_ids_forwardable(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        with _approval_setup(monkeypatch, tmp_path) as pending_dir:
            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "remember", "content": "forward staged verbatim"},
                {"action": "remember", "content": "second staged write"},
            ]}))
            assert resp["status"] == "staged"
            staged = resp["staged"]
            # Contract: 'staged' is a list of raw string pending IDs on
            # both provider surfaces (before the fix the standalone
            # surface returned dict objects here).
            assert isinstance(staged, list)
            assert len(staged) == 2
            assert all(isinstance(pid, str) and pid for pid in staged)
            # Every staged ID resolves to a real pending record.
            for pid in staged:
                assert (pending_dir / f"{pid}.json").is_file()
            # Legacy surface keeps its historical pending_ids alias in
            # sync with the unified 'staged' key.
            if "pending_ids" in resp:
                assert resp["pending_ids"] == staged
            if "staged_count" in resp:
                assert resp["staged_count"] == len(staged)

            # The client contract: forward response['staged'] verbatim to
            # mnemosyne_apply_pending. Before the fix this raised
            # invalid:empty on the standalone surface (dicts) and
            # KeyError on the legacy surface (no 'staged' key).
            applied = json.loads(provider._handle_apply_pending({"pending_ids": staged}))
            assert applied["applied_count"] == 2
            assert applied["failed_count"] == 0
            rows = _wm_rows(db_path)
            assert len(rows) == 2


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_staged_actions_metadata_present_on_both_surfaces(provider_module_name, monkeypatch, tmp_path):
    module = _import_provider(provider_module_name)
    with _make_provider(module) as (provider, db_path):
        with _approval_setup(monkeypatch, tmp_path):
            monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
            resp = json.loads(provider._handle_batch({"operations": [
                {"action": "remember", "content": "meta one"},
                {"action": "forget", "memory_id": "00000000-dead-beef"},
            ]}))
            assert resp["status"] == "staged"
            staged = resp["staged"]
            actions = resp["staged_actions"]
            # Additive metadata field: parallel list of {action, pending_id}
            # dicts, present on BOTH surfaces, never overloading 'staged'.
            assert isinstance(actions, list)
            assert len(actions) == len(staged)
            for entry, pid in zip(actions, staged):
                assert entry["pending_id"] == pid
                assert entry["action"] in ("remember", "forget")
            assert [e["action"] for e in actions] == ["remember", "forget"]
