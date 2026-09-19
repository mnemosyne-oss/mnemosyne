"""Regression tests for issue #914 — Hermes provider per-write author stamps.

The provider surfaces (legacy ``hermes_memory_provider`` and standalone
``mnemosyne_hermes``) must route author identity into their writes the way
the MCP surface does: tool-arg > provider identity (agent_identity > env) >
None, passed via ``remember(..., author_id=...)``, with ``beam.author_id``
left unset so automatic prefetch recall stays session-scoped. Batch-level
default author stamps every operation that does not carry its own.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"

HERMES_AUTHOR_ID = "hermes-test-author"
HERMES_AUTHOR_TYPE = "profile"


def _wm_row(db_path, memory_id):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT author_id, author_type FROM working_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        conn.close()


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
        provider._beam = beam_cls(session_id="hermes_auth-test", db_path=db_path)
        provider._default_scope = "session"
        provider._agent_context = "primary"
        provider._skip_contexts = set()
        try:
            yield provider, db_path
        finally:
            provider._beam.conn.close()


# ---------------------------------------------------------------------------
# Provider-surface write path (both implementations)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_remember_env_author_stamps_row(provider_module_name, monkeypatch):
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", HERMES_AUTHOR_TYPE)
    with _make_provider(module) as (provider, db_path):
        payload = json.loads(provider._handle_remember({"content": "stamped write"}))
        mid = payload["memory_id"]
        assert _wm_row(db_path, mid) == (HERMES_AUTHOR_ID, HERMES_AUTHOR_TYPE)
        # Read identity stayed unset: prefetch must remain session-scoped.
        assert provider._beam.author_id is None
        assert provider._beam.author_type is None


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_remember_tool_arg_author_wins_over_env(provider_module_name, monkeypatch):
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    with _make_provider(module) as (provider, db_path):
        payload = json.loads(provider._handle_remember({
            "content": "explicit author write",
            "author_id": "per-call-author",
            "author_type": "agent",
        }))
        mid = payload["memory_id"]
        assert _wm_row(db_path, mid) == ("per-call-author", "agent")


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_remember_batch_env_author_stamps_rows(provider_module_name, monkeypatch):
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", HERMES_AUTHOR_TYPE)
    with _make_provider(module) as (provider, db_path):
        payload = json.loads(provider._handle_batch({
            "operations": [
                {"action": "remember", "content": "batch write one"},
                {"action": "remember", "content": "batch write two"},
            ],
        }))
        assert payload["status"] == "ok"
        ids = [r["memory_id"] for r in payload["results"]]
        assert _wm_row(db_path, ids[0]) == (HERMES_AUTHOR_ID, HERMES_AUTHOR_TYPE)
        assert _wm_row(db_path, ids[1]) == (HERMES_AUTHOR_ID, HERMES_AUTHOR_TYPE)


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_remember_batch_per_operation_author_wins(provider_module_name, monkeypatch):
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    with _make_provider(module) as (provider, db_path):
        payload = json.loads(provider._handle_batch({
            "operations": [
                {"action": "remember", "content": "batch write one",
                 "author_id": "op-author", "author_type": "agent"},
            ],
        }))
        assert payload["status"] == "ok"
        mid = payload["results"][0]["memory_id"]
        assert _wm_row(db_path, mid) == ("op-author", "agent")


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_batch_staging_preserves_per_operation_authors(provider_module_name, monkeypatch):
    """#914 regression: per-op author stamps survive approval staging.

    validate_batch_operations() normalizes each op to {index, action,
    payload}; the staging loop must read author fields from op["payload"],
    not the top-level op, or every op silently falls back to the
    batch/env default.

    #926 (CodeRabbit F1/F4): the same op-vs-payload lookup bug silently
    emptied every other staged field (content '', importance 0.5 ...), and
    _handle_apply_pending then deleted the record without writing memory.
    Asserting only the author fields could not catch it, so every staged
    field here is asserted against a NON-default value.
    """
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", HERMES_AUTHOR_TYPE)
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
    staged_payloads = []
    monkeypatch.setattr(
        module,
        "_stage_pending_write",
        lambda payload: staged_payloads.append(payload) or f"pid-{len(staged_payloads)}",
    )
    with _make_provider(module) as (provider, _db_path):
        payload = json.loads(provider._handle_batch({
            "operations": [
                {"action": "remember", "content": "batch write one",
                 "importance": 0.91, "source": "op-source-a",
                 "scope": "global", "metadata": {"op": "a"},
                 "veracity": "stated", "valid_until": "2030-01-01",
                 "author_id": "op-author-a", "author_type": "agent"},
                {"action": "remember", "content": "batch write two",
                 "importance": 0.23, "source": "op-source-b",
                 "scope": "session", "metadata": {"op": "b"},
                 "veracity": "inferred", "valid_until": "2030-02-02",
                 "author_id": "op-author-b", "author_type": "profile"},
            ],
        }))
    assert payload["status"] == "staged"
    assert len(staged_payloads) == 2
    assert staged_payloads[0]["author_id"] == "op-author-a"
    assert staged_payloads[0]["author_type"] == "agent"
    assert staged_payloads[1]["author_id"] == "op-author-b"
    assert staged_payloads[1]["author_type"] == "profile"
    # F1: every op field must come from payload, not the top-level op.
    assert staged_payloads[0]["content"] == "batch write one"
    assert staged_payloads[0]["importance"] == 0.91
    assert staged_payloads[0]["source"] == "op-source-a"
    assert staged_payloads[0]["scope"] == "global"
    assert staged_payloads[0]["metadata"] == {"op": "a"}
    assert staged_payloads[0]["veracity"] == "stated"
    assert staged_payloads[0]["valid_until"] == "2030-01-01"
    assert staged_payloads[1]["content"] == "batch write two"
    assert staged_payloads[1]["importance"] == 0.23
    assert staged_payloads[1]["source"] == "op-source-b"
    assert staged_payloads[1]["scope"] == "session"
    assert staged_payloads[1]["metadata"] == {"op": "b"}
    assert staged_payloads[1]["veracity"] == "inferred"
    assert staged_payloads[1]["valid_until"] == "2030-02-02"


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_batch_staging_falls_back_to_batch_default_author(provider_module_name, monkeypatch):
    """Ops without their own author fall back to the batch/env default.

    #926 (CodeRabbit F1/F4): asserting a non-default importance here as
    well — a default-only assertion cannot distinguish a payload lookup
    from an op-level lookup miss.
    """
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", HERMES_AUTHOR_TYPE)
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
    staged_payloads = []
    monkeypatch.setattr(
        module,
        "_stage_pending_write",
        lambda payload: staged_payloads.append(payload) or "pid-0",
    )
    with _make_provider(module) as (provider, _db_path):
        payload = json.loads(provider._handle_batch({
            "operations": [
                {"action": "remember", "content": "batch write one",
                 "importance": 0.77},
            ],
        }))
    assert payload["status"] == "staged"
    assert staged_payloads[0]["author_id"] == HERMES_AUTHOR_ID
    assert staged_payloads[0]["author_type"] == HERMES_AUTHOR_TYPE
    assert staged_payloads[0]["content"] == "batch write one"
    assert staged_payloads[0]["importance"] == 0.77


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_shared_remember_env_author_stamps_surface_row(provider_module_name, monkeypatch):
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    with _make_provider(module) as (provider, db_path):
        shared_path = Path(db_path).parent / "shared" / "mnemosyne.db"
        provider._shared_surface_path = shared_path
        provider._ensure_surface_beam()
        payload = json.loads(provider._handle_shared_remember({
            "content": "shared stamped write",
            "kind": "preference",
        }))
        mid = payload["memory_id"]
        assert _wm_row(shared_path, mid) == (HERMES_AUTHOR_ID, None)
        provider._surface_beam.conn.close()


# ---------------------------------------------------------------------------
# Shared-surface per-write author stamping (cross-agent metadata writes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_shared_remember_tool_arg_author_wins_over_env(provider_module_name, monkeypatch):
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    with _make_provider(module) as (provider, db_path):
        shared_path = Path(db_path).parent / "shared" / "mnemosyne.db"
        provider._shared_surface_path = shared_path
        provider._ensure_surface_beam()
        payload = json.loads(provider._handle_shared_remember({
            "content": "shared explicit author write",
            "kind": "preference",
            "author_id": "per-call-author",
            "author_type": "agent",
        }))
        mid = payload["memory_id"]
        assert _wm_row(shared_path, mid) == ("per-call-author", "agent")
        provider._surface_beam.conn.close()


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_batch_staging_then_apply_pending_commits_content(
    provider_module_name, monkeypatch, tmp_path
):
    """#926 (CodeRabbit F1) — the data-loss consequence, end to end.

    At the reviewed head the staging loop read op-level fields, so every
    staged batch write carried content='' / importance=0.5. On replay
    ``_handle_apply_pending`` treats empty content as a dead record:
    it records "empty content" and DELETES the file without writing
    memory. This drives the real stage -> apply cycle against a temp
    HERMES_HOME (only ``get_hermes_home`` is faked) and asserts the
    memory actually lands.
    """
    import types

    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", HERMES_AUTHOR_TYPE)
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
    fake_constants = types.ModuleType("hermes_constants")
    setattr(fake_constants, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setitem(sys.modules, "hermes_constants", fake_constants)

    with _make_provider(module) as (provider, db_path):
        staged = json.loads(provider._handle_batch({
            "operations": [
                {"action": "remember", "content": "staged round trip content",
                 "importance": 0.91, "author_id": "op-author",
                 "author_type": "agent"},
            ],
        }))
        assert staged["status"] == "staged", staged
        pending_id = (
            staged["pending_ids"][0]
            if "pending_ids" in staged
            else staged["staged"][0]["pending_id"]
        )

        applied = json.loads(provider._handle_apply_pending({
            "pending_ids": [pending_id],
        }))
        assert applied["failed_count"] == 0, applied
        assert applied["applied_count"] == 1, applied
        mid = applied["applied"][0]["memory_id"]

        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT content, importance, author_id, author_type "
                "FROM working_memory WHERE id = ?",
                (mid,),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None, "apply_pending claimed success but wrote no row"
        assert tuple(row) == (
            "staged round trip content", 0.91, "op-author", "agent",
        ), tuple(row)


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_batch_staging_carries_memory_id_for_id_based_ops(
    provider_module_name, monkeypatch
):
    """#926 (CodeRabbit F1): id-based ops keep their memory_id when staged.

    The integration dict read ``op.get("memory_id")`` while the standalone
    dict did not carry the key at all; both must read it from payload so
    the staged record preserves the target id for the approval replay.
    """
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
    staged_payloads = []
    monkeypatch.setattr(
        module,
        "_stage_pending_write",
        lambda payload: staged_payloads.append(payload) or "pid-0",
    )
    with _make_provider(module) as (provider, _db_path):
        payload = json.loads(provider._handle_batch({
            "operations": [
                {"action": "update", "memory_id": "wm-target-1",
                 "content": "updated content", "importance": 0.42},
            ],
        }))
    assert payload["status"] == "staged"
    assert staged_payloads[0]["memory_id"] == "wm-target-1"
    assert staged_payloads[0]["content"] == "updated content"
    assert staged_payloads[0]["importance"] == 0.42


def _import_provider(package: str):
    """Import a provider package from its own source root, mirroring the
    module-swap pattern in test_hermes_provider_parity.py.

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


# ---------------------------------------------------------------------------
# Mirror-write parity: on_memory_write (builtin memory tool) on both surfaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
@pytest.mark.parametrize("target", ["user", "session"])
def test_on_memory_write_mirror_stamps_author(provider_module_name, target, monkeypatch):
    """The builtin-memory mirror write must carry the author on BOTH surfaces.

    CodeRabbit review finding on PR #926 (comment 4040483068's sibling, review
    5230504707): the sibling provider forwarded its write-identity kwargs in
    ``on_memory_write`` while ``mnemosyne_hermes`` did not, so the same
    ``builtin_memory_*`` write was stamped on one fork and NULL on the other.
    Parametrized over both targets (user -> global, session -> session) so the
    stamp is asserted on every branch of the scope mapping.
    """
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", HERMES_AUTHOR_ID)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", HERMES_AUTHOR_TYPE)
    with _make_provider(module) as (provider, db_path):
        label = f"mirror write {provider_module_name} {target}"
        provider.on_memory_write("add", target, label)
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute(
                "SELECT author_id, author_type FROM working_memory WHERE content = ?",
                (label,),
            ).fetchone()
        finally:
            conn.close()
        assert row is not None, "mirror write did not reach working_memory"
        assert tuple(row) == (HERMES_AUTHOR_ID, HERMES_AUTHOR_TYPE), (
            f"{provider_module_name}: mirror write lost its author stamp: {tuple(row)!r}"
        )
        # Read identity must stay unset so prefetch remains session-scoped.
        assert provider._beam.author_id is None
        assert provider._beam.author_type is None
