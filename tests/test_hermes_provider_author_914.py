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
                 "author_id": "op-author-a", "author_type": "agent"},
                {"action": "remember", "content": "batch write two",
                 "author_id": "op-author-b", "author_type": "profile"},
            ],
        }))
    assert payload["status"] == "staged"
    assert len(staged_payloads) == 2
    assert staged_payloads[0]["author_id"] == "op-author-a"
    assert staged_payloads[0]["author_type"] == "agent"
    assert staged_payloads[1]["author_id"] == "op-author-b"
    assert staged_payloads[1]["author_type"] == "profile"


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_batch_staging_falls_back_to_batch_default_author(provider_module_name, monkeypatch):
    """Ops without their own author fall back to the batch/env default."""
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
                {"action": "remember", "content": "batch write one"},
            ],
        }))
    assert payload["status"] == "staged"
    assert staged_payloads[0]["author_id"] == HERMES_AUTHOR_ID
    assert staged_payloads[0]["author_type"] == HERMES_AUTHOR_TYPE


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
