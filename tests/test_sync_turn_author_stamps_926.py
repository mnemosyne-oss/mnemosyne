"""Regression tests for sync_turn() author attribution on both provider surfaces.

Maintainer repro on PR #926 (dplush review 5254704471, issue comment
5739583178): the primary automatic capture path was still unattributed.
``sync_turn()`` in both providers called ``remember()`` with
content/source/importance/scope/extract_entities only -- no
``author_id``/``author_type`` and no ``**self._write_identity_kwargs()``.
With ``agent_identity='sphinx'`` and ``MNEMOSYNE_AUTHOR_ID='env-author'``
set, the stored working rows came back with
``(author_id, author_type) = (NULL, NULL)``.

Both provider copies must resolve the write identity through
``_write_identity_kwargs()`` and pass it through each ``remember()`` call in
``sync_turn()`` -- same narrow per-write path the identity-signal capture
already uses. The Beam read identity stays unset so prefetch recall remains
session/channel scoped.

These tests drive real BeamMemory rows (temp DB) rather than a mocked beam,
so they exercise the actual write path end to end on both surfaces.
"""

from __future__ import annotations

import importlib
import sqlite3
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"

ENV_AUTHOR_ID = "env-author"
ENV_AUTHOR_TYPE = "profile"
AGENT_IDENTITY = "sphinx"

USER_TEXT = "sync turn user text that must be attributed"
ASSISTANT_TEXT = "sync turn assistant text that must be attributed"


def _wm_row(db_path, content):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT author_id, author_type FROM working_memory WHERE content = ?",
            (content,),
        ).fetchone()
        return tuple(row) if row is not None else None
    finally:
        conn.close()


@pytest.fixture(autouse=True)
def _restore_provider_modules():
    """Restore the pre-test module identities after each test.

    ``_import_provider`` re-imports the provider packages and the real
    ``mnemosyne`` core (the repo-root ``__init__.py`` stub shadows the core
    under pytest importlib mode). Other test modules bind their imports at
    collection time, so swapping the ``mnemosyne.*`` namespace would orphan
    those references. Snapshot every touched module and restore the whole
    namespace afterwards.
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
    """Construct a minimal provider + real BeamMemory in a temp DB."""
    beam_cls = getattr(module, "_BEAM_CLS")  # set by _import_provider
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "mnemosyne.db"
        provider = module.MnemosyneMemoryProvider()
        provider._beam = beam_cls(session_id="hermes_sync-stamps", db_path=db_path)
        provider._default_scope = "session"
        provider._agent_context = "primary"
        provider._skip_contexts = set()
        provider._sync_roles = {"user", "assistant"}
        provider._auto_sleep_enabled = False
        try:
            yield provider, db_path
        finally:
            provider._beam.conn.close()


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
# sync_turn() write path (both implementations)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_sync_turn_stamps_agent_identity_on_stored_rows(
    provider_module_name, monkeypatch
):
    """Maintainer repro: agent_identity wins over the env fallback.

    With agent_identity='sphinx' AND MNEMOSYNE_AUTHOR_ID='env-author' set,
    the conversation capture must store rows stamped 'sphinx' -- not
    (NULL, NULL), and not the env fallback.
    """
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", ENV_AUTHOR_ID)
    monkeypatch.delenv("MNEMOSYNE_AUTHOR_TYPE", raising=False)
    with _make_provider(module) as (provider, db_path):
        provider._agent_identity = AGENT_IDENTITY
        provider.sync_turn(USER_TEXT, ASSISTANT_TEXT, session_id="stamps")

        user_row = _wm_row(db_path, f"[USER] {USER_TEXT}")
        assert user_row is not None, "user capture did not reach working_memory"
        assert user_row == (AGENT_IDENTITY, None), (
            f"{provider_module_name}: sync_turn user row carries {user_row!r}; "
            "expected the resolved agent identity"
        )
        assistant_row = _wm_row(db_path, f"[ASSISTANT] {ASSISTANT_TEXT}")
        assert assistant_row is not None, (
            "assistant capture did not reach working_memory"
        )
        assert assistant_row == (AGENT_IDENTITY, None), (
            f"{provider_module_name}: sync_turn assistant row carries "
            f"{assistant_row!r}; expected the resolved agent identity"
        )
        # Read identity stayed unset: prefetch must remain session-scoped.
        assert provider._beam.author_id is None
        assert provider._beam.author_type is None


@pytest.mark.parametrize("provider_module_name", [
    "hermes_memory_provider",
    "mnemosyne_hermes",
])
def test_sync_turn_stamps_env_author_without_agent_identity(
    provider_module_name, monkeypatch
):
    """No agent_identity: the env author still stamps both captured rows."""
    module = _import_provider(provider_module_name)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", ENV_AUTHOR_ID)
    monkeypatch.setenv("MNEMOSYNE_AUTHOR_TYPE", ENV_AUTHOR_TYPE)
    with _make_provider(module) as (provider, db_path):
        provider.sync_turn(USER_TEXT, ASSISTANT_TEXT, session_id="stamps")

        user_row = _wm_row(db_path, f"[USER] {USER_TEXT}")
        assert user_row is not None, "user capture did not reach working_memory"
        assert user_row == (ENV_AUTHOR_ID, ENV_AUTHOR_TYPE), (
            f"{provider_module_name}: sync_turn user row carries {user_row!r}; "
            "expected the resolved env author identity"
        )
        assistant_row = _wm_row(db_path, f"[ASSISTANT] {ASSISTANT_TEXT}")
        assert assistant_row is not None, (
            "assistant capture did not reach working_memory"
        )
        assert assistant_row == (ENV_AUTHOR_ID, ENV_AUTHOR_TYPE), (
            f"{provider_module_name}: sync_turn assistant row carries "
            f"{assistant_row!r}; expected the resolved env author identity"
        )
        # Read identity stayed unset: prefetch must remain session-scoped.
        assert provider._beam.author_id is None
        assert provider._beam.author_type is None
