"""Tests for lazy import self-healing in mnemosyne.core.embeddings.

Root cause (already proven): ``np`` and ``TextEmbedding`` are bound at import
time, and ``_FASTEMBED_AVAILABLE`` is evaluated once at import. When
``mnemosyne.core.embeddings`` is imported during a window where the dense stack
is unreachable (the Hermes plugin-loader import window, or a venv without the
embeddings extra), both globals bind to None and ``available()`` returns False
for the whole process lifetime, even after the stack becomes importable.

These tests pin four behaviours:
1. ``available()`` re-attempts the import when the globals are empty, so dense
   retrieval self-heals without a reload;
2. a failing import logs exactly one warning naming the failing module rather
   than degrading silently;
3. ``embed_query()`` never memoises a failure — during an outage it returns
   ``None`` without touching the cache, so no stale entry can outlive recovery;
4. if an entry did get cached by an older revision, recovery drops it.

The tests install stub ``numpy``/``fastembed`` modules, so they do NOT require
the optional ``embeddings`` extra to run — the retry and cache behaviour is what
is under test, not real embedding quality.
"""
from __future__ import annotations

import builtins
import fnmatch
import importlib
import logging
import os
import sys
import types
from functools import lru_cache

import pytest

from mnemosyne.core import embeddings

_DENSE_MODULES = ("numpy", "fastembed")

# Ambient settings that would change which branch of embed_query() runs.
_ISOLATED_ENV = (
    "MNEMOSYNE_NO_EMBEDDINGS",
    "MNEMOSYNE_SKIP_EMBEDDINGS",
    "MNEMOSYNE_EMBEDDINGS_OFF",
    "MNEMOSYNE_EMBEDDING_API_URL",
    "MNEMOSYNE_EMBEDDING_API_KEY",
    "MNEMOSYNE_EMBEDDINGS_VIA_API",
)


def _isolate_config(monkeypatch, tmp_path):
    """Neutralise ambient embedding config and point the cache at tmp_path.

    Call this BEFORE a reload and again AFTER it: ``importlib.reload`` re-runs
    the module body, which re-computes ``_FASTEMBED_CACHE_DIR`` from the
    environment and would otherwise undo the patch (and let a stub model write
    into the real home directory).
    """
    for key in _ISOLATED_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(embeddings, "_FASTEMBED_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture
def isolated_embeddings(monkeypatch, tmp_path):
    """Undo every test patch before rebuilding the shared module globals."""
    try:
        with monkeypatch.context() as patch:
            _isolate_config(patch, tmp_path)
            yield patch
    finally:
        importlib.reload(embeddings)


def _site_package_entries():
    return [
        p for p in sys.path
        if fnmatch.fnmatch(os.path.basename(p.rstrip(os.sep)), "*-packages")
    ]


def _make_blocked_import(original_import):
    def _blocked_import(name, *args, **kwargs):
        if name in _DENSE_MODULES or name.split(".")[0] in _DENSE_MODULES:
            raise ModuleNotFoundError(
                f"No module named '{name}' (simulated dense-stack import window)",
                name=name,
            )
        return original_import(name, *args, **kwargs)
    return _blocked_import


class _StubVector(list):
    """Minimal stand-in for ``numpy.ndarray`` on the query path."""

    def astype(self, dtype):
        return self


class _StubTextEmbedding:
    """Produces deterministic vectors without loading a real model."""

    def __init__(self, *args, **kwargs):
        pass

    def embed(self, texts):
        return [_StubVector([0.1, 0.2, 0.3]) for _ in texts]


def _install_stub_dense_stack():
    """Register stub ``numpy``/``fastembed`` modules and return the previous entries.

    Using stubs keeps the recovery assertions independent of whether the optional
    ``embeddings`` extra is installed, which is the state a CI run without it is
    in. The previous entries (possibly None) are restored by the caller.
    """
    np_stub = types.ModuleType("numpy")
    setattr(np_stub, "__version__", "0.0.0-stub")
    setattr(np_stub, "float32", "float32")

    fastembed_stub = types.ModuleType("fastembed")
    setattr(fastembed_stub, "TextEmbedding", _StubTextEmbedding)

    previous = {name: sys.modules.get(name) for name in _DENSE_MODULES}
    sys.modules["numpy"] = np_stub
    sys.modules["fastembed"] = fastembed_stub
    return previous, (np_stub, _StubTextEmbedding)


def _restore_stub_dense_stack(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def test_available_self_heals_after_dense_stack_returns(isolated_embeddings, tmp_path):
    monkeypatch = isolated_embeddings

    query = "mnemosyne lazy import cache probe"
    original_path = list(sys.path)
    original_import = builtins.__import__
    previous_modules = None
    try:
        # Simulate the import window: the dense stack cannot be imported at all.
        # Blocking the import (rather than ejecting loaded C extensions) keeps
        # re-import safe: deleting an already-loaded extension makes a later
        # import fail with "cannot load module more than once per process".
        builtins.__import__ = _make_blocked_import(original_import)
        importlib.reload(embeddings)
        _isolate_config(monkeypatch, tmp_path)  # reload re-computed the defaults

        assert embeddings.available() is False
        assert embeddings.np is None
        assert embeddings.TextEmbedding is None
        assert embeddings._FASTEMBED_AVAILABLE is False

        # A query taken during the outage returns None and caches NOTHING:
        # memoising the failure is what let a stale None outlive recovery.
        assert embeddings.embed_query(query) is None
        cached_before = embeddings._embed_query_cached.cache_info()
        assert cached_before.currsize == 0, cached_before

        # The stack becomes importable again. With the real extra installed the
        # bindings come from it; without it (CI, minimal install) the stubs
        # stand in. Either way the retry happens without a reload.
        previous_modules, (stub_np, stub_text_embedding) = _install_stub_dense_stack()
        builtins.__import__ = original_import

        assert embeddings.available() is True
        assert embeddings.np is stub_np
        assert embeddings.TextEmbedding is stub_text_embedding
        assert embeddings._FASTEMBED_AVAILABLE is False  # frozen alias, unused

        # The same query now produces a vector, and only that success is cached.
        assert embeddings.embed_query(query) is not None
        recomputed = embeddings._embed_query_cached.cache_info()
        assert recomputed.misses == 1, recomputed
        assert recomputed.hits == 0, recomputed
        assert recomputed.currsize == 1, recomputed
    finally:
        builtins.__import__ = original_import
        sys.path[:] = original_path
        if previous_modules is not None:
            _restore_stub_dense_stack(previous_modules)


def test_recovery_clears_a_preexisting_cached_failure(isolated_embeddings, tmp_path):
    """Defence in depth: an entry cached by an older revision is dropped.

    embed_query() no longer caches failures, but a process upgraded in place can
    already hold one. ``_ensure_fastembed()`` must clear the cache when the stack
    first binds.
    """
    monkeypatch = isolated_embeddings

    calls = {"n": 0}

    @lru_cache(maxsize=8)
    def _fake_cached(prefixed, _caps=(0, 0), api_url=None):
        calls["n"] += 1
        return None

    monkeypatch.setattr(embeddings, "_embed_query_cached", _fake_cached)
    _fake_cached("warm", (0, 0), None)
    assert _fake_cached.cache_info().currsize == 1

    # Force the "stack is unbound, then becomes importable" transition.
    monkeypatch.setattr(embeddings, "np", None)
    monkeypatch.setattr(embeddings, "TextEmbedding", None)
    monkeypatch.setattr(embeddings, "_DENSE_STACK_IMPORTED", False)

    previous_modules = None
    try:
        previous_modules, _ = _install_stub_dense_stack()
        importlib.invalidate_caches()
        assert embeddings._is_fastembed_available() is True
        assert _fake_cached.cache_info().currsize == 0
    finally:
        if previous_modules is not None:
            _restore_stub_dense_stack(previous_modules)


def test_import_failure_warns_exactly_once(isolated_embeddings, caplog, tmp_path):
    original_path = list(sys.path)
    original_import = builtins.__import__
    try:
        builtins.__import__ = _make_blocked_import(original_import)
        caplog.set_level(logging.WARNING, logger="mnemosyne.core.embeddings")
        importlib.reload(embeddings)
        # Call repeatedly to prove the warning budget is per-process, not per-call.
        results = [embeddings.available() for _ in range(5)]
        assert results == [False] * 5
    finally:
        builtins.__import__ = original_import
        sys.path[:] = original_path

    records = [
        r for r in caplog.records
        if r.name == "mnemosyne.core.embeddings" and r.levelno >= logging.WARNING
    ]
    assert len(records) == 1, records
    msg = records[0].getMessage()
    assert "numpy" in msg or "fastembed" in msg, msg


def test_embed_query_recovers_without_calling_available(isolated_embeddings, tmp_path):
    """A direct caller never touches available(); embed_query() must retry anyway."""
    monkeypatch = isolated_embeddings

    query = "mnemosyne direct embed_query recovery probe"
    original_path = list(sys.path)
    original_import = builtins.__import__
    previous_modules = None
    try:
        builtins.__import__ = _make_blocked_import(original_import)
        importlib.reload(embeddings)
        _isolate_config(monkeypatch, tmp_path)

        assert embeddings.embed_query(query) is None
        assert embeddings.embed_query(query) is None
        # Neither call memoised a failure.
        assert embeddings._embed_query_cached.cache_info().currsize == 0

        previous_modules, _ = _install_stub_dense_stack()
        builtins.__import__ = original_import

        # No available() call in between: this must still recover.
        assert embeddings.embed_query(query) is not None
    finally:
        builtins.__import__ = original_import
        sys.path[:] = original_path
        if previous_modules is not None:
            _restore_stub_dense_stack(previous_modules)
