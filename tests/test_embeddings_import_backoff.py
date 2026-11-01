"""Dense dependency recovery without repeated broken-package initialization."""
from __future__ import annotations

import builtins
import importlib.util
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
from types import SimpleNamespace

import pytest

from mnemosyne.core import embeddings


@pytest.fixture
def dense_imports(monkeypatch):
    """Replace only optional imports; run the real import/recovery coordinator."""
    for name, value in (("np", None), ("TextEmbedding", None),
                        ("_DENSE_STACK_IMPORTED", False),
                        ("_DENSE_STACK_WARNED", False), ("_IMPORTING", False)):
        monkeypatch.setattr(embeddings, name, value)
    monkeypatch.setattr(embeddings, "_DENSE_STACK_RETRY_AT", 0.0, raising=False)
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(embeddings, "time", SimpleNamespace(monotonic=lambda: clock.now))
    numpy = SimpleNamespace()
    text_embedding = type("TextEmbedding", (), {})
    fastembed = SimpleNamespace(TextEmbedding=text_embedding)
    state = SimpleNamespace(failing="numpy", error=RuntimeError("broken wheel"),
                            attempts=[], clock=clock, numpy=numpy,
                            text_embedding=text_embedding)
    original_import = builtins.__import__

    def import_optional(name, *args, **kwargs):
        if name in ("numpy", "fastembed"):
            state.attempts.append(name)
            if name == state.failing:
                raise state.error
            return numpy if name == "numpy" else fastembed
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_optional)
    return state


@pytest.mark.parametrize("module", ["numpy", "fastembed"])
@pytest.mark.parametrize("error_type", [RuntimeError, ImportError])
def test_broken_import_backs_off_then_recovers(dense_imports, module, error_type):
    """Removing either exception branch's deadline repeats costly imports."""
    state = dense_imports
    state.failing = module
    state.error = error_type("broken wheel")
    assert embeddings._ensure_fastembed() is False
    first_attempts = list(state.attempts)
    for _ in range(4):
        assert embeddings._ensure_fastembed() is False
    assert state.attempts == first_attempts
    assert embeddings.np is None
    assert embeddings.TextEmbedding is None

    # Recovery waits only for the bounded deadline, without sleeping.
    state.failing = None
    state.clock.now = 100.5
    assert embeddings._ensure_fastembed() is False
    state.clock.now = 101.0
    assert embeddings._ensure_fastembed() is True
    assert embeddings.np is state.numpy
    assert embeddings.TextEmbedding is state.text_embedding
    assert embeddings._DENSE_STACK_RETRY_AT == 0.0


@pytest.mark.parametrize("module", ["numpy", "fastembed"])
def test_missing_dependency_is_immediately_retryable(dense_imports, module):
    """Backing off ModuleNotFoundError would delay recovery at the same clock time."""
    state = dense_imports
    state.failing = module
    state.error = ModuleNotFoundError("dependency absent", name=module)
    assert embeddings._ensure_fastembed() is False
    state.failing = None
    assert embeddings._ensure_fastembed() is True


@pytest.mark.parametrize("module", ["numpy", "fastembed"])
def test_failure_names_module_and_recovery_restores_warning_budget(
    dense_imports, monkeypatch, caplog, module,
):
    """Wrong attribution, early partial binding, or a stale warning flag fails."""
    state = dense_imports
    state.failing = module
    caplog.set_level(logging.WARNING, logger=embeddings.__name__)
    assert embeddings._ensure_fastembed() is False
    assert embeddings.np is None
    assert embeddings.TextEmbedding is None
    assert len(caplog.records) == 1
    assert f"could not import {module}:" in caplog.records[0].getMessage()
    assert "broken wheel" in caplog.records[0].getMessage()

    state.clock.now = 101.0
    state.failing = None
    assert embeddings._ensure_fastembed() is True
    monkeypatch.setattr(embeddings, "np", None)
    monkeypatch.setattr(embeddings, "TextEmbedding", None)
    state.failing = module
    assert embeddings._ensure_fastembed() is False
    assert len(caplog.records) == 2
    assert f"could not import {module}:" in caplog.records[1].getMessage()


@pytest.mark.parametrize("error_type, expected_attempts", [(RuntimeError, 1), (ModuleNotFoundError, 8)])
def test_concurrent_available_warns_once(
    dense_imports, monkeypatch, caplog, error_type, expected_attempts,
):
    """Both retry policies emit one warning to simultaneous callers."""
    dense_imports.error = error_type("unavailable dependency", name="numpy") if error_type is ModuleNotFoundError else error_type("unavailable dependency")
    monkeypatch.setattr(embeddings, "_is_disabled", lambda: False)
    monkeypatch.setattr(embeddings, "_is_api_model", lambda _: False)
    caplog.set_level(logging.WARNING, logger=embeddings.__name__)
    barrier = threading.Barrier(8)

    def check_available():
        barrier.wait(timeout=5)
        return embeddings.available()

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda _: check_available(), range(8))) == [False] * 8
    assert dense_imports.attempts == ["numpy"] * expected_attempts
    assert len(caplog.records) == 1


@pytest.mark.parametrize("module", ["numpy", "fastembed"])
def test_import_time_failure_does_not_defer_first_runtime_warning(
    dense_imports, monkeypatch, caplog, module,
):
    """Alias evaluation must not consume the runtime warning or cooldown."""
    dense_imports.failing = module
    caplog.set_level(logging.WARNING, logger=embeddings.__name__)
    monkeypatch.setattr(embeddings, "_IMPORTING", True)
    assert embeddings._ensure_fastembed() is False
    assert caplog.records == []
    monkeypatch.setattr(embeddings, "_IMPORTING", False)
    assert embeddings._ensure_fastembed() is False
    assert len(caplog.records) == 1
    assert f"could not import {module}:" in caplog.records[0].getMessage()


@pytest.mark.parametrize("module", ["numpy", "fastembed"])
@pytest.mark.parametrize("missing", ["onnxruntime", "numpy._core", None])
def test_missing_transitive_dependency_backs_off(dense_imports, module, missing):
    """A missing dependency inside an installed package must not cause an import storm."""
    state = dense_imports
    state.failing = module
    state.error = ModuleNotFoundError("installed package is broken", name=missing)
    assert embeddings._ensure_fastembed() is False
    attempts = list(state.attempts)
    state.failing = None
    assert embeddings._ensure_fastembed() is False
    assert state.attempts == attempts
    state.clock.now = 101.0
    assert embeddings._ensure_fastembed() is True


def test_fresh_module_survives_broken_numpy_and_recovers(monkeypatch, caplog):
    """An initial NumPy RuntimeError must leave the lazy recovery path usable."""
    original_import = builtins.__import__
    state = SimpleNamespace(broken=True)
    numpy = SimpleNamespace()
    fastembed = SimpleNamespace(TextEmbedding=type("TextEmbedding", (), {}))

    def import_optional(name, *args, **kwargs):
        if name == "numpy":
            if state.broken:
                raise RuntimeError("broken NumPy initialization")
            return numpy
        if name == "fastembed":
            return fastembed
        return original_import(name, *args, **kwargs)

    caplog.set_level(logging.WARNING, logger=embeddings.__name__)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", import_optional)
        spec = importlib.util.spec_from_file_location(
            "embedding_fresh_import_probe", embeddings.__file__,
        )
        fresh = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fresh)
        assert fresh.np is None
        assert fresh._IMPORTING is False
        assert fresh._ensure_fastembed() is False
        assert "could not import numpy" in caplog.text
        state.broken = False
        patch.setattr(fresh, "_DENSE_STACK_RETRY_AT", 0.0)
        assert fresh._ensure_fastembed() is True
        assert fresh.np is numpy
        assert fresh.TextEmbedding is fastembed.TextEmbedding
