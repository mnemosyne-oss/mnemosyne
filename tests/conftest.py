"""
Shared test fixtures for Mnemosyne test suite.

Provides fixtures that handle SQLite thread-local connection cleanup
to prevent "database is locked" and UNIQUE constraint collisions
between tests, and that default-disable the local LLM so tests don't
make real CPU inference calls when a model is available on disk.
"""

import pytest


def _close_cached_connections():
    """Close and reset thread-local SQLite connection caches in both modules."""
    for mod_path in (
        "mnemosyne.core.beam",
        "mnemosyne.core.memory",
    ):
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            tl = getattr(mod, "_thread_local", None)
            if tl is not None and hasattr(tl, "conn") and tl.conn is not None:
                try:
                    tl.conn.close()
                except Exception:
                    pass
                tl.conn = None
                if hasattr(tl, "db_path"):
                    tl.db_path = None
        except Exception:
            pass

    # Reset the global Mnemosyne default instance to avoid cross-test
    # contamination of the singleton
    try:
        from mnemosyne.core import memory as _mem_mod
        _mem_mod._default_instance = None
        _mem_mod._default_bank = "default"
    except Exception:
        pass

    # Reset hermes_plugin singleton
    try:
        import hermes_plugin
        hermes_plugin._memory_instance = None
        hermes_plugin._current_session_id = None
        hermes_plugin._triple_store = None
    except Exception:
        pass

    # Reset host LLM backend registry to prevent cross-test contamination.
    # The registry is a process-global; a test that forgets to unregister
    # would otherwise bleed into the next.
    try:
        from mnemosyne.core import llm_backends as _llm_backends_mod
        _llm_backends_mod._backend = None
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_thread_local_connections():
    """
    Auto-use fixture that resets thread-local SQLite connection caches
    before and after every test. This prevents connection leakage between
    tests that use different database paths.

    Both mnemosyne.core.beam and mnemosyne.core.memory maintain their own
    thread-local caches (_thread_local.conn / _thread_local.db_path).
    When tests create instances with different db_paths, the old connection
    is never closed, leading to "database is locked" errors.
    """
    _close_cached_connections()
    yield
    _close_cached_connections()


# ---------------------------------------------------------------------------
# Local-LLM default-disable (test environment hygiene)
# ---------------------------------------------------------------------------
#
# Background: in dev environments that have a GGUF model file on disk plus
# `llama-cpp-python` or `ctransformers` installed (e.g. the Hermes Agent
# venv at ~/.hermes/hermes-agent/venv with tinyllama at
# ~/.hermes/mnemosyne/models/), `mnemosyne.core.local_llm._load_llm()`
# auto-loads the model on first call. Any test that exercises a
# summarization path (`beam.sleep()`, `consolidate_to_episodic`, SHMR, the
# extraction fallback) then runs real CPU inference -- 5-30 seconds per
# call, multiplied by the number of items consolidated. The full suite
# goes from ~20 seconds (CI: no model on disk) to 15+ minutes locally.
#
# This fixture default-disables `_load_llm` and clears its cached state
# at the start of every test, so tests behave like CI by default.
#
# Tests that need to exercise an LLM-enabled code path should explicitly
# opt in by either:
#   - Using the `local_llm_enabled` fixture below, or
#   - Calling `monkeypatch.setattr(local_llm, "_load_llm", lambda: fake)`
#     with their own fake callable.
#
# Both `unittest.mock.patch.object(local_llm, "_load_llm", ...)` (context
# manager) and `monkeypatch.setattr(local_llm, "_load_llm", ...)` applied
# after the autouse fixture override our default-disable -- pytest's
# monkeypatch is function-scoped and later setattrs win.


@pytest.fixture(autouse=True)
def _disable_local_llm_inference(monkeypatch):
    """
    Default-disable the local LLM model loader for every test.

    Concretely, replaces ``mnemosyne.core.local_llm._load_llm`` with a
    function that returns None, and clears the module's cached
    availability flags so previously-set state from another import path
    doesn't leak in.

    Effect on the public API:
      - ``local_llm.llm_available()`` returns False (since path 3 --
        ``_load_llm()`` -- now yields None and ``_llm_available`` stays
        None, so ``bool(None) == False``).
      - ``local_llm._call_local_llm(...)`` returns None (short-circuits
        on ``llm is None``).
      - Sleep / extraction / SHMR code paths that gate on
        ``llm_available()`` take the deterministic non-LLM branch.

    Tests that want a working LLM path should opt in via the
    ``local_llm_enabled`` fixture below, or set their own
    ``_load_llm`` mock.
    """
    try:
        from mnemosyne.core import local_llm
    except Exception:
        # local_llm import failures shouldn't abort the test session.
        yield
        return

    monkeypatch.setattr(local_llm, "_llm_available", None, raising=False)
    monkeypatch.setattr(local_llm, "_llm_instance", None, raising=False)
    monkeypatch.setattr(local_llm, "_llm_backend", None, raising=False)
    monkeypatch.setattr(local_llm, "_load_llm", lambda: None, raising=True)
    yield


@pytest.fixture
def local_llm_enabled(monkeypatch):
    """
    Opt-in fixture for tests that need a working LLM path.

    Installs a deterministic fake LLM via ``_load_llm``: the returned
    object is callable (mimicking the ctransformers callable interface)
    AND has a ``create_chat_completion`` method (mimicking the
    llama-cpp-python interface), so tests don't have to care which
    backend a given code path expects.

    Usage::

        def test_summarization(local_llm_enabled, ...):
            # llm_available() is True here, and any code that calls
            # _call_local_llm gets a fake completion.

    Override the fake response per-test by capturing the fixture and
    setting ``local_llm_enabled.response``:

        def test_x(local_llm_enabled):
            local_llm_enabled.response = "custom output"
            ...
    """
    from mnemosyne.core import local_llm

    class _FakeLLM:
        response = "fake summary"

        # ctransformers-style: instance is callable
        def __call__(self, prompt, *args, **kwargs):
            return self.response

        # llama-cpp-python-style: create_chat_completion
        def create_chat_completion(self, messages, **kwargs):
            return {"choices": [{"message": {"content": self.response}}]}

    fake = _FakeLLM()
    monkeypatch.setattr(local_llm, "_load_llm", lambda: fake, raising=True)
    monkeypatch.setattr(local_llm, "_llm_instance", fake, raising=False)
    monkeypatch.setattr(local_llm, "_llm_available", True, raising=False)
    monkeypatch.setattr(local_llm, "_llm_backend", "llamacpp", raising=False)
    return fake
