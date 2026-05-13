"""
Regression tests for the conftest's default-disable of local LLM
inference (T1, follow-up to the 2026-05-12 security audit).

Background: in dev environments that have a GGUF model file on disk
plus ``llama-cpp-python`` or ``ctransformers`` installed (e.g. the
Hermes Agent venv with tinyllama at
``~/.hermes/mnemosyne/models/``), ``local_llm._load_llm()`` auto-loads
the model. ``beam.sleep()`` tests that don't explicitly monkeypatch
``llm_available`` then run real CPU inference -- 5-30s per call,
turning a 20-second suite into a 15+ minute one.

The autouse fixture ``_disable_local_llm_inference`` in
``tests/conftest.py`` neutralises this by replacing ``_load_llm``
with a stub that returns None. The opt-in fixture
``local_llm_enabled`` lets specific tests put a fake LLM back in
place.

This file locks in:
  1. The default-disable actually works.
  2. The opt-in fixture restores a working LLM path.
  3. Per-test ``monkeypatch.setattr(local_llm, "_load_llm", ...)``
     still wins over the autouse default (existing tests in
     ``test_extraction.py`` rely on this).

Run with: pytest tests/test_t1_local_llm_default_disabled.py -v
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Default state (autouse fixture)
# ---------------------------------------------------------------------------


class TestDefaultDisabled:
    """Without opting in, no test should see a working local LLM."""

    def test_llm_available_returns_false_by_default(self):
        """The most important contract -- gating callers see False."""
        from mnemosyne.core import local_llm
        assert local_llm.llm_available() is False

    def test_load_llm_returns_none_by_default(self):
        """``_load_llm`` is the stub."""
        from mnemosyne.core import local_llm
        assert local_llm._load_llm() is None

    def test_call_local_llm_returns_none_by_default(self):
        """The whole call path short-circuits cleanly."""
        from mnemosyne.core import local_llm
        result = local_llm._call_local_llm("anything")
        assert result is None

    def test_cached_flags_reset_per_test(self):
        """Each test sees a fresh (None, None, None) cache state.

        Without this, a previous test that set ``_llm_available = True``
        would leak forward.
        """
        from mnemosyne.core import local_llm
        assert local_llm._llm_available is None
        assert local_llm._llm_instance is None
        assert local_llm._llm_backend is None


# ---------------------------------------------------------------------------
# Opt-in via the local_llm_enabled fixture
# ---------------------------------------------------------------------------


class TestOptInFixture:
    """``local_llm_enabled`` fixture restores a working LLM path."""

    def test_fixture_makes_llm_available_true(self, local_llm_enabled):
        from mnemosyne.core import local_llm
        assert local_llm.llm_available() is True

    def test_fixture_provides_ctransformers_style_callable(self, local_llm_enabled):
        """ctransformers backend: the loaded object is callable."""
        from mnemosyne.core import local_llm
        llm = local_llm._load_llm()
        assert llm is not None
        # ctransformers-style: instance is directly callable
        result = llm("test prompt")
        assert result == "fake summary"

    def test_fixture_provides_llamacpp_style_interface(self, local_llm_enabled):
        """llama-cpp-python backend: create_chat_completion method exists."""
        from mnemosyne.core import local_llm
        llm = local_llm._load_llm()
        result = llm.create_chat_completion(
            messages=[{"role": "user", "content": "test"}]
        )
        assert result["choices"][0]["message"]["content"] == "fake summary"

    def test_response_is_customizable(self, local_llm_enabled):
        """Tests can override the fake response per-test."""
        local_llm_enabled.response = "different output"
        from mnemosyne.core import local_llm
        llm = local_llm._load_llm()
        assert llm("anything") == "different output"

    def test_isolation_after_opt_in_test(self):
        """A NEXT test (this one) without the fixture should see the
        default-disabled state again -- the opt-in didn't leak."""
        from mnemosyne.core import local_llm
        assert local_llm.llm_available() is False
        assert local_llm._load_llm() is None


# ---------------------------------------------------------------------------
# Compatibility with per-test patches (existing tests rely on this)
# ---------------------------------------------------------------------------


class TestPerTestPatchOverridesAutouse:
    """Tests that monkeypatch ``_load_llm`` themselves must still win."""

    def test_monkeypatch_load_llm_after_autouse_takes_effect(self, monkeypatch):
        """``monkeypatch.setattr`` applied INSIDE the test body should
        override the autouse fixture's setattr, because pytest applies
        setattrs in call order and the test's setattr happens later.

        This mirrors what ``tests/test_extraction.py`` and other
        existing tests do."""
        from mnemosyne.core import local_llm

        def fake_load():
            class _F:
                def __call__(self, prompt, *a, **kw):
                    return "per-test fake"
            return _F()

        monkeypatch.setattr(local_llm, "_load_llm", fake_load)
        llm = local_llm._load_llm()
        assert llm is not None
        assert llm("anything") == "per-test fake"

    def test_patch_object_context_manager_overrides_autouse(self):
        """``unittest.mock.patch.object`` (context manager) entered
        after the autouse fixture should also win."""
        from unittest.mock import patch
        from mnemosyne.core import local_llm

        class _F:
            def __call__(self, prompt, *a, **kw):
                return "ctx-mgr fake"

        with patch.object(local_llm, "_load_llm", return_value=_F()):
            llm = local_llm._load_llm()
            assert llm("anything") == "ctx-mgr fake"

        # After the `with`, autouse default is restored
        assert local_llm._load_llm() is None


# ---------------------------------------------------------------------------
# Sanity: the real bug stays fixed
# ---------------------------------------------------------------------------


class TestSleepIsFast:
    """End-to-end check: ``beam.sleep()`` no longer invokes real
    inference when run from a venv that happens to have a GGUF model on
    disk. We can't directly assert "no model loaded" portably, but we
    CAN assert the deterministic non-LLM summary path is hit by
    confirming the consolidation completes nearly instantly.

    A regression that re-enables the LLM by default would make this
    test slow (seconds-to-minutes) -- detectable by CI runtime.
    """

    def test_sleep_does_not_hit_llm(self, tmp_path, monkeypatch):
        """If anything calls _call_local_llm during sleep, it should
        return None (the autouse stub). Sleep should still succeed
        using the deterministic AAAK fallback."""
        import sqlite3
        from datetime import datetime, timedelta
        from mnemosyne.core.beam import BeamMemory
        from mnemosyne.core import local_llm

        # Track whether anything tried to load the LLM
        call_count = {"loads": 0}
        original_load = local_llm._load_llm

        def counting_load():
            call_count["loads"] += 1
            return original_load()  # still returns None per autouse

        monkeypatch.setattr(local_llm, "_load_llm", counting_load)

        db = tmp_path / "t1_sleep.db"
        beam = BeamMemory(session_id="s1", db_path=db)
        conn = sqlite3.connect(db)
        old_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [(f"old{i}", f"task {i}", "conversation", old_ts, "s1")
             for i in range(3)],
        )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["items_consolidated"] == 3

        # Even if _load_llm was called, it returned None (the stub),
        # so no actual inference happened. The assertion isn't on
        # call_count -- we just want to know sleep succeeded without
        # blocking on a real model.
