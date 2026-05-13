"""
Regression tests for C13: prevent memory-context double-injection.

Pre-fix, when Hermes loaded both the ``MnemosyneMemoryProvider``
(canonical surface) AND the legacy ``hermes_plugin`` (composed by the
provider's ``register()`` or independently discovered when an extra
``plugin.yaml`` is found), TWO pre-turn memory-injection paths fired
on every LLM call:

  1. ``MnemosyneMemoryProvider.prefetch()`` rendered a
     ``## Mnemosyne Context`` block.
  2. ``hermes_plugin._on_pre_llm_call()`` rendered a separate
     ``MNEMOSYNE CONTEXT / MNEMOSYNE RECALL`` block.

Both ran their own ``beam.recall()`` and wrote to the system prompt,
doubling per-turn token cost and confusing the agent with two
differently-formatted views of similar content.

Post-fix: when the provider's ``initialize()`` runs successfully in a
non-skip context, the module-level
``hermes_memory_provider._provider_active = True`` flag is set. The
plugin's pre_llm_call hook reads the flag and returns ``None`` so
memory injection happens through the canonical provider surface only.

The standalone plugin-only install (no MemoryProvider package, or the
provider was never initialized) is unaffected: the flag stays ``False``
and the plugin hook injects context as before.

Run with: pytest tests/test_c13_memory_context_single_injection.py -v
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Helper fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_provider_active():
    """Restore both module-level C13 globals after each test.

    Module-level globals leak across tests. The autouse conftest
    fixture ``_close_cached_connections`` already covers a handful of
    similar globals; this fixture targets the C13 flag AND its
    underlying refcount so a prior test that incremented can't make a
    later test's decrement fail to reach zero."""
    import hermes_memory_provider as hmp
    original_flag = hmp._provider_active
    original_count = hmp._active_provider_count
    # Also start each test from a clean baseline so the SAME test
    # under different orderings always begins at the same state.
    hmp._provider_active = False
    hmp._active_provider_count = 0
    yield
    hmp._provider_active = original_flag
    hmp._active_provider_count = original_count


@pytest.fixture
def fake_ctx():
    """Hermes-like context that records every register_hook call."""

    class _Ctx:
        def __init__(self):
            self.tools = []
            self.hooks = {}
            self.cli_commands = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, name, fn):
            self.hooks.setdefault(name, []).append(fn)

        def register_cli_command(self, **kwargs):
            self.cli_commands.append(kwargs)

        def register_memory_provider(self, provider):
            self.memory_provider = provider

    return _Ctx()


# ---------------------------------------------------------------------------
# The flag itself
# ---------------------------------------------------------------------------


class TestProviderActiveFlag:
    """``hermes_memory_provider._provider_active`` is the single source of
    truth for whether the canonical surface is the active memory-
    injection path."""

    def test_default_is_false(self, reset_provider_active):
        """Out of the box, before any initialize(), the flag is False --
        so a standalone plugin install still works the way it did
        pre-C13."""
        import hermes_memory_provider as hmp
        hmp._provider_active = False  # reset for a clean baseline
        assert hmp._provider_active is False

    def test_set_true_on_provider_initialize(self, reset_provider_active, tmp_path, monkeypatch):
        """After provider.initialize() completes in a non-skip context,
        the flag is True."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        import hermes_memory_provider as hmp
        hmp._provider_active = False  # force a clean start
        provider = hmp.MnemosyneMemoryProvider()
        provider.initialize(session_id="t1", hermes_home=str(tmp_path / "h"))
        assert hmp._provider_active is True

    def test_stays_false_in_skip_context(self, reset_provider_active, tmp_path):
        """Subagent / cron / skill_loop contexts don't activate the
        provider, so the plugin path should still run (its existing
        behavior for those contexts -- separate concern from C13)."""
        import hermes_memory_provider as hmp
        hmp._provider_active = False
        provider = hmp.MnemosyneMemoryProvider()
        provider.initialize(
            session_id="t1", agent_context="subagent",
            hermes_home=str(tmp_path / "h"),
        )
        assert hmp._provider_active is False, (
            "skip-context init must not set the flag -- otherwise the "
            "plugin hook would silently defer in a context where the "
            "provider isn't injecting either"
        )

    def test_stays_false_on_init_failure(self, reset_provider_active, tmp_path, monkeypatch):
        """Codex review #1: on this branch (C27 not yet merged), the
        provider's system_prompt_block returns "" when _beam is None, so
        if BeamMemory init fails AND we silenced the plugin too, the
        user would see ZERO indication memory is broken. Preserve the
        plugin path as a legacy fallback until C27 lands."""
        from unittest.mock import patch
        import hermes_memory_provider as hmp
        hmp._provider_active = False
        provider = hmp.MnemosyneMemoryProvider()
        with patch(
            "hermes_memory_provider._get_beam_class",
            side_effect=RuntimeError("simulated"),
        ):
            provider.initialize(session_id="t1", hermes_home=str(tmp_path / "h"))
        assert hmp._provider_active is False, (
            "on init failure the provider didn't actually become active "
            "(no _beam) -- leaving the plugin path enabled preserves a "
            "legacy fallback that keeps memory surface functional rather "
            "than silently breaking both paths"
        )

    def test_reset_on_shutdown(self, reset_provider_active, tmp_path, monkeypatch):
        """shutdown() resets the flag so a process that later reuses
        the plugin path gets the legacy injection back."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        import hermes_memory_provider as hmp
        hmp._provider_active = False
        provider = hmp.MnemosyneMemoryProvider()
        provider.initialize(session_id="t1", hermes_home=str(tmp_path / "h"))
        assert hmp._provider_active is True
        provider.shutdown()
        assert hmp._provider_active is False

    def test_primary_then_skip_reinit_deactivates(self, reset_provider_active, tmp_path, monkeypatch):
        """Codex review #2: a primary->skip-context re-init in the same
        process must deactivate so the plugin's pre_llm_call still
        injects context for the subagent session (the plugin has no
        skip-context check of its own; pre-C13 it always injected, and
        we don't want to silently break that for subagents)."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        import hermes_memory_provider as hmp
        hmp._provider_active = False
        provider = hmp.MnemosyneMemoryProvider()

        # Primary init activates
        provider.initialize(session_id="t1", hermes_home=str(tmp_path / "h"))
        assert hmp._provider_active is True

        # Re-init in skip context must deactivate
        provider.initialize(
            session_id="t2", agent_context="subagent",
            hermes_home=str(tmp_path / "h"),
        )
        assert hmp._provider_active is False, (
            "primary->skip re-init must deactivate so the plugin's "
            "pre_llm_call still runs for the subagent session (codex #2)"
        )

    def test_multiple_providers_refcount(self, reset_provider_active, tmp_path, monkeypatch):
        """Codex review #3: shutting down one provider with another
        still active must NOT deactivate the flag globally. Use a
        refcount; flag stays True until ALL providers shut down."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        import hermes_memory_provider as hmp
        hmp._provider_active = False
        hmp._active_provider_count = 0

        provider_a = hmp.MnemosyneMemoryProvider()
        provider_b = hmp.MnemosyneMemoryProvider()

        provider_a.initialize(session_id="a", hermes_home=str(tmp_path / "ha"))
        assert hmp._provider_active is True
        assert hmp._active_provider_count == 1

        provider_b.initialize(session_id="b", hermes_home=str(tmp_path / "hb"))
        assert hmp._provider_active is True
        assert hmp._active_provider_count == 2

        provider_a.shutdown()
        assert hmp._provider_active is True, (
            "shutting down provider A while B is still active must NOT "
            "globally reset the flag -- B's prefetch path is still live "
            "and the plugin must keep deferring (codex #3)"
        )
        assert hmp._active_provider_count == 1

        provider_b.shutdown()
        assert hmp._provider_active is False
        assert hmp._active_provider_count == 0

    def test_redundant_initialize_does_not_double_count(self, reset_provider_active, tmp_path, monkeypatch):
        """Per-instance idempotency: re-running initialize() on the same
        instance in the same context must not increment the count
        twice. Otherwise shutdown() leaves a stuck-positive count."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        import hermes_memory_provider as hmp
        hmp._provider_active = False
        hmp._active_provider_count = 0

        provider = hmp.MnemosyneMemoryProvider()
        provider.initialize(session_id="t1", hermes_home=str(tmp_path / "h"))
        provider.initialize(session_id="t1", hermes_home=str(tmp_path / "h"))
        assert hmp._active_provider_count == 1, (
            f"re-initialize must not double-increment, got "
            f"{hmp._active_provider_count}"
        )
        provider.shutdown()
        assert hmp._active_provider_count == 0
        assert hmp._provider_active is False

    def test_shutdown_on_never_activated_is_noop(self, reset_provider_active):
        """Defensive: a fresh provider that never initialized (or
        initialized in skip context) must not produce a negative
        count when shutdown is called."""
        import hermes_memory_provider as hmp
        hmp._provider_active = False
        hmp._active_provider_count = 5  # simulate other active instances

        provider = hmp.MnemosyneMemoryProvider()
        provider.shutdown()
        assert hmp._active_provider_count == 5, (
            "shutdown on a never-activated instance must not decrement "
            "the count -- otherwise out-of-order lifecycle drives the "
            "count negative and breaks subsequent providers"
        )


# ---------------------------------------------------------------------------
# Plugin hook respects the flag
# ---------------------------------------------------------------------------


class TestPluginHookDefersWhenProviderActive:
    """``hermes_plugin._on_pre_llm_call`` returns ``None`` when the
    provider-active flag is set, deferring memory injection to the
    canonical surface."""

    def test_returns_none_when_provider_active(self, reset_provider_active):
        """Headline assertion: with provider active, the hook is a
        no-op. No recall query is made, no context block is rendered."""
        import hermes_memory_provider as hmp
        import hermes_plugin

        hmp._provider_active = True
        result = hermes_plugin._on_pre_llm_call(
            session_id="any", history=[],
        )
        assert result is None, (
            "with the provider active, the plugin hook must defer -- "
            "otherwise both paths inject a memory-context block into "
            "the system prompt every turn (C13)"
        )

    def test_runs_normally_when_provider_inactive(self, reset_provider_active, monkeypatch):
        """With the flag False (standalone plugin install), the hook
        runs as before. We don't assert on the full output; we just
        verify _get_memory was called -- the smoking gun that the
        defer-check didn't short-circuit."""
        import hermes_memory_provider as hmp
        import hermes_plugin

        hmp._provider_active = False

        calls = []

        class _FakeMem:
            def get_context(self, limit=5):
                calls.append(("get_context", limit))
                return []

            def recall(self, query, **kwargs):
                calls.append(("recall", query))
                return []

        monkeypatch.setattr(hermes_plugin, "_get_memory", lambda **kw: _FakeMem())

        result = hermes_plugin._on_pre_llm_call(
            session_id="s1", history=[{"role": "user", "content": "hello"}],
        )

        # Hook ran -- it called _get_memory at least once
        assert calls, (
            "with provider inactive, plugin hook must run its legacy "
            "logic (the standalone install path)"
        )
        assert any(c[0] == "get_context" for c in calls)

    def test_defer_does_not_call_memory(self, reset_provider_active, monkeypatch):
        """When deferring, the plugin must NOT touch the memory
        instance -- otherwise it would consume the recall/cache for no
        reason and defeat the purpose of the C13 fix."""
        import hermes_memory_provider as hmp
        import hermes_plugin

        hmp._provider_active = True

        memory_was_touched = {"yes": False}

        def _boom(**kwargs):
            memory_was_touched["yes"] = True
            raise AssertionError(
                "with provider active, _on_pre_llm_call must NOT call "
                "_get_memory -- it should defer before any memory access"
            )

        monkeypatch.setattr(hermes_plugin, "_get_memory", _boom)
        result = hermes_plugin._on_pre_llm_call(
            session_id="s1", history=[],
        )
        assert result is None
        assert memory_was_touched["yes"] is False


# ---------------------------------------------------------------------------
# Standalone plugin install path -- the fallback that must keep working
# ---------------------------------------------------------------------------


class TestStandalonePluginInstall:
    """When ``hermes_memory_provider`` is not importable (older standalone
    plugin install), the hook falls through to legacy behavior."""

    def test_hook_works_when_provider_module_missing(self, monkeypatch):
        """Simulate the standalone-plugin install where the
        MemoryProvider package isn't installed. The ImportError on the
        flag lookup must be caught AND the legacy injection logic must
        actually run (codex review #6 -- testing "no exception" alone
        would be satisfied by an unconditional `return None`)."""
        import sys
        import hermes_plugin

        # Track that the legacy path actually executed
        legacy_calls = []

        class _FakeMem:
            def get_context(self, limit=5):
                legacy_calls.append("get_context")
                return []

            def recall(self, query, **kwargs):
                legacy_calls.append("recall")
                return []

        monkeypatch.setattr(hermes_plugin, "_get_memory", lambda **kw: _FakeMem())

        # Temporarily remove the provider module from sys.modules so the
        # `from hermes_memory_provider import _provider_active` lookup
        # raises ImportError as it would for a standalone install.
        original = sys.modules.pop("hermes_memory_provider", None)
        try:
            class _FailingFinder:
                def find_module(self, name, path=None):
                    if name == "hermes_memory_provider":
                        raise ImportError("simulated: not installed")
                    return None

                def find_spec(self, name, path, target=None):
                    if name == "hermes_memory_provider":
                        raise ImportError("simulated: not installed")
                    return None

            sys.meta_path.insert(0, _FailingFinder())
            try:
                # Hook must not raise AND must run legacy injection
                hermes_plugin._on_pre_llm_call(
                    session_id="s1", history=[],
                )
            finally:
                sys.meta_path.pop(0)
        finally:
            if original is not None:
                sys.modules["hermes_memory_provider"] = original

        # The legacy path actually executed -- proves the ImportError
        # didn't accidentally short-circuit the whole function (which
        # would have been a different kind of regression: standalone
        # plugin installs would silently stop injecting memory).
        assert legacy_calls, (
            "standalone install must keep injecting memory -- the "
            "ImportError catch can't short-circuit the legacy logic"
        )
        assert "get_context" in legacy_calls


# ---------------------------------------------------------------------------
# End-to-end: provider initialize -> plugin hook defers -> no double inject
# ---------------------------------------------------------------------------


class TestEndToEndSingleInjection:
    """Full lifecycle: initializing the provider silences the plugin
    pre_llm_call hook, leaving only the provider's prefetch as the
    memory-injection path."""

    def test_provider_initialize_silences_plugin_hook(
        self, reset_provider_active, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
        import hermes_memory_provider as hmp
        import hermes_plugin

        hmp._provider_active = False  # clean baseline

        # Pre-init: plugin hook runs (legacy install)
        recall_calls_before = []

        class _Mem:
            def get_context(self, limit=5):
                recall_calls_before.append("ctx")
                return []

            def recall(self, q, **kw):
                recall_calls_before.append("recall")
                return []

        monkeypatch.setattr(hermes_plugin, "_get_memory", lambda **kw: _Mem())
        hermes_plugin._on_pre_llm_call(session_id="s", history=[])
        assert recall_calls_before, (
            "baseline check: pre-init the plugin hook must do work"
        )

        # Now initialize the provider
        provider = hmp.MnemosyneMemoryProvider()
        provider.initialize(session_id="t1", hermes_home=str(tmp_path / "h"))
        assert hmp._provider_active is True

        # Plugin hook must now defer (no calls to _get_memory)
        recall_calls_after = []

        def _boom(**kw):
            recall_calls_after.append("called")
            raise AssertionError("plugin hook must not call _get_memory after provider init")

        monkeypatch.setattr(hermes_plugin, "_get_memory", _boom)
        result = hermes_plugin._on_pre_llm_call(session_id="s", history=[])
        assert result is None
        assert not recall_calls_after, (
            "after provider initialize, plugin hook must defer -- otherwise "
            "both paths run a recall every turn"
        )
