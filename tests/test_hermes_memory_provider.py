"""Tests for MnemosyneMemoryProvider host-LLM lifecycle hooks.

Covers decisions A6 (bounded on_session_end), C7 (shutdown unregisters
the host backend), and the registration flow added to initialize().
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core.llm_backends import get_host_llm_backend


# ---------------------------------------------------------------------------
# initialize() registration
# ---------------------------------------------------------------------------

def test_initialize_registers_host_llm_when_register_returns_true(monkeypatch):
    provider = MnemosyneMemoryProvider()
    # Stub BeamMemory so we don't touch the filesystem.
    monkeypatch.setattr("hermes_memory_provider._get_beam_class", lambda: lambda **kwargs: MagicMock())
    # Stub the registration call so the test does not depend on the real
    # adapter behavior — we only verify the hook is invoked and survives.
    with patch("hermes_memory_provider.hermes_llm_adapter.register_hermes_host_llm", return_value=True) as mock_reg:
        provider.initialize(session_id="test-session")
    mock_reg.assert_called_once()


def test_initialize_does_not_fail_when_register_raises(monkeypatch):
    provider = MnemosyneMemoryProvider()
    monkeypatch.setattr("hermes_memory_provider._get_beam_class", lambda: lambda **kwargs: MagicMock())
    with patch(
        "hermes_memory_provider.hermes_llm_adapter.register_hermes_host_llm",
        side_effect=RuntimeError("boom"),
    ):
        # Must not raise.
        provider.initialize(session_id="test-session")
    # initialize() is allowed to leave _beam set even when registration explodes.
    assert provider._beam is not None


def test_initialize_does_not_fail_when_register_returns_false(monkeypatch):
    provider = MnemosyneMemoryProvider()
    monkeypatch.setattr("hermes_memory_provider._get_beam_class", lambda: lambda **kwargs: MagicMock())
    with patch("hermes_memory_provider.hermes_llm_adapter.register_hermes_host_llm", return_value=False):
        provider.initialize(session_id="test-session")
    assert provider._beam is not None


def test_initialize_skips_for_non_primary_context(monkeypatch):
    """REGRESSION: subagent/cron/flush contexts still skip initialization entirely."""
    provider = MnemosyneMemoryProvider()
    with patch("hermes_memory_provider.hermes_llm_adapter.register_hermes_host_llm") as mock_reg:
        provider.initialize(session_id="x", agent_context="cron")
    mock_reg.assert_not_called()
    assert provider._beam is None


# ---------------------------------------------------------------------------
# shutdown() unregistration (decision C7)
# ---------------------------------------------------------------------------

def test_shutdown_clears_host_backend(monkeypatch):
    """After shutdown(), the host LLM backend must be unregistered."""
    from hermes_memory_provider import hermes_llm_adapter

    provider = MnemosyneMemoryProvider()
    # Manually register to simulate a live session.
    hermes_llm_adapter.register_hermes_host_llm()
    assert get_host_llm_backend() is not None

    provider.shutdown()
    assert get_host_llm_backend() is None
    assert provider._beam is None


def test_shutdown_swallows_unregister_failure(monkeypatch):
    """If unregistering raises, shutdown() must still complete."""
    provider = MnemosyneMemoryProvider()
    with patch(
        "hermes_memory_provider.hermes_llm_adapter.unregister_hermes_host_llm",
        side_effect=RuntimeError("boom"),
    ):
        provider.shutdown()  # must not raise
    assert provider._beam is None


# ---------------------------------------------------------------------------
# on_session_end() bounded daemon thread (decision A6)
# ---------------------------------------------------------------------------

def _make_provider_with_blocking_sleep(sleep_duration: float, timeout: float = 0.5):
    """Build a provider whose _beam.sleep() blocks for `sleep_duration` seconds.

    The provider's join timeout is shortened to keep the test suite fast.
    """
    beam = MagicMock()
    beam.sleep.side_effect = lambda: time.sleep(sleep_duration)
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider.SESSION_END_SLEEP_TIMEOUT_SECONDS = timeout
    return provider, beam


def test_on_session_end_returns_within_timeout_when_sleep_blocks():
    """A6 contract: blocking sleep must not block on_session_end past the join cap."""
    # Production timeout is 15s; test uses 0.5s for speed and a 5s outer ceiling.
    provider, beam = _make_provider_with_blocking_sleep(sleep_duration=5.0, timeout=0.5)

    start = time.monotonic()
    provider.on_session_end(messages=[])
    elapsed = time.monotonic() - start

    # 0.5s join cap + slack. A regression making on_session_end synchronous
    # would take ~5s here.
    assert elapsed < 2.0, f"on_session_end took {elapsed:.2f}s, expected <2s"
    beam.sleep.assert_called_once()


def test_on_session_end_logs_warning_on_timeout(caplog):
    provider, _ = _make_provider_with_blocking_sleep(sleep_duration=5.0, timeout=0.5)
    with caplog.at_level("WARNING", logger="hermes_memory_provider"):
        provider.on_session_end(messages=[])
    msgs = [r.getMessage() for r in caplog.records]
    assert any("timed out" in m for m in msgs), msgs


def test_session_end_timeout_default_matches_design():
    """The production default should remain 15s (decision A6)."""
    assert MnemosyneMemoryProvider.SESSION_END_SLEEP_TIMEOUT_SECONDS == 15


def test_on_session_end_completes_when_sleep_is_fast():
    """Fast sleep must be allowed to finish; no warning emitted."""
    beam = MagicMock()
    # No-op sleep returns immediately.
    beam.sleep.return_value = None
    provider = MnemosyneMemoryProvider()
    provider._beam = beam

    provider.on_session_end(messages=[])
    beam.sleep.assert_called_once()


def test_on_session_end_no_op_without_beam():
    """REGRESSION: on_session_end skips work entirely when not initialized."""
    provider = MnemosyneMemoryProvider()
    provider._beam = None
    # Must not raise.
    provider.on_session_end(messages=[])


def test_on_session_end_logs_when_sleep_raises_in_daemon_thread(caplog):
    """Codex finding 3: exceptions from beam.sleep() now happen in the daemon
    thread; the wrapper must catch them and log at DEBUG instead of letting
    the traceback escape uncaught."""
    beam = MagicMock()
    beam.sleep.side_effect = RuntimeError("synthetic explosion")
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider.SESSION_END_SLEEP_TIMEOUT_SECONDS = 1.0  # plenty of time for the raise to happen

    with caplog.at_level("DEBUG", logger="hermes_memory_provider"):
        provider.on_session_end(messages=[])
    # Wait for daemon thread to fully run the wrapper.
    if provider._session_end_thread is not None:
        provider._session_end_thread.join(timeout=2.0)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("session-end sleep failed" in m and "synthetic explosion" in m for m in msgs), msgs


def test_shutdown_drains_in_flight_session_end_thread(caplog):
    """Codex finding 4: shutdown() must briefly wait for an in-flight
    session_end thread before clearing the host backend, otherwise the
    daemon thread's late host call sees backend=None and degrades to remote."""
    from hermes_memory_provider import hermes_llm_adapter

    # Make the session_end thread block for ~0.4s — longer than the
    # session_end timeout (0.1s) but well within the shutdown drain.
    beam = MagicMock()
    sleep_started = []
    sleep_finished = []

    def slow_sleep():
        sleep_started.append(True)
        time.sleep(0.4)
        sleep_finished.append(True)

    beam.sleep.side_effect = slow_sleep
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider.SESSION_END_SLEEP_TIMEOUT_SECONDS = 0.1
    provider.SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 1.0

    # Start session_end (returns after 0.1s; daemon keeps running)
    provider.on_session_end(messages=[])
    assert provider._session_end_thread is not None
    assert provider._session_end_thread.is_alive(), "daemon should still be running"

    # Register the host backend so we can observe it being cleared
    hermes_llm_adapter.register_hermes_host_llm()
    assert get_host_llm_backend() is not None

    # Shutdown should drain the in-flight thread BEFORE clearing the backend
    provider.shutdown()

    # By now the daemon thread should have finished
    assert sleep_started, "daemon should have started"
    assert sleep_finished, "shutdown should have drained the in-flight daemon"
    assert get_host_llm_backend() is None  # cleared after drain


def test_shutdown_proceeds_when_drain_times_out(caplog):
    """If the drain takes longer than SHUTDOWN_DRAIN_TIMEOUT_SECONDS, shutdown
    proceeds (we don't want shutdown to block indefinitely either)."""
    from hermes_memory_provider import hermes_llm_adapter

    beam = MagicMock()
    beam.sleep.side_effect = lambda: time.sleep(5.0)
    provider = MnemosyneMemoryProvider()
    provider._beam = beam
    provider.SESSION_END_SLEEP_TIMEOUT_SECONDS = 0.05
    provider.SHUTDOWN_DRAIN_TIMEOUT_SECONDS = 0.2

    provider.on_session_end(messages=[])
    hermes_llm_adapter.register_hermes_host_llm()

    start = time.monotonic()
    with caplog.at_level("DEBUG", logger="hermes_memory_provider"):
        provider.shutdown()
    elapsed = time.monotonic() - start

    # Total shutdown should be bounded by drain timeout + small slack.
    assert elapsed < 1.0, f"shutdown took {elapsed:.2f}s, expected <1s"
    assert get_host_llm_backend() is None
    msgs = [r.getMessage() for r in caplog.records]
    assert any("session-end thread still running" in m for m in msgs), msgs


def test_shutdown_drain_default_matches_design():
    """Production drain default should remain 2s."""
    assert MnemosyneMemoryProvider.SHUTDOWN_DRAIN_TIMEOUT_SECONDS == 2
