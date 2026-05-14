"""Tests for MnemosyneMemoryProvider host-LLM lifecycle hooks.

Covers decisions A6 (bounded on_session_end), C7 (shutdown unregisters
the host backend), and the registration flow added to initialize().
"""

from __future__ import annotations

import json
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


def test_session_end_timeout_default_matches_design(monkeypatch):
    """The production default should remain 15s when no env var is set."""
    monkeypatch.delenv("MNEMOSYNE_SESSION_END_TIMEOUT", raising=False)
    # Re-import so class attrs re-evaluate
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider.SESSION_END_SLEEP_TIMEOUT_SECONDS == 15


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


# ---------------------------------------------------------------------------
# Env var override and validation
# ---------------------------------------------------------------------------

def test_session_end_timeout_default_is_15(monkeypatch):
    """When MNEMOSYNE_SESSION_END_TIMEOUT is absent, default is 15s."""
    monkeypatch.delenv("MNEMOSYNE_SESSION_END_TIMEOUT", raising=False)
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider.SESSION_END_SLEEP_TIMEOUT_SECONDS == 15


def test_session_end_timeout_env_override(monkeypatch):
    """When MNEMOSYNE_SESSION_END_TIMEOUT is set, value is used."""
    monkeypatch.setenv("MNEMOSYNE_SESSION_END_TIMEOUT", "30.5")
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider.SESSION_END_SLEEP_TIMEOUT_SECONDS == 30.5


def test_session_end_timeout_invalid_env_falls_back(monkeypatch):
    """Invalid MNEMOSYNE_SESSION_END_TIMEOUT falls back to default."""
    monkeypatch.setenv("MNEMOSYNE_SESSION_END_TIMEOUT", "not-a-number")
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider.SESSION_END_SLEEP_TIMEOUT_SECONDS == 15


def test_auto_sleep_timeout_default_is_5(monkeypatch):
    """When MNEMOSYNE_AUTO_SLEEP_TIMEOUT is absent, default is 5s."""
    monkeypatch.delenv("MNEMOSYNE_AUTO_SLEEP_TIMEOUT", raising=False)
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider._AUTO_SLEEP_TIMEOUT_SECONDS == 5


def test_auto_sleep_timeout_env_override(monkeypatch):
    """When MNEMOSYNE_AUTO_SLEEP_TIMEOUT is set, value is used."""
    monkeypatch.setenv("MNEMOSYNE_AUTO_SLEEP_TIMEOUT", "10.0")
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider._AUTO_SLEEP_TIMEOUT_SECONDS == 10.0


def test_auto_sleep_timeout_invalid_env_falls_back(monkeypatch):
    """Invalid MNEMOSYNE_AUTO_SLEEP_TIMEOUT falls back to default."""
    monkeypatch.setenv("MNEMOSYNE_AUTO_SLEEP_TIMEOUT", "garbage")
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider._AUTO_SLEEP_TIMEOUT_SECONDS == 5


def test_shutdown_drain_timeout_default_is_2(monkeypatch):
    """When MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT is absent, default is 2s."""
    monkeypatch.delenv("MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT", raising=False)
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider.SHUTDOWN_DRAIN_TIMEOUT_SECONDS == 2


def test_shutdown_drain_timeout_env_override(monkeypatch):
    """When MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT is set, value is used."""
    monkeypatch.setenv("MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT", "12.0")
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider.SHUTDOWN_DRAIN_TIMEOUT_SECONDS == 12.0


def test_shutdown_drain_timeout_invalid_env_falls_back(monkeypatch):
    """Invalid MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT falls back to default."""
    monkeypatch.setenv("MNEMOSYNE_SHUTDOWN_DRAIN_TIMEOUT", "bad")
    import importlib
    import hermes_memory_provider as hmp
    importlib.reload(hmp)
    assert hmp.MnemosyneMemoryProvider.SHUTDOWN_DRAIN_TIMEOUT_SECONDS == 2


# ---------------------------------------------------------------------------
# C12.b — REMEMBER_SCHEMA + _handle_remember per-call kwargs parity
# ---------------------------------------------------------------------------
#
# BeamMemory.remember() accepts extract, metadata, veracity per call. The
# plugin's REMEMBER_SCHEMA used to only expose content/importance/source/
# scope/valid_until/extract_entities, so callers passing any of the missing

def test_remember_schema_advertises_extract_and_metadata_and_veracity():
    """[C12.b] REMEMBER_SCHEMA must advertise the per-call kwargs that
    beam.remember() actually supports, so Hermes' tool-arg validator
    accepts them instead of stripping them as unknown fields."""
    from hermes_memory_provider import REMEMBER_SCHEMA

    props = REMEMBER_SCHEMA["parameters"]["properties"]
    assert "extract" in props, (
        "REMEMBER_SCHEMA missing 'extract' — LLM fact-triple extraction "
        "is unreachable through the plugin"
    )
    assert "metadata" in props, (
        "REMEMBER_SCHEMA missing 'metadata' — caller-supplied tags / "
        "source-doc IDs get silently dropped"
    )
    assert "veracity" in props, (
        "REMEMBER_SCHEMA missing 'veracity' — every plugin-stored memory "
        "defaults to 'unknown', defeating recall's veracity weighting"
    )
    # Sanity-check the advertised types so a typo doesn't slip in.
    assert props["extract"]["type"] == "boolean"
    assert props["metadata"]["type"] == "object"
    assert props["veracity"]["type"] == "string"


def test_handle_remember_passes_extract_metadata_veracity_to_beam(monkeypatch):
    """[C12.b] _handle_remember must forward extract / metadata / veracity
    to beam.remember(). Pre-fix the args were either ignored (no .get())
    or never wired into the beam call."""
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    beam = MagicMock()
    beam.remember.return_value = "mem-123"
    provider._beam = beam

    args = {
        "content": "Sarah leads Project Falcon, started 2026-04-01.",
        "extract": True,
        "metadata": {"source_doc": "kickoff-deck.pdf", "page": 3},
        "veracity": "stated",
    }
    provider._handle_remember(args)

    beam.remember.assert_called_once()
    kwargs = beam.remember.call_args.kwargs
    assert kwargs.get("extract") is True, (
        "extract=True was not forwarded to beam.remember — LLM fact "
        "extraction is unreachable through the plugin tool"
    )
    assert kwargs.get("metadata") == {"source_doc": "kickoff-deck.pdf", "page": 3}, (
        f"metadata not forwarded to beam.remember; got {kwargs.get('metadata')!r}"
    )
    assert kwargs.get("veracity") == "stated", (
        f"veracity not forwarded to beam.remember; got {kwargs.get('veracity')!r}"
    )


def test_handle_remember_defaults_when_new_kwargs_omitted(monkeypatch):
    """[C12.b] Pre-existing callers that don't pass the new kwargs must not
    break: extract defaults False, metadata defaults None, veracity defaults
    'unknown'. Verifies the schema bump is backward-compatible."""
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    beam = MagicMock()
    beam.remember.return_value = "mem-456"
    provider._beam = beam

    provider._handle_remember({"content": "minimal call"})

    kwargs = beam.remember.call_args.kwargs
    assert kwargs.get("extract", False) is False
    # metadata may be None or absent; both are acceptable as "not set"
    assert kwargs.get("metadata") in (None, {}), kwargs.get("metadata")
    # veracity may be "unknown" (passed through) or absent (beam default)
    assert kwargs.get("veracity", "unknown") == "unknown"


def test_handle_remember_clamps_unknown_veracity_to_unknown(monkeypatch, caplog):
    """[C12.b — adversarial review] An LLM typo or a caller passing a
    non-canonical veracity label (e.g. 'STATED' capitalization, 'state'
    truncation, 'random_garbage') must be clamped to 'unknown' at the
    trust boundary. Beam itself does not validate; the row would persist
    with the junk label and pollute the contamination filter
    (`veracity != 'stated'`). Locks the handler-side allowlist."""
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    beam = MagicMock()
    beam.remember.return_value = "mem-789"
    provider._beam = beam

    # 'STATED' (capitalization) — should normalize to 'stated', not clamp.
    provider._handle_remember({"content": "x", "veracity": "STATED"})
    assert beam.remember.call_args.kwargs.get("veracity") == "stated"

    # 'state' (truncated) — not in allowlist, must clamp to 'unknown'.
    beam.remember.reset_mock()
    with caplog.at_level("WARNING", logger="hermes_memory_provider"):
        provider._handle_remember({"content": "y", "veracity": "state"})
    assert beam.remember.call_args.kwargs.get("veracity") == "unknown"
    assert any("unknown veracity" in r.getMessage() for r in caplog.records), (
        "handler should log a warning when clamping a bad veracity label"
    )

    # 'random_garbage' — not in allowlist, must clamp to 'unknown'.
    beam.remember.reset_mock()
    provider._handle_remember({"content": "z", "veracity": "random_garbage"})
    assert beam.remember.call_args.kwargs.get("veracity") == "unknown"


def test_handle_remember_response_echoes_metadata(monkeypatch):
    """[C12.b — adversarial review] The response JSON echoes extract /
    extract_entities / veracity already; metadata should be in the same
    surface for symmetry so callers can confirm what got applied."""
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    beam = MagicMock()
    beam.remember.return_value = "mem-meta"
    provider._beam = beam

    payload = {"content": "x", "metadata": {"source_doc": "deck.pdf", "page": 7}}
    response = provider._handle_remember(payload)
    parsed = json.loads(response)
    assert parsed.get("metadata") == {"source_doc": "deck.pdf", "page": 7}, (
        f"response missing metadata echo: {parsed!r}"
    )


# ---------------------------------------------------------------------------
# Issue #45 followup — RECALL_SCHEMA + _handle_recall scoring weight forwarding
# ---------------------------------------------------------------------------
#
# Adversarial review of issue #45's PR caught that the Hermes-side recall
# surface here also drops vec_weight / fts_weight / importance_weight. The
# RECALL_SCHEMA's description literally says "50% vector + 30% FTS5 + 20%
# importance" but never lets clients tune those weights. Same shape as the
# C12.b REMEMBER fix in this same file — schema mismatch with what
# BeamMemory.recall actually accepts.

def test_recall_schema_advertises_scoring_weights():
    """[issue #45 followup] RECALL_SCHEMA must advertise the per-call scoring
    weights that BeamMemory.recall accepts (beam.py:1296-1298) so Hermes'
    tool-arg validator accepts them instead of stripping as unknown fields."""
    from hermes_memory_provider import RECALL_SCHEMA

    props = RECALL_SCHEMA["parameters"]["properties"]
    for key in ("vec_weight", "fts_weight", "importance_weight"):
        assert key in props, (
            f"RECALL_SCHEMA missing {key!r} — schema description claims "
            f"'50% vector + 30% FTS5 + 20% importance' but never lets the "
            f"client tune those weights"
        )
        assert props[key]["type"] == "number"


def test_handle_recall_forwards_scoring_weights_to_beam(monkeypatch):
    """[issue #45 followup] _handle_recall must forward vec_weight /
    fts_weight / importance_weight when the caller supplies them, so the
    schema-advertised tuning actually takes effect on ranking."""
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    beam = MagicMock()
    beam.recall.return_value = []
    provider._beam = beam

    provider._handle_recall({
        "query": "anything",
        "limit": 3,
        "vec_weight": 0.55,
        "fts_weight": 0.25,
        "importance_weight": 0.20,
    })

    kwargs = beam.recall.call_args.kwargs
    assert kwargs.get("vec_weight") == 0.55, (
        f"_handle_recall did not forward vec_weight; kwargs={kwargs!r}"
    )
    assert kwargs.get("fts_weight") == 0.25
    assert kwargs.get("importance_weight") == 0.20


def test_handle_recall_omits_weights_when_caller_does_not_supply():
    """[issue #45 followup] When caller omits the weight kwargs, the handler
    must NOT pass spurious values to beam.recall — beam treats None as
    "fall back to env var or default" via _normalize_weights, and forcing
    0.0 / 0.5 / etc. would override that resolution."""
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    beam = MagicMock()
    beam.recall.return_value = []
    provider._beam = beam

    provider._handle_recall({"query": "anything", "limit": 3})

    kwargs = beam.recall.call_args.kwargs
    # Acceptable: the kwarg is not in beam.recall's call OR is explicitly None.
    # Failing path: a numeric default like 0.5 / 0.0 leaked through.
    for key in ("vec_weight", "fts_weight", "importance_weight"):
        val = kwargs.get(key, "OMITTED")
        assert val in (None, "OMITTED"), (
            f"_handle_recall forwarded {key}={val!r} when caller omitted it; "
            f"this overrides beam's env/default resolution. Either pass None "
            f"or omit the kwarg entirely."
        )

# ---------------------------------------------------------------------------
# profile_isolation tests — bank-based per-profile memory isolation (issue #72)
# ---------------------------------------------------------------------------


class TestSanitizeBankName:
    """Tests for _sanitize_bank_name static method."""

    def test_normal_name_passes_through(self):
        assert MnemosyneMemoryProvider._sanitize_bank_name("work") == "work"
        assert MnemosyneMemoryProvider._sanitize_bank_name("my_profile") == "my_profile"
        assert MnemosyneMemoryProvider._sanitize_bank_name("test-123") == "test-123"

    def test_lowercases(self):
        assert MnemosyneMemoryProvider._sanitize_bank_name("Work") == "work"
        assert MnemosyneMemoryProvider._sanitize_bank_name("MY_PROFILE") == "my_profile"

    def test_replaces_special_chars_with_underscore(self):
        assert MnemosyneMemoryProvider._sanitize_bank_name("my profile") == "my_profile"
        assert MnemosyneMemoryProvider._sanitize_bank_name("my.profile") == "my_profile"
        assert MnemosyneMemoryProvider._sanitize_bank_name("my@profile!") == "my_profile"
        assert MnemosyneMemoryProvider._sanitize_bank_name("profile (v2)") == "profile_v2"

    def test_collapses_consecutive_underscores(self):
        assert MnemosyneMemoryProvider._sanitize_bank_name("a  b") == "a_b"

    def test_strips_leading_trailing_underscores(self):
        assert MnemosyneMemoryProvider._sanitize_bank_name("_leading") == "leading"
        assert MnemosyneMemoryProvider._sanitize_bank_name("trailing_") == "trailing"
        assert MnemosyneMemoryProvider._sanitize_bank_name("___both___") == "both"

    def test_path_traversal_chars_are_sanitized(self):
        """.. and / replaced with underscores during sanitization, not rejected."""
        assert MnemosyneMemoryProvider._sanitize_bank_name("../../etc") == "etc"
        assert MnemosyneMemoryProvider._sanitize_bank_name("a/b") == "a_b"

    def test_caps_length_at_64(self):
        long_name = "a" * 100
        result = MnemosyneMemoryProvider._sanitize_bank_name(long_name)
        assert len(result) == 64
        assert result == "a" * 64

    def test_empty_returns_default(self):
        assert MnemosyneMemoryProvider._sanitize_bank_name("") == "default"
        assert MnemosyneMemoryProvider._sanitize_bank_name("   ") == "default"
        assert MnemosyneMemoryProvider._sanitize_bank_name("!!!") == "default"

    def test_numeric_start_is_valid(self):
        """Numbers are alphanumeric, pass the start-is-alnum check."""
        assert MnemosyneMemoryProvider._sanitize_bank_name("123name") == "123name"


class TestResolveProfileBank:
    """Tests for _resolve_profile_bank method."""

    def test_uses_agent_identity_first(self):
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = "sphinx"
        provider._hermes_home = "/home/user/.hermes/profiles/work"
        assert provider._resolve_profile_bank() == "sphinx"

    def test_skips_generic_identities(self):
        """'primary', 'default', 'none' fall through to hermes_home."""
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = "primary"
        provider._hermes_home = "/home/user/.hermes/profiles/work"
        assert provider._resolve_profile_bank() == "work"

    def test_falls_back_to_hermes_home_basename(self):
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = ""
        provider._hermes_home = "/home/user/.hermes/profiles/work"
        assert provider._resolve_profile_bank() == "work"

    def test_skips_default_hermes_home(self):
        """'.hermes' and 'hermes' basenames fall through to 'default'."""
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = ""
        provider._hermes_home = "/home/user/.hermes"
        assert provider._resolve_profile_bank() == "default"

    def test_returns_default_when_no_signal(self):
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = ""
        provider._hermes_home = ""
        assert provider._resolve_profile_bank() == "default"

    def test_sanitizes_hermes_home_with_special_chars(self):
        provider = MnemosyneMemoryProvider()
        provider._agent_identity = ""
        provider._hermes_home = "/home/user/.hermes/profiles/My Profile (v2)"
        assert provider._resolve_profile_bank() == "my_profile_v2"


class TestInitializeProfileIsolation:
    """Tests for initialize() with profile_isolation flag."""

    def test_uses_mnemosyne_when_isolation_enabled(self, monkeypatch):
        """profile_isolation=true routes through Mnemosyne(bank=...)."""
        provider = MnemosyneMemoryProvider()
        mem = MagicMock()
        mem.beam = MagicMock()

        # Mnemosyne is imported locally inside initialize()
        import mnemosyne
        monkeypatch.setattr(mnemosyne, "Mnemosyne", MagicMock(return_value=mem))

        provider.initialize(
            session_id="test123",
            agent_identity="sphinx",
            hermes_home="/home/user/.hermes/profiles/sphinx",
            profile_isolation=True,
        )
        mnemosyne.Mnemosyne.assert_called_once()
        kwargs = mnemosyne.Mnemosyne.call_args.kwargs
        assert kwargs["bank"] == "sphinx"
        assert kwargs["session_id"] == "hermes_test123"

    def test_uses_beam_memory_when_isolation_disabled(self, monkeypatch):
        """Default: uses BeamMemory directly (legacy behavior)."""
        provider = MnemosyneMemoryProvider()
        beam_mock = MagicMock()
        beam_constructor = MagicMock(return_value=beam_mock)
        monkeypatch.setattr("hermes_memory_provider._get_beam_class", lambda: beam_constructor)

        provider.initialize(session_id="test123", hermes_home="/some/path")
        beam_constructor.assert_called_once_with(session_id="hermes_test123")

    def test_uses_default_bank_when_no_profile_signal(self, monkeypatch):
        """Isolation ON but no profile signal -> 'default' bank."""
        provider = MnemosyneMemoryProvider()
        mem = MagicMock()
        mem.beam = MagicMock()

        import mnemosyne
        monkeypatch.setattr(mnemosyne, "Mnemosyne", MagicMock(return_value=mem))

        provider.initialize(
            session_id="test123",
            agent_identity="",
            hermes_home="",
            profile_isolation=True,
        )
        kwargs = mnemosyne.Mnemosyne.call_args.kwargs
        assert kwargs["bank"] == "default"

    def test_survives_mnemosyne_failure(self, monkeypatch):
        """Mnemosyne(bank=...) failure logs warning, does not crash."""
        provider = MnemosyneMemoryProvider()

        def boom(*a, **kw):
            raise RuntimeError("db locked")

        import mnemosyne
        monkeypatch.setattr(mnemosyne, "Mnemosyne", MagicMock(side_effect=boom))

        # Must not raise
        provider.initialize(
            session_id="test123",
            agent_identity="sphinx",
            profile_isolation=True,
        )
        assert provider._beam is None

    def test_skips_generic_identity_uses_hermes_home(self, monkeypatch):
        """Identity 'primary' skipped, hermes_home basename used instead."""
        provider = MnemosyneMemoryProvider()
        mem = MagicMock()
        mem.beam = MagicMock()

        import mnemosyne
        monkeypatch.setattr(mnemosyne, "Mnemosyne", MagicMock(return_value=mem))

        provider.initialize(
            session_id="test",
            agent_identity="primary",
            hermes_home="/home/user/.hermes/profiles/work",
            profile_isolation=True,
        )
        kwargs = mnemosyne.Mnemosyne.call_args.kwargs
        assert kwargs["bank"] == "work"
