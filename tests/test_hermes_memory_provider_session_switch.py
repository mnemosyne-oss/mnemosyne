"""Regression tests for Hermes session lifecycle rebinding."""

from __future__ import annotations

import sqlite3
import threading
from unittest.mock import Mock

from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core.beam import BeamMemory


def test_shutdown_closes_root_audit_log():
    provider = MnemosyneMemoryProvider()
    audit = Mock()
    provider._audit = audit

    provider.shutdown()

    audit.close.assert_called_once_with()
    assert provider._audit is None


def test_session_switch_waits_for_inflight_turn_before_resetting_counter():
    started = threading.Event()
    release = threading.Event()

    class BlockingBeam:
        session_id = "hermes_SESS-A"
        channel_id = "hermes_SESS-A"

        def remember(self, **kwargs):
            started.set()
            assert release.wait(timeout=5)

    provider = MnemosyneMemoryProvider()
    provider._beam = BlockingBeam()
    provider._session_id = "hermes_SESS-A"
    provider._skip_contexts = set()
    provider._agent_context = "primary"
    provider._sync_roles = {"user"}
    provider._auto_sleep_enabled = False
    provider._default_scope = "default"
    provider._should_filter = lambda _content: False
    provider._capture_identity_signals = lambda _content: None

    turn = threading.Thread(
        target=provider.sync_turn,
        args=("user text before switch", ""),
    )
    turn.start()
    assert started.wait(timeout=5)

    switched = threading.Thread(
        target=provider.on_session_switch,
        args=("SESS-B",),
        kwargs={"reset": True},
    )
    switched.start()
    switched.join(timeout=0.05)
    assert switched.is_alive()

    release.set()
    turn.join(timeout=5)
    switched.join(timeout=5)

    assert not turn.is_alive()
    assert not switched.is_alive()
    assert provider._session_id == "hermes_SESS-B"
    assert provider._turn_count == 0


def test_sync_turn_writes_to_the_new_session_after_switch(tmp_path):
    db_path = tmp_path / "mnemosyne.db"
    provider = MnemosyneMemoryProvider()
    beam = BeamMemory(session_id="hermes_SESS-A", db_path=db_path)
    provider._beam = beam
    provider._session_id = "hermes_SESS-A"
    provider._skip_contexts = set()
    provider._agent_context = "primary"
    provider._sync_roles = {"user"}
    provider._auto_sleep_enabled = False
    provider._turn_count = 8
    provider._reflect_calls_this_session = 2

    try:
        provider.on_session_switch(
            "SESS-B",
            parent_session_id="SESS-A",
            reset=True,
            reason="new_session",
        )
        provider.sync_turn(
            "user text after the switch",
            "assistant text after the switch",
            session_id="SESS-B",
        )

        with sqlite3.connect(db_path) as conn:
            rows = conn.execute(
                "SELECT session_id, content FROM working_memory ORDER BY id"
            ).fetchall()

        assert rows == [("hermes_SESS-B", "[USER] user text after the switch")]
        assert provider._session_id == "hermes_SESS-B"
        assert provider._beam.session_id == "hermes_SESS-B"
        assert provider._beam.channel_id == "hermes_SESS-B"
        assert provider._turn_count == 1
        assert provider._reflect_calls_this_session == 0
    finally:
        beam.conn.close()


def test_initialize_gateway_scope_survives_switch_without_gateway_kwarg(tmp_path):
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        "SESS-A",
        hermes_home=str(tmp_path),
        gateway_session_key="gateway-topic",
    )

    try:
        assert provider._session_id == "hermes_gateway-topic"
        assert provider._beam.session_id == "hermes_gateway-topic"
        assert provider._beam.channel_id == "hermes_gateway-topic"

        provider.on_session_switch("transient-child-session")

        assert provider._session_id == "hermes_gateway-topic"
        assert provider._beam.session_id == "hermes_gateway-topic"
        assert provider._beam.channel_id == "hermes_gateway-topic"
    finally:
        beam = provider._beam
        provider.shutdown()
        if beam is not None:
            beam.conn.close()
