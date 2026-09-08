from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import mnemosyne_hermes
import pytest
from datetime import datetime as _real_dt
from datetime import timedelta as _real_td
from datetime import timezone as _real_tz
from mnemosyne_hermes import _get_working_memory_ttl_hours
from mnemosyne.core.config import MnemosyneConfig
from mnemosyne_hermes import MnemosyneMemoryProvider


def _shutdown_and_close(provider: MnemosyneMemoryProvider) -> None:
    memory = provider._memory
    beam = provider._beam
    connections = []
    for owner in (memory, beam):
        conn = getattr(owner, "conn", None)
        if conn is not None and all(conn is not existing for existing in connections):
            connections.append(conn)

    provider.shutdown()
    for conn in connections:
        conn.close()


class _EmptyCanonicalStore:
    def list(self, _owner_id: str) -> list[dict[str, Any]]:
        return []


class _RecordingBeam:
    def __init__(self, session_id: str = "hermes_SESS-A") -> None:
        self.session_id = session_id
        self.channel_id = session_id
        self.author_id = None
        self.db_path = None
        self.canonical = _EmptyCanonicalStore()
        self.observed_sessions: list[tuple[str, str]] = []

    def recall(self, **_kwargs: Any) -> list[dict[str, Any]]:
        self.observed_sessions.append((self.session_id, self.channel_id))
        return []

    def remember(self, **_kwargs: Any) -> None:
        self.observed_sessions.append((self.session_id, self.channel_id))


class _ObservedRLock:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.waiting = threading.Event()

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        self.waiting.set()
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> _ObservedRLock:
        self.acquire()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


def _provider_with_recording_beam(
    *,
    session_id: str = "hermes_SESS-A",
    gateway_session_key: str = "",
) -> tuple[MnemosyneMemoryProvider, _RecordingBeam]:
    provider = MnemosyneMemoryProvider()
    beam = _RecordingBeam(session_id)
    provider._beam = beam
    provider._session_id = session_id
    provider._gateway_session_key = gateway_session_key
    provider._agent_context = "primary"
    provider._skip_contexts = set()
    provider._sync_roles = {"user"}
    provider._auto_sleep_enabled = False
    provider._should_filter = lambda _content: False
    provider._capture_identity_signals = lambda _content: None
    return provider, beam


def _call_surface(
    provider: MnemosyneMemoryProvider,
    surface: str,
    session_id: str,
) -> None:
    if surface == "prefetch":
        provider.prefetch("query for active session", session_id=session_id)
    else:
        provider.sync_turn(
            "user text for active session",
            "",
            session_id=session_id,
        )


def test_switch_then_sync_turn_writes_new_session_and_resets_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(MnemosyneConfig, "_instance", None)
    provider = MnemosyneMemoryProvider()
    provider.initialize("SESS-A", hermes_home=str(tmp_path), auto_sleep=False)
    beam = provider._beam
    assert beam is not None
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
            "",
            session_id="SESS-B",
        )

        conn = sqlite3.connect(beam.db_path)
        try:
            rows = conn.execute(
                "SELECT session_id, channel_id, content FROM working_memory ORDER BY id"
            ).fetchall()
        finally:
            conn.close()

        assert rows == [
            (
                "hermes_SESS-B",
                "hermes_SESS-B",
                "[USER] user text after the switch",
            )
        ]
        assert provider._session_id == "hermes_SESS-B"
        assert beam.session_id == "hermes_SESS-B"
        assert beam.channel_id == "hermes_SESS-B"
        assert provider._turn_count == 1
        assert provider._reflect_calls_this_session == 0
    finally:
        _shutdown_and_close(provider)


def test_switch_then_memory_write_uses_new_session_in_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(MnemosyneConfig, "_instance", None)
    provider = MnemosyneMemoryProvider()
    provider.initialize("SESS-A", hermes_home=str(tmp_path), auto_sleep=False)
    beam = provider._beam
    assert beam is not None

    try:
        provider.on_session_switch("SESS-B")
        provider.on_memory_write("add", "project", "builtin write after switch")

        row = beam.conn.execute(
            "SELECT session_id, channel_id, source, scope FROM working_memory "
            "WHERE content = ?",
            ("builtin write after switch",),
        ).fetchone()

        assert tuple(row) == (
            "hermes_SESS-B",
            "hermes_SESS-B",
            "builtin_memory_project",
            "session",
        )
    finally:
        _shutdown_and_close(provider)


def test_switch_without_reset_preserves_session_counters() -> None:
    provider, beam = _provider_with_recording_beam()
    provider._turn_count = 7
    provider._reflect_calls_this_session = 2

    provider.on_session_switch("SESS-B", reset=False)

    assert provider._session_id == "hermes_SESS-B"
    assert beam.session_id == "hermes_SESS-B"
    assert provider._turn_count == 7
    assert provider._reflect_calls_this_session == 2


@pytest.mark.parametrize("empty_id", ["", "   "])
def test_empty_session_id_is_a_no_op(empty_id: str) -> None:
    provider, beam = _provider_with_recording_beam()
    provider._turn_count = 7
    provider._reflect_calls_this_session = 2

    provider.on_session_switch(empty_id, reset=True)

    assert provider._session_id == "hermes_SESS-A"
    assert beam.session_id == "hermes_SESS-A"
    assert beam.channel_id == "hermes_SESS-A"
    assert provider._turn_count == 7
    assert provider._reflect_calls_this_session == 2


def test_switch_without_beam_updates_provider_state_without_raising() -> None:
    provider = MnemosyneMemoryProvider()
    provider._beam = None

    provider.on_session_switch("SESS-B")

    assert provider._session_id == "hermes_SESS-B"


@pytest.mark.parametrize("surface", ["prefetch", "sync_turn"])
@pytest.mark.parametrize(
    ("session_id", "expected_session"),
    [("SESS-B", "hermes_SESS-B"), ("", "hermes_SESS-A")],
)
def test_per_call_session_id_scopes_only_the_locked_beam_access(
    surface: str,
    session_id: str,
    expected_session: str,
) -> None:
    provider, beam = _provider_with_recording_beam()

    _call_surface(provider, surface, session_id)

    assert beam.observed_sessions == [(expected_session, expected_session)]
    assert provider._session_id == "hermes_SESS-A"
    assert beam.session_id == "hermes_SESS-A"
    assert beam.channel_id == "hermes_SESS-A"


@pytest.mark.parametrize("surface", ["prefetch", "sync_turn"])
def test_per_call_session_id_cannot_override_gateway_scope(surface: str) -> None:
    provider, beam = _provider_with_recording_beam(
        session_id="hermes_gateway-topic",
        gateway_session_key="gateway-topic",
    )

    _call_surface(provider, surface, "transient-child-session")

    assert beam.observed_sessions == [("hermes_gateway-topic", "hermes_gateway-topic")]
    assert provider._session_id == "hermes_gateway-topic"
    assert beam.session_id == "hermes_gateway-topic"
    assert beam.channel_id == "hermes_gateway-topic"


def test_temporary_sync_scope_does_not_advance_durable_session_counters() -> None:
    provider, beam = _provider_with_recording_beam()
    provider._turn_count = 9
    provider._auto_sleep_enabled = True
    auto_sleep_sessions: list[tuple[str, str]] = []
    provider._maybe_auto_sleep = lambda: auto_sleep_sessions.append(
        (provider._session_id, beam.session_id)
    )

    provider.sync_turn(
        "temporary child session turn",
        "",
        session_id="SESS-B",
    )

    assert beam.observed_sessions == [("hermes_SESS-B", "hermes_SESS-B")]
    assert provider._turn_count == 9
    assert auto_sleep_sessions == []
    assert provider._session_id == "hermes_SESS-A"
    assert beam.session_id == "hermes_SESS-A"
    assert beam.channel_id == "hermes_SESS-A"


def test_gateway_scoped_sync_turn_advances_the_durable_counter() -> None:
    provider, beam = _provider_with_recording_beam(
        session_id="hermes_gateway-topic",
        gateway_session_key="gateway-topic",
    )
    provider._turn_count = 8
    provider._auto_sleep_enabled = False

    provider.sync_turn(
        "gateway child session turn",
        "",
        session_id="transient-child-session",
    )

    assert beam.observed_sessions == [("hermes_gateway-topic", "hermes_gateway-topic")]
    assert provider._turn_count == 9
    assert provider._session_id == "hermes_gateway-topic"


def test_auto_sleep_trigger_revalidates_the_durable_turn_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auto_sleep_entered = threading.Event()
    release_auto_sleep = threading.Event()
    expected_sessions: list[str] = []

    class _SourceBeam(_RecordingBeam):
        db_path = "memory.db"
        author_type = "agent"

        def get_working_stats(self) -> dict[str, int]:
            return {"total": 2}

        def _count_unconsolidated_before(self, _cutoff: str) -> int:
            return 1

        def sleep_all_sessions(self) -> None:
            raise AssertionError("auto-sleep must use a worker-local Beam")

    class _WorkerBeam:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any) -> None:
            type(self).calls.append(dict(kwargs))

        def sleep_all_sessions(self) -> None:
            raise AssertionError("a stale auto-sleep trigger must be discarded")

    provider, _ = _provider_with_recording_beam()
    provider._beam = _SourceBeam()
    provider._turn_count = 9
    provider._auto_sleep_enabled = True
    provider._auto_sleep_threshold = 1
    provider._reserve_reflection_budget_locked = lambda _reason: None
    original_auto_sleep = provider._maybe_auto_sleep

    def blocked_auto_sleep(*, expected_session_id: str = "") -> None:
        expected_sessions.append(expected_session_id)
        auto_sleep_entered.set()
        if not release_auto_sleep.wait(timeout=5):
            raise TimeoutError("auto-sleep trigger was not released")
        original_auto_sleep(expected_session_id=expected_session_id)

    provider._maybe_auto_sleep = blocked_auto_sleep
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _WorkerBeam)

    sync_turn = threading.Thread(
        target=provider.sync_turn,
        args=("durable session turn before switch", ""),
    )
    sync_turn.start()
    assert auto_sleep_entered.wait(timeout=5)

    provider.on_session_switch("SESS-B", reset=True)
    release_auto_sleep.set()
    sync_turn.join(timeout=5)

    assert not sync_turn.is_alive()
    assert expected_sessions == ["hermes_SESS-A"]
    assert _WorkerBeam.calls == []
    assert provider._sync_turn_diagnostics()["failed"] == 0
    assert provider._session_id == "hermes_SESS-B"
    assert provider._turn_count == 0


def test_gateway_scope_survives_compression_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(MnemosyneConfig, "_instance", None)
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        "SESS-A",
        hermes_home=str(tmp_path),
        gateway_session_key="gateway-topic",
        auto_sleep=False,
    )
    beam = provider._beam
    assert beam is not None

    try:
        provider.on_session_switch(
            "compression-child-session",
            parent_session_id="SESS-A",
            reason="compression",
        )

        assert provider._session_id == "hermes_gateway-topic"
        assert beam.session_id == "hermes_gateway-topic"
        assert beam.channel_id == "hermes_gateway-topic"
    finally:
        _shutdown_and_close(provider)


def test_profile_isolation_switch_preserves_bank_db_and_explicit_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(MnemosyneConfig, "_instance", None)
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        "SESS-A",
        hermes_home=str(tmp_path / "profile-home"),
        agent_identity="profile-alpha",
        profile_isolation=True,
        channel_id="caller-channel",
        auto_sleep=False,
    )
    memory = provider._memory
    beam = provider._beam
    assert memory is not None
    assert beam is not None
    original_bank = memory.bank
    original_db_path = memory.db_path

    try:
        provider.on_session_switch("SESS-B")

        assert provider._memory is memory
        assert memory.bank == original_bank
        assert memory.db_path == original_db_path
        assert beam.db_path == original_db_path
        assert provider._session_id == "hermes_SESS-B"
        assert memory.session_id == "hermes_SESS-B"
        assert beam.session_id == "hermes_SESS-B"
        assert memory.channel_id == "caller-channel"
        assert beam.channel_id == "caller-channel"
    finally:
        _shutdown_and_close(provider)


def test_switch_waits_for_inflight_turn_without_mixing_sessions() -> None:
    turn_started = threading.Event()
    release_turn = threading.Event()
    switch_done = threading.Event()
    observations: list[tuple[tuple[str, str], bool, tuple[str, str]]] = []
    thread_errors: list[BaseException] = []

    class _BlockingBeam(_RecordingBeam):
        def remember(self, **_kwargs: Any) -> None:
            before = (self.session_id, self.channel_id)
            turn_started.set()
            released = release_turn.wait(timeout=5)
            after = (self.session_id, self.channel_id)
            observations.append((before, released, after))

    provider, _ = _provider_with_recording_beam()
    beam = _BlockingBeam()
    provider._beam = beam
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock

    def run_turn() -> None:
        try:
            provider.sync_turn(
                "user text before switch",
                "",
                session_id="SESS-A",
            )
        except BaseException as exc:
            thread_errors.append(exc)

    def run_switch() -> None:
        try:
            provider.on_session_switch("SESS-B", reset=True)
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            switch_done.set()

    turn = threading.Thread(target=run_turn)
    turn.start()
    assert turn_started.wait(timeout=5)
    beam_lock.waiting.clear()

    switch = threading.Thread(target=run_switch)
    switch.start()
    assert beam_lock.waiting.wait(timeout=5)
    switch_blocked = not switch_done.wait(timeout=0.1)

    release_turn.set()
    turn.join(timeout=5)
    switch.join(timeout=5)

    assert not turn.is_alive()
    assert not switch.is_alive()
    assert not thread_errors
    assert switch_blocked
    assert switch_done.is_set()
    assert observations == [
        (
            ("hermes_SESS-A", "hermes_SESS-A"),
            True,
            ("hermes_SESS-A", "hermes_SESS-A"),
        )
    ]
    assert provider._session_id == "hermes_SESS-B"
    assert beam.session_id == "hermes_SESS-B"
    assert beam.channel_id == "hermes_SESS-B"
    assert provider._turn_count == 0


def test_switch_waits_for_inflight_tool_write_without_mixing_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_started = threading.Event()
    release_tool = threading.Event()
    switch_done = threading.Event()
    observations: list[tuple[tuple[str, str], bool, tuple[str, str]]] = []
    tool_results: list[str] = []
    thread_errors: list[BaseException] = []

    class _BlockingToolBeam(_RecordingBeam):
        def remember(self, **_kwargs: Any) -> str:
            before = (self.session_id, self.channel_id)
            tool_started.set()
            released = release_tool.wait(timeout=5)
            after = (self.session_id, self.channel_id)
            observations.append((before, released, after))
            return "memory-id"

    provider, _ = _provider_with_recording_beam()
    beam = _BlockingToolBeam()
    provider._beam = beam
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    provider.has_tool = lambda name: name == "mnemosyne_remember"
    monkeypatch.setattr(mnemosyne_hermes, "_write_approval_enabled", lambda: False)

    def run_tool() -> None:
        try:
            tool_results.append(
                provider.handle_tool_call(
                    "mnemosyne_remember",
                    {"content": "tool write before switch"},
                )
            )
        except BaseException as exc:
            thread_errors.append(exc)

    def run_switch() -> None:
        try:
            provider.on_session_switch("SESS-B")
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            switch_done.set()

    tool = threading.Thread(target=run_tool)
    tool.start()
    assert tool_started.wait(timeout=5)
    beam_lock.waiting.clear()

    switch = threading.Thread(target=run_switch)
    switch.start()
    try:
        assert beam_lock.waiting.wait(timeout=5)
        assert not switch_done.is_set()
    finally:
        release_tool.set()
        tool.join(timeout=5)
        switch.join(timeout=5)

    assert not tool.is_alive()
    assert not switch.is_alive()
    assert not thread_errors
    assert switch_done.is_set()
    assert len(tool_results) == 1
    assert '"status": "stored"' in tool_results[0]
    assert observations == [
        (
            ("hermes_SESS-A", "hermes_SESS-A"),
            True,
            ("hermes_SESS-A", "hermes_SESS-A"),
        )
    ]
    assert provider._session_id == "hermes_SESS-B"
    assert beam.session_id == "hermes_SESS-B"
    assert beam.channel_id == "hermes_SESS-B"


def test_switch_waits_for_inflight_memory_write_without_mixing_sessions() -> None:
    write_started = threading.Event()
    release_write = threading.Event()
    switch_done = threading.Event()
    observations: list[tuple[tuple[str, str], bool, tuple[str, str]]] = []
    thread_errors: list[BaseException] = []

    class _BlockingBeam(_RecordingBeam):
        def remember(self, **_kwargs: Any) -> None:
            before = (self.session_id, self.channel_id)
            write_started.set()
            released = release_write.wait(timeout=5)
            after = (self.session_id, self.channel_id)
            observations.append((before, released, after))

    provider, _ = _provider_with_recording_beam()
    beam = _BlockingBeam()
    provider._beam = beam
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock

    def write() -> None:
        try:
            provider.on_memory_write("add", "session", "memory before switch")
        except BaseException as exc:
            thread_errors.append(exc)

    def switch() -> None:
        try:
            provider.on_session_switch("SESS-B")
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            switch_done.set()

    write_thread = threading.Thread(target=write)
    write_thread.start()
    assert write_started.wait(timeout=5)
    beam_lock.waiting.clear()

    switch_thread = threading.Thread(target=switch)
    switch_thread.start()
    assert beam_lock.waiting.wait(timeout=5)
    switch_blocked = not switch_done.wait(timeout=0.1)

    release_write.set()
    write_thread.join(timeout=5)
    switch_thread.join(timeout=5)

    assert not write_thread.is_alive()
    assert not switch_thread.is_alive()
    assert not thread_errors
    assert switch_blocked
    assert switch_done.is_set()
    assert observations == [
        (
            ("hermes_SESS-A", "hermes_SESS-A"),
            True,
            ("hermes_SESS-A", "hermes_SESS-A"),
        )
    ]
    assert provider._session_id == "hermes_SESS-B"
    assert beam.session_id == "hermes_SESS-B"
    assert beam.channel_id == "hermes_SESS-B"


def test_failed_init_switches_pending_retry_to_latest_gateway_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    class _FailOnceBeam:
        author_id = None

        def __init__(self, **kwargs: Any) -> None:
            calls.append(dict(kwargs))
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            self.session_id = kwargs["session_id"]
            self.channel_id = kwargs.get("channel_id") or self.session_id
            self.db_path = kwargs.get("db_path")

    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _FailOnceBeam)
    provider = MnemosyneMemoryProvider()
    provider.initialize("SESS-A", gateway_session_key="gateway-topic")
    assert provider._retry_init_args is not None

    provider.on_session_switch("SESS-B")
    assert provider._retry_init_args[0] == "SESS-B"
    assert provider._retry_init_args[1]["gateway_session_key"] == "gateway-topic"
    provider._retry_init_at = 0
    provider._maybe_retry_init()

    assert provider._retry_init_args is None
    assert provider._session_id == "hermes_gateway-topic"
    assert provider._beam is not None
    assert provider._beam.session_id == "hermes_gateway-topic"
    assert [call["session_id"] for call in calls] == [
        "hermes_gateway-topic",
        "hermes_gateway-topic",
    ]
    provider.shutdown()


def test_retry_cannot_publish_old_session_after_switch() -> None:
    provider = MnemosyneMemoryProvider()
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    provider._retry_init_args = ("SESS-A", {})
    provider._retry_init_at = 0
    init_started = threading.Event()
    release_init = threading.Event()
    switch_done = threading.Event()
    init_releases: list[bool] = []
    thread_errors: list[BaseException] = []

    def blocked_initialize(session_id: str, **_kwargs: Any) -> None:
        init_started.set()
        init_releases.append(release_init.wait(timeout=5))
        provider._beam = _RecordingBeam(f"hermes_{session_id}")
        provider._session_id = f"hermes_{session_id}"

    provider.initialize = blocked_initialize

    def run_retry() -> None:
        try:
            provider._maybe_retry_init()
        except BaseException as exc:
            thread_errors.append(exc)

    def run_switch() -> None:
        try:
            provider.on_session_switch("SESS-B")
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            switch_done.set()

    retry = threading.Thread(target=run_retry)
    retry.start()
    assert init_started.wait(timeout=5)
    beam_lock.waiting.clear()

    switch = threading.Thread(target=run_switch)
    switch.start()
    assert beam_lock.waiting.wait(timeout=5)
    switch_blocked = not switch_done.wait(timeout=0.1)

    release_init.set()
    retry.join(timeout=5)
    switch.join(timeout=5)

    assert not retry.is_alive()
    assert not switch.is_alive()
    assert not thread_errors
    assert init_releases == [True]
    assert switch_blocked
    assert switch_done.is_set()
    assert provider._session_id == "hermes_SESS-B"
    assert provider._beam is not None
    assert provider._beam.session_id == "hermes_SESS-B"
    assert provider._beam.channel_id == "hermes_SESS-B"


def test_direct_initialize_is_atomic_with_session_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_started = threading.Event()
    release_init = threading.Event()
    switch_done = threading.Event()
    init_releases: list[bool] = []
    thread_errors: list[BaseException] = []

    class _BlockingBeam:
        author_id = None

        def __init__(self, **kwargs: Any) -> None:
            self.session_id = kwargs["session_id"]
            self.channel_id = kwargs.get("channel_id") or self.session_id
            self.db_path = kwargs.get("db_path")
            init_started.set()
            init_releases.append(release_init.wait(timeout=5))

    provider = MnemosyneMemoryProvider()
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _BlockingBeam)

    def initialize() -> None:
        try:
            provider.initialize("SESS-A", auto_sleep=False)
        except BaseException as exc:
            thread_errors.append(exc)

    def switch() -> None:
        try:
            provider.on_session_switch("SESS-B")
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            switch_done.set()

    init_thread = threading.Thread(target=initialize)
    init_thread.start()
    assert init_started.wait(timeout=5)
    beam_lock.waiting.clear()

    switch_thread = threading.Thread(target=switch)
    switch_thread.start()
    assert beam_lock.waiting.wait(timeout=5)
    switch_blocked = not switch_done.wait(timeout=0.1)

    release_init.set()
    init_thread.join(timeout=5)
    switch_thread.join(timeout=5)

    assert not init_thread.is_alive()
    assert not switch_thread.is_alive()
    assert not thread_errors
    assert init_releases == [True]
    assert switch_blocked
    assert switch_done.is_set()
    assert provider._session_id == "hermes_SESS-B"
    assert provider._beam is not None
    assert provider._beam.session_id == "hermes_SESS-B"
    assert provider._beam.channel_id == "hermes_SESS-B"
    provider.shutdown()


def test_direct_initialize_blocks_sync_turn_until_beam_is_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_started = threading.Event()
    release_init = threading.Event()
    sync_done = threading.Event()
    writes: list[tuple[str, str]] = []
    thread_errors: list[BaseException] = []

    class _BlockingBeam:
        author_id = None

        def __init__(self, **kwargs: Any) -> None:
            self.session_id = kwargs["session_id"]
            self.channel_id = kwargs.get("channel_id") or self.session_id
            self.db_path = kwargs.get("db_path")
            init_started.set()
            if not release_init.wait(timeout=5):
                thread_errors.append(TimeoutError("initialize was not released"))

        def remember(self, **_kwargs: Any) -> None:
            writes.append((self.session_id, self.channel_id))

    provider = MnemosyneMemoryProvider()
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _BlockingBeam)

    def initialize() -> None:
        try:
            provider.initialize("SESS-A", auto_sleep=False)
        except BaseException as exc:
            thread_errors.append(exc)

    def sync() -> None:
        try:
            provider.sync_turn("turn submitted during initialize", "")
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            sync_done.set()

    init_thread = threading.Thread(target=initialize)
    init_thread.start()
    assert init_started.wait(timeout=5)
    beam_lock.waiting.clear()

    sync_thread = threading.Thread(target=sync)
    sync_thread.start()
    assert beam_lock.waiting.wait(timeout=5)
    sync_blocked = not sync_done.wait(timeout=0.1)

    release_init.set()
    init_thread.join(timeout=5)
    sync_thread.join(timeout=5)

    assert not init_thread.is_alive()
    assert not sync_thread.is_alive()
    assert not thread_errors
    assert sync_blocked
    assert sync_done.is_set()
    assert writes == [("hermes_SESS-A", "hermes_SESS-A")]
    provider.shutdown()


def test_explicit_channel_collision_survives_real_initialize_and_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(MnemosyneConfig, "_instance", None)
    provider = MnemosyneMemoryProvider()
    provider.initialize(
        "SESS-A",
        hermes_home=str(tmp_path),
        channel_id="hermes_SESS-A",
        auto_sleep=False,
    )
    beam = provider._beam
    assert beam is not None

    try:
        provider.on_session_switch("SESS-B")

        assert provider._session_id == "hermes_SESS-B"
        assert beam.session_id == "hermes_SESS-B"
        assert beam.channel_id == "hermes_SESS-A"
    finally:
        _shutdown_and_close(provider)


def test_session_end_reservation_and_snapshot_share_switch_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation_entered = threading.Event()
    release_reservation = threading.Event()
    worker_started = threading.Event()
    release_worker = threading.Event()
    switch_done = threading.Event()
    reservation_releases: list[bool] = []
    worker_releases: list[bool] = []
    thread_errors: list[BaseException] = []

    class _SourceBeam:
        session_id = "hermes_SESS-A"
        db_path = "memory.db"
        author_id = "author"
        author_type = "agent"
        channel_id = "channel"
        canonical_owner_id = "profile-owner"
        agent_context = "cron"

    class _SleepBeam:
        calls: list[dict[str, Any]] = []
        identities: list[tuple[str | None, str | None]] = []

        def __init__(self, **kwargs: Any) -> None:
            type(self).calls.append(dict(kwargs))

        def sleep(self) -> None:
            type(self).identities.append(
                (
                    getattr(self, "canonical_owner_id", None),
                    getattr(self, "agent_context", None),
                )
            )
            worker_started.set()
            worker_releases.append(release_worker.wait(timeout=5))

    provider = MnemosyneMemoryProvider()
    provider._beam = _SourceBeam()
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    provider.SESSION_END_SLEEP_TIMEOUT_SECONDS = 1

    def reserve_locked(_reason: str) -> None:
        reservation_entered.set()
        reservation_releases.append(release_reservation.wait(timeout=5))

    provider._reserve_reflection_budget_locked = reserve_locked
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _SleepBeam)

    def run_session_end() -> None:
        try:
            provider.on_session_end([])
        except BaseException as exc:
            thread_errors.append(exc)

    def run_switch() -> None:
        try:
            provider.on_session_switch("SESS-B")
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            switch_done.set()

    session_end = threading.Thread(target=run_session_end)
    session_end.start()
    assert reservation_entered.wait(timeout=5)
    beam_lock.waiting.clear()

    switch = threading.Thread(target=run_switch)
    switch.start()
    assert beam_lock.waiting.wait(timeout=5)
    switch_blocked = not switch_done.wait(timeout=0.1)

    release_reservation.set()
    assert worker_started.wait(timeout=5)
    release_worker.set()
    session_end.join(timeout=5)
    switch.join(timeout=5)
    if provider._session_end_thread is not None:
        provider._session_end_thread.join(timeout=5)

    assert not session_end.is_alive()
    assert not switch.is_alive()
    assert not thread_errors
    assert reservation_releases == [True]
    assert worker_releases == [True]
    assert switch_blocked
    assert switch_done.is_set()
    assert _SleepBeam.calls == [
        {
            "session_id": "hermes_SESS-A",
            "db_path": "memory.db",
            "author_id": "author",
            "author_type": "agent",
            "channel_id": "channel",
        }
    ]
    assert _SleepBeam.identities == [("profile-owner", "cron")]


def test_session_end_sleep_blocks_concurrent_sync_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_started = threading.Event()
    release_sleep = threading.Event()
    turn_write_started = threading.Event()
    writes: list[tuple[str, str]] = []

    class _SourceBeam(_RecordingBeam):
        db_path = "memory.db"
        author_id = "author"
        author_type = "agent"

        def remember(self, **_kwargs: Any) -> None:
            turn_write_started.set()
            writes.append((self.session_id, self.channel_id))

    class _SleepBeam:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def sleep(self) -> None:
            sleep_started.set()
            assert release_sleep.wait(timeout=5)

    provider, _ = _provider_with_recording_beam()
    provider._beam = _SourceBeam()
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    provider.SESSION_END_SLEEP_TIMEOUT_SECONDS = 1
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _SleepBeam)

    session_end = threading.Thread(target=provider.on_session_end, args=([],))
    session_end.start()
    assert sleep_started.wait(timeout=5)
    beam_lock.waiting.clear()

    sync_turn = threading.Thread(
        target=provider.sync_turn,
        args=("user write while session-end sleep runs", ""),
    )
    sync_turn.start()
    try:
        assert beam_lock.waiting.wait(timeout=5)
        assert not turn_write_started.is_set()
    finally:
        release_sleep.set()
        session_end.join(timeout=5)
        sync_turn.join(timeout=5)
        if provider._session_end_thread is not None:
            provider._session_end_thread.join(timeout=5)

    assert not session_end.is_alive()
    assert not sync_turn.is_alive()
    assert turn_write_started.is_set()
    assert writes == [("hermes_SESS-A", "hermes_SESS-A")]


def test_session_end_sleep_blocks_concurrent_memory_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleep_started = threading.Event()
    release_sleep = threading.Event()
    write_started = threading.Event()
    sleep_releases: list[bool] = []
    writes: list[tuple[str, str]] = []
    thread_errors: list[BaseException] = []

    class _SourceBeam(_RecordingBeam):
        db_path = "memory.db"
        author_id = "author"
        author_type = "agent"

        def remember(self, **_kwargs: Any) -> None:
            write_started.set()
            writes.append((self.session_id, self.channel_id))

    class _SleepBeam:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def sleep(self) -> None:
            sleep_started.set()
            sleep_releases.append(release_sleep.wait(timeout=5))

    provider, _ = _provider_with_recording_beam()
    provider._beam = _SourceBeam()
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    provider.SESSION_END_SLEEP_TIMEOUT_SECONDS = 10
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _SleepBeam)

    def end_session() -> None:
        try:
            provider.on_session_end([])
        except BaseException as exc:
            thread_errors.append(exc)

    def write() -> None:
        try:
            provider.on_memory_write("add", "session", "write during sleep")
        except BaseException as exc:
            thread_errors.append(exc)

    session_end = threading.Thread(target=end_session)
    session_end.start()
    assert sleep_started.wait(timeout=5)
    beam_lock.waiting.clear()

    write_thread = threading.Thread(target=write)
    write_thread.start()
    attempted_lock = beam_lock.waiting.wait(timeout=5)
    write_blocked = not write_started.wait(timeout=0.1)

    release_sleep.set()
    session_end.join(timeout=5)
    write_thread.join(timeout=5)
    if provider._session_end_thread is not None:
        provider._session_end_thread.join(timeout=5)

    assert not session_end.is_alive()
    assert not write_thread.is_alive()
    assert not thread_errors
    assert sleep_releases == [True]
    assert attempted_lock
    assert write_blocked
    assert write_started.is_set()
    assert writes == [("hermes_SESS-A", "hermes_SESS-A")]


def test_auto_sleep_reservation_and_snapshot_share_switch_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reservation_entered = threading.Event()
    release_reservation = threading.Event()
    worker_started = threading.Event()
    release_worker = threading.Event()
    switch_done = threading.Event()
    thread_errors: list[BaseException] = []
    reservation_releases: list[bool] = []
    worker_releases: list[bool] = []

    class _SourceBeam:
        session_id = "hermes_SESS-A"
        db_path = "memory.db"
        author_id = "author"
        author_type = "agent"
        channel_id = "channel"
        canonical_owner_id = "profile-owner"
        agent_context = "cron"

        def get_working_stats(self) -> dict[str, int]:
            return {"total": 2}

        def _count_unconsolidated_before(self, _cutoff: str) -> int:
            return 1

        def sleep(self) -> None:
            raise AssertionError("auto-sleep must use a worker-local Beam")

        def sleep_all_sessions(self) -> None:
            raise AssertionError("auto-sleep must use a worker-local Beam")

    class _WorkerBeam:
        calls: list[dict[str, Any]] = []
        identities: list[tuple[str | None, str | None]] = []

        def __init__(self, **kwargs: Any) -> None:
            type(self).calls.append(dict(kwargs))

        def sleep(self) -> None:
            type(self).identities.append(
                (
                    getattr(self, "canonical_owner_id", None),
                    getattr(self, "agent_context", None),
                )
            )
            worker_started.set()
            worker_releases.append(release_worker.wait(timeout=5))

        def sleep_all_sessions(self) -> None:
            type(self).identities.append(
                (
                    getattr(self, "canonical_owner_id", None),
                    getattr(self, "agent_context", None),
                )
            )
            worker_started.set()
            worker_releases.append(release_worker.wait(timeout=5))

    provider = MnemosyneMemoryProvider()
    provider._beam = _SourceBeam()
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    provider._auto_sleep_threshold = 1
    provider._auto_sleep_enabled = True
    provider._AUTO_SLEEP_TIMEOUT_SECONDS = 1

    def reserve_locked(_reason: str) -> None:
        reservation_entered.set()
        reservation_releases.append(release_reservation.wait(timeout=5))

    provider._reserve_reflection_budget_locked = reserve_locked
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _WorkerBeam)

    def run_auto_sleep() -> None:
        try:
            provider._maybe_auto_sleep()
        except BaseException as exc:
            thread_errors.append(exc)

    auto_sleep = threading.Thread(target=run_auto_sleep)
    auto_sleep.start()
    assert reservation_entered.wait(timeout=5), thread_errors
    assert not thread_errors
    beam_lock.waiting.clear()

    def run_switch() -> None:
        try:
            provider.on_session_switch("SESS-B")
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            switch_done.set()

    switch = threading.Thread(target=run_switch)
    switch.start()
    assert beam_lock.waiting.wait(timeout=5)
    switch_blocked = not switch_done.wait(timeout=0.1)

    release_reservation.set()
    assert worker_started.wait(timeout=5)
    release_worker.set()
    auto_sleep.join(timeout=5)
    switch.join(timeout=5)

    assert not auto_sleep.is_alive()
    assert not switch.is_alive()
    assert not thread_errors
    assert reservation_releases == [True]
    assert worker_releases == [True]
    assert switch_blocked
    assert switch_done.is_set()
    assert _WorkerBeam.calls == [
        {
            "session_id": "hermes_SESS-A",
            "db_path": "memory.db",
            "author_id": "author",
            "author_type": "agent",
            "channel_id": "channel",
        }
    ]
    assert _WorkerBeam.identities == [("profile-owner", "cron")]


def test_auto_sleep_eligibility_and_snapshot_share_switch_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stats_started = threading.Event()
    release_stats = threading.Event()
    worker_started = threading.Event()
    release_worker = threading.Event()
    switch_done = threading.Event()
    stats_sessions: list[tuple[str, str]] = []
    eligibility_sessions: list[tuple[str, str]] = []
    thread_errors: list[BaseException] = []

    class _SourceBeam:
        session_id = "hermes_SESS-A"
        channel_id = "hermes_SESS-A"
        db_path = "memory.db"
        author_id = "author"
        author_type = "agent"
        canonical_owner_id = "profile-owner"
        agent_context = "primary"

        def get_working_stats(self) -> dict[str, int]:
            stats_sessions.append((self.session_id, self.channel_id))
            stats_started.set()
            if not release_stats.wait(timeout=5):
                raise TimeoutError("auto-sleep stats were not released")
            return {"total": 2}

        def _count_unconsolidated_before(self, _cutoff: str) -> int:
            eligibility_sessions.append((self.session_id, self.channel_id))
            return 1

        def sleep_all_sessions(self) -> None:
            raise AssertionError("auto-sleep must use a worker-local Beam")

    class _WorkerBeam:
        calls: list[dict[str, Any]] = []

        def __init__(self, **kwargs: Any) -> None:
            type(self).calls.append(dict(kwargs))

        def sleep(self) -> None:
            worker_started.set()
            if not release_worker.wait(timeout=5):
                raise TimeoutError("auto-sleep worker was not released")

    provider = MnemosyneMemoryProvider()
    provider._beam = _SourceBeam()
    provider._session_id = "hermes_SESS-A"
    provider._auto_sleep_threshold = 1
    provider._auto_sleep_enabled = True
    provider._AUTO_SLEEP_TIMEOUT_SECONDS = 1
    beam_lock = _ObservedRLock()
    provider._beam_access_lock = beam_lock
    provider._reserve_reflection_budget_locked = lambda _reason: None
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _WorkerBeam)

    def run_auto_sleep() -> None:
        try:
            provider._maybe_auto_sleep()
        except BaseException as exc:
            thread_errors.append(exc)

    def run_switch() -> None:
        try:
            provider.on_session_switch("SESS-B")
        except BaseException as exc:
            thread_errors.append(exc)
        finally:
            switch_done.set()

    auto_sleep = threading.Thread(target=run_auto_sleep)
    auto_sleep.start()
    assert stats_started.wait(timeout=5)
    beam_lock.waiting.clear()

    switch = threading.Thread(target=run_switch)
    switch.start()
    attempted_lock = beam_lock.waiting.wait(timeout=5)
    switch_blocked = not switch_done.wait(timeout=0.1)

    release_stats.set()
    release_worker.set()
    auto_sleep.join(timeout=5)
    switch.join(timeout=5)

    assert not auto_sleep.is_alive()
    assert not switch.is_alive()
    assert not thread_errors
    assert attempted_lock
    assert switch_blocked
    assert switch_done.is_set()
    assert worker_started.is_set()
    assert stats_sessions == [("hermes_SESS-A", "hermes_SESS-A")]
    assert eligibility_sessions == [("hermes_SESS-A", "hermes_SESS-A")]
    assert _WorkerBeam.calls == [
        {
            "session_id": "hermes_SESS-A",
            "db_path": "memory.db",
            "author_id": "author",
            "author_type": "agent",
            "channel_id": "hermes_SESS-A",
        }
    ]
    assert provider._session_id == "hermes_SESS-B"


def test_reinitialize_rebuilds_beam_bound_tool_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mnemosyne_hermes import persona_adapter as persona_adapter_module
    from mnemosyne_hermes import sync_adapter as sync_adapter_module

    sync_constructed: list[str] = []
    persona_constructed: list[str] = []
    sync_handled: list[str] = []
    persona_handled: list[str] = []
    sync_shutdown: list[str] = []

    class _SyncAdapter:
        def __init__(self, beam: Any, _config: dict[str, Any]) -> None:
            self._beam = beam
            sync_constructed.append(beam.session_id)

        def handle_tool_call(self, _tool_name: str, _args: dict[str, Any]) -> str:
            sync_handled.append(self._beam.session_id)
            return self._beam.session_id

        def shutdown(self) -> None:
            sync_shutdown.append(self._beam.session_id)

    class _PersonaAdapter:
        def __init__(self, beam: Any, _config: dict[str, Any]) -> None:
            self._beam = beam
            persona_constructed.append(beam.session_id)

        def handle_tool_call(self, _tool_name: str, _args: dict[str, Any]) -> str:
            persona_handled.append(self._beam.session_id)
            return self._beam.session_id

    class _ReinitializedBeam(_RecordingBeam):
        author_type = "agent"

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(kwargs["session_id"])
            self.channel_id = kwargs.get("channel_id") or self.session_id
            self.db_path = kwargs.get("db_path")

    monkeypatch.setattr(sync_adapter_module, "SyncAdapter", _SyncAdapter)
    monkeypatch.setattr(persona_adapter_module, "PersonaAdapter", _PersonaAdapter)
    monkeypatch.setattr(
        mnemosyne_hermes,
        "_get_beam_class",
        lambda: _ReinitializedBeam,
    )

    provider, _ = _provider_with_recording_beam()
    provider.has_tool = lambda _name: True
    try:
        assert (
            provider.handle_tool_call("mnemosyne_sync_status", {})
            == "hermes_shared_surface"
        )
        assert (
            provider.handle_tool_call("mnemosyne_persona_list", {})
            == "hermes_SESS-A"
        )

        provider.initialize("SESS-B", auto_sleep=False)

        assert (
            provider.handle_tool_call("mnemosyne_sync_status", {})
            == "hermes_shared_surface"
        )
        assert (
            provider.handle_tool_call("mnemosyne_persona_list", {})
            == "hermes_SESS-B"
        )
        assert sync_constructed == ["hermes_shared_surface", "hermes_shared_surface"]
        assert persona_constructed == ["hermes_SESS-A", "hermes_SESS-B"]
        assert sync_handled == ["hermes_shared_surface", "hermes_shared_surface"]
        assert persona_handled == ["hermes_SESS-A", "hermes_SESS-B"]
        assert sync_shutdown == ["hermes_shared_surface"]
    finally:
        provider.shutdown()

    assert sync_shutdown == ["hermes_shared_surface", "hermes_shared_surface"]
    assert provider._provider_sync_adapter is None
    assert provider._provider_persona_adapter is None


def test_reinitialize_closes_existing_audit_log() -> None:
    provider = MnemosyneMemoryProvider()
    audit = Mock()
    provider._audit = audit

    provider.initialize("SESS-B", agent_context="subagent")

    audit.close.assert_called_once_with()
    assert provider._audit is None


def test_shutdown_closes_existing_audit_log() -> None:
    provider = MnemosyneMemoryProvider()
    audit = Mock()
    sync_adapter = Mock()
    persona_adapter = Mock()
    provider._audit = audit
    provider._provider_sync_adapter = sync_adapter
    provider._provider_persona_adapter = persona_adapter

    provider.shutdown()

    audit.close.assert_called_once_with()
    sync_adapter.shutdown.assert_called_once_with()
    persona_adapter.shutdown.assert_called_once_with()
    assert provider._audit is None
    assert provider._provider_sync_adapter is None
    assert provider._provider_persona_adapter is None


def test_auto_sleep_worker_uses_session_scoped_sleep_not_sleep_all_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto-sleep must consolidate only the triggering session (#771).

    A replica of a shared-surface DB carries aged-but-unconsolidated rows for
    the MCP's fixed ``mcp_{bank}`` session. If auto-sleep selects
    ``sleep_all_sessions()`` via capability probing, the first write on the
    replica sweeps the whole backlog into a gist in one pass (407 -> 147 rows
    in the issue). The worker must run the session-scoped ``sleep()`` on an
    isolated beam bound to the triggering session.
    """
    worker_calls: list[str] = []
    worker_kwargs: list[dict[str, Any]] = []

    class _SourceBeam:
        session_id = "hermes_SESS-A"
        channel_id = "hermes_SESS-A"
        db_path = "memory.db"
        author_id = "author"
        author_type = "agent"
        canonical_owner_id = "profile-owner"
        agent_context = "primary"

        def get_working_stats(self) -> dict[str, int]:
            return {"total": 2}

        def _count_unconsolidated_before(self, _cutoff: str) -> int:
            return 1

        def sleep(self) -> None:
            raise AssertionError("auto-sleep must use a worker-local Beam")

        def sleep_all_sessions(self) -> None:
            raise AssertionError("auto-sleep must use a worker-local Beam")

    class _WorkerBeam:
        def __init__(self, **kwargs: Any) -> None:
            worker_kwargs.append(dict(kwargs))

        def sleep(self) -> None:
            worker_calls.append("sleep")

        def sleep_all_sessions(self) -> None:
            worker_calls.append("sleep_all_sessions")

    provider = MnemosyneMemoryProvider()
    provider._beam = _SourceBeam()
    provider._session_id = "hermes_SESS-A"
    provider._auto_sleep_threshold = 1
    provider._auto_sleep_enabled = True
    provider._AUTO_SLEEP_TIMEOUT_SECONDS = 5
    provider._reserve_reflection_budget_locked = lambda _reason: None
    monkeypatch.setattr(mnemosyne_hermes, "_get_beam_class", lambda: _WorkerBeam)

    provider._maybe_auto_sleep(expected_session_id="hermes_SESS-A")

    assert worker_calls == ["sleep"], f"auto-sleep must be session-scoped: {worker_calls}"
    assert worker_kwargs[0]["session_id"] == "hermes_SESS-A"
    assert worker_kwargs[0]["db_path"] == "memory.db"


def test_auto_sleep_disabled_via_auto_sleep_enabled_config_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """auto_sleep_enabled: false must disable auto-sleep via the core config (#771).

    The provider previously read only the Hermes ``auto_sleep`` key; the core
    config key ``auto_sleep_enabled`` (set via ``mnemosyne config set``) never
    reached it, so operators could not opt out of the capability-selected
    sleep path. The fix bridges ``_read_config_key()`` to the core
    ``MnemosyneConfig`` singleton; this test pins that bridge by mocking
    ``get_config()`` with the Hermes config left empty, so it fails if the
    bridge is removed.
    """
    monkeypatch.delenv("MNEMOSYNE_AUTO_SLEEP_ENABLED", raising=False)
    (tmp_path / "config.yaml").write_text(
        "memory:\n"
        "  provider: mnemosyne\n"
        "  mnemosyne: {}\n"
    )

    fake_config = Mock()
    fake_config.get.side_effect = lambda key, default=None: (
        False if key == "auto_sleep_enabled" else default
    )
    monkeypatch.setattr("mnemosyne.core.config.get_config", lambda: fake_config)

    provider = MnemosyneMemoryProvider()
    provider._hermes_home = str(tmp_path)
    provider._auto_sleep_enabled = True
    provider._apply_provider_config({})

    assert provider._auto_sleep_enabled is False


class TestAutoSleepGateCutoffUtc:
    """Twin of the root provider's cutoff-format pin for the packaged
    mnemosyne_hermes auto-sleep gate: the eligibility cutoff must be the
    SPACE-form naive UTC instant derived from the AWARE UTC clock (F11-2,
    CodeRabbit 3942181776 parity across both provider copies)."""

    def test_snapshot_gate_receives_naive_utc_space_form_cutoff(self, monkeypatch):
        captured = {}

        class _SourceBeam:
            session_id = "hermes_SESS-TWIN"
            db_path = "memory.db"
            author_id = "author"
            author_type = "agent"
            channel_id = "channel"

            def get_working_stats(self):
                return {"total": 50}

            def _count_unconsolidated_before(self, cutoff):
                captured["cutoff"] = cutoff
                return 0  # below-threshold result: gate returns before any worker

        provider = mnemosyne_hermes.MnemosyneMemoryProvider()
        provider._beam = _SourceBeam()
        provider._auto_sleep_threshold = 10

        # UTC+2 host shape: naive wall clock 14:30, aware UTC 12:30Z.
        fixed_utc = _real_dt(2026, 4, 1, 12, 30, 0, tzinfo=_real_tz.utc)

        class _Frozen(_real_dt):
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return fixed_utc.astimezone(tz)
                return cls(2026, 4, 1, 14, 30, 0)

        monkeypatch.setattr(mnemosyne_hermes, "datetime", _Frozen)

        result = provider._auto_sleep_snapshot_locked()
        assert result is None  # eligible == 0 -> no snapshot
        cutoff = captured.get("cutoff")
        assert cutoff is not None, "eligibility gate was never called"
        expected = (
            fixed_utc.replace(tzinfo=None)
            - _real_td(hours=_get_working_memory_ttl_hours() // 2)
        ).strftime("%Y-%m-%d %H:%M:%S")
        assert cutoff == expected, (
            f"packaged gate must receive the naive UTC cutoff derived from "
            f"the aware UTC clock, got {cutoff!r}, expected {expected!r}"
        )
