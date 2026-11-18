"""Regression tests for #1062: canonical_facts stamps must share created_at's clock.

``created_at`` is SQLite ``CURRENT_TIMESTAMP`` (naive UTC, ``YYYY-MM-DD HH:MM:SS``).
``CanonicalStore`` used to stamp ``valid_from`` / ``valid_until`` with
``datetime.now().isoformat()``, i.e. naive *local* wall-clock time, so on any
host away from UTC the two columns of one row disagreed by the host's offset
(the same class as #525 for ``working_memory.valid_until``).

Every test runs under a non-UTC host timezone. POSIX ``TZ`` strings are used
(fixed offsets, no DST, no tzdata dependency) so the skew is unambiguous
against a few-seconds tolerance.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne.core import canonical as canonical_module
from mnemosyne.core.canonical import CanonicalStore, forget_canonical, remember_canonical

SKEW_TOLERANCE = timedelta(seconds=5)


@pytest.fixture(params=["NYT5", "JST-9"], ids=["utc-5", "utc+9"])
def non_utc_tz(request, monkeypatch):
    """Run under a non-UTC host timezone, restoring the original TZ afterwards."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset() unavailable on this platform")
    original = os.environ.get("TZ")
    monkeypatch.setenv("TZ", request.param)
    time.tzset()
    yield request.param
    if original is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", original)
    time.tzset()


@pytest.fixture
def store(tmp_path):
    s = CanonicalStore(db_path=tmp_path / "canonical.db")
    yield s
    s.conn.close()


def _parse_utc(value: str) -> datetime:
    """Parse a stored timestamp as UTC (naive values are UTC, like SQLite's)."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def test_host_tz_is_actually_offset(non_utc_tz):
    """Guard the fixture: local wall clock must differ from UTC, or the rest proves nothing."""
    local = datetime.now().replace(microsecond=0)
    utc = _utc_now().replace(tzinfo=None, microsecond=0)
    assert abs(local - utc) >= timedelta(hours=4)


def test_forget_stamps_valid_until_in_created_at_clock(non_utc_tz, store):
    store.remember("jessi", "identity", "name", "My name is Jessi.")
    assert store.forget("jessi", "identity", "name") is True

    row = store.history("jessi", "identity", "name")[0]
    assert row["valid_until"] is not None
    created = _parse_utc(row["created_at"])
    until = _parse_utc(row["valid_until"])

    assert until >= created
    assert abs(_utc_now() - until) <= SKEW_TOLERANCE
    assert abs(until - created) <= SKEW_TOLERANCE
    # Same shape as created_at, so the two columns compare directly as text
    # and through SQLite's date functions.
    assert len(row["valid_until"]) == len(row["created_at"])
    assert row["valid_until"] >= row["created_at"]
    skew = store.conn.execute(
        "SELECT (julianday(valid_until) - julianday(created_at)) * 86400.0 "
        "FROM canonical_facts WHERE id = ?",
        (row["id"],),
    ).fetchone()[0]
    assert 0 <= skew <= SKEW_TOLERANCE.total_seconds()


def test_valid_from_uses_created_at_clock_too(non_utc_tz, store):
    row = store.remember("jessi", "identity", "name", "My name is Jessi.")
    assert abs(_utc_now() - _parse_utc(row["valid_from"])) <= SKEW_TOLERANCE
    assert row["valid_from"] >= row["created_at"]


def test_supersede_stamps_prior_row_in_utc(non_utc_tz, store):
    """remember() closes the prior row with the same clock as forget()."""
    store.remember("jessi", "identity", "name", "My name is Jessi.")
    store.remember("jessi", "identity", "name", "I go by Jess now.")

    newest, oldest = store.history("jessi", "identity", "name")
    assert (newest["version"], oldest["version"]) == (2, 1)
    assert abs(_utc_now() - _parse_utc(oldest["valid_until"])) <= SKEW_TOLERANCE
    assert oldest["valid_until"] >= oldest["created_at"]
    # The superseded row closes exactly when its successor opens.
    assert oldest["valid_until"] == newest["valid_from"]


def test_history_order_survives_forget_and_reremember(non_utc_tz, store):
    store.remember("jessi", "identity", "name", "v1")
    store.remember("jessi", "identity", "name", "v2")
    assert store.forget("jessi", "identity", "name") is True
    assert store.recall("jessi", "identity", "name") is None
    store.remember("jessi", "identity", "name", "v3")

    hist = store.history("jessi", "identity", "name")
    assert [h["version"] for h in hist] == [3, 2, 1]
    assert [h["body"] for h in hist] == ["v3", "v2", "v1"]
    assert hist[0]["valid_until"] is None
    assert all(h["valid_until"] is not None for h in hist[1:])
    # Timestamps never run backwards as versions climb.
    stamps = [_parse_utc(h["valid_from"]) for h in reversed(hist)]
    assert stamps == sorted(stamps)
    closes = [_parse_utc(h["valid_until"]) for h in reversed(hist[1:])]
    assert closes == sorted(closes)
    for h in hist[1:]:
        assert _parse_utc(h["valid_until"]) >= _parse_utc(h["valid_from"])


def test_module_level_helpers_stamp_utc(non_utc_tz, tmp_path):
    db = tmp_path / "helpers.db"
    remember_canonical("jessi", "identity", "name", "My name is Jessi.", db_path=db)
    assert forget_canonical("jessi", "identity", "name", db_path=db) is True

    store = CanonicalStore(db_path=db)
    try:
        row = store.history("jessi", "identity", "name")[0]
        assert abs(_utc_now() - _parse_utc(row["valid_until"])) <= SKEW_TOLERANCE
        assert row["valid_until"] >= row["created_at"]
    finally:
        store.conn.close()


def _make_stepping_clock():
    """Build a ``datetime`` whose ``now()`` steps 10 minutes per call from a fixed UTC instant.

    ``now()`` with no tz returns naive *host-local* time and ``now(tz)`` returns
    that instant in ``tz``, exactly like the real ``datetime.now``, so a
    local-time stamp would land hours away from the UTC one under the fixture TZ.
    A fresh class per call keeps the position from leaking between tests.
    """

    class _SteppingClock(datetime):
        _utc = datetime(2026, 9, 26, 16, 0, 0, tzinfo=timezone.utc)

        @classmethod
        def now(cls, tz=None):
            cls._utc = cls._utc + timedelta(minutes=10)
            if tz is None:
                return cls._utc.astimezone().replace(tzinfo=None)
            return cls._utc.astimezone(tz)

    return _SteppingClock


def _versions_as_of(store, as_of_utc: str):
    """Versions live at ``as_of_utc`` using plain text comparison on the stamps."""
    rows = store.conn.execute(
        "SELECT version FROM canonical_facts "
        "WHERE owner_id = ? AND category = ? AND name = ? "
        "AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?) "
        "ORDER BY version",
        ("jessi", "identity", "name", as_of_utc, as_of_utc),
    ).fetchall()
    return [r[0] for r in rows]


def test_as_of_lookup_over_stamps_matches_utc_timeline(non_utc_tz, store, monkeypatch):
    monkeypatch.setattr(canonical_module, "datetime", _make_stepping_clock())
    # UTC timeline: v1 opens 16:10, v2 supersedes 16:20, forget 16:30, v3 opens 16:40.
    store.remember("jessi", "identity", "name", "v1")
    store.remember("jessi", "identity", "name", "v2")
    store.forget("jessi", "identity", "name")
    store.remember("jessi", "identity", "name", "v3")

    assert _versions_as_of(store, "2026-09-26 16:05:00") == []
    assert _versions_as_of(store, "2026-09-26 16:15:00") == [1]
    assert _versions_as_of(store, "2026-09-26 16:25:00") == [2]
    assert _versions_as_of(store, "2026-09-26 16:35:00") == []
    assert _versions_as_of(store, "2026-09-26 16:45:00") == [3]

    rows = {h["version"]: h for h in store.history("jessi", "identity", "name")}
    assert rows[1]["valid_from"] == "2026-09-26 16:10:00"
    assert rows[1]["valid_until"] == "2026-09-26 16:20:00"
    assert rows[2]["valid_until"] == "2026-09-26 16:30:00"
    assert rows[3]["valid_from"] == "2026-09-26 16:40:00"
    assert rows[3]["valid_until"] is None
