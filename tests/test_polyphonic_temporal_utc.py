"""Regression tests for #1094: the polyphonic temporal voice must score in UTC.

``working_memory.timestamp`` is naive UTC (beam.py stamps
``datetime.now(timezone.utc).replace(tzinfo=None)``). The temporal reader did:

    if row_dt.tzinfo is not None:
        row_dt = row_dt.astimezone().replace(tzinfo=None)   # -> local
    age = datetime.now() - row_dt                          # -> local

Both sides are local, so it looks self-consistent, but a row written in UTC reads as
``offset`` hours older than it is. With a 7-day time constant that is a flat rank shift for
every row, and the 7-day window moves with the host.

Follows the pattern from test_canonical_valid_until_utc.py (#1087): POSIX TZ strings with
``time.tzset()``, so the skew is unambiguous and needs no tzdata. Fixed offsets only —
DST is out of scope, and the behaviour under test does not depend on it.
"""

import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("numpy")

from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine  # noqa: E402

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
def db(tmp_path):
    path = tmp_path / "wm.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        "CREATE TABLE working_memory ("
        "id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT,"
        "session_id TEXT, importance REAL, valid_until TEXT, superseded_by TEXT);"
    )
    conn.commit()
    conn.close()
    yield path
    sqlite3.connect(str(path)).close()


def _insert(path, mid, age_hours, *, aware=False, offset_hours=0):
    """Store one row ``age_hours`` old.

    ``aware`` writes an offset-bearing timestamp; ``offset_hours`` is that offset, so 0 is
    the case where the written digits happen to match the instant. A non-zero offset is what
    separates a TEXT comparison from julianday().
    """
    moment = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    if not aware:
        stamp = moment.replace(tzinfo=None).isoformat()
    else:
        stamp = moment.astimezone(timezone(timedelta(hours=offset_hours))).isoformat()
    _stamp(path, mid, stamp)


def _stamp(path, mid, stamp):
    """Store one row with an exact timestamp string."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
        "VALUES (?, ?, 'conversation', ?, 's1', 1.0)",
        (mid, f"memory {mid}", stamp),
    )
    conn.commit()
    conn.close()


def _temporal(db):
    # The voice is keyword-gated: _temporal_voice returns [] unless the query carries a
    # temporal term ("recent", "yesterday", ...). Without one the test measures the gate,
    # not the scoring, and every row comes back missing.
    return PolyphonicRecallEngine(db_path=db)._temporal_voice(query="what happened recently")


def _age_days(results, memory_id):
    for r in results:
        if r.memory_id == memory_id:
            return r.metadata["age_days"]
    return None


def test_host_tz_is_actually_offset(non_utc_tz):
    """Guard the fixture: local wall clock must differ from UTC, or the rest proves nothing."""
    local = datetime.now().replace(microsecond=0)
    utc = datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    assert abs(local - utc) >= timedelta(hours=4)


def test_naive_utc_row_ages_by_its_real_age(db, non_utc_tz):
    """A naive row is UTC. Its age must be hours, not hours + host offset."""
    _insert(db, "m1", age_hours=1)
    age = _age_days(_temporal(db), "m1")
    assert age is not None, "row was not returned"
    assert abs(age - (1 / 24)) < 0.01, f"a 1h-old row aged as {age} days"


def test_aware_row_ages_the_same_as_naive(db, non_utc_tz):
    """An offset-bearing row must score the same as its naive-UTC twin."""
    _insert(db, "naive", age_hours=1, aware=False)
    _insert(db, "aware", age_hours=1, aware=True)

    results = _temporal(db)
    naive_age = _age_days(results, "naive")
    aware_age = _age_days(results, "aware")

    assert naive_age is not None and aware_age is not None, results
    assert abs(naive_age - aware_age) < 0.01, (naive_age, aware_age)
    assert abs(naive_age - (1 / 24)) < 0.01, naive_age


def test_offset_never_ages_a_row_backwards(db, non_utc_tz):
    """A row must never read as younger than it is, in either direction."""
    _insert(db, "fresh", age_hours=1)
    _insert(db, "brand_new", age_hours=0)

    results = _temporal(db)
    for mid in ("fresh", "brand_new"):
        age = _age_days(results, mid)
        assert age is not None, mid
        assert age >= -0.01, f"{mid} aged {age} days"


def test_older_rows_rank_below_newer_ones(db, non_utc_tz):
    """The whole point of the voice: a 1h row must outrank a 100h row, by a lot."""
    _insert(db, "newest", age_hours=1)
    _insert(db, "oldest", age_hours=100)

    results = {r.memory_id: r for r in _temporal(db)}
    assert "newest" in results and "oldest" in results, sorted(results)
    assert results["newest"].score > results["oldest"].score, {
        k: v.score for k, v in results.items()
    }


def test_seven_day_window_is_utc(db, non_utc_tz):
    """The window boundary is 7 days in UTC, for naive rows and for offset rows.

    The offsets are chosen so each row's *written* digits fall on the wrong side of a TEXT
    comparison, which is the defect this pins. dplush's review of #1097 caught the first
    attempt at these cases: I put "inside" at +05:00 and "outside" at -05:00, which moves both
    rows further away from the boundary and so passes even with raw TEXT comparison — the
    tests could not tell julianday() from its defect. Reversing the offsets is what makes
    them discriminating:

        inside  6d23h written -05:00  reads as 7d00h of wall clock -> wrongly excluded
        outside 7d1h  written +05:00  reads as 6d20h of wall clock -> wrongly included

    With julianday() both land correctly, because it compares instants.
    """
    _insert(db, "inside", age_hours=24 * 6 + 23)   # 6d23h
    _insert(db, "outside", age_hours=24 * 7 + 1)  # 7d1h
    _insert(db, "inside_offset", age_hours=24 * 6 + 23, aware=True, offset_hours=-5)
    _insert(db, "outside_offset", age_hours=24 * 7 + 1, aware=True, offset_hours=5)

    ids = {r.memory_id for r in _temporal(db)}
    assert "inside" in ids and "inside_offset" in ids, sorted(ids)
    assert "outside" not in ids and "outside_offset" not in ids, sorted(ids)


def test_mixed_formats_admit_the_newest_before_limit(db, non_utc_tz):
    """25 rows, half naive and half offset-bearing: admission must follow the instant.

    Asserts the complete ordered id list, not the length and the two extremes. Length 20 with
    m00 present and m24 absent is also satisfied by the wrong twenty rows — dplush's review of
    #1097 showed this case passing under a raw-TEXT mutation that admitted an older row while
    dropping a newer one. The exact list is the only assertion that cannot survive that.

    Rows are interleaved and the odd ones carry +05:00, so their written digits are five hours
    behind their instant: under TEXT ordering they sink and displace newer naive rows.
    """
    for i in range(25):
        _insert(db, f"m{i:02d}", age_hours=i, aware=bool(i % 2), offset_hours=5)

    ids = [r.memory_id for r in _temporal(db)]
    assert ids == [f"m{i:02d}" for i in range(20)], ids


def test_invalid_and_null_timestamps_are_skipped(db, non_utc_tz):
    """A malformed or NULL stamp is dropped, as before — and the good row still comes back."""
    _insert(db, "good", age_hours=1)
    conn = sqlite3.connect(str(db))
    for mid, stamp in (("bad", "not-a-timestamp"), ("null_ts", None)):
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
            "VALUES (?, ?, 'conversation', ?, 's1', 1.0)",
            (mid, mid, stamp),
        )
    conn.commit()
    conn.close()

    ids = {r.memory_id for r in _temporal(db)}
    assert "good" in ids, sorted(ids)
    assert "bad" not in ids and "null_ts" not in ids, sorted(ids)


def test_top20_admits_the_newest_rows(db, non_utc_tz):
    """Selection happens before LIMIT 20, so the newest rows are the ones admitted."""
    for i in range(25):
        _insert(db, f"m{i:02d}", age_hours=i)  # m00 is the newest

    ids = [r.memory_id for r in _temporal(db)]
    assert len(ids) == 20, len(ids)
    assert "m00" in ids, "the newest row was not admitted"
    assert "m24" not in ids, "the oldest row was admitted; the cut is not by recency"
