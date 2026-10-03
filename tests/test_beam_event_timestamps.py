"""Regression tests: event timestamps + content event dates (PR).

Covers the maintainer review demands:
- chronological (UTC-instant) selection, not lexicographic, incl. mixed
  ISO offsets and the naive-as-UTC policy
- contract split: event_timestamp writes `timestamp` only; event_date is
  never manufactured from ingest time
- sleep() propagates a content event date only when all non-null source
  event_dates agree
- end-to-end through sleep() with mixed-offset sources
"""
from __future__ import annotations

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import mnemosyne.core.beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.beam import _latest_iso_string


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


# ------------------------------------------------------------ pure selection
class TestLatestIsoString:
    def test_picks_chronological_latest_across_mixed_offsets(self):
        # Lexicographic max would pick "2026-04-01T00:30:00+02:00"; the UTC
        # instant is 2026-03-31T22:30:00Z, EARLIER than 23:00:00Z.
        values = ["2026-04-01T00:30:00+02:00", "2026-03-31T23:00:00+00:00"]
        assert _latest_iso_string(values) == "2026-03-31T23:00:00+00:00"

    def test_naive_treated_as_utc(self):
        # Naive policy: 2026-04-01T01:00:00 (naive, as UTC) is later than
        # 2026-03-31T23:30:00+00:00.
        values = ["2026-04-01T01:00:00", "2026-03-31T23:30:00+00:00"]
        assert _latest_iso_string(values) == "2026-04-01T01:00:00"

    def test_preserves_original_representation(self):
        # Tie at 06:00Z between the +00:00 and +02:00 forms: strict ordering
        # keeps the first-seen representation.
        values = ["2026-01-01T06:00:00+00:00", "2026-01-01T08:00:00+02:00"]
        assert _latest_iso_string(values) == "2026-01-01T06:00:00+00:00"

    def test_skips_invalid_and_empty(self):
        values = ["not-a-date", "", None, "2026-05-05T00:00:00"]
        assert _latest_iso_string(values) == "2026-05-05T00:00:00"

    def test_all_invalid_returns_none(self):
        assert _latest_iso_string(["garbage", ""]) is None


# ------------------------------------------------- contract: consolidate API
class TestConsolidateContract:
    def test_event_timestamp_writes_timestamp_only(self, temp_db):
        beam = BeamMemory(session_id="c1", db_path=temp_db)
        ts = "2026-04-01T12:00:00"
        mid = beam.consolidate_to_episodic(
            summary="- test", source_wm_ids=[], source="probe",
            event_timestamp=ts)
        row = beam.conn.execute(
            "SELECT timestamp, event_date, event_date_precision FROM episodic_memory WHERE id=?",
            (mid,)).fetchone()
        assert row[0] == ts
        assert row[1] is None  # no event date manufactured
        assert row[2] == "unknown"  # schema default precision, untouched
        beam.conn.execute("DELETE FROM episodic_memory WHERE id=?", (mid,))
        beam.conn.commit()

    def test_event_date_written_only_when_passed(self, temp_db):
        beam = BeamMemory(session_id="c2", db_path=temp_db)
        mid = beam.consolidate_to_episodic(
            summary="- test", source_wm_ids=[], source="probe",
            event_date="2020-01-02", event_date_precision="day")
        row = beam.conn.execute(
            "SELECT event_date, event_date_precision FROM episodic_memory WHERE id=?",
            (mid,)).fetchone()
        assert row[0] == "2020-01-02"
        assert row[1] == "day"
        beam.conn.execute("DELETE FROM episodic_memory WHERE id=?", (mid,))
        beam.conn.commit()

    def test_event_date_default_precision(self, temp_db):
        beam = BeamMemory(session_id="c3", db_path=temp_db)
        mid = beam.consolidate_to_episodic(
            summary="- test", source_wm_ids=[], source="probe",
            event_date="2020-01-02")
        row = beam.conn.execute(
            "SELECT event_date_precision FROM episodic_memory WHERE id=?",
            (mid,)).fetchone()
        assert row[0] == "unknown"
        beam.conn.execute("DELETE FROM episodic_memory WHERE id=?", (mid,))
        beam.conn.commit()


# ------------------------------------------------------------- sleep() E2E
def _seed_old_wm(conn, session_id, rows):
    """Seed working_memory rows through an existing connection (post-init)."""
    for row in rows:
        rid, content, ts, ed, edp = row
        if ed:
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id, event_date, event_date_precision) "
                "VALUES (?, ?, 'conversation', ?, ?, ?, ?)",
                (rid, content, ts, session_id, ed, edp))
        else:
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
                "VALUES (?, ?, 'conversation', ?, ?)",
                (rid, content, ts, session_id))
    conn.commit()


class TestGetReturnsEventMetadata:
    def test_get_exposes_event_date_and_precision(self, temp_db):
        # CodeRabbit: get() selected event_date/event_date_precision but the
        # episodic return mapping discarded them — callers could not read
        # what consolidation/export/import preserve.
        beam = BeamMemory(session_id="get1", db_path=temp_db)
        mid = beam.consolidate_to_episodic(
            summary="- getter", source_wm_ids=[], source="probe",
            event_timestamp="2026-04-01T12:00:00",
            event_date="2020-01-02", event_date_precision="day")
        got = beam.get(mid)
        assert got is not None
        assert got["memory_store"] == "episodic"
        assert got["event_date"] == "2020-01-02"
        assert got["event_date_precision"] == "day"

    def test_get_working_row_has_no_event_fields(self, temp_db):
        # working rows carry no event columns: get() must not fabricate them
        beam = BeamMemory(session_id="get2", db_path=temp_db)
        wid = beam.remember("wm getter", source="probe")
        got = beam.get(wid)
        assert got is not None
        assert got["memory_store"] == "working"
        assert "event_date" not in got


class TestNoOverrideBackwardCompat:
    def test_no_overrides_leave_defaults(self, temp_db):
        # Without any kwargs the row keeps its generated timestamp and both
        # event fields stay unset — the pre-PR call contract is unchanged.
        beam = BeamMemory(session_id="noc", db_path=temp_db)
        mid = beam.consolidate_to_episodic(summary="- plain", source_wm_ids=[], source="probe")
        row = beam.conn.execute(
            "SELECT timestamp, event_date, event_date_precision FROM episodic_memory WHERE id=?",
            (mid,)).fetchone()
        assert row[0] is not None and row[0] != ""
        assert row[1] is None
        assert row[2] == "unknown"
        beam.conn.execute("DELETE FROM episodic_memory WHERE id=?", (mid,))
        beam.conn.commit()


class TestExportImportRoundTrip:
    def test_event_date_and_precision_survive_round_trip(self, temp_db):
        beam = BeamMemory(session_id="exp", db_path=temp_db)
        mid = beam.consolidate_to_episodic(
            summary="- round trip", source_wm_ids=[], source="probe",
            event_timestamp="2026-04-01T12:00:00",
            event_date="2020-01-02", event_date_precision="day")
        data = beam.export_to_dict()
        beam2 = BeamMemory(session_id="imp", db_path=temp_db.with_name("rt2.db"))
        beam2.import_from_dict(data, force=True)
        row = beam2.conn.execute(
            "SELECT timestamp, event_date, event_date_precision FROM episodic_memory WHERE id=?",
            (mid,)).fetchone()
        assert row is not None, "episodic row lost in export/import"
        assert row[0] == "2026-04-01T12:00:00"
        assert row[1] == "2020-01-02"
        assert row[2] == "day"


class TestSleepEventDates:
    def _episodic_rows(self, db_path):
        conn = sqlite3.connect(str(db_path))
        try:
            return conn.execute(
                "SELECT timestamp, event_date, event_date_precision, content FROM episodic_memory"
            ).fetchall()
        finally:
            conn.close()

    def test_sleep_selects_latest_by_utc_instant(self, temp_db):
        # 04-01T00:30+02:00 == 03-31T22:30Z (earlier); 03-31T23:00Z is the
        # latest instant. Lexicographic max would pick the +02:00 row.
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(beam.conn, "s1", [
            ("a", "content a", "2026-04-01T00:30:00+02:00", None, None),
            ("b", "content b", "2026-03-31T23:00:00+00:00", None, None),
        ])
        res = beam.sleep(dry_run=False)
        assert res["status"] == "consolidated"
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        # stored NORMALIZED (no offset suffix): lexicographic date filters stay correct
        assert rows[0][0] == "2026-03-31T23:00:00"
        assert rows[0][1] is None

    def test_sleep_propagates_consistent_content_event_date(self, temp_db):
        beam = BeamMemory(session_id="s2", db_path=temp_db)
        _seed_old_wm(beam.conn, "s2", [
            ("a", "content a", (datetime.now() - timedelta(hours=200)).isoformat(), "2020-01-02", "day"),
            ("b", "content b", (datetime.now() - timedelta(hours=199)).isoformat(), "2020-01-02", "day"),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] == "2020-01-02"
        assert rows[0][2] == "day"

    def test_sleep_leaves_event_date_unknown_when_sources_disagree(self, temp_db):
        beam = BeamMemory(session_id="s3", db_path=temp_db)
        _seed_old_wm(beam.conn, "s3", [
            ("a", "content a", (datetime.now() - timedelta(hours=200)).isoformat(), "2020-01-02", "day"),
            ("b", "content b", (datetime.now() - timedelta(hours=199)).isoformat(), "2021-06-15", "day"),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] is None  # never manufacture from conflicting content dates

    def test_sleep_does_not_stamp_from_single_dated_row(self, temp_db):
        # Round-2 MEDIUM-2: one dated row among undated rows must NOT stamp
        # the summary covering all of them.
        beam = BeamMemory(session_id="s5", db_path=temp_db)
        _seed_old_wm(beam.conn, "s5", [
            ("a", "content a", (datetime.now() - timedelta(hours=200)).isoformat(), "2020-01-02", "day"),
            ("b", "content b", (datetime.now() - timedelta(hours=199)).isoformat(), None, None),
            ("c", "content c", (datetime.now() - timedelta(hours=198)).isoformat(), None, None),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] is None

    def test_sleep_agrees_on_date_not_precision(self, temp_db):
        # Round-2 LOW-1: agreement is on the date; precision variants carry.
        beam = BeamMemory(session_id="s6", db_path=temp_db)
        _seed_old_wm(beam.conn, "s6", [
            ("a", "content a", (datetime.now() - timedelta(hours=200)).isoformat(), "2020-01-02", "day"),
            ("b", "content b", (datetime.now() - timedelta(hours=199)).isoformat(), "2020-01-02", "month"),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] == "2020-01-02"

    def test_sleep_whitespace_event_date_counts_undated(self, temp_db):
        # whitespace-only event_date must count as undated (not stamp the group)
        beam = BeamMemory(session_id="s7", db_path=temp_db)
        _seed_old_wm(beam.conn, "s7", [
            ("a", "content a", (datetime.now() - timedelta(hours=200)).isoformat(), "2020-01-02", "day"),
            ("b", "content b", (datetime.now() - timedelta(hours=199)).isoformat(), "   ", "day"),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] is None

    def test_sleep_precision_from_newest_dated_row(self, temp_db):
        # same date, different precisions: carried precision comes from the
        # NEWEST row (matching the timestamp rule), not the oldest
        beam = BeamMemory(session_id="s8", db_path=temp_db)
        _seed_old_wm(beam.conn, "s8", [
            ("old", "content old", (datetime.now() - timedelta(hours=300)).isoformat(), "2020-01-02", "month"),
            ("new", "content new", (datetime.now() - timedelta(hours=100)).isoformat(), "2020-01-02", "day"),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] == "2020-01-02"
        assert rows[0][2] == "day"

    def test_sleep_precision_from_latest_instant_with_crossed_offsets(self, temp_db):
        # CodeRabbit Minor: naive timestamps sort lexicographically ==
        # chronologically, so the precision rule is untested against mixed
        # offsets. Here the STRING-max row is the EARLIER instant:
        #   a: 2026-04-01T00:30:00+02:00 == 2026-03-31T22:30Z (string max)
        #   b: 2026-03-31T23:00:00+00:00 == 2026-03-31T23:00Z (instant max)
        # Precision must follow the chronologically latest (b -> "month"),
        # not the lexicographically greatest string (a -> "day").
        beam = BeamMemory(session_id="s8x", db_path=temp_db)
        _seed_old_wm(beam.conn, "s8x", [
            ("a", "content a", "2026-04-01T00:30:00+02:00", "2020-01-02", "day"),
            ("b", "content b", "2026-03-31T23:00:00+00:00", "2020-01-02", "month"),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] == "2020-01-02"
        assert rows[0][2] == "month", \
            f"precision must come from the chronologically latest instant, got {rows[0][2]!r}"

    def test_sleep_poison_timestamp_does_not_abort_batch(self, temp_db):
        # round-3 HIGH-1 lineage: an unparseable timestamp must never abort
        # sleep(). The claim pre-filter now EXCLUDES every poison form the
        # Python gate would reject (SQL acceptance equals Python
        # acceptance, F12-2): julian-day numbers, calendar-impossible
        # dates and plain junk never occupy batch slots, so eligible rows
        # behind them consolidate instead of starving.
        beam = BeamMemory(session_id="s9", db_path=temp_db)
        _seed_old_wm(beam.conn, "s9", [
            ("a", "content a", "2461000", "2020-01-02", "day"),  # julian-day poison
            ("c", "content c", "2026-02-30T00:00:00", "2020-01-02", "day"),  # impossible date
            ("b", "content b", "2026-03-31T23:00:00+00:00", "2020-01-02", "day"),
        ])
        res = beam.sleep(dry_run=False)  # must not raise
        assert res["status"] == "consolidated"
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] == "2020-01-02"
        # precision from the newest row (b); the poisons never reach the batch
        assert rows[0][2] == "day"
        remaining = {
            r[0] for r in beam.conn.execute(
                "SELECT id FROM working_memory WHERE consolidated_at IS NULL")
        }
        assert {"a", "c"} <= remaining, (
            f"poison rows must stay unconsolidated (excluded pre-claim): {remaining}"
        )

    def test_sleep_normalized_storage_keeps_date_filters_correct(self, temp_db):
        # round-3 MEDIUM-1: an offset-bearing winner must be stored WITHOUT
        # the offset so recall's lexicographic date filters see the right day.
        # 2026-04-01T00:30+02:00 == 2026-03-31T22:30Z: normalized storage keeps
        # it on 2026-03-31, not 2026-04-01.
        beam = BeamMemory(session_id="s10", db_path=temp_db)
        _seed_old_wm(beam.conn, "s10", [
            ("a", "content a", "2026-04-01T00:30:00+02:00", None, None),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][0].startswith("2026-03-31T22:30:00")

    def test_sleep_whitespace_event_date_pinned(self, temp_db):
        # round-3 MEDIUM-2 pin: ALL-whitespace dates count undated (strip
        # applied on the truthiness side too, not just comparison)
        beam = BeamMemory(session_id="s11", db_path=temp_db)
        _seed_old_wm(beam.conn, "s11", [
            ("a", "content a", (datetime.now() - timedelta(hours=200)).isoformat(), "  ", "day"),
            ("b", "content b", (datetime.now() - timedelta(hours=199)).isoformat(), "  ", "day"),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] is None

    def test_sleep_multi_source_groups_stay_independent(self, temp_db):
        # Two source groups in ONE sleep batch: each summary must carry only
        # its OWN group's latest timestamp and agreed date (the original bug
        # took max() over the whole batch).
        beam = BeamMemory(session_id="s12", db_path=temp_db)
        _seed_old_wm(beam.conn, "s12", [
            ("a1", "content a", "2026-01-01T10:00:00", "2020-01-02", "day"),
            ("a2", "content a2", "2026-01-05T10:00:00", "2020-01-02", "day"),
            ("b1", "content b", "2026-03-01T10:00:00", "2021-06-15", "day"),
            ("b2", "content b2", "2026-02-01T10:00:00", "2021-06-15", "day"),
        ])
        # different sources force different groups: seed via source column
        beam.conn.execute("UPDATE working_memory SET source='convA' WHERE id IN ('a1','a2')")
        beam.conn.execute("UPDATE working_memory SET source='convB' WHERE id IN ('b1','b2')")
        beam.conn.commit()
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 2
        by_date = {r[1]: r for r in rows}
        # group A: latest ts 2026-01-05, date 2020-01-02
        ra = by_date["2020-01-02"]
        assert ra[0] == "2026-01-05T10:00:00"
        # group B: latest ts 2026-03-01 (NOT the batch-wide max), date 2021-06-15
        rb = by_date["2021-06-15"]
        assert rb[0] == "2026-03-01T10:00:00"

    def test_normalized_timestamp_keeps_recall_date_filters_correct(self, temp_db):
        # The stored naive-UTC value must answer date filters on the right day:
        # 2026-04-01T00:30+02:00 == 2026-03-31T22:30Z -> visible through
        # to_date=2026-03-31, absent from from_date=2026-04-01. Deterministic:
        # the episodic summary is the ONLY recallable row for the query.
        beam = BeamMemory(session_id="s13", db_path=temp_db)
        _seed_old_wm(beam.conn, "s13", [
            ("a", "content a", "2026-04-01T00:30:00+02:00", None, None),
        ])
        beam.sleep(dry_run=False)
        row = beam.conn.execute(
            "SELECT id FROM episodic_memory ORDER BY rowid DESC LIMIT 1").fetchone()
        ep_id = row[0]

        to_hit = beam.recall("content a", to_date="2026-03-31")
        from_hit = beam.recall("content a", from_date="2026-04-01")
        to_ids = {r["id"] for r in to_hit} if to_hit else set()
        from_ids = {r["id"] for r in from_hit} if from_hit else set()
        # present on its true UTC day; absent once the window starts 04-01
        assert ep_id in to_ids, f"summary missing from to_date=2026-03-31: {to_ids}"
        assert ep_id not in from_ids, f"summary leaked into from_date=2026-04-01: {from_ids}"

    def test_sleep_mixed_offsets_and_consistent_dates_together(self, temp_db):
        beam = BeamMemory(session_id="s4", db_path=temp_db)
        _seed_old_wm(beam.conn, "s4", [
            ("a", "content a", "2026-04-01T00:30:00+02:00", "2020-01-02", "day"),
            ("b", "content b", "2026-03-31T23:00:00+00:00", "2020-01-02", "day"),
        ])
        beam.sleep(dry_run=False)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][0] == "2026-03-31T23:00:00"  # normalized UTC
        assert rows[0][1] == "2020-01-02"


# ------------------------------------------------- R5: chronological TTL
class TestChronologicalTTLSelection:
    """Maintainer review 5117679525 finding 1: lexicographic timestamp
    comparison misclassifies offset-bearing rows at the cutoff."""

    def test_positive_offset_row_eligible_at_cutoff(self, temp_db, monkeypatch):
        # 2026-04-01T00:30:00+02:00 == 2026-03-31T22:30Z. With a UTC
        # cutoff of 2026-04-01T00:00:00 the row is OLDER than the cutoff
        # -> eligible. A raw TEXT compare wrongly rejects it.
        beam = BeamMemory(session_id="ttl1", db_path=temp_db)
        monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 168)
        # freeze the clock: now = 2026-04-08 -> cutoff = 2026-04-01T00:00:00
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 4, 8)
        monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
        _seed_old_wm(beam.conn, "ttl1", [
            ("plus2", "content plus", "2026-04-01T00:30:00+02:00", None, None),
        ])
        res = beam.sleep(dry_run=True)
        # dry_run reports planned consolidation; the row must be selected
        assert res.get("status") != "no_op", "offset row wrongly excluded by lexicographic compare"

    def test_negative_offset_row_excluded_at_cutoff(self, temp_db, monkeypatch):
        # 2026-03-31T23:30:00-02:00 == 2026-04-01T01:30Z. Newer than the
        # 2026-04-01T00:00:00 cutoff -> NOT eligible. TEXT compare wrongly
        # includes it (sorts before the cutoff string).
        beam = BeamMemory(session_id="ttl2", db_path=temp_db)
        monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 168)
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 4, 4, 12, 0, 0)
        monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
        _seed_old_wm(beam.conn, "ttl2", [
            ("minus2", "content minus", "2026-03-31T23:30:00-02:00", None, None),
        ])
        res = beam.sleep(dry_run=True)
        # eligible rows -> 'dry_run' status; nothing eligible -> 'no_op'.
        # The negative-offset row is NEWER than the cutoff: it must not be
        # selected, so no consolidation is planned.
        assert res.get("status") == "no_op", (
            f"offset row wrongly included by lexicographic compare: {res}"
        )

    def test_count_gate_uses_chronology(self, temp_db):
        # _count_unconsolidated_before must agree with sleep() selection
        beam = BeamMemory(session_id="ttl3", db_path=temp_db)
        _seed_old_wm(beam.conn, "ttl3", [
            ("plus2", "content plus", "2026-04-01T00:30:00+02:00", None, None),
        ])
        n = beam._count_unconsolidated_before(
            beam_module._utc_cutoff_sql("2026-04-01T00:00:00")
        )
        assert n == 1, "count gate still lexicographic"

    def test_poison_rows_sort_below_real_rows(self, temp_db, monkeypatch):
        # Alpha-leading poison ('not-a-timestamp') sorts ABOVE every real
        # row under raw TEXT compare and would be stranded forever; the
        # IS-NULL tiebreak pins all poison below real rows.
        beam = BeamMemory(session_id="ttl4", db_path=temp_db)
        monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 168)
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 4, 4, 12, 0, 0)
        monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
        _seed_old_wm(beam.conn, "ttl4", [
            ("real", "real content", "2026-04-05T12:00:00", None, None),
        ])
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES ('poison', 'poison content', 'conversation', 'not-a-timestamp', 'ttl4')"
        )
        beam.conn.commit()
        # Production ordering shape (sleep claim): the shared
        # _SQL_CHRONO_TS expression — not a hand-rolled variant.
        order = [r["id"] for r in beam.conn.execute(
            f"SELECT id FROM working_memory "
            f"WHERE COALESCE(session_id, 'default') = 'ttl4' "
            f"ORDER BY {beam_module._SQL_CHRONO_TS} ASC"
        )]
        assert order[0] == "poison", f"poison not pinned lowest: {order}"
        res = beam.sleep(dry_run=False)
        # The real row is NEWER than the cutoff, so nothing is claimable:
        # no_op is correct. The contract being pinned: poison does not
        # get claimed-and-stranded (claim filter skips it), and the real
        # row is untouched with no summary.
        assert res.get("status") == "no_op", (
            f"unexpected sleep outcome: {res}"
        )
        n_real = beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'real'"
        ).fetchone()[0]
        n_poison_claimed = beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'poison'"
            " AND consolidated_at IS NOT NULL"
        ).fetchone()[0]
        n_ep = beam.conn.execute(
            "SELECT COUNT(*) FROM episodic_memory"
        ).fetchone()[0]
        assert n_real == 1 and n_poison_claimed == 0 and n_ep == 0, (
            "poison row must not be claimed/summarized"
        )


class TestTrimWorkingMemoryChronology:
    def test_offset_row_trimmed_by_utc_instant(self, monkeypatch):
        """Trim must judge an offset-bearing row by its UTC instant via the
        PRODUCTION _trim_working_memory predicate (a hand-written DELETE
        cannot catch a regression there). The row 2026-04-04T12:00+02:00
        (== 10:00Z) trimmed at frozen now 2026-04-11 12:00 UTC with
        TTL=168h has cutoff 2026-04-04 12:00Z: the row is 2h older than
        the cutoff -> exactly one outcome, trimmed."""
        import tempfile
        from pathlib import Path
        tmp = Path(tempfile.mkdtemp()) / "trim.db"
        beam = BeamMemory(session_id="trim1", db_path=tmp)
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES ('tz', 'content', 'conversation', '2026-04-04T12:00:00+02:00', 'trim1')"
        )
        beam.conn.commit()

        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 4, 11, 12, 0, 0)
        monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
        monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 168)
        beam._trim_working_memory()
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'tz'"
        ).fetchone()[0]
        assert n == 0, "offset row is 2h past the UTC cutoff and must be trimmed"


class TestOverrideValidation:
    def test_bad_event_date_raises_and_persists_nothing(self, temp_db):
        beam = BeamMemory(session_id="val1", db_path=temp_db)
        with pytest.raises(ValueError, match="event_date"):
            beam.consolidate_to_episodic(
                summary="- bad", source_wm_ids=[], source="probe",
                event_date="2020-1-2")
        n = beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        assert n == 0, "invalid override must not persist a partial row"

    def test_bad_precision_raises(self, temp_db):
        beam = BeamMemory(session_id="val2", db_path=temp_db)
        with pytest.raises(ValueError, match="event_date_precision"):
            beam.consolidate_to_episodic(
                summary="- bad", source_wm_ids=[], source="probe",
                event_date="2020-01-02", event_date_precision="fortnight")

    def test_bad_event_timestamp_raises(self, temp_db):
        beam = BeamMemory(session_id="val3", db_path=temp_db)
        with pytest.raises(ValueError, match="event_timestamp"):
            beam.consolidate_to_episodic(
                summary="- bad", source_wm_ids=[], source="probe",
                event_timestamp="not-a-timestamp")

    def test_sleep_survives_garbage_stored_event_date(self, temp_db, monkeypatch):
        # C1: a garbage event_date aggregated from stored rows must NOT
        # raise out of sleep() after the claim (would strand the group);
        # the summary is stored without an event date instead.
        beam = BeamMemory(session_id="san1", db_path=temp_db)
        monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 168)
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 4, 8)
        monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
        _seed_old_wm(beam.conn, "san1", [
            ("a", "content a", "2026-04-03T10:00:00", "2020-1-2", "day"),   # garbage date
            ("b", "content b", "2026-04-03T11:00:00", "2020-1-2", "day"),
        ])
        res = beam.sleep(dry_run=False)  # must not raise
        assert res.get("status") == "consolidated"
        rows = beam.conn.execute(
            "SELECT event_date, event_date_precision FROM episodic_memory"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] is None, "garbage date must degrade to undated, not raise"
        assert rows[0][1] == "unknown"


class TestFailureInjectionAtomicity:
    def test_mid_transaction_failure_leaves_no_row(self, temp_db, monkeypatch):
        # F4: a failure between INSERT and commit must leave the episodic
        # table EMPTY (guarded transaction rolls the whole row back).
        beam = BeamMemory(session_id="fa1", db_path=temp_db)
        real_execute = beam.conn.cursor

        class ExplodingCursor:
            def __init__(self, inner):
                self._inner = inner
            def execute(self, sql, *a, **k):
                if "UPDATE episodic_memory SET timestamp" in sql:
                    raise sqlite3.OperationalError("injected failure")
                return self._inner.execute(sql, *a, **k)
            def __getattr__(self, name):
                return getattr(self._inner, name)

        monkeypatch.setattr(
            beam.conn, "cursor", lambda: ExplodingCursor(real_execute())
        )
        with pytest.raises(sqlite3.OperationalError, match="injected"):
            beam.consolidate_to_episodic(
                summary="- doomed", source_wm_ids=[], source="probe",
                event_timestamp="2026-04-01T12:00:00")
        n = beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        assert n == 0, "partial episodic row persisted across a mid-transaction failure"


class TestWMExportPinnedAndDates:
    def test_round_trip_carries_event_fields_and_pinned(self, temp_db):
        beam = BeamMemory(session_id="wm1", db_path=temp_db)
        _seed_old_wm(beam.conn, "wm1", [
            ("dated", "dated content", "2026-04-05T10:00:00", "2020-01-02", "day"),
        ])
        beam.conn.execute(
            "UPDATE working_memory SET pinned = 1 WHERE id = 'dated'"
        )
        beam.conn.commit()
        data = beam.export_to_dict()
        wm_row = next(r for r in data["working_memory"] if r["id"] == "dated")
        assert wm_row["event_date"] == "2020-01-02"
        assert wm_row["event_date_precision"] == "day"
        assert wm_row["pinned"] == 1

        beam2 = BeamMemory(session_id="wm1", db_path=temp_db.with_name("wm2.db"))
        stats = beam2.import_from_dict(data)
        assert stats["working_memory"]["inserted"] == 1
        row = beam2.conn.execute(
            "SELECT event_date, event_date_precision, pinned FROM working_memory WHERE id='dated'"
        ).fetchone()
        assert row[0] == "2020-01-02"
        assert row[1] == "day"
        assert row[2] == 1

    def test_restore_then_sleep_carries_event_date(self, temp_db, monkeypatch):
        # The maintainer's scenario: backup/restore must not strip the
        # content dates that a later sleep() aggregates.
        beam = BeamMemory(session_id="rs1", db_path=temp_db)
        _seed_old_wm(beam.conn, "rs1", [
            ("a", "content a", "2026-04-03T10:00:00", "2020-01-02", "day"),
            ("b", "content b", "2026-04-03T11:00:00", "2020-01-02", "day"),
        ])
        data = beam.export_to_dict()
        beam2 = BeamMemory(session_id="rs1", db_path=temp_db.with_name("rs2.db"))
        stats = beam2.import_from_dict(data)
        assert stats["working_memory"]["inserted"] == 2
        monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 168)
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 4, 8)
        monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
        res = beam2.sleep(dry_run=False)
        assert res.get("status") == "consolidated"
        row = beam2.conn.execute(
            "SELECT event_date, event_date_precision FROM episodic_memory"
        ).fetchone()
        assert row[0] == "2020-01-02", "restored rows lost their event_date before sleep"
        assert row[1] == "day"


class TestForceCutoffSentinel:
    """SQLite-version independence: the force-consolidation sentinel
    (9999-12-31T23:59:59.999999) must make every unconsolidated row
    eligible even on SQLite builds whose SQL datetime() returns NULL for
    julian-day overflow values (observed on CI's runners; fine on 3.53.x).
    The cutoff is normalized in Python (_utc_cutoff_sql) and bound as a
    plain string."""

    def test_force_sleep_consolidates_regardless_of_sqlite_datetime(self, temp_db):
        import sqlite3 as _sq
        beam = BeamMemory(session_id="force1", db_path=temp_db)
        # simulate the CI-side failure mode: SQL datetime() on the raw
        # sentinel returns NULL -> NULL comparison -> no rows. The Python
        # normalization must make the bound value a plain comparable string.
        _seed_old_wm(beam.conn, "force1", [
            ("fresh", "fresh row", "2099-01-01T00:00:00", None, None),
        ])
        res = beam.sleep(dry_run=False, force=True)
        assert res.get("status") == "consolidated", (
            f"force sentinel failed (SQLite {'.'.join(map(str, _sq.sqlite_version_info))}): {res}"
        )
        assert res.get("items_consolidated") == 1


class TestR5ReviewFixes:
    """Regressions for the adversarial-implementation-review findings."""

    def test_trim_never_deletes_pinned_rows(self, temp_db, monkeypatch):
        # R5-H1: pinned rows are exempt from trim (TTL and survivor-set).
        beam = BeamMemory(session_id="pin1", db_path=temp_db)
        monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 168)
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 4, 30)
        monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
        _seed_old_wm(beam.conn, "pin1", [
            ("pinned-old", "pinned content", "2026-03-01T10:00:00", None, None),
        ])
        beam.conn.execute("UPDATE working_memory SET pinned = 1 WHERE id = 'pinned-old'")
        beam.conn.commit()
        beam._trim_working_memory()
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'pinned-old'"
        ).fetchone()[0]
        assert n == 1, "trim deleted a pinned row (R5-H1)"

    def test_now_timestamp_not_immortal(self, temp_db, monkeypatch):
        # R5-H2: stored 'now' must not read the live clock — it degrades
        # to the epoch and becomes eligible for consolidation.
        beam = BeamMemory(session_id="nowts", db_path=temp_db)
        monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 168)
        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 4, 4, 12, 0, 0)
        monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES ('nowrow', 'content', 'conversation', 'now', 'nowts')"
        )
        beam.conn.commit()
        res = beam.sleep(dry_run=True)
        assert res.get("status") != "no_op", (
            f"'now' row stranded as immortal: {res}"
        )

    def test_event_timestamp_override_stored_naive_utc(self, temp_db):
        # R5-M1: an offset-bearing override must land naive-UTC so recall's
        # lexicographic date filters see the right day.
        beam = BeamMemory(session_id="ov1", db_path=temp_db)
        mid = beam.consolidate_to_episodic(
            summary="- override", source_wm_ids=[], source="probe",
            event_timestamp="2026-04-01T00:30:00+02:00")
        row = beam.conn.execute(
            "SELECT timestamp FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row[0] == "2026-03-31T22:30:00", (
            f"override stored raw: {row[0]}"
        )

    def test_unicode_digit_date_rejected(self, temp_db):
        # R5-M3: Arabic-Indic digits parse via strptime but sort wrong.
        beam = BeamMemory(session_id="uni1", db_path=temp_db)
        with pytest.raises(ValueError, match="event_date"):
            beam.consolidate_to_episodic(
                summary="- uni", source_wm_ids=[], source="probe",
                event_date="\u0662\u0660\u0662\u0666-01-01")

    def test_caller_transaction_not_stolen(self, temp_db):
        # R5-M2: a caller with an open transaction keeps ownership.
        beam = BeamMemory(session_id="own1", db_path=temp_db)
        beam.conn.execute("BEGIN")
        beam.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES ('marker', 'caller marker', 'conversation', '2026-01-01T00:00:00', 'own1')"
        )
        assert beam.conn.in_transaction
        beam.consolidate_to_episodic(
            summary="- owned", source_wm_ids=[], source="probe")
        assert beam.conn.in_transaction, (
            "consolidate_to_episodic stole the caller's transaction (R5-M2)"
        )
        beam.conn.rollback()
        n_ep = beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        n_wm = beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = 'marker'"
        ).fetchone()[0]
        assert n_ep == 0 and n_wm == 0


# ------------------------------------------------------- round-3 (dplush P1s)
class TestOffsetAndQuarantineRegressions:
    """dplush review 5120106306 (Sep 5): UTC cutoffs, import validation,
    event-under-caller-txn, calendar-valid event dates."""

    def test_trim_aware_row_judged_on_utc_instant(self, temp_db, monkeypatch):
        # dplush's repro discipline: with TZ=Europe/Berlin, an offset-
        # bearing row must be judged on its UTC instant. Frozen now =
        # 2026-04-11 12:00 (naive==UTC contract), TTL 1h → cutoff
        # 2026-04-11 11:00Z. Row 2026-04-11T10:00:00+02:00 == 08:00Z:
        # 3h past the cutoff → TRIMMED. Wall-clock judging (the old bug)
        # would read 10:00 local < 11:00 "local" the same way — so also
        # assert the UTC sibling at 12:30Z (30min
        # old) SURVIVES; only the UTC-instant predicate passes both.
        import time as _time
        monkeypatch.setenv("TZ", "Europe/Berlin")
        _time.tzset()
        try:
            monkeypatch.setattr(beam_module, "WORKING_MEMORY_TTL_HOURS", 1)

            class FakeDatetime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return cls(2026, 4, 11, 12, 0, 0)
            monkeypatch.setattr(beam_module, "datetime", FakeDatetime)
            beam = BeamMemory(session_id="s-tz", db_path=temp_db)
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source,"
                " timestamp, importance) VALUES (?,?,?,?,?,?)",
                ("aware-old", "s-tz", "offset row past cutoff",
                 "conversation", "2026-04-11T10:00:00+02:00", 0.5),
            )
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source,"
                " timestamp, importance) VALUES (?,?,?,?,?,?)",
                ("aware-young", "s-tz", "offset row inside cutoff",
                 "conversation", "2026-04-11T12:30:00+00:00", 0.5),
            )
            beam.conn.commit()
            beam._trim_working_memory()
            n_old = beam.conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE id='aware-old'"
            ).fetchone()[0]
            n_young = beam.conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE id='aware-young'"
            ).fetchone()[0]
        finally:
            monkeypatch.undo()
            _time.tzset()
        assert n_old == 0, "row past its UTC cutoff must be trimmed"
        assert n_young == 1, "row inside its UTC cutoff must survive"

    def test_import_preserves_unusable_timestamp(self, temp_db):
        """Round-4 dplush P1: import must NOT silently drop rows with
        unusable timestamps — preserve them with a quarantine (epoch)
        timestamp so the restore isn't lossy. The epoch means trim will
        collect them after TTL unless re-dated."""
        beam = BeamMemory(session_id="s-imp", db_path=temp_db)
        data = {"working_memory": [
            {"id": "good", "content": "valid row", "source": "conversation",
             "timestamp": "2026-09-05T10:00:00", "session_id": "s-imp",
             "importance": 0.5},
            {"id": "bad-none", "content": "none ts", "source": "conversation",
             "timestamp": None, "session_id": "s-imp", "importance": 0.5},
            {"id": "bad-junk", "content": "junk ts", "source": "conversation",
             "timestamp": "not-a-date", "session_id": "s-imp",
             "importance": 0.5},
        ]}
        stats = beam.import_from_dict(data)
        assert stats["working_memory"]["inserted"] == 3, (
            "all rows must be preserved (bad-timestamp rows get epoch ts)"
        )
        assert stats["working_memory"].get("imported_bad_timestamp", 0) == 2
        n = beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        assert n == 3, "no rows silently dropped from a backup restore"
        # The quarantine must exist in DATA, not just the log (round-6 F1):
        # bad rows carry the epoch timestamp and pinned=1 so neither trim
        # nor sleep collects them until re-dated/unpinned.
        for bad_id in ("bad-none", "bad-junk"):
            row = beam.conn.execute(
                "SELECT timestamp, pinned FROM working_memory WHERE id = ?",
                (bad_id,),
            ).fetchone()
            assert row["timestamp"] == "1970-01-01T00:00:00", (
                f"{bad_id}: epoch quarantine not stored: {row['timestamp']!r}"
            )
            assert row["pinned"] == 1, f"{bad_id}: quarantined row not pinned"
        good = beam.conn.execute(
            "SELECT timestamp, pinned FROM working_memory WHERE id = 'good'"
        ).fetchone()
        assert good["timestamp"] == "2026-09-05T10:00:00"
        assert not good["pinned"]

    def test_event_gate_raises_under_caller_txn(self, temp_db):
        beam = BeamMemory(session_id="s-gate", db_path=temp_db)
        events = []
        beam._event_emitter = lambda evt: events.append(evt)
        beam.remember("meeting on May 29")
        wid = beam.conn.execute(
            "SELECT id FROM working_memory LIMIT 1").fetchone()[0]
        beam.conn.execute("BEGIN")
        try:
            with pytest.raises(beam_module.MemoryTransactionStateError):
                beam.consolidate_to_episodic(
                    "summary", [wid], emit_event=True
                )
            # Entry-raise: no partial effect visible in the open txn.
            n = beam.conn.execute(
                "SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
            assert n == 0, "raise must happen before any write"
        finally:
            beam.conn.rollback()
        # Opt-out path consolidates fine inside the caller txn.
        beam.conn.execute("BEGIN")
        mid = beam.consolidate_to_episodic(
            "summary opt-out", [wid], emit_event=False
        )
        beam.conn.commit()
        assert mid
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        assert n == 1

    def test_event_date_calendar_validated(self, temp_db):
        beam = BeamMemory(session_id="s-date", db_path=temp_db)
        with pytest.raises(ValueError, match="real calendar date"):
            beam.consolidate_to_episodic(
                "s", [], event_date="2026-02-31", emit_event=False
            )


# --------
# Import quarantine + sanitizer (CodeRabbit 5122732528 + adversarial review passes): no abort mid-file, epoch+pinned quarantine, monotone force
# import pins, UTC normalization on remediation, sleep starvation behind
# unplaceable backlogs, UTC sleep cutoffs, reclaim clamps.
# --------


class TestImportEventDateSanitizer:
    def test_truthy_non_strings_degrade(self):
        for v in [123, True, 0.5, ["x"], {"a": 1}]:
            assert beam_module._sanitize_import_event_date(v, "day") == (None, "unknown")

    def test_falsy_degrade(self):
        for v in [None, "", "   ", 0, False]:
            assert beam_module._sanitize_import_event_date(v, "day") == (None, "unknown")

    def test_list_precision_degrades(self):
        assert beam_module._sanitize_import_event_date("2026-01-01", ["day"]) == ("2026-01-01", "unknown")
        assert beam_module._sanitize_import_event_date("2026-01-01", None) == ("2026-01-01", "unknown")

    def test_valid_pairs_survive(self):
        assert beam_module._sanitize_import_event_date("2026-05-05", "week") == ("2026-05-05", "week")
        assert beam_module._sanitize_import_event_date(" 2026-05-05 ", "month") == ("2026-05-05", "month")


class TestImportQuarantineNoAbort:
    def test_poison_second_row_leaves_first_committed(self, temp_db):
        beam = BeamMemory(session_id="s-abort", db_path=temp_db)
        data = {
            "working_memory": [
                {"id": "good", "content": "g", "source": "c", "timestamp": "2020-01-01T00:00:00", "importance": 0.5},
                {"id": "poison", "content": "p", "source": "c", "timestamp": "2020-01-01T00:00:00",
                 "importance": 0.5, "event_date": 123},
                {"id": "poison2", "content": "p2", "source": "c", "timestamp": "2020-01-01T00:00:00",
                 "importance": 0.5, "event_date": "2026-13-40", "event_date_precision": ["day"]},
            ]
        }
        stats = beam.import_from_dict(data)
        ids = {r["id"] for r in beam.conn.execute("SELECT id FROM working_memory")}
        assert ids == {"good", "poison", "poison2"}, f"import aborted mid-file: {ids}"
        assert not beam.conn.in_transaction
        assert stats["working_memory"]["inserted"] == 3
        rows = {
            r["id"]: r
            for r in beam.conn.execute(
                "SELECT id, event_date, event_date_precision FROM working_memory"
            )
        }
        assert rows["poison"]["event_date"] is None
        assert rows["poison"]["event_date_precision"] == "unknown"
        assert rows["poison2"]["event_date"] is None
        assert rows["poison2"]["event_date_precision"] == "unknown"


class TestForceImportPinMonotone:
    def test_absent_pin_key_does_not_unpin(self, temp_db):
        beam = BeamMemory(session_id="s-pin", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance, pinned) "
            "VALUES ('r1','s-pin','c','conversation','2020-01-01T00:00:00',0.5,1)")
        beam.conn.commit()
        data = {"working_memory": [{"id": "r1", "content": "c2", "source": "conversation",
                                    "timestamp": "2020-01-01T00:00:00", "importance": 0.5}]}
        beam.import_from_dict(data, force=True)
        row = beam.conn.execute("SELECT pinned FROM working_memory WHERE id='r1'").fetchone()
        assert row[0] == 1, "force import silently unpinned a pinned row"

    def test_explicit_zero_pin_does_not_unpin_via_force_import(self, temp_db):
        beam = BeamMemory(session_id="s-pin2", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance, pinned) "
            "VALUES ('r1','s-pin2','c','conversation','2020-01-01T00:00:00',0.5,1)")
        beam.conn.commit()
        data = {"working_memory": [{"id": "r1", "content": "c2", "source": "conversation",
                                    "session_id": "s-pin2", "timestamp": "2020-01-01T00:00:00",
                                    "importance": 0.5, "pinned": 0}]}
        beam.import_from_dict(data, force=True)
        row = beam.conn.execute("SELECT pinned FROM working_memory WHERE id='r1'").fetchone()
        assert row[0] == 1, "force import with pinned=0 must not lower an existing pin"
        assert beam.update_working("r1", pinned=0)
        assert beam.conn.execute("SELECT pinned FROM working_memory WHERE id='r1'").fetchone()[0] == 0

    def test_force_import_pinned_payload_raises_unpinned_row(self, temp_db):
        # Monotone semantics, raise direction: a force import with an
        # explicit pinned=1 must RAISE an unpinned live row to pinned —
        # the max(existing, incoming) guard. A backup-restore with
        # force=True must not silently drop the restored pin.
        beam = BeamMemory(session_id="s-pin3", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance, pinned) "
            "VALUES ('r1','s-pin3','c','conversation','2020-01-01T00:00:00',0.5,0)")
        beam.conn.commit()
        data = {"working_memory": [{"id": "r1", "content": "c2", "source": "conversation",
                                    "session_id": "s-pin3", "timestamp": "2020-01-01T00:00:00",
                                    "importance": 0.5, "pinned": 1}]}
        beam.import_from_dict(data, force=True)
        row = beam.conn.execute("SELECT pinned FROM working_memory WHERE id='r1'").fetchone()
        assert row[0] == 1, "force import with pinned=1 must raise an unpinned row"

    def test_update_working_pin_and_redate(self, temp_db):
        beam = BeamMemory(session_id="s-upd", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
            "VALUES ('r1','s-upd','c','conversation','1970-01-01T00:00:00',0.5)")
        beam.conn.commit()
        assert beam.update_working("r1", pinned=1)
        assert beam.conn.execute("SELECT pinned FROM working_memory WHERE id='r1'").fetchone()[0] == 1
        assert beam.update_working("r1", timestamp="2026-04-01T10:00:00")
        ts = beam.conn.execute("SELECT timestamp FROM working_memory WHERE id='r1'").fetchone()[0]
        assert ts == "2026-04-01T10:00:00"
        # offset-bearing ts stored as naive UTC (lexical compare correctness)
        assert beam.update_working("r1", timestamp="2026-04-01T00:30:00+02:00")
        ts = beam.conn.execute("SELECT timestamp FROM working_memory WHERE id='r1'").fetchone()[0]
        assert ts == "2026-03-31T22:30:00"
        with pytest.raises(ValueError):
            beam.update_working("r1", timestamp="not-a-date")


class TestImportTimestampParsing:
    def test_z_suffix_and_offsets_accepted(self):
        assert beam_module._import_timestamp_ok("2026-09-05T12:00:00Z")
        assert beam_module._import_timestamp_ok("2026-04-01T00:30:00+02:00")
        assert beam_module._import_timestamp_ok("2026-04-01T00:30:00-05:00")
        assert not beam_module._import_timestamp_ok(123)
        assert not beam_module._import_timestamp_ok(["2026-04-01T00:30:00Z"])
        assert not beam_module._import_timestamp_ok("garbage")
        assert not beam_module._import_timestamp_ok(None)


class TestSleepBacklogSurvivesUnplaceableRows:
    def test_eligible_row_consolidates_behind_junk_backlog(self, temp_db, monkeypatch):
        import mnemosyne.core.local_llm as local_llm_mod
        monkeypatch.setattr(local_llm_mod, "llm_available", lambda: False)
        beam = BeamMemory(session_id="s-starve", db_path=temp_db)
        monkeypatch.setattr(beam_module, "SLEEP_BATCH_SIZE", 20)
        junk_count = beam_module.SLEEP_BATCH_SIZE + 100
        rows = [("junk%05d" % i, "s-starve", "junk", "conversation", "not-a-ts", 0.5)
                for i in range(junk_count)]
        rows.append(("real-eligible", "s-starve", "old real row", "conversation",
                     "2020-01-01T00:00:00", 0.5))
        beam.conn.executemany(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
            "VALUES (?,?,?,?,?,?)", rows)
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "consolidated", f"eligible row starved: {res}"
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM episodic_memory WHERE summary_of LIKE '%real-eligible%'"
        ).fetchone()[0]
        assert n == 1, "eligible row was never summarized"

    def test_all_junk_reports_filtered_count(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s-starve2", db_path=temp_db)
        monkeypatch.setattr(beam_module, "SLEEP_BATCH_SIZE", 20)
        junk_count = beam_module.SLEEP_BATCH_SIZE + 100
        rows = [("junk%05d" % i, "s-starve2", "junk", "conversation", "not-a-ts", 0.5)
                for i in range(junk_count)]
        beam.conn.executemany(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
            "VALUES (?,?,?,?,?,?)", rows)
        beam.conn.commit()
        res = beam.sleep(dry_run=True)
        assert res.get("filtered_unplaceable", 0) == junk_count, f"backlog invisible: {res}"


class TestSleepCutoffIsUtc:
    def test_non_utc_host_uses_utc_cutoff(self, temp_db, monkeypatch):
        """Discriminating variant: a UTC+2 host whose wall clock (05:30)
        differs from UTC (03:30). With TTL 2h the UTC cutoff is 02:30Z —
        a row stamped 04:00Z is NOT claimable. A naive-local cutoff
        (04:30 local) WOULD claim it, so this test fails if sleep() ever
        reverts to the local wall clock."""
        import mnemosyne.core.beam as beam_mod
        monkeypatch.setattr(beam_mod, "WORKING_MEMORY_TTL_HOURS", 2)
        beam = BeamMemory(session_id="s-berlin", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
            "VALUES ('r-gap','s-berlin','gap row','conversation','2026-01-01T04:00:00',0.5)")
        beam.conn.commit()
        from datetime import datetime as _dt

        class _Frozen(_dt):
            # UTC+2 host: naive wall clock reads 05:30, UTC reads 03:30.
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return cls(2026, 1, 1, 3, 30, 0, tzinfo=tz)
                return cls(2026, 1, 1, 5, 30, 0)

        monkeypatch.setattr(beam_mod, "datetime", _Frozen)
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "no_op", f"naive-local cutoff leaked in: {res}"
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id='r-gap'"
        ).fetchone()[0]
        assert n == 1, "row must remain unconsolidated under a UTC cutoff"


class TestReclaimClamp:
    def test_huge_stale_after_seconds_no_overflow(self, temp_db):
        beam = BeamMemory(session_id="s-rec", db_path=temp_db)
        res = beam.reclaim_orphans(stale_after_seconds=10**12)
        assert res["status"] == "no_op", f"clamp failed: {res}"
        assert res["stale_after_seconds"] == 10**9, f"clamp not applied: {res}"
        # the returned cutoff must remain a parseable ISO instant
        assert isinstance(datetime.fromisoformat(res["cutoff"]), datetime)

    def test_non_finite_rejected(self, temp_db):
        beam = BeamMemory(session_id="s-rec2", db_path=temp_db)
        for bad in [float("inf"), float("nan")]:
            with pytest.raises(ValueError):
                beam.reclaim_orphans(stale_after_seconds=bad)


class TestEligibilityGateMatchesSleepClaims:
    """The auto-sleep count gate and sleep() must agree on which rows are
    claimable. sleep() pre-filters unparseable (unplaceable) timestamps
    before batching; a gate that still counts them reports eligibility
    forever while every pass consolidates nothing (CodeRabbit 3942181782)."""

    def test_count_gate_excludes_unplaceable_but_keeps_sentinels(self, temp_db):
        beam = BeamMemory(session_id="s-gate", db_path=temp_db)
        for rid, ts in (("poison", "not-a-ts"), ("real", "2020-01-01T00:00:00"),
                        ("sentinel", "now")):
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
                "VALUES (?,?,?,?,?,?)",
                (rid, "s-gate", "c-" + rid, "conversation", ts, 0.5))
        beam.conn.commit()
        # julian-day numbers, calendar-impossible dates and Excel-style
        # serials are SQLite-parseable but Python-unplaceable: they must
        # be excluded exactly like plain junk (F12-2).
        for rid, ts in (("julian", "2461000"), ("impossible", "2026-02-30T00:00:00"),
                        ("excel", "45000")):
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
                "VALUES (?,?,?,?,?,?)",
                (rid, "s-gate", "c-" + rid, "conversation", ts, 0.5))
        beam.conn.commit()
        n = beam._count_unconsolidated_before(
            beam_module._utc_cutoff_sql("2021-01-01T00:00:00"))
        # real + sentinel qualify; every unplaceable form must NOT count
        # (without the predicate their 1970-degraded CASE forms count too,
        # so a regression makes this assertion fail).
        assert n == 2, f"gate disagrees with sleep claimability: {n}"

    def test_sleep_all_sessions_skips_unplaceable_only_session(self, temp_db):
        # A session holding ONLY unparseable rows must never be scanned or
        # re-opened by sleep_all_sessions: sleep()'s placeability pre-filter
        # can never claim those rows, so eligibility must be 0. Without the
        # predicate their 1970-degraded CASE form makes the session
        # eligible on every maintenance pass forever.
        beam = BeamMemory(session_id="s-maint", db_path=temp_db)
        for rid in ("j1", "j2"):
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
                "VALUES (?,?,?,?,?,?)",
                (rid, "alien-poisoned", "c", "conversation", "not-a-ts", 0.5))
        beam.conn.commit()
        res = beam.sleep_all_sessions(dry_run=True)
        assert res["status"] == "no_op", f"poisoned session scanned: {res}"
        assert res["sessions_scanned"] == 0, f"poisoned session scanned: {res}"
        assert res["session_results"] == []


class TestImportNormalizesAcceptedTimestamps:
    """Accepted offset/Z timestamps are stored as naive UTC so recall()'s
    lexicographic date filters and the UTC cutoffs stay aligned on the
    import path too (CodeRabbit 3942181785)."""

    def _ep_row(self, conn, mid):
        return conn.execute(
            "SELECT timestamp FROM episodic_memory WHERE id=?", (mid,)).fetchone()[0]

    def test_working_offset_stored_naive_utc(self, temp_db):
        beam = BeamMemory(session_id="s-imp-utc", db_path=temp_db)
        data = {"working_memory": [{
            "id": "w1", "content": "c", "source": "conversation",
            "timestamp": "2026-04-01T00:30:00+02:00", "importance": 0.5}]}
        beam.import_from_dict(data)
        row = beam.conn.execute(
            "SELECT timestamp FROM working_memory WHERE id='w1'").fetchone()
        assert row[0] == "2026-03-31T22:30:00", f"offset not normalized: {row[0]!r}"

    def test_episodic_z_suffix_stored_naive_utc(self, temp_db):
        beam = BeamMemory(session_id="s-imp-utc2", db_path=temp_db)
        data = {"episodic_memory": [{
            "id": "e1", "content": "c", "source": "probe",
            "timestamp": "2026-04-01T00:30:00Z"}]}
        beam.import_from_dict(data)
        row = beam.conn.execute(
            "SELECT timestamp FROM episodic_memory WHERE id='e1'").fetchone()
        assert row[0] == "2026-04-01T00:30:00", f"Z not normalized: {row[0]!r}"

    def test_quarantine_branch_keeps_epoch(self, temp_db):
        # regression: the normalize-else must not rewrite the epoch branch
        beam = BeamMemory(session_id="s-imp-utc3", db_path=temp_db)
        data = {"working_memory": [{
            "id": "w2", "content": "c", "source": "conversation",
            "timestamp": "garbage", "importance": 0.5}]}
        beam.import_from_dict(data)
        row = beam.conn.execute(
            "SELECT timestamp, pinned FROM working_memory WHERE id='w2'").fetchone()
        assert row[0] == "1970-01-01T00:00:00", f"epoch quarantine lost: {row[0]!r}"
        assert row[1] == 1


class TestPoisonClassesExcludedEverywhere:
    """F12-4: julian/numeric/impossible-date timestamps are excluded by
    every eligibility surface (claim pre-filter, count gate, sleep_all
    sessions) exactly like plain junk — they must never occupy claim
    slots (starvation) nor make poisoned sessions look eligible."""

    def test_real_row_consolidates_behind_julian_backlog(self, temp_db, monkeypatch):
        import mnemosyne.core.local_llm as local_llm_mod
        monkeypatch.setattr(local_llm_mod, "llm_available", lambda: False)
        beam = BeamMemory(session_id="s-julian-starve", db_path=temp_db)
        monkeypatch.setattr(beam_module, "SLEEP_BATCH_SIZE", 20)
        junk_count = beam_module.SLEEP_BATCH_SIZE + 100
        rows = [("jul%05d" % i, "s-julian-starve", "junk", "conversation", "2461000", 0.5)
                for i in range(junk_count)]
        rows.append(("real-eligible", "s-julian-starve", "old real row", "conversation",
                     "2020-01-01T00:00:00", 0.5))
        beam.conn.executemany(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
            "VALUES (?,?,?,?,?,?)", rows)
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "consolidated", f"real row starved: {res}"
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM episodic_memory WHERE summary_of LIKE '%real-eligible%'"
        ).fetchone()[0]
        assert n == 1, "eligible row was never summarized"
        remaining = beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE timestamp='2461000' AND consolidated_at IS NULL"
        ).fetchone()[0]
        assert remaining == junk_count, "julian rows must stay unconsolidated"

    def test_sleep_all_sessions_skips_julian_only_session(self, temp_db):
        beam = BeamMemory(session_id="s-maint2", db_path=temp_db)
        for rid in ("k1", "k2"):
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
                "VALUES (?,?,?,?,?,?)",
                (rid, "alien-julian", "c", "conversation", "2461000", 0.5))
        beam.conn.commit()
        res = beam.sleep_all_sessions(dry_run=True)
        assert res["status"] == "no_op", f"julian session scanned: {res}"
        assert res["sessions_scanned"] == 0

    def test_sleep_all_sessions_skips_pinned_quarantine_session(self, temp_db):
        # Import quarantine stores epoch + pinned=1; the eligibility query
        # must exclude pinned rows exactly like sleep()'s claim does
        # (F12-3) — otherwise the store's own junk producer makes every
        # maintenance pass scan a session that can never consolidate.
        beam = BeamMemory(session_id="s-maint3", db_path=temp_db)
        data = {"working_memory": [
            {"id": "q1", "content": "garbage-ts row", "source": "conversation",
             "timestamp": "not-a-ts", "importance": 0.5},
            {"id": "q2", "content": "garbage-ts row 2", "source": "conversation",
             "timestamp": "not-a-ts", "importance": 0.5}]}
        beam.import_from_dict(data)
        # payload rows without a session key land in 'default'; move them
        # to an alien session so sleep_all_sessions sees a foreign store
        beam.conn.execute(
            "UPDATE working_memory SET session_id='alien-quarantined' WHERE session_id='default'")
        beam.conn.commit()
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE session_id='alien-quarantined' AND pinned=1"
        ).fetchone()[0] == 2
        res = beam.sleep_all_sessions(dry_run=True)
        assert res["status"] == "no_op", f"pinned-quarantine session scanned: {res}"
        assert res["sessions_scanned"] == 0, f"pinned-quarantine session scanned: {res}"


class TestImportPaddedAndLastRecalled:
    def test_padded_offset_import_stored_naive_utc(self, temp_db):
        # F12-1: _import_timestamp_ok accepts whitespace-padded values;
        # the normalize branch must strip before parsing or the import
        # crashes mid-restore.
        beam = BeamMemory(session_id="s-pad", db_path=temp_db)
        data = {"working_memory": [
            {"id": "w1", "content": "c", "source": "conversation",
             "timestamp": "  2026-04-01T00:30:00+02:00  ", "importance": 0.5},
            {"id": "w2", "content": "c2", "source": "conversation",
             "timestamp": " 2026-04-01T00:30:00Z ", "importance": 0.5}]}
        stats = beam.import_from_dict(data)
        assert stats["working_memory"]["inserted"] == 2
        r1 = beam.conn.execute(
            "SELECT timestamp FROM working_memory WHERE id='w1'").fetchone()[0]
        r2 = beam.conn.execute(
            "SELECT timestamp FROM working_memory WHERE id='w2'").fetchone()[0]
        assert r1 == "2026-03-31T22:30:00", f"padded offset not normalized: {r1!r}"
        assert r2 == "2026-04-01T00:30:00", f"padded Z not normalized: {r2!r}"

    def test_numeric_last_recalled_self_heals(self, temp_db):
        # F11-1: SQLite NUMERIC affinity stores numeric-looking last_recalled
        # as int; the get_context bump must self-heal, not raise
        # AttributeError on the hot path.
        beam = BeamMemory(session_id="s-numrec", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance, last_recalled) "
            "VALUES ('r1','s-numrec','c','conversation','2020-01-01T00:00:00',0.5,1700000000)")
        beam.conn.commit()
        ctx = beam.get_context(limit=10)
        assert ctx is not None
        row = beam.conn.execute(
            "SELECT last_recalled FROM working_memory WHERE id='r1'").fetchone()[0]
        assert isinstance(row, str) and row.startswith("20"), f"not healed: {row!r}"


class TestRound13GapClosers:
    """Three mutation-survivor gaps from the round-13 audit: claim-site
    julian exclusion with a NEWER real row, the episodic padded-import
    strip, and the count-gate pinned filter."""

    def test_newer_real_row_consolidates_behind_julian_backlog(self, temp_db, monkeypatch):
        # Discriminating variant: the real row is dated AFTER the julian
        # rows' SQLite placement (2026 vs 2025-11-20), so under a revert
        # of the claim-site exclusion the julian backlog occupies the
        # whole batch (they sort first) and the real row starves.
        import mnemosyne.core.local_llm as local_llm_mod
        monkeypatch.setattr(local_llm_mod, "llm_available", lambda: False)
        beam = BeamMemory(session_id="s-julian-starve2", db_path=temp_db)
        monkeypatch.setattr(beam_module, "SLEEP_BATCH_SIZE", 20)
        junk_count = beam_module.SLEEP_BATCH_SIZE + 100
        rows = [("jul%05d" % i, "s-julian-starve2", "junk", "conversation", "2461000", 0.5)
                for i in range(junk_count)]
        rows.append(("real-2026", "s-julian-starve2", "newer real row", "conversation",
                     "2026-01-01T00:00:00", 0.5))
        beam.conn.executemany(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
            "VALUES (?,?,?,?,?,?)", rows)
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "consolidated", f"real row starved: {res}"
        n = beam.conn.execute(
            "SELECT COUNT(*) FROM episodic_memory WHERE summary_of LIKE '%real-2026%'"
        ).fetchone()[0]
        assert n == 1, "newer real row was never summarized"

    def test_padded_episodic_import_stored_naive_utc(self, temp_db):
        # F12-1 parity on the episodic path: whitespace-padded accepted
        # timestamps must not crash the import normalize branch.
        beam = BeamMemory(session_id="s-pad-ep", db_path=temp_db)
        data = {"episodic_memory": [
            {"id": "e1", "content": "c", "source": "probe",
             "timestamp": "  2026-04-01T00:30:00+02:00  "}]}
        stats = beam.import_from_dict(data)
        assert stats["episodic_memory"]["inserted"] == 1
        row = beam.conn.execute(
            "SELECT timestamp FROM episodic_memory WHERE id='e1'").fetchone()
        assert row[0] == "2026-03-31T22:30:00", f"padded episodic not normalized: {row[0]!r}"

    def test_count_gate_excludes_pinned_rows(self, temp_db):
        # The count gate mirrors sleep()'s claim, which excludes pinned:
        # three pinned-quarantine rows must count 0, not churn the
        # auto-sleep gate forever.
        beam = BeamMemory(session_id="s-gate-pin", db_path=temp_db)
        for rid in ("q1", "q2", "q3"):
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance, pinned) "
                "VALUES (?,?,?,?,?,?,1)",
                (rid, "s-gate-pin", "c", "conversation", "1970-01-01T00:00:00", 0.5))
        beam.conn.commit()
        n = beam._count_unconsolidated_before(
            beam_module._utc_cutoff_sql("2030-01-01T00:00:00"))
        assert n == 0, f"count gate counts pinned rows: {n}"

    def test_sleep_noop_reports_pinned_exempt(self, temp_db):
        # C4: a pinned-only unconsolidated backlog yields a no_op that
        # names the exempt rows instead of a bare "nothing to do".
        beam = BeamMemory(session_id="s-pinvis", db_path=temp_db)
        for rid in ("q1", "q2"):
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance, pinned) "
                "VALUES (?,?,?,?,?,?,1)",
                (rid, "s-pinvis", "c", "conversation", "1970-01-01T00:00:00", 0.5))
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res["status"] == "no_op"
        assert res.get("pinned_exempt") == 2, f"pinned exempt invisible: {res}"


class TestQueryCacheInvalidation:
    """C2/C3: mutating paths must invalidate the enhanced-recall query
    cache only when their writes are visible to other connections."""

    def test_update_working_invalidates_after_commit(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s-inv1", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance, pinned) "
            "VALUES ('r1','s-inv1','c','conversation','2020-01-01T00:00:00',0.5,1)")
        beam.conn.commit()
        calls = []
        monkeypatch.setattr(beam, "_invalidate_query_cache_after_commit",
                            lambda op: calls.append(op))
        assert beam.update_working("r1", pinned=0)
        assert calls == ["update_working"], f"no invalidation after update: {calls}"

    def test_update_working_noop_does_not_invalidate(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="s-inv2", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
            "VALUES ('r1','s-inv2','c','conversation','2020-01-01T00:00:00',0.5)")
        beam.conn.commit()
        calls = []
        monkeypatch.setattr(beam, "_invalidate_query_cache_after_commit",
                            lambda op: calls.append(op))
        assert not beam.update_working("r1")  # nothing to update
        assert calls == []

    def test_consolidate_caller_txn_defers_invalidation(self, temp_db, monkeypatch):
        # Under a caller-owned transaction the row is not yet visible to
        # other connections; invalidating now would let a concurrent
        # reader repopulate the cache from the pre-consolidation state.
        beam = BeamMemory(session_id="s-inv3", db_path=temp_db)
        calls = []
        monkeypatch.setattr(beam, "_invalidate_query_cache_after_commit",
                            lambda op: calls.append(op))
        with beam.conn:  # caller-owned transaction
            # open a real txn first (the context manager alone is lazy)
            beam.conn.execute(
                "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance) "
                "VALUES ('txn-open','s-inv3','c','conversation','2020-01-01T00:00:00',0.5)")
            assert beam.conn.in_transaction
            mid = beam.consolidate_to_episodic(
                summary="- caller txn", source_wm_ids=[], source="probe",
                emit_event=False)
            assert calls == [], f"invalidated inside caller txn: {calls}"
            assert beam.conn.in_transaction
        # after the caller commits, the writer-side contract says the
        # caller invalidates; here the next mutation will do it
        assert mid


class TestSleepNonTextEventDate:
    """dplush P1: legacy/foreign rows can carry non-text stored
    event_date values (BLOB/number); sleep() must degrade them to an
    undated summary instead of raising AFTER the source rows were
    claimed — a raise there strands rows with claim markers set and no
    episodic summary."""

    @staticmethod
    def _episodic_rows(db_path):
        beam = BeamMemory(session_id="_probe", db_path=db_path)
        return beam.conn.execute(
            "SELECT id, event_date, event_date_precision, summary_of "
            "FROM episodic_memory ORDER BY rowid").fetchall()

    def test_blob_event_date_degrades_to_undated_summary(self, temp_db, monkeypatch):
        import sqlite3 as _sqlite3
        import mnemosyne.core.local_llm as local_llm_mod
        monkeypatch.setattr(local_llm_mod, "llm_available", lambda: False)
        beam = BeamMemory(session_id="s-blobdate", db_path=temp_db)
        # BOTH rows carry the SAME BLOB date: the pre-fix aggregation
        # reached _event_date_valid(bytes) and raised a TypeError after
        # the claim commit (stranding the rows). Identical blob values
        # also prove the degradation is per-VALUE, not per-group.
        blob = _sqlite3.Binary(b"\x00\x01binary-date")
        beam.conn.execute(
            "INSERT INTO working_memory "
            "(id, session_id, content, source, timestamp, importance, event_date, event_date_precision) "
            "VALUES ('a','s-blobdate','content a','conversation','2020-01-01T00:00:00',0.5,?, 'day')",
            (blob,))
        beam.conn.execute(
            "INSERT INTO working_memory "
            "(id, session_id, content, source, timestamp, importance, event_date, event_date_precision) "
            "VALUES ('b','s-blobdate','content b','conversation','2020-01-02T00:00:00',0.5,?, 'day')",
            (blob,))
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "consolidated", f"sleep failed: {res}"
        # the summary was written with NO manufactured date and unknown
        # precision (the blob row is undated, so the group is undated)
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1, f"expected one summary, got {len(rows)}"
        assert rows[0][1] is None, f"undated group must not carry event_date: {rows[0][1]!r}"
        assert rows[0][2] == "unknown", f"precision must be unknown: {rows[0][2]!r}"
        # no stranding: every source row is claimed AND summarized
        for rid in ("a", "b"):
            n = beam.conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE id=? AND consolidated_at IS NOT NULL",
                (rid,)).fetchone()[0]
            assert n == 1, f"row {rid} stranded (claim marker missing)"
        summary_of = beam.conn.execute(
            "SELECT summary_of FROM episodic_memory").fetchone()[0]
        assert "a" in summary_of and "b" in summary_of, summary_of

    def test_public_override_non_string_raises_valueerror(self, temp_db):
        beam = BeamMemory(session_id="s-pub", db_path=temp_db)
        import pytest as _pytest
        for bad in (b"\x00blob", 0, [], {}, 20200102):
            with _pytest.raises(ValueError, match="must be a string"):
                beam.consolidate_to_episodic(
                    summary="- x", source_wm_ids=[], source="probe",
                    event_date=bad, event_date_precision="day",
                    emit_event=False)
        # None alone remains "not provided" (undated summary)
        mid = beam.consolidate_to_episodic(
            summary="- none", source_wm_ids=[], source="probe",
            event_date=None, event_date_precision="day",
            emit_event=False)
        row = beam.conn.execute(
            "SELECT event_date FROM episodic_memory WHERE id=?",
            (mid,)).fetchone()
        assert row[0] is None

    def test_public_unhashable_precision_raises_valueerror(self, temp_db):
        beam = BeamMemory(session_id="s-pub2", db_path=temp_db)
        import pytest as _pytest
        with _pytest.raises(ValueError, match="must be a string"):
            beam.consolidate_to_episodic(
                summary="- x", source_wm_ids=[], source="probe",
                event_date="2020-01-02", event_date_precision={},
                emit_event=False)

    def test_public_non_string_timestamp_raises_valueerror(self, temp_db):
        beam = BeamMemory(session_id="s-pub3", db_path=temp_db)
        import pytest as _pytest
        with _pytest.raises(ValueError, match="must be a string"):
            beam.consolidate_to_episodic(
                summary="- x", source_wm_ids=[], source="probe",
                event_timestamp=b"\x00blob", emit_event=False)

    def test_blob_precision_on_agreeing_dates_keeps_date_unknown_precision(
            self, temp_db, monkeypatch):
        # F2/F3 witness: the precision isinstance guard must degrade a
        # BLOB precision to 'unknown' while the agreed DATE survives.
        import sqlite3 as _sqlite3
        import mnemosyne.core.local_llm as local_llm_mod
        monkeypatch.setattr(local_llm_mod, "llm_available", lambda: False)
        beam = BeamMemory(session_id="s-blobprec", db_path=temp_db)
        blob = _sqlite3.Binary(b"day-ish")
        for rid, ts in (("a", "2020-01-01T00:00:00"), ("b", "2020-01-02T00:00:00")):
            beam.conn.execute(
                "INSERT INTO working_memory "
                "(id, session_id, content, source, timestamp, importance, event_date, event_date_precision) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (rid, "s-blobprec", "c", "conversation", ts, 0.5,
                 "2020-01-02", blob))
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "consolidated", res
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] == "2020-01-02", "agreed date must survive"
        assert rows[0][2] == "unknown", (
            f"BLOB precision must degrade to unknown: {rows[0][2]!r}")

    def test_blob_valid_until_does_not_strand(self, temp_db, monkeypatch):
        # F1 witness (audit15-pr2): non-text valid_until on a stored row
        # must not raise post-claim; the group consolidates and no row is
        # stranded with claim markers set and no summary.
        import sqlite3 as _sqlite3
        import mnemosyne.core.local_llm as local_llm_mod
        monkeypatch.setattr(local_llm_mod, "llm_available", lambda: False)
        beam = BeamMemory(session_id="s-blobvu", db_path=temp_db)
        blob = _sqlite3.Binary(b"\x00\x01valid-until")
        for rid, ts in (("a", "2020-01-01T00:00:00"), ("b", "2020-01-02T00:00:00")):
            beam.conn.execute(
                "INSERT INTO working_memory "
                "(id, session_id, content, source, timestamp, importance, valid_until) "
                "VALUES (?,?,?,?,?,?,?)",
                (rid, "s-blobvu", "c", "conversation", ts, 0.5, blob))
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "consolidated", f"sleep failed: {res}"
        for rid in ("a", "b"):
            n = beam.conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE id=? AND consolidated_at IS NOT NULL",
                (rid,)).fetchone()[0]
            assert n == 1, f"row {rid} stranded"
        summary_of = beam.conn.execute(
            "SELECT summary_of FROM episodic_memory").fetchone()[0]
        assert "a" in summary_of and "b" in summary_of


class TestRound16Witnesses:
    """Audit16 residuals: OverflowError-class valid_until, falsy
    non-string valid_until remap, NULL-precision load-bearing guard,
    and the public event_timestamp overflow ValueError."""

    @staticmethod
    def _episodic_rows(db_path):
        beam = BeamMemory(session_id="_probe", db_path=db_path)
        return beam.conn.execute(
            "SELECT id, event_date, event_date_precision, summary_of "
            "FROM episodic_memory ORDER BY rowid").fetchall()

    def test_year9999_valid_until_does_not_strand(self, temp_db, monkeypatch):
        # F1: '9999-12-31T00:01:00-23:59' overflows the datetime
        # arithmetic inside parsing; it must be treated as unparseable,
        # never raise post-claim (stranding the group).
        import mnemosyne.core.local_llm as local_llm_mod
        monkeypatch.setattr(local_llm_mod, "llm_available", lambda: False)
        beam = BeamMemory(session_id="s-9999", db_path=temp_db)
        for rid, ts in (("a", "2020-01-01T00:00:00"), ("b", "2020-01-02T00:00:00")):
            beam.conn.execute(
                "INSERT INTO working_memory "
                "(id, session_id, content, source, timestamp, importance, valid_until) "
                "VALUES (?,?,?,?,?,?,?)",
                (rid, "s-9999", "c", "conversation", ts, 0.5,
                 "9999-12-31T00:01:00-23:59"))
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "consolidated", f"sleep failed: {res}"
        for rid in ("a", "b"):
            n = beam.conn.execute(
                "SELECT COUNT(*) FROM working_memory WHERE id=? AND consolidated_at IS NOT NULL",
                (rid,)).fetchone()[0]
            assert n == 1, f"row {rid} stranded"

    def test_falsy_non_string_valid_until_imports_as_none(self, temp_db):
        # F2: falsy non-strings (b'', 0) must be remapped to no-expiry
        # (NULL), not stored raw (raw b''/0 reads as expired).
        beam = BeamMemory(session_id="s-falsyvu", db_path=temp_db)
        data = {"working_memory": [
            {"id": "w1", "content": "c", "source": "conversation",
             "timestamp": "2020-01-01T00:00:00", "importance": 0.5,
             "valid_until": 0},
            {"id": "w2", "content": "c2", "source": "conversation",
             "timestamp": "2020-01-02T00:00:00", "importance": 0.5,
             "valid_until": b""},
        ]}
        stats = beam.import_from_dict(data)
        assert stats["working_memory"]["inserted"] == 2
        for rid in ("w1", "w2"):
            row = beam.conn.execute(
                "SELECT valid_until FROM working_memory WHERE id=?",
                (rid,)).fetchone()[0]
            assert row is None, f"{rid}: falsy non-string must map to NULL: {row!r}"

    def test_null_precision_row_does_not_strand(self, temp_db, monkeypatch):
        # F3: the newest row carrying a NULL event_date_precision must
        # degrade to 'unknown' (the isinstance guard is load-bearing).
        import mnemosyne.core.local_llm as local_llm_mod
        monkeypatch.setattr(local_llm_mod, "llm_available", lambda: False)
        beam = BeamMemory(session_id="s-nullprec", db_path=temp_db)
        for rid, ts in (("a", "2020-01-01T00:00:00"), ("b", "2020-01-02T00:00:00")):
            beam.conn.execute(
                "INSERT INTO working_memory "
                "(id, session_id, content, source, timestamp, importance, event_date, event_date_precision) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (rid, "s-nullprec", "c", "conversation", ts, 0.5,
                 "2020-01-02", None))
        beam.conn.commit()
        res = beam.sleep(dry_run=False)
        assert res.get("status") == "consolidated", f"sleep failed: {res}"
        rows = self._episodic_rows(temp_db)
        assert len(rows) == 1
        assert rows[0][1] == "2020-01-02"
        assert rows[0][2] == "unknown", (
            f"NULL precision must degrade to unknown: {rows[0][2]!r}")

    def test_public_year9999_timestamp_raises_valueerror(self, temp_db):
        beam = BeamMemory(session_id="s-9999pub", db_path=temp_db)
        import pytest as _pytest
        with _pytest.raises(ValueError, match="parseable ISO-8601"):
            beam.consolidate_to_episodic(
                summary="- x", source_wm_ids=[], source="probe",
                event_timestamp="9999-12-31T00:01:00-23:59",
                emit_event=False)
