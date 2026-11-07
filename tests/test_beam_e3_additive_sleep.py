"""Regression tests for E3 — additive sleep().

Pre-E3 contract: BeamMemory.sleep() consolidated old working_memory rows
into an episodic_memory summary and then DELETED the source rows. The
new contract (per maintainer decision 2026-05-10: "Kill summarize-and-
delete. Originals stay.") is additive:

  - Source working_memory rows REMAIN after sleep
  - A new `consolidated_at` TIMESTAMP column on working_memory is set on
    the rows sleep processed
  - sleep() doesn't pick up rows that already have consolidated_at set
    (so calling sleep twice doesn't double-summarize the same originals)
  - Recall surfaces both the original working_memory row AND the
    episodic summary row when both are relevant

This blocks experiment Arm B (ADD-only ingest) of the BEAM-recovery
experiment.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _wm_count(db_path, session_id=None):
    conn = sqlite3.connect(str(db_path))
    try:
        if session_id is None:
            return conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        return conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
    finally:
        conn.close()


def _consolidated_rows(db_path, session_id):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT id, consolidated_at FROM working_memory "
            "WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()


def _seed_old_wm(db_path, session_id, n, ts_offset_hours=200):
    """Insert n old working_memory rows for the given session. Uses
    distinct content per row so each is uniquely identifiable."""
    conn = sqlite3.connect(str(db_path))
    ts = (datetime.now() - timedelta(hours=ts_offset_hours)).isoformat()
    rows = [
        (f"e3-{session_id}-{i}", f"e3-content-{i}", "conversation", ts, session_id)
        for i in range(n)
    ]
    conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return [r[0] for r in rows]


def _utc_naive_iso(hours_ago: float) -> str:
    """Claim timestamp in the same clock sleep() writes (naive UTC), so the
    staleness check is independent of the machine's local timezone."""
    return (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=hours_ago)).isoformat()


class TestE3AdditiveSleep:

    def test_sleep_preserves_working_memory_rows(self, temp_db):
        """Post-E3, source working_memory rows survive sleep()."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        seeded_ids = _seed_old_wm(temp_db, "s1", n=3)

        assert _wm_count(temp_db, "s1") == 3

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["items_consolidated"] == 3

        # The whole point of E3: originals stay.
        post_count = _wm_count(temp_db, "s1")
        assert post_count == 3, (
            f"sleep() deleted working_memory rows — pre-E3 behavior. "
            f"Expected 3 originals to remain, got {post_count}."
        )

        # IDs unchanged.
        conn = sqlite3.connect(str(temp_db))
        post_ids = sorted(r[0] for r in conn.execute(
            "SELECT id FROM working_memory WHERE session_id = 's1'"
        ).fetchall())
        conn.close()
        assert post_ids == sorted(seeded_ids)

    def test_sleep_marks_consolidated_at(self, temp_db):
        """After sleep, every processed row has consolidated_at set."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=2)

        beam.sleep(dry_run=False)

        rows = _consolidated_rows(temp_db, "s1")
        assert len(rows) == 2
        for row_id, consolidated_at in rows:
            assert consolidated_at is not None, (
                f"row {row_id} survived sleep but consolidated_at is NULL "
                f"— sleep didn't mark it, so the next sleep cycle will "
                f"re-consolidate it and produce a duplicate summary."
            )

    def test_sleep_does_not_reconsolidate_marked_rows(self, temp_db):
        """Calling sleep twice doesn't pick up already-marked rows."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=2)

        first = beam.sleep(dry_run=False)
        assert first["items_consolidated"] == 2
        first_episodic = sqlite3.connect(str(temp_db)).execute(
            "SELECT COUNT(*) FROM episodic_memory"
        ).fetchone()[0]

        # Second sleep should find no eligible rows (status no_op) and
        # crucially not create a second summary.
        second = beam.sleep(dry_run=False)
        assert second["status"] == "no_op", (
            f"sleep re-processed already-consolidated rows; got {second}"
        )

        second_episodic = sqlite3.connect(str(temp_db)).execute(
            "SELECT COUNT(*) FROM episodic_memory"
        ).fetchone()[0]
        assert second_episodic == first_episodic, (
            f"sleep produced a duplicate summary on the second call: "
            f"first={first_episodic}, second={second_episodic}"
        )

    def test_dry_run_does_not_mark_consolidated_at(self, temp_db):
        """Dry run must not mutate working_memory (no consolidated_at writes)."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=2)

        result = beam.sleep(dry_run=True)
        assert result["status"] == "dry_run"

        rows = _consolidated_rows(temp_db, "s1")
        for row_id, consolidated_at in rows:
            assert consolidated_at is None, (
                f"dry_run set consolidated_at on row {row_id} — "
                f"dry_run must be side-effect-free."
            )

    def test_consolidated_originals_remain_recallable(self, temp_db, monkeypatch):
        """[E3] Originals stay queryable through recall() after sleep.
        Uses LLM-disabled path so unique tokens survive into the
        aaak-encoded summary; the assertion is that recall returns hits
        whose content matches the ORIGINAL working_memory row content,
        not just the summary."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: False
        )
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        # Distinctive token only in the original, very unlikely to
        # appear in an aaak-encoded summary.
        conn = sqlite3.connect(str(temp_db))
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("orig1", "original wm marker zzzunique42", "conversation", old_ts, "s1"),
        )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)

        results = beam.recall("zzzunique42", top_k=10)
        assert results, "consolidated original is unreachable through recall()"
        # The exact-token original should be findable; the wm row carries
        # tier=='working' in the post-recall shape.
        assert any(
            r.get("tier") == "working" and "zzzunique42" in (r.get("content") or "")
            for r in results
        ), (
            f"recall did not surface the original working_memory row that "
            f"carries the unique token. E3 contract requires originals "
            f"remain recallable. Got: "
            f"{[(r.get('tier'), (r.get('content') or '')[:40]) for r in results]}"
        )

    def test_sleep_remains_session_scoped_with_marker(self, temp_db):
        """E3 version of the legacy test_sleep_remains_session_scoped:
        cross-session row stays untouched (no consolidated_at marker)
        because the caller's session is s1."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        _seed_old_wm(temp_db, "s2", n=1)

        beam.sleep(dry_run=False)

        conn = sqlite3.connect(str(temp_db))
        rows = conn.execute(
            "SELECT session_id, consolidated_at FROM working_memory ORDER BY session_id"
        ).fetchall()
        conn.close()

        # Both rows still exist; only s1's is marked.
        assert len(rows) == 2
        by_session = {r[0]: r[1] for r in rows}
        assert by_session["s1"] is not None, "s1 row not marked consolidated"
        assert by_session["s2"] is None, (
            "s2 row was marked despite caller being session s1 — cross-"
            "session leak on the consolidation marker"
        )

    def test_consolidated_at_column_added_on_legacy_db(self, temp_db):
        """Pre-E3 databases that don't have consolidated_at must get the
        column added by init_beam() with existing rows backfilled to
        'already consolidated' (preserving the pre-E3 expectation that
        old rows are gone)."""
        # Simulate a legacy DB: create the working_memory table without
        # consolidated_at, insert a row, then re-init.
        legacy_conn = sqlite3.connect(str(temp_db))
        legacy_conn.execute("""
            CREATE TABLE working_memory (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                timestamp TEXT,
                session_id TEXT DEFAULT 'default',
                importance REAL DEFAULT 0.5,
                metadata_json TEXT,
                veracity TEXT DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        legacy_conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("legacy-1", "legacy content", "conversation", old_ts, "s1"),
        )
        legacy_conn.commit()
        legacy_conn.close()

        # init_beam via BeamMemory construction should add the column.
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        conn = sqlite3.connect(str(temp_db))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(working_memory)").fetchall()]
        conn.close()
        assert "consolidated_at" in cols, (
            "init_beam did not add consolidated_at column to legacy DB"
        )

        # Existing legacy row should be backfilled to "already consolidated"
        # — without this, the first post-upgrade sleep would blast a huge
        # batch over the pre-E3 backlog.
        rows = _consolidated_rows(temp_db, "s1")
        assert len(rows) == 1
        assert rows[0][0] == "legacy-1"
        assert rows[0][1] is not None, (
            "pre-E3 row was not backfilled; the first sleep cycle would "
            "incorrectly re-consolidate everything"
        )
        # Backfilled value must be a parseable ISO timestamp.
        datetime.fromisoformat(rows[0][1])

        # And sleep should NOT re-process the backfilled row.
        result = beam.sleep(dry_run=False)
        assert result["status"] == "no_op", (
            f"Pre-E3 backfilled row was incorrectly re-consolidated: {result}"
        )

    def test_consolidated_at_is_iso_timestamp(self, temp_db):
        """The marker value must be a parseable ISO-8601 timestamp.
        Downstream consumers compute age from consolidated_at; a sentinel
        like '1' or '' would silently break them."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=2)

        beam.sleep(dry_run=False)

        rows = _consolidated_rows(temp_db, "s1")
        # The writers stamp UTC with the tzinfo stripped
        # (datetime.now(timezone.utc).replace(tzinfo=None).isoformat()), so the
        # comparison has to be UTC too. datetime.now() is local wall clock: on any
        # host that is not UTC the two differ by exactly the offset, and the
        # assertion reads that as a corrupt marker rather than a clock mismatch.
        # This is the test half of #1094; the polyphonic reader was the production half.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for row_id, consolidated_at in rows:
            parsed = datetime.fromisoformat(consolidated_at)
            # Within 60s of NOW (generous for CI clock skew).
            delta = abs((now - parsed).total_seconds())
            assert delta < 60, (
                f"consolidated_at for {row_id} is {consolidated_at}, "
                f"{delta:.1f}s from NOW — outside the expected window"
            )

    def test_consolidated_at_idempotent_on_modern_db(self, temp_db):
        """A second BeamMemory init on the same DB must not crash and
        must NOT clobber existing consolidated_at values."""
        beam1 = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        beam1.sleep(dry_run=False)

        rows_first = _consolidated_rows(temp_db, "s1")
        first_marker = rows_first[0][1]
        assert first_marker is not None

        # Re-init.
        BeamMemory(session_id="s1", db_path=temp_db)

        rows_second = _consolidated_rows(temp_db, "s1")
        assert rows_second[0][1] == first_marker, (
            "second init clobbered the existing consolidated_at value"
        )

        # PRAGMA shows the column exactly once.
        conn = sqlite3.connect(str(temp_db))
        cols = [r[1] for r in conn.execute("PRAGMA table_info(working_memory)").fetchall()]
        conn.close()
        assert cols.count("consolidated_at") == 1

    def test_dry_run_writes_nothing(self, temp_db):
        """Dry run must not mutate working_memory, episodic_memory, or
        consolidation_log. Pre-E3 only DELETE was suppressed; post-E3
        the UPDATE marker is a second mutation that must also be skipped."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=2)

        result = beam.sleep(dry_run=True)
        assert result["status"] == "dry_run"

        conn = sqlite3.connect(str(temp_db))
        wm_count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        ep_count = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        log_count = conn.execute("SELECT COUNT(*) FROM consolidation_log").fetchone()[0]
        marked = conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE consolidated_at IS NOT NULL"
        ).fetchone()[0]
        conn.close()

        assert wm_count == 2, "dry_run deleted working_memory rows"
        assert ep_count == 0, "dry_run wrote an episodic summary"
        assert log_count == 0, "dry_run wrote to consolidation_log"
        assert marked == 0, "dry_run set consolidated_at"

    def test_trim_exempts_consolidated_rows(self, temp_db):
        """[E3 trim policy] _trim_working_memory must NOT delete rows
        whose consolidated_at is set. Without this exemption, the
        'originals stay' contract collapses at the TTL boundary."""
        from mnemosyne.core.beam import WORKING_MEMORY_TTL_HOURS

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Seed a row, consolidate it, then move the row's timestamp
        # past the trim TTL window. The trim runs implicitly on every
        # remember(); explicitly invoke it to be unambiguous.
        _seed_old_wm(temp_db, "s1", n=1)
        beam.sleep(dry_run=False)
        rows_after_sleep = _consolidated_rows(temp_db, "s1")
        assert rows_after_sleep[0][1] is not None

        # Backdate the row past the trim cutoff (= WORKING_MEMORY_TTL_HOURS).
        very_old = (
            datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS + 5)
        ).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE working_memory SET timestamp = ?", (very_old,)
        )
        conn.commit()
        conn.close()

        # Trigger trim by writing a new (fresh) row through beam.remember.
        beam.remember("fresh content not eligible for sleep", source="conversation")

        # The consolidated row must survive.
        count = _wm_count(temp_db, "s1")
        assert count == 2, (
            f"trim deleted a consolidated row; the additive contract is "
            f"broken at the TTL boundary. Expected 2 rows (1 consolidated "
            f"+ 1 fresh), got {count}"
        )

    def test_dedup_remember_clears_consolidated_at(self, temp_db):
        """When the same content is remembered again after sleep, the
        dedup-update path must clear consolidated_at so the refreshed
        occurrence becomes eligible for sleep again."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = "duplicate-content-for-dedup-test"

        # Insert and consolidate.
        mem_id = beam.remember(content, source="conversation")
        conn = sqlite3.connect(str(temp_db))
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn.execute(
            "UPDATE working_memory SET timestamp = ? WHERE id = ?",
            (old_ts, mem_id),
        )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)
        rows = _consolidated_rows(temp_db, "s1")
        assert rows[0][1] is not None, "sleep didn't mark the row"

        # Re-remember the same content.
        beam.remember(content, source="conversation")

        rows = _consolidated_rows(temp_db, "s1")
        assert rows[0][1] is None, (
            "dedup-update path did not clear consolidated_at; the "
            "refreshed occurrence is permanently skipped"
        )

    def test_concurrent_sleep_atomic_claim(self, temp_db):
        """Two BeamMemory instances calling sleep() on the same DB must
        not both summarize the same originals. The atomic claim — UPDATE
        WHERE consolidated_at IS NULL with rowcount check — gates this."""
        beam_a = BeamMemory(session_id="s1", db_path=temp_db)
        beam_b = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=3)

        result_a = beam_a.sleep(dry_run=False)
        result_b = beam_b.sleep(dry_run=False)

        # Exactly one summary should exist; the second sleep should be
        # a no_op (all rows claimed) or skip-already-consolidated.
        conn = sqlite3.connect(str(temp_db))
        ep_count = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        conn.close()
        assert ep_count == 1, (
            f"concurrent sleeps produced {ep_count} summaries; the atomic "
            f"claim failed. result_a={result_a}, result_b={result_b}"
        )

    def test_sleep_all_sessions_reclaims_stale_orphan_claims_first(self, temp_db):
        """A crash between sleep()'s claim and its episodic insert leaves rows
        with consolidated_at set and no summary. The maintenance pass must
        reclaim them so they are summarized instead of skipped forever."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=2)
        stale_claim = _utc_naive_iso(hours_ago=2)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = ?",
            (stale_claim, stale_claim),
        )
        conn.commit()
        conn.close()

        result = beam.sleep_all_sessions(dry_run=False)

        assert result["reclaimed_orphans"] == 2
        assert result["items_consolidated"] == 2
        conn = sqlite3.connect(str(temp_db))
        try:
            summary_of = conn.execute(
                "SELECT summary_of FROM episodic_memory"
            ).fetchall()
        finally:
            conn.close()
        covered = {i for (s,) in summary_of for i in (s or "").split(",") if i}
        assert {"e3-s1-0", "e3-s1-1"} <= covered

    def test_sleep_all_sessions_dry_run_does_not_reclaim(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        stale_claim = _utc_naive_iso(hours_ago=2)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = ?",
            (stale_claim, stale_claim),
        )
        conn.commit()
        conn.close()

        result = beam.sleep_all_sessions(dry_run=True)

        assert _consolidated_rows(temp_db, "s1")[0][1] == stale_claim
        conn = sqlite3.connect(str(temp_db))
        try:
            claim = conn.execute("SELECT consolidation_claimed_at FROM working_memory").fetchone()[0]
        finally:
            conn.close()
        assert claim == stale_claim, "dry run cleared the claim marker"
        assert result["reclaimed_orphans"] == 0

    def test_reclaim_orphans_clears_stale_claim_without_summary(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        stale_claim = _utc_naive_iso(hours_ago=2)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = ? WHERE id = ?",
            (stale_claim, stale_claim, "e3-s1-0"),
        )
        conn.commit()
        conn.close()

        result = beam.reclaim_orphans(stale_after_seconds=1)

        assert result["status"] == "reclaimed"
        assert result["reclaimed"] == 1
        assert result["candidate_ids"] == ["e3-s1-0"]
        assert _consolidated_rows(temp_db, "s1")[0][1] is None

    def test_reclaim_orphans_preserves_referenced_claims_and_avoids_partial_matches(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        stale_claim = _utc_naive_iso(hours_ago=2)
        conn = sqlite3.connect(str(temp_db))
        conn.executemany(
            """
            INSERT INTO working_memory (id, content, source, timestamp, session_id, consolidated_at, consolidation_claimed_at)
            VALUES (?, ?, 'conversation', ?, 's1', ?, ?)
            """,
            [
                ("wm1", "orphan", old_ts, stale_claim, stale_claim),
                ("wm10", "referenced", old_ts, stale_claim, stale_claim),
            ],
        )
        conn.execute(
            """
            INSERT INTO episodic_memory (id, content, source, timestamp, session_id, summary_of)
            VALUES ('ep1', 'summary', 'sleep_consolidation', ?, 's1', 'wm10')
            """,
            (datetime.now().isoformat(),),
        )
        conn.commit()
        conn.close()

        result = beam.reclaim_orphans(stale_after_seconds=1)

        assert result["reclaimed"] == 1
        rows = dict(_consolidated_rows(temp_db, "s1"))
        assert rows["wm1"] is None
        assert rows["wm10"] == stale_claim

    def test_reclaim_orphans_respects_stale_threshold(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        stale_claim = _utc_naive_iso(hours_ago=2)
        fresh_claim = _utc_naive_iso(hours_ago=0)
        conn = sqlite3.connect(str(temp_db))
        conn.executemany(
            """
            INSERT INTO working_memory (id, content, source, timestamp, session_id, consolidated_at, consolidation_claimed_at)
            VALUES (?, ?, 'conversation', ?, 's1', ?, ?)
            """,
            [
                ("stale", "stale orphan", old_ts, stale_claim, stale_claim),
                ("fresh", "fresh claim", old_ts, fresh_claim, fresh_claim),
            ],
        )
        conn.commit()
        conn.close()

        result = beam.reclaim_orphans(stale_after_seconds=60)

        assert result["reclaimed"] == 1
        rows = dict(_consolidated_rows(temp_db, "s1"))
        assert rows["stale"] is None
        assert rows["fresh"] == fresh_claim

    def test_reclaim_orphans_dry_run_writes_nothing(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        stale_claim = _utc_naive_iso(hours_ago=2)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = ? WHERE id = ?",
            (stale_claim, stale_claim, "e3-s1-0"),
        )
        conn.commit()
        conn.close()

        result = beam.reclaim_orphans(dry_run=True, stale_after_seconds=1)

        assert result["status"] == "dry_run"
        assert result["candidates"] == 1
        assert result["reclaimed"] == 0
        assert _consolidated_rows(temp_db, "s1")[0][1] == stale_claim

    def test_reclaim_orphans_ignores_legacy_backfill_without_claim_marker(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        stale_backfill = _utc_naive_iso(hours_ago=2)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = NULL WHERE id = ?",
            (stale_backfill, "e3-s1-0"),
        )
        conn.commit()
        conn.close()

        result = beam.reclaim_orphans(stale_after_seconds=1)

        assert result["status"] == "no_op"
        assert result["reclaimed"] == 0
        assert _consolidated_rows(temp_db, "s1")[0][1] == stale_backfill

    def test_successful_sleep_clears_claim_marker(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)

        result = beam.sleep(dry_run=False)

        assert result["items_consolidated"] == 1
        conn = sqlite3.connect(str(temp_db))
        marker = conn.execute(
            "SELECT consolidation_claimed_at FROM working_memory WHERE id = 'e3-s1-0'"
        ).fetchone()[0]
        conn.close()
        assert marker is None

    def test_sleep_all_sessions_force_consolidates_fresh_rows(self, temp_db, monkeypatch):
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: False
        )
        beam = BeamMemory(session_id="maintenance", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1, ts_offset_hours=0)

        result = beam.sleep_all_sessions(dry_run=False, force=True)

        assert result["sessions_consolidated"] == 1
        assert result["items_consolidated"] == 1
        conn = sqlite3.connect(str(temp_db))
        summary_of = conn.execute(
            "SELECT summary_of FROM episodic_memory WHERE source = 'sleep_consolidation'"
        ).fetchone()[0]
        final = conn.execute(
            "SELECT consolidated_at, consolidation_claimed_at FROM working_memory WHERE id = 'e3-s1-0'"
        ).fetchone()
        conn.close()
        assert summary_of == "e3-s1-0"
        assert final[0] is not None
        assert final[1] is None

    def test_manual_reclaim_then_sleep_all_sessions_consolidates_orphan(self, temp_db, monkeypatch):
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: False
        )
        beam = BeamMemory(session_id="maintenance", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        stale_claim = _utc_naive_iso(hours_ago=2)
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE working_memory SET consolidated_at = ?, consolidation_claimed_at = ? WHERE id = ?",
            (stale_claim, stale_claim, "e3-s1-0"),
        )
        conn.commit()
        conn.close()

        reclaim = beam.reclaim_orphans(stale_after_seconds=1)
        result = beam.sleep_all_sessions(dry_run=False, force=True)

        assert reclaim["reclaimed"] == 1
        assert result["items_consolidated"] == 1
        conn = sqlite3.connect(str(temp_db))
        summary_of = conn.execute(
            "SELECT summary_of FROM episodic_memory WHERE source = 'sleep_consolidation'"
        ).fetchone()[0]
        final = conn.execute(
            "SELECT consolidated_at, consolidation_claimed_at FROM working_memory WHERE id = 'e3-s1-0'"
        ).fetchone()
        conn.close()
        assert summary_of == "e3-s1-0"
        assert final[0] is not None
        assert final[1] is None

    # --- review findings on automatic orphan recovery (#1322) -----------------

    @staticmethod
    def _claim(db_path, ids, stamp, summarized=False):
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executemany(
                "UPDATE working_memory SET consolidated_at = ?, "
                "consolidation_claimed_at = ? WHERE id = ?",
                [(stamp, stamp, i) for i in ids],
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _summaries(db_path):
        conn = sqlite3.connect(str(db_path))
        try:
            return [r[0] for r in conn.execute(
                "SELECT summary_of FROM episodic_memory WHERE source = 'sleep_consolidation'")]
        finally:
            conn.close()

    def test_recovery_never_exposes_the_row_to_trim(self, temp_db, monkeypatch):
        """Reclaim used to NULL consolidated_at, ending the trim exemption: a
        remember() landing right after the reclaim commit deleted the 200h-old
        original before any summary existed. A trim run just before each
        session's sleep stands in for that remember()."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = _seed_old_wm(temp_db, "s1", n=1)
        self._claim(temp_db, ids, _utc_naive_iso(hours_ago=2))
        real_sleep = BeamMemory.sleep

        def trim_then_sleep(self_, *a, **k):
            self_._trim_working_memory()
            return real_sleep(self_, *a, **k)

        monkeypatch.setattr(BeamMemory, "sleep", trim_then_sleep)
        beam.sleep_all_sessions(dry_run=False)
        assert _wm_count(temp_db, "s1") == 1, "trim deleted the original during recovery"
        assert self._summaries(temp_db) == [ids[0]]

    def test_recovery_beyond_one_sleep_batch_protects_every_row(self, temp_db, monkeypatch):
        """With a batch of 2, 3 recovered rows used to leave 1 unprotected and
        unsummarized for the next remember() to trim."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        monkeypatch.setattr("mnemosyne.core.beam.SLEEP_BATCH_SIZE", 2)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = _seed_old_wm(temp_db, "s1", n=3)
        self._claim(temp_db, ids, _utc_naive_iso(hours_ago=2))
        beam.sleep_all_sessions(dry_run=False)
        beam._trim_working_memory()  # what the next remember() would run
        assert _wm_count(temp_db, "s1") == 3, "an unsummarized recovered row was trimmed"
        assert all(r[1] is not None for r in _consolidated_rows(temp_db, "s1"))

    def test_paused_holder_cannot_write_a_second_summary(self, temp_db, monkeypatch):
        """A pauses after claiming, B recovers and summarizes, A resumes: A must
        write nothing and must not clear B's claim marker."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        a = BeamMemory(session_id="s1", db_path=temp_db)
        b = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        real = BeamMemory.consolidate_to_episodic
        state = {"paused": False}

        def a_pauses_then_b_runs(self_, *args, **kw):
            if self_ is a and not state["paused"]:
                state["paused"] = True
                # while A is "paused", its claim ages past the stale window
                old = _utc_naive_iso(hours_ago=2)
                c = sqlite3.connect(str(temp_db))
                c.execute("UPDATE working_memory SET consolidated_at=?, consolidation_claimed_at=?", (old, old))
                c.commit(); c.close()
                b.sleep_all_sessions(dry_run=False)  # B recovers + summarizes
            return real(self_, *args, **kw)

        monkeypatch.setattr(BeamMemory, "consolidate_to_episodic", a_pauses_then_b_runs)
        res = a.sleep(dry_run=False)
        assert self._summaries(temp_db) == ["e3-s1-0"], "two summaries for one original"
        assert res["claims_lost"] == 1 and res["summaries_created"] == 0
        assert _consolidated_rows(temp_db, "s1")[0][1] is not None
        c = sqlite3.connect(str(temp_db))
        claim = c.execute("SELECT consolidation_claimed_at FROM working_memory").fetchone()[0]
        c.close()
        assert claim is None, "B's finished row should carry no claim marker"

    def test_stale_holder_cannot_clear_a_newer_claim_marker(self, temp_db, monkeypatch):
        """A finishes its summary, but B re-claimed the row first: A's claim-clear
        must not erase B's marker."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        newer = "2999-01-01T00:00:00"
        real = BeamMemory.consolidate_to_episodic

        def write_then_get_reclaimed(self_, *a, **k):
            out = real(self_, *a, **k)
            c = sqlite3.connect(str(temp_db))
            c.execute("UPDATE working_memory SET consolidation_claimed_at = ?", (newer,))
            c.commit(); c.close()
            return out

        monkeypatch.setattr(BeamMemory, "consolidate_to_episodic", write_then_get_reclaimed)
        beam.sleep(dry_run=False)
        c = sqlite3.connect(str(temp_db))
        claim = c.execute("SELECT consolidation_claimed_at FROM working_memory").fetchone()[0]
        c.close()
        assert claim == newer, "the previous holder cleared a newer claim marker"

    def test_large_orphan_backlog_is_not_blocked_by_the_bind_limit(self, temp_db, monkeypatch):
        """999 stale ids + the cutoff exceeded a 999-variable limit: the pass
        returned no_op / 0 / 0 errors and left every claim untouched."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = _seed_old_wm(temp_db, "s1", n=1000)
        self._claim(temp_db, ids, _utc_naive_iso(hours_ago=2))
        beam.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        res = beam.reclaim_orphans(stale_after_seconds=1, limit=1000)
        assert res["reclaimed"] == 1000 and res["candidates"] == 1000
        remaining = sqlite3.connect(str(temp_db)).execute(
            "SELECT COUNT(*) FROM working_memory WHERE consolidated_at IS NOT NULL").fetchone()[0]
        assert remaining == 0

    def test_sleep_recovers_a_large_backlog_under_the_bind_limit(self, temp_db, monkeypatch):
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = _seed_old_wm(temp_db, "s1", n=1000)
        self._claim(temp_db, ids, _utc_naive_iso(hours_ago=2))
        beam.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        res = beam.sleep_all_sessions(dry_run=False)
        assert res["reclaimed_orphans"] == 1000 and res["items_consolidated"] == 1000
        assert res["errors"] == 0

    def test_historical_summary_does_not_block_recovery_of_the_latest_claim(self, temp_db, monkeypatch):
        """sleep -> remember() refresh (clears markers) -> second sleep claims and
        crashes: the first summary names the same id but did not complete this claim."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = _seed_old_wm(temp_db, "s1", n=1)
        beam.sleep(dry_run=False)
        assert self._summaries(temp_db) == [ids[0]]
        # refresh: markers cleared, row eligible again; summary row predates the new claim
        new_claim = (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2))
        c = sqlite3.connect(str(temp_db))
        c.execute("UPDATE episodic_memory SET created_at = ?", ((new_claim - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),))
        c.execute("UPDATE working_memory SET consolidated_at=?, consolidation_claimed_at=?", (new_claim.isoformat(),) * 2)
        c.commit(); c.close()
        res = beam.reclaim_orphans(stale_after_seconds=1, dry_run=True)
        assert res["candidates"] == 1, "an earlier claim's summary blocked recovery"
        out = beam.sleep_all_sessions(dry_run=False)
        assert out["reclaimed_orphans"] == 1 and out["items_consolidated"] == 1
        assert len(self._summaries(temp_db)) == 2

    def test_completed_claim_is_not_recovered(self, temp_db):
        """A summary written after the claim completes it: never re-summarize."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        beam.sleep(dry_run=False)
        stamp = _utc_naive_iso(hours_ago=2)
        c = sqlite3.connect(str(temp_db))
        c.execute("UPDATE working_memory SET consolidated_at=?, consolidation_claimed_at=?", (stamp, stamp))
        c.commit(); c.close()
        assert beam.reclaim_orphans(stale_after_seconds=1, dry_run=True)["candidates"] == 0

    def test_reclamation_failure_is_reported_not_hidden(self, temp_db, monkeypatch):
        """A recovery failure (e.g. 'database is locked') used to be logged only:
        the pass returned errors=0. The same failure injected into both the old
        reclaim call and the new in-sleep claim step must surface as an error."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        ids = _seed_old_wm(temp_db, "s1", n=1)
        self._claim(temp_db, ids, _utc_naive_iso(hours_ago=2))

        def busy(self_, *a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(BeamMemory, "reclaim_orphans", busy)
        monkeypatch.setattr(BeamMemory, "_claim_sleep_rows", busy, raising=False)
        res = beam.sleep_all_sessions(dry_run=False)
        assert res["errors"] == 1, res
        assert res["error_details"][0]["session_id"] == "s1"

    def test_export_import_preserves_consolidated_at(self, temp_db):
        """Backup round-trip must preserve consolidated_at so the
        importing DB doesn't re-summarize already-slept rows on next
        sleep."""
        import tempfile as _tempfile

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=2)
        beam.sleep(dry_run=False)

        original_rows = _consolidated_rows(temp_db, "s1")
        original_markers = {r[0]: r[1] for r in original_rows}

        # Export then re-import into a fresh DB.
        export = beam.export_to_dict()
        with _tempfile.TemporaryDirectory() as td:
            dest_path = Path(td) / "restored.db"
            beam_dest = BeamMemory(session_id="s1", db_path=dest_path)
            beam_dest.import_from_dict(export)

            restored_rows = _consolidated_rows(dest_path, "s1")
            assert len(restored_rows) == 2
            for row_id, consolidated_at in restored_rows:
                assert consolidated_at == original_markers[row_id], (
                    f"export/import lost consolidated_at for {row_id}: "
                    f"original={original_markers[row_id]}, "
                    f"restored={consolidated_at}"
                )

            # And sleep on the restored DB must be a no-op.
            result = beam_dest.sleep(dry_run=False)
            assert result["status"] == "no_op"
