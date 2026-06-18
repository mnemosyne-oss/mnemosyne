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
from datetime import datetime, timedelta
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
        now = datetime.now()
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

    def test_reclaim_orphans_clears_stale_claim_without_summary(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        _seed_old_wm(temp_db, "s1", n=1)
        stale_claim = (datetime.now() - timedelta(hours=2)).isoformat()
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
        stale_claim = (datetime.now() - timedelta(hours=2)).isoformat()
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
        stale_claim = (datetime.now() - timedelta(hours=2)).isoformat()
        fresh_claim = datetime.now().isoformat()
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
        stale_claim = (datetime.now() - timedelta(hours=2)).isoformat()
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
        stale_backfill = (datetime.now() - timedelta(hours=2)).isoformat()
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
        stale_claim = (datetime.now() - timedelta(hours=2)).isoformat()
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
