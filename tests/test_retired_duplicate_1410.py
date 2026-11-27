"""Exact-content deduplication only reuses active working memory."""

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.memory import Mnemosyne


@pytest.mark.parametrize(
    "retirement", ["invalidate", "supersede", "supersede-no-expiry", "expire"]
)
def test_remember_reasserts_retired_content_as_new_active_row(tmp_path, retirement):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    first = memory.remember(content, importance=0.9)
    assert first is not None
    assert any(hit["id"] == first for hit in memory.recall("metric units"))

    replacement = None
    if retirement in ("supersede", "supersede-no-expiry"):
        replacement = memory.remember("Prefers replies in imperial units")
        assert replacement is not None
        assert memory.invalidate(first, replacement_id=replacement)
        # A future expiry or NULL leaves superseded_by as the only exclusion.
        valid_until = None
        if retirement == "supersede":
            valid_until = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        memory.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?", (valid_until, first)
        )
        memory.conn.commit()
    elif retirement == "invalidate":
        assert memory.invalidate(first)
    else:
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        memory.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?", (past, first)
        )
        memory.conn.commit()

    old = memory.conn.execute(
        "SELECT valid_until, superseded_by, timestamp, created_at, metadata_json "
        "FROM working_memory WHERE id = ?", (first,)
    ).fetchone()
    if replacement is not None:
        assert old["valid_until"] == valid_until
        assert old["superseded_by"] == replacement
    second = memory.remember(content, importance=0.9)
    rows = memory.conn.execute(
        "SELECT id, valid_until, superseded_by FROM working_memory "
        "WHERE session_id = ? AND content = ? ORDER BY rowid", ("s1", content)
    ).fetchall()
    assert second != first
    assert [row["id"] for row in rows] == [first, second]
    assert (rows[0]["valid_until"], rows[0]["superseded_by"]) == tuple(old)[:2]
    assert tuple(memory.conn.execute(
        "SELECT valid_until, superseded_by, timestamp, created_at, metadata_json "
        "FROM working_memory WHERE id = ?", (first,)
    ).fetchone()) == tuple(old)
    assert rows[1]["valid_until"] is None
    assert rows[1]["superseded_by"] is None
    assert any(hit["id"] == second for hit in memory.recall("metric units"))
    assert all(hit["id"] != first for hit in memory.recall("metric units"))

    # A further identical write still deduplicates against the active version.
    assert memory.remember(content) == second
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE session_id = ? AND content = ?",
        ("s1", content),
    ).fetchone()[0] == 2


def test_reassertion_evicts_cached_empty_enhanced_recall(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"

    def recall():
        return memory.recall_enhanced(
            content,
            top_k=3,
            use_weibull=False,
            use_mmr=False,
            use_intent=False,
            use_synonyms=False,
        )

    original = memory.remember(content, importance=0.9)
    assert original is not None
    assert memory.invalidate(original)

    assert recall() == []
    cache = memory._query_cache
    hits = cache.stats()["hits"]
    assert recall() == []
    assert cache.stats()["hits"] == hits + 1

    restored = memory.remember(content, importance=0.9)
    assert restored not in {None, original}
    assert [hit["id"] for hit in recall()] == [restored]


@pytest.mark.parametrize("all_sessions", [False, True], ids=["session", "all-sessions"])
@pytest.mark.parametrize("case", ["reasserted", "expired-only", "future-expiry"])
def test_sleep_summarizes_only_active_working_rows(tmp_path, monkeypatch, all_sessions, case):
    """Expired inputs stay out of summaries and do not hide live sources."""
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
    clock = datetime(2030, 6, 15, 12, tzinfo=timezone.utc)

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock.replace(tzinfo=None) if tz is None else clock.astimezone(tz)

    monkeypatch.setattr(beam_module, "datetime", FixedClock)
    memory = Mnemosyne(session_id="s1", db_path=tmp_path / "memory.db")
    active_text = "observatory relay insulation policy"
    retired_text = active_text if case == "reasserted" else "retired cedar packing"
    retired = memory.remember(retired_text, source="notes")
    assert retired is not None and memory.invalidate(retired)

    # Advance the clock so a reassertion can generate a distinct ID.
    clock += timedelta(seconds=1)
    active_ids = []
    expiry = None
    if case != "expired-only":
        active_ids.append(memory.remember(active_text, source="notes"))
    if case == "future-expiry":
        expiry = clock + timedelta(hours=1)
        offset_expiry = expiry.astimezone(timezone(timedelta(hours=-7))).isoformat()
        active_ids.append(memory.remember(
            "copper relay housing inspection", source="notes", valid_until=offset_expiry
        ))
        # An expired oldest row must not consume a slot before the LIMIT.
        monkeypatch.setattr(beam_module, "SLEEP_BATCH_SIZE", len(active_ids))
    assert None not in active_ids and retired not in active_ids

    backlog = memory.beam._count_unconsolidated_before("9999-12-31 00:00:00")
    if all_sessions:
        result = memory.beam.sleep_all_sessions(force=True)
    else:
        result = memory.sleep(force=True)

    summaries = memory.beam.conn.execute(
        "SELECT id, summary_of, valid_until, content FROM episodic_memory "
        "WHERE source = 'sleep_consolidation'"
    ).fetchall()
    assert backlog == len(active_ids)
    assert result.get("items_consolidated", 0) == len(active_ids)
    if all_sessions:
        assert result["sessions_scanned"] == int(bool(active_ids))
    retired_after = memory.beam.conn.execute(
        "SELECT valid_until, consolidated_at FROM working_memory WHERE id = ?", (retired,)
    ).fetchone()
    assert retired_after is not None and retired_after["consolidated_at"] is None
    assert datetime.fromisoformat(retired_after["valid_until"]) < clock

    if not active_ids:
        assert result["status"] == "no_op"
        assert summaries == []
        assert memory.recall(retired_text) == []
        return

    (summary,) = summaries
    assert set(summary["summary_of"].split(",")) == set(active_ids)
    assert "retired cedar" not in summary["content"]
    if expiry is None:
        assert summary["valid_until"] is None
    else:
        assert datetime.fromisoformat(summary["valid_until"]) == expiry
    recalled_ids = {hit["id"] for hit in memory.recall(active_text, top_k=10)}
    assert active_ids[0] in recalled_ids
    assert retired not in recalled_ids
    summarized_hits = memory.recall(active_text, top_k=10, source="sleep_consolidation")
    assert summary["id"] in {hit["id"] for hit in summarized_hits}


def test_duplicate_control_and_opt_out(tmp_path):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    first = memory.remember(content)
    assert memory.remember(content, memory_id="caller-provided") == first
    second = memory.remember(content, dedupe=False)
    assert second not in {None, first}
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE session_id = ? AND content = ?",
        ("s1", content),
    ).fetchone()[0] == 2


def test_caller_provided_id_used_for_reassertion(tmp_path):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    old_id = memory.remember(content)
    assert old_id is not None
    assert memory.invalidate(old_id)
    assert memory.remember(content, memory_id="new-external-id") == "new-external-id"
    assert memory.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", ("new-external-id",)
    ).fetchone()[0] is None


@pytest.mark.parametrize("superseded", [False, True], ids=["expired", "superseded"])
def test_reassertion_reports_occupied_explicit_id(tmp_path, superseded):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    retired = memory.remember(content, memory_id="caller-supplied")
    assert retired == "caller-supplied"
    replacement = memory.remember("Prefers replies in imperial units") if superseded else None
    if superseded:
        assert replacement is not None
    assert memory.invalidate(retired, replacement_id=replacement)
    read_retired = "SELECT * FROM working_memory WHERE id = ?"
    before = tuple(memory.conn.execute(read_retired, (retired,)).fetchone())

    with pytest.raises(beam_module.MemoryIdConflictError):
        memory.remember(content, memory_id=retired)
    assert tuple(memory.conn.execute(read_retired, (retired,)).fetchone()) == before
    assert memory.remember(content, memory_id="unused-id") == "unused-id"


def test_future_expiration_remains_eligible_for_deduplication(tmp_path):
    memory = BeamMemory(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    original = memory.remember(content, valid_until=future)
    assert original is not None
    assert memory.remember(content) == original
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?", (content,)
    ).fetchone()[0] == 1


@pytest.mark.parametrize(
    ("expiry_delta_ms", "offset_hours", "reasserts"),
    [
        (-1, 0, True),
        (0, 0, True),
        (1, 0, False),
        (-1, 7, True),
        (0, 7, True),
        (1, -5, False),
    ],
)
def test_exact_dedup_expiry_boundary(tmp_path, monkeypatch, expiry_delta_ms, offset_hours, reasserts):
    memory = Mnemosyne(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    original = memory.remember(content)
    assert original is not None

    lookup_time = datetime(2026, 1, 20, 12, tzinfo=timezone.utc)

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                return lookup_time.replace(tzinfo=None)
            return lookup_time.astimezone(tz)

    expiry = (lookup_time + timedelta(milliseconds=expiry_delta_ms)).astimezone(
        timezone(timedelta(hours=offset_hours))
    ).isoformat()
    memory.beam.conn.execute(
        "UPDATE working_memory SET valid_until = ? WHERE id = ?", (expiry, original)
    )
    memory.beam.conn.commit()

    monkeypatch.setattr(beam_module, "datetime", FixedDatetime)
    result = memory.remember(content)

    assert result is not None
    assert (result != original) is reasserts


def test_wrapper_reassertion_writes_new_legacy_id(tmp_path):
    memory = Mnemosyne(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    original = memory.remember(content)
    assert original is not None
    assert memory.invalidate(original)
    restored = memory.remember(content)
    assert restored not in {None, original}
    assert memory.conn.execute(
        "SELECT id FROM memories WHERE id = ?", (restored,)
    ).fetchone()[0] == restored
    assert memory.beam.conn.execute(
        "SELECT id FROM working_memory WHERE id = ? AND superseded_by IS NULL",
        (restored,),
    ).fetchone()[0] == restored


def test_wrapper_reassertion_with_sqlite_clock_before_invalidation(tmp_path):
    memory = Mnemosyne(session_id="s1", db_path=tmp_path / "memory.db")
    content = "Prefers replies in metric units"
    # SQLite's 'now' can be 1 ms behind Python's invalidation timestamp.
    # Override only that SQLite clock on the real wrapper connection; use
    # SQLite itself for ISO parsing and Julian-day conversion in both cases.
    earlier = None
    with sqlite3.connect(":memory:") as clock:
        def sqlite_julianday(value):
            return clock.execute("SELECT julianday(?)", (
                earlier if value == "now" and earlier is not None else value,
            )).fetchone()[0]

        # Register before first remember so its cached dedup query uses this
        # function; only the simulated SQLite 'now' changes after invalidate.
        memory.beam.conn.create_function("julianday", 1, sqlite_julianday)
        original = memory.remember(content)
        assert original is not None
        assert memory.invalidate(original)
        retired = memory.beam.conn.execute(
            "SELECT valid_until, superseded_by, timestamp, created_at, metadata_json "
            "FROM working_memory WHERE id = ?", (original,)
        ).fetchone()
        assert retired["superseded_by"] is None
        assert retired["valid_until"] is not None
        earlier = (datetime.fromisoformat(retired["valid_until"]) -
                   timedelta(milliseconds=1)).isoformat()
        assert clock.execute(
            "SELECT julianday(?) > julianday(?)", (retired["valid_until"], earlier)
        ).fetchone()[0] == 1
        restored = memory.remember(content)

    assert restored not in {None, original}
    assert tuple(memory.beam.conn.execute(
        "SELECT valid_until, superseded_by, timestamp, created_at, metadata_json "
        "FROM working_memory WHERE id = ?", (original,)
    ).fetchone()) == tuple(retired)
    assert memory.beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE session_id = ? AND content = ?",
        ("s1", content),
    ).fetchone()[0] == 2
    assert memory.conn.execute(
        "SELECT id FROM memories WHERE id = ?", (restored,)
    ).fetchone()[0] == restored
