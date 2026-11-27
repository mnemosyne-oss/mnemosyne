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
