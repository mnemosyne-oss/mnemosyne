"""The actual retention operation must use one normalized chronological key."""
from datetime import datetime, timezone

import pytest

import mnemosyne.core.beam as bm


@pytest.mark.parametrize("newest_stamp", [
    " 2026-04-11T11:00:00Z ",
    " 2026-04-11T13:00:00+02:00 ",
    " 2026-04-11T06:00:00-05:00 ",
    " 2026-04-11 11:00:00 ",
])
def test_trim_keeps_padded_newest_and_preserves_exempt_rows(tmp_path, monkeypatch, newest_stamp):
    beam = bm.BeamMemory(db_path=tmp_path / "trim.db", session_id="retention")

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 4, 11, 12, tzinfo=timezone.utc)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    monkeypatch.setattr(bm, "datetime", FrozenDatetime)
    monkeypatch.setattr(bm, "WORKING_MEMORY_TTL_HOURS", 24)
    monkeypatch.setattr(bm, "WORKING_MEMORY_MAX_ITEMS", 1)
    rows = [
        ("older", "2026-04-11T10:00:00Z", "retention", 0, None),
        ("newest", newest_stamp, "retention", 0, None),
        ("invalid", "not-a-timestamp", "retention", 0, None),
        ("expired", " 2026-04-09T11:00:00Z ", "retention", 0, None),
        ("pinned", "2020-01-01T00:00:00Z", "retention", 1, None),
        ("consolidated", "2020-01-01T00:00:00Z", "retention", 0, "2026-04-10"),
        ("other-session", "2020-01-01T00:00:00Z", "other", 0, None),
    ]
    try:
        beam.conn.executemany(
            "INSERT INTO working_memory(id, content, timestamp, session_id, pinned, consolidated_at) "
            "VALUES (?, 'Retention witness', ?, ?, ?, ?)", rows,
        )
        beam.conn.commit()
        assert beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == len(rows)
        beam._trim_working_memory()
        survivors = dict(beam.conn.execute("SELECT id, timestamp FROM working_memory"))
        assert set(survivors) == {"newest", "pinned", "consolidated", "other-session"}
        assert survivors["newest"] == newest_stamp, "Retention must not rewrite stored values"
        # Repeating trim cannot displace the same survivor or its exempt neighbors.
        beam._trim_working_memory()
        assert dict(beam.conn.execute("SELECT id, timestamp FROM working_memory")) == survivors
    finally:
        beam.conn.close()
