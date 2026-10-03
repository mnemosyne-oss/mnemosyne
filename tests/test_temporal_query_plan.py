"""Query-plan and budget evidence for the SQL `_temporal_voice` actually runs.

dplush's review of #1097 was right that this file measured the wrong statement: the plan and
timing tests executed a separate raw `timestamp > ?` query, so they passed while the shipped
path used `julianday()` and lost indexed selection. Every assertion below therefore reads the
production SQL out of the module instead of retyping it — retyping is how the previous
version drifted from the code it claimed to cover.

Local medians over 20 executions, 50k rows, synthetic in-memory (not a production claim):

    julianday(timestamp)  SCAN + USE TEMP B-TREE FOR ORDER BY   ~4.7 ms
    timestamp > ?         SEARCH USING idx_wm_timestamp        ~0.013 ms

The julianday() form is the correct one — #1094 requires mixed-format window membership and
chronology before LIMIT, and TEXT comparison gets both wrong. It costs a full scan because
julianday() is not indexable under idx_wm_timestamp, and no schema migration is authorised.
So the cost is accepted, bounded below, and revisited only if an expression index lands.

Run: pytest tests/test_temporal_query_plan.py -q
"""

import sqlite3
import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("numpy")

from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine  # noqa: E402

ROWS = 50_000
# Accepted cost of instant-correct selection with no expression index. Measured ~4.7 ms
# locally; the ceiling is loose because runner speed varies, but tight enough to catch a
# regression into a per-row re-parse or a lost secondary filter.
ACCEPTED_SCAN_MS = 250.0

SCHEMA = """
CREATE TABLE working_memory (
    id TEXT PRIMARY KEY, content TEXT, source TEXT, timestamp TEXT,
    session_id TEXT, importance REAL, valid_until TEXT, superseded_by TEXT
);
CREATE INDEX idx_wm_timestamp ON working_memory(timestamp);
"""


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = tmp_path_factory.mktemp("plan") / "wm.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
        "VALUES (?, 'x', 'conversation', ?, 's1', 1.0)",
        [(f"m{i}", (now - timedelta(minutes=i)).isoformat()) for i in range(ROWS)],
    )
    conn.commit()
    conn.execute("ANALYZE")
    conn.close()
    return PolyphonicRecallEngine(db_path=path)


def _shipped_sql():
    """The statement `_temporal_voice` runs, lifted from the module source.

    The f-string interpolates ``{echo_clause}`` — an empty string when no ids are excluded —
    so the literal braces are stripped here before the SQL is handed to EXPLAIN. Anything
    else and sqlite3 reads the braces as syntax.
    """
    from mnemosyne.core import polyphonic_recall as mod

    with open(mod.__file__, encoding="utf-8") as fh:
        text = fh.read()
    start = text.index("SELECT id, content, timestamp, importance")
    end = text.index('"""', start)
    sql = " ".join(text[start:end].split())
    return sql.replace("{echo_clause}", "").strip()


def _cutoff():
    return (datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=7)).isoformat()


def test_plan_is_read_from_the_production_statement():
    """Guard against the drift that caused the review finding."""
    sql = _shipped_sql()
    assert "julianday(timestamp) > julianday(?)" in sql, sql
    assert "ORDER BY julianday(timestamp) DESC" in sql, sql
    assert "LIMIT 20" in sql, sql


def test_shipped_query_plan_is_pinned(engine):
    """Pin the plan the shipped query gets today.

    SCAN + temp B-tree is the accepted cost: julianday() is not indexable under
    idx_wm_timestamp. If this ever reports SEARCH, an expression index landed and
    ACCEPTED_SCAN_MS should be revisited in the same change.
    """
    sql = _shipped_sql()
    conn = sqlite3.connect(str(engine.db_path))
    plan = " ".join(row[3] for row in conn.execute(f"EXPLAIN QUERY PLAN {sql}", (_cutoff(),)))
    conn.close()
    assert "idx_wm_timestamp" not in plan, (
        "the shipped query now uses idx_wm_timestamp — an expression index was added, so "
        f"ACCEPTED_SCAN_MS should be revisited. plan={plan}"
    )


def test_shipped_query_stays_within_the_accepted_budget(engine):
    """Median wall clock of the real statement, 20 executions on 50k rows."""
    sql = _shipped_sql()
    conn = sqlite3.connect(str(engine.db_path))
    samples = []
    for _ in range(20):
        start = time.perf_counter()
        conn.execute(sql, (_cutoff(),)).fetchall()
        samples.append((time.perf_counter() - start) * 1000)
    conn.close()
    samples.sort()
    median = samples[len(samples) // 2]
    assert median < ACCEPTED_SCAN_MS, {"median_ms": median, "rows": ROWS}


def test_instant_ordering_holds_on_the_real_reader(tmp_path):
    """The behaviour the scan buys, through `_temporal_voice` itself.

    09:00-05:00 is 14:00 UTC and 12:00+02:00 is 10:00 UTC. As TEXT the second sorts first;
    with julianday() the first does, which is what decides admission before LIMIT 20.

    A small dedicated table, not the 50k fixture: the point is the ordering, and reusing the
    budget fixture would put 50k rows between these two and the top of the result.
    """
    path = tmp_path / "mixed.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    now = datetime.now(timezone.utc)
    conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) "
        "VALUES (?, 'x', 'conversation', ?, 's1', 1.0)",
        [
            ("older_by_instant",
             (now - timedelta(hours=4)).astimezone(timezone(timedelta(hours=2))).isoformat()),
            ("newer_by_instant",
             (now - timedelta(hours=1)).astimezone(timezone(timedelta(hours=-5))).isoformat()),
        ],
    )
    conn.commit()
    conn.close()

    reader = PolyphonicRecallEngine(db_path=path)
    got = [r.memory_id for r in reader._temporal_voice(query="what happened recently")]
    assert got == ["newer_by_instant", "older_by_instant"], got