"""Regression tests for consolidation write-lock hygiene in sleep().

sleep() processes eligible working_memory rows in per-source groups. After
each group it clears the group's consolidation_claimed_at markers with an
UPDATE. Pre-fix, that UPDATE was never committed, so it opened an implicit
write transaction that stayed open across the NEXT group's LLM summarization
calls (remote API: seconds to minutes per chunk; the local GGUF fallback has
no timeout at all). While that transaction was open, every other connection
failed with "database is locked" after busy_timeout, and the WAL could not
checkpoint. Observed in production as a multi-hour wedge with a frozen WAL.

The fix commits immediately after the claim-clear UPDATE, so no write
transaction is open on the beam connection while an LLM call runs.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core import local_llm
from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _seed_old_wm(db_path, session_id, source, n, start=0):
    """Insert n old working_memory rows for the given session/source."""
    conn = sqlite3.connect(str(db_path))
    ts = (datetime.now() - timedelta(hours=200)).isoformat()
    rows = [
        (
            f"lh-{source}-{i}",
            f"lock-hygiene content {source} {i}",
            source,
            ts,
            session_id,
        )
        for i in range(start, start + n)
    ]
    conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


class TestSleepLockHygiene:

    def test_no_open_write_txn_during_llm_summarization(self, temp_db, monkeypatch):
        """Each group's LLM summarize call must run with NO open transaction
        on the beam connection. Pre-fix, group N's claim-clear UPDATE was
        left uncommitted, so group N+1's summarize call ran inside an open
        write transaction that held the SQLite write lock for the whole
        (potentially unbounded) LLM call."""
        beam = BeamMemory(session_id="lock-hygiene", db_path=temp_db)
        # Skip the model-refresh proposal LLM path; this test targets the
        # summarization path only.
        beam.agent_context = "cron"

        # Two sources -> two consolidation groups -> at least two
        # summarize calls, so the second observes the state the first
        # group's claim-clear left behind.
        _seed_old_wm(temp_db, "lock-hygiene", "conversation", n=2)
        _seed_old_wm(temp_db, "lock-hygiene", "notes", n=2)

        txn_open_during_summarize = []

        monkeypatch.setattr(local_llm, "llm_available", lambda: True)
        monkeypatch.setattr(
            local_llm,
            "chunk_memories_by_budget",
            lambda lines, source=None: [lines],
        )

        def fake_summarize(lines, source=None):
            txn_open_during_summarize.append(beam.conn.in_transaction)
            return f"summary of {len(lines)} items"

        monkeypatch.setattr(local_llm, "summarize_memories", fake_summarize)

        result = beam.sleep()

        assert result["status"] == "consolidated"
        assert result["summaries_created"] == 2
        assert len(txn_open_during_summarize) == 2
        assert txn_open_during_summarize == [False, False], (
            "sleep() held an open write transaction on the beam connection "
            "while an LLM summarization call ran; the claim-clear UPDATE "
            f"was not committed. States: {txn_open_during_summarize}"
        )

    def test_sleep_leaves_no_open_transaction(self, temp_db, monkeypatch):
        """After sleep() returns, the beam connection has no open
        transaction, so other connections can write and the WAL can
        checkpoint."""
        beam = BeamMemory(session_id="lock-hygiene", db_path=temp_db)
        beam.agent_context = "cron"
        _seed_old_wm(temp_db, "lock-hygiene", "conversation", n=3)

        monkeypatch.setattr(local_llm, "llm_available", lambda: False)

        result = beam.sleep()

        assert result["status"] == "consolidated"
        assert beam.conn.in_transaction is False

        # A second connection must be able to write immediately.
        other = sqlite3.connect(str(temp_db), timeout=1.0)
        try:
            other.execute(
                "UPDATE working_memory SET last_recalled = ? WHERE session_id = ?",
                (datetime.now().isoformat(), "lock-hygiene"),
            )
            other.commit()
        finally:
            other.close()

    def test_embed_failure_does_not_drop_summary(self, temp_db, monkeypatch):
        """The pre-INSERT embed() call must not abort the episodic insert.
        embed() can raise on the local path (model load failures re-raise
        as RuntimeError); the summary row is the payload and must land,
        just without a vector."""
        from mnemosyne.core import embeddings as _emb

        beam = BeamMemory(session_id="lock-hygiene", db_path=temp_db)

        monkeypatch.setattr(_emb, "available", lambda: True)

        def raising_embed(texts):
            raise RuntimeError("Failed to load embedding model")

        monkeypatch.setattr(_emb, "embed", raising_embed)

        beam.consolidate_to_episodic(
            summary="summary that must survive an embed failure",
            source_wm_ids=["lh-x-0"],
            source="sleep_consolidation",
        )

        row = beam.conn.execute(
            "SELECT content FROM episodic_memory WHERE source = ?",
            ("sleep_consolidation",),
        ).fetchone()
        assert row is not None
        assert "must survive" in row["content"]
        assert beam.conn.in_transaction is False


class TestRecallTouchRollback:
    """The recall touch in get_context() runs an explicit transaction.
    Pre-fix, a failure inside it (e.g. "database is locked" while a
    consolidation pass is writing) propagated with the transaction still
    open on the long-lived thread-local connection, so every later write
    on that thread failed "database is locked" instantly until something
    reset it. Observed in production as a self-healing storm of instant
    tool failures minutes after the real writer had finished."""

    def test_touch_failure_leaves_no_open_transaction(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="touch-rollback", db_path=temp_db)
        beam.remember("touch rollback seed", source="test")

        real_commit = beam.conn.commit
        calls = {"n": 0}

        def failing_commit():
            calls["n"] += 1
            if calls["n"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_commit()

        monkeypatch.setattr(beam.conn, "commit", failing_commit)
        with pytest.raises(sqlite3.OperationalError):
            beam.get_context(limit=5)
        monkeypatch.setattr(beam.conn, "commit", real_commit)

        # The failed touch must have been ROLLED BACK, not merely abandoned:
        # the seeded row's recall bookkeeping is unchanged.
        row = beam.conn.execute(
            "SELECT recall_count, last_recalled FROM working_memory "
            "WHERE content = ?",
            ("touch rollback seed",),
        ).fetchone()
        assert row is not None
        assert row["recall_count"] == 0
        assert row["last_recalled"] is None

        # Pre-fix: the connection is still inside the touch transaction here,
        # and this write fails "database is locked" (stale snapshot) or
        # "cannot start a transaction within a transaction".
        assert not beam.conn.in_transaction
        beam.remember("write after failed touch", source="test")
        rows = beam.get_context(limit=5)
        assert any("write after failed touch" in r["content"] for r in rows)
