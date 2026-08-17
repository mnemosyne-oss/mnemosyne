"""
Regression tests for `consolidate_fact` concurrent same-SPO race.

Pre-fix: `consolidate_fact` did SELECT-by-SPO then conditional
INSERT or UPDATE without transaction serialization. Two threads
both passing the no-match SELECT and both attempting INSERT raced
on the deterministic PRIMARY KEY — one INSERT succeeded, the
other raised `IntegrityError` which propagated up to (or was
swallowed by) the caller's broad `except: pass`. Result: silent
data loss — the second thread's observation was never recorded.

Bayesian confidence updates were also race-vulnerable. The formula
`new = old + (1 - old) * weight * 0.3` is path-dependent: two
concurrent UPDATEs that both read the same baseline confidence
both compute the same new value, then both write — the second
UPDATE overwrites the first's effect rather than compounding it.

Post-fix: `consolidate_fact` wraps the SELECT-then-INSERT/UPDATE
in `BEGIN IMMEDIATE` so concurrent calls serialize at SQLite's
writer-lock level. Nested-transaction safe (skips BEGIN if the
caller already opened one, e.g., E2's `_deferred_commits`).

These tests pin:
  - N threads consolidating the same SPO produce exactly one row
    with `mention_count == N`.
  - N threads consolidating distinct SPOs produce N rows.
  - The nested-transaction path (caller in `BEGIN`) doesn't crash
    and produces the same end state.
  - No IntegrityError surfaces under contention.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import List

import pytest

from mnemosyne.core.veracity_consolidation import VeracityConsolidator


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Shared DB path; each thread opens its own connection."""
    # Pre-init the schema by constructing once. Each thread then
    # opens its own VeracityConsolidator(db_path=temp_db) which
    # builds its own connection but finds the schema already created.
    db_path = tmp_path / "concurrency.db"
    VeracityConsolidator(db_path=db_path)
    return db_path


def _consolidate_in_thread(
    db_path: Path,
    subject: str,
    predicate: str,
    object: str,
    veracity: str,
    source: str,
    barrier: threading.Barrier,
    exceptions: List[BaseException],
):
    """Thread worker: wait on barrier so all threads attempt the
    consolidate_fact concurrently, then record any unexpected
    exception so the test can fail visibly."""
    try:
        cons = VeracityConsolidator(db_path=db_path)
        # Configure busy_timeout on this connection so BEGIN IMMEDIATE
        # waits for the lock instead of immediately raising
        # OperationalError under contention. Matches what
        # beam._get_connection does at line 187.
        cons.conn.execute("PRAGMA busy_timeout=5000")
        barrier.wait()  # synchronize threads to maximize contention
        cons.consolidate_fact(subject, predicate, object, veracity, source)
    except BaseException as exc:
        exceptions.append(exc)


# ---------------------------------------------------------------------------
# Core contract: concurrent same-SPO produces exactly one row
# ---------------------------------------------------------------------------


def test_two_threads_same_spo_produce_one_row_count_2(temp_db):
    """Pre-fix this test would either crash with IntegrityError on
    one thread (silent data loss when the upstream `except: pass`
    swallowed it) or leave mention_count = 1 (one observation
    lost). Post-fix: serialized via BEGIN IMMEDIATE → exactly one
    row, mention_count == 2."""
    exceptions: List[BaseException] = []
    barrier = threading.Barrier(2)
    t1 = threading.Thread(target=_consolidate_in_thread, args=(
        temp_db, "Alice", "is", "developer", "stated", "src_a",
        barrier, exceptions,
    ))
    t2 = threading.Thread(target=_consolidate_in_thread, args=(
        temp_db, "Alice", "is", "developer", "stated", "src_b",
        barrier, exceptions,
    ))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert not exceptions, (
        f"thread raised under contention: {exceptions}"
    )

    cons = VeracityConsolidator(db_path=temp_db)
    rows = cons.conn.execute(
        "SELECT mention_count FROM consolidated_facts "
        "WHERE subject = 'Alice'"
    ).fetchall()
    assert len(rows) == 1, (
        f"expected exactly 1 row, got {len(rows)} — race produced "
        "duplicate rows"
    )
    assert rows[0]["mention_count"] == 2, (
        f"expected mention_count == 2, got {rows[0]['mention_count']} "
        "— one observation lost to race"
    )


def test_eight_threads_same_spo_produce_one_row_count_8(temp_db):
    """Higher contention: 8 threads all consolidating identical
    SPO. Pre-fix would lose multiple observations to IntegrityError;
    post-fix all 8 are recorded."""
    exceptions: List[BaseException] = []
    barrier = threading.Barrier(8)
    threads = [
        threading.Thread(target=_consolidate_in_thread, args=(
            temp_db, "Carol", "leads", "team", "stated", f"src_{i}",
            barrier, exceptions,
        ))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not exceptions, (
        f"{len(exceptions)} threads raised: {exceptions[:3]}"
    )

    cons = VeracityConsolidator(db_path=temp_db)
    rows = cons.conn.execute(
        "SELECT mention_count, sources_json FROM consolidated_facts "
        "WHERE subject = 'Carol'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["mention_count"] == 8, (
        f"expected mention_count == 8 from 8 threads, got "
        f"{rows[0]['mention_count']} — race lost observations"
    )


def test_eight_threads_distinct_spos_produce_eight_rows(temp_db):
    """Different SPOs from different threads shouldn't block each
    other excessively (BEGIN IMMEDIATE serializes but doesn't drop
    writes). All 8 distinct rows should land."""
    exceptions: List[BaseException] = []
    barrier = threading.Barrier(8)
    threads = [
        threading.Thread(target=_consolidate_in_thread, args=(
            temp_db, f"Person{i}", "is", "engineer", "stated", f"src_{i}",
            barrier, exceptions,
        ))
        for i in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not exceptions

    cons = VeracityConsolidator(db_path=temp_db)
    count = cons.conn.execute(
        "SELECT COUNT(*) FROM consolidated_facts "
        "WHERE subject LIKE 'Person%'"
    ).fetchone()[0]
    assert count == 8, (
        f"expected 8 distinct rows for 8 distinct SPOs, got {count}"
    )


# ---------------------------------------------------------------------------
# Nested-transaction safety
# ---------------------------------------------------------------------------


def test_consolidate_fact_nested_in_outer_transaction(temp_db):
    """When the caller is already in a transaction (e.g., E2's
    `_deferred_commits` wrapping a batch enrichment loop),
    consolidate_fact must NOT try to start its own BEGIN IMMEDIATE
    (which would raise `OperationalError: cannot start a
    transaction within a transaction`). It should detect the
    outer tx via `conn.in_transaction` and skip the BEGIN."""
    cons = VeracityConsolidator(db_path=temp_db)
    # Force an outer transaction.
    cons.conn.execute("BEGIN")
    assert cons.conn.in_transaction

    # This should NOT raise; should use the existing transaction.
    fact = cons.consolidate_fact("Dan", "is", "designer", "stated", "src_x")
    assert fact.subject == "Dan"

    # Commit the outer transaction so the row persists.
    cons.conn.commit()
    assert not cons.conn.in_transaction

    # Verify the row was written.
    rows = cons.conn.execute(
        "SELECT mention_count FROM consolidated_facts "
        "WHERE subject = 'Dan'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["mention_count"] == 1


def test_consolidate_fact_rolls_back_own_transaction_on_error(temp_db):
    """If consolidate_fact opened its own BEGIN IMMEDIATE and the
    body raises, the transaction must be rolled back (no partial
    writes leaking into the DB). We trigger an error by simulating
    a constraint failure mid-call."""
    cons = VeracityConsolidator(db_path=temp_db)
    # Pre-populate with a row that the next consolidate_fact will
    # find via the SPO match, taking the UPDATE branch.
    cons.consolidate_fact("Eve", "is", "scientist", "stated", "src_a")

    # Monkey-patch the bayesian_update to raise — this fires inside
    # the SELECT-then-UPDATE path, after our BEGIN IMMEDIATE.
    def fail(current_confidence, veracity):
        raise RuntimeError("simulated mid-update failure")

    cons.bayesian_update = fail  # type: ignore

    with pytest.raises(RuntimeError, match="simulated"):
        cons.consolidate_fact("Eve", "is", "scientist", "stated", "src_b")

    # Restore.
    del cons.bayesian_update  # type: ignore

    # State should be unchanged from the first call — mention_count
    # still 1, sources still just ["src_a"].
    row = cons.conn.execute(
        "SELECT mention_count, sources_json FROM consolidated_facts "
        "WHERE subject = 'Eve'"
    ).fetchone()
    assert row["mention_count"] == 1, (
        f"failed update leaked into DB: mention_count = "
        f"{row['mention_count']}"
    )


# ---------------------------------------------------------------------------
# Bayesian confidence under contention
# ---------------------------------------------------------------------------


def test_concurrent_updates_compound_confidence_correctly(temp_db):
    """The Bayesian confidence formula is path-dependent — each
    update reads the CURRENT confidence and computes a delta. Pre-
    fix, two concurrent UPDATEs both read the same baseline,
    computed the same new value, and the second overwrote the
    first → only one of the two updates' effects landed. Post-fix:
    serialized via BEGIN IMMEDIATE → both updates compound and
    the final confidence is strictly greater than the
    single-update result."""
    # First call: establish a baseline single-update confidence
    # value for comparison.
    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Frank", "is", "DBA", "stated", "src_seed")
    seed_row = cons.conn.execute(
        "SELECT confidence FROM consolidated_facts WHERE subject = 'Frank'"
    ).fetchone()
    seed_confidence = seed_row["confidence"]

    # Now fire 4 concurrent updates with separate connections.
    exceptions: List[BaseException] = []
    barrier = threading.Barrier(4)
    threads = [
        threading.Thread(target=_consolidate_in_thread, args=(
            temp_db, "Frank", "is", "DBA", "stated", f"src_{i}",
            barrier, exceptions,
        ))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not exceptions

    row = cons.conn.execute(
        "SELECT confidence, mention_count FROM consolidated_facts "
        "WHERE subject = 'Frank'"
    ).fetchone()
    # Each of the 4 threads should have applied the Bayesian update.
    # mention_count is the most direct evidence the serialization
    # held — each update increments by 1.
    assert row["mention_count"] == 5, (
        f"expected mention_count == 5 (seed + 4 concurrent updates), "
        f"got {row['mention_count']} — race lost update(s)"
    )
    # Final confidence must be strictly greater than the seed —
    # at least one of the 4 updates must have compounded the
    # confidence (pre-fix could have lost all 4 to the
    # last-writer-wins race and left confidence == seed).
    assert row["confidence"] > seed_confidence, (
        f"final confidence {row['confidence']} not greater than "
        f"seed {seed_confidence} — concurrent updates didn't compound"
    )


# ---------------------------------------------------------------------------
# /review hardening (commit 2) — 4-source convergence findings
# ---------------------------------------------------------------------------


class TestReviewHardening:
    """Closes the must-fix /review army findings on commit 1:
      1. _record_conflict premature commit (4-source HIGH)
      2. Silent fallthrough on BEGIN IMMEDIATE failure (4-source HIGH)
      3. WAL + busy_timeout not set on VeracityConsolidator's own
         connection (Claude C3)
      4. Test reliability — deterministic race-window widening to
         prove the original race actually triggers without the fix
         (Claude H1 + Codex adversarial MED)
    """

    def test_record_conflict_does_not_commit_when_nested(self, temp_db):
        """`_record_conflict` must accept `commit=False` so the
        atomicity of consolidate_fact's BEGIN IMMEDIATE scope is
        preserved when conflicts are recorded. Pre-fix the
        unconditional commit ended our outer transaction
        mid-method — the fact INSERT became durable before all
        conflicts were recorded."""
        from mnemosyne.core.veracity_consolidation import VeracityConsolidator

        cons = VeracityConsolidator(db_path=temp_db)

        # Manually open a transaction; call _record_conflict with
        # commit=False; verify the transaction is still open.
        cons.conn.execute("BEGIN IMMEDIATE")
        assert cons.conn.in_transaction
        cons._record_conflict("fact_x", "fact_y", "test", commit=False)
        # Tx should still be open — we said commit=False.
        assert cons.conn.in_transaction, (
            "_record_conflict committed despite commit=False"
        )
        cons.conn.commit()  # clean up

    def test_record_conflict_default_still_commits(self, temp_db):
        """Backward compat: callers that didn't pass commit= should
        still get the pre-fix behavior of committing immediately."""
        from mnemosyne.core.veracity_consolidation import VeracityConsolidator

        cons = VeracityConsolidator(db_path=temp_db)
        # No outer tx; _record_conflict should commit its own work.
        cons._record_conflict("fact_a", "fact_b", "test")
        assert not cons.conn.in_transaction
        # And the conflict row should be visible.
        count = cons.conn.execute(
            "SELECT COUNT(*) FROM conflicts WHERE fact_a_id = 'fact_a'"
        ).fetchone()[0]
        assert count == 1

    def test_begin_immediate_failure_raises_does_not_silently_proceed(
        self, temp_db
    ):
        """If BEGIN IMMEDIATE raises (lock held past busy_timeout,
        or any other OperationalError), `consolidate_fact` must
        propagate the error rather than fall through to the
        unprotected SELECT-then-write path. /review caught the
        pre-fix silent-fallthrough as 4-source HIGH — it
        reintroduced the exact race the method was trying to close."""
        from mnemosyne.core.veracity_consolidation import VeracityConsolidator

        cons = VeracityConsolidator(db_path=temp_db)

        # Simulate BEGIN IMMEDIATE failure by holding the writer
        # lock from a separate connection with a short busy_timeout
        # on our writer.
        cons.conn.execute("PRAGMA busy_timeout=100")  # 100ms only
        blocker = sqlite3.connect(str(temp_db))
        try:
            blocker.execute("BEGIN IMMEDIATE")  # hold the lock
            # Our consolidate_fact should fail rather than proceed.
            with pytest.raises(sqlite3.OperationalError):
                cons.consolidate_fact(
                    "Locked", "is", "blocked", "stated", "src",
                )
        finally:
            blocker.rollback()
            blocker.close()

    def test_consolidator_sets_wal_and_busy_timeout(self, tmp_path):
        """`VeracityConsolidator.__init__` should configure WAL +
        busy_timeout when it owns the connection. Without WAL the
        BEGIN IMMEDIATE lock blocks readers too; without
        busy_timeout the second writer fails instantly under
        contention. /review (Claude C3) caught the missing PRAGMAs."""
        from mnemosyne.core.veracity_consolidation import VeracityConsolidator

        db_path = tmp_path / "pragma_check.db"
        cons = VeracityConsolidator(db_path=db_path)

        # Verify journal_mode matches the configured mode (WAL unless
        # MNEMOSYNE_JOURNAL_MODE overrides it)
        from mnemosyne.core.journal import journal_mode
        mode = cons.conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
        assert mode.lower() == journal_mode(), (
            f"expected journal_mode={journal_mode()}, got {mode!r} — "
            "VeracityConsolidator.__init__ didn't apply PRAGMA"
        )

        # Verify busy_timeout is set (non-zero).
        timeout = cons.conn.execute(
            "PRAGMA busy_timeout"
        ).fetchone()[0]
        assert timeout > 0, (
            f"expected busy_timeout > 0, got {timeout} — "
            "VeracityConsolidator.__init__ didn't apply PRAGMA"
        )

    def test_partial_conflict_rollback_undoes_fact_insert(self, temp_db):
        """If a conflict-record INSERT fails mid-loop (after the
        fact INSERT but before all conflicts are recorded), the
        BEGIN IMMEDIATE scope's rollback should undo the fact
        INSERT. Pre-fix _record_conflict's commit between the
        fact INSERT and the conflict INSERT meant a failed
        second conflict would leave the fact + first conflict
        durable but later conflicts missing — partial state."""
        from mnemosyne.core.veracity_consolidation import VeracityConsolidator

        cons = VeracityConsolidator(db_path=temp_db)
        # Seed two existing facts so the new fact will produce
        # 2 conflicts when inserted.
        cons.consolidate_fact("Kate", "is", "X", "stated", "src_x")
        cons.consolidate_fact("Kate", "is", "Y", "stated", "src_y")
        # Both Kate-is-X and Kate-is-Y are in the DB.

        # Now monkey-patch _record_conflict to fail on the second
        # call. The first call should be deferred (commit=False);
        # the failure on the second should trigger the outer
        # rollback, undoing the fact INSERT entirely.
        call_count = {"n": 0}
        original = cons._record_conflict

        def fail_second(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 2:
                raise sqlite3.OperationalError("simulated mid-loop failure")
            return original(*args, **kwargs)

        cons._record_conflict = fail_second  # type: ignore

        with pytest.raises(sqlite3.OperationalError, match="simulated"):
            cons.consolidate_fact("Kate", "is", "Z", "stated", "src_z")

        del cons._record_conflict  # type: ignore

        # The new fact (Kate, is, Z) must NOT be in the DB.
        rows = cons.conn.execute(
            "SELECT subject, object FROM consolidated_facts "
            "WHERE subject = 'Kate' AND object = 'Z'"
        ).fetchall()
        assert len(rows) == 0, (
            "fact INSERT not rolled back after conflict-record "
            "failure — partial state leaked"
        )

    def test_race_window_widening_demonstrates_serialization(
        self, temp_db, monkeypatch
    ):
        """A more deterministic regression test than the
        barrier-only concurrency tests. Injects a small sleep
        between SELECT and INSERT to widen the race window. With
        BEGIN IMMEDIATE serialization in place, two threads still
        produce exactly one row + mention_count=2. Without it
        (pre-fix), the wide-open window guarantees an
        IntegrityError on one of them."""
        from mnemosyne.core.veracity_consolidation import VeracityConsolidator
        import time

        # Track whether the original bayesian_update was called
        # during the SELECT-then-write critical section, then
        # inject a sleep AFTER SELECT but BEFORE INSERT/UPDATE to
        # widen the race window.
        original_bayesian = VeracityConsolidator.bayesian_update

        def slow_bayesian(self, current_confidence, veracity):
            time.sleep(0.05)  # 50ms window
            return original_bayesian(self, current_confidence, veracity)

        monkeypatch.setattr(
            VeracityConsolidator, "bayesian_update", slow_bayesian
        )

        # Seed the row so concurrent calls go through the UPDATE
        # branch (which is where bayesian_update fires).
        seeder = VeracityConsolidator(db_path=temp_db)
        seeder.consolidate_fact("Liam", "is", "seeded", "stated", "src_seed")

        # Fire 4 concurrent updates; each delays inside
        # bayesian_update. Without BEGIN IMMEDIATE serialization,
        # at least one would race and lose its update. With
        # serialization, all 4 land and mention_count == 5.
        exceptions: List[BaseException] = []
        barrier = threading.Barrier(4)
        threads = [
            threading.Thread(target=_consolidate_in_thread, args=(
                temp_db, "Liam", "is", "seeded", "stated", f"src_{i}",
                barrier, exceptions,
            ))
            for i in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not exceptions, (
            f"race-widened test surfaced exception: {exceptions[:2]}"
        )

        row = seeder.conn.execute(
            "SELECT mention_count FROM consolidated_facts "
            "WHERE subject = 'Liam'"
        ).fetchone()
        assert row["mention_count"] == 5, (
            f"expected mention_count == 5 (seed + 4 concurrent "
            f"updates with widened race window), got "
            f"{row['mention_count']} — serialization lost updates"
        )
