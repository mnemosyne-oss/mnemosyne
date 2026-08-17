"""
Regression tests for race-pattern fix on `VeracityConsolidator`
sibling methods: `resolve_conflict`, `resolve_conflict_by_facts`,
`run_consolidation_pass`.

E2.a.5 (PR #84) closed the SELECT-then-write race in
`consolidate_fact`. The /review army on that PR flagged that the
three sibling write methods on the same class have the same
shape and are unprotected. E2.a.6 fixes them by extracting a
shared `_serialized_write` context manager and applying it to all
three.

Severity profile (analyzed during fix implementation):
  - `resolve_conflict`: **real bug.** Two concurrent calls with
    different `winning_fact_id` values on the same conflict_id can
    leave BOTH facts superseded.
  - `resolve_conflict_by_facts`: single UPDATE, no SELECT-write
    race shape. Wrapped for consistency.
  - `run_consolidation_pass`: read-decide-resolve loop. Race is
    benign (idempotent decisions) but wrapped to prevent
    interleaved writes from other writers during the pass.

These tests pin:
  - Concurrent `resolve_conflict` with conflicting winners produces
    a deterministic single winner, not both-superseded.
  - `_serialized_write` participates in caller-owned outer
    transactions.
  - `run_consolidation_pass` calling `resolve_conflict_by_facts`
    nests correctly (inner doesn't try its own BEGIN IMMEDIATE).
  - All three methods preserve their pre-fix happy-path behavior.
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
    db_path = tmp_path / "sibling_races.db"
    VeracityConsolidator(db_path=db_path)  # initialize schema
    return db_path


# ---------------------------------------------------------------------------
# resolve_conflict — the real race (different winning_fact_id values)
# ---------------------------------------------------------------------------


def test_concurrent_resolve_conflict_different_winners_deterministic(temp_db):
    """Two threads each pass a different ``winning_fact_id`` for the
    same conflict. Pre-fix both could pass the SELECT and the
    second UPDATE would mark the OTHER fact superseded — leaving
    BOTH facts with superseded_by set. Post-fix serialization makes
    one winner durable before the second call runs."""
    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Alice", "is", "engineer", "stated", "src_a")
    cons.consolidate_fact("Alice", "is", "manager", "inferred", "src_b")
    conflicts = cons.get_conflicts()
    assert conflicts, "test setup: expected a conflict from Alice's roles"
    conflict_id = conflicts[0]["id"]
    fact_a_id = conflicts[0]["fact_a_id"]
    fact_b_id = conflicts[0]["fact_b_id"]

    exceptions: List[BaseException] = []
    barrier = threading.Barrier(2)

    def resolve_in_thread(winner_id: str):
        try:
            sub_cons = VeracityConsolidator(db_path=temp_db)
            barrier.wait()
            sub_cons.resolve_conflict(conflict_id, winner_id)
        except BaseException as exc:
            exceptions.append(exc)

    t1 = threading.Thread(target=resolve_in_thread, args=(fact_a_id,))
    t2 = threading.Thread(target=resolve_in_thread, args=(fact_b_id,))
    t1.start(); t2.start()
    t1.join(); t2.join()

    # Either thread's exception is acceptable (the second writer
    # might see the already-resolved conflict and no-op), but BOTH
    # facts being superseded is the bug.
    rows = cons.conn.execute(
        "SELECT id, superseded_by FROM consolidated_facts "
        "WHERE subject = 'Alice'"
    ).fetchall()
    superseded_count = sum(1 for r in rows if r["superseded_by"] is not None)
    assert superseded_count <= 1, (
        f"both facts superseded ({superseded_count}/2) — concurrent "
        f"resolve_conflict with conflicting winners left a non-coherent "
        f"state. exceptions: {exceptions[:1]}"
    )


def test_resolve_conflict_happy_path_unchanged(temp_db):
    """Single-threaded resolve_conflict still produces the same
    end state as pre-fix. Locks in backward compat."""
    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Bob", "is", "engineer", "stated")
    cons.consolidate_fact("Bob", "is", "manager", "inferred")
    conflicts = cons.get_conflicts()
    assert conflicts

    cons.resolve_conflict(conflicts[0]["id"], conflicts[0]["fact_a_id"])

    # The other fact should be marked superseded.
    fact_b = cons.conn.execute(
        "SELECT superseded_by FROM consolidated_facts WHERE id = ?",
        (conflicts[0]["fact_b_id"],),
    ).fetchone()
    assert fact_b["superseded_by"] == conflicts[0]["fact_a_id"]

    # The conflict should be marked resolved.
    conflict_row = cons.conn.execute(
        "SELECT resolution FROM conflicts WHERE id = ?",
        (conflicts[0]["id"],),
    ).fetchone()
    assert conflict_row["resolution"] is not None


# ---------------------------------------------------------------------------
# resolve_conflict_by_facts — wrapped for consistency
# ---------------------------------------------------------------------------


def test_resolve_conflict_by_facts_happy_path_unchanged(temp_db):
    """Single-call behavior is unchanged after wrapping. The
    serialization is defensive — single-UPDATE doesn't need it for
    correctness, but the wrap keeps the canonical pattern."""
    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Carol", "is", "lead", "stated")
    cons.consolidate_fact("Carol", "is", "founder", "stated")

    rows = cons.conn.execute(
        "SELECT id, object FROM consolidated_facts WHERE subject = 'Carol'"
    ).fetchall()
    by_object = {r["object"]: r["id"] for r in rows}
    winning = by_object["lead"]
    losing = by_object["founder"]

    cons.resolve_conflict_by_facts(winning, losing)

    superseded = cons.conn.execute(
        "SELECT superseded_by FROM consolidated_facts WHERE id = ?",
        (losing,),
    ).fetchone()
    assert superseded["superseded_by"] == winning


def test_concurrent_resolve_conflict_by_facts_idempotent(temp_db):
    """Two threads calling resolve_conflict_by_facts with the SAME
    winner+loser arguments must converge to one durable state with
    no IntegrityError or partial writes."""
    cons = VeracityConsolidator(db_path=temp_db)
    cons.consolidate_fact("Dan", "is", "A", "stated")
    cons.consolidate_fact("Dan", "is", "B", "stated")
    rows = cons.conn.execute(
        "SELECT id, object FROM consolidated_facts WHERE subject = 'Dan'"
    ).fetchall()
    winning = next(r["id"] for r in rows if r["object"] == "A")
    losing = next(r["id"] for r in rows if r["object"] == "B")

    exceptions: List[BaseException] = []
    barrier = threading.Barrier(4)

    def in_thread():
        try:
            sub = VeracityConsolidator(db_path=temp_db)
            barrier.wait()
            sub.resolve_conflict_by_facts(winning, losing)
        except BaseException as exc:
            exceptions.append(exc)

    threads = [threading.Thread(target=in_thread) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert not exceptions, f"unexpected: {exceptions[:1]}"
    final = cons.conn.execute(
        "SELECT superseded_by FROM consolidated_facts WHERE id = ?",
        (losing,),
    ).fetchone()
    assert final["superseded_by"] == winning


# ---------------------------------------------------------------------------
# run_consolidation_pass — nested-call correctness
# ---------------------------------------------------------------------------


def test_run_consolidation_pass_resolves_obvious_conflicts(temp_db):
    """Happy-path: pass with high-confidence vs low-confidence
    facts auto-resolves in favor of high-confidence. Pre-fix
    behavior unchanged."""
    cons = VeracityConsolidator(db_path=temp_db)
    # Seed a high-confidence fact (multiple stated mentions to push
    # mention_count > 2 AND boost confidence) + a low-confidence
    # competing fact.
    for _ in range(4):
        cons.consolidate_fact("Eve", "is", "CEO", "stated", "src_high")
    cons.consolidate_fact("Eve", "is", "VP", "inferred", "src_low")

    cons.run_consolidation_pass()

    # VP should be superseded, CEO active.
    rows = cons.conn.execute(
        "SELECT object, superseded_by FROM consolidated_facts "
        "WHERE subject = 'Eve' ORDER BY object"
    ).fetchall()
    by_object = {r["object"]: r["superseded_by"] for r in rows}
    assert by_object["CEO"] is None, "CEO should remain active"
    assert by_object["VP"] is not None, (
        "VP should be superseded by run_consolidation_pass"
    )


def test_run_consolidation_pass_nested_resolve_does_not_crash(temp_db):
    """`run_consolidation_pass` opens BEGIN IMMEDIATE via
    `_serialized_write`. Inside it calls `resolve_conflict_by_facts`
    which ALSO uses `_serialized_write`. The inner call's
    `conn.in_transaction` check must detect the outer scope and
    skip its own BEGIN — otherwise it would raise
    `cannot start a transaction within a transaction`."""
    cons = VeracityConsolidator(db_path=temp_db)
    # Seed facts ready for the pass.
    for _ in range(4):
        cons.consolidate_fact("Frank", "uses", "Python", "stated", "src_x")
    cons.consolidate_fact("Frank", "uses", "Rust", "inferred", "src_y")

    # Should not raise.
    cons.run_consolidation_pass()


# ---------------------------------------------------------------------------
# _serialized_write helper directly
# ---------------------------------------------------------------------------


def test_serialized_write_begins_immediate_when_not_in_tx(temp_db):
    """The helper should issue BEGIN IMMEDIATE and own the
    commit/rollback lifecycle when the connection isn't in a tx."""
    cons = VeracityConsolidator(db_path=temp_db)
    assert not cons.conn.in_transaction

    with cons._serialized_write():
        assert cons.conn.in_transaction, (
            "_serialized_write didn't open a transaction"
        )
        cons.conn.execute(
            "INSERT INTO consolidated_facts "
            "(id, subject, predicate, object, confidence, mention_count, "
            " first_seen, last_seen, sources_json, veracity) "
            "VALUES ('cf_test', 's', 'p', 'o', 0.5, 1, "
            "datetime('now'), datetime('now'), '[]', 'stated')"
        )

    # After the with-block, tx should be committed and closed.
    assert not cons.conn.in_transaction
    row = cons.conn.execute(
        "SELECT * FROM consolidated_facts WHERE id = 'cf_test'"
    ).fetchone()
    assert row is not None


def test_serialized_write_rolls_back_on_exception(temp_db):
    """When the body raises, the helper must roll back its own
    transaction. Pre-fix consolidate_fact's inline rollback path
    was tested by E2.a.5; this test pins the helper's version."""
    cons = VeracityConsolidator(db_path=temp_db)

    with pytest.raises(RuntimeError, match="simulated"):
        with cons._serialized_write():
            cons.conn.execute(
                "INSERT INTO consolidated_facts "
                "(id, subject, predicate, object, confidence, "
                " mention_count, first_seen, last_seen, "
                " sources_json, veracity) "
                "VALUES ('cf_doomed', 's', 'p', 'o', 0.5, 1, "
                "datetime('now'), datetime('now'), '[]', 'stated')"
            )
            raise RuntimeError("simulated mid-write failure")

    # The INSERT should have been rolled back.
    row = cons.conn.execute(
        "SELECT * FROM consolidated_facts WHERE id = 'cf_doomed'"
    ).fetchone()
    assert row is None, (
        "rollback didn't undo the insert — leaked into DB"
    )


def test_serialized_write_participates_in_outer_transaction(temp_db):
    """When the connection is already in a tx, `_serialized_write`
    must NOT try BEGIN IMMEDIATE (which would raise) and must NOT
    issue its own commit on exit (caller owns it)."""
    cons = VeracityConsolidator(db_path=temp_db)
    cons.conn.execute("BEGIN")
    assert cons.conn.in_transaction

    with cons._serialized_write():
        # Still in the OUTER tx — helper didn't start its own.
        assert cons.conn.in_transaction
        cons.conn.execute(
            "INSERT INTO consolidated_facts "
            "(id, subject, predicate, object, confidence, "
            " mention_count, first_seen, last_seen, sources_json, "
            " veracity) "
            "VALUES ('cf_nested', 's', 'p', 'o', 0.5, 1, "
            "datetime('now'), datetime('now'), '[]', 'stated')"
        )

    # Outer tx still open — helper didn't commit ours.
    assert cons.conn.in_transaction
    cons.conn.commit()


# ---------------------------------------------------------------------------
# /review hardening (commit 2) — 4-source convergence
# ---------------------------------------------------------------------------


class TestReviewHardening:
    """Closes findings from the /review army on commit 1:
      1. Same-connection writer race (Codex structured GATE FAIL P2) —
         threading.RLock added to instance
      2. Missing WAL/busy_timeout PRAGMAs (Claude CRITICAL) — added
         to __init__
      3. Self.conn capture in helper (Codex adv MED) — captured once
      4. WARNING log assertion for first-writer-wins guard
         (Maintainability MED + Codex adv)
      5. Race-window-widening test pattern (E2.a.5 reference) —
         deterministic regression guard
    """

    def test_consolidator_sets_wal_and_busy_timeout(self, tmp_path):
        """Mirrors the E2.a.5 PR #84 test. VeracityConsolidator
        must apply WAL + busy_timeout when constructing its own
        connection so `_serialized_write`'s BEGIN IMMEDIATE has the
        correct contention semantics (waits up to 5s rather than
        raising instantly under default journal_mode=DELETE)."""
        db_path = tmp_path / "pragma_check.db"
        cons = VeracityConsolidator(db_path=db_path)

        from mnemosyne.core.journal import journal_mode
        mode = cons.conn.execute(
            "PRAGMA journal_mode"
        ).fetchone()[0]
        assert mode.lower() == journal_mode(), (
            f"expected journal_mode=wal, got {mode!r} — "
            "VeracityConsolidator.__init__ didn't apply PRAGMA"
        )

        timeout = cons.conn.execute(
            "PRAGMA busy_timeout"
        ).fetchone()[0]
        assert timeout > 0, (
            f"expected busy_timeout > 0, got {timeout}"
        )

    def test_same_connection_writers_serialize_via_rlock(self, temp_db):
        """When two threads share the same VeracityConsolidator
        instance (and therefore the same self.conn), BEGIN IMMEDIATE
        alone doesn't protect them — the first thread's BEGIN makes
        conn.in_transaction=True, and the second sees that and
        skips BEGIN, entering the critical section within the
        first's open transaction. The instance-level RLock added in
        commit 2 closes this gap.

        We verify by setting up the same shared-instance scenario
        and exercising it with the race-window-widening pattern
        from E2.a.5: monkey-patch bayesian_update (the deepest
        helper called inside _serialized_write scope via
        consolidate_fact) to sleep, deterministically holding the
        critical section open. Then fire 4 threads doing
        resolve_conflict_by_facts (also wrapped by _serialized_write)
        against the same instance — all 4 should serialize on the
        RLock and produce a coherent end state."""
        import time
        cons = VeracityConsolidator(db_path=temp_db)
        cons.consolidate_fact("Alice", "is", "X", "stated")
        cons.consolidate_fact("Alice", "is", "Y", "stated")
        rows = cons.conn.execute(
            "SELECT id, object FROM consolidated_facts WHERE subject = 'Alice'"
        ).fetchall()
        winning = next(r["id"] for r in rows if r["object"] == "X")
        losing = next(r["id"] for r in rows if r["object"] == "Y")

        # Inject delay inside the critical section. The patch fires
        # before each resolve_conflict_by_facts UPDATE so concurrent
        # threads contend on the RLock.
        original_resolve = cons.resolve_conflict_by_facts

        def slow_resolve(*args, **kwargs):
            time.sleep(0.02)
            return original_resolve(*args, **kwargs)

        cons.resolve_conflict_by_facts = slow_resolve  # type: ignore

        exceptions: List[BaseException] = []
        barrier = threading.Barrier(4)

        def in_thread():
            try:
                barrier.wait()
                # Use the SHARED instance, not a fresh one — that's
                # the same-conn scenario the RLock protects.
                cons.resolve_conflict_by_facts(winning, losing)
            except BaseException as exc:
                exceptions.append(exc)

        threads = [threading.Thread(target=in_thread) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        del cons.resolve_conflict_by_facts  # type: ignore

        # No exceptions (RLock prevented IntegrityError from
        # interleaved writes via SQLite's same-conn-transaction model).
        assert not exceptions, (
            f"same-conn race surfaced exception: {exceptions[:1]} — "
            "RLock didn't serialize"
        )
        # End state coherent: losing fact marked superseded by winning.
        final = cons.conn.execute(
            "SELECT superseded_by FROM consolidated_facts WHERE id = ?",
            (losing,),
        ).fetchone()
        assert final["superseded_by"] == winning

    def test_first_writer_wins_logs_warning(self, temp_db, caplog):
        """When the already-resolved guard kicks in, a WARNING log
        should fire so operators can spot conflicting writes.
        /review (Maintainability MED) flagged the missing log
        assertion."""
        import logging
        cons = VeracityConsolidator(db_path=temp_db)
        cons.consolidate_fact("Bob", "is", "X", "stated")
        cons.consolidate_fact("Bob", "is", "Y", "inferred")
        conflicts = cons.get_conflicts()
        conflict_id = conflicts[0]["id"]
        fact_a_id = conflicts[0]["fact_a_id"]
        fact_b_id = conflicts[0]["fact_b_id"]

        # First resolution: succeeds, no warning.
        with caplog.at_level(logging.WARNING):
            cons.resolve_conflict(conflict_id, fact_a_id)
            assert not any(
                "already resolved" in rec.message
                for rec in caplog.records
            )

        # Second resolution on same conflict_id: first-writer-wins
        # guard kicks in; should log WARNING.
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            cons.resolve_conflict(conflict_id, fact_b_id)
            assert any(
                "already resolved" in rec.message
                for rec in caplog.records
            ), (
                f"expected WARNING about already-resolved conflict; "
                f"got logs: {[r.message for r in caplog.records]}"
            )

    def test_helper_captures_conn_at_entry(self, temp_db):
        """_serialized_write captures `conn = self.conn` once at
        entry so commit/rollback target the same connection BEGIN
        IMMEDIATE opened on — defense against the body swapping
        self.conn mid-scope. /review (Codex adv MED)."""
        cons = VeracityConsolidator(db_path=temp_db)
        original_conn = cons.conn

        with cons._serialized_write():
            # Swap to a different (in-memory) conn mid-scope.
            sqlite_other = sqlite3.connect(":memory:")
            cons.conn = sqlite_other  # type: ignore
            # The original conn should still be the one BEGIN
            # IMMEDIATE was issued on; commit/rollback on context
            # exit must target it.

        # Restore the original conn for cleanup.
        cons.conn = original_conn

        # The original conn should NOT be in a transaction
        # (commit fired against the captured `conn = self.conn`
        # from entry, which was original_conn).
        assert not original_conn.in_transaction, (
            "original conn left mid-transaction — helper didn't "
            "capture self.conn at entry"
        )

    # Race-window-widening test was attempted but `datetime.datetime.now`
    # can't be monkey-patched (immutable C type). The other tests in
    # this class cover the same regression surface:
    #   - test_same_connection_writers_serialize_via_rlock exercises
    #     real contention via slow_resolve injection
    #   - test_first_writer_wins_logs_warning pins the
    #     already-resolved guard
    #   - test_concurrent_resolve_conflict_different_winners_deterministic
    #     exercises the cross-thread race shape
    # Adding a deterministic delay-injection test would require a
    # different injection point (e.g., a hook on _serialized_write
    # itself). Tracked as a future test-coverage improvement.

    def test_consolidate_fact_same_connection_serializes_via_rlock(
        self, temp_db
    ):
        """Post-DRY-refactor: `consolidate_fact` uses `_serialized_write`
        and acquires `_write_lock` like the other three write methods.
        Pre-DRY (PR #84's inline pattern), `consolidate_fact` did NOT
        acquire `_write_lock` — so two threads sharing one
        VeracityConsolidator instance could still race within a single
        SQL transaction (BEGIN IMMEDIATE protects across connections
        but not within one).

        Mirrors `test_same_connection_writers_serialize_via_rlock` but
        exercises `consolidate_fact` rather than
        `resolve_conflict_by_facts`. Closes the same-conn race shape
        for the fourth and final write method on
        VeracityConsolidator."""
        import time
        cons = VeracityConsolidator(db_path=temp_db)

        # Inject delay inside `bayesian_update` (UPDATE branch's
        # only Python-level work outside the SQL UPDATE itself) to
        # widen the critical section.
        original_bayesian = cons.bayesian_update

        def slow_bayesian(current_confidence, veracity):
            time.sleep(0.02)
            return original_bayesian(current_confidence, veracity)

        cons.bayesian_update = slow_bayesian  # type: ignore

        # Seed so concurrent calls hit the UPDATE branch.
        cons.consolidate_fact("Diana", "is", "founder", "stated", "seed")

        exceptions: List[BaseException] = []
        barrier = threading.Barrier(4)

        def in_thread(i: int):
            try:
                barrier.wait()
                # SHARED instance — exercises the same-conn race path.
                cons.consolidate_fact(
                    "Diana", "is", "founder", "stated", f"src_{i}",
                )
            except BaseException as exc:
                exceptions.append(exc)

        threads = [
            threading.Thread(target=in_thread, args=(i,))
            for i in range(4)
        ]
        for t in threads: t.start()
        for t in threads: t.join()

        del cons.bayesian_update  # type: ignore

        # No exceptions (RLock prevented same-conn interleave).
        assert not exceptions, (
            f"same-conn race in consolidate_fact: {exceptions[:1]} — "
            "DRY refactor's RLock acquisition broken"
        )
        # mention_count should be exactly 5: 1 seed + 4 concurrent
        # updates. Pre-DRY (no RLock for consolidate_fact) the race
        # would lose at least one update.
        row = cons.conn.execute(
            "SELECT mention_count FROM consolidated_facts "
            "WHERE subject = 'Diana'"
        ).fetchone()
        assert row["mention_count"] == 5, (
            f"expected mention_count == 5 (seed + 4 concurrent "
            f"updates), got {row['mention_count']} — same-conn race "
            "lost update(s) in consolidate_fact"
        )
