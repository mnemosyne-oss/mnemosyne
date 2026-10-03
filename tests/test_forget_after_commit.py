"""
Tests for after-commit event emission in forget() (issue #963).

Pre-fix, ``Mnemosyne.forget()`` emitted ``MEMORY_INVALIDATED``
immediately after the ``_deferred_commits`` block. When a caller-owned
transaction was still open, the event fired before the caller's commit
— a phantom event if the caller then rolled back.

Post-fix, ``forget()`` emits at once when it owned the transaction and
queues an after-commit hook on the connection otherwise. The hook fires
on the next real commit and is discarded unseen on rollback.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from mnemosyne.core.memory import Mnemosyne


def _mem_with_events(tmp_path: Path, name: str = "forget_ac.db"):
    """Mnemosyne with a capturing emitter; returns (mem, events)."""
    mem = Mnemosyne(session_id="ac-test", db_path=tmp_path / name)
    events: list = []
    mem._emit_wrapper = lambda *args, **kwargs: events.append((args, kwargs))  # noqa: SLF001
    return mem, events


def test_owned_transaction_emits_immediately(tmp_path: Path):
    """Without a caller transaction, the event fires on return (unchanged)."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("owned row", source="test")

    assert mem.forget(mid) is True
    assert events == [(("MEMORY_INVALIDATED", mid), {})]


def test_caller_owned_commit_fires_event_after_commit(tmp_path: Path):
    """With a caller transaction open, no event until the caller commits."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("caller row", source="test")
    mem.conn.execute("BEGIN")

    assert mem.forget(mid) is True
    assert events == []

    mem.conn.commit()
    assert events == [(("MEMORY_INVALIDATED", mid), {})]


def test_caller_owned_rollback_suppresses_event(tmp_path: Path):
    """A caller rollback discards the queued event; the row survives."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("rollback row", source="test")
    mem.conn.execute("BEGIN")

    assert mem.forget(mid) is True
    mem.conn.rollback()

    assert events == []
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)
    ).fetchone()[0] == 1


def test_rollback_clears_stale_hooks(tmp_path: Path):
    """Hooks queued before a rollback never fire on a later commit."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("stale row", source="test")
    mem.conn.execute("BEGIN")
    assert mem.forget(mid) is True
    mem.conn.rollback()

    mem.beam.remember("unrelated row", source="test")
    mem.conn.commit()

    assert events == []


def test_failing_hook_does_not_break_commit(tmp_path: Path):
    """A raising hook is skipped; the commit itself still succeeds."""
    mem, _ = _mem_with_events(tmp_path)
    mem.conn.execute("BEGIN")
    mem.conn.execute(
        "INSERT INTO working_memory (id, content) VALUES ('hook-row', 'x')"
    )
    mem.conn._after_commit_hooks.append(lambda: 1 / 0)  # noqa: SLF001

    mem.conn.commit()  # must not raise

    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = 'hook-row'"
    ).fetchone()[0] == 1


def test_hook_registered_by_hook_defers_to_next_commit(tmp_path: Path):
    """A hook queued from inside a running hook fires on the next commit."""
    mem, _ = _mem_with_events(tmp_path)
    fired: list = []
    mem.conn.execute("BEGIN")
    mem.conn.execute(
        "INSERT INTO working_memory (id, content) VALUES ('re-row', 'x')"
    )

    def first() -> None:
        """Queue the follow-up hook; it must not run in this drain."""
        fired.append("first")
        mem.conn._after_commit_hooks.append(lambda: fired.append("second"))  # noqa: SLF001

    mem.conn._after_commit_hooks.append(first)  # noqa: SLF001
    mem.conn.commit()
    assert fired == ["first"]
    # A no-op commit (nothing open) must not fire the queued hook: only a
    # real committed transaction drains.
    mem.conn.commit()
    assert fired == ["first"]
    mem.conn.execute("BEGIN")
    mem.conn.execute(
        "INSERT INTO working_memory (id, content) VALUES ('re-row-2', 'x')"
    )
    mem.conn.commit()
    assert fired == ["first", "second"]


def test_legacy_only_emit_follows_caller_commit(tmp_path: Path):
    """Legacy-only cleanup returns false and emits no invalidation."""
    mem, events = _mem_with_events(tmp_path)
    mem.conn.execute(
        "CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, content TEXT, "
        "source TEXT, timestamp TEXT, session_id TEXT, importance REAL, metadata_json TEXT)"
    )
    mem.conn.execute(
        "INSERT INTO memories (id, content, session_id) VALUES ('leg-1', 'x', 'ac-test')"
    )
    mem.conn.commit()

    mem.conn.execute("BEGIN")
    assert mem.forget("leg-1") is False
    assert events == []
    mem.conn.commit()
    assert events == []
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = 'leg-1'"
    ).fetchone()[0] == 0

    mem.conn.execute(
        "INSERT INTO memories (id, content, session_id) VALUES ('leg-2', 'x', 'ac-test')"
    )
    mem.conn.commit()
    mem.conn.execute("BEGIN")
    assert mem.forget("leg-2") is False
    mem.conn.rollback()
    assert events == []
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM memories WHERE id = 'leg-2'"
    ).fetchone()[0] == 1


def test_no_raw_transaction_ending_sql(tmp_path: Path):
    """Pin the hook-aware invariant: no raw COMMIT/ROLLBACK/END statements.

    After-commit hooks live on _BeamConnection.commit()/rollback(). A raw
    transaction-ending statement would bypass them, so the codebase must
    keep routing transaction control through those methods.
    """
    del tmp_path  # static source scan; no fixture DB needed
    import re

    core = Path(__file__).resolve().parent.parent / "mnemosyne" / "core"
    # Scan complete file content (not line-by-line) so a multiline
    # execute("...") call cannot split the statement across the match.
    # ROLLBACK TO SAVEPOINT is excluded: it neither ends the transaction
    # nor bypasses the hooks (\s also spans line breaks in the lookahead).
    pattern = re.compile(
        r"""execute\s*\(\s*['\"](COMMIT|END|ROLLBACK(?!\s+TO))\b""", re.IGNORECASE
    )
    offenders = []
    for p in sorted(core.glob("*.py")):
        text = p.read_text()
        for m in pattern.finditer(text):
            lineno = text.count("\n", 0, m.start()) + 1
            offenders.append(f"{p.name}:{lineno}")
    assert offenders == []


def test_savepoint_rollback_discards_queued_event(tmp_path: Path):
    """A ROLLBACK TO undoing forget() must also discard its queued event.

    Regression for the savepoint gap in #963: the hook queue is
    connection-wide, so without savepoint tracking the row comes back
    while MEMORY_INVALIDATED still fires on commit.
    """
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("savepoint row", source="test")
    mem.conn.execute("BEGIN")
    mem.conn.execute("SAVEPOINT caller")
    assert mem.forget(mid) is True
    mem.conn.execute("ROLLBACK TO caller")
    mem.conn.execute("RELEASE caller")
    mem.conn.commit()

    assert events == []
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)
    ).fetchone()[0] == 1


def test_released_savepoint_keeps_queued_event(tmp_path: Path):
    """RELEASE merges into the outer scope: the event fires on commit."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("released row", source="test")
    mem.conn.execute("BEGIN")
    mem.conn.execute("SAVEPOINT caller")
    assert mem.forget(mid) is True
    mem.conn.execute("RELEASE caller")
    assert events == []
    mem.conn.commit()

    assert events == [(("MEMORY_INVALIDATED", mid), {})]
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)
    ).fetchone()[0] == 0


def test_cursor_savepoint_rollback_discards_queued_event(tmp_path: Path):
    """A cursor-level ROLLBACK TO discards the hook like any other (#963)."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("cursor row", source="test")
    cur = mem.conn.cursor()
    mem.conn.execute("BEGIN")
    cur.execute("SAVEPOINT caller")
    assert mem.forget(mid) is True
    cur.execute("ROLLBACK TO caller")
    cur.execute("RELEASE caller")
    mem.conn.commit()

    assert events == []
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)
    ).fetchone()[0] == 1


def test_outermost_savepoint_release_emits_on_implicit_commit(tmp_path: Path):
    """RELEASE of a bare SAVEPOINT commits: the event fires at once (#963)."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("outer row", source="test")
    mem.conn.execute("SAVEPOINT caller")
    assert mem.forget(mid) is True
    assert events == []
    mem.conn.execute("RELEASE caller")

    assert events == [(("MEMORY_INVALIDATED", mid), {})]
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)
    ).fetchone()[0] == 0


def test_cursor_outermost_savepoint_release_emits_on_implicit_commit(tmp_path: Path):
    """A bare cursor SAVEPOINT + RELEASE commits: the event fires (#963)."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("cursor outer row", source="test")
    cur = mem.conn.cursor()
    cur.execute("SAVEPOINT caller")
    assert mem.forget(mid) is True
    assert events == []
    cur.execute("RELEASE caller")

    assert events == [(("MEMORY_INVALIDATED", mid), {})]
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)
    ).fetchone()[0] == 0


def test_connection_executescript_drains_hook_before_script(tmp_path: Path):
    """Connection executescript emits at SQLite's implicit commit (#963)."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("script row", source="test")
    mem.conn.execute("BEGIN")
    assert mem.forget(mid) is True
    assert events == []

    mem.conn.executescript(
        "CREATE TABLE script_marker (value TEXT);"
        "INSERT INTO script_marker VALUES ('after-commit');"
    )

    assert events == [(("MEMORY_INVALIDATED", mid), {})]
    assert mem.conn._after_commit_hooks == []  # noqa: SLF001
    assert mem.conn.execute("SELECT value FROM script_marker").fetchone()[0] == (
        "after-commit"
    )


def test_cursor_executescript_drains_hook_when_script_fails(tmp_path: Path):
    """Cursor executescript does not retain hooks after implicit commit."""
    mem, events = _mem_with_events(tmp_path)
    mid = mem.beam.remember("failing script row", source="test")
    mem.conn.execute("BEGIN")
    assert mem.forget(mid) is True

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        mem.conn.cursor().executescript(
            "CREATE TABLE script_marker (value TEXT);"
            "INSERT INTO missing_script_table VALUES ('fails');"
        )

    assert events == [(("MEMORY_INVALIDATED", mid), {})]
    assert mem.conn._after_commit_hooks == []  # noqa: SLF001
    assert mem.conn.in_transaction is False


@pytest.mark.parametrize("statement", ["COMMIT", "END"])
@pytest.mark.parametrize("use_cursor", [False, True])
def test_raw_commit_terminators_drain_hooks(
    tmp_path: Path, statement: str, use_cursor: bool
):
    """Raw COMMIT/END must emit queued hooks in either execute path (#963)."""
    mem, events = _mem_with_events(tmp_path, f"raw-{statement}-{use_cursor}.db")
    mid = mem.beam.remember("raw commit row", source="test")
    mem.conn.execute("BEGIN")
    assert mem.forget(mid) is True

    executor = mem.conn.cursor().execute if use_cursor else mem.conn.execute
    executor(statement)

    assert events == [(("MEMORY_INVALIDATED", mid), {})]
    assert mem.conn._after_commit_hooks == []  # noqa: SLF001


@pytest.mark.parametrize("use_cursor", [False, True])
def test_raw_rollback_transaction_clears_hooks(tmp_path: Path, use_cursor: bool):
    """Raw ROLLBACK TRANSACTION must suppress queued hooks in both paths."""
    mem, events = _mem_with_events(tmp_path, f"raw-rollback-{use_cursor}.db")
    mid = mem.beam.remember("raw rollback row", source="test")
    mem.conn.execute("BEGIN")
    assert mem.forget(mid) is True

    executor = mem.conn.cursor().execute if use_cursor else mem.conn.execute
    executor("ROLLBACK TRANSACTION")

    assert events == []
    assert mem.conn._after_commit_hooks == []  # noqa: SLF001
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)
    ).fetchone()[0] == 1


@pytest.mark.parametrize(
    "statement, survives",
    [("-- audit\nCOMMIT", False), ("/* audit */ ROLLBACK TRANSACTION", True)],
)
@pytest.mark.parametrize("use_cursor", [False, True])
def test_commented_raw_terminators_keep_hooks_consistent(
    tmp_path: Path, statement: str, survives: bool, use_cursor: bool
):
    """Leading SQL comments must not bypass transaction hook handling."""
    mem, events = _mem_with_events(
        tmp_path, f"commented-{survives}-{use_cursor}.db"
    )
    mid = mem.beam.remember("commented transaction row", source="test")
    mem.conn.execute("BEGIN")
    assert mem.forget(mid) is True

    executor = mem.conn.cursor().execute if use_cursor else mem.conn.execute
    executor(statement)

    assert events == ([] if survives else [(("MEMORY_INVALIDATED", mid), {})])
    assert mem.conn._after_commit_hooks == []  # noqa: SLF001
    assert mem.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (mid,)
    ).fetchone()[0] == (1 if survives else 0)
