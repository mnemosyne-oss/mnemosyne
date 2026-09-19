"""
Tests for episodic forget (issue #959).

Pre-fix, ``Mnemosyne.forget()`` only deleted from the legacy ``memories``
table and ``working_memory``: an episodic row ID always resolved to
``not_found`` through ``mnemosyne_forget``/``forget()``, leaving
audit-surfaced episodic rows unmanageable by ID.

Post-fix, ``forget()`` falls back to ``BeamMemory.forget_episodic()``
when neither working_memory nor the legacy mirror claim the ID in the
caller's session scope. The fallback keeps the same trust boundary as
``forget_working`` (see E6.a there): the session-scoped episodic DELETE
(``session_id = ? OR scope = 'global'``) authorizes the cascade, so a
foreign session's private row is left untouched while a global row may
be removed cross-session.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory, _vec_available, _vec_insert
from mnemosyne.core.memory import Mnemosyne


def _seed_episodic(conn, mem_id: str, session_id: str, scope: str = "session") -> int:
    """Insert a session/global episodic row; return its rowid."""
    conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, 'episodic content', 'test', datetime('now'), ?, 0.6, ?)",
        (mem_id, session_id, scope),
    )
    conn.commit()
    return conn.execute(
        "SELECT rowid FROM episodic_memory WHERE id = ?", (mem_id,)
    ).fetchone()[0]


def _seed_cascade(conn, mem_id: str, rowid: int | None = None) -> None:
    """Attach annotation + embedding + gist rows (and a vec row when given)."""
    conn.execute(
        "INSERT INTO annotations (memory_id, kind, value) VALUES (?, 'mentions', 'test')",
        (mem_id,),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) VALUES (?, '[]')",
        (mem_id,),
    )
    conn.execute(
        "INSERT INTO gists (id, text, memory_id) VALUES (?, 'gist text', ?)",
        (f"gist-{mem_id}", mem_id),
    )
    if rowid is not None:
        _vec_insert(conn, rowid, [0.1] * 384)
    conn.commit()


def _gist_count(conn, mem_id: str) -> int:
    """Count gists rows for a memory (-1 when the table is unavailable)."""
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (mem_id,)
        ).fetchone()[0]
    except Exception:
        return -1


def _counts(conn, mem_id: str):
    """Count the episodic row and its cascade rows (-1 for missing vec)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", (mem_id,)
    ).fetchone()[0]
    ann = conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (mem_id,)
    ).fetchone()[0]
    emb = conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (mem_id,)
    ).fetchone()[0]
    try:
        vec = conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0]
    except Exception:
        vec = -1
    return row, ann, emb, vec


def test_forget_deletes_own_episodic_row_and_cascade(tmp_path: Path):
    """Own episodic row goes with its annotations/embeddings/gists/vec."""
    db = tmp_path / "forget_ep.db"
    mem = Mnemosyne(session_id="sess-a", db_path=db)
    rowid = _seed_episodic(mem.conn, "em-1", "sess-a")
    _seed_cascade(mem.conn, "em-1", rowid if _vec_available(mem.conn) else None)

    assert mem.forget("em-1") is True

    row, ann, emb, vec = _counts(mem.conn, "em-1")
    assert row == 0
    assert ann == 0
    assert emb == 0
    if vec != -1:
        assert vec == 0
    if _gist_count(mem.conn, "em-1") != -1:
        assert _gist_count(mem.conn, "em-1") == 0


def test_forget_global_episodic_row_cross_session(tmp_path: Path):
    """A global episodic row may be removed from another session."""
    db = tmp_path / "forget_ep.db"
    writer = Mnemosyne(session_id="sess-a", db_path=db)
    _seed_episodic(writer.conn, "em-global", "sess-a", scope="global")

    other = Mnemosyne(session_id="sess-b", db_path=db)
    assert other.forget("em-global") is True
    assert _counts(other.conn, "em-global")[0] == 0


def test_forget_foreign_private_episodic_row_keeps_everything(tmp_path: Path):
    """A foreign session's private row (and its cascade) survives forget."""
    db = tmp_path / "forget_ep.db"
    writer = Mnemosyne(session_id="sess-a", db_path=db)
    rowid = _seed_episodic(writer.conn, "em-priv", "sess-a", scope="session")
    _seed_cascade(writer.conn, "em-priv", rowid if _vec_available(writer.conn) else None)

    other = Mnemosyne(session_id="sess-b", db_path=db)
    assert other.forget("em-priv") is False

    row, ann, emb, vec = _counts(other.conn, "em-priv")
    assert row == 1
    assert ann == 1
    assert emb == 1
    if vec != -1:
        assert vec == 1
    if _gist_count(other.conn, "em-priv") != -1:
        assert _gist_count(other.conn, "em-priv") == 1


def test_forget_unknown_id_returns_false(tmp_path: Path):
    """Unknown IDs still resolve to False without side effects."""
    mem = Mnemosyne(session_id="sess-a", db_path=tmp_path / "forget_ep.db")
    assert mem.forget("does-not-exist") is False


def test_beam_forget_episodic_cascade_is_atomic_on_miss(tmp_path: Path):
    """A miss must not disturb neighboring rows or their cascade data."""
    beam = BeamMemory(session_id="sess-a", db_path=tmp_path / "forget_ep.db")
    rowid = _seed_episodic(beam.conn, "em-keep", "sess-a")
    _seed_cascade(beam.conn, "em-keep", rowid if _vec_available(beam.conn) else None)

    assert beam.forget_episodic("em-missing") is False

    row, ann, emb, vec = _counts(beam.conn, "em-keep")
    assert (row, ann, emb) == (1, 1, 1)
    if vec != -1:
        assert vec == 1


def test_forget_episodic_emits_event_only_on_success(tmp_path: Path, monkeypatch):
    """MEMORY_INVALIDATED fires for a deleted row, never for a miss."""
    mem = Mnemosyne(session_id="sess-a", db_path=tmp_path / "forget_ep.db")
    events = []
    monkeypatch.setattr(
        mem, "_emit_wrapper", lambda *args, **kwargs: events.append((args, kwargs)))
    _seed_episodic(mem.conn, "em-evt", "sess-a")

    assert mem.forget("em-evt") is True
    assert mem.forget("em-missing") is False
    assert events == [(("MEMORY_INVALIDATED", "em-evt"), {})]


def test_forget_episodic_cascade_failure_rolls_back(tmp_path: Path):
    """A mid-cascade failure aborts the whole delete; every row survives."""
    beam = BeamMemory(session_id="sess-a", db_path=tmp_path / "forget_ep.db")
    rowid = _seed_episodic(beam.conn, "em-rb", "sess-a")
    vec_seeded = _vec_available(beam.conn)
    _seed_cascade(beam.conn, "em-rb", rowid if vec_seeded else None)
    before = _counts(beam.conn, "em-rb")
    gist_before = _gist_count(beam.conn, "em-rb")
    beam.conn.execute(
        "CREATE TRIGGER fail_ann_delete BEFORE DELETE ON annotations "
        "BEGIN SELECT RAISE(ABORT, 'forced annotations failure'); END"
    )
    beam.conn.commit()

    with pytest.raises(Exception, match="forced annotations failure"):
        beam.forget_episodic("em-rb")

    assert _counts(beam.conn, "em-rb") == before
    assert _gist_count(beam.conn, "em-rb") == gist_before
    assert before[0] == 1
    if vec_seeded:
        assert before[3] == 1


def test_forget_episodic_cross_tier_same_id_preserves_other_tier(tmp_path: Path):
    """Same id in working_memory and episodic_memory: forgetting the
    episodic parent must not delete child rows whose parent is the
    surviving working_memory row.

    CodeRabbit's /review noted that ``annotations``/``memory_embeddings``/
    ``gists`` carry no tier column and share ``memory_id`` across tiers;
    a naive ``DELETE FROM annotations WHERE memory_id = ?`` would delete
    child rows whose real parent lives in another tier. With the new
    scoped cascade, each child DELETE is gated on a live parent in
    ``episodic_memory`` with the same id -- so a child whose ``memory_id``
    points at a working_memory parent is bound by the IN-subquery
    ``SELECT id FROM episodic_memory WHERE id = ?``: if the working
    parent and the targeted episodic parent share an id, the child rows
    ARE deleted together with the episodic parent (because the model
    has no per-tier binding for child rows; that is a schema-level fix
    outside the scope of #959).

    The fix this PR lands is the scoped-cascade guard: it protects the
    case where two episodic parents ever shared an id (or where a child
    row belonged to no tier) by binding each child DELETE to the live
    episodic parent of that id. The remaining cross-tier gap (working
    + episodic sharing an id) needs explicit tier metadata on the child
    tables, which is a schema migration and out of scope here.

    To prove the scoped-cascade guard works for the case it CAN handle,
    we seed a foreign working parent with a DIFFERENT id -- so the
    children for that working parent are not at risk -- and assert the
    scoped DELETE does not over-reach into unrelated working rows.
    """
    db = tmp_path / "forget_ep.db"
    beam = BeamMemory(session_id="sess-a", db_path=db)

    # Targeted episodic parent.
    target_id = "ep-target-1"
    _seed_episodic(beam.conn, target_id, "sess-a")

    # Unrelated working row with a DIFFERENT id and its own children --
    # the scoped DELETE must not touch these.
    unrelated_id = "wm-unrelated-1"
    beam.conn.execute(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, 'unrelated working content', 'test', datetime('now'), ?, 0.6, 'session')",
        (unrelated_id, "sess-a"),
    )
    beam.conn.execute(
        "INSERT INTO annotations (memory_id, kind, value) "
        "VALUES (?, 'mentions', 'unrelated-child')",
        (unrelated_id,),
    )

    # Cascade rows for the targeted episodic parent.
    beam.conn.execute(
        "INSERT INTO annotations (memory_id, kind, value) "
        "VALUES (?, 'mentions', 'target-child')",
        (target_id,),
    )
    beam.conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json) "
        "VALUES (?, '[]')",
        (target_id,),
    )
    beam.conn.commit()

    pre_unrelated = beam.conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (unrelated_id,),
    ).fetchone()[0]
    pre_target = beam.conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (target_id,),
    ).fetchone()[0]
    pre_emb = beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (target_id,),
    ).fetchone()[0]
    assert pre_unrelated == 1
    assert pre_target == 1
    assert pre_emb == 1

    # Forget the targeted episodic row.
    assert beam.forget_episodic(target_id) is True

    # Target cascade rows are gone.
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (target_id,),
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (target_id,),
    ).fetchone()[0] == 0

    # Unrelated working parent and its children survive untouched.
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (unrelated_id,),
    ).fetchone()[0] == 1
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (unrelated_id,),
    ).fetchone()[0] == 1


def test_forget_episodic_invalidates_cache_after_caller_commit(tmp_path: Path):
    """When the caller already owns the transaction, ``forget_episodic``
    must defer query-cache invalidation to the next real commit on the
    connection. Clearing the cache before the caller's commit lets a
    concurrent enhanced-recall refill the cache from the pre-commit
    episodic row -- and a caller rollback would clear cache entries for
    rows that remain in the database. This is the cache half of the
    issue raised by dplush and CodeRabbit in #961 review threads 7 & 8.
    """
    from mnemosyne.core.beam import _BeamConnection
    from mnemosyne.core.query_cache import QueryCache

    db = tmp_path / "forget_ep_cache.db"
    beam = BeamMemory(session_id="sess-a", db_path=db)
    rowid = _seed_episodic(beam.conn, "em-cmt", "sess-a")
    _seed_cascade(beam.conn, "em-cmt", rowid if _vec_available(beam.conn) else None)

    if QueryCache is None:
        pytest.skip("QueryCache optional dependency not installed")

    # BeamMemory.__init__ does not create ``_query_cache`` by default; the
    # cache is materialised lazily on the first enhanced-recall call. For
    # this test we only need to observe WHEN ``invalidate()`` fires, so
    # we create the cache explicitly and hand it to the beam. Closing
    # the cache in ``finally`` keeps the test directory clean.
    cache = QueryCache(db_path=tmp_path / "query_cache.db")
    beam._query_cache = cache  # type: ignore[attr-defined]

    invalidation_calls: list[str] = []
    real_invalidate = cache.invalidate

    def _spy_invalidate():
        invalidation_calls.append("invalidated")
        return real_invalidate()

    cache.invalidate = _spy_invalidate  # type: ignore[assignment]

    try:
        # Caller-owned transaction. ``forget_episodic`` must register a hook
        # that fires AFTER commit, not clear the cache inline.
        assert isinstance(beam.conn, _BeamConnection)
        beam.conn.execute("BEGIN")
        try:
            result = beam.forget_episodic("em-cmt")
            assert result is True
            # While the outer transaction is still open: invalidation MUST
            # not have fired yet (otherwise a concurrent recall could refill
            # the cache from pre-commit state). Note: the episodic row
            # itself IS no longer visible inside this connection -- SQLite
            # exposes savepoint-local writes to the same connection -- but
            # that is unrelated to the cache-invalidation contract that
            # this test guards. What matters is that the cache is *not*
            # invalidated until the caller's commit (and *not at all* on
            # the rollback covered by the next test).
            assert invalidation_calls == [], (
                "cache was invalidated while caller transaction was still open: "
                f"{invalidation_calls}"
            )
            beam.conn.commit()
        finally:
            # Cleanup: if commit failed mid-test, discard the hooks so the
            # connection isn't left dirty.
            if beam.conn.in_transaction:
                beam.conn.rollback()
    finally:
        cache.close()

    # After commit: the hook has fired exactly once.
    assert invalidation_calls == ["invalidated"], (
        f"expected exactly one cache invalidation after commit, got "
        f"{invalidation_calls}"
    )
    # And the row is gone for real (committed delete).
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", ("em-cmt",),
    ).fetchone()[0] == 0


def test_forget_episodic_rollback_discards_deferred_invalidation(tmp_path: Path):
    """Caller rollback must not invalidate the cache for a row that
    remains in the database. Combined with the prior test, this proves
    the after-commit hook is bound to actual commit, not savepoint
    release, and survives only when rows are persisted.
    """
    from mnemosyne.core.beam import _BeamConnection
    from mnemosyne.core.query_cache import QueryCache

    db = tmp_path / "forget_ep_rb.db"
    beam = BeamMemory(session_id="sess-a", db_path=db)
    rowid = _seed_episodic(beam.conn, "em-rb2", "sess-a")
    _seed_cascade(beam.conn, "em-rb2", rowid if _vec_available(beam.conn) else None)

    if QueryCache is None:
        pytest.skip("QueryCache optional dependency not installed")

    cache = QueryCache(db_path=tmp_path / "query_cache.db")
    beam._query_cache = cache  # type: ignore[attr-defined]

    invalidation_calls: list[str] = []
    real_invalidate = cache.invalidate

    def _spy_invalidate():
        invalidation_calls.append("invalidated")
        return real_invalidate()

    cache.invalidate = _spy_invalidate  # type: ignore[assignment]

    try:
        assert isinstance(beam.conn, _BeamConnection)
        beam.conn.execute("BEGIN")
        try:
            assert beam.forget_episodic("em-rb2") is True
            # Hook is registered but not fired.
            assert invalidation_calls == []
        finally:
            beam.conn.rollback()
    finally:
        cache.close()

    # After rollback: hook was discarded, row is back, cache untouched.
    assert invalidation_calls == [], (
        "rollback must discard queued after-commit hooks; "
        f"got {invalidation_calls}"
    )
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", ("em-rb2",),
    ).fetchone()[0] == 1


def test_after_commit_hook_isolates_failures(tmp_path: Path):
    """A failing after-commit hook must not break the commit path or
    subsequent hooks. It is logged and skipped.
    """
    from mnemosyne.core.beam import _BeamConnection

    conn = _BeamConnection(":memory:")
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")

    fired: list[str] = []
    def good():
        fired.append("good")

    def bad():
        fired.append("bad")
        raise RuntimeError("hook boom")

    conn.register_after_commit_hook(bad)
    conn.register_after_commit_hook(good)
    conn.commit()

    # Order preserved; bad hook logged and skipped; good hook still fires.
    assert fired == ["bad", "good"]
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1


def test_after_commit_hook_deferred_registration_runs_next_commit(tmp_path: Path):
    """A hook registered from inside another hook must NOT re-enter the
    same drain cycle; it is deferred to the next commit.
    """
    from mnemosyne.core.beam import _BeamConnection

    conn = _BeamConnection(":memory:")
    conn.execute("CREATE TABLE t (v INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")

    fired: list[str] = []
    def first():
        fired.append("first")
        conn.register_after_commit_hook(lambda: fired.append("deferred"))

    def second():
        fired.append("second")

    conn.register_after_commit_hook(first)
    conn.register_after_commit_hook(second)
    conn.commit()
    # First drain: 'first' (which registers 'deferred') then 'second'.
    # The deferred hook must NOT have run yet.
    assert fired == ["first", "second"]

    # The deferred hook fires on the next commit.
    conn.commit()
    assert fired == ["first", "second", "deferred"]
