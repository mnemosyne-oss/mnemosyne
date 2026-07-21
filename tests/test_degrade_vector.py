"""Regression tests for [C18.b]: degrade_episodic updates content text but
leaves stale dense embeddings. Pre-fix the embedding stored in vec_episodes
or memory_embeddings still represented the ORIGINAL content even after the
content was compressed/truncated, causing dense recall to score against
content that no longer exists in the row.

Two tests:
1. With embeddings provider available, degrade regenerates the embedding
   to match the new compressed content.
2. With embeddings provider unavailable, degrade invalidates (deletes)
   the stale embedding so dense recall doesn't return semantically
   misleading results.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _content_to_vec(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic content-encoding 'embedding'. Different content
    produces different vectors AND different sign patterns, so the
    sign-bit binarizer in binary_vectors.py also produces different
    bytes. Without sign-pattern variation, two distinct strings would
    binarize to identical bytes and the binary_vector regeneration
    assertion couldn't tell whether refresh ran."""
    v = np.zeros(dim, dtype=np.float32)
    if not text:
        return v
    # Spread 32 bits of the content hash across the leading dims as
    # signed ±1.0 — one sign per hash bit. Two distinct strings produce
    # two distinct hashes → ~16 differing sign bits → distinct binary
    # vectors after sign-bit binarization. (Python's hash() is salted
    # per process by PYTHONHASHSEED but stable within a process, which
    # is all we need here.)
    h = hash(text) & 0xFFFFFFFF
    for i in range(32):
        v[i] = 1.0 if (h >> i) & 1 else -1.0
    # Length and first-char in dims past the sign-bit block so the
    # float-embedding-differs assertions still see length/content
    # signal layered on top of the hash-driven sign pattern.
    v[32] = float(len(text))
    v[33] = float(ord(text[0]))
    return v


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Patch the embeddings module: available() returns True, embed()
    returns content-deterministic vectors, and force the in-memory
    fallback path so we don't need sqlite-vec loaded."""
    from mnemosyne.core import embeddings as emb

    monkeypatch.setattr(emb, "available", lambda: True)
    monkeypatch.setattr(
        emb, "embed",
        lambda texts: np.stack([_content_to_vec(t) for t in texts]),
    )
    # Force the memory_embeddings fallback path; sqlite-vec presence
    # varies across test environments and the bug is identical for
    # both stores.
    monkeypatch.setattr(beam_module, "_vec_available", lambda conn: False)
    return emb


def _read_fallback_embedding(db_path, memory_id):
    """Return the serialized embedding stored in memory_embeddings for
    the given memory_id, or None if missing."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT embedding_json FROM memory_embeddings WHERE memory_id = ?",
            (memory_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _read_binary_vector(db_path, memory_id):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT binary_vector FROM episodic_memory WHERE id = ?",
            (memory_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _read_vec_embedding(db_path, rowid):
    """Read the real sqlite-vec payload using a separately opened connection."""
    import sqlite_vec

    conn = sqlite3.connect(str(db_path))
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        row = conn.execute(
            "SELECT embedding FROM vec_episodes WHERE rowid = ?", (rowid,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


@pytest.fixture
def sqlite_vec_embeddings(monkeypatch):
    """Use the installed sqlite-vec extension with deterministic embeddings."""
    pytest.importorskip("sqlite_vec")
    from mnemosyne.core import embeddings as emb

    monkeypatch.setattr(emb, "available", lambda: True)
    monkeypatch.setattr(
        emb, "embed",
        lambda texts: np.stack([_content_to_vec(t) for t in texts]),
    )
    return emb


class TestDegradeEpisodicVectorRefresh:

    def test_sqlite_vec_refresh_keeps_savepoint_until_outer_commit(
        self, temp_db, sqlite_vec_embeddings
    ):
        """A sqlite-vec refresh must not commit/release degrade_row early."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        if not beam_module._vec_available(beam.conn):
            pytest.skip("sqlite-vec vec_episodes unavailable in this build")

        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        memory_id = beam.consolidate_to_episodic(
            summary=original, source_wm_ids=["fake-wm"], importance=0.6
        )
        rowid = beam.conn.execute(
            "SELECT rowid FROM episodic_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        original_vec = _read_vec_embedding(temp_db, rowid)
        assert original_vec is not None

        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        beam.conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        beam.conn.commit()

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier2_to_tier3"] == 1

        # A separate connection can observe both mutations only after
        # degrade_episodic's outer transaction commits.
        conn = sqlite3.connect(str(temp_db))
        try:
            content, tier = conn.execute(
                "SELECT content, tier FROM episodic_memory WHERE id = ?", (memory_id,)
            ).fetchone()
        finally:
            conn.close()
        assert content != original
        assert tier == 3
        assert _read_vec_embedding(temp_db, rowid) != original_vec

    def test_sqlite_vec_refresh_failure_rolls_back_row_and_vector(
        self, temp_db, sqlite_vec_embeddings, monkeypatch
    ):
        """A post-insert refresh failure rolls back content, tier, and vec0 data."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        if not beam_module._vec_available(beam.conn):
            pytest.skip("sqlite-vec vec_episodes unavailable in this build")

        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        memory_id = beam.consolidate_to_episodic(
            summary=original, source_wm_ids=["fake-wm"], importance=0.6
        )
        rowid = beam.conn.execute(
            "SELECT rowid FROM episodic_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        original_vec = _read_vec_embedding(temp_db, rowid)
        assert original_vec is not None

        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        beam.conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        beam.conn.commit()

        real_insert = beam_module._vec_table_insert

        def insert_then_fail(conn, table, vec_rowid, embedding, *, commit=True):
            real_insert(conn, table, vec_rowid, embedding, commit=commit)
            raise RuntimeError("simulated failure after sqlite-vec insert")

        monkeypatch.setattr(beam_module, "_vec_table_insert", insert_then_fail)

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier2_to_tier3"] == 0
        conn = sqlite3.connect(str(temp_db))
        try:
            content, tier = conn.execute(
                "SELECT content, tier FROM episodic_memory WHERE id = ?", (memory_id,)
            ).fetchone()
        finally:
            conn.close()
        assert content == original
        assert tier == 2
        assert _read_vec_embedding(temp_db, rowid) == original_vec

    def test_tier_2_to_tier_3_regenerates_embedding(self, temp_db, fake_embeddings):
        """When tier 2→3 truncation changes content, the embedding stored
        in memory_embeddings must update to match the new content."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        # Long original content that will be truncated by tier 2→3 (TIER3_MAX_CHARS=300)
        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        assert len(original) > beam_module.TIER3_MAX_CHARS

        memory_id = beam.consolidate_to_episodic(
            summary=original,
            source_wm_ids=["fake-wm"],
            importance=0.6,
        )

        original_embedding = _read_fallback_embedding(temp_db, memory_id)
        assert original_embedding is not None, (
            "memory_embeddings should contain a row after consolidate_to_episodic"
        )

        # Backdate to make the row eligible for tier 2→3 and set tier=2 so it
        # hits the truncation path (skips the LLM-summarization tier 1→2 path
        # which is a no-op when local_llm is unavailable).
        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier2_to_tier3"] == 1, (
            f"Expected one tier 2→3 transition, got {result}"
        )

        conn = sqlite3.connect(str(temp_db))
        new_content = conn.execute(
            "SELECT content FROM episodic_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        conn.close()
        assert new_content != original, "tier 2→3 should have truncated the content"

        post_embedding = _read_fallback_embedding(temp_db, memory_id)
        assert post_embedding is not None, (
            "memory_embeddings row missing after degrade; expected regenerated, "
            "not deleted, when the embeddings provider is available"
        )
        assert post_embedding != original_embedding, (
            "memory_embeddings still holds the pre-degradation embedding — "
            "dense recall would score against original content while displaying "
            "truncated content. C18.b regeneration did not run."
        )

    def test_tier_2_to_tier_3_regenerates_binary_vector(self, temp_db, fake_embeddings):
        """The binary_vector column on episodic_memory must also update
        to match the new content."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        memory_id = beam.consolidate_to_episodic(
            summary=original,
            source_wm_ids=["fake-wm"],
            importance=0.6,
        )

        if beam_module._mib is None:
            pytest.skip("binary vectorization not available in this build")

        original_bv = _read_binary_vector(temp_db, memory_id)
        assert original_bv is not None

        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        conn.commit()
        conn.close()

        beam.degrade_episodic(dry_run=False)

        post_bv = _read_binary_vector(temp_db, memory_id)
        assert post_bv is not None, (
            "binary_vector should be present (regenerated, not nulled) when "
            "the embedding provider is available"
        )
        assert post_bv != original_bv, (
            "binary_vector still holds pre-degradation bytes — same C18.b drift"
        )

    def test_tier_1_to_tier_2_llm_path_regenerates_embedding(
        self, temp_db, fake_embeddings, monkeypatch
    ):
        """Tier 1→2 with the LLM-summarization path active: monkeypatched
        local_llm replaces content with a stub summary. The embedding
        must regenerate to match. Closes the test gap from /review —
        same _refresh_episodic_embedding path is exercised but a
        regression specific to the tier 1→2 hookup (e.g., wrong arg)
        wouldn't slip through."""
        from mnemosyne.core import local_llm

        monkeypatch.setattr(local_llm, "llm_available", lambda: True)
        monkeypatch.setattr(
            local_llm, "summarize_memories",
            lambda lines, source="": "STUB SUMMARY produced by LLM (deterministic)",
        )

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Long content (>300 chars) so the tier 1→2 path hits the LLM branch
        long_original = ("OriginalLongFactPrefix " + "padding " * 100).strip()
        assert len(long_original) > 300

        memory_id = beam.consolidate_to_episodic(
            summary=long_original,
            source_wm_ids=["wm-1"],
            importance=0.6,
            tier=1,  # start hot: this test exercises the tier 1→2 transition
        )
        original_embedding = _read_fallback_embedding(temp_db, memory_id)
        assert original_embedding is not None

        # Backdate so the row is eligible for tier 1→2 (the row was created at
        # tier 1 above, so we just need the timestamp).
        old_ts = (datetime.now() - timedelta(days=beam_module.TIER2_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE episodic_memory SET created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier1_to_tier2"] == 1, (
            f"Expected one tier 1→2 transition, got {result}"
        )

        # Content should now be the stub summary
        conn = sqlite3.connect(str(temp_db))
        new_content = conn.execute(
            "SELECT content FROM episodic_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        conn.close()
        assert new_content.startswith("STUB SUMMARY"), (
            f"tier 1→2 LLM path didn't replace content: {new_content[:60]!r}"
        )

        # Embedding must match new (summary) content, not original
        post_embedding = _read_fallback_embedding(temp_db, memory_id)
        assert post_embedding is not None
        assert post_embedding != original_embedding, (
            "Embedding still reflects pre-LLM content; tier 1→2 path "
            "did not call _refresh_episodic_embedding"
        )

    def test_refresh_failure_rolls_back_content_update(
        self, temp_db, fake_embeddings, monkeypatch
    ):
        """[C18.b /review finding #1] If _refresh_episodic_embedding raises
        after the UPDATE statement runs, the SAVEPOINT must roll back the
        content mutation so we don't commit content=NEW with embedding=OLD
        (the very drift this PR fixes). Pre-fix, the broad except in the
        loop body swallowed the refresh exception and the UPDATE stayed
        staged in the implicit transaction."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        memory_id = beam.consolidate_to_episodic(
            summary=original,
            source_wm_ids=["fake-wm"],
            importance=0.6,
        )

        # Backdate + mark tier 2 so degrade hits the tier 2→3 path.
        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        conn.commit()
        conn.close()

        # Force the refresh to raise mid-call so we exercise the SAVEPOINT
        # rollback path. We patch on the instance so other beam instances
        # in the same test session aren't affected.
        def boom(*args, **kwargs):
            raise RuntimeError("simulated refresh failure")
        monkeypatch.setattr(beam, "_refresh_episodic_embedding", boom)

        result = beam.degrade_episodic(dry_run=False)
        # The row should NOT count as consolidated since the savepoint
        # rolled back.
        assert result["tier2_to_tier3"] == 0, (
            f"Refresh raised but row still counted as degraded: {result}"
        )

        # Critically: content must remain at the original (rollback worked),
        # NOT the truncated form. If the SAVEPOINT didn't roll back, the
        # UPDATE would have committed and we'd see truncated content + stale
        # embedding — exactly the C18.b drift.
        conn = sqlite3.connect(str(temp_db))
        post_content = conn.execute(
            "SELECT content, tier FROM episodic_memory WHERE id = ?", (memory_id,)
        ).fetchone()
        conn.close()
        assert post_content[0] == original, (
            f"SAVEPOINT did not roll back: content was mutated despite refresh "
            f"failure. Got {post_content[0][:60]!r}, expected original."
        )
        assert post_content[1] == 2, (
            f"SAVEPOINT did not roll back tier change: got tier={post_content[1]}"
        )

    def test_tier_2_to_tier_3_invalidates_when_provider_unavailable(
        self, temp_db, monkeypatch
    ):
        """If embeddings provider is unavailable at degrade time, the stale
        embedding rows must be invalidated so dense recall can't return
        semantically misleading hits."""
        from mnemosyne.core import embeddings as emb

        # Phase 1: provider available — seed.
        monkeypatch.setattr(emb, "available", lambda: True)
        monkeypatch.setattr(
            emb, "embed",
            lambda texts: np.stack([_content_to_vec(t) for t in texts]),
        )
        monkeypatch.setattr(beam_module, "_vec_available", lambda conn: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        original = ("ORIGINAL_DETAILED_CONTEXT " * 30).strip()
        memory_id = beam.consolidate_to_episodic(
            summary=original,
            source_wm_ids=["fake-wm"],
            importance=0.6,
        )
        assert _read_fallback_embedding(temp_db, memory_id) is not None

        # Phase 2: provider goes unavailable BEFORE degrade.
        monkeypatch.setattr(emb, "available", lambda: False)

        old_ts = (datetime.now() - timedelta(days=beam_module.TIER3_DAYS + 1)).isoformat()
        conn = sqlite3.connect(str(temp_db))
        conn.execute(
            "UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?",
            (old_ts, memory_id),
        )
        conn.commit()
        conn.close()

        beam.degrade_episodic(dry_run=False)

        post_embedding = _read_fallback_embedding(temp_db, memory_id)
        assert post_embedding is None, (
            "Stale memory_embeddings row remained after degrade with no embeddings "
            "provider. Should have been deleted to avoid ranking against content "
            "that no longer matches the row's text."
        )

        post_bv = _read_binary_vector(temp_db, memory_id)
        if beam_module._mib is not None:
            assert post_bv is None, (
                "binary_vector should be NULLed when the embedding provider is "
                "unavailable at degrade time"
            )
