"""Regression tests for E5 -- wire PolyphonicRecallEngine under feature flag.

Pre-E5: `PolyphonicRecallEngine` (mnemosyne/core/polyphonic_recall.py)
existed as a complete 4-voice + RRF + diversity-rerank + budget-aware
context-assembly implementation, but it was dead code -- no production
caller imported it. Commit 9f96ded's "polyphonic recall" was actually
inline graph_bonus + fact_bonus added to BeamMemory's linear scorer
(beam.py:2101-2156), not the engine.

Post-E5:
  - `MNEMOSYNE_POLYPHONIC_RECALL=1` activates the polyphonic engine
    inside `BeamMemory.recall()`
  - Default (flag unset or "0"): existing linear scorer runs unchanged --
    production behavior preserved
  - Flag ON: engine produces ranked candidates via RRF fusion across 4
    voices (vector + graph + fact + temporal), diversity-reranks them,
    and assembles context within budget. The inline graph_bonus and
    fact_bonus terms in the linear scorer are bypassed (engine handles
    those itself).
  - Per-result `voice_scores` field carries provenance -- operators can
    see WHICH voices contributed to a given ranking.
  - Engine reuses BeamMemory's shared sqlite connection rather than
    spawning 4+ new connections per recall call.

This unblocks the experiment's "strongest available recall layer" --
all arms run with flag=ON to test against the polyphonic engine
instead of the inline bonus shortcut.
"""

import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def disable_llm(monkeypatch):
    """Force deterministic non-LLM paths."""
    monkeypatch.setattr(
        "mnemosyne.core.local_llm.llm_available", lambda: False
    )


def _seed_recallable_content(beam, items):
    """Seed working_memory + episodic_memory with content that recall
    can find. items is a list of (content, source, importance) tuples."""
    for content, source, importance in items:
        beam.remember(content, source=source, importance=importance)


class TestE5FeatureFlag:

    def test_flag_off_uses_linear_scorer(self, temp_db, monkeypatch, disable_llm):
        """[E5] When MNEMOSYNE_POLYPHONIC_RECALL is unset or '0', recall
        runs the existing linear scorer. Production behavior unchanged.

        Post-Gap-G (May 2026): both engines now populate `voice_scores`
        for analysis parity, but with different keys. Linear keys are
        {vec, fts, keyword, importance, recency_decay}; polyphonic keys
        are {vector, graph, fact, temporal}. The signal that linear ran
        is the absence of polyphonic-specific keys, not the absence of
        the dict itself."""
        monkeypatch.delenv("MNEMOSYNE_POLYPHONIC_RECALL", raising=False)

        beam = BeamMemory(session_id="e5-off", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Alice mentioned the deploy timeline", "conversation", 0.7),
            ("Bob owns the auth refactor", "fact", 0.8),
        ])

        results = beam.recall("deploy", top_k=10)
        assert results, "recall returned 0 -- sanity check"
        _POLYPHONIC_KEYS = {"vector", "graph", "fact", "temporal"}
        for r in results:
            vs = r.get("voice_scores", {})
            assert not (set(vs.keys()) & _POLYPHONIC_KEYS), (
                f"linear path leaked polyphonic voice keys: {set(vs.keys())}"
            )

    def test_flag_off_explicit_zero(self, temp_db, monkeypatch, disable_llm):
        """Setting the env var to '0' is the explicit way to opt out;
        must behave identically to unset."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")

        beam = BeamMemory(session_id="e5-off-zero", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("explicit-zero content here", "test", 0.5),
        ])

        results = beam.recall("explicit-zero", top_k=10)
        _POLYPHONIC_KEYS = {"vector", "graph", "fact", "temporal"}
        for r in results:
            vs = r.get("voice_scores", {})
            assert not (set(vs.keys()) & _POLYPHONIC_KEYS)

    def test_flag_on_uses_polyphonic_engine(
        self, temp_db, monkeypatch, disable_llm
    ):
        """[E5] Flag ON: results carry voice_scores provenance -- the
        signal that the engine ran instead of the linear scorer.
        Per-signal observability is a hard requirement (the
        OpenViking-style 'show me which voice contributed').
        """
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")

        beam = BeamMemory(session_id="e5-on", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Alice deployed the auth service last Tuesday", "convo", 0.7),
            ("Bob filed a bug about the auth refactor", "convo", 0.7),
            ("Carol approved Alice's deploy plan", "convo", 0.8),
        ])

        results = beam.recall("Alice deploy auth", top_k=10)
        assert results, "polyphonic recall returned 0 -- engine wired wrong"
        # At least one result must carry voice_scores; an empty dict is
        # not enough (would mean RRF combined nothing). The engine's
        # RRF accumulates contributions from each contributing voice.
        any_with_voices = any(
            r.get("voice_scores") for r in results
        )
        assert any_with_voices, (
            f"no result carries voice_scores; engine didn't run or its "
            f"output wasn't mapped back. Got: {results}"
        )

    def test_flag_on_then_off_swaps_paths(
        self, temp_db, monkeypatch, disable_llm
    ):
        """The flag is read PER CALL, not at __init__ time -- operators
        can toggle the engine without rebuilding BeamMemory. Critical
        for A/B experiments inside the same process.

        Uses a query with a capitalized entity ('Alice') so the graph
        voice has something to bite on without requiring embeddings."""
        beam = BeamMemory(session_id="e5-toggle", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Alice asked about the toggle behavior", "convo", 0.6),
            ("Alice followed up with another comment", "convo", 0.6),
        ])

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        on_results = beam.recall("Alice toggle", top_k=10)

        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
        off_results = beam.recall("Alice toggle", top_k=10)

        # Post-Gap-G: both engines populate voice_scores with different
        # keys. Engine identity is the polyphonic-specific keys
        # (vector/graph/fact/temporal), not the field's mere presence.
        _POLYPHONIC_KEYS = {"vector", "graph", "fact", "temporal"}
        on_has_polyphonic_keys = any(
            (set(r.get("voice_scores", {}).keys()) & _POLYPHONIC_KEYS)
            for r in on_results
        )
        off_has_polyphonic_keys = any(
            (set(r.get("voice_scores", {}).keys()) & _POLYPHONIC_KEYS)
            for r in off_results
        )
        assert on_has_polyphonic_keys, (
            f"flag=ON didn't engage the engine. on_results={on_results}"
        )
        assert not off_has_polyphonic_keys, "flag=OFF still ran the engine"


class TestE5EnginePlumbing:

    def test_diversity_hydrates_source_content(self, temp_db):
        """Content Jaccard must use DB content, not a missing result attribute."""
        from mnemosyne.core.polyphonic_recall import (
            PolyphonicRecallEngine,
            RecallResult,
        )

        beam = BeamMemory(session_id="e5-content", db_path=temp_db)
        first_id = beam.remember("Alice deployed the auth service")
        second_id = beam.remember("Bob fixed the billing dashboard")
        engine = PolyphonicRecallEngine(db_path=temp_db, conn=beam.conn)
        combined = engine._combine_voices([
            RecallResult(first_id, 0.8, "vector", {}),
            RecallResult(second_id, 0.7, "vector", {}),
        ])

        engine._hydrate_result_content(combined)
        assert combined[first_id].content == "Alice deployed the auth service"
        assert combined[second_id].content == "Bob fixed the billing dashboard"
        assert engine._estimate_similarity(
            combined[first_id], combined[second_id]
        ) == pytest.approx(1 / 9)

    def test_diversity_handles_unmapped_ids_without_shared_connection(self, temp_db):
        """Standalone engines keep synthetic IDs harmless during reranking."""
        from mnemosyne.core.polyphonic_recall import (
            PolyphonicRecallEngine,
            RecallResult,
        )

        engine = PolyphonicRecallEngine(db_path=temp_db)
        combined = engine._combine_voices([
            RecallResult("synthetic:missing", 0.8, "graph", {}),
        ])
        engine._hydrate_result_content(combined)

        result = combined["synthetic:missing"]
        assert result.content == ""
        assert engine._estimate_similarity(result, result) == 0.0

    def test_engine_accepts_shared_connection(self, temp_db):
        """[E5 connection reuse] PolyphonicRecallEngine.__init__ must
        accept conn= so BeamMemory can share its thread-local
        connection. Without this each recall call would spawn multiple
        new SQLite connections (one per subsystem + one for the engine
        itself) -- wasteful under load and inconsistent with the
        post-9f96ded EpisodicGraph / VeracityConsolidator pattern.

        Post-E5.a: the standalone BinaryVectorStore subsystem was
        removed (vector voice now queries memory_embeddings directly),
        so the engine's connection plumbing surface shrank to
        graph + consolidator + the engine's own self.conn."""
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        # Bring up the schema via BeamMemory so the engine has tables to query.
        beam = BeamMemory(session_id="e5-conn", db_path=temp_db)
        shared_conn = beam.conn

        # The constructor must accept conn= without raising.
        engine = PolyphonicRecallEngine(db_path=temp_db, conn=shared_conn)

        # Verify the engine and its subsystems all use the shared
        # connection (this is the whole point of accepting it).
        assert engine.conn is shared_conn, (
            "engine self.conn is not the shared connection"
        )
        assert engine.graph.conn is shared_conn, (
            "graph is not using the shared connection"
        )
        assert engine.consolidator.conn is shared_conn, (
            "consolidator is not using the shared connection"
        )

    def test_engine_recall_handles_empty_corpus(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Flag ON against an empty DB: graceful empty return, no crash."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="e5-empty", db_path=temp_db)

        results = beam.recall("anything", top_k=10)
        # Empty list is acceptable; the contract is "doesn't crash" not
        # "returns hits when there are no rows."
        assert isinstance(results, list)

    def test_engine_recall_handles_no_embeddings(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Flag ON when embeddings are unavailable: vector voice
        returns empty, but graph/fact/temporal voices can still
        contribute. No crash."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        # Disable embeddings by monkeypatching available() to False.
        monkeypatch.setattr(
            "mnemosyne.core.embeddings.available", lambda: False
        )

        beam = BeamMemory(session_id="e5-noemb", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Alice did things yesterday", "convo", 0.6),
        ])

        # Use a temporal-keyword query so the temporal voice can
        # contribute even without entities matching the graph voice.
        results = beam.recall("yesterday Alice", top_k=10)
        # No crash. Results may or may not be non-empty depending on
        # which voices found matches.
        assert isinstance(results, list)


class TestE5FilterEnforcement:
    """[/review P1] Under flag=ON the engine path must enforce the
    same isolation/validity filters as the linear path. Pre-fix it
    returned rows globally, leaking cross-session and expired
    content."""

    def test_engine_path_isolates_by_session(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Two sessions, same DB. Recall from session A must NOT
        return session B's session-scoped rows."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam_a = BeamMemory(session_id="A", db_path=temp_db)
        beam_b = BeamMemory(session_id="B", db_path=temp_db)
        beam_a.remember("Alice talked about session A secret", source="conv", importance=0.7, scope="session")
        beam_b.remember("Bob talked about session B secret", source="conv", importance=0.7, scope="session")

        # Recall from A must not surface B's content.
        results = beam_a.recall("Bob", top_k=20)
        for r in results:
            assert "session B secret" not in (r.get("content") or ""), (
                f"engine path leaked session B content into session A "
                f"recall: {r}"
            )

    def test_engine_path_returns_global_scope_cross_session(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Global-scope rows MUST surface across sessions -- that's
        the design of `scope='global'`."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam_a = BeamMemory(session_id="A", db_path=temp_db)
        beam_b = BeamMemory(session_id="B", db_path=temp_db)
        beam_a.remember("Alice global preference for dark mode", source="pref", importance=0.9, scope="global")

        results = beam_b.recall("Alice", top_k=20)
        contents = [r.get("content", "") for r in results]
        assert any("dark mode" in c for c in contents), (
            f"global-scope row didn't surface in cross-session recall: "
            f"{contents}"
        )

    def test_engine_path_filters_superseded(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Rows with superseded_by set are tombstoned -- they must
        NOT surface in recall regardless of flag state."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        old_id = beam.remember("Alice prefers Vim", source="pref", importance=0.7)
        beam.conn.execute(
            "UPDATE working_memory SET superseded_by = ? WHERE id = ?",
            ("new-id", old_id),
        )
        beam.conn.commit()

        results = beam.recall("Alice Vim", top_k=20)
        for r in results:
            assert r["id"] != old_id, (
                f"engine path returned a superseded row: {r}"
            )

    def test_engine_path_filters_expired(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Rows whose valid_until has passed must not surface."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        past = (datetime.now() - timedelta(days=1)).isoformat()
        mid = beam.remember("Alice expired content", source="conv", importance=0.7)
        beam.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?",
            (past, mid),
        )
        beam.conn.commit()

        results = beam.recall("Alice", top_k=20)
        for r in results:
            assert r["id"] != mid, (
                f"engine path returned an expired row: {r}"
            )

    def test_engine_path_honors_author_filter(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Caller-supplied author_id filter must apply on engine path."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(
            session_id="s1", db_path=temp_db, author_id="alice"
        )
        beam.remember("Alice said hi", source="conv", importance=0.7)

        beam2 = BeamMemory(
            session_id="s1", db_path=temp_db, author_id="bob"
        )
        beam2.remember("Alice nodded", source="conv", importance=0.7)

        # Filter by author_id=alice: should not see bob's row.
        results = beam.recall("Alice", author_id="alice", top_k=20)
        for r in results:
            assert r.get("author_id") == "alice", (
                f"engine path ignored author_id filter: {r}"
            )


class TestE5MultiplierComposition:
    """[/review HIGH] E4 veracity multiplier + tier degradation
    multiplier must compose with the engine's RRF combined_score.
    Otherwise flag=ON erases the E4 work."""

    def test_veracity_multiplier_applies_on_engine_path(
        self, temp_db, monkeypatch, disable_llm
    ):
        """'stated' content should rank higher than 'unknown'
        content on the engine path, just as it does on the linear
        path post-E4."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        stated_id = beam.remember(
            "Alice prefers Vim editor", source="pref",
            importance=0.7, veracity="stated",
        )
        unknown_id = beam.remember(
            "Alice probably uses Vim editor", source="conv",
            importance=0.7, veracity="unknown",
        )

        results = beam.recall("Alice", top_k=20)
        scores = {r["id"]: r["score"] for r in results}
        if stated_id in scores and unknown_id in scores:
            assert scores[stated_id] > scores[unknown_id], (
                f"engine path didn't apply veracity multiplier; "
                f"stated={scores[stated_id]}, unknown={scores[unknown_id]}"
            )


class TestE5TelemetryUpdates:
    """[/review HIGH] recall_count / last_recalled must update on
    the engine path. Linear path updates them; not doing so under
    flag=ON silently breaks decay/usage signals."""

    def test_engine_path_increments_recall_count(
        self, temp_db, monkeypatch, disable_llm
    ):
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("Alice unique recall content", source="conv", importance=0.7)

        # Before recall: recall_count = 0.
        pre = sqlite3.connect(str(temp_db)).execute(
            "SELECT recall_count FROM working_memory WHERE id = ?",
            (mid,),
        ).fetchone()[0]
        assert (pre or 0) == 0

        results = beam.recall("Alice", top_k=10)
        # The row should be in results (with the unique Alice entity).
        ids = [r["id"] for r in results]
        assert mid in ids, f"row not in results: ids={ids}"

        # After: recall_count incremented, last_recalled set.
        post_row = sqlite3.connect(str(temp_db)).execute(
            "SELECT recall_count, last_recalled FROM working_memory "
            "WHERE id = ?", (mid,),
        ).fetchone()
        assert post_row[0] >= 1, (
            f"engine path did NOT increment recall_count; got {post_row[0]}"
        )
        assert post_row[1] is not None, (
            f"engine path did NOT set last_recalled; got {post_row[1]}"
        )


class TestE5EngineCache:
    """[/review maintainability] The engine should be lazily cached
    on the BeamMemory instance so subsystem constructors don't
    re-fire on every recall."""

    def test_engine_is_cached_between_calls(
        self, temp_db, monkeypatch, disable_llm
    ):
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Alice cached test", source="conv", importance=0.7)

        beam.recall("Alice", top_k=10)
        engine_first = beam._polyphonic_engine
        assert engine_first is not None, "engine wasn't cached after first call"

        beam.recall("Alice", top_k=10)
        engine_second = beam._polyphonic_engine
        assert engine_second is engine_first, (
            "engine instance changed between calls -- cache isn't holding"
        )


class TestE5ResultShape:

    def test_polyphonic_results_have_combined_score(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Flag ON: every result dict carries the RRF combined_score
        in the `score` field (existing shape) so downstream
        consumers don't need a special case."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="e5-shape", db_path=temp_db)
        _seed_recallable_content(beam, [
            ("Carol approved the launch", "convo", 0.8),
            ("Carol pushed back on the date", "convo", 0.7),
        ])

        results = beam.recall("Carol", top_k=10)
        for r in results:
            assert "score" in r, f"missing score field: {r}"
            assert isinstance(r["score"], (int, float))
            assert r["score"] > 0, f"non-positive score: {r}"

    def test_polyphonic_results_have_content(
        self, temp_db, monkeypatch, disable_llm
    ):
        """Flag ON: result dicts include the actual memory content,
        not just an id. Downstream consumers compose prompts from
        `content`, so the engine path must fetch and map it back from
        working_memory / episodic_memory."""
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="e5-content", db_path=temp_db)
        unique_token = "e5contentmarkerxyz"
        _seed_recallable_content(beam, [
            (f"row carrying {unique_token} content", "convo", 0.7),
        ])

        results = beam.recall(unique_token, top_k=10)
        # The engine may or may not surface the row depending on
        # whether any of its voices match (FTS isn't part of the
        # engine; entity extraction is the graph voice's only input).
        # If we get hits, they MUST have content.
        for r in results:
            assert r.get("content"), (
                f"result has no content; engine didn't map row back: {r}"
            )
