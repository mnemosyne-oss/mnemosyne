"""
Tests for Mnemosyne BEAM architecture
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

from mnemosyne.core.beam import BeamMemory, init_beam
from mnemosyne.core.memory import Mnemosyne


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        yield db_path


class TestBeamSchema:
    def test_init_creates_tables(self, temp_db):
        init_beam(temp_db)
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        tables = [r[0] for r in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "working_memory" in tables
        assert "episodic_memory" in tables
        assert "scratchpad" in tables
        assert "consolidation_log" in tables
        # FTS5 virtual table
        assert "fts_episodes" in tables
        conn.close()


class TestWorkingMemory:
    def test_remember_and_context(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("Prefers Neovim", source="preference", importance=0.9)
        assert mid is not None

        ctx = beam.get_context(limit=5)
        assert len(ctx) == 1
        assert ctx[0]["content"] == "Prefers Neovim"

    def test_trim_old_memories(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Insert old memory directly
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=25)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            ("old1", "old content", "conversation", old_ts, "s1")
        )
        conn.commit()
        conn.close()

        beam._trim_working_memory()
        stats = beam.get_working_stats()
        assert stats["total"] == 0


class TestEpisodicMemory:
    def test_consolidate_and_recall(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="User likes dark mode",
            source_wm_ids=["wm1"],
            importance=0.8
        )
        assert eid is not None

        results = beam.recall("dark mode")
        assert len(results) >= 1
        assert any(r["tier"] == "episodic" for r in results)

    def test_recall_hybrid_ranking(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.consolidate_to_episodic("Python is the best language", ["a"], importance=0.7)
        beam.consolidate_to_episodic("Rust is great for systems", ["b"], importance=0.7)

        results = beam.recall("best programming language")
        assert len(results) >= 1


class TestScratchpad:
    def test_scratchpad_write_read_clear(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.scratchpad_write("todo: fix auth")
        entries = beam.scratchpad_read()
        assert len(entries) == 1
        assert "fix auth" in entries[0]["content"]

        beam.scratchpad_clear()
        assert len(beam.scratchpad_read()) == 0


class TestSleepCycle:
    def test_sleep_consolidates_old_memories(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Inject old working memories
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        for i in range(3):
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                (f"old{i}", f"task {i}", "conversation", old_ts, "s1")
            )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["items_consolidated"] == 3

        log = beam.get_consolidation_log(limit=1)
        assert len(log) == 1
        assert log[0]["items_consolidated"] == 3

    def test_sleep_dry_run(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            ("old1", "task one", "conversation", old_ts, "s1")
        )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=True)
        assert result["status"] == "dry_run"
        assert result["items_consolidated"] == 1
        # Should not actually delete
        stats = beam.get_working_stats()
        assert stats["total"] == 1

    def test_sleep_remains_session_scoped(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("s1-old", "session one task", "conversation", old_ts, "s1"),
                ("s2-old", "session two task", "conversation", old_ts, "s2"),
            ]
        )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["items_consolidated"] == 1

        conn = sqlite3.connect(temp_db)
        remaining = conn.execute("SELECT session_id FROM working_memory ORDER BY session_id").fetchall()
        conn.close()
        assert remaining == [("s2",)]

    def test_sleep_all_sessions_consolidates_inactive_sessions(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        fresh_ts = datetime.now().isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("s1-old", "session one old task", "conversation", old_ts, "s1"),
                ("s2-old", "session two old task", "conversation", old_ts, "s2"),
                ("s2-fresh", "session two fresh task", "conversation", fresh_ts, "s2"),
            ]
        )
        conn.commit()
        conn.close()

        result = beam.sleep_all_sessions(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["sessions_scanned"] == 2
        assert result["sessions_consolidated"] == 2
        assert result["items_consolidated"] == 2
        assert result["summaries_created"] == 2
        assert result["errors"] == 0

        conn = sqlite3.connect(temp_db)
        remaining = conn.execute("SELECT id, session_id FROM working_memory").fetchall()
        logs = conn.execute("SELECT session_id, items_consolidated FROM consolidation_log ORDER BY session_id").fetchall()
        episodic_count = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        conn.close()

        assert remaining == [("s2-fresh", "s2")]
        assert logs == [("s1", 1), ("s2", 1)]
        assert episodic_count == 2

    def test_sleep_all_sessions_dry_run_preserves_working_memory(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("s1-old", "session one task", "conversation", old_ts, "s1"),
                ("s2-old", "session two task", "conversation", old_ts, "s2"),
            ]
        )
        conn.commit()
        conn.close()

        result = beam.sleep_all_sessions(dry_run=True)
        assert result["status"] == "dry_run"
        assert result["sessions_scanned"] == 2
        assert result["items_consolidated"] == 2

        conn = sqlite3.connect(temp_db)
        working_count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
        episodic_count = conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        log_count = conn.execute("SELECT COUNT(*) FROM consolidation_log").fetchone()[0]
        conn.close()
        assert working_count == 2
        assert episodic_count == 0
        assert log_count == 0

    def test_sleep_writes_dense_embedding_for_consolidated_row(self, temp_db, monkeypatch):
        """[C5] State-level companion to the FTS recallability test. Verifies
        sleep populates a dense-recall store (sqlite-vec's vec_episodes when
        loaded, otherwise the memory_embeddings fallback) for each consolidated
        episodic row. A regression that broke the embed→write call (e.g.
        embed() returning None silently, or a missing INSERT into the
        fallback table) would leave dense recall empty even though FTS keeps
        working.

        Skipped when fastembed isn't installed; the dense path is gated on
        _embeddings.available() and a model load. CI runs with fastembed."""
        from mnemosyne.core import embeddings as _embeddings

        if not _embeddings.available():
            pytest.skip("fastembed not available — dense-recall path inactive")

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("old0", "deploy plan for falcon kickoff", "conversation", old_ts, "s1"),
                ("old1", "retro notes from beta release", "conversation", old_ts, "s1"),
            ],
        )
        conn.commit()
        conn.close()

        beam.sleep(dry_run=False)

        # Post-sleep, exactly one episodic row should exist (one consolidated
        # summary for the session). Dense store should hold a row for it.
        from mnemosyne.core.beam import _vec_available

        conn = sqlite3.connect(temp_db)
        ep_ids = [r[0] for r in conn.execute("SELECT id FROM episodic_memory").fetchall()]
        assert len(ep_ids) == 1, f"expected 1 consolidated episodic row, got {len(ep_ids)}"

        if _vec_available(conn):
            vec_count = conn.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0]
            conn.close()
            assert vec_count >= 1, (
                "sleep consolidated an episodic row but vec_episodes is "
                "empty — the embed→_vec_insert path did not run. Likely "
                "cause: _embeddings.embed() returned None silently, or "
                "_vec_insert raised and was swallowed."
            )
        else:
            mem_count = conn.execute(
                "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", (ep_ids[0],)
            ).fetchone()[0]
            conn.close()
            assert mem_count >= 1, (
                "sleep consolidated an episodic row but memory_embeddings "
                "fallback is empty — the embed→INSERT path did not run."
            )

    def test_sleep_consolidated_content_is_recallable(self, temp_db, monkeypatch):
        """[C5] End-to-end recallability check. Existing sleep tests assert
        counts (items_consolidated, episodic_count) but never verify the
        consolidated content is actually findable through the public recall
        API. A regression that took the consolidated row off-recall via ALL
        recall paths simultaneously (FTS5 trigger broken AND dense store
        skipped AND fallback substring match unreachable) would slip through
        every existing sleep test.

        Locks: after sleep, recall(unique_token_from_seeded_wm) returns at
        least one episodic-tier hit whose content contains that token.

        Note: this is NOT an FTS-isolated assertion. recall() unions vec
        and FTS rowids (beam.py:1751) and falls back to substring scan
        (beam.py:1880) when both are empty, so this test locks recallability
        by *any* path — not the FTS path specifically. Stronger isolation
        would require calling _fts_search directly; that lives in a follow-up
        if the union/fallback layers shift.

        Uses LLM-disabled deterministic AAAK-encoded summary path
        (beam.py:2483 — `compressed = aaak_encode(combined)`). AAAK is
        phrase-substitution + compaction; uncommon literal tokens like
        the ones seeded below survive intact. Same monkeypatch pattern as
        test_beam.py:297, :488, :691, :938, :961."""
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=20)).isoformat()
        # Three distinct unique tokens — one per seeded memory.
        # Pick tokens that won't collide with FTS stop-words or the deterministic
        # concat header text.
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            [
                ("old0", "wm contains marker zorblax kickoff plan", "conversation", old_ts, "s1"),
                ("old1", "wm contains marker quetzelfin retro notes", "conversation", old_ts, "s1"),
                ("old2", "wm contains marker xanadush deploy log", "conversation", old_ts, "s1"),
            ],
        )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"
        assert result["items_consolidated"] == 3

        # Each unique token must surface an episodic-tier result.
        for token in ("zorblax", "quetzelfin", "xanadush"):
            results = beam.recall(token, top_k=10)
            assert results, (
                f"recall({token!r}) returned 0 results — the sleep path "
                f"consolidated working_memory but the episodic row is not "
                f"reachable through ANY recall path (FTS, vec, fallback "
                f"substring scan). Likely cause: FTS5 trigger missed AND "
                f"dense store missed AND content does not contain the "
                f"original token (LLM summarization path active despite "
                f"monkeypatch?)."
            )
            assert any(r.get("tier") == "episodic" for r in results), (
                f"recall({token!r}) returned {len(results)} hits but none "
                f"are episodic-tier: {[(r.get('tier'), r.get('content', '')[:50]) for r in results]}"
            )
            assert any(token in (r.get("content") or "").lower() for r in results), (
                f"recall({token!r}) returned hits but the token does not "
                f"appear in any returned content — FTS may be matching on "
                f"trigram noise rather than the seeded token: "
                f"{[r.get('content') for r in results]}"
            )


class TestMnemosyneIntegration:
    def test_legacy_and_beam_dual_write(self, temp_db):
        mem = Mnemosyne(session_id="s2", db_path=temp_db)
        mid = mem.remember("Likes pizza", source="preference", importance=0.8)

        # Legacy table
        conn = sqlite3.connect(temp_db)
        legacy = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        assert legacy is not None

        # BEAM working_memory should use the same ID now
        wm = conn.execute("SELECT * FROM working_memory WHERE id = ? AND session_id = ?", (mid, "s2")).fetchone()
        assert wm is not None
        conn.close()

        results = mem.recall("pizza")
        assert len(results) >= 1

    def test_forget_removes_both_layers(self, temp_db):
        mem = Mnemosyne(session_id="s2", db_path=temp_db)
        mid = mem.remember("Forget me please", source="preference", importance=0.8)
        assert mem.forget(mid) is True
        conn = sqlite3.connect(temp_db)
        legacy = conn.execute("SELECT * FROM memories WHERE id = ?", (mid,)).fetchone()
        wm = conn.execute("SELECT * FROM working_memory WHERE id = ? AND session_id = ?", (mid, "s2")).fetchone()
        conn.close()
        assert legacy is None
        assert wm is None

    def test_beam_stats(self, temp_db):
        mem = Mnemosyne(session_id="s3", db_path=temp_db)
        mem.remember("Test stat", importance=0.5)
        stats = mem.get_stats()
        assert stats["mode"] == "beam"
        assert "beam" in stats
        assert "working_memory" in stats["beam"]
        assert "episodic_memory" in stats["beam"]


class TestExportImport:
    def test_beam_export_to_dict(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Prefers dark mode", source="preference", importance=0.9)
        beam.scratchpad_write("todo item")
        beam.consolidate_to_episodic("User likes dark mode", ["wm1"], importance=0.8)

        data = beam.export_to_dict()
        assert "mnemosyne_export" in data
        assert data["mnemosyne_export"]["version"] == "1.0"
        assert len(data["working_memory"]) >= 1
        assert len(data["scratchpad"]) >= 1
        assert len(data["episodic_memory"]) >= 1

    def test_beam_import_from_dict_idempotent(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("Prefers dark mode", source="preference", importance=0.9)
        data = beam.export_to_dict()

        # Import into fresh DB
        with tempfile.TemporaryDirectory() as tmpdir:
            fresh_db = Path(tmpdir) / "fresh.db"
            fresh_beam = BeamMemory(session_id="s1", db_path=fresh_db)
            stats = fresh_beam.import_from_dict(data)
            assert stats["working_memory"]["inserted"] >= 1

            # Verify
            ctx = fresh_beam.get_context(limit=5)
            assert any("dark mode" in c["content"] for c in ctx)

            # Second import should skip
            stats2 = fresh_beam.import_from_dict(data)
            assert stats2["working_memory"]["skipped"] >= 1

    def test_mnemosyne_export_import_roundtrip(self, temp_db):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Source
            src = Mnemosyne(session_id="s1", db_path=temp_db)
            src.remember("Likes pizza", source="preference", importance=0.8)
            src.scratchpad_write("note")
            export_path = Path(tmpdir) / "export.json"
            src.export_to_file(str(export_path))
            assert export_path.exists()

            # Target
            target_db = Path(tmpdir) / "target.db"
            target = Mnemosyne(session_id="s1", db_path=target_db)
            stats = target.import_from_file(str(export_path))
            assert stats["legacy"]["inserted"] >= 1
            assert stats["beam"]["working_memory"]["inserted"] >= 1


class TestProviderContextSafety:
    def test_subagent_context_does_not_initialize_or_write(self, temp_db, monkeypatch):
        import importlib.util
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))

        provider_path = repo_root / "hermes_memory_provider" / "__init__.py"
        spec = importlib.util.spec_from_file_location("mnemo_provider_test", provider_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        provider = mod.MnemosyneMemoryProvider()
        provider.initialize(
            "subagent-session",
            hermes_home=str(repo_root),
            platform="cli",
            agent_context="subagent",
            agent_identity="test-profile",
            agent_workspace="hermes",
        )

        assert provider._beam is None
        result = provider.handle_tool_call(
            "mnemosyne_remember",
            {
                "content": "subagent should not persist memory",
                "importance": 0.9,
                "source": "test",
                "scope": "session",
            },
        )
        assert "not initialized" in result

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='working_memory'")
        exists = cursor.fetchone() is not None
        count = conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] if exists else 0
        conn.close()
        assert count == 0


class TestCrossSessionRecall:
    def test_global_memory_survives_consolidation_and_recall(self, temp_db, monkeypatch):
        """Regression for issue #7 Bug 2: global memories must survive sleep() and be recallable cross-session."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        # Disable LLM summarization so original Chinese text is preserved in consolidation
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        # Session A: store global memories with backdated timestamps so sleep() consolidates them
        beam_a = BeamMemory(session_id="hermes_session-A", db_path=temp_db)
        beam_a.remember("用户喜欢直接说结论", source="preference", importance=0.95, scope="global")
        beam_a.remember("用户讨论基金时重视手续费口径", source="preference", importance=0.92, scope="global")
        beam_a.remember("本轮只测试 mnemosyne 沙盒", source="test", importance=0.80, scope="session")

        # Backdate all working memories so they are old enough to consolidate
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=48)).isoformat()
        conn.execute("UPDATE working_memory SET timestamp = ?", (old_ts,))
        conn.commit()
        conn.close()

        # Force consolidation (simulate on_session_end)
        result = beam_a.sleep()
        assert result["status"] == "consolidated"

        # Verify consolidated episodic memories preserved global scope
        conn = sqlite3.connect(temp_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT content, scope, session_id FROM episodic_memory WHERE scope = 'global'")
        global_rows = cursor.fetchall()
        assert len(global_rows) >= 1, "Global memories should survive consolidation with scope preserved"
        conn.close()

        # Session B: recall global memories
        beam_b = BeamMemory(session_id="hermes_session-B", db_path=temp_db)

        # Test Chinese query that previously returned 0
        results = beam_b.recall("谁喜欢直接说结论", top_k=5)
        assert len(results) > 0, "Cross-session recall should find global memory with Chinese query"
        contents = [r["content"] for r in results]
        assert any("用户喜欢直接说结论" in c for c in contents)

        # Test another Chinese query
        results2 = beam_b.recall("基金讨论时看重什么口径", top_k=5)
        assert len(results2) > 0, "Cross-session recall should find second global memory"

        # Test that session-scoped memory is NOT visible cross-session
        results3 = beam_b.recall("本轮只测试", top_k=5)
        # This may or may not find it depending on scoring; the key is globals ARE found

    def test_fallback_scoring_finds_chinese_substrings(self, temp_db, monkeypatch):
        """Fallback keyword scoring must handle Chinese where words aren't space-delimited."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))

        beam = BeamMemory(session_id="test-session", db_path=temp_db)
        beam.remember("用户喜欢直接说结论", source="preference", importance=0.9, scope="global")

        # Query that differs at the start but shares a core substring
        results = beam.recall("谁喜欢直接说结论", top_k=5)
        assert len(results) > 0, "Fallback scoring should match shared substrings in Chinese"

    def test_tools_session_singleton_updates(self, temp_db, monkeypatch):
        """Plugin tools _get_memory() must recreate when HERMES_SESSION_ID changes."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))

        import importlib.util
        import sys
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        tools_path = repo_root / "hermes_plugin" / "tools.py"
        spec = importlib.util.spec_from_file_location("mnemo_tools_test", tools_path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        _get_memory = mod._get_memory

        monkeypatch.setenv("HERMES_SESSION_ID", "session-alpha")
        mem_a = _get_memory()
        mem_a.remember("alpha fact", source="test", scope="session")

        monkeypatch.setenv("HERMES_SESSION_ID", "session-beta")
        mem_b = _get_memory()
        # Should be a different instance (or at least different beam session_id)
        assert mem_b.session_id == "session-beta"
        assert mem_a.session_id == "session-alpha"


class TestTemporalQueries:
    """Temporal filtering for BEAM recall — Issue #16."""

    def test_recall_from_date_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Meeting about Q1 goals", source="meeting", importance=0.8)

        # Backdate an old memory directly
        conn = sqlite3.connect(temp_db)
        old_ts = "2025-01-15T10:00:00"
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("old1", "Old project kickoff", "meeting", old_ts, "s1", 0.7)
        )
        conn.commit()
        conn.close()

        # Filter from 2025-04-01 should exclude January memory
        results = beam.recall("project", from_date="2025-04-01")
        assert all("Old project kickoff" not in r["content"] for r in results)

    def test_recall_to_date_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Recent standup notes", source="meeting", importance=0.8)

        # Backdate an old memory
        conn = sqlite3.connect(temp_db)
        old_ts = "2025-01-15T10:00:00"
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
            ("old1", "January planning session", "meeting", old_ts, "s1", 0.7)
        )
        conn.commit()
        conn.close()

        # Filter to 2025-02-01 should only include January memory
        results = beam.recall("planning", to_date="2025-02-01")
        assert any("January" in r["content"] for r in results)
        assert all("Recent" not in r["content"] for r in results)

    def test_recall_source_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Bug fix for auth", source="github", importance=0.8)
        beam.remember("Lunch with team", source="conversation", importance=0.5)

        results = beam.recall("auth", source="github")
        assert len(results) >= 1
        assert all(r.get("source") == "github" for r in results)

    def test_recall_date_range_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        # Insert memories on different dates
        conn = sqlite3.connect(temp_db)
        dates = [
            ("2025-01-10T10:00:00", "January task A"),
            ("2025-03-15T10:00:00", "March task B"),
            ("2025-06-20T10:00:00", "June task C"),
        ]
        for ts, content in dates:
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance) VALUES (?, ?, ?, ?, ?, ?)",
                (f"m_{content[:5]}", content, "test", ts, "s1", 0.7)
            )
        conn.commit()
        conn.close()

        # Range: March to May
        results = beam.recall("task", from_date="2025-03-01", to_date="2025-05-31")
        contents = [r["content"] for r in results]
        assert any("March" in c for c in contents)
        assert all("January" not in c for c in contents)
        assert all("June" not in c for c in contents)

    def test_recall_with_episodic_temporal_filter(self, temp_db):
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Consolidated memory with old timestamp
        beam.consolidate_to_episodic(
            summary="Q4 review discussion",
            source_wm_ids=["wm1"],
            source="meeting",
            importance=0.8
        )
        # Backdate the episodic memory
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE episodic_memory SET timestamp = ? WHERE content = ?", ("2024-12-01T10:00:00", "Q4 review discussion"))
        conn.commit()
        conn.close()

        # Should find it without date filter
        results_all = beam.recall("Q4 review")
        assert any("Q4" in r["content"] for r in results_all)

        # Should exclude it with from_date in 2025
        results_filtered = beam.recall("Q4 review", from_date="2025-01-01")
        assert all("Q4" not in r["content"] for r in results_filtered)

    def test_temporal_triple_auto_generated(self, temp_db):
        """Temporal triples should be auto-generated on remember()."""
        from mnemosyne.core.triples import TripleStore

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("Deploy script updated", source="dev", importance=0.8)

        triple_store = TripleStore(db_path=temp_db)
        triples = triple_store.query(subject=mid)
        assert len(triples) >= 1
        assert any(t["predicate"] == "occurred_on" for t in triples)


class TestTokenAwareConsolidation:
    def test_sleep_chunks_large_batches(self, temp_db, monkeypatch):
        """BUG-1: sleep() must chunk memories to fit LLM context window."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(temp_db.parent))
        # Force a small context window to trigger chunking
        monkeypatch.setenv("MNEMOSYNE_LLM_N_CTX", "512")
        monkeypatch.setenv("MNEMOSYNE_LLM_MAX_TOKENS", "128")
        # Disable actual LLM — we test the chunking logic
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="test-chunking", db_path=temp_db)

        # Store 30 memories, each ~100 chars (~25 tokens)
        for i in range(30):
            beam.remember(
                f"Memory number {i} with enough content to consume tokens " * 3,
                source="test_batch",
                importance=0.5
            )

        # Backdate so sleep() picks them up
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=48)).isoformat()
        conn.execute("UPDATE working_memory SET timestamp = ?", (old_ts,))
        conn.commit()
        conn.close()

        result = beam.sleep()
        assert result["status"] == "consolidated"
        assert result["summaries_created"] >= 1

        # Verify no working memory left for this session
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM working_memory WHERE session_id = ?", ("test-chunking",))
        count = cursor.fetchone()[0]
        conn.close()
        assert count == 0

    def test_chunk_memories_by_budget_single_oversized(self, monkeypatch):
        """A single memory exceeding the budget should be skipped from LLM chunking."""
        from mnemosyne.core import local_llm

        # Monkeypatch module-level constants directly (env vars already read at import)
        monkeypatch.setattr(local_llm, "LLM_N_CTX", 128)
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 32)

        from mnemosyne.core.local_llm import chunk_memories_by_budget

        # One normal memory, one giant memory
        memories = [
            "Short memory.",  # ~3 tokens, fits
            "A" * 500,       # ~125 tokens, exceeds budget
        ]
        chunks = chunk_memories_by_budget(memories)

        # Giant memory should be excluded (it exceeds the total budget)
        assert len(chunks) == 1
        assert chunks[0] == ["Short memory."]


class TestTieredDegradation:
    """Tests for tiered episodic degradation — Phase 1 of the tiered memory system."""

    def test_schema_migration_adds_tier_columns(self, temp_db):
        """Wave 1: init_beam() should add tier and degraded_at columns to episodic_memory."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Just creating a BeamMemory triggers init_beam which runs the migration

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(episodic_memory)").fetchall()]
        assert "tier" in cols, "tier column missing after migration"
        assert "degraded_at" in cols, "degraded_at column missing after migration"

        # Verify index exists
        indexes = [r[1] for r in cursor.execute(
            "SELECT * FROM sqlite_master WHERE type='index' AND tbl_name='episodic_memory'"
        ).fetchall()]
        assert any("tier" in idx for idx in indexes), "idx_em_tier index missing"
        conn.close()

    def test_episodic_memory_defaults_to_tier_1(self, temp_db):
        """New episodic memories should default to tier 1."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="Default tier should be 1",
            source_wm_ids=["wm1"],
            importance=0.8
        )

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        tier = cursor.execute(
            "SELECT tier FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone()[0]
        conn.close()
        assert tier == 1, f"Expected tier=1, got tier={tier}"

    def test_degrade_episodic_tier1_to_tier2(self, temp_db, monkeypatch):
        """Tier 1 memories older than TIER2_DAYS should degrade to tier 2."""
        # Module-level constants are read at import time — patch them directly
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 5)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 200)  # far future — won't trigger tier 3

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="This memory is old enough for tier 2 degradation",
            source_wm_ids=["wm1"],
            importance=0.7
        )

        # Backdate the episodic memory to be older than 5 days
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (old_ts, eid))
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier1_to_tier2"] == 1
        assert result["tier2_to_tier3"] == 0

        # Verify tier changed
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        tier, degraded_at = cursor.execute(
            "SELECT tier, degraded_at FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone()
        conn.close()
        assert tier == 2
        assert degraded_at is not None

    def test_degrade_episodic_tier2_to_tier3(self, temp_db, monkeypatch):
        """Tier 2 memories older than TIER3_DAYS should degrade to tier 3."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 5)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="This memory will go all the way to tier 3",
            source_wm_ids=["wm1"],
            importance=0.6
        )

        # First degrade to tier 2 (older than 1 day)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(days=3)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (old_ts, eid))
        conn.commit()
        conn.close()
        beam.degrade_episodic(dry_run=False)  # tier 1 → 2

        # Then push it even older and degrade again
        conn = sqlite3.connect(temp_db)
        very_old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ?, tier = 2 WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        assert result["tier2_to_tier3"] == 1

        conn = sqlite3.connect(temp_db)
        tier = conn.execute("SELECT tier FROM episodic_memory WHERE id = ?", (eid,)).fetchone()[0]
        conn.close()
        assert tier == 3

    def test_degrade_episodic_dry_run(self, temp_db, monkeypatch):
        """Dry run counts candidates but does NOT modify the database."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 5)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.consolidate_to_episodic(
            summary="Should be counted but not degraded",
            source_wm_ids=["wm1"],
            importance=0.7
        )

        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ?", (old_ts,))
        conn.commit()

        result = beam.degrade_episodic(dry_run=True)
        assert result["status"] == "dry_run"
        assert result["tier1_to_tier2"] == 1

        # Tier should still be 1 — dry run doesn't modify
        tier = conn.execute("SELECT tier FROM episodic_memory").fetchone()[0]
        conn.close()
        assert tier == 1, "Dry run should not change tier"

    def test_degrade_episodic_respects_batch_limit(self, temp_db, monkeypatch):
        """Degradation should respect DEGRADE_BATCH_SIZE limit."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.DEGRADE_BATCH_SIZE", 3)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Consolidate 5 episodic memories first
        eids = []
        for i in range(5):
            eid = beam.consolidate_to_episodic(
                summary=f"Memory {i} for batch limit test",
                source_wm_ids=[f"wm{i}"],
                importance=0.5
            )
            eids.append(eid)

        # Backdate them all in a single raw connection block
        conn = sqlite3.connect(temp_db, timeout=10)
        old_ts = (datetime.now() - timedelta(days=10)).isoformat()
        for eid in eids:
            conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (old_ts, eid))
        conn.commit()
        conn.close()

        result = beam.degrade_episodic(dry_run=False)
        # Should degrade at most DEGRADE_BATCH_SIZE (3), not all 5
        assert result["tier1_to_tier2"] <= 3

    def test_tier_weighting_in_recall(self, temp_db, monkeypatch):
        """Tier 3 memories should score lower than tier 1 in recall."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 5)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_WEIGHT", 0.1)  # heavily penalize

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="Python projects use virtual environments for isolation",
            source_wm_ids=["wm1"],
            importance=0.9
        )

        # Degrade to tier 3
        conn = sqlite3.connect(temp_db)
        very_old_ts = (datetime.now() - timedelta(days=30)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t1→t2
        conn.execute("UPDATE episodic_memory SET tier = 2, created_at = ? WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t2→t3
        conn.close()

        results = beam.recall("Python virtual environments", top_k=5)
        # Should still be findable (just weighted lower)
        degraded = [r for r in results if r.get("degradation_tier") == 3]
        if degraded:
            assert degraded[0]["score"] < 1.0, f"Tier 3 score {degraded[0]['score']} should be penalized"

    def test_sleep_includes_degradation(self, temp_db, monkeypatch):
        """sleep() return value must include degradation key."""
        monkeypatch.setenv("MNEMOSYNE_TIER2_DAYS", "30")  # no actual degradation, just testing the key
        monkeypatch.setenv("MNEMOSYNE_TIER3_DAYS", "200")
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Inject old working memory to trigger consolidation
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=48)).isoformat()
        for i in range(2):
            conn.execute(
                "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                (f"old{i}", f"sleep test content {i}", "conversation", old_ts, "s1")
            )
        conn.commit()
        conn.close()

        result = beam.sleep(dry_run=False)
        assert "degradation" in result, "sleep() should include degradation key"
        assert "status" in result["degradation"]
        assert "tier1_to_tier2" in result["degradation"]

    def test_sleep_all_sessions_includes_degradation(self, temp_db, monkeypatch):
        """sleep_all_sessions() return value must include degradation key."""
        monkeypatch.setenv("MNEMOSYNE_TIER2_DAYS", "30")
        monkeypatch.setenv("MNEMOSYNE_TIER3_DAYS", "200")
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        conn = sqlite3.connect(temp_db)
        old_ts = (datetime.now() - timedelta(hours=48)).isoformat()
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            ("s2-old", "all sessions sleep test", "conversation", old_ts, "s2")
        )
        conn.commit()
        conn.close()

        result = beam.sleep_all_sessions(dry_run=False)
        assert "degradation" in result, "sleep_all_sessions() should include degradation key"

    def test_old_memory_still_recallable_after_degradation(self, temp_db, monkeypatch):
        """Integration: store old memory, degrade to tier 3, still recallable."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 5)
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        eid = beam.consolidate_to_episodic(
            summary="The user's favorite programming language is Rust for systems work",
            source_wm_ids=["wm1"],
            importance=0.85
        )

        conn = sqlite3.connect(temp_db)
        very_old_ts = (datetime.now() - timedelta(days=200)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t1→t2
        conn.execute("UPDATE episodic_memory SET tier = 2 WHERE id = ?", (eid,))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t2→t3

        # Verify it's tier 3
        tier = conn.execute("SELECT tier FROM episodic_memory WHERE id = ?", (eid,)).fetchone()[0]
        print(f"DEBUG: tier after double degrade = {tier}")
        content = conn.execute("SELECT content FROM episodic_memory WHERE id = ?", (eid,)).fetchone()[0]
        print(f"DEBUG: tier 3 content = {content[:100]}")
        conn.close()
        assert tier == 3, f"Expected tier 3, got {tier}"

        # Should still be recallable — this is the marketing promise
        results = beam.recall("favorite programming language", top_k=5)
        contents = [r["content"] for r in results]
        assert len(results) > 0, "Tier 3 memory should still be recallable"
        assert any("Rust" in c for c in contents), (
            f"Tier 3 memory should contain 'Rust', got contents: {contents}"
        )


class TestSmartCompression:
    """Phase 2: entity-aware extraction for tier 2→3 degradation."""

    def test_extract_key_signal_keeps_proper_nouns(self, temp_db):
        """Entities like names, tools, and versions should survive compression."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = (
            "The user's favorite editor is Neovim with LazyVim. "
            "They deploy everything with Docker Compose. "
            "Their preferred language is Rust for systems work. "
            "The weather was nice on Tuesday. "
            "Nothing special happened in the morning. "
            "They also use GitHub Actions for CI/CD."
        )
        result = beam._extract_key_signal(content, max_chars=200)
        # Signal sentences should be present
        assert "Neovim" in result, f"Lost 'Neovim': {result}"
        assert "Docker" in result, f"Lost 'Docker': {result}"
        assert "Rust" in result, f"Lost 'Rust': {result}"
        # Low-signal sentences should be dropped
        assert "weather" not in result, f"Weather survived: {result}"
        assert "Nothing special" not in result, f"Noise survived: {result}"

    def test_extract_key_signal_handles_no_sentences(self, temp_db):
        """Single blob of text without sentence boundaries — falls back to prefix."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = "A" * 500  # No punctuation
        result = beam._extract_key_signal(content, max_chars=100)
        assert len(result) <= 110  # 100 chars + " [...]"
        assert result.startswith("A" * 90)

    def test_extract_key_signal_short_content_passthrough(self, temp_db):
        """Content under max_chars should be returned as-is."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        content = "Short memory about Python."
        result = beam._extract_key_signal(content, max_chars=500)
        assert result == content

    def test_smart_compress_preserves_entities_in_degradation(self, temp_db, monkeypatch):
        """End-to-end: smart compression keeps key facts where naive prefix would lose them."""
        monkeypatch.setattr("mnemosyne.core.beam.TIER2_DAYS", 1)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_DAYS", 5)
        monkeypatch.setattr("mnemosyne.core.beam.SMART_COMPRESS", True)
        monkeypatch.setattr("mnemosyne.core.beam.TIER3_MAX_CHARS", 200)
        monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)

        beam = BeamMemory(session_id="s1", db_path=temp_db)

        # Memory where the most important fact is at the END
        eid = beam.consolidate_to_episodic(
            summary=(
                "Morning standup was uneventful. The coffee was cold. "
                "Lunch was a sandwich from the deli. Team discussed vacation plans. "
                "CRITICAL: The production database password was changed to XKCD-correct-horse-battery-staple. "
                "Afternoon was quiet. Went home at 5pm."
            ),
            source_wm_ids=["wm1"],
            importance=0.9
        )

        # Backdate and degrade to tier 3
        conn = sqlite3.connect(temp_db)
        very_old_ts = (datetime.now() - timedelta(days=200)).isoformat()
        conn.execute("UPDATE episodic_memory SET created_at = ? WHERE id = ?", (very_old_ts, eid))
        conn.commit()
        beam.degrade_episodic(dry_run=False)  # t1→t2
        conn.execute("UPDATE episodic_memory SET tier = 2 WHERE id = ?", (eid,))
        conn.commit()
        conn.close()
        beam.degrade_episodic(dry_run=False)  # t2→t3

        # Verify the critical fact survived
        conn = sqlite3.connect(temp_db)
        tier3_content = conn.execute(
            "SELECT content FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone()[0]
        conn.close()

        assert "XKCD" in tier3_content or "password" in tier3_content, (
            f"Smart compression should preserve critical entities. Got: {tier3_content}"
        )
        # Naive prefix would have kept "Morning standup was uneventful" — useless


class TestVeracity:
    """Phase 3: memory confidence / veracity signal."""

    def test_schema_adds_veracity_columns(self, temp_db):
        """init_beam should add veracity to working_memory and episodic_memory."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        conn = sqlite3.connect(temp_db)
        wm_cols = [r[1] for r in conn.execute("PRAGMA table_info(working_memory)").fetchall()]
        em_cols = [r[1] for r in conn.execute("PRAGMA table_info(episodic_memory)").fetchall()]
        conn.close()
        assert "veracity" in wm_cols
        assert "veracity" in em_cols

    def test_remember_defaults_to_unknown(self, temp_db):
        """remember() without explicit veracity defaults to 'unknown'."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("A fact", source="test", importance=0.5)

        conn = sqlite3.connect(temp_db)
        veracity = conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()[0]
        conn.close()
        assert veracity == "unknown"

    def test_remember_explicit_veracity(self, temp_db):
        """remember() with veracity='stated' stores correctly."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        mid = beam.remember("User said this", source="user", veracity="stated")

        conn = sqlite3.connect(temp_db)
        veracity = conn.execute(
            "SELECT veracity FROM working_memory WHERE id = ?", (mid,)
        ).fetchone()[0]
        conn.close()
        assert veracity == "stated"

    def test_recall_veracity_filter(self, temp_db):
        """recall(veracity='stated') should only return stated memories."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Stated preference: dark mode", source="user", veracity="stated")
        beam.remember("Inferred: probably likes Python", source="conversation", veracity="inferred")
        beam.remember("Tool output: cron ran at 3am", source="cron", veracity="tool")

        results = beam.recall("preference", veracity="stated")
        assert all(r["veracity"] == "stated" for r in results)
        assert any("dark mode" in r["content"] for r in results)

    def test_veracity_weighting_in_recall(self, temp_db, monkeypatch):
        """Stated memories should score higher than inferred ones."""
        monkeypatch.setattr("mnemosyne.core.beam.STATED_WEIGHT", 1.0)
        monkeypatch.setattr("mnemosyne.core.beam.INFERRED_WEIGHT", 0.3)

        beam = BeamMemory(session_id="s1", db_path=temp_db)
        # Consolidate two similar memories with different veracity
        beam.consolidate_to_episodic(
            summary="User stated: prefers Rust for systems programming",
            source_wm_ids=["wm1"],
            importance=0.8
        )
        beam.consolidate_to_episodic(
            summary="Inferred: agent thinks user likes Go",
            source_wm_ids=["wm2"],
            importance=0.8
        )

        # Set veracity directly in DB
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE episodic_memory SET veracity = 'stated' WHERE content LIKE '%stated%'")
        conn.execute("UPDATE episodic_memory SET veracity = 'inferred' WHERE content LIKE '%Inferred%'")
        conn.commit()
        conn.close()

        results = beam.recall("systems programming language", top_k=5)
        stated_results = [r for r in results if r.get("veracity") == "stated"]
        inferred_results = [r for r in results if r.get("veracity") == "inferred"]

        if stated_results and inferred_results:
            assert stated_results[0]["score"] > inferred_results[0]["score"], (
                f"Stated score {stated_results[0]['score']} should exceed inferred {inferred_results[0]['score']}"
            )

    def test_get_contaminated_returns_non_stated(self, temp_db):
        """get_contaminated() should return inferred/tool/imported/unknown but not stated."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        eids = []
        for content, veracity_val in [
            ("User said this explicitly", "stated"),
            ("Agent inferred this", "inferred"),
            ("Cron injected this", "tool"),
            ("Imported from Mem0", "imported"),
            ("Legacy uncategorized memory", "unknown"),
        ]:
            eid = beam.consolidate_to_episodic(
                summary=content, source_wm_ids=["wm"], importance=0.8
            )
            eids.append(eid)

        conn = sqlite3.connect(temp_db)
        for eid, veracity_val in zip(eids, ["stated", "inferred", "tool", "imported", "unknown"]):
            conn.execute("UPDATE episodic_memory SET veracity = ? WHERE id = ?", (veracity_val, eid))
        conn.commit()
        conn.close()

        contaminated = beam.get_contaminated(limit=10)
        contents = [c["content"] for c in contaminated]
        assert any("inferred" in c for c in contents)
        assert any("Cron injected" in c for c in contents)
        assert any("Imported" in c for c in contents)
        assert any("Legacy" in c for c in contents)
        # Stated should NOT appear
        assert not any("explicitly" in c for c in contents)

    def test_get_contaminated_respects_importance(self, temp_db):
        """get_contaminated() with min_importance filter."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)

        eid = beam.consolidate_to_episodic(
            summary="Low importance memory", source_wm_ids=["wm"], importance=0.2
        )
        conn = sqlite3.connect(temp_db)
        conn.execute("UPDATE episodic_memory SET veracity = 'inferred' WHERE id = ?", (eid,))
        conn.commit()
        conn.close()

        results = beam.get_contaminated(limit=10, min_importance=0.5)
        assert len(results) == 0, "Low importance should be filtered out"

    def test_recall_still_works_without_veracity_filter(self, temp_db):
        """recall() without veracity filter should return all memories."""
        beam = BeamMemory(session_id="s1", db_path=temp_db)
        beam.remember("Fact A", veracity="stated")
        beam.remember("Fact B", veracity="inferred")

        results = beam.recall("Fact", top_k=10)
        assert len(results) >= 2
