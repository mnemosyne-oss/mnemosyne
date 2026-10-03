"""Regression tests: bounded recall content cap (PR).

Covers the maintainer review demands:
- default stays 500 (public contract unchanged); env opts into a higher cap
- invalid/non-positive/empty env values fall back with a warning
- the cap is part of the enhanced-recall cache identity, so entries created
  under one cap are never served under another
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    """Observe the library default regardless of the deployment env OR of
    when beam was first imported: the module constant resolves at import,
    so reset both the env and the resolved constant. conftest.py also scrubs
    the env at module top (before any test module imports beam)."""
    monkeypatch.delenv("MNEMOSYNE_RECALL_CONTENT_CAP", raising=False)
    monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 500)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    return tmp_path / "content_cap.db"


class TestCapDefaults:
    def test_default_is_500(self):
        # Pins the SHIPPED default via a fresh subprocess import: the
        # autouse fixture assigns the module attribute before this test
        # runs, so comparing against in-process state would be vacuous
        # (it passes even if the shipped initializer changed). A clean
        # interpreter reads production beam.py with no env and no
        # fixture interference — fails under any default mutation.
        prog = (
            "import sys\n"
            f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
            "from mnemosyne.core import beam\n"
            "print(beam.RECALL_CONTENT_CAP)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", prog],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "MNEMOSYNE_NO_EMBEDDINGS": "1"},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "500"

    def test_shipped_default_mutation_is_caught(self, tmp_path):
        # Meta-test: prove the subprocess probe above is not vacuous.
        # Mutate the shipped initializer in a scratch copy the way a
        # regression would (default 999) and require the same probe to
        # FAIL there.
        import shutil

        scratch = tmp_path / "repo"
        shutil.copytree(
            _REPO_ROOT / "mnemosyne", scratch / "mnemosyne",
            ignore=shutil.ignore_patterns(
                "__pycache__", "*.pyc",
            ),
        )
        assert set(scratch.iterdir()) == {scratch / "mnemosyne"}
        init = scratch / "mnemosyne" / "core" / "beam.py"
        source = init.read_text()
        mutated = source.replace(
            'RECALL_CONTENT_CAP = _env_int("MNEMOSYNE_RECALL_CONTENT_CAP", 500)',
            'RECALL_CONTENT_CAP = _env_int("MNEMOSYNE_RECALL_CONTENT_CAP", 999)',
        )
        assert mutated != source, "shipped initializer pattern not found"
        init.write_text(mutated)
        prog = (
            "import sys\n"
            f"sys.path.insert(0, {str(scratch)!r})\n"
            "from mnemosyne.core import beam\n"
            "from pathlib import Path\n"
            f"assert str(Path(beam.__file__).resolve()) == {str(init)!r}\n"
            "print(beam.RECALL_CONTENT_CAP)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", prog],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "MNEMOSYNE_NO_EMBEDDINGS": "1"},
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == "999", (
            "probe must observe the shipped initializer, not fixture state"
        )

    def test_env_opt_in_raises(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_RECALL_CONTENT_CAP", "4000")
        assert beam_module._env_int("MNEMOSYNE_RECALL_CONTENT_CAP", 500) == 4000

    def test_invalid_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_RECALL_CONTENT_CAP", "garbage")
        assert beam_module._env_int("MNEMOSYNE_RECALL_CONTENT_CAP", 500) == 500

    def test_empty_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_RECALL_CONTENT_CAP", "")
        assert beam_module._env_int("MNEMOSYNE_RECALL_CONTENT_CAP", 500) == 500

    def test_zero_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_RECALL_CONTENT_CAP", "0")
        assert beam_module._env_int("MNEMOSYNE_RECALL_CONTENT_CAP", 500) == 500

    def test_fresh_import_zero_env_uses_default(self):
        assert self._fresh_import_cap("0") == "500"

    def test_negative_env_falls_back(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_RECALL_CONTENT_CAP", "-400")
        assert beam_module._env_int("MNEMOSYNE_RECALL_CONTENT_CAP", 500) == 500

    @staticmethod
    def _fresh_import_cap(env_value: str) -> str:
        """Import production beam.py in a clean interpreter with the env var
        set, and read the resolved module constant — catches regressions
        that hardcode the default or read a different env key (in-process
        _env_int tests cannot distinguish those)."""
        prog = (
            "import sys\n"
            f"sys.path.insert(0, {str(_REPO_ROOT)!r})\n"
            "from mnemosyne.core import beam\n"
            "print(beam.RECALL_CONTENT_CAP)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", prog],
            capture_output=True, text=True, timeout=120,
            env={
                **os.environ,
                "MNEMOSYNE_NO_EMBEDDINGS": "1",
                "MNEMOSYNE_RECALL_CONTENT_CAP": env_value,
            },
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip()

    def test_fresh_import_respects_opt_in(self):
        assert self._fresh_import_cap("4000") == "4000"

    def test_fresh_import_invalid_env_uses_default(self):
        assert self._fresh_import_cap("garbage") == "500"

    def test_fresh_import_empty_env_uses_default(self):
        assert self._fresh_import_cap("") == "500"

    def test_fresh_import_negative_env_uses_default(self):
        assert self._fresh_import_cap("-400") == "500"


class TestRecallTruncation:
    def test_default_cap_truncates_at_500(self, temp_db):
        beam = BeamMemory(session_id="cap", db_path=temp_db)
        long_mem = "Alice a un passe-temps particulier " + " ".join(f"detail{n}" for n in range(120)) + " fin souvenir"
        assert len(long_mem) > 500
        mid = beam.remember(long_mem)
        try:
            res = beam.recall("passe-temps particulier detail5", top_k=10)
            items = res if isinstance(res, list) else getattr(res, "results", [])
            row = next((it for it in items if "passe-temps particulier" in (it.get("content") or "")), None)
            assert row is not None
            assert len(row["content"]) == 500
        finally:
            beam.conn.execute("DELETE FROM working_memory WHERE id=?", (mid,))
            beam.conn.commit()

    def test_opt_in_cap_truncates_at_raised_boundary(self, temp_db, monkeypatch):
        # M3: content longer than the RAISED cap must truncate AT the cap
        # (a test with content between 500 and 4000 would pass even if the
        # truncation were completely broken).
        monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 4000)
        beam = BeamMemory(session_id="cap2", db_path=temp_db)
        long_mem = "Alice a un passe-temps particulier " + " ".join(f"detail{n}" for n in range(500)) + " fin souvenir"
        assert len(long_mem) > 4000
        mid = beam.remember(long_mem)
        try:
            res = beam.recall("passe-temps particulier detail5", top_k=10)
            items = res if isinstance(res, list) else getattr(res, "results", [])
            row = next((it for it in items if "passe-temps particulier" in (it.get("content") or "")), None)
            assert row is not None
            assert len(row["content"]) == 4000
        finally:
            beam.conn.execute("DELETE FROM working_memory WHERE id=?", (mid,))
            beam.conn.commit()


class TestBoundaryEnforcement:
    """P1: the cap must hold at the shared public-result boundary — every
    recall() exit path (linear, polyphonic, MEMORIA supplements, fact merge)
    — not only on the six linear producer slices."""

    LONG = "boundary cap probe content " + " ".join(f"seg{n}" for n in range(120)) + " end"

    def _seed(self, beam, monkeypatch, cap=100):
        monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", cap)
        assert len(self.LONG) > cap
        beam.remember(self.LONG, source="conversation", dedupe=False)

    def test_polyphonic_mapper_boundary(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="p1", db_path=temp_db)
        self._seed(beam, monkeypatch)
        mid = beam.conn.execute(
            "SELECT id FROM working_memory ORDER BY rowid DESC LIMIT 1"
        ).fetchone()["id"]
        row = beam.conn.execute(
            "SELECT id, content, source, timestamp, session_id, importance,"
            " recall_count, last_recalled, scope, author_id, author_type,"
            " channel_id, veracity, memory_type FROM working_memory WHERE id = ?",
            (mid,),
        ).fetchone()
        d = beam._polyphonic_row_to_dict(row, tier_label="working")
        assert len(d["content"]) == 100

    def test_polyphonic_fetch_boundary(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="p1f", db_path=temp_db)
        self._seed(beam, monkeypatch)
        mid = beam.conn.execute(
            "SELECT id FROM working_memory ORDER BY rowid DESC LIMIT 1"
        ).fetchone()["id"]
        d = beam._fetch_polyphonic_row(beam.conn.cursor(), mid)
        assert d is not None
        assert len(d["content"]) == 100

    def test_fact_merge_boundary(self, temp_db, monkeypatch):
        # The fact-voice merge path shares the boundary via recall()'s
        # return; force it by enabling fact recall and confirming no
        # over-cap content leaves recall() even when facts are long.
        monkeypatch.setenv("MNEMOSYNE_FACT_RECALL_ENABLED", "1")
        beam = BeamMemory(session_id="p3", db_path=temp_db)
        self._seed(beam, monkeypatch)
        res = beam.recall("boundary cap probe content", top_k=10)
        items = res if isinstance(res, list) else getattr(res, "results", [])
        assert items
        assert all(len(it.get("content", "")) <= 100 for it in items)

    def test_recall_explain_boundary(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="p4", db_path=temp_db)
        self._seed(beam, monkeypatch)
        res = beam.recall("boundary cap probe content", top_k=10, explain=True)
        results = res["results"]
        assert results
        assert all(len(it["content"]) <= 100 for it in results)

    def test_boundary_function_tolerates_non_results(self):
        # Contract: the boundary is a no-op on non-list payloads (defensive;
        # recall never returns these, but the helper must not crash if the
        # producers change shape).
        assert beam_module._cap_recall_results(None) is None
        assert beam_module._cap_recall_results("short") == "short"


class TestEnhancedCacheIdentity:
    def _key(self, beam, monkeypatch, cap_value):
        monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", cap_value)
        runtime = beam_module._cross_session_enabled()  # noqa: F841
        from types import SimpleNamespace
        runtime = SimpleNamespace(cross_session=False)
        return beam._enhanced_recall_cache_key(
            original_query="q", expanded_query="q", top_k=10, runtime=runtime,
            use_weibull=False, use_mmr=False, use_intent=False, use_synonyms=False,
            use_associative=False, associative_depth=0, mmr_lambda=0.5,
            recall_kwargs={},
        )

    def test_cap_changes_cache_identity(self, temp_db, monkeypatch):
        beam = BeamMemory(session_id="cache", db_path=temp_db)
        k500 = self._key(beam, monkeypatch, 500)
        k4000 = self._key(beam, monkeypatch, 4000)
        assert k500 != k4000, "entries under different caps must not cross-hit"


class TestFunctionalCapBoundaries:
    """Exercise public recall boundaries for polyphonic, fact-merge and
    persisted-cache paths, rather than only their internal helpers."""

    def _long_content(self, n: int) -> str:
        return ("x" * 10 + " ") * (n // 11 + 1)

    def test_polyphonic_recall_capped_normal_and_explain(self, temp_db, monkeypatch):
        # Stub the engine seam with over-cap content; the boundary at
        # recall() must cap the returned payload in BOTH modes.
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
        beam = BeamMemory(session_id="s-cap", db_path=temp_db)

        # Real rows (hydration by memory_id must find them); the raw stored
        # content exceeds the cap, so a faithful pipeline output is over-cap
        # BEFORE the boundary.
        raw = self._long_content(918)
        real_ids = [
            beam.remember(raw + f" part{i}", source="conversation")
            for i in range(3)
        ]

        class _FakeEngine:
            def recall(self, query, query_embedding=None, top_k=10, **kw):
                from mnemosyne.core.polyphonic_recall import PolyphonicResult
                return [
                    PolyphonicResult(
                        memory_id=rid, content=raw + f" part{i}",
                        combined_score=0.9 - i * 0.01,
                        voice_scores={"vector": 0.9 - i * 0.01},
                        metadata={},
                    )
                    for i, rid in enumerate(real_ids)
                ]

        monkeypatch.setattr(
            beam, "_get_polyphonic_engine", lambda: _FakeEngine()
        )

        results = beam.recall("long content probe", top_k=10)
        assert results, "polyphonic recall must return the stubbed rows"
        for r in results:
            assert len(r["content"]) <= beam_module.RECALL_CONTENT_CAP, (
                "polyphonic normal-mode payload must respect the cap"
            )

        explained = beam.recall("long content probe", top_k=10, explain=True)
        payload = explained if isinstance(explained, dict) else {"results": explained}
        items = payload.get("results") or []
        assert items, "explain mode must still surface the rows"
        for r in items:
            content = r.get("content", "") if isinstance(r, dict) else getattr(r, "content", "")
            assert len(content) <= beam_module.RECALL_CONTENT_CAP, (
                "polyphonic explain-mode payload must respect the cap"
            )

    def test_fact_merge_capped_with_tier(self, temp_db, monkeypatch):
        # Real path: facts table + MNEMOSYNE_FACT_RECALL_ENABLED=1; the row
        # must come back tier='fact' AND capped.
        monkeypatch.setenv("MNEMOSYNE_FACT_RECALL_ENABLED", "1")
        beam = BeamMemory(session_id="s-cap", db_path=temp_db)
        beam.conn.execute(
            "INSERT INTO facts (fact_id, session_id, subject, predicate, object,"
            " timestamp, confidence) VALUES (?,?,?,?,?,?,?)",
            (
                "fact-cap-1", beam.session_id, "Alice", "works at",
                "BigCorp " + "detail " * 120,
                "2026-09-05T12:00:00", 0.9,
            ),
        )
        beam.conn.commit()

        results = beam.recall("where does Alice work", top_k=10)
        fact_rows = [r for r in results if r.get("tier") == "fact"]
        assert fact_rows, "fact merge must surface the seeded row with tier='fact'"
        for r in fact_rows:
            assert len(r["content"]) <= beam_module.RECALL_CONTENT_CAP

    def test_persisted_cache_rejects_foreign_cap_entry(self, temp_db, monkeypatch):
        # Cap A entry persisted, cap B query on the same DB: key identity
        # must MISS (not serve the A entry), and the served content follows
        # cap B. The cache key reads the module attribute at call time.
        monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
        monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 5000)
        beam_a = BeamMemory(session_id="s-cap", db_path=temp_db)
        # Content must lexically match the query: recall legitimately
        # returns [] for non-matching stores, which would make the cap
        # assertions below vacuous.
        matching = (
            "cap identity probe notes: " + self._long_content(2000)
        )
        beam_a.remember(matching, source="conversation")
        warm = beam_a.recall_enhanced("cap identity probe", top_k=5)
        warm_results = warm.get("results") if isinstance(warm, dict) else warm
        assert warm_results, (
            "warm call must return the matching row — an empty warm result "
            "proves nothing about the cache"
        )

        cache_path = temp_db.parent / "query_cache.db"
        assert cache_path.exists(), (
            "warming recall_enhanced must persist query_cache.db — without "
            "this assertion the test passes vacuously (no cache involved)"
        )

        monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 100)
        beam_b = BeamMemory(session_id="s-cap", db_path=temp_db)
        out_b = beam_b.recall_enhanced("cap identity probe", top_k=5)
        results = out_b.get("results") if isinstance(out_b, dict) else out_b
        assert results, (
            "cap-B call must return the matching row (cache MISS by cap "
            "identity, recomputed at cap B)"
        )
        for r in results:
            assert len(r["content"]) <= 100, (
                "a persisted cap-A entry must never be served under cap B; "
                "content must be re-capped at B"
            )


@pytest.mark.parametrize("cap", [1, 500, 4000])
def test_enhanced_associative_cold_and_cached_content_cap(temp_db, monkeypatch, cap):
    from mnemosyne.core.episodic_graph import GraphEdge

    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", cap)
    memory = BeamMemory(session_id="cap-graph", db_path=temp_db)
    try:
        seed = memory.remember("quasar observatory " + "detail " * 100, dedupe=False)
        target = memory.remember("unrelated heliotrope", dedupe=False)
        edge_type = "rel" if cap == 1 else "related-" + "é星" * cap
        raw_content = f"[Associative: {edge_type}]"
        assert len(raw_content) > cap
        assert memory.episodic_graph is not None
        memory.episodic_graph.add_edge(
            GraphEdge(seed, target, edge_type, 0.8, "2026-01-01T00:00:00")
        )
        def query():
            return memory.recall_enhanced(
                "quasar observatory", top_k=1, use_associative=True,
                use_synonyms=False, use_weibull=False, use_mmr=False, use_intent=False,
            )

        cold = query()
        assert {r["id"] for r in cold} == {seed, target}
        assoc = next(r for r in cold if r["tier"] == "associative")
        assert assoc["content"] == raw_content[:cap]
        assert assoc["connecting_edge"] == edge_type
        assert all(len(r["content"]) <= cap for r in cold)

        hits = []
        original_get = memory._query_cache.get_opaque

        def get_cached(key):
            cached = original_get(key)
            hits.append((key, cached))
            return cached

        def no_recompute(*args, **kwargs):
            pytest.fail("warm recall must use the real persisted cache")

        monkeypatch.setattr(memory._query_cache, "get_opaque", get_cached)
        monkeypatch.setattr(memory, "recall", no_recompute)
        warm = query()
        assert warm == cold
        assert hits and hits[-1][1] == cold
        assert temp_db.with_name("query_cache.db").exists()
        # Inspect the serialized payload independently of the public hit cap.
        assert original_get(hits[-1][0]) == cold

        # A same-key oversized entry must also respect the public boundary.
        oversized = [dict(row, content="é星" * (cap + 1)) for row in cold]
        memory._query_cache.put_opaque(hits[-1][0], oversized)
        expected = [dict(row, content=row["content"][:cap]) for row in oversized]
        assert query() == expected
    finally:
        memory.conn.close()


@pytest.mark.parametrize("source", ["raw", "consolidated"])
def test_direct_fact_recall_respects_content_cap(temp_db, monkeypatch, source):
    """The public fact_recall surface must share the recall content bound."""
    monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 80)
    memory = BeamMemory(session_id="fact-cap", db_path=temp_db)
    subject = "Capstone"
    predicate = "records"
    obj = "boundary " + "detail " * 80
    try:
        if source == "raw":
            memory.conn.execute(
                "INSERT INTO facts (fact_id, session_id, subject, predicate, object, "
                "timestamp, source_msg_id, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "fact-cap-raw", memory.session_id, subject, predicate, obj,
                    "2026-01-01T00:00:00", "source-1", 0.9,
                ),
            )
        else:
            memory.conn.execute(
                "INSERT INTO consolidated_facts "
                "(id, subject, predicate, object, confidence, mention_count, "
                "first_seen, last_seen, sources_json, veracity) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "fact-cap-consolidated", subject, predicate, obj, 0.9, 1,
                    "2026-01-01T00:00:00", "2026-01-01T00:00:00", "[]", "stated",
                ),
            )
        memory.conn.commit()
        results = memory.fact_recall("capstone", top_k=10)
        assert len(results) == 1
        assert results[0]["content"] == f"{subject} {predicate} {obj}"[:80]
    finally:
        memory.conn.close()


def test_shmr_reflect_receives_capped_fact_recall_content(temp_db, monkeypatch):
    """SHMR's automatic fact retrieval must not re-expose an over-cap fact."""
    from mnemosyne.core import shmr

    monkeypatch.setattr(beam_module, "RECALL_CONTENT_CAP", 80)
    memory = BeamMemory(session_id="fact-cap-shmr", db_path=temp_db)
    obj = "private-tail-sentinel " + "detail " * 80
    memory.conn.execute(
        "INSERT INTO facts (fact_id, session_id, subject, predicate, object, "
        "timestamp, source_msg_id, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            "fact-cap-shmr", memory.session_id, "Capstone", "records", obj,
            "2026-01-01T00:00:00", "source-2", 0.9,
        ),
    )
    memory.conn.commit()
    prompts = []

    def capture(prompt):
        prompts.append(prompt)
        return "A sufficiently long synthesized answer."

    monkeypatch.setattr(shmr, "_call_llm", capture)
    try:
        assert shmr.reflect(memory, "capstone")
        assert len(prompts) == 1
        assert f"Capstone records {obj}"[:80] in prompts[0]
        assert "detail detail detail detail detail detail detail detail" not in prompts[0]
    finally:
        memory.conn.close()
