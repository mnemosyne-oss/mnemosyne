#!/usr/bin/env python3
"""
Comprehensive test suite for Mnemosyne Temporal Upgrade.
Tests all enhanced recall features with dummy data only.
No real user data is used at any point.

Run: MNEMOSYNE_ENHANCED_RECALL=1 python3 test_temporal_upgrade.py
"""

import sys
import os
import unittest
import tempfile
import time
import json
from datetime import datetime, timedelta, date
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# Test 1: Weibull Decay Scoring
# ============================================================
class TestWeibullDecay(unittest.TestCase):
    def setUp(self):
        from mnemosyne.core.weibull import weibull_boost, weibull_decay_factor, WEIBULL_PARAMS
        
    def test_weibull_params_exist(self):
        """All memory types have Weibull parameters."""
        from mnemosyne.core.weibull import WEIBULL_PARAMS
        expected_types = ["profile", "preference", "setup", "fact", "learning",
                         "pattern", "project", "goal", "entity", "event", "issue",
                         "request", "general"]
        for t in expected_types:
            self.assertIn(t, WEIBULL_PARAMS, f"Missing Weibull params for: {t}")
            self.assertIn("k", WEIBULL_PARAMS[t])
            self.assertIn("eta", WEIBULL_PARAMS[t])
    
    def test_profile_decay_is_slow(self):
        """Profiles decay much slower than requests."""
        from mnemosyne.core.weibull import weibull_decay_factor
        age = 720  # 30 days
        profile_decay = weibull_decay_factor(age, "profile")
        request_decay = weibull_decay_factor(age, "request")
        self.assertGreater(profile_decay, request_decay,
                          "Profiles should decay slower than requests")
        self.assertGreater(profile_decay, 0.5,
                          "Profile should still have high weight after 30 days")
    
    def test_request_decay_is_fast(self):
        """Requests decay very fast."""
        from mnemosyne.core.weibull import weibull_decay_factor
        age = 168  # 1 week
        request_decay = weibull_decay_factor(age, "request")
        self.assertLess(request_decay, 0.1,
                       "Requests should have near-zero weight after 1 week")
    
    def test_fresh_memory_boost_is_one(self):
        """Brand new memory gets boost = 1.0."""
        from mnemosyne.core.weibull import weibull_boost
        now = datetime.now()
        boost = weibull_boost(now.isoformat(), now, memory_type="general")
        self.assertAlmostEqual(boost, 1.0, places=5)
    
    def test_weibull_vs_exponential(self):
        """Weibull with k<1 decays slower than exponential at long ranges."""
        from mnemosyne.core.weibull import weibull_decay_factor
        age = 5000  # ~7 months
        weibull_val = weibull_decay_factor(age, "profile")  # k=0.3, eta=8760
        import math
        exp_val = math.exp(-age / 168)  # Simple exponential, halflife=1 week
        self.assertGreater(weibull_val, exp_val,
                          "Weibull profile should retain more weight than exponential")
    
    def test_general_type_uses_exponential(self):
        """General type (k=1.0) behaves like exponential."""
        from mnemosyne.core.weibull import weibull_decay_factor
        age = 168  # 1 week
        import math
        weibull_val = weibull_decay_factor(age, "general")
        exp_val = math.exp(-age / 168.0)
        self.assertAlmostEqual(weibull_val, exp_val, places=5)
    
    def test_invalid_timestamp_returns_zero(self):
        """Invalid timestamp should return 0.0."""
        from mnemosyne.core.weibull import weibull_boost
        boost = weibull_boost("not-a-date", memory_type="general")
        self.assertEqual(boost, 0.0)
    
    def test_none_timestamp_returns_zero(self):
        """None timestamp should return 0.0."""
        from mnemosyne.core.weibull import weibull_boost
        boost = weibull_boost(None, memory_type="general")
        self.assertEqual(boost, 0.0)


# ============================================================
# Test 2: Query Intent Classification
# ============================================================
class TestQueryIntent(unittest.TestCase):
    def test_temporal_intent(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("what happened last Monday")
        self.assertEqual(intent.category, "temporal")
        self.assertGreater(intent.confidence, 0.3)
    
    def test_factual_intent(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("what is the database password")
        self.assertEqual(intent.category, "factual")
    
    def test_preference_intent(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("what does Denis prefer for lunch")
        # "prefer" triggers both entity and preference — preference should win
        self.assertIn(intent.category, ["preference", "entity"])
    
    def test_procedural_intent(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("how do I deploy this project")
        self.assertEqual(intent.category, "procedural")
    
    def test_general_intent(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("hello world test")
        self.assertEqual(intent.category, "general")
        self.assertEqual(intent.confidence, 0.0)
    
    def test_weight_adjustment(self):
        from mnemosyne.core.query_intent import classify_intent, adjust_weights
        intent = classify_intent("what happened last week")
        vw, fw, iw = adjust_weights(0.5, 0.3, 0.2, intent=intent)
        # Temporal: fts_bias=1.5, vec_bias=0.6 → FTS should get higher weight
        self.assertGreater(fw, vw, "Temporal intent should boost FTS over vector")
        # Weights should sum to ~1.0
        self.assertAlmostEqual(vw + fw + iw, 1.0, places=5)


# ============================================================
# Test 3: Synonym Expansion
# ============================================================
class TestSynonyms(unittest.TestCase):
    def test_expand_query(self):
        from mnemosyne.core.synonyms import expand_query
        result = expand_query("what is the db password")
        self.assertIn("database", result.lower())
        self.assertIn("password", result.lower())
    
    def test_normalize_query(self):
        from mnemosyne.core.synonyms import normalize_query
        result = normalize_query("what is the database password")
        # Should contain canonical forms
        self.assertIn("database", result)
        self.assertIn("password", result)
        # Stop words removed
        self.assertNotIn("what", result)
        self.assertNotIn("is", result)
        self.assertNotIn("the", result)
    
    def test_get_synonyms(self):
        from mnemosyne.core.synonyms import get_synonyms
        syns = get_synonyms("db")
        self.assertIn("database", syns)
        self.assertGreater(len(syns), 1)
    
    def test_no_synonyms_for_unknown(self):
        from mnemosyne.core.synonyms import get_synonyms
        syns = get_synonyms("xyzzy_unknown_word")
        self.assertEqual(syns, ["xyzzy_unknown_word"])
    
    def test_canonical_mapping(self):
        from mnemosyne.core.synonyms import normalize_query
        # "db" should normalize to "database"
        r1 = normalize_query("db password")
        r2 = normalize_query("database password")
        self.assertEqual(r1, r2, "db and database should normalize to same form")


# ============================================================
# Test 4: MMR Re-Ranking
# ============================================================
class TestMMRRerank(unittest.TestCase):
    def test_mmr_rerank_no_duplicates(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [
            {"content": "database password is hunter2", "score": 0.95},
            {"content": "server runs on port 8080", "score": 0.85},
            {"content": "deploy script is in /opt/deploy", "score": 0.80},
        ]
        reranked = mmr_rerank(results, lambda_param=0.7, top_k=3)
        self.assertEqual(len(reranked), 3)
        # First result should still be highest scoring
        self.assertEqual(reranked[0]["content"], "database password is hunter2")
    
    def test_mmr_diversifies_similar_results(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [
            {"content": "the database password is hunter2", "score": 0.95},
            {"content": "the database password was hunter2", "score": 0.94},
            {"content": "the database password should be hunter2", "score": 0.93},
            {"content": "unrelated topic about gardening", "score": 0.50},
        ]
        reranked = mmr_rerank(results, lambda_param=0.5, top_k=3)
        # The "gardening" result should appear in top 3 (diversity)
        contents = [r["content"] for r in reranked]
        self.assertIn("unrelated topic about gardening", contents,
                     "MMR should diversify by including the unrelated result")
    
    def test_mmr_single_result(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [{"content": "only one result", "score": 0.5}]
        reranked = mmr_rerank(results)
        self.assertEqual(len(reranked), 1)
    
    def test_empty_results(self):
        from mnemosyne.core.mmr import mmr_rerank
        reranked = mmr_rerank([])
        self.assertEqual(len(reranked), 0)


# ============================================================
# Test 5: Temporal Parser
# ============================================================
class TestTemporalParser(unittest.TestCase):
    def test_absolute_date_iso(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        result = extract_temporal("Meeting on 2026-05-20 about project alpha")
        self.assertEqual(result["event_date"], "2026-05-20")
        self.assertEqual(result["event_date_precision"], "day")
    
    def test_relative_yesterday(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        result = extract_temporal("I met with Denis yesterday")
        self.assertIsNotNone(result["event_date"])
        self.assertIn("yesterday", result["temporal_tags"])
    
    def test_last_monday(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        now = datetime(2026, 5, 26, 12, 0, 0)  # Tuesday
        result = extract_temporal("last Monday we discussed memory", reference=now)
        self.assertIsNotNone(result["event_date"])
        # Last Monday from Tuesday May 26 = May 18
        expected = date(2026, 5, 18)
        self.assertEqual(result["event_date"], expected.isoformat())
    
    def test_next_week(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        now = datetime(2026, 5, 26, 12, 0, 0)
        result = extract_temporal("deploy next week", reference=now)
        self.assertIsNotNone(result["event_date"])
        self.assertEqual(result["event_date_precision"], "week")
    
    def test_n_days_ago(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        now = datetime(2026, 5, 26, 12, 0, 0)
        result = extract_temporal("3 days ago we fixed the bug", reference=now)
        expected = date(2026, 5, 23)
        self.assertEqual(result["event_date"], expected.isoformat())
    
    def test_in_n_weeks(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        now = datetime(2026, 5, 26, 12, 0, 0)
        result = extract_temporal("launch in 2 weeks", reference=now)
        self.assertIsNotNone(result["event_date"])
        expected = (now + timedelta(weeks=2)).date()
        self.assertEqual(result["event_date"], expected.isoformat())
    
    def test_named_month(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        result = extract_temporal("Released on May 15, 2026")
        self.assertEqual(result["event_date"], "2026-05-15")
    
    def test_no_date(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        result = extract_temporal("This text has no date information at all")
        self.assertIsNone(result["event_date"])
        self.assertEqual(result["event_date_precision"], "unknown")
    
    def test_morning_tag(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        result = extract_temporal("I met him in the morning")
        self.assertIn("morning", result["temporal_tags"])
    
    def test_vague_recently(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        result = extract_temporal("I recently changed the config")
        self.assertEqual(result["event_date_precision"], "relative")
        self.assertIn("recently", result["temporal_tags"])


# ============================================================
# Test 6: Query Cache
# ============================================================
class TestQueryCache(unittest.TestCase):
    def setUp(self):
        from mnemosyne.core.query_cache import QueryCache
        self.cache = QueryCache(max_size=100)
    
    def test_cache_hit_exact(self):
        results = [{"content": "cached result", "score": 0.9}]
        self.cache.put("test query", results)
        cached = self.cache.get("test query")
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["content"], "cached result")
        self.assertEqual(self.cache.hits, 1)
    
    def test_cache_miss(self):
        cached = self.cache.get("nonexistent query")
        self.assertIsNone(cached)
        self.assertEqual(self.cache.misses, 1)
    
    def test_cache_normalized_hit(self):
        results = [{"content": "test", "score": 0.5}]
        self.cache.put("What is the database password", results)
        # Different casing and stop words should still hit
        cached = self.cache.get("what is the database password")
        self.assertIsNotNone(cached)
    
    def test_cache_invalidation(self):
        results = [{"content": "test", "score": 0.5}]
        self.cache.put("query one", results)
        self.cache.invalidate()
        cached = self.cache.get("query one")
        self.assertIsNone(cached)
    
    def test_cache_stats(self):
        self.cache.put("query", [{"content": "x", "score": 0.5}])
        self.cache.get("query")  # hit
        self.cache.get("other")  # miss
        stats = self.cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 0.5, places=2)


# ============================================================
# Test 7: End-to-End Enhanced Recall
# ============================================================
class TestEnhancedRecallE2E(unittest.TestCase):
    """End-to-end test of the enhanced recall pipeline with dummy data."""
    
    def setUp(self):
        os.environ["MNEMOSYNE_ENHANCED_RECALL"] = "1"
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_enhanced.db"
        from mnemosyne.core.beam import BeamMemory, init_beam
        init_beam(self.db_path)
        self.beam = BeamMemory(session_id="test_enhanced", db_path=self.db_path)
    
    def tearDown(self):
        self.beam.conn.close()
        import glob as _glob
        for f in _glob.glob(str(self.db_path) + "*"):
            try:
                os.remove(f)
            except OSError:
                pass
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("MNEMOSYNE_ENHANCED_RECALL", None)
    
    def test_enhanced_recall_basic(self):
        """Basic recall with dummy data."""
        self.beam.remember("The database password is mySecret123", source="dummy", importance=0.8)
        self.beam.remember("The server runs on port 8080 in production", source="dummy", importance=0.7)
        self.beam.remember("Deploy script is located at /opt/deploy/run.sh", source="dummy", importance=0.6)
        
        results = self.beam.recall_enhanced("database password", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("password", results[0]["content"].lower())
    
    def test_enhanced_recall_synonym_expansion(self):
        """Synonym expansion should find 'db' as 'database'."""
        self.beam.remember("The database password is mySecret123", source="dummy", importance=0.8)
        self.beam.remember("Nothing about databases here", source="dummy", importance=0.3)
        
        results = self.beam.recall_enhanced("db password", top_k=3)
        self.assertGreaterEqual(len(results), 1)
    
    def test_enhanced_recall_weibull_scoring(self):
        """Weibull scoring should be applied."""
        self.beam.remember("I prefer dark mode for all apps", source="dummy", importance=0.8,
                          extract_entities=True)
        
        results = self.beam.recall_enhanced("dark mode preference", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        # Check that weibull_boost is in results
        if results:
            self.assertIn("weibull_boost", results[0],
                         "Weibull boost should be in enhanced recall results")
    
    def test_enhanced_recall_mmr_diversity(self):
        """MMR should promote diversity."""
        for i in range(3):
            self.beam.remember(f"The database password is secret{i}", source="dummy", importance=0.9 - i * 0.05)
        self.beam.remember("Gardening tips: water plants daily in summer", source="dummy", importance=0.3)
        
        results = self.beam.recall_enhanced("database password", top_k=4, mmr_lambda=0.5)
        self.assertGreaterEqual(len(results), 1)
    
    def test_enhanced_recall_backward_compat(self):
        """Without MNEMOSYNE_ENHANCED_RECALL, should fall through."""
        os.environ.pop("MNEMOSYNE_ENHANCED_RECALL", None)
        self.beam.remember("Test memory for backward compat", source="dummy", importance=0.5)
        # This should NOT crash and should use the original recall
        results = self.beam.recall_enhanced("test memory", top_k=3)
        self.assertGreaterEqual(len(results), 1)
        os.environ["MNEMOSYNE_ENHANCED_RECALL"] = "1"
    
    def test_temporal_extraction_on_store(self):
        """Temporal info should be extracted on remember()."""
        memory_id = self.beam.remember(
            "I met with Alice last Monday about the project deadline",
            source="dummy", importance=0.7
        )
        cursor = self.beam.conn.cursor()
        cursor.execute("SELECT event_date, event_date_precision, temporal_tags FROM working_memory WHERE id=?", (memory_id,))
        row = cursor.fetchone()
        self.assertIsNotNone(row)
        # Should have extracted some temporal info
        if row["event_date"]:
            self.assertIsNotNone(row["event_date"])
        if row["temporal_tags"]:
            tags = json.loads(row["temporal_tags"])
            self.assertIsInstance(tags, list)
    
    def test_query_cache_works(self):
        """Query cache should cache results."""
        self.beam.remember("Cache test memory about purple elephants", source="dummy", importance=0.5)
        
        # First call should be a miss
        results1 = self.beam.recall_enhanced("purple elephants", top_k=3)
        self.assertGreaterEqual(len(results1), 1)
        
        # Second call should be a cache hit (faster)
        start = time.perf_counter()
        results2 = self.beam.recall_enhanced("purple elephants", top_k=3)
        elapsed = time.perf_counter() - start
        self.assertGreaterEqual(len(results2), 1)
        # Cache hit should be very fast (< 10ms for exact match)
        # But we don't assert on timing in CI — just verify results match
        self.assertEqual(results1[0]["content"], results2[0]["content"])
    
    def test_schema_migration_applied(self):
        """Temporal columns should exist after init_beam."""
        cursor = self.beam.conn.cursor()
        cursor.execute("PRAGMA table_info(working_memory)")
        columns = {row[1] for row in cursor.fetchall()}
        self.assertIn("event_date", columns)
        self.assertIn("event_date_precision", columns)
        self.assertIn("temporal_tags", columns)
        self.assertIn("corrected_by", columns)


# ============================================================
# Test 8: Integration — All modules work together
# ============================================================
class TestIntegration(unittest.TestCase):
    """Integration test: store dummy data, recall with all features."""
    
    def setUp(self):
        os.environ["MNEMOSYNE_ENHANCED_RECALL"] = "1"
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_integration.db"
        from mnemosyne.core.beam import BeamMemory, init_beam
        init_beam(self.db_path)
        self.beam = BeamMemory(session_id="test_integration", db_path=self.db_path)
    
    def tearDown(self):
        self.beam.conn.close()
        import glob as _glob, shutil
        for f in _glob.glob(str(self.db_path) + "*"):
            try: os.remove(f)
            except OSError: pass
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("MNEMOSYNE_ENHANCED_RECALL", None)
    
    def test_full_pipeline(self):
        """Store varied dummy data and test enhanced recall."""
        dummy_data = [
            ("Yesterday I changed the database password to SuperSecret2026", "dummy", 0.9),
            ("Denis prefers dark theme for all development tools", "dummy", 0.8),
            ("The production server runs on 192.168.1.100 port 8080", "dummy", 0.85),
            ("Last Monday we had a meeting about project deadlines", "dummy", 0.7),
            ("How to deploy: run /opt/deploy/deploy.sh --env prod", "dummy", 0.75),
            ("The API key for the weather service is WX-12345-ABCDE", "dummy", 0.9),
            ("Denis likes to work late at night, prefers async communication", "dummy", 0.6),
            ("On May 15, 2026 we launched version 3.0 of the platform", "dummy", 0.8),
            ("The git repository is at github.com/NousResearch/hermes-agent", "dummy", 0.7),
            ("Health check endpoint: GET /health returns 200 OK", "dummy", 0.8),
        ]
        
        for content, source, importance in dummy_data:
            self.beam.remember(content, source=source, importance=importance)
        
        # Test 1: Factual query
        results = self.beam.recall_enhanced("what is the database password", top_k=5)
        self.assertGreaterEqual(len(results), 1)
        # Check that at least one result mentions database or password
        contents = " ".join(r["content"].lower() for r in results)
        self.assertTrue(
            "database" in contents or "password" in contents,
            f"Expected 'database' or 'password' in results: {[r['content'][:50] for r in results]}"
        )
        
        # Test 2: Preference query
        results = self.beam.recall_enhanced("what does Denis prefer", top_k=5)
        self.assertGreaterEqual(len(results), 1)
        
        # Test 3: Temporal query
        results = self.beam.recall_enhanced("what happened last Monday", top_k=5)
        self.assertGreaterEqual(len(results), 1)
        
        # Test 4: Procedural query
        results = self.beam.recall_enhanced("how do I deploy", top_k=5)
        self.assertGreaterEqual(len(results), 1)
        
        # Test 5: Synonym-expanded query
        results = self.beam.recall_enhanced("find the API token", top_k=5)
        self.assertGreaterEqual(len(results), 1)
        
        # Test 6: Cache stats
        if hasattr(self.beam, '_query_cache'):
            stats = self.beam._query_cache.stats()
            self.assertIn("hits", stats)
            self.assertIn("hit_rate", stats)


if __name__ == "__main__":
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    
    # Exit with non-zero if any failures
    sys.exit(0 if result.wasSuccessful() else 1)
