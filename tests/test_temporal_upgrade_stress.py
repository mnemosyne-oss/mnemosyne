#!/usr/bin/env python3
"""
RIGOROUS stress and edge-case test suite for Mnemosyne Temporal Upgrade.
Tests worst-case scenarios, boundary conditions, concurrency, and logic flaws.

Run: MNEMOSYNE_ENHANCED_RECALL=1 python3 test_temporal_upgrade_stress.py
"""

import sys, os, unittest, tempfile, time, json, math, threading, random, string
from datetime import datetime, timedelta, date
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# ============================================================
# STRESS 1: Weibull Decay — worst-case inputs
# ============================================================
class TestWeibullEdgeCases(unittest.TestCase):
    """Push Weibull decay to its limits."""
    
    @classmethod
    def setUpClass(cls):
        from mnemosyne.core.weibull import weibull_boost, weibull_decay_factor, WEIBULL_PARAMS
    
    def test_none_memory_type(self):
        """None memory type should fall back to general."""
        from mnemosyne.core.weibull import weibull_boost, weibull_decay_factor
        now = datetime.now()
        # Should not crash
        b1 = weibull_boost(now.isoformat(), now, memory_type=None)
        self.assertGreaterEqual(b1, 0.0)
        b2 = weibull_decay_factor(10.0, None)
        self.assertGreaterEqual(b2, 0.0)
    
    def test_empty_string_memory_type(self):
        from mnemosyne.core.weibull import weibull_decay_factor
        b = weibull_decay_factor(100.0, "")
        self.assertGreaterEqual(b, 0.0)
    
    def test_unknown_memory_type(self):
        from mnemosyne.core.weibull import weibull_decay_factor
        b = weibull_decay_factor(100.0, "this_type_does_not_exist_xyz")
        self.assertGreaterEqual(b, 0.0)
    
    def test_zero_age(self):
        from mnemosyne.core.weibull import weibull_decay_factor
        for mem_type in ["profile", "preference", "general", "event", "request"]:
            b = weibull_decay_factor(0.0, mem_type)
            self.assertAlmostEqual(b, 1.0, places=5, msg=f"Type {mem_type}: zero age should give 1.0")
    
    def test_negative_age(self):
        """Negative age (future memory) should give 1.0."""
        from mnemosyne.core.weibull import weibull_decay_factor
        b = weibull_decay_factor(-100.0, "general")
        self.assertAlmostEqual(b, 1.0, places=5)
    
    def test_extreme_age(self):
        """100-year-old memory should still be computable."""
        from mnemosyne.core.weibull import weibull_decay_factor
        age = 100 * 365 * 24  # 100 years in hours
        b = weibull_decay_factor(age, "profile")
        self.assertGreaterEqual(b, 0.0)
        self.assertLess(b, 1.0)
    
    def test_zero_eta_protection(self):
        """If eta=0 somehow, should not divide by zero."""
        from mnemosyne.core.weibull import weibull_decay_factor, WEIBULL_PARAMS
        # Temporarily add a zero-eta type
        WEIBULL_PARAMS["zero_eta_test"] = {"k": 1.0, "eta": 0.0}
        b = weibull_decay_factor(100.0, "zero_eta_test")
        self.assertEqual(b, 0.0)
        del WEIBULL_PARAMS["zero_eta_test"]
    
    def test_negative_eta_protection(self):
        """Negative eta should not crash."""
        from mnemosyne.core.weibull import weibull_decay_factor, WEIBULL_PARAMS
        WEIBULL_PARAMS["neg_eta_test"] = {"k": 1.0, "eta": -10.0}
        b = weibull_decay_factor(100.0, "neg_eta_test")
        self.assertEqual(b, 0.0)
        del WEIBULL_PARAMS["neg_eta_test"]
    
    def test_bogus_timestamp_formats(self):
        from mnemosyne.core.weibull import weibull_boost
        bogus = ["", " ", "abc", "2026", "2026-13-45", "not-a-date-at-all", "2026-05-20T25:99:99"]
        for ts in bogus:
            b = weibull_boost(ts, memory_type="general")
            self.assertEqual(b, 0.0, f"Bogus timestamp '{ts}' should return 0.0")
    
    def test_integer_timestamp(self):
        from mnemosyne.core.weibull import weibull_boost
        b = weibull_boost(42, memory_type="general")
        self.assertEqual(b, 0.0)
    
    def test_override_halflife_works(self):
        from mnemosyne.core.weibull import weibull_boost
        now = datetime.now()
        ago = (now - timedelta(hours=24)).isoformat()
        b = weibull_boost(ago, now, memory_type="general", halflife_hours=48.0)
        self.assertGreater(b, 0.5)  # Should be >50% with 48h halflife
    
    def test_monotonic_decay(self):
        """Older things should decay more."""
        from mnemosyne.core.weibull import weibull_decay_factor
        prev = 1.0
        for age in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            b = weibull_decay_factor(age, "event")
            self.assertLessEqual(b, prev, f"Decay should be monotonic: age={age}")
            prev = b


# ============================================================
# STRESS 2: MMR — diversity correctness under extreme conditions
# ============================================================
class TestMMREdgeCases(unittest.TestCase):
    def test_all_identical_content(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [{"content": "exact same text", "score": s} for s in [0.9, 0.8, 0.7, 0.6, 0.5]]
        reranked = mmr_rerank(results, lambda_param=0.5, top_k=3)
        self.assertEqual(len(reranked), 3)
        # All identical, MMR should still work (just picks top scores)
    
    def test_lambda_zero(self):
        """Pure diversity (lambda=0) — should pick most diverse."""
        from mnemosyne.core.mmr import mmr_rerank
        results = [
            {"content": "aaa xxx yyy", "score": 0.9},
            {"content": "aaa xxx yyy", "score": 0.89},
            {"content": "completely different topic", "score": 0.1},
        ]
        reranked = mmr_rerank(results, lambda_param=0.0, top_k=2)
        # First pick is highest score, second pick should be diverse
        contents = [r["content"] for r in reranked]
        self.assertIn("completely different topic", contents)
    
    def test_lambda_one(self):
        """Pure relevance (lambda=1) — same as sort by score."""
        from mnemosyne.core.mmr import mmr_rerank
        results = [
            {"content": "first", "score": 0.9},
            {"content": "second", "score": 0.8},
            {"content": "third", "score": 0.7},
        ]
        reranked = mmr_rerank(results, lambda_param=1.0, top_k=3)
        self.assertEqual(reranked[0]["content"], "first")
        self.assertEqual(reranked[1]["content"], "second")
    
    def test_negative_lambda(self):
        """Negative lambda should be handled gracefully."""
        from mnemosyne.core.mmr import mmr_rerank
        results = [{"content": f"item {i}", "score": 1.0 - i * 0.1} for i in range(5)]
        reranked = mmr_rerank(results, lambda_param=-0.5, top_k=3)
        self.assertEqual(len(reranked), 3)
    
    def test_lambda_above_one(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [{"content": f"item {i}", "score": 0.5} for i in range(5)]
        reranked = mmr_rerank(results, lambda_param=2.5, top_k=3)
        self.assertEqual(len(reranked), 3)
    
    def test_all_zero_scores(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [{"content": f"item {i}", "score": 0.0} for i in range(5)]
        reranked = mmr_rerank(results, top_k=3)
        self.assertEqual(len(reranked), 3)
    
    def test_missing_score_field(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [{"content": f"item {i}"} for i in range(5)]  # No score field
        reranked = mmr_rerank(results, top_k=3)
        self.assertEqual(len(reranked), 3)
    
    def test_single_word_content(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [
            {"content": "a", "score": 0.9},
            {"content": "b", "score": 0.8},
            {"content": "c", "score": 0.7},
        ]
        reranked = mmr_rerank(results, top_k=3)
        self.assertEqual(len(reranked), 3)
    
    def test_unicode_content(self):
        from mnemosyne.core.mmr import mmr_rerank
        results = [
            {"content": "数据库密码是hunter2", "score": 0.9},
            {"content": "服务器端口8080", "score": 0.8},
            {"content": "部署脚本位置", "score": 0.7},
        ]
        reranked = mmr_rerank(results, top_k=3)
        self.assertEqual(len(reranked), 3)


# ============================================================
# STRESS 3: Query Cache — concurrency, eviction, persistence
# ============================================================
class TestQueryCacheEdgeCases(unittest.TestCase):
    def setUp(self):
        from mnemosyne.core.query_cache import QueryCache
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "cache.db"
        self.cache = QueryCache(db_path=self.db_path, max_size=10, ttl_seconds=1)
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
    
    def test_eviction_at_max_size(self):
        results = [{"content": f"result {i}", "score": 0.5} for i in range(3)]
        for i in range(15):
            self.cache.put(f"query {i}", results)
        stats = self.cache.stats()
        self.assertLessEqual(stats["size"], 10, "Cache should evict at max_size")
    
    def test_concurrent_puts(self):
        """Multiple threads putting simultaneously should not crash."""
        errors = []
        def put_many(prefix, count):
            try:
                for i in range(count):
                    self.cache.put(f"{prefix}_{i}", [{"content": f"x{i}", "score": 0.5}])
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for prefix in ["a", "b", "c", "d"]:
            t = threading.Thread(target=put_many, args=(prefix, 10))
            threads.append(t)
        
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        
        self.assertEqual(len(errors), 0, f"Concurrent puts had errors: {errors}")
        stats = self.cache.stats()
    
    def test_concurrent_gets_and_puts(self):
        """Concurrent reads and writes should not deadlock."""
        errors = []
        barrier = threading.Barrier(4)
        
        def worker(op_type, iterations):
            try:
                barrier.wait()
                for i in range(iterations):
                    if op_type == "put":
                        self.cache.put(f"{op_type}_{i}", [{"content": str(i), "score": 0.5}])
                    elif op_type == "get":
                        self.cache.get(f"query_{i % 20}")
                    elif op_type == "mixed":
                        if i % 2 == 0:
                            self.cache.put(f"mixed_{i}", [{"content": str(i), "score": 0.5}])
                        else:
                            self.cache.get(f"mixed_{i-1}")
            except Exception as e:
                errors.append(f"{op_type}: {e}")
        
        threads = [
            threading.Thread(target=worker, args=("put", 50)),
            threading.Thread(target=worker, args=("get", 50)),
            threading.Thread(target=worker, args=("mixed", 50)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
        
        self.assertEqual(len(errors), 0, f"Concurrent ops had errors: {errors}")
    
    def test_invalidation_during_operations(self):
        """Invalidating while threads are operating should be safe."""
        errors = []
        
        def worker():
            try:
                for i in range(50):
                    self.cache.put(f"w_{i}", [{"content": str(i), "score": 0.5}])
                    self.cache.get(f"w_{i}")
            except Exception as e:
                errors.append(str(e))
        
        t = threading.Thread(target=worker)
        t.start()
        
        # Invalidate multiple times during operations
        for _ in range(10):
            time.sleep(0.01)
            self.cache.invalidate()
        
        t.join(timeout=5)
        self.assertEqual(len(errors), 0)
    
    def test_repeated_invalidation(self):
        for _ in range(20):
            self.cache.put(f"q_{_}", [{"content": "x", "score": 0.5}])
        for _ in range(10):
            self.cache.invalidate()
        # Should end up empty
        stats = self.cache.stats()
        self.assertEqual(stats["size"], 0)
    
    def test_persistence_across_instances(self):
        from mnemosyne.core.query_cache import QueryCache
        self.cache.put("persist_test", [{"content": "persisted", "score": 0.9}])
        
        # New instance with same DB
        cache2 = QueryCache(db_path=self.db_path)
        cached = cache2.get("persist_test")
        self.assertIsNotNone(cached)
        self.assertEqual(cached[0]["content"], "persisted")
    
    def test_get_with_no_embedding_correctly(self):
        """Cache should still check Tier 1 even without embedding."""
        self.cache.put("no_emb_test", [{"content": "test", "score": 0.5}])
        # Get without embedding
        cached = self.cache.get("no_emb_test", embedding=None)
        self.assertIsNotNone(cached)
    
    def test_normalized_variants_hit(self):
        """Different spacing, casing should still hit Tier 1."""
        self.cache.put(" THE  Database  PASSWORD ", [{"content": "found", "score": 0.9}])
        cached = self.cache.get("the database password")
        self.assertIsNotNone(cached)


# ============================================================
# STRESS 4: Temporal Parser — edge dates, malformed input
# ============================================================
class TestTemporalParserEdgeCases(unittest.TestCase):
    def test_empty_text(self):
        from mnemosyne.core.temporal_parser import extract_temporal, parse_nl_date
        r = extract_temporal("")
        self.assertIsNone(r["event_date"])
        self.assertEqual(r["event_date_precision"], "unknown")
    
    def test_only_stop_words(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        r = extract_temporal("the and or but if")
        self.assertIsNone(r["event_date"])
    
    def test_very_long_text(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        long_text = "blah " * 1000 + " on 2026-05-20 " + "blah " * 1000
        r = extract_temporal(long_text)
        self.assertEqual(r["event_date"], "2026-05-20")
    
    def test_multiple_dates_returns_first(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        # "2026-01-15" comes before "2026-12-25" in the text
        r = extract_temporal("On 2026-01-15 we planned, on 2026-12-25 we deployed")
        self.assertEqual(r["event_date"], "2026-01-15")
    
    def test_february_29_leap_year(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        r = extract_temporal("Meeting on 2024-02-29")  # 2024 is a leap year
        self.assertEqual(r["event_date"], "2024-02-29")
    
    def test_non_leap_feb_29(self):
        """2026-02-29 does not exist — parser should handle gracefully."""
        from mnemosyne.core.temporal_parser import extract_temporal
        r = extract_temporal("Meeting on 2026-02-29")  # Not a real date
        # Parser should NOT crash, but may return None (graceful degradation)
        self.assertIsNotNone(r)  # Should return dict, not crash
    
    def test_dates_before_epoch(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        r = extract_temporal("Founded on 1900-06-15")
        self.assertEqual(r["event_date"], "1900-06-15")
    
    def test_next_monday_from_sunday(self):
        from mnemosyne.core.temporal_parser import parse_nl_date
        ref = datetime(2026, 5, 31, 12, 0, 0)  # Sunday
        result = parse_nl_date("next Monday", reference=ref)
        self.assertIsNotNone(result)
        d, prec, tags = result
        # Next Monday from Sunday May 31 = June 1
        self.assertEqual(d, date(2026, 6, 1))
    
    def test_last_monday_from_monday(self):
        from mnemosyne.core.temporal_parser import parse_nl_date
        ref = datetime(2026, 5, 25, 12, 0, 0)  # Monday
        result = parse_nl_date("last Monday", reference=ref)
        d, prec, tags = result
        # Last Monday from Monday = 7 days ago
        self.assertEqual(d, date(2026, 5, 18))
    
    def test_this_monday_from_tuesday(self):
        from mnemosyne.core.temporal_parser import parse_nl_date
        ref = datetime(2026, 5, 26, 12, 0, 0)  # Tuesday
        result = parse_nl_date("this Monday", reference=ref)
        d, prec, tags = result
        # This Monday from Tuesday = yesterday
        self.assertEqual(d, date(2026, 5, 25))
    
    def test_90_days_ago(self):
        from mnemosyne.core.temporal_parser import parse_nl_date
        ref = datetime(2026, 5, 26, 12, 0, 0)
        result = parse_nl_date("90 days ago", reference=ref)
        d, prec, tags = result
        self.assertEqual(d, date(2026, 2, 25))
    
    def test_in_365_days(self):
        from mnemosyne.core.temporal_parser import parse_nl_date
        ref = datetime(2026, 5, 26, 12, 0, 0)
        result = parse_nl_date("in 365 days", reference=ref)
        d, prec, tags = result
        self.assertEqual(d, date(2027, 5, 26))
    
    def test_mixed_named_times(self):
        from mnemosyne.core.temporal_parser import extract_temporal
        # "afternoon" + date
        r = extract_temporal("Meeting on 2026-05-20 in the afternoon")
        self.assertIn("afternoon", r["temporal_tags"])
        self.assertIsNotNone(r["event_date"])


# ============================================================
# STRESS 5: Synonyms — adversarial input
# ============================================================
class TestSynonymsEdgeCases(unittest.TestCase):
    def test_empty_query(self):
        from mnemosyne.core.synonyms import expand_query, normalize_query
        self.assertEqual(normalize_query(""), "")
        self.assertEqual(expand_query(""), "")
    
    def test_only_stop_words(self):
        from mnemosyne.core.synonyms import expand_query, normalize_query
        result = normalize_query("the and or but if a an")
        self.assertEqual(result, "")  # All words are stop words
    
    def test_very_long_query(self):
        from mnemosyne.core.synonyms import expand_query
        long_q = "database " * 200
        result = expand_query(long_q)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
    
    def test_special_characters(self):
        from mnemosyne.core.synonyms import expand_query, normalize_query
        # Should handle special chars gracefully
        result = expand_query("db!@#$%^&*()password")
        self.assertIsInstance(result, str)
    
    def test_numbers_in_query(self):
        from mnemosyne.core.synonyms import normalize_query
        result = normalize_query("port 8080 and server 192.168.1.1")
        self.assertIn("8080", result)
    
    def test_all_synonym_words(self):
        """Query where every content word has a synonym."""
        from mnemosyne.core.synonyms import expand_query
        # db->database, pass->password
        result = expand_query("db pass")
        self.assertIn("database", result)
        self.assertIn("password", result)


# ============================================================
# STRESS 6: Query Intent — ambiguous queries
# ============================================================
class TestQueryIntentEdgeCases(unittest.TestCase):
    def test_empty_query(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("")
        self.assertEqual(intent.category, "general")
        self.assertEqual(intent.confidence, 0.0)
    
    def test_only_stop_words(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("the and or but if")
        self.assertEqual(intent.category, "general")
    
    def test_mixed_intents(self):
        """Query that matches multiple intent patterns."""
        from mnemosyne.core.query_intent import classify_intent
        # "when" = temporal, "is" = factual
        intent = classify_intent("when is the database password changed")
        # Should pick the higher-scoring one
        self.assertIn(intent.category, ["temporal", "factual"])
    
    def test_single_character_query(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("?")
        self.assertEqual(intent.category, "general")
    
    def test_unicode_query(self):
        from mnemosyne.core.query_intent import classify_intent
        intent = classify_intent("数据库密码是什么")  # Chinese
        self.assertIsNotNone(intent.category)


# ============================================================
# STRESS 7: Enhanced Recall — heavy load, mixed scenarios
# ============================================================
class TestEnhancedRecallStress(unittest.TestCase):
    """Store 200 dummy items and recall 50 queries."""
    
    def setUp(self):
        os.environ["MNEMOSYNE_ENHANCED_RECALL"] = "1"
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "stress_test.db"
        from mnemosyne.core.beam import BeamMemory, init_beam
        init_beam(self.db_path)
        self.beam = BeamMemory(session_id="stress_test", db_path=self.db_path)
    
    def tearDown(self):
        self.beam.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("MNEMOSYNE_ENHANCED_RECALL", None)
    
    def test_bulk_store_and_recall(self):
        """Store 200 dummy items and recall 50 queries."""
        import random
        
        # Store varied dummy data
        categories = ["tech", "personal", "project", "meeting", "config"]
        for i in range(200):
            cat = random.choice(categories)
            content = f"Dummy {cat} memory #{i}: "
            if cat == "tech":
                content += random.choice([
                    "database password updated",
                    "server running on port",
                    "deploy script executed",
                    "API endpoint returning 200",
                    "health check passed",
                ])
            elif cat == "personal":
                content += random.choice([
                    "met with colleague yesterday",
                    "prefer dark theme",
                    "lunch at noon",
                    "worked late on Friday",
                ])
            elif cat == "meeting":
                content += random.choice([
                    "last Monday project review",
                    "next week planning session",
                    "on 2026-05-15 budget meeting",
                    "today standup notes",
                ])
            else:
                content += f"item number {i}"
            
            self.beam.remember(content, source=f"dummy_{cat}", importance=random.uniform(0.3, 0.9))
        
        # Run 50 varied recalls
        queries = [
            "database password",
            "what happened last Monday",
            "how do I deploy",
            "preferences dark theme",
            "port 8080",
            "meeting yesterday",
            "API endpoint health",
            "what did we discuss about budget",
            "server configuration",
            "next week plans",
            # Duplicates to test cache
            "database password",
            "what happened last Monday",
            "database password",
            "preferences dark theme",
            "database password",
        ]
        
        for _ in range(2):  # Run twice to test cache
            for q in queries:
                results = self.beam.recall_enhanced(q, top_k=5)
                self.assertIsInstance(results, list)
                # Results should always be <= top_k
                self.assertLessEqual(len(results), 5)
    
    def test_size_limits(self):
        """top_k=0, top_k=1, top_k=very_large."""
        for k in [0, 1, 50, 200]:
            results = self.beam.recall_enhanced("test", top_k=k)
            self.assertIsInstance(results, list)
            if k == 0:
                self.assertEqual(len(results), 0)
    
    def test_feature_toggle_off(self):
        """All features toggled off should still work."""
        results = self.beam.recall_enhanced("test", top_k=3,
            use_cache=False, use_weibull=False, use_mmr=False,
            use_intent=False, use_synonyms=False, use_associative=False)
        self.assertIsInstance(results, list)
    
    def test_associative_not_crash_without_graph(self):
        """Even without episodic graph, associative should not crash."""
        results = self.beam.recall_enhanced("test", top_k=3, use_associative=True)
        self.assertIsInstance(results, list)
    
    def test_no_embedding_fallback(self):
        """Results should still work even if embedding is not available."""
        # Our test setup may or may not have embeddings loaded.
        # The recall should work regardless.
        results = self.beam.recall_enhanced("database password", top_k=3)
        self.assertIsInstance(results, list)


# ============================================================
# STRESS 8: Schema Migration — idempotency & safety
# ============================================================
class TestSchemaMigration(unittest.TestCase):
    def test_init_beam_idempotent(self):
        """Calling init_beam multiple times should be safe."""
        from mnemosyne.core.beam import init_beam
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "migrate_test.db"
        
        # Call init_beam 3 times
        for i in range(3):
            init_beam(db_path)
        
        # Verify columns exist
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(working_memory)")
        columns = {row[1] for row in cursor.fetchall()}
        
        expected = {"event_date", "event_date_precision", "temporal_tags", "corrected_by"}
        for col in expected:
            self.assertIn(col, columns, f"Column {col} missing after init_beam")
        
        conn.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    
    def test_existing_db_migration(self):
        """Migration on a pre-existing DB should add columns without data loss."""
        from mnemosyne.core.beam import BeamMemory, init_beam
        tmpdir = tempfile.mkdtemp()
        db_path = Path(tmpdir) / "existing.db"
        
        # Create DB, add data, then migrate
        beam = BeamMemory(session_id="pre_migrate", db_path=db_path)
        mid = beam.remember("Pre-migration test memory", source="test", importance=0.5)
        
        # Verify migration columns exist
        cursor = beam.conn.cursor()
        cursor.execute("PRAGMA table_info(working_memory)")
        columns = {row[1] for row in cursor.fetchall()}
        self.assertIn("event_date", columns)
        
        # Verify data still exists
        cursor.execute("SELECT content FROM working_memory WHERE id=?", (mid,))
        row = cursor.fetchone()
        self.assertEqual(row["content"], "Pre-migration test memory")
        
        beam.conn.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# STRESS 9: Temporal extraction on store
# ============================================================
class TestTemporalStore(unittest.TestCase):
    def setUp(self):
        os.environ["MNEMOSYNE_ENHANCED_RECALL"] = "1"
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "temporal_store.db"
        from mnemosyne.core.beam import BeamMemory, init_beam
        init_beam(self.db_path)
        self.beam = BeamMemory(session_id="temporal_store", db_path=self.db_path)
    
    def tearDown(self):
        self.beam.conn.close()
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        os.environ.pop("MNEMOSYNE_ENHANCED_RECALL", None)
    
    def test_temporal_extraction_sets_columns(self):
        mid = self.beam.remember(
            "I met with Denis last Monday about the Hermes project",
            source="dummy", importance=0.7
        )
        cursor = self.beam.conn.cursor()
        cursor.execute(
            "SELECT event_date, event_date_precision, temporal_tags FROM working_memory WHERE id=?",
            (mid,)
        )
        row = cursor.fetchone()
        # Should have extracted some temporal data
        self.assertIsNotNone(row["event_date"])
        self.assertNotEqual(row["event_date_precision"], "unknown")
        tags = json.loads(row["temporal_tags"])
        self.assertGreater(len(tags), 0)
    
    def test_no_date_text_leaves_null(self):
        mid = self.beam.remember(
            "This text has absolutely no date information whatsoever",
            source="dummy", importance=0.3
        )
        cursor = self.beam.conn.cursor()
        cursor.execute("SELECT event_date FROM working_memory WHERE id=?", (mid,))
        row = cursor.fetchone()
        # Should be NULL or empty
        self.assertIsNone(row["event_date"])


# ============================================================
if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
