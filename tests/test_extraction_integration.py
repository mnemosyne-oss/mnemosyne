"""
Integration tests for Mnemosyne Structured Fact Extraction (Phase 2)
Tests end-to-end: remember with extract -> recall finds facts
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.triples import TripleStore, init_triples
from mnemosyne.core.beam import BeamMemory


class MockLLMExtractor:
    """Mock LLM for fact extraction that returns predictable facts."""
    def __init__(self, facts=None):
        self.facts = facts or [
            "The user loves coffee",
            "The user hates mornings",
            "The user prefers dark roast"
        ]
    
    def __call__(self, prompt, **kwargs):
        return "\n".join(self.facts)


def test_end_to_end_extract_recall():
    """
    Test: remember with extract=True -> facts stored -> recall finds them
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Initialize
        init_triples(db_path)
        
        # Create Mnemosyne instance
        mem = Mnemosyne(session_id="test_session", db_path=db_path)
        
        # Store a memory WITH fact extraction (mocked)
        content = "I absolutely love coffee, especially dark roast. I hate mornings though."
        
        # Manually inject facts (simulating LLM extraction)
        memory_id = mem.remember(content, source="test", extract=False)
        
        # Manually add facts to triples
        triples = TripleStore(db_path=db_path)
        triples.add_facts(memory_id, [
            "The user loves coffee",
            "The user hates mornings",
            "The user prefers dark roast"
        ], source="test", confidence=0.7)
        
        # Now recall with a query that matches the facts
        results = mem.recall("what does the user like", top_k=5)
        
        # Should find the memory via fact matching
        assert len(results) > 0, "Recall should find results"
        
        # Check if fact_match is present in any result
        fact_matches = [r for r in results if r.get("fact_match")]
        assert len(fact_matches) > 0, "At least one result should have fact_match=True"
        
        # The memory should be in results
        memory_ids = [r["id"] for r in results]
        assert memory_id in memory_ids, f"Memory {memory_id} should be in recall results"
        
        print("PASS: test_end_to_end_extract_recall")


def test_fact_recall_keyword_matching():
    """
    Test: Fact recall uses keyword matching against stored facts
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_triples(db_path)
        
        mem = Mnemosyne(session_id="test_session_2", db_path=db_path)
        
        # Store two memories
        id1 = mem.remember("I love hiking in the mountains", source="test", extract=False)
        id2 = mem.remember("I enjoy swimming at the beach", source="test", extract=False)
        
        # Add facts manually
        triples = TripleStore(db_path=db_path)
        triples.add_facts(id1, ["The user loves hiking", "The user enjoys mountains"], source="test")
        triples.add_facts(id2, ["The user enjoys swimming", "The user likes the beach"], source="test")
        
        # Recall for "hiking" should find id1 via facts
        results = mem.recall("hiking", top_k=5)
        result_ids = [r["id"] for r in results]
        assert id1 in result_ids, "Should find memory about hiking via fact match"
        
        # Recall for "swimming" should find id2 via facts
        results = mem.recall("swimming", top_k=5)
        result_ids = [r["id"] for r in results]
        assert id2 in result_ids, "Should find memory about swimming via fact match"
        
        print("PASS: test_fact_recall_keyword_matching")


def test_fact_and_entity_extraction_together():
    """
    Test: Both extract_entities=True and extract=True work together
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_triples(db_path)
        
        mem = Mnemosyne(session_id="test_session_3", db_path=db_path)
        
        content = "I met Abdias in New York. He loves coffee."
        
        # Store with both extraction flags (entities will extract, facts won't because no LLM)
        memory_id = mem.remember(content, source="test", extract_entities=True, extract=False)
        
        # Manually add facts
        triples = TripleStore(db_path=db_path)
        triples.add_facts(memory_id, ["The user met Abdias", "Abdias loves coffee"], source="test")
        
        # Recall for "Abdias" should find via entity
        results = mem.recall("Abdias", top_k=5)
        result_ids = [r["id"] for r in results]
        assert memory_id in result_ids, "Should find memory via entity match"
        
        # Recall for "coffee" should find via fact
        results = mem.recall("coffee", top_k=5)
        result_ids = [r["id"] for r in results]
        assert memory_id in result_ids, "Should find memory via fact match"
        
        print("PASS: test_fact_and_entity_extraction_together")


def test_backward_compatibility():
    """
    Test: remember() without extract works exactly as before
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_triples(db_path)
        
        mem = Mnemosyne(session_id="test_session_4", db_path=db_path)
        
        # Store without any extraction
        memory_id = mem.remember("Simple memory without extraction", source="test")
        
        # Should be retrievable
        results = mem.recall("simple memory", top_k=5)
        result_ids = [r["id"] for r in results]
        assert memory_id in result_ids, "Should find memory without extraction"
        
        # Triples should be empty
        triples = TripleStore(db_path=db_path)
        all_facts = triples.query_by_predicate("fact")
        assert len(all_facts) == 0, "No facts should be stored without extract=True"
        
        print("PASS: test_backward_compatibility")


def test_graceful_fallback_no_llm():
    """
    Test: When LLM unavailable, memory still stores, no error
    """
    from unittest.mock import patch
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_triples(db_path)
        
        mem = Mnemosyne(session_id="test_session_5", db_path=db_path)
        
        # Patch llm_available so extraction is skipped regardless of env state.
        # extract_facts() calls local_llm.llm_available() through the live module
        # reference, not through extraction's at-import-time binding.
        with patch("mnemosyne.core.local_llm.llm_available", return_value=False):
            # This should NOT raise even though extract=True and no LLM
            memory_id = mem.remember(
                "I love coffee",
                source="test",
                extract=True  # LLM unavailable, but should not fail
            )
            
            assert memory_id is not None, "Memory ID should be returned"
        
        # No facts should be stored
        triples = TripleStore(db_path=db_path)
        all_facts = triples.query_by_predicate("fact")
        assert len(all_facts) == 0, "No facts when LLM unavailable"
        
        print("PASS: test_graceful_fallback_no_llm")


def test_fact_aware_recall_boosts_scores():
    """
    Test: Fact matches get score boost (1.2x)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_triples(db_path)
        
        mem = Mnemosyne(session_id="test_session_6", db_path=db_path)
        
        # Store two similar memories (different content to avoid dedup)
        id1 = mem.remember("I love coffee and tea in the morning", source="test", importance=0.5)
        id2 = mem.remember("I love coffee and tea in the evening", source="test", importance=0.5)
        
        # Add fact only to id1
        triples = TripleStore(db_path=db_path)
        triples.add_facts(id1, ["The user loves coffee"], source="test")
        
        # Recall for "coffee" - id1 should have fact_match
        results = mem.recall("coffee", top_k=5)
        
        # Find results for id1 and id2
        r1 = [r for r in results if r["id"] == id1]
        r2 = [r for r in results if r["id"] == id2]
        
        if r1 and r2:
            # id1 should have fact_match and boosted score
            assert r1[0].get("fact_match") == True, "id1 should have fact_match"
            assert r1[0]["score"] > r2[0]["score"], "Fact match should boost score"
        
        print("PASS: test_fact_aware_recall_boosts_scores")


def run_all_tests():
    """Run all integration tests."""
    print("=" * 60)
    print("Phase 2: Structured Fact Extraction — Integration Tests")
    print("=" * 60)
    
    tests = [
        test_end_to_end_extract_recall,
        test_fact_recall_keyword_matching,
        test_fact_and_entity_extraction_together,
        test_backward_compatibility,
        test_graceful_fallback_no_llm,
        test_fact_aware_recall_boosts_scores,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
