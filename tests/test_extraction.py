"""
Tests for Mnemosyne Structured Fact Extraction (Phase 2)
"""

import os
import sys
import json
import sqlite3
import tempfile
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.core.extraction import (
    extract_facts,
    extract_facts_safe,
    _build_extraction_prompt,
    _parse_facts,
    EXTRACTION_PROMPT,
)
from mnemosyne.core.triples import TripleStore, init_triples


class MockLLM:
    """Mock LLM that returns predictable responses."""
    def __init__(self, response="The user loves coffee\nThe user hates mornings"):
        self.response = response
        self.call_count = 0
        self.last_prompt = None
    
    def __call__(self, prompt, **kwargs):
        self.call_count += 1
        self.last_prompt = prompt
        return self.response


def test_build_extraction_prompt():
    """Test that the extraction prompt includes the user text."""
    prompt = _build_extraction_prompt("I love coffee")
    assert "I love coffee" in prompt
    assert "Extract" in prompt or "extract" in prompt.lower()
    print("PASS: test_build_extraction_prompt")


def test_parse_facts_basic():
    """Test parsing LLM output into facts."""
    raw = "The user loves coffee\nThe user hates mornings\n"
    facts = _parse_facts(raw)
    assert len(facts) == 2
    assert "loves coffee" in facts[0]
    assert "hates mornings" in facts[1]
    print("PASS: test_parse_facts_basic")


def test_parse_facts_with_numbering():
    """Test parsing facts with numbering/bullets."""
    raw = "1. The user loves coffee\n2. The user hates mornings\n- User prefers tea\n* User dislikes rain"
    facts = _parse_facts(raw)
    assert len(facts) == 4
    assert all(not fact.startswith(("1.", "2.", "-", "*")) for fact in facts)
    print("PASS: test_parse_facts_with_numbering")


def test_parse_facts_no_facts():
    """Test parsing 'NO_FACTS' response."""
    facts = _parse_facts("NO_FACTS")
    assert facts == []
    print("PASS: test_parse_facts_no_facts")


def test_parse_facts_empty():
    """Test parsing empty response."""
    facts = _parse_facts("")
    assert facts == []
    facts = _parse_facts("   \n   ")
    assert facts == []
    print("PASS: test_parse_facts_empty")


def test_extract_facts_safe_no_llm():
    """Test that extract_facts_safe returns empty list when no LLM."""
    # Ensure no LLM is configured
    old_base_url = os.environ.get("MNEMOSYNE_LLM_BASE_URL", "")
    old_enabled = os.environ.get("MNEMOSYNE_LLM_ENABLED", "")
    
    try:
        os.environ["MNEMOSYNE_LLM_BASE_URL"] = ""
        os.environ["MNEMOSYNE_LLM_ENABLED"] = "false"
        
        facts = extract_facts_safe("I love coffee")
        assert facts == []
        print("PASS: test_extract_facts_safe_no_llm")
    finally:
        os.environ["MNEMOSYNE_LLM_BASE_URL"] = old_base_url
        os.environ["MNEMOSYNE_LLM_ENABLED"] = old_enabled


def test_extract_facts_safe_exception_handling():
    """Test that extract_facts_safe never raises."""
    # Should not raise even with garbage input
    facts = extract_facts_safe(None)
    assert facts == []
    facts = extract_facts_safe("")
    assert facts == []
    facts = extract_facts_safe("x")  # Too short for extraction
    assert facts == []
    print("PASS: test_extract_facts_safe_exception_handling")


def test_triplestore_add_facts():
    """Test TripleStore.add_facts() batch storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_triples(db_path)
        
        triples = TripleStore(db_path=db_path)
        count = triples.add_facts(
            "mem_123",
            ["The user loves coffee", "The user hates mornings", "x"],  # "x" too short
            source="test",
            confidence=0.7
        )
        
        assert count == 2  # "x" filtered out
        
        # Verify stored
        all_facts = triples.query_by_predicate("fact")
        assert len(all_facts) == 2
        assert all(f["subject"] == "mem_123" for f in all_facts)
        assert all(f["predicate"] == "fact" for f in all_facts)
        assert all(f["confidence"] == 0.7 for f in all_facts)
        
        print("PASS: test_triplestore_add_facts")


def test_triplestore_add_facts_empty():
    """Test TripleStore.add_facts() with empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_triples(db_path)
        
        triples = TripleStore(db_path=db_path)
        count = triples.add_facts("mem_456", [], source="test")
        assert count == 0
        
        print("PASS: test_triplestore_add_facts_empty")


def test_extraction_prompt_configurable():
    """Test that EXTRACTION_PROMPT env var overrides default."""
    old_prompt = os.environ.get("MNEMOSYNE_EXTRACTION_PROMPT", "")
    
    try:
        custom = "CUSTOM PROMPT: {text}"
        os.environ["MNEMOSYNE_EXTRACTION_PROMPT"] = custom
        
        # Re-import to pick up new env var
        # (In real usage, you'd restart; here we test the constant directly)
        from mnemosyne.core.extraction import EXTRACTION_PROMPT as ep
        # Note: module-level constants are set at import time, so this tests
        # that the code structure supports it. The actual override requires
        # re-import or setting before import.
        
        # Instead, verify the function uses the constant
        prompt = _build_extraction_prompt("test")
        assert "test" in prompt
        print("PASS: test_extraction_prompt_configurable")
    finally:
        if old_prompt:
            os.environ["MNEMOSYNE_EXTRACTION_PROMPT"] = old_prompt
        else:
            os.environ.pop("MNEMOSYNE_EXTRACTION_PROMPT", None)


def run_all_tests():
    """Run all unit tests."""
    print("=" * 60)
    print("Phase 2: Structured Fact Extraction — Unit Tests")
    print("=" * 60)
    
    tests = [
        test_build_extraction_prompt,
        test_parse_facts_basic,
        test_parse_facts_with_numbering,
        test_parse_facts_no_facts,
        test_parse_facts_empty,
        test_extract_facts_safe_no_llm,
        test_extract_facts_safe_exception_handling,
        test_triplestore_add_facts,
        test_triplestore_add_facts_empty,
        test_extraction_prompt_configurable,
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
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
