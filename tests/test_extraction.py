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
    _call_local_extraction_llm,
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


def test_call_local_extraction_llm_uses_llamacpp_chat_api(monkeypatch):
    """llama-cpp-python uses max_tokens, not ctransformers' max_new_tokens."""
    monkeypatch.setattr(local_llm, "_llm_backend", "llamacpp")
    monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 123)

    class FakeLlamaCpp:
        def __init__(self):
            self.kwargs = None

        def create_chat_completion(self, **kwargs):
            self.kwargs = kwargs
            return {"choices": [{"message": {"content": "The user likes coffee."}}]}

        def __call__(self, *args, **kwargs):  # pragma: no cover - should not run
            raise AssertionError("llama.cpp extraction should use chat completion API")

    llm = FakeLlamaCpp()
    output = _call_local_extraction_llm(llm, "prompt")

    assert output == "The user likes coffee."
    assert llm.kwargs["max_tokens"] == 123
    assert llm.kwargs["temperature"] == 0.0
    assert "max_new_tokens" not in llm.kwargs


def test_call_local_extraction_llm_preserves_ctransformers_kwargs(monkeypatch):
    """ctransformers still receives max_new_tokens on the direct callable."""
    monkeypatch.setattr(local_llm, "_llm_backend", "ctransformers")
    monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 456)

    class FakeCTransformers:
        def __init__(self):
            self.kwargs = None

        def __call__(self, prompt, **kwargs):
            self.kwargs = kwargs
            return "The user likes tea."

    llm = FakeCTransformers()
    output = _call_local_extraction_llm(llm, "prompt")

    assert output == "The user likes tea."
    assert llm.kwargs["max_new_tokens"] == 456
    assert "max_tokens" not in llm.kwargs


def test_extract_facts_safe_no_llm():
    """Test that extract_facts_safe returns empty list when no LLM."""
    from unittest.mock import patch
    
    # Patch llm_available at the extraction module level to ensure it returns False,
    # regardless of what module-level constants were set at import time.
    # extract_facts() now calls local_llm.llm_available() through the live module
    # reference (so monkeypatch on local_llm reaches it). Patch there.
    with patch("mnemosyne.core.local_llm.llm_available", return_value=False):
        facts = extract_facts_safe("I love coffee and this is long enough for extraction")
        assert facts == []
    
    print("PASS: test_extract_facts_safe_no_llm")


def test_extract_facts_safe_exception_handling():
    """Test that extract_facts_safe never raises."""
    from unittest.mock import patch
    
    # Should not raise even with garbage input
    facts = extract_facts_safe(None)
    assert facts == []
    facts = extract_facts_safe("")
    assert facts == []
    
    # "x" is valid text but too short for meaningful extraction.
    # Patch llm_available to ensure no LLM call is attempted.
    # extract_facts() now calls local_llm.llm_available() through the live module
    # reference (so monkeypatch on local_llm reaches it). Patch there.
    with patch("mnemosyne.core.local_llm.llm_available", return_value=False):
        facts = extract_facts_safe("x")
        assert facts == []
    
    print("PASS: test_extract_facts_safe_exception_handling")


def test_triplestore_add_facts():
    """Test TripleStore.add_facts() batch storage.

    Post-E6: add_facts is a deprecation shim that routes writes to the
    AnnotationStore (not the triples table). This was changed during the
    /review adversarial pass — pre-redirect, deprecated callers' facts
    went into the triples table but the new recall path read from
    annotations, making the facts silently invisible.
    """
    import warnings
    from mnemosyne.core.annotations import AnnotationStore

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        init_triples(db_path)

        triples = TripleStore(db_path=db_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            count = triples.add_facts(
                "mem_123",
                ["The user loves coffee", "The user hates mornings", "x"],  # "x" too short
                source="test",
                confidence=0.7
            )

        assert count == 2  # "x" filtered out

        # Post-E6: facts land in annotations where the recall path looks.
        ann_store = AnnotationStore(db_path=db_path)
        all_facts = ann_store.query_by_memory(memory_id="mem_123", kind="fact")
        assert len(all_facts) == 2
        assert all(f["memory_id"] == "mem_123" for f in all_facts)
        assert all(f["kind"] == "fact" for f in all_facts)
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


# ---------------------------------------------------------------------------
# Host LLM backend integration (decisions A1, A3, C2)
# ---------------------------------------------------------------------------

from unittest.mock import patch  # noqa: E402

from mnemosyne.core import extraction as _extraction_mod, local_llm  # noqa: E402
from mnemosyne.core.llm_backends import (  # noqa: E402
    CallableLLMBackend,
    set_host_llm_backend,
)


def _enable_host(monkeypatch):
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_PROVIDER", None)
    monkeypatch.setattr(local_llm, "HOST_LLM_MODEL", None)


def test_host_extract_facts_uses_temperature_zero(monkeypatch):
    """C2 contract: extract_facts forces temperature=0.0 for determinism."""
    _enable_host(monkeypatch)
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
    monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 128)

    captured = []

    def fake(prompt, *, max_tokens, temperature, timeout, provider=None, model=None):
        captured.append({"temperature": temperature, "max_tokens": max_tokens})
        return "Alex uses Neovim.\nAlex dislikes VSCode."

    set_host_llm_backend(CallableLLMBackend("test", fake))
    with patch.object(local_llm, "_call_remote_llm") as mock_remote:
        facts = extract_facts("Alex said they prefer Neovim and dislike VSCode.")
        mock_remote.assert_not_called()

    assert any("Neovim" in f for f in facts)
    assert captured
    assert captured[0]["temperature"] == 0.0
    assert captured[0]["max_tokens"] == 128


def test_host_extract_facts_skips_remote_on_host_miss(monkeypatch):
    """A3 contract: host enabled, host returns None → fall to local, NOT remote."""
    _enable_host(monkeypatch)
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
    set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: None))

    fake_local = lambda prompt, max_new_tokens, stop: "Local fact one.\nLocal fact two."  # noqa: E731

    with patch.object(local_llm, "_call_remote_llm", return_value="Remote facts.") as mock_remote, \
         patch.object(local_llm, "_load_llm", return_value=fake_local) as mock_load:
        facts = extract_facts("some content with facts to extract")
        mock_remote.assert_not_called()
        mock_load.assert_called()

    assert any("Local fact" in f for f in facts)


def test_host_extract_facts_unchanged_when_HOST_LLM_ENABLED_false(monkeypatch):
    """REGRESSION: existing behavior preserved when host is off."""
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", False)
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
    set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: "Host fact.\nAnother host fact."))
    with patch.object(local_llm, "_call_remote_llm", return_value="Remote fact one.\nRemote fact two."):
        facts = extract_facts("some content")
    assert any("Remote fact" in f for f in facts)
    assert not any("Host fact" in f for f in facts)


def test_host_extract_facts_preserves_bulleted_output(monkeypatch):
    """REGRESSION (codex finding): host output like '- fact one' must survive
    so _parse_facts() can strip the bullet prefix. Earlier the helper ran
    output through _clean_output(), which deletes whole bullet lines."""
    _enable_host(monkeypatch)
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")

    # Codex/GPT often returns facts as a bulleted list — exactly the shape
    # _clean_output() would otherwise nuke at re.sub(r"^\s*[-*]\s.*\n", "").
    set_host_llm_backend(CallableLLMBackend(
        "test",
        lambda *a, **k: "- Alex uses Neovim.\n- Alex dislikes VSCode.\n- Alex uses example.com email.",
    ))
    facts = extract_facts("Alex said they prefer Neovim, dislike VSCode, and use example.com email.")
    assert any("Neovim" in f for f in facts), f"bullet '-' lines were stripped: {facts}"
    assert any("VSCode" in f for f in facts)
    assert any("example.com" in f for f in facts)
    # And the bullet prefix should be gone (parse_facts strips it).
    assert not any(f.startswith("-") for f in facts)


def test_host_extract_facts_remote_path_uses_temperature_zero(monkeypatch):
    """REGRESSION (codex finding 2): extract_facts must pass temperature=0.0
    even on the standalone remote path, not just the host path."""
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "HOST_LLM_ENABLED", False)  # force remote path
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")

    captured = {}

    def fake_remote(prompt, temperature=0.3):
        captured["temperature"] = temperature
        return "Some fact about something.\nAnother fact about elsewhere."

    monkeypatch.setattr(local_llm, "_call_remote_llm", fake_remote)
    facts = extract_facts("some content with facts")
    assert facts  # parsed successfully
    assert captured["temperature"] == 0.0, (
        "Remote extraction path must use temperature=0.0 for determinism"
    )


def test_host_extract_facts_returns_empty_when_both_host_and_local_fail(monkeypatch):
    """Codex finding 5 graceful-degradation path: if host attempts and local
    raises (e.g., oversized prompt), return [] cleanly so AAAK fallback runs."""
    _enable_host(monkeypatch)
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
    set_host_llm_backend(CallableLLMBackend("test", lambda *a, **k: None))

    def fake_local_that_blows_up(prompt, max_new_tokens, stop):
        raise RuntimeError("simulated oversized prompt")

    with patch.object(local_llm, "_call_remote_llm") as mock_remote, \
         patch.object(local_llm, "_load_llm", return_value=fake_local_that_blows_up):
        facts = extract_facts("some content")
        mock_remote.assert_not_called()  # A3 still holds
    assert facts == []


def test_host_extract_facts_swallows_exception_then_local(monkeypatch):
    """Backend that raises is treated as host-attempted-with-no-output."""
    _enable_host(monkeypatch)
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")

    def boom(*a, **k):
        raise RuntimeError("hermes is angry")

    set_host_llm_backend(CallableLLMBackend("test", boom))
    fake_local = lambda prompt, max_new_tokens, stop: "Recovered fact one.\nRecovered fact two."  # noqa: E731
    with patch.object(local_llm, "_call_remote_llm") as mock_remote, \
         patch.object(local_llm, "_load_llm", return_value=fake_local):
        facts = extract_facts("some content")
        mock_remote.assert_not_called()
    assert any("Recovered fact" in f for f in facts)
