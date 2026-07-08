"""Tests for the core write filter pipeline (Layer 1, issues #406 + #428).

Covers:
- Regex ignore pattern matching (the extracted _should_filter logic)
- Secret detection (API keys, tokens, passwords)
- classify_memory_write() decision routing
- should_remember() with classifier modes (off/warn/strict)
- Curated default patterns catch common noise
- Backward compat: classifier off = only regex patterns apply
"""

import os
import pytest

from mnemosyne.core.filters import (
    DEFAULT_NOISE_PATTERNS,
    SECRET_PATTERNS,
    WriteDecision,
    classify_memory_write,
    detect_secrets,
    matches_patterns,
    should_remember,
)


# ---------------------------------------------------------------------------
# matches_patterns
# ---------------------------------------------------------------------------

class TestMatchesPatterns:
    def test_empty_patterns_returns_false(self):
        assert matches_patterns("anything", []) is False

    def test_simple_regex_match(self):
        assert matches_patterns("pip install foo", [r"^\s*(\$|>)\s*pip\s"]) is False
        # The pattern expects a $ prefix; without it, no match
        assert matches_patterns("$ pip install foo", [r"^\s*(\$|>)\s*pip\s"]) is True

    def test_case_insensitive(self):
        assert matches_patterns("PIP INSTALL FOO", [r"pip\sinstall"]) is True

    def test_invalid_pattern_skipped(self):
        # Invalid regex should not raise
        assert matches_patterns("test", [r"[invalid", r"valid"]) is False

    def test_multiple_patterns_first_match(self):
        assert matches_patterns("heartbeat", [r"^\[?heartbeat\]?$", r"other"]) is True


# ---------------------------------------------------------------------------
# detect_secrets
# ---------------------------------------------------------------------------

class TestDetectSecrets:
    def test_openai_key(self):
        hits = detect_secrets("My key is sk-abc123def456ghi789jkl012mno345pqr678")
        assert "api_key_prefix" in hits

    def test_aws_key(self):
        hits = detect_secrets("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert "aws_access_key" in hits

    def test_github_token(self):
        hits = detect_secrets("ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
        assert "github_token" in hits

    def test_jwt(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        hits = detect_secrets(f"token: {jwt}")
        assert "jwt_token" in hits

    def test_password_assignment(self):
        hits = detect_secrets("password = hunter2supersecret")
        assert "secret_assignment" in hits

    def test_private_key_block(self):
        hits = detect_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIJKQIBAA")
        assert "private_key_block" in hits

    def test_connection_string(self):
        hits = detect_secrets("postgres://user:secretpass@localhost:5432/db")
        assert "connection_string_with_credentials" in hits

    def test_env_assignment(self):
        hits = detect_secrets("DB_PASS=supersecret123")
        assert "env_secret_assignment" in hits

    def test_no_secrets_in_clean_content(self):
        hits = detect_secrets("User prefers concise responses in English.")
        assert hits == []

    def test_never_echoes_raw_secret(self):
        raw_secret = "sk-abc123def456ghi789jkl012mno345pqr678"
        hits = detect_secrets(f"My key is {raw_secret}")
        # The hits list contains labels, not the raw secret
        for hit in hits:
            assert raw_secret not in hit

    def test_empty_content(self):
        assert detect_secrets("") == []


# ---------------------------------------------------------------------------
# classify_memory_write
# ---------------------------------------------------------------------------

class TestClassifyMemoryWrite:
    def test_allows_valuable_content(self):
        decision = classify_memory_write("User prefers concise responses in English.")
        assert decision.action == "allow"
        assert decision.target == "memory"

    def test_rejects_empty_content(self):
        decision = classify_memory_write("")
        assert decision.action == "reject"
        assert decision.reason == "empty_content"

    def test_rejects_whitespace_only(self):
        decision = classify_memory_write("   \n\t  ")
        assert decision.action == "reject"
        assert decision.reason == "empty_content"

    def test_rejects_secret(self):
        decision = classify_memory_write("My API key is sk-abc123def456ghi789jkl012mno345pqr678")
        assert decision.action == "reject"
        assert decision.reason == "secret_detected"
        assert decision.confidence >= 0.9

    def test_rejects_terminal_output(self):
        decision = classify_memory_write("$ pip install foo\nCollecting foo\nSuccessfully installed foo")
        assert decision.action == "reject"
        assert "noise_pattern_match" in decision.reason

    def test_rejects_stack_trace(self):
        content = "Traceback (most recent call last):\n  File \"test.py\", line 10, in <module>\n    raise ValueError('bad')"
        decision = classify_memory_write(content)
        assert decision.action == "reject"

    def test_rejects_heartbeat(self):
        decision = classify_memory_write("heartbeat")
        assert decision.action == "reject"

    def test_rejects_trivial_ok(self):
        decision = classify_memory_write("ok")
        assert decision.action == "reject"

    def test_rejects_large_dump(self):
        # 60 lines of non-sentence content, >1000 chars total
        content = "\n".join(["some random data line that is long enough"] * 60)
        decision = classify_memory_write(content)
        assert decision.action == "reject"
        assert "dump" in decision.reason

    def test_value_keywords_reduce_score(self):
        content = "The user prefers using pytest for testing in this project. Always remember to run tests before committing."
        decision = classify_memory_write(content)
        assert decision.action == "allow"

    def test_custom_ignore_patterns(self):
        # Custom pattern that's not in defaults
        decision = classify_memory_write("weather forecast: rain today", ignore_patterns=[r"weather\s+forecast"])
        assert decision.action == "reject"

    def test_decision_is_json_serializable(self):
        decision = classify_memory_write("test content")
        d = decision.to_dict()
        assert "action" in d
        assert "target" in d
        assert "reason" in d
        assert "confidence" in d
        assert "warnings" in d


# ---------------------------------------------------------------------------
# should_remember
# ---------------------------------------------------------------------------

class TestShouldRemember:
    def test_classifier_off_allows_normal_content(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_WRITE_CLASSIFIER", raising=False)
        monkeypatch.delenv("MNEMOSYNE_IGNORE_PATTERNS", raising=False)
        should, decision = should_remember("User prefers concise responses.")
        assert should is True
        assert decision.action == "allow"

    def test_classifier_off_regex_still_filters(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_WRITE_CLASSIFIER", raising=False)
        should, decision = should_remember(
            "$ pip install foo",
            ignore_patterns=[r"^\s*\$\s*pip\s"],
        )
        assert should is False
        assert decision.reason == "ignore_pattern_match"

    def test_strict_mode_rejects_noise(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", "strict")
        should, decision = should_remember("$ pip install foo\nCollecting foo")
        assert should is False
        assert decision.action == "reject"

    def test_strict_mode_allows_valuable(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", "strict")
        should, decision = should_remember("User prefers pytest for testing.")
        assert should is True

    def test_strict_mode_rejects_secret(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", "strict")
        should, decision = should_remember("password = hunter2supersecret")
        assert should is False
        assert "secret" in decision.reason

    def test_warn_mode_allows_but_warns(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", "warn")
        should, decision = should_remember("ok")
        assert should is True
        assert decision.action == "allow"
        assert len(decision.warnings) > 0

    def test_warn_mode_allows_valuable_silently(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", "warn")
        should, decision = should_remember("User prefers concise responses in English.")
        assert should is True
        assert decision.warnings == []

    def test_env_ignore_patterns_respected(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_WRITE_CLASSIFIER", raising=False)
        monkeypatch.setenv("MNEMOSYNE_IGNORE_PATTERNS", r"^custom\snoise")
        should, decision = should_remember("custom noise line here")
        assert should is False

    def test_invalid_classifier_mode_defaults_off(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", "bogus")
        should, decision = should_remember("normal content")
        assert should is True
