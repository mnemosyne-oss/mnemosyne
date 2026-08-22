"""Tests for the core write filter pipeline (Layer 1, issues #406 + #428).

Covers:
- Regex ignore pattern matching (the extracted _should_filter logic)
- Secret detection (API keys, tokens, passwords)
- classify_memory_write() decision routing
- should_remember() with classifier modes (off/warn/strict)
- Curated default patterns catch common noise
- Backward compat: classifier off = only regex patterns apply
"""

import pytest

from mnemosyne.core.filters import (
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
        # nosec - test fixture, not a real secret
        hits = detect_secrets("My key is sk-abc123def456ghi789jkl012mno345pqr678")
        assert "api_key_prefix" in hits

    def test_aws_key(self):
        # nosec - test fixture, not a real secret
        hits = detect_secrets("AWS key: AKIAIOSFODNN7EXAMPLE")
        assert "aws_access_key" in hits

    def test_github_token(self):
        # nosec - test fixture, not a real secret
        hits = detect_secrets("ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123456789")
        assert "github_token" in hits

    def test_slack_token(self):
        # nosec - test fixture, not a real secret
        hits = detect_secrets("xoxb-1234567890-abcdefghij")
        assert "slack_token" in hits

    def test_google_api_key(self):
        # nosec - test fixture, not a real secret
        hits = detect_secrets("AIzaSyA1234567890abcdefghijklmnopqrstuvwxyz_")
        assert "google_api_key" in hits

    def test_jwt(self):
        # nosec - test fixture, not a real secret
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        hits = detect_secrets(f"token: {jwt}")
        assert "jwt_token" in hits

    def test_password_assignment(self):
        # nosec - test fixture, not a real secret
        hits = detect_secrets("password = hunter2supersecret")
        assert "secret_assignment" in hits

    def test_private_key_block(self):
        # nosec - test fixture, not a real secret
        hits = detect_secrets("-----BEGIN RSA PRIVATE KEY-----\nMIIJKQIBAA")
        assert "private_key_block" in hits

    def test_connection_string(self):
        # nosec - test fixture, not a real secret
        hits = detect_secrets("postgres://user:secretpass@localhost:5432/db")
        assert "connection_string_with_credentials" in hits

    def test_env_assignment(self):
        # nosec - test fixture, not a real secret
        hits = detect_secrets("DB_PASS=supersecret123")
        assert "env_secret_assignment" in hits

    def test_no_secrets_in_clean_content(self):
        hits = detect_secrets("User prefers concise responses in English.")
        assert hits == []

    def test_never_echoes_raw_secret(self):
        # nosec - test fixture, not a real secret
        raw_secret = "sk-abc123def456ghi789jkl012mno345pqr678"
        hits = detect_secrets(f"My key is {raw_secret}")
        # The hits list contains labels, not the raw secret
        for hit in hits:
            assert raw_secret not in hit

    def test_empty_content(self):
        assert detect_secrets("") == []


# ---------------------------------------------------------------------------
# detect_secrets — CJK-labelled secrets (issue #806, lane 1)
# ---------------------------------------------------------------------------

class TestDetectSecretsCJK:
    @pytest.mark.parametrize(
        ("label", "value"),
        [
            ("密码", "s3cr3t_pa55word_x1y2z3w4"),
            ("密钥", "AbCdEfGhIjKlMnOp"),
            ("令牌", "ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123"),
            ("口令", "qwerty1234567890"),
            ("私钥", "MIIEvQIBADANBgkqhkiG9w0BAQEFAASC"),
            ("パスワード", "abcdefgh1234"),
            ("秘密鍵", "Jw8kLm2pQr7sVx9z"),
            ("トークン", "token_AbCdEfGhIjKlMn"),
            ("비밀번호", "qwerty123456"),
            ("키", "key_AbCdEfGhIjKlMn"),
        ],
    )
    @pytest.mark.parametrize("separator", [":", "=", "：", "＝"])
    def test_all_cjk_labels_and_separators(self, label, value, separator):
        # nosec - test fixtures
        hits = detect_secrets(f"{label}{separator}{value}")
        assert "cjk_secret_assignment" in hits

    def test_chinese_quoted_value(self):
        # nosec - test fixture
        hits = detect_secrets('令牌："ghp_AbCdEfGhIjKlMnOpQrStUvWxYz0123"')
        assert "cjk_secret_assignment" in hits

    def test_positive_secret_with_trailing_cjk_prose(self):
        """Trailing CJK punctuation + prose must not hide a secret."""
        # nosec - test fixture
        hits = detect_secrets("数据库密码：s3cr3t_pa55word_x1y2z3w4，请勿外传")
        assert "cjk_secret_assignment" in hits

    def test_positive_secret_with_trailing_cjk_period(self):
        # nosec - test fixture
        hits = detect_secrets("数据库密码：s3cr3t_pa55word_x1y2z3w4。")
        assert "cjk_secret_assignment" in hits

    @pytest.mark.parametrize("separator", ["：", "＝"])
    def test_positive_english_label_fullwidth_separator(self, separator):
        """An English label with a fullwidth separator must be detected."""
        # nosec - test fixture
        hits = detect_secrets(f"password{separator}s3cr3t_pa55word_x1y2z3w4")
        assert "secret_assignment" in hits

    @pytest.mark.parametrize(
        "prose",
        [
            "password：建议每90天更换一次",
            "password＝建议每90天更换一次",
        ],
    )
    def test_english_label_fullwidth_separator_policy_prose_not_detected(self, prose):
        """Fullwidth separators must not make English labels match CJK prose."""
        assert detect_secrets(prose) == []

    def test_negative_chinese_policy_prose_after_label(self):
        """Ordinary Chinese policy prose after a label must not be a secret."""
        hits = detect_secrets("密码：建议每90天更换一次")
        assert hits == []

    def test_negative_chinese_architecture_prose(self):
        hits = detect_secrets("配置：采用微服务架构部署项目，并约定所有配置走环境变量")
        assert hits == []

    def test_negative_chinese_plain_name(self):
        hits = detect_secrets("名称：Alice 的昵称")
        assert hits == []

    def test_negative_value_too_short(self):
        hits = detect_secrets("数据库密码：abc1234")
        assert hits == []

    def test_negative_symbols_only_value(self):
        hits = detect_secrets("密码：！！！！！！！！")
        assert hits == []

    def test_negative_ascii_symbols_followed_by_text(self):
        hits = detect_secrets("密码：!!!!!!!! nextword")
        assert hits == []

    def test_exact_eight_character_alphanumeric_value_is_detected(self):
        # nosec - test fixture
        assert "cjk_secret_assignment" in detect_secrets("密码：Abc12345")

    def test_exact_eight_character_all_digit_value_is_detected(self):
        # nosec - test fixture
        assert "cjk_secret_assignment" in detect_secrets("密码：12345678")

    def test_exact_eight_character_ascii_symbol_value_is_rejected(self):
        assert detect_secrets("密码：!!!!!!!!") == []

    @pytest.mark.parametrize("character", ["ſ", "ı", "İ", "K"])
    def test_unicode_casefold_equivalent_is_not_ascii_alphanumeric(self, character):
        """Unicode case-fold equivalents must not satisfy the ASCII boundary."""
        assert detect_secrets(f"密码：!!!!!!!!{character}") == []

    def test_japanese_policy_prose_is_not_detected(self):
        assert detect_secrets("パスワード：定期的に変更してください") == []

    def test_korean_policy_prose_is_not_detected(self):
        assert detect_secrets("비밀번호: 90일마다변경") == []

    def test_negative_mixed_cjk_value(self):
        """A value mixing CJK prose with digits is still prose, not a secret."""
        hits = detect_secrets("数据库密码：我的密码是12345678")
        assert hits == []

    def test_negative_ascii_prefix_then_cjk(self):
        """An ASCII prefix followed by CJK prose must not be a secret."""
        hits = detect_secrets("密码：abc12345我的密码")
        assert hits == []

    @pytest.mark.parametrize(
        "char",
        [
            "\u4e00",  # Han unified (Chinese)
            "\u3042",  # Hiragana (Japanese)
            "\uac00",  # Hangul syllable (Korean)
            "\u3400",  # CJK Extension A
            "\uf900",  # CJK Compatibility Ideograph
            "\U00020000",  # CJK Extension B (non-BMP)
            "\u1100",  # Hangul Jamo
            "\u3130",  # Hangul Compatibility Jamo
            "\uff86",  # halfwidth Katakana
            "\u2e80",  # CJK Radicals Supplement
            "\u2f00",  # Kangxi Radical
            "\u3005",  # Ideographic iteration mark
            "\u3006",  # Ideographic closing mark
            "\u3007",  # Ideographic number zero
            "\u31f0",  # Katakana Phonetic Extensions
            "\U000323b0",  # CJK Extension J boundary
            "\U0003347f",  # CJK Extension J upper boundary
            "\uff10",  # fullwidth digit
            "\uff21",  # fullwidth Latin uppercase letter
            "\uff41",  # fullwidth Latin lowercase letter
            "\uffa0",  # halfwidth Hangul filler
            "\uffbf",
            "\uffc1",
            "\uffc8",
            "\uffc9",
            "\uffd0",
            "\uffd1",
            "\uffd8",
            "\uffd9",
        ],
    )
    def test_negative_ascii_prefix_then_cjk_block(self, char):
        """CJK or fullwidth prose after an ASCII prefix must reject it."""
        hits = detect_secrets(f"密码：abc12345{char}")
        assert hits == []


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

    @pytest.mark.parametrize("terminator", ["。", "！", "？", "."])
    def test_allows_multiline_sentences_with_cjk_or_line_end_punctuation(self, terminator):
        content = "\n".join(
            [f"This is a complete sentence with useful content{terminator}"] * 60
        )

        decision = classify_memory_write(content)

        assert decision.action == "allow"

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
        assert decision.action == "allow"

    def test_strict_mode_rejects_secret(self, monkeypatch):
        # nosec - test fixture
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
        assert decision.reason == "ignore_pattern_match"

    def test_invalid_classifier_mode_defaults_off(self, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_WRITE_CLASSIFIER", "bogus")
        should, decision = should_remember("normal content")
        assert should is True
        assert decision.action == "allow"

    def test_empty_list_override_disables_patterns(self, monkeypatch):
        """Passing ignore_patterns=[] should NOT load from env."""
        monkeypatch.delenv("MNEMOSYNE_WRITE_CLASSIFIER", raising=False)
        monkeypatch.setenv("MNEMOSYNE_IGNORE_PATTERNS", r"^should_match")
        # Use content that matches the env pattern but no default noise patterns
        should, decision = should_remember("should_match this content here", ignore_patterns=[])
        # With empty list override, env patterns are NOT loaded
        assert should is True
        assert decision.action == "allow"
