"""
Memory write filter pipeline — core-level noise prevention.

Placed in ``mnemosyne.core`` so every entry point (Hermes provider, MCP server,
Python SDK, CLI) benefits, not just the Hermes plugin layer.

The pipeline has two stages:

1. **Regex ignore patterns** — the existing ``ignore_patterns`` mechanism,
   extracted to core so it is reusable by all callers.  Matches via
   ``re.search(pattern, content, re.IGNORECASE)``.

2. **Secret detection** — flags content that looks like API keys, tokens, or
   passwords.  Does not delete; returns a ``WriteDecision`` with
   ``action="reject"`` and ``reason="secret_detected"`` so the caller can
   decide how to surface it.

For v1 this is deterministic only — no LLM calls.  The ``classify_memory_write``
function returns a structured ``WriteDecision`` that callers inspect before
persisting.

Config is read from env vars (mirroring the pattern in ``beam.py``):

- ``MNEMOSYNE_IGNORE_PATTERNS`` — newline- or comma-separated regex patterns
- ``MNEMOSYNE_WRITE_CLASSIFIER`` — ``off`` (default), ``warn``, or ``strict``

When ``off``, the classifier is a no-op and existing ``remember()`` behavior is
unchanged.  ``warn`` proceeds with the write but returns decision metadata.
``strict`` rejects writes classified as ``reject``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated default patterns
# ---------------------------------------------------------------------------

# Noise patterns — terminal spam, command output, heartbeats, stack traces,
# cron noise, transient status.  These are the common offenders identified
# from community discussion and existing ``memoria_audit.py`` classifiers.
DEFAULT_NOISE_PATTERNS: List[str] = [
    # --- Terminal / shell command output ---
    r"^\s*(\$|>|#)\s*(pip|npm|npx|yarn|cargo|brew|apt|dnf|pacman)\s",
    r"^\s*(Collecting|Downloading|Installing|Building|Successfully installed)",
    r"^\s*Requirement already satisfied",
    r"^\s*(added|removed|changed)\s+\d+\s+package",
    r"^\s*(npm warn|npm error|npm notice)",
    r"^\s*(total\s+\d+|drwx|-\w+-\w+\s)",  # ls -la output
    r"^\s*(Macintosh|Windows)\s*$",  # uname header lines
    # --- Heartbeats / cron noise ---
    r"^\[?(heartbeat|ping|pong|alive|ok)\]?$",
    r"^\s*(tick|tock)\s*$",
    r"^cron\s+(started|completed|skipped|tick)",
    # --- Stack traces / debug logs ---
    r"^Traceback \(most recent call last\):",
    r"^\s+File \"[^\"]+\", line \d+",
    r"^\s+(raise|return)\s+\w+Error",
    r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s+\d{4}-\d{2}-\d{2}",
    r"^\s*at\s+.*\(.+:\d+:\d+\)",  # JS-style stack frames
    # --- Transient status / task progress ---
    r"^(Phase|Step|Stage)\s+\d+\s+(done|complete|started|pending)",
    r"^(PR|Issue|Commit|Merge)\s*#\d+\s+(fixed|done|merged|closed)",
    r"^\s*(TODO|FIXME|HACK|XXX)\b",
    # --- Empty / trivial ---
    r"^\s*$",  # empty content
    r"^(ok|done|yes|no|sure|thanks|got it)\.?$",
]

# Secret patterns — API keys, tokens, passwords, private keys.
# Each entry is a (label, regex) pair so labels stay attached to their
# pattern regardless of ordering — avoids the parallel-list desync bug.
SECRET_PATTERNS: List[str] = [
    r"(?:sk|pk|rk)-[a-zA-Z0-9]{20,}",  # OpenAI-style
    r"AKIA[0-9A-Z]{16}",  # AWS access key
    r"gh[pousr]_[A-Za-z0-9]{36}",  # GitHub token
    r"xox[baprs]-[A-Za-z0-9-]+",  # Slack token
    r"AIza[0-9A-Za-z_\-]{35}",  # Google API key
    r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",  # JWT
    r"(?i)(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)"
    r"\s*[=:]\s*['\"]?[^\s'\"<>{}]{8,}",  # generic assignment
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
    r"(?:postgres|mysql|mongodb|redis)://[^:]+:[^@]+@",
    r"(?i)^\s*(?:DB_PASS|SECRET_KEY|AUTH_TOKEN|API_SECRET)\s*=",
]

# CJK character ranges treated as natural prose, not credential values.
# A value containing CJK characters is almost always ordinary text
# ("密码：建议每90天更换一次" is a policy note, not a secret), so the
# credential-value predicate below excludes these ranges.
_CJK_PROSE_RANGES = (
    "\u4e00-\u9fff"  # Han ideographs (Chinese, Japanese, Korean hanja)
    "\u3040-\u30ff"  # Hiragana + Katakana (Japanese)
    "\uac00-\ud7af"  # Hangul syllables (Korean)
    "\u3000-\u303f"  # CJK symbols / punctuation
    "\uff00-\uffef"  # Fullwidth forms
    "\u3400-\u4dbf"  # CJK Extension A
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\U00020000-\U0003347f"  # CJK Extensions B-J (non-BMP Han)
    "\u1100-\u11ff"  # Hangul Jamo
    "\u3130-\u318f"  # Hangul Compatibility Jamo
    "\u31f0-\u31ff"  # Katakana Phonetic Extensions
    "\u2e80-\u2eff"  # CJK Radicals Supplement
    "\u2f00-\u2fdf"  # Kangxi Radicals
)

# CJK prose characters that must not directly follow a credential token:
# ideographs, kana, hangul, radicals, and word-like fullwidth forms. Excludes
# CJK/fullwidth punctuation so "，请勿外传" after a token is still detected.
_CJK_IDEOGRAPHIC = (
    "\u4e00-\u9fff"  # Han ideographs
    "\u3040-\u30ff"  # Hiragana + Katakana
    "\uff66-\uff9f"  # Halfwidth Katakana
    "\uff10-\uff19"  # Fullwidth digits
    "\uff21-\uff3a"  # Fullwidth Latin uppercase
    "\uff41-\uff5a"  # Fullwidth Latin lowercase
    "\uffa0-\uffbe"  # Halfwidth Hangul filler + letters
    "\uffbf-\uffc1"  # Reserved allocation gap
    "\uffc2-\uffc7"  # Halfwidth Hangul letters
    "\uffc8-\uffc9"  # Reserved allocation gap
    "\uffca-\uffcf"  # Halfwidth Hangul letters
    "\uffd0-\uffd1"  # Reserved allocation gap
    "\uffd2-\uffd7"  # Halfwidth Hangul letters
    "\uffd8-\uffd9"  # Reserved allocation gap
    "\uffda-\uffdc"  # Halfwidth Hangul letters
    "\uac00-\ud7af"  # Hangul syllables
    "\u1100-\u11ff"  # Hangul Jamo
    "\u3130-\u318f"  # Hangul Compatibility Jamo
    "\u31f0-\u31ff"  # Katakana Phonetic Extensions
    "\u3005-\u3007"  # Ideographic iteration/closing marks and number zero
    "\u3400-\u4dbf"  # CJK Extension A
    "\uf900-\ufaff"  # CJK Compatibility Ideographs
    "\U00020000-\U0003347f"  # CJK Extensions B-J (non-BMP Han)
    "\u2e80-\u2eff"  # CJK Radicals Supplement
    "\u2f00-\u2fdf"  # Kangxi Radicals
)

# Credential-value predicate: 8+ characters, no whitespace / quotes / brackets,
# no CJK characters, and at least one ASCII letter or digit. The token is the
# contiguous non-CJK run after the separator, and it must not be directly
# followed by CJK prose, so mixed prose like "abc12345我的密码" is never treated
# as a secret while "s3cr3t_pa55word_x1y2z3w4" (even with trailing "，请勿外传")
# still is.
_CREDENTIAL_VALUE = (
    r"(?=(?:(?![" + _CJK_PROSE_RANGES + r"])[^\s'\"<>{}])*(?-i:[A-Za-z0-9]))"
    r"(?:(?![" + _CJK_PROSE_RANGES + r"])[^\s'\"<>{}]){8,}"
    r"(?![" + _CJK_IDEOGRAPHIC + r"])"
)

# Curated CJK secret labels (exact labels only, never generic substrings).
_CJK_SECRET_LABELS = (
    "密码|密钥|令牌|口令|私钥|"        # Chinese
    "パスワード|秘密鍵|トークン|"     # Japanese
    "비밀번호|키"                     # Korean
)

# Paired (label, regex) structure — the single source of truth.
SECRET_LABELED_PATTERNS: List[tuple] = [
    ("api_key_prefix", r"(?:sk|pk|rk)-[a-zA-Z0-9]{20,}"),
    ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
    ("github_token", r"gh[pousr]_[A-Za-z0-9]{36}"),
    ("slack_token", r"xox[baprs]-[A-Za-z0-9-]+"),
    ("google_api_key", r"AIza[0-9A-Za-z_\-]{35}"),
    ("jwt_token", r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+"),
    ("secret_assignment", r"(?i)(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)"
                          r"\s*(?:"
                          r"[=:]\s*['\"]?[^\s'\"<>{}]{8,}|"
                          r"[：＝]\s*['\"]?" + _CREDENTIAL_VALUE + r"['\"]?)"),
    ("cjk_secret_assignment",
     r"(?:" + _CJK_SECRET_LABELS + r")"
     r"\s*[:=：＝]\s*['\"]?" + _CREDENTIAL_VALUE + r"['\"]?"),
    ("private_key_block", r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    ("connection_string_with_credentials", r"(?:postgres|mysql|mongodb|redis)://[^:]+:[^@]+@"),
    ("env_secret_assignment", r"(?i)^\s*(?:DB_PASS|SECRET_KEY|AUTH_TOKEN|API_SECRET)\s*="),
]

# A sentence boundary is either a CJK terminator or an ASCII terminator at
# whitespace/end-of-content. The latter avoids treating dots in paths,
# versions, or identifiers as sentence boundaries while still recognizing
# both ``. `` and line-ending ``.``.
_SENTENCE_BOUNDARY_RE = re.compile(r"[。！？]|[.!?](?=\s|\Z)")


def _count_sentence_boundaries(content: str) -> int:
    """Count sentence-ending punctuation without treating arbitrary dots as prose."""
    return len(_SENTENCE_BOUNDARY_RE.findall(content))

# Compiled pattern cache
_compiled_noise: Optional[List[re.Pattern]] = None
_compiled_secrets: Optional[List[tuple]] = None  # list of (label, compiled_regex)


def _compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    """Compile a list of regex strings, skipping invalid ones."""
    compiled = []
    for p in patterns:
        try:
            compiled.append(re.compile(p, re.IGNORECASE))
        except re.error:
            logger.debug("Invalid pattern, skipping: %r", p)
    return compiled


def _get_compiled_noise() -> List[re.Pattern]:
    global _compiled_noise
    if _compiled_noise is None:
        _compiled_noise = _compile_patterns(DEFAULT_NOISE_PATTERNS)
    return _compiled_noise


def _get_compiled_secrets() -> List[tuple]:
    """Return compiled (label, regex) pairs, cached."""
    global _compiled_secrets
    if _compiled_secrets is None:
        _compiled_secrets = []
        for label, pattern in SECRET_LABELED_PATTERNS:
            try:
                _compiled_secrets.append((label, re.compile(pattern, re.IGNORECASE)))
            except re.error:
                logger.debug("Invalid secret pattern %r, skipping", pattern)
    return _compiled_secrets


# ---------------------------------------------------------------------------
# Write decision
# ---------------------------------------------------------------------------

@dataclass
class WriteDecision:
    """Result of classifying a memory write candidate.

    Mirrors the shape proposed in issue #406.
    """
    action: str  # "allow" | "reject" | "rewrite"
    target: str = "memory"  # where to route ("memory" | "none" | "scratchpad")
    reason: str = ""
    confidence: float = 1.0
    warnings: List[str] = field(default_factory=list)
    safer_content: Optional[str] = None  # set when action == "rewrite"

    def to_dict(self) -> Dict:
        return {
            "action": self.action,
            "target": self.target,
            "reason": self.reason,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "safer_content": self.safer_content,
        }


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _parse_patterns(raw: str) -> List[str]:
    """Parse a newline-separated pattern string into a list.

    Does NOT split on commas — regex patterns like ``a{2,4}`` contain commas
    as quantifier bounds.  Users separate patterns with newlines.
    """
    if not raw:
        return []
    parts = raw.split("\n")
    return [p.strip() for p in parts if p.strip()]


def _load_ignore_patterns_from_env() -> List[str]:
    """Read MNEMOSYNE_IGNORE_PATTERNS env var."""
    raw = os.environ.get("MNEMOSYNE_IGNORE_PATTERNS", "")
    return _parse_patterns(raw)


def _load_classifier_mode() -> str:
    """Read MNEMOSYNE_WRITE_CLASSIFIER env var. Returns 'off' | 'warn' | 'strict'."""
    mode = os.environ.get("MNEMOSYNE_WRITE_CLASSIFIER", "off").strip().lower()
    if mode not in ("off", "warn", "strict"):
        logger.warning("Unknown MNEMOSYNE_WRITE_CLASSIFIER=%r, defaulting to 'off'", mode)
        return "off"
    return mode


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def matches_patterns(content: str, patterns: List[str]) -> bool:
    """Check if content matches any regex pattern.

    This is the core extraction of the provider's ``_should_filter`` logic,
    available to all callers (MCP, SDK, CLI) not just the Hermes plugin.
    """
    if not patterns:
        return False
    for pattern in patterns:
        try:
            if re.search(pattern, content, re.IGNORECASE):
                return True
        except re.error:
            logger.debug("Invalid ignore pattern %r, skipping", pattern)
    return False


def detect_secrets(content: str) -> List[str]:
    """Check if content contains secret-like strings.

    Returns a list of pattern descriptions (not the matched values) for
    safe logging/reporting.  Never echoes the raw secret.
    """
    if not content:
        return []
    hits = []
    for label, compiled_pat in _get_compiled_secrets():
        if compiled_pat.search(content):
            hits.append(label)
    return hits


def classify_memory_write(
    content: str,
    ignore_patterns: Optional[List[str]] = None,
) -> WriteDecision:
    """Classify a memory write candidate.

    Deterministic only (no LLM) for v1.

    Args:
        content: The text to evaluate.
        ignore_patterns: Optional list of regex patterns.  If None, reads
            from ``MNEMOSYNE_IGNORE_PATTERNS`` env var.  Combined with
            ``DEFAULT_NOISE_PATTERNS``.

    Returns:
        ``WriteDecision`` with ``action`` indicating whether to allow,
        reject, or rewrite the write.
    """
    if not content or not content.strip():
        return WriteDecision(
            action="reject", target="none",
            reason="empty_content", confidence=1.0,
        )

    # --- Stage 1: secret detection (highest priority) ---
    secret_hits = detect_secrets(content)
    if secret_hits:
        return WriteDecision(
            action="reject", target="none",
            reason="secret_detected",
            confidence=0.95,
            warnings=[f"Secret-like pattern matched: {', '.join(secret_hits)}"],
        )

    # --- Stage 2: noise pattern matching (compiled cache) ---
    # Use the compiled default noise patterns for speed, plus any
    # user-supplied patterns from ignore_patterns or env.
    compiled_noise = _get_compiled_noise()
    for pat in compiled_noise:
        if pat.search(content):
            return WriteDecision(
                action="reject", target="none",
                reason="noise_pattern_match",
                confidence=0.8,
            )

    # User-supplied patterns (from arg or env) — not cached since they
    # change per-call.
    user_patterns = ignore_patterns if ignore_patterns is not None else _load_ignore_patterns_from_env()
    if user_patterns and matches_patterns(content, user_patterns):
        return WriteDecision(
            action="reject", target="none",
            reason="noise_pattern_match",
            confidence=0.8,
        )

    # --- Stage 3: heuristic noise signals (not regex, but structural) ---
    # High line count + low semantic structure (no sentences) is likely a dump.
    line_count = content.count("\n") + 1
    if line_count > 50 and len(content) > 1000:
        # Check if it looks like structured text (has sentences)
        sentences = _count_sentence_boundaries(content)
        if sentences < line_count * 0.1:
            return WriteDecision(
                action="reject", target="none",
                reason="likely_dump_high_linecount_low_structure",
                confidence=0.6,
            )

    return WriteDecision(action="allow", target="memory", confidence=1.0)


def should_remember(
    content: str,
    ignore_patterns: Optional[List[str]] = None,
    classifier_mode: Optional[str] = None,
) -> Tuple[bool, WriteDecision]:
    """Decide whether content should be persisted to memory.

    This is the main entry point for ``remember()`` callers.  It combines
    regex filtering with the write classifier.

    Args:
        content: The text to evaluate.
        ignore_patterns: Optional regex patterns.  If None, reads from env.
        classifier_mode: ``'off'``, ``'warn'``, or ``'strict'``.  If None,
            reads from ``MNEMOSYNE_WRITE_CLASSIFIER`` env var.

    Returns:
        ``(should_write, decision)`` — ``should_write`` is True if the
        caller should proceed with the write.  When the classifier is
        ``off``, always returns ``(True, WriteDecision(allow))`` for
        backward compatibility (the regex-only path is still checked
        via ``matches_patterns`` when patterns are supplied).

        When ``strict``, returns ``(False, decision)`` for any ``reject``.
        When ``warn``, always returns ``(True, decision)`` but the
        decision carries warnings for the caller to inspect.
    """
    mode = classifier_mode or _load_classifier_mode()

    # When classifier is off, only apply regex ignore_patterns (backward
    # compat with the provider's _should_filter behavior).
    if mode == "off":
        # Only load from env when ignore_patterns is None (not an empty list,
        # which is an intentional "disable extra patterns" override).
        patterns = ignore_patterns if ignore_patterns is not None else _load_ignore_patterns_from_env()
        if patterns and matches_patterns(content, patterns):
            return False, WriteDecision(
                action="reject", target="none",
                reason="ignore_pattern_match", confidence=1.0,
            )
        return True, WriteDecision(action="allow", target="memory")

    # warn or strict: run full classifier
    decision = classify_memory_write(content, ignore_patterns=ignore_patterns)

    if mode == "strict" and decision.action == "reject":
        return False, decision

    # warn mode: always allow, but return the decision with warnings.
    # Align target with the new allow state so downstream consumers see
    # consistent action/target.
    if mode == "warn" and decision.action == "reject":
        decision.warnings.append(
            f"Write allowed in warn mode but classified as reject: {decision.reason}"
        )
        decision.action = "allow"
        decision.target = "memory"
        return True, decision

    return True, decision
