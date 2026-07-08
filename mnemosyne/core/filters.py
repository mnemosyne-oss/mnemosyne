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
# Match common secret shapes without capturing the full value.
SECRET_PATTERNS: List[str] = [
    # API key prefixes (well-known services)
    r"(?:sk|pk|rk)-[a-zA-Z0-9]{20,}",  # OpenAI-style
    r"AKIA[0-9A-Z]{16}",  # AWS access key
    r"gh[pousr]_[A-Za-z0-9]{36}",  # GitHub token
    r"xox[baprs]-[A-Za-z0-9-]+",  # Slack token
    r"AIza[0-9A-Za-z_\-]{35}",  # Google API key
    r"eyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+",  # JWT
    # Generic secret assignments
    r"(?i)(?:password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key)"
    r"\s*[=:]\s*['\"]?[^\s'\"<>{}]{8,}",
    # Private key blocks
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",
    # Connection strings with credentials
    r"(?:postgres|mysql|mongodb|redis)://[^:]+:[^@]+@",
    # .env-style assignments
    r"(?i)^\s*(?:DB_PASS|SECRET_KEY|AUTH_TOKEN|API_SECRET)\s*=",
]

# Compiled pattern cache
_compiled_noise: Optional[List[re.Pattern]] = None
_compiled_secrets: Optional[List[re.Pattern]] = None


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


def _get_compiled_secrets() -> List[re.Pattern]:
    global _compiled_secrets
    if _compiled_secrets is None:
        _compiled_secrets = _compile_patterns(SECRET_PATTERNS)
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
    """Parse a comma- or newline-separated pattern string into a list."""
    if not raw:
        return []
    parts = raw.replace(",", "\n").split("\n")
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
    compiled = _get_compiled_secrets()
    # Map each compiled pattern back to a human-readable label
    labels = [
        "api_key_prefix", "aws_access_key", "github_token", "slack_token",
        "google_api_key", "jwt_token", "secret_assignment", "private_key_block",
        "connection_string_with_credentials", "env_secret_assignment",
    ]
    for i, pat in enumerate(compiled):
        if pat.search(content):
            label = labels[i] if i < len(labels) else f"secret_pattern_{i}"
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

    # --- Stage 2: noise pattern matching ---
    # Merge user-supplied patterns with curated defaults.
    patterns = list(DEFAULT_NOISE_PATTERNS)
    if ignore_patterns is not None:
        patterns.extend(ignore_patterns)
    else:
        patterns.extend(_load_ignore_patterns_from_env())

    if matches_patterns(content, patterns):
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
        sentences = content.count(". ")
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
        patterns = ignore_patterns or _load_ignore_patterns_from_env()
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

    # warn mode: always allow, but return the decision with warnings
    if mode == "warn" and decision.action == "reject":
        decision.warnings.append(
            f"Write allowed in warn mode but classified as reject: {decision.reason}"
        )
        decision.action = "allow"
        return True, decision

    return True, decision
