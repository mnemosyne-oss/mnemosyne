"""Fact-extraction diagnostics (C13.b).

Pre-C13.b: fact extraction had five silent-failure layers (cloud HTTP
errors, JSON parse failures, local LLM exceptions, no-LLM-available
fallback, outer try/except wrappers). Operators got zero signal that
extraction was running blind — fact-recall and the graph voice
silently degraded with no log line, no counter, no audit trail.

This module exposes a process-global counter set that records each
extraction attempt's outcome at every tier (host / remote / local /
cloud). Operators can query via `get_extraction_stats()` to see:
  - How many attempts have run since process start
  - How many succeeded at each tier
  - How many failed at each tier and why (recent error samples)
  - The current success rate

Thread-safe (a single threading.Lock gates all mutations). Process-
global because extraction calls fan out from many sites
(`extract_facts_safe`, `ExtractionClient`, the batch benchmark adapter)
and operators want one aggregate view, not per-instance fragmentation.

The diagnostics are read-only signal; they never affect extraction
behavior. Failures are still swallowed by the caller's `except`
blocks — diagnostics just surface the silence.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque, Dict, Optional


logger = logging.getLogger(__name__)


# Max recent-error samples kept per tier. Bounded queue so a noisy
# failing tier doesn't accumulate unbounded memory.
_MAX_ERROR_SAMPLES_PER_TIER = 10

# Cap on the raw error message length kept in samples — prevents
# log-flood / content leakage from upstream LLM errors that include
# the full prompt.
_ERROR_MESSAGE_CAP = 200

# Canonical extraction tiers. Each tier corresponds to a transport
# the extraction path can try: a Hermes-provided host LLM, an
# OpenAI-compatible remote endpoint, the local ctransformers GGUF
# model, or the cloud OpenRouter ExtractionClient. Operators look at
# per-tier success rates to know where to invest fix effort.
#
# `wrapper` is a synthetic tier for failures that escaped through the
# `extract_facts_safe` outer try/except whose tier of origin can't be
# determined post-hoc. /review caught the prior pattern of attributing
# every outer-wrapper failure to `local` — that inflated `local`'s
# failure count and misled operators about local-LLM health.
EXTRACTION_TIERS = ("host", "remote", "local", "cloud", "wrapper")


def _safe_for_log(value) -> str:
    """[C13.b /review] Sanitize a string for log inclusion. Strips
    control characters / newlines that a hostile or malformed
    exception `__repr__` could inject into log aggregators, and caps
    length. Returns a single-line representation."""
    if value is None:
        return ""
    s = str(value)
    # Replace newlines, tabs, and other control chars with spaces.
    sanitized = "".join(
        c if (c.isprintable() and c != "\x1b") else " "
        for c in s
    )
    return sanitized[:200]


@dataclass
class _TierStats:
    """Per-tier counters."""
    attempts: int = 0
    successes: int = 0
    no_output: int = 0  # tier ran but returned empty / no parseable facts
    failures: int = 0   # tier raised an exception
    error_samples: Deque = field(
        default_factory=lambda: deque(maxlen=_MAX_ERROR_SAMPLES_PER_TIER)
    )


class ExtractionDiagnostics:
    """Process-global extraction-attempt counters.

    Designed for low-overhead instrumentation: each method takes
    sub-microsecond on the happy path. The single `_lock` gates all
    mutations so concurrent extraction calls from different threads
    accumulate correctly. Reads (`snapshot`, `success_rate`) take the
    same lock briefly.

    Diagnostics are signal-only — they never alter extraction
    behavior. Callers continue to swallow failures via their own
    try/except blocks; this class exists so operators can see WHAT
    is being swallowed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tier_stats: Dict[str, _TierStats] = {
            tier: _TierStats() for tier in EXTRACTION_TIERS
        }
        # Bird's-eye counters spanning all tiers.
        self._total_calls = 0       # outer extract_facts() invocations
        self._total_successes = 0   # at least one tier returned facts
        self._total_failures = 0    # every tier failed (no facts AND error)
        self._total_empty = 0       # all tiers returned no facts but didn't error
        self._created_at = datetime.now().isoformat()

    @staticmethod
    def _truncate_error(msg: str) -> str:
        if msg is None:
            return ""
        s = str(msg)
        if len(s) > _ERROR_MESSAGE_CAP:
            return s[:_ERROR_MESSAGE_CAP] + "...[truncated]"
        return s

    @staticmethod
    def _validate_tier(tier: str) -> None:
        if tier not in EXTRACTION_TIERS:
            raise ValueError(
                f"unknown extraction tier {tier!r}; "
                f"valid tiers: {EXTRACTION_TIERS}"
            )

    def record_attempt(self, tier: str) -> None:
        """Record that an extraction attempt at `tier` is starting."""
        self._validate_tier(tier)
        with self._lock:
            self._tier_stats[tier].attempts += 1

    def record_success(self, tier: str, fact_count: int = 0) -> None:
        """Record that `tier` returned non-empty facts."""
        self._validate_tier(tier)
        with self._lock:
            self._tier_stats[tier].successes += 1

    def record_no_output(self, tier: str) -> None:
        """Record that `tier` ran without exception but returned no
        parseable facts. Distinguishes "LLM said nothing" from "LLM
        crashed" — operators triage these differently."""
        self._validate_tier(tier)
        with self._lock:
            self._tier_stats[tier].no_output += 1

    def record_failure(self, tier: str, exc: Optional[BaseException] = None,
                       reason: Optional[str] = None) -> None:
        """Record that `tier` failed with an exception or a named
        reason (e.g. 'json_parse_failed', 'no_api_key',
        'model_not_loaded'). At least one of `exc` / `reason` must be
        supplied so the sample is useful."""
        self._validate_tier(tier)
        with self._lock:
            stats = self._tier_stats[tier]
            stats.failures += 1
            sample = {"at": datetime.now().isoformat()}
            if exc is not None:
                sample["type"] = type(exc).__name__
                sample["msg"] = self._truncate_error(repr(exc))
            elif reason is not None:
                sample["type"] = "reason"
                sample["msg"] = self._truncate_error(reason)
            else:
                sample["type"] = "unspecified"
                sample["msg"] = ""
            # Always include `reason` when supplied so operators
            # alerting on a specific reason string can find it
            # regardless of whether an exception was also captured.
            if reason is not None:
                sample["reason"] = reason
            stats.error_samples.append(sample)

    def record_call(self, *, succeeded: bool, all_empty: bool = False) -> None:
        """Record the outcome of an outer extraction call. Once per
        `extract_facts()` invocation.

        Args:
            succeeded: True if at least one tier returned facts.
            all_empty: True if every tier ran but none returned
                facts (no exceptions). Mutually exclusive with
                succeeded. When both are False, the call counts as a
                hard failure (every tier errored or no tier ran).
        """
        with self._lock:
            self._total_calls += 1
            if succeeded:
                self._total_successes += 1
            elif all_empty:
                self._total_empty += 1
            else:
                self._total_failures += 1

    def success_rate(self) -> float:
        """Fraction of outer calls that returned facts. 0.0 when no
        calls have run yet."""
        with self._lock:
            if self._total_calls == 0:
                return 0.0
            return self._total_successes / self._total_calls

    def snapshot(self) -> Dict:
        """Return a JSON-serializable view of current state.

        Shape:
            {
              "created_at": iso-timestamp,
              "snapshot_at": iso-timestamp,
              "totals": {
                "calls": N, "successes": N, "failures": N, "empty": N,
                "success_rate": 0.0-1.0,
              },
              "by_tier": {
                "host": {
                  "attempts": N, "successes": N, "no_output": N,
                  "failures": N, "error_samples": [{...}],
                },
                ... (same shape for remote / local / cloud)
              },
            }

        Safe to call from monitoring code; takes the lock briefly,
        returns plain Python types ready for JSON serialization.
        """
        with self._lock:
            by_tier = {}
            for tier in EXTRACTION_TIERS:
                stats = self._tier_stats[tier]
                # Deep-copy the sample dicts so a caller mutating
                # the snapshot can't mutate the diagnostics' internal
                # state. /review caught the alias.
                by_tier[tier] = {
                    "attempts": stats.attempts,
                    "successes": stats.successes,
                    "no_output": stats.no_output,
                    "failures": stats.failures,
                    "error_samples": [dict(s) for s in stats.error_samples],
                }
            rate = 0.0 if self._total_calls == 0 else (
                self._total_successes / self._total_calls
            )
            return {
                "created_at": self._created_at,
                "snapshot_at": datetime.now().isoformat(),
                "totals": {
                    "calls": self._total_calls,
                    "successes": self._total_successes,
                    "failures": self._total_failures,
                    "empty": self._total_empty,
                    "success_rate": rate,
                },
                "by_tier": by_tier,
            }

    def reset(self) -> None:
        """Reset all counters. Useful for tests and for operators
        wanting a fresh measurement window. Resets `created_at` too."""
        with self._lock:
            self._tier_stats = {
                tier: _TierStats() for tier in EXTRACTION_TIERS
            }
            self._total_calls = 0
            self._total_successes = 0
            self._total_failures = 0
            self._total_empty = 0
            self._created_at = datetime.now().isoformat()


# Process-global singleton. Lazy initialization so import-time cost
# is zero; first call to `get_diagnostics()` builds the instance.
_singleton_lock = threading.Lock()
_singleton: Optional[ExtractionDiagnostics] = None


def get_diagnostics() -> ExtractionDiagnostics:
    """Return the process-global ExtractionDiagnostics instance.

    Thread-safe lazy init. Operators monitoring extraction health
    should pin this reference once and snapshot periodically.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:  # double-check under lock
                _singleton = ExtractionDiagnostics()
    return _singleton


def get_extraction_stats() -> Dict:
    """Convenience: return a snapshot of the current diagnostics.
    Equivalent to `get_diagnostics().snapshot()` — handy for
    one-shot operator queries."""
    return get_diagnostics().snapshot()


def reset_extraction_stats() -> None:
    """Convenience: reset the process-global diagnostics. Primarily
    for tests and for operators starting a fresh measurement
    window."""
    get_diagnostics().reset()
