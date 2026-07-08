"""
Memory hygiene — noise audit and safe cleanup for existing stores.

Addresses issue #428: existing stored noise (terminal spam, command output,
heartbeats, stack traces, secrets already in the DB) that ``ignore_patterns``
cannot prevent because it was written before the filter existed, or via entry
points that bypassed it (MCP, SDK, CLI — fixed in the companion Layer 1 patch).

Two operations:

1. ``audit_noise()`` — scan ``working_memory`` + ``memories`` (primary ingest
   targets), score each row for noise likelihood + secret presence, return
   ranked candidates.  Dry-run by default.

2. ``clean_noise()`` — process audit output in batches, apply one of three
   actions (delete / archive / keep), write a full audit log to
   ``hygiene_audit_log`` table.  Requires explicit confirmation.

Both operations are deterministic (no LLM) for v1.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mnemosyne.core.filters import (
    DEFAULT_NOISE_PATTERNS,
    detect_secrets,
    matches_patterns,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema: hygiene_audit_log table
# ---------------------------------------------------------------------------

HYGIENE_LOG_DDL = """
CREATE TABLE IF NOT EXISTS hygiene_audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT NOT NULL,
    table_name TEXT NOT NULL,
    action TEXT NOT NULL,           -- 'deleted' | 'archived' | 'kept' | 'flagged'
    reason TEXT,
    noise_score REAL,
    secret_flags TEXT,              -- JSON array of secret labels
    original_content_preview TEXT,  -- first 200 chars for audit trail
    original_metadata TEXT,         -- JSON of original metadata_json
    timestamp TEXT NOT NULL,
    session_id TEXT
)
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NoiseCandidate:
    """A single memory row evaluated for noise."""
    memory_id: str
    table_name: str  # "working_memory" | "memories" | "episodic_memory"
    content_preview: str  # first 200 chars
    noise_score: float  # 0.0 (valuable) to 1.0 (definitely noise)
    noise_reasons: List[str] = field(default_factory=list)
    secret_flags: List[str] = field(default_factory=list)
    importance: float = 0.5
    source: str = ""
    timestamp: str = ""
    suggested_action: str = "keep"  # "delete" | "archive" | "keep" | "flag"
    content_length: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AuditReport:
    """Result of an audit_noise() call."""
    total_scanned: int = 0
    candidates: List[NoiseCandidate] = field(default_factory=list)
    tables_scanned: List[str] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_scanned": self.total_scanned,
            "candidates": [c.to_dict() for c in self.candidates],
            "tables_scanned": self.tables_scanned,
            "summary": self.summary,
        }


@dataclass
class CleanResult:
    """Result of a clean_noise() call."""
    deleted: int = 0
    archived: int = 0
    kept: int = 0
    flagged: int = 0
    errors: List[str] = field(default_factory=list)
    log_entries: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Noise scoring
# ---------------------------------------------------------------------------

# Additional noise indicators for existing DB content (beyond the regex
# patterns in filters.py). These are structural / heuristic signals.
_NOISE_KEYWORDS = {
    "done", "ok", "yes", "no", "sure", "thanks", "got it", "acknowledged",
    "heartbeat", "ping", "pong", "tick", "tock", "alive",
}

_VALUE_KEYWORDS = {
    "prefer", "always remember", "never", "project", "environment",
    "convention", "insight", "decision", "user", "config", "setup",
    "password policy", "deployment", "architecture",
}


def _score_noise(content: str, importance: float, source: str) -> Tuple[float, List[str]]:
    """Score a single content string for noise likelihood.

    Returns (score, reasons) where score is 0.0-1.0 (higher = more likely noise)
    and reasons is a list of human-readable strings explaining the score.
    """
    reasons: List[str] = []
    score = 0.0

    if not content or not content.strip():
        return 1.0, ["empty_content"]

    content_lower = content.lower().strip()

    # 1. Regex pattern match (reuse filters.DEFAULT_NOISE_PATTERNS)
    if matches_patterns(content, DEFAULT_NOISE_PATTERNS):
        score = max(score, 0.8)
        reasons.append("noise_pattern_match")

    # 2. Secret detection
    secrets = detect_secrets(content)
    if secrets:
        score = max(score, 0.9)
        reasons.append(f"secret_detected:{','.join(secrets)}")

    # 3. Very short / trivial content
    if len(content_lower) < 15 and content_lower in _NOISE_KEYWORDS:
        score = max(score, 0.7)
        reasons.append("trivial_keyword")

    # 4. Terminal output markers
    terminal_markers = ["collecting ", "downloading ", "installing ", "requirement already",
                        "successfully installed", "npm warn", "npm error",
                        "total ", "drwx", "-rw-r--r--"]
    if any(m in content_lower for m in terminal_markers):
        score = max(score, 0.85)
        reasons.append("terminal_output")

    # 5. Stack trace markers
    if "traceback" in content_lower or "  file \"" in content:
        score = max(score, 0.85)
        reasons.append("stack_trace")

    # 6. High line count + low semantic structure (likely a dump)
    line_count = content.count("\n") + 1
    if line_count > 30 and len(content) > 1000:
        sentences = content.count(". ")
        if sentences < line_count * 0.1:
            score = max(score, 0.65)
            reasons.append("likely_dump")

    # 7. Low importance penalty (existing importance field)
    if importance < 0.2:
        score = max(score, 0.5)
        reasons.append("low_importance")

    # 8. Value signals (reduce score)
    if any(kw in content_lower for kw in _VALUE_KEYWORDS):
        score = min(score, 0.3)
        reasons.append("value_keyword_present")

    # 9. Source-based heuristic
    if source in ("heartbeat", "cron", "debug", "terminal"):
        score = max(score, 0.7)
        reasons.append(f"noisy_source:{source}")

    return score, reasons


def _suggest_action(score: float, secret_flags: List[str]) -> str:
    """Suggest an action based on noise score and secret presence."""
    if secret_flags:
        return "flag"  # secrets get flagged for operator review, never auto-deleted
    if score >= 0.8:
        return "delete"
    if score >= 0.5:
        return "archive"
    return "keep"


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def _ensure_hygiene_log_table(conn: sqlite3.Connection) -> None:
    """Create the hygiene_audit_log table if it doesn't exist."""
    conn.execute(HYGIENE_LOG_DDL)
    conn.commit()


def _scan_table(
    conn: sqlite3.Connection,
    table_name: str,
    limit: int,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Scan a table for noise candidates. Returns rows as dicts."""
    cursor = conn.cursor()
    # We deliberately select all columns we need; the schemas for
    # working_memory, memories, and episodic_memory all share the core
    # (id, content, source, timestamp, session_id, importance, metadata_json)
    # shape, with episodic_memory having extra columns we don't need.
    cursor.execute(
        f"SELECT id, content, source, timestamp, session_id, importance, metadata_json "
        f"FROM {table_name} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def audit_noise(
    db_path: Path,
    limit: int = 200,
    tables: Optional[List[str]] = None,
    min_score: float = 0.3,
) -> AuditReport:
    """Audit a memory database for noise.

    Scans ``working_memory`` and ``memories`` (primary ingest targets) by
    default.  Returns ranked candidates sorted by noise score descending.

    Args:
        db_path: Path to the mnemosyne SQLite database.
        limit: Maximum rows to scan per table.
        tables: Override which tables to scan. Default: working_memory + memories.
        min_score: Only include candidates with noise_score >= this threshold.

    Returns:
        ``AuditReport`` with ranked candidates.
    """
    if tables is None:
        tables = ["working_memory", "memories"]

    report = AuditReport(tables_scanned=tables)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        for table in tables:
            # Check table exists
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            )
            if cursor.fetchone() is None:
                logger.debug("Table %s does not exist, skipping", table)
                continue

            rows = _scan_table(conn, table, limit)
            report.total_scanned += len(rows)

            for row in rows:
                content = row.get("content", "") or ""
                importance = row.get("importance", 0.5) or 0.5
                source = row.get("source", "") or ""

                score, reasons = _score_noise(content, importance, source)
                secrets = detect_secrets(content)

                if score < min_score and not secrets:
                    continue

                suggested = _suggest_action(score, secrets)
                candidate = NoiseCandidate(
                    memory_id=row.get("id", ""),
                    table_name=table,
                    content_preview=content[:200],
                    noise_score=round(score, 4),
                    noise_reasons=reasons,
                    secret_flags=secrets,
                    importance=importance,
                    source=source,
                    timestamp=row.get("timestamp", "") or "",
                    suggested_action=suggested,
                    content_length=len(content),
                )
                report.candidates.append(candidate)
    finally:
        conn.close()

    # Sort by noise score descending (most likely noise first)
    report.candidates.sort(key=lambda c: c.noise_score, reverse=True)

    # Build summary
    report.summary = {
        "total_candidates": len(report.candidates),
        "by_action": {},
        "with_secrets": sum(1 for c in report.candidates if c.secret_flags),
    }
    for c in report.candidates:
        report.summary["by_action"][c.suggested_action] = \
            report.summary["by_action"].get(c.suggested_action, 0) + 1

    return report


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def clean_noise(
    db_path: Path,
    candidates: List[NoiseCandidate],
    action: str = "archive",
    confirm: bool = False,
    dry_run: bool = True,
) -> CleanResult:
    """Process noise candidates and apply cleanup actions.

    Args:
        db_path: Path to the mnemosyne SQLite database.
        candidates: Candidates from ``audit_noise()``.
        action: Override action for all candidates: ``delete``, ``archive``,
            ``flag``, or ``keep``.  If empty, uses each candidate's
            ``suggested_action``.
        confirm: Must be True for any destructive action.  If False,
            only dry-run analysis is performed.
        dry_run: If True, no changes are made; returns what *would* happen.

    Returns:
        ``CleanResult`` with counts and any errors.
    """
    result = CleanResult()

    if dry_run:
        for c in candidates:
            effective_action = action if action != "keep" else c.suggested_action
            if effective_action == "delete":
                result.deleted += 1
            elif effective_action == "archive":
                result.archived += 1
            elif effective_action == "flag":
                result.flagged += 1
            else:
                result.kept += 1
        return result

    if not confirm:
        result.errors.append("Confirmation required for non-dry-run cleanup. Pass confirm=True.")
        return result

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    try:
        _ensure_hygiene_log_table(conn)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for c in candidates:
            effective_action = action if action != "keep" else c.suggested_action

            try:
                # Fetch original content + metadata for audit log
                cursor.execute(
                    f"SELECT content, metadata_json FROM {c.table_name} WHERE id = ?",
                    (c.memory_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    result.errors.append(f"Row not found: {c.table_name}:{c.memory_id}")
                    continue

                original_content = row["content"] or ""
                original_metadata = row["metadata_json"] or "{}"

                if effective_action == "delete":
                    cursor.execute(
                        f"DELETE FROM {c.table_name} WHERE id = ?",
                        (c.memory_id,),
                    )
                    result.deleted += 1
                    log_action = "deleted"
                elif effective_action == "archive":
                    # Archive = set importance to 0 and add metadata flag.
                    # This decays the row out of active retrieval without
                    # hard delete. Reversible.
                    meta = json.loads(original_metadata) if original_metadata else {}
                    meta["_archived"] = True
                    meta["_archived_at"] = now
                    meta["_archive_reason"] = c.noise_reasons
                    cursor.execute(
                        f"UPDATE {c.table_name} SET importance = 0, metadata_json = ? WHERE id = ?",
                        (json.dumps(meta), c.memory_id),
                    )
                    result.archived += 1
                    log_action = "archived"
                elif effective_action == "flag":
                    # Flag = mark in metadata for operator review. No content change.
                    meta = json.loads(original_metadata) if original_metadata else {}
                    meta["_hygiene_flagged"] = True
                    meta["_hygiene_flag_reason"] = c.secret_flags or c.noise_reasons
                    cursor.execute(
                        f"UPDATE {c.table_name} SET metadata_json = ? WHERE id = ?",
                        (json.dumps(meta), c.memory_id),
                    )
                    result.flagged += 1
                    log_action = "flagged"
                else:
                    result.kept += 1
                    log_action = "kept"

                # Write audit log entry
                cursor.execute(
                    """INSERT INTO hygiene_audit_log
                       (memory_id, table_name, action, reason, noise_score,
                        secret_flags, original_content_preview, original_metadata,
                        timestamp, session_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c.memory_id,
                        c.table_name,
                        log_action,
                        json.dumps(c.noise_reasons),
                        c.noise_score,
                        json.dumps(c.secret_flags),
                        original_content[:200],
                        original_metadata,
                        now,
                        None,  # session_id not tracked at log level
                    ),
                )
                result.log_entries += 1

            except Exception as e:
                result.errors.append(f"Error processing {c.table_name}:{c.memory_id}: {e}")
                logger.warning("Hygiene cleanup error for %s:%s: %s",
                               c.table_name, c.memory_id, e)

        conn.commit()
    except Exception as e:
        conn.rollback()
        result.errors.append(f"Transaction failed: {e}")
    finally:
        conn.close()

    return result


# ---------------------------------------------------------------------------
# Restore (reversibility)
# ---------------------------------------------------------------------------

def restore_archived(
    db_path: Path,
    log_entry_ids: Optional[List[int]] = None,
    limit: int = 100,
) -> int:
    """Restore archived memories by reading the hygiene_audit_log.

    Only restores ``archived`` entries (not ``deleted`` — those are gone,
    but the audit log preserves the original content preview + metadata
    for manual reconstruction if needed).

    Args:
        db_path: Path to the database.
        log_entry_ids: Specific log entry IDs to restore. If None, restores
            the most recent ``limit`` archived entries.
        limit: Max entries to restore when log_entry_ids is None.

    Returns:
        Number of entries restored.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    restored = 0

    try:
        cursor = conn.cursor()

        if log_entry_ids:
            placeholders = ",".join("?" * len(log_entry_ids))
            cursor.execute(
                f"SELECT * FROM hygiene_audit_log WHERE id IN ({placeholders}) AND action = 'archived'",
                log_entry_ids,
            )
        else:
            cursor.execute(
                "SELECT * FROM hygiene_audit_log WHERE action = 'archived' ORDER BY id DESC LIMIT ?",
                (limit,),
            )

        entries = cursor.fetchall()

        for entry in entries:
            table = entry["table_name"]
            memory_id = entry["memory_id"]
            original_meta = entry["original_metadata"] or "{}"

            # Restore: clear archive flags, restore original metadata
            meta = json.loads(original_meta)
            meta.pop("_archived", None)
            meta.pop("_archived_at", None)
            meta.pop("_archive_reason", None)

            cursor.execute(
                f"UPDATE {table} SET importance = ?, metadata_json = ? WHERE id = ?",
                (0.5, json.dumps(meta), memory_id),
            )
            if cursor.rowcount > 0:
                restored += 1

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error("Restore failed: %s", e)
    finally:
        conn.close()

    return restored
