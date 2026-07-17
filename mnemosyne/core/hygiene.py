"""
Memory hygiene — noise audit and safe cleanup for existing stores.

Addresses issue #428: existing stored noise (terminal spam, command output,
heartbeats, stack traces, secrets already in the DB) that ``ignore_patterns``
cannot prevent because it was written before the filter existed, or via entry
points that bypassed it (MCP, SDK, CLI — fixed in the companion Layer 1 patch).

Two operations:

1. ``audit_noise()`` — scan ``working_memory`` + ``memories`` +
   ``episodic_memory`` (when present), score each row for noise likelihood +
   secret presence, return ranked candidates.  Dry-run by default.

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
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

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
    summary: Dict[str, Any] = field(default_factory=dict)

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

    # 8. Value signals (reduce score) — BUT skip dampening if secrets
    # were detected, so secret-bearing content stays high-scored and
    # surfaces in the audit report instead of being hidden.
    if not secrets:
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


def _quote_identifier(identifier: str) -> str:
    """Return a safely quoted SQLite identifier or raise ValueError."""
    if not identifier or identifier[0].isdigit() or not all(c.isalnum() or c == "_" for c in identifier):
        raise ValueError(f"Invalid table identifier: {identifier}")
    return f'"{identifier}"'


def _scan_table(
    conn: sqlite3.Connection,
    table_name: str,
    limit: int,
    offset: int = 0,
    *,
    after: Optional[Tuple[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Scan a table for noise candidates. Returns rows as dicts."""
    cursor = conn.cursor()
    quoted_table = _quote_identifier(table_name)
    # Audit scoring needs only this fixed contract. Metadata is intentionally
    # excluded so a read-only doctor scan never loads raw metadata.
    base_query = (
        f"SELECT id, content, source, timestamp, session_id, importance "
        f"FROM {quoted_table}"
    )
    if after is None:
        cursor.execute(
            f"{base_query} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
    else:
        # Keyset pagination avoids repeatedly walking a growing OFFSET on large
        # live tables; callers should process each batch before asking for more.
        after_ts, after_id = after
        cursor.execute(
            f"{base_query} "
            "WHERE timestamp < ? OR (timestamp = ? AND id < ?) "
            "ORDER BY timestamp DESC, id DESC LIMIT ?",
            (after_ts, after_ts, after_id, limit),
        )
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    )
    return cursor.fetchone() is not None


def _default_audit_tables() -> List[str]:
    return ["working_memory", "memories", "episodic_memory"]


def _build_audit_summary(
    candidates: List[NoiseCandidate], *, table_counts: Dict[str, int]
) -> Dict[str, Any]:
    by_action: Dict[str, int] = {}
    by_table: Dict[str, int] = {table: 0 for table in table_counts}
    for c in candidates:
        by_action[c.suggested_action] = by_action.get(c.suggested_action, 0) + 1
        by_table[c.table_name] = by_table.get(c.table_name, 0) + 1
    return {
        "total_candidates": len(candidates),
        "by_action": by_action,
        "by_table": by_table,
        "table_counts": table_counts,
        "with_secrets": sum(1 for c in candidates if c.secret_flags),
    }


def audit_noise(
    db_path: Path,
    limit: int = 200,
    tables: Optional[List[str]] = None,
    min_score: float = 0.3,
    *,
    offset: int = 0,
    scan_all: bool = False,
    batch_size: int = 1000,
    conn: Optional[sqlite3.Connection] = None,
    content_preview_transform: Optional[Callable[[str], str]] = None,
) -> AuditReport:
    """Audit a memory database for noise.

    Scans ``working_memory``, ``memories``, and ``episodic_memory`` by
    default when those tables exist. Returns ranked candidates sorted by noise
    score descending. The operation is read-only.

    Args:
        db_path: Path to the mnemosyne SQLite database.
        limit: Maximum rows to scan per table unless ``scan_all`` is true.
        tables: Override which tables to scan.
        min_score: Only include candidates with noise_score >= this threshold.
        offset: Row offset per table for paginated scans.
        scan_all: If true, page through all rows in each selected table.
        batch_size: Batch size used when ``scan_all`` is true.
        content_preview_transform: Optional bounded preview renderer for a
            caller with a stricter content-safety contract.

    Returns:
        ``AuditReport`` with ranked candidates.
    """
    if tables is None:
        tables = _default_audit_tables()

    if limit < 0:
        raise ValueError("limit must be >= 0")
    if offset < 0:
        raise ValueError("offset must be >= 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    report = AuditReport(tables_scanned=tables)
    table_counts: Dict[str, int] = {}
    owns_connection = conn is None
    if conn is None:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

    def scan_batches(table_name: str) -> Iterator[List[Dict[str, Any]]]:
        if not scan_all:
            yield _scan_table(conn, table_name, limit, offset=offset)
            return

        after: Optional[Tuple[str, str]] = None
        current_offset = offset
        while True:
            batch = _scan_table(
                conn,
                table_name,
                batch_size,
                offset=current_offset if after is None else 0,
                after=after,
            )
            if not batch:
                break
            yield batch
            last = batch[-1]
            after = (last.get("timestamp", "") or "", last.get("id", "") or "")
            current_offset = 0

    try:
        for table in tables:
            _quote_identifier(table)
            if not _table_exists(conn, table):
                logger.debug("Table %s does not exist, skipping", table)
                table_counts[table] = 0
                continue

            table_scanned = 0
            for rows in scan_batches(table):
                table_scanned += len(rows)
                report.total_scanned += len(rows)

                for row in rows:
                    content = row.get("content", "") or ""
                    importance_raw = row.get("importance")
                    importance = 0.5 if importance_raw is None else importance_raw
                    source = row.get("source", "") or ""

                    score, reasons = _score_noise(content, importance, source)
                    secrets = detect_secrets(content)

                    if score < min_score and not secrets:
                        continue

                    suggested = _suggest_action(score, secrets)
                    candidate = NoiseCandidate(
                        memory_id=row.get("id", ""),
                        table_name=table,
                        content_preview=(
                            content_preview_transform(content)
                            if content_preview_transform is not None
                            else content[:200]
                        ),
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
            table_counts[table] = table_scanned
    finally:
        if owns_connection:
            conn.close()

    report.candidates.sort(key=lambda c: c.noise_score, reverse=True)
    report.summary = _build_audit_summary(report.candidates, table_counts=table_counts)
    return report


def noise_summary(
    db_path: Path,
    limit: int = 200,
    tables: Optional[List[str]] = None,
    min_score: float = 0.3,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Return a PII-safe noise summary without content previews."""
    report = audit_noise(
        db_path=db_path,
        limit=limit,
        tables=tables,
        min_score=min_score,
        conn=conn,
    )
    table_counts = report.summary.get("table_counts", {})
    candidates_by_table = report.summary.get("by_table", {})
    ratios = {
        table: (round(candidates_by_table.get(table, 0) / count, 4) if count else 0.0)
        for table, count in table_counts.items()
    }
    return {
        "status": "ok",
        "tables_scanned": report.tables_scanned,
        "total_scanned": report.total_scanned,
        "total_candidates": len(report.candidates),
        "candidate_ratio": round(len(report.candidates) / report.total_scanned, 4) if report.total_scanned else 0.0,
        "by_table": candidates_by_table,
        "table_counts": table_counts,
        "candidate_ratio_by_table": ratios,
        "by_action": report.summary.get("by_action", {}),
        "with_secrets": report.summary.get("with_secrets", 0),
        "min_score": min_score,
        "limit_per_table": limit,
    }


def doctor_hygiene_summary(
    db_path: Path,
    limit: int = 200,
    candidate_limit: int = 20,
    min_score: float = 0.3,
    *,
    conn: Optional[sqlite3.Connection] = None,
) -> Dict[str, Any]:
    """Return a bounded, redacted hygiene view for the read-only doctor.

    A caller may supply the doctor's ``mode=ro``/``query_only`` connector.
    This helper owns no mutation path and omits raw metadata, embeddings,
    BLOBs, sources, and timestamps from candidates.
    """

    if not isinstance(candidate_limit, int) or isinstance(candidate_limit, bool) or candidate_limit < 0:
        raise ValueError("candidate_limit must be a non-negative integer")
    from mnemosyne.doctor import safe_preview

    try:
        report = audit_noise(
            db_path=db_path,
            limit=limit,
            min_score=min_score,
            conn=conn,
            content_preview_transform=lambda content: safe_preview(content, max_length=120),
        )
    except sqlite3.Error:
        return {"status": "unavailable", "error_class": "sqlite_error", "candidates": []}
    except Exception:
        return {"status": "unknown", "error_class": "runtime_error", "candidates": []}

    summary = report.summary
    return {
        "status": "ok",
        "total_scanned": report.total_scanned,
        "total_candidates": len(report.candidates),
        "candidate_ratio": round(len(report.candidates) / report.total_scanned, 4) if report.total_scanned else 0.0,
        "with_secrets": summary.get("with_secrets", 0),
        "limit_per_table": limit,
        "candidate_limit": candidate_limit,
        "candidates": [
            {
                "table": candidate.table_name,
                "noise_score": candidate.noise_score,
                "reasons": list(candidate.noise_reasons),
                "secret_flags": list(candidate.secret_flags),
                "suggested_action": candidate.suggested_action,
                "preview": safe_preview(candidate.content_preview, max_length=120),
            }
            for candidate in report.candidates[:candidate_limit]
        ],
    }


def hygiene_status(
    db_path: Path, *, include_noise_summary: bool = True, limit: int = 200
) -> Dict[str, Any]:
    """Return PII-safe hygiene status and optional current noise summary."""
    status: Dict[str, Any] = {"status": "ok", "audit_log": {}}
    conn = sqlite3.connect(str(db_path))
    try:
        if _table_exists(conn, "hygiene_audit_log"):
            total = int(conn.execute("SELECT COUNT(*) FROM hygiene_audit_log").fetchone()[0])
            by_action = {
                row[0]: int(row[1])
                for row in conn.execute(
                    "SELECT action, COUNT(*) FROM hygiene_audit_log GROUP BY action"
                ).fetchall()
            }
            last_ts = conn.execute("SELECT MAX(timestamp) FROM hygiene_audit_log").fetchone()[0]
            status["audit_log"] = {
                "present": True,
                "total_entries": total,
                "by_action": by_action,
                "last_timestamp": last_ts,
            }
        else:
            status["audit_log"] = {"present": False, "total_entries": 0, "by_action": {}}
    finally:
        conn.close()

    if include_noise_summary:
        status["noise_summary"] = noise_summary(db_path=db_path, limit=limit)
    return status


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
                    # hard delete. Reversible — original importance is
                    # preserved in metadata so restore_archived can recover
                    # the exact value instead of guessing.
                    meta = json.loads(original_metadata) if original_metadata else {}
                    # Fetch and preserve original importance before zeroing.
                    cursor.execute(
                        f"SELECT importance FROM {c.table_name} WHERE id = ?",
                        (c.memory_id,),
                    )
                    imp_row = cursor.fetchone()
                    original_importance = imp_row["importance"] if imp_row else 0.5
                    meta["_archived"] = True
                    meta["_archived_at"] = now
                    meta["_archive_reason"] = c.noise_reasons
                    meta["_original_importance"] = original_importance
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

            # Read the CURRENT row metadata (which has _original_importance
            # stored during archive), not the audit log's original_metadata
            # snapshot (which is pre-archive and lacks the saved importance).
            cursor.execute(
                f"SELECT metadata_json FROM {table} WHERE id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                continue
            current_meta = row["metadata_json"] or "{}"
            meta = json.loads(current_meta)

            # Use the preserved _original_importance from metadata if
            # available; fall back to 0.5 for entries archived before
            # the importance-preservation fix was deployed.
            saved_importance = meta.pop("_original_importance", 0.5)
            meta.pop("_archived", None)
            meta.pop("_archived_at", None)
            meta.pop("_archive_reason", None)

            cursor.execute(
                f"UPDATE {table} SET importance = ?, metadata_json = ? WHERE id = ?",
                (saved_importance, json.dumps(meta), memory_id),
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
