"""Memory audit log for Mnemosyne provider.

Records memory mutation events in a SQLite table co-located with the
active provider DB. Non-blocking, fire-and-forget — audit failures
never break memory operations.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    action TEXT NOT NULL,
    memory_id TEXT,
    bank TEXT,
    scope TEXT,
    profile TEXT,
    session_id TEXT,
    source_tool TEXT,
    reason TEXT,
    metadata_json TEXT
)
"""

_INSERT = """
INSERT INTO memory_events
    (timestamp, action, memory_id, bank, scope, profile, session_id, source_tool, reason, metadata_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


class AuditLog:
    """Append-only audit log backed by a SQLite table."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_table()

    def _ensure_table(self) -> None:
        try:
            self._conn = sqlite3.connect(str(self._db_path), timeout=5)
            self._conn.execute(_CREATE_TABLE)
            self._conn.commit()
        except Exception as exc:
            logger.warning("audit: failed to create table: %s", exc)
            self._conn = None

    def record(
        self,
        action: str,
        *,
        memory_id: Optional[str] = None,
        bank: Optional[str] = None,
        scope: Optional[str] = None,
        profile: Optional[str] = None,
        session_id: Optional[str] = None,
        source_tool: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record one audit event. Never raises."""
        if self._conn is None:
            return
        try:
            meta_str = json.dumps(metadata) if metadata else None
            self._conn.execute(
                _INSERT,
                (
                    time.time(),
                    action,
                    memory_id,
                    bank,
                    scope,
                    profile,
                    session_id,
                    source_tool,
                    reason,
                    meta_str,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.debug("audit: failed to record event: %s", exc)

    def query(self, limit: int = 50) -> list[Dict[str, Any]]:
        """Return recent events. For diagnostics/testing."""
        if self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                "SELECT event_id, timestamp, action, memory_id, bank, scope, "
                "profile, session_id, source_tool, reason, metadata_json "
                "FROM memory_events ORDER BY event_id DESC LIMIT ?",
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            return []

    def count(self) -> int:
        if self._conn is None:
            return 0
        try:
            return self._conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
        except Exception:
            return 0

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
