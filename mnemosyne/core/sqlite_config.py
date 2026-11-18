"""Shared SQLite connection configuration."""

from __future__ import annotations

import os
import sqlite3

DEFAULT_BUSY_TIMEOUT_MS = 5000


def configure_busy_timeout(conn: sqlite3.Connection) -> None:
    """Apply Mnemosyne's environment-configured SQLite busy timeout."""
    try:
        timeout_ms = int(
            os.environ.get("MNEMOSYNE_BUSY_TIMEOUT_MS", str(DEFAULT_BUSY_TIMEOUT_MS))
        )
    except ValueError:
        timeout_ms = DEFAULT_BUSY_TIMEOUT_MS
    conn.execute(f"PRAGMA busy_timeout={timeout_ms}")
