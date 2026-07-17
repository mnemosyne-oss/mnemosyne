"""
Mnemosyne Cost Logger
Tracks memory injection costs over time for benchmarking.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict

DEFAULT_LOG_DIR = Path.home() / ".mnemosyne" / "data"
DEFAULT_LOG_DB = DEFAULT_LOG_DIR / "cost_log.db"


def _get_conn(db_path: Path = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_LOG_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_cost_log(db_path: Path = None):
    conn = _get_conn(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cost_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            memory_count INTEGER,
            token_count INTEGER,
            estimated_cost_usd REAL,
            model TEXT DEFAULT 'default',
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def log_cost(session_id: str, memory_count: int, token_count: int,
             estimated_cost_usd: float, model: str = "default",
             db_path: Path = None):
    init_cost_log(db_path)
    conn = _get_conn(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO cost_entries (session_id, memory_count, token_count, estimated_cost_usd, model, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (session_id, memory_count, token_count, estimated_cost_usd, model, datetime.now().isoformat()))
    conn.commit()


def get_cost_stats(session_id: str = None, db_path: Path = None) -> Dict:
    init_cost_log(db_path)
    conn = _get_conn(db_path)
    cursor = conn.cursor()
    
    if session_id:
        cursor.execute("""
            SELECT COUNT(*) as calls, SUM(memory_count) as total_memories,
                   SUM(token_count) as total_tokens, SUM(estimated_cost_usd) as total_cost
            FROM cost_entries WHERE session_id = ?
        """, (session_id,))
    else:
        cursor.execute("""
            SELECT COUNT(*) as calls, SUM(memory_count) as total_memories,
                   SUM(token_count) as total_tokens, SUM(estimated_cost_usd) as total_cost
            FROM cost_entries
        """)
    
    row = cursor.fetchone()
    return {
        "total_calls": row["calls"] or 0,
        "total_memories_injected": row["total_memories"] or 0,
        "total_tokens": row["total_tokens"] or 0,
        "total_estimated_cost_usd": round(row["total_cost"] or 0, 6),
    }
