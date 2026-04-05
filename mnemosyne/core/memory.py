"""
Mnemosyne Core - Direct SQLite Integration
No HTTP, no server, just pure Python + SQLite

This is the heart of Mnemosyne — a zero-dependency memory system
that delivers sub-millisecond performance through direct SQLite access.
"""

import sqlite3
import json
import hashlib
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

# Single shared connection per thread
_thread_local = threading.local()

# Default data directory
DEFAULT_DATA_DIR = Path.home() / ".mnemosyne" / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"

# Allow override via environment
import os
if os.environ.get("MNEMOSYNE_DATA_DIR"):
    DEFAULT_DATA_DIR = Path(os.environ.get("MNEMOSYNE_DATA_DIR"))
    DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"


def _get_connection(db_path: Path = None) -> sqlite3.Connection:
    """Get thread-local database connection"""
    if not hasattr(_thread_local, 'conn') or _thread_local.conn is None:
        path = db_path or DEFAULT_DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        _thread_local.conn = sqlite3.connect(str(path), check_same_thread=False)
        _thread_local.conn.row_factory = sqlite3.Row
    return _thread_local.conn


def init_db(db_path: Path = None):
    """Initialize database schema"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source TEXT,
            timestamp TEXT,
            session_id TEXT DEFAULT 'default',
            importance REAL DEFAULT 0.5,
            metadata_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON memories(source)")
    
    conn.commit()


# Initialize on module load
init_db()


def generate_id(content: str) -> str:
    """Generate unique ID for memory"""
    return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]


def calculate_relevance(query_words: List[str], content: str) -> float:
    """
    Calculate relevance score between query and content.
    
    Uses a combination of:
    - Exact word matches (higher weight)
    - Partial word matches
    - Word frequency
    """
    content_lower = content.lower()
    content_words = set(content_lower.split())
    
    exact_matches = sum(1 for word in query_words if word.lower() in content_words)
    partial_matches = sum(
        1 for word in query_words 
        for content_word in content_words
        if word.lower() in content_word or content_word in word.lower()
    )
    
    # Exact matches count more
    score = (exact_matches * 1.0 + partial_matches * 0.3) / max(len(query_words), 1)
    return min(score, 1.0)  # Cap at 1.0


class Mnemosyne:
    """
    Native memory interface - no HTTP, direct SQLite.
    
    This class provides the main interface to the Mnemosyne memory system.
    Each instance can have its own session_id for multi-tenant scenarios.
    
    Args:
        session_id: Unique identifier for this memory session
        db_path: Optional custom database path
    
    Example:
        >>> mem = Mnemosyne(session_id="user_123")
        >>> mem.remember("Likes Python", importance=0.8)
        >>> results = mem.recall("programming preferences")
    """
    
    def __init__(self, session_id: str = "default", db_path: Path = None):
        self.session_id = session_id
        self.db_path = db_path or DEFAULT_DB_PATH
        self.conn = _get_connection(self.db_path)
        init_db(self.db_path)
    
    def remember(self, content: str, source: str = "conversation",
                 importance: float = 0.5, metadata: Dict = None) -> str:
        """
        Store a memory directly to SQLite.
        
        Args:
            content: The information to remember
            source: Origin of the memory ('conversation', 'preference', 'fact')
            importance: Priority from 0.0 to 1.0 (0.9+ for critical facts)
            metadata: Optional structured data as a dictionary
            
        Returns:
            memory_id: Unique identifier for the stored memory
            
        Example:
            >>> mem.remember("User prefers dark mode", importance=0.9)
            'a1b2c3d4e5f67890'
        """
        memory_id = generate_id(content)
        timestamp = datetime.now().isoformat()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO memories (id, content, source, timestamp, session_id, importance, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, content, source, timestamp, self.session_id,
            importance, json.dumps(metadata or {})
        ))
        self.conn.commit()
        
        return memory_id
    
    def recall(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search memories with relevance scoring.
        
        Uses a custom scoring algorithm combining:
        - Keyword relevance (50%)
        - Importance boost (30%)
        - Recency boost (20%)
        
        Args:
            query: Search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of memory dictionaries with relevance scores
            
        Example:
            >>> results = mem.recall("dark mode")
            >>> results[0]['content']
            'User prefers dark mode'
            >>> results[0]['score']
            0.95
        """
        query_words = query.split()
        cursor = self.conn.cursor()
        
        # Get recent memories (limit to prevent memory bloat)
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id, importance
            FROM memories
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT 1000
        """, (self.session_id,))
        
        rows = cursor.fetchall()
        results = []
        
        for row in rows:
            relevance = calculate_relevance(query_words, row['content'])
            
            if relevance > 0:
                # Boost by importance
                importance_boost = row['importance'] * 0.3
                
                # Time decay boost
                try:
                    age_hours = (datetime.now() - datetime.fromisoformat(row['timestamp'])).total_seconds() / 3600
                    if age_hours < 24:
                        recency_boost = 0.2
                    elif age_hours < 168:  # 1 week
                        recency_boost = 0.1
                    else:
                        recency_boost = 0
                except:
                    recency_boost = 0
                
                final_score = (relevance * 0.5) + importance_boost + recency_boost
                
                results.append({
                    "id": row['id'],
                    "content": row['content'][:500],  # Limit content length
                    "source": row['source'],
                    "timestamp": row['timestamp'],
                    "session_id": row['session_id'],
                    "score": round(final_score, 3),
                    "importance": row['importance']
                })
        
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]
    
    def get_context(self, limit: int = 10) -> List[Dict]:
        """
        Get recent memories from current session for context injection.
        
        This is used by the Hermes plugin to auto-inject memories
        before LLM calls.
        
        Args:
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of recent memory dictionaries
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, source, timestamp, importance
            FROM memories
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.session_id, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict:
        """
        Get memory system statistics.
        
        Returns:
            Dictionary containing:
            - total_memories: Total count of stored memories
            - total_sessions: Number of unique sessions
            - sources: Breakdown by source type
            - last_memory: Timestamp of most recent memory
            - database: Path to database file
            - mode: Storage mode (always 'native_sqlite')
        """
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM memories")
        sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT source, COUNT(*) FROM memories GROUP BY source")
        sources = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute("SELECT timestamp FROM memories ORDER BY timestamp DESC LIMIT 1")
        last = cursor.fetchone()
        
        return {
            "total_memories": total,
            "total_sessions": sessions,
            "sources": sources,
            "last_memory": last[0] if last else None,
            "database": str(self.db_path),
            "mode": "native_sqlite"
        }
    
    def forget(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.
        
        Args:
            memory_id: The unique identifier of the memory to delete
            
        Returns:
            True if memory was found and deleted, False otherwise
        """
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ? AND session_id = ?", 
                      (memory_id, self.session_id))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def update(self, memory_id: str, content: str = None, 
               importance: float = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: The unique identifier of the memory to update
            content: New content (optional)
            importance: New importance score (optional)
            
        Returns:
            True if memory was found and updated, False otherwise
        """
        cursor = self.conn.cursor()
        
        updates = []
        params = []
        
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)
        
        if not updates:
            return False
        
        params.extend([memory_id, self.session_id])
        cursor.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ? AND session_id = ?",
            params
        )
        self.conn.commit()
        return cursor.rowcount > 0


# Global instance for module-level convenience functions
_default_instance = None

def _get_default():
    """Get or create the default Mnemosyne instance"""
    global _default_instance
    if _default_instance is None:
        _default_instance = Mnemosyne()
    return _default_instance


# Module-level convenience functions
def remember(content: str, source: str = "conversation", 
             importance: float = 0.5, metadata: Dict = None) -> str:
    """Store a memory using the global instance"""
    return _get_default().remember(content, source, importance, metadata)


def recall(query: str, top_k: int = 5) -> List[Dict]:
    """Search memories using the global instance"""
    return _get_default().recall(query, top_k)


def get_context(limit: int = 10) -> List[Dict]:
    """Get session context using the global instance"""
    return _get_default().get_context(limit)


def get_stats() -> Dict:
    """Get stats using the global instance"""
    return _get_default().get_stats()


def forget(memory_id: str) -> bool:
    """Delete memory using the global instance"""
    return _get_default().forget(memory_id)


def update(memory_id: str, content: str = None, importance: float = None) -> bool:
    """Update memory using the global instance"""
    return _get_default().update(memory_id, content, importance)
