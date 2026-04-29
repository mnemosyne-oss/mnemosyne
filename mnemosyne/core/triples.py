"""
Mnemosyne Temporal Triples
Time-aware knowledge graph on top of SQLite.
Tracks when facts were true, enabling contradiction detection and historical queries.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

DEFAULT_DB = Path.home() / ".hermes" / "mnemosyne" / "data" / "triples.db"


def _get_conn(db_path = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else DEFAULT_DB
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_triples(db_path: Path = None):
    conn = _get_conn(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS triples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            valid_from TEXT NOT NULL,
            valid_until TEXT,
            source TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_triples_valid_from ON triples(valid_from)")
    
    conn.commit()


class TripleStore:
    """
    Temporal knowledge graph for Mnemosyne.
    
    Example:
        >>> kg = TripleStore()
        >>> kg.add("Maya", "assigned_to", "auth-migration", valid_from="2026-01-15")
        >>> kg.query("Maya", as_of="2026-01-20")
    """
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or DEFAULT_DB
        init_triples(self.db_path)
        self.conn = _get_conn(self.db_path)
    
    def add(self, subject: str, predicate: str, object: str,
            valid_from: str = None, source: str = "inferred",
            confidence: float = 1.0) -> int:
        """
        Add a temporal triple. Automatically closes previous matching triples.
        """
        valid_from = valid_from or datetime.now().isoformat()[:10]
        
        # Invalidate previous triples for same (subject, predicate)
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE triples
            SET valid_until = ?
            WHERE subject = ? AND predicate = ? AND valid_until IS NULL
        """, (valid_from, subject, predicate))
        
        # Insert new triple
        cursor.execute("""
            INSERT INTO triples (subject, predicate, object, valid_from, source, confidence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (subject, predicate, object, valid_from, source, confidence))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def query(self, subject: str = None, predicate: str = None,
              object: str = None, as_of: str = None) -> List[Dict]:
        """
        Query triples, optionally as of a specific date.
        """
        cursor = self.conn.cursor()
        as_of = as_of or datetime.now().isoformat()[:10]
        
        conditions = []
        params = []
        
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if object:
            conditions.append("object = ?")
            params.append(object)
        
        # Temporal filter: valid at as_of date
        conditions.append("valid_from <= ?")
        params.append(as_of)
        conditions.append("(valid_until IS NULL OR valid_until > ?)")
        params.append(as_of)
        
        where_clause = " AND ".join(conditions)
        cursor.execute(f"SELECT * FROM triples WHERE {where_clause} ORDER BY valid_from DESC", params)
        
        return [dict(row) for row in cursor.fetchall()]

    def query_by_predicate(self, predicate: str, object: str = None, subject: str = None) -> List[Dict]:
        """
        Query triples by predicate, optionally filtering by object or subject.
        
        Useful for entity queries: find all memories that mention a specific entity.
        
        Examples:
            >>> kg.query_by_predicate("mentions", "Abdias")
            # Returns all triples where someone/something mentions Abdias
            
            >>> kg.query_by_predicate("mentions", subject="memory_123")
            # Returns entities mentioned by memory_123
        """
        cursor = self.conn.cursor()
        
        conditions = ["predicate = ?"]
        params = [predicate]
        
        if object:
            conditions.append("object = ?")
            params.append(object)
        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        
        where_clause = " AND ".join(conditions)
        cursor.execute(f"SELECT * FROM triples WHERE {where_clause} ORDER BY created_at DESC", params)
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_distinct_objects(self, predicate: str) -> List[str]:
        """
        Get all distinct object values for a given predicate.
        
        Useful for building entity lists: get all known entities that have been mentioned.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT DISTINCT object FROM triples WHERE predicate = ? ORDER BY object",
            (predicate,)
        )
        return [row["object"] for row in cursor.fetchall()]

    def add_facts(self, memory_id: str, facts: List[str], source: str = "", confidence: float = 0.7) -> int:
        """
        Batch-store extracted facts as triples.

        Args:
            memory_id: The subject memory ID
            facts: List of fact strings to store
            source: Source identifier
            confidence: Confidence score for extracted facts (default 0.7)

        Returns:
            Number of facts stored
        """
        if not facts:
            return 0

        stored = 0
        for fact in facts:
            if fact and len(fact) > 10:
                self.add(
                    subject=memory_id,
                    predicate="fact",
                    object=fact,
                    source=source,
                    confidence=confidence
                )
                stored += 1

        return stored

    def export_all(self) -> List[Dict]:
        """Export all triples to a list of dictionaries."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, subject, predicate, object, valid_from, valid_until,
                   source, confidence, created_at
            FROM triples
            ORDER BY id
        """)
        return [dict(row) for row in cursor.fetchall()]

    def import_all(self, triples: List[Dict], force: bool = False) -> Dict:
        """
        Import triples from a list of dictionaries.
        Idempotent by default: skips records whose id already exists.
        Set force=True to overwrite.
        Returns import statistics.
        """
        stats = {"inserted": 0, "skipped": 0, "overwritten": 0}
        cursor = self.conn.cursor()
        for item in triples:
            tid = item.get("id")
            cursor.execute("SELECT 1 FROM triples WHERE id = ?", (tid,))
            exists = cursor.fetchone() is not None
            if exists and not force:
                stats["skipped"] += 1
                continue
            if exists and force:
                cursor.execute("DELETE FROM triples WHERE id = ?", (tid,))
                stats["overwritten"] += 1
            else:
                stats["inserted"] += 1
            cursor.execute("""
                INSERT INTO triples (id, subject, predicate, object, valid_from,
                                     valid_until, source, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tid, item.get("subject"), item.get("predicate"), item.get("object"),
                item.get("valid_from"), item.get("valid_until"),
                item.get("source", "imported"), item.get("confidence", 1.0),
                item.get("created_at")
            ))
        self.conn.commit()
        return stats


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def add_triple(subject: str, predicate: str, object: str,
               valid_from: str = None, source: str = "inferred",
               confidence: float = 1.0, db_path: Path = None) -> int:
    """
    Add a temporal triple without instantiating TripleStore manually.
    Optional db_path aligns with BEAM memory database when used from Hermes.
    """
    store = TripleStore(db_path=db_path)
    return store.add(subject, predicate, object,
                     valid_from=valid_from, source=source, confidence=confidence)


def query_triples(subject: str = None, predicate: str = None,
                  object: str = None, as_of: str = None,
                  db_path: Path = None) -> List[Dict]:
    """
    Query temporal triples without instantiating TripleStore manually.
    Optional db_path aligns with BEAM memory database when used from Hermes.
    """
    store = TripleStore(db_path=db_path)
    return store.query(subject=subject, predicate=predicate,
                       object=object, as_of=as_of)
