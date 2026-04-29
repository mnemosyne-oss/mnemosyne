"""
Mnemosyne BEAM Architecture
============================
Bilevel Episodic-Associative Memory

Three SQLite tables:
- working_memory: hot, recent context (auto-injected into prompts)
- episodic_memory: long-term storage with native vector + FTS5 search
- scratchpad: temporary agent reasoning workspace

Native sqlite-vec for vector search.
FTS5 for full-text retrieval.
Hybrid ranking: 50% vector + 30% FTS rank + 20% importance.
"""

import sqlite3
import json
import hashlib
import threading
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Set
from pathlib import Path

from mnemosyne.core import embeddings as _embeddings

# sqlite-vec optional dependency
try:
    import sqlite_vec
    _SQLITE_VEC_AVAILABLE = True
except Exception:
    _SQLITE_VEC_AVAILABLE = False
    sqlite_vec = None

_thread_local = threading.local()

# On Fly.io and other ephemeral VMs, only ~/.hermes is persisted.
# Default to the legacy Hermes path so memories survive restarts.
DEFAULT_DATA_DIR = Path.home() / ".hermes" / "mnemosyne" / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"

import os
if os.environ.get("MNEMOSYNE_DATA_DIR"):
    DEFAULT_DATA_DIR = Path(os.environ.get("MNEMOSYNE_DATA_DIR"))
    DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"

# Config
EMBEDDING_DIM = 384  # bge-small-en-v1.5
WORKING_MEMORY_MAX_ITEMS = int(os.environ.get("MNEMOSYNE_WM_MAX_ITEMS", "10000"))
WORKING_MEMORY_TTL_HOURS = int(os.environ.get("MNEMOSYNE_WM_TTL_HOURS", "24"))
EPISODIC_RECALL_LIMIT = int(os.environ.get("MNEMOSYNE_EP_LIMIT", "50000"))
SLEEP_BATCH_SIZE = int(os.environ.get("MNEMOSYNE_SLEEP_BATCH", "5000"))
SCRATCHPAD_MAX_ITEMS = int(os.environ.get("MNEMOSYNE_SP_MAX", "1000"))
RECENCY_HALFLIFE_HOURS = float(os.environ.get("MNEMOSYNE_RECENCY_HALFLIFE", "168"))  # 1 week default

# Vector compression: float32 | int8 | bit
VEC_TYPE = os.environ.get("MNEMOSYNE_VEC_TYPE", "int8").lower()
if VEC_TYPE not in ("float32", "int8", "bit"):
    VEC_TYPE = "float32"


def _get_connection(db_path: Path = None) -> sqlite3.Connection:
    """Get thread-local database connection with extensions loaded."""
    path = db_path or DEFAULT_DB_PATH
    if not hasattr(_thread_local, 'conn') or _thread_local.conn is None or getattr(_thread_local, 'db_path', None) != str(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        if _SQLITE_VEC_AVAILABLE:
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
            except Exception:
                pass  # Some environments don't support load_extension
        _thread_local.conn = conn
        _thread_local.db_path = str(path)
    return _thread_local.conn


def _detect_vec_type(conn: sqlite3.Connection) -> str:
    """
    Detect whether sqlite-vec supports int8/bit.
    Falls back to float32 if the requested type is unavailable.
    """
    if not _SQLITE_VEC_AVAILABLE:
        return "float32"
    if VEC_TYPE == "float32":
        return "float32"
    cursor = conn.cursor()
    test_type = VEC_TYPE  # int8 or bit
    try:
        cursor.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS _vec_test USING vec0(embedding {test_type}[{EMBEDDING_DIM}])")
        cursor.execute("DROP TABLE IF EXISTS _vec_test")
        conn.commit()
        return test_type
    except Exception:
        conn.rollback()
        # Try int8 as fallback from bit
        if test_type == "bit":
            try:
                cursor.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS _vec_test USING vec0(embedding int8[{EMBEDDING_DIM}])")
                cursor.execute("DROP TABLE IF EXISTS _vec_test")
                conn.commit()
                return "int8"
            except Exception:
                conn.rollback()
        return "float32"


def init_beam(db_path: Path = None):
    """Initialize BEAM schema."""
    conn = _get_connection(db_path)
    cursor = conn.cursor()

    # --- WORKING MEMORY ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS working_memory (
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_session ON working_memory(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_timestamp ON working_memory(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_source ON working_memory(source)")

    # --- EPISODIC MEMORY ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS episodic_memory (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            id TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            timestamp TEXT,
            session_id TEXT DEFAULT 'default',
            importance REAL DEFAULT 0.5,
            metadata_json TEXT,
            summary_of TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_session ON episodic_memory(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_timestamp ON episodic_memory(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_source ON episodic_memory(source)")

    # --- SCRATCHPAD ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scratchpad (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            session_id TEXT DEFAULT 'default',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sp_session ON scratchpad(session_id)")

    # Detect supported vector type
    effective_vec_type = _detect_vec_type(conn)

    # --- sqlite-vec VIRTUAL TABLE ---
    if _SQLITE_VEC_AVAILABLE:
        try:
            cursor.execute(f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_episodes USING vec0(
                    embedding {effective_vec_type}[{EMBEDDING_DIM}]
                )
            """)
        except sqlite3.OperationalError:
            pass  # May already exist or extension not loadable

    # --- FTS5 VIRTUAL TABLE for episodic ---
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_episodes USING fts5(
            content,
            content='episodic_memory',
            content_rowid='rowid'
        )
    """)

    # --- FTS5 VIRTUAL TABLE for working memory (autonomous) ---
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_working USING fts5(
            id UNINDEXED,
            content
        )
    """)

    # --- FTS5 Triggers for episodic ---
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS em_ai AFTER INSERT ON episodic_memory BEGIN
            INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS em_ad AFTER DELETE ON episodic_memory BEGIN
            INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS em_au AFTER UPDATE ON episodic_memory BEGIN
            INSERT INTO fts_episodes(fts_episodes, rowid, content) VALUES ('delete', old.rowid, old.content);
            INSERT INTO fts_episodes(rowid, content) VALUES (new.rowid, new.content);
        END
    """)

    # --- FTS5 Triggers for working memory ---
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS wm_ai AFTER INSERT ON working_memory BEGIN
            INSERT INTO fts_working(id, content) VALUES (new.id, new.content);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS wm_ad AFTER DELETE ON working_memory BEGIN
            DELETE FROM fts_working WHERE id = old.id;
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS wm_au AFTER UPDATE ON working_memory BEGIN
            DELETE FROM fts_working WHERE id = old.id;
            INSERT INTO fts_working(id, content) VALUES (new.id, new.content);
        END
    """)

    # --- Consolidation Log ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS consolidation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            items_consolidated INTEGER,
            summary_preview TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --- memory_embeddings: fallback for environments without sqlite-vec ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()

    # --- Migration: recall tracking columns (v2.1) ---
    _add_column_if_missing(conn, "working_memory", "recall_count", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "working_memory", "last_recalled", "TIMESTAMP DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "recall_count", "INTEGER DEFAULT 0")
    _add_column_if_missing(conn, "episodic_memory", "last_recalled", "TIMESTAMP DEFAULT NULL")

    # --- Migration: temporal validity + scope (v2.2) ---
    _add_column_if_missing(conn, "working_memory", "valid_until", "TIMESTAMP DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "superseded_by", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "scope", "TEXT DEFAULT 'global'")
    _add_column_if_missing(conn, "episodic_memory", "valid_until", "TIMESTAMP DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "superseded_by", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "scope", "TEXT DEFAULT 'global'")


def _generate_id(content: str) -> str:
    return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str):
    """Safely add a column if it doesn't already exist (SQLite migration helper)."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in cursor.fetchall()}
    if column not in existing:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()


def _recency_decay(timestamp_str: str, halflife_hours: float = RECENCY_HALFLIFE_HOURS) -> float:
    """Calculate recency decay factor. 1.0 = brand new, ~0.5 = one halflife old."""
    if not timestamp_str:
        return 0.5  # Unknown age = neutral
    try:
        ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=None)
        hours_old = max(0.0, (datetime.now() - ts).total_seconds() / 3600.0)
        return math.exp(-hours_old / halflife_hours)
    except Exception:
        return 0.5


def _vec_available(conn: sqlite3.Connection) -> bool:
    if not _SQLITE_VEC_AVAILABLE:
        return False
    try:
        conn.execute("SELECT 1 FROM vec_episodes LIMIT 0")
        return True
    except Exception:
        return False


def _extract_and_store_entities(beam: "BeamMemory", memory_id: str, content: str):
    """
    Extract entities from content and store as triples.
    Called internally by remember() when extract_entities=True.
    """
    try:
        from mnemosyne.core.entities import extract_entities_regex
        from mnemosyne.core.triples import TripleStore
        
        entities = extract_entities_regex(content)
        if not entities:
            return
        
        triples = TripleStore(db_path=beam.db_path)
        for entity in entities:
            triples.add(
                subject=memory_id,
                predicate="mentions",
                object=entity,
                source="regex",
                confidence=0.8
            )
    except Exception:
        # Entity extraction is best-effort; never fail remember() because of it
        pass


def _extract_and_store_facts(beam: "BeamMemory", memory_id: str, content: str, source: str = ""):
    """
    Extract structured facts from content using LLM and store as triples.
    Called internally by remember() when extract=True.
    """
    try:
        from mnemosyne.core.extraction import extract_facts_safe
        from mnemosyne.core.triples import TripleStore
        
        facts = extract_facts_safe(content)
        if not facts:
            return
        
        triples = TripleStore(db_path=beam.db_path)
        triples.add_facts(memory_id, facts, source=source, confidence=0.7)
    except Exception:
        # Fact extraction is best-effort; never fail remember() because of it
        pass


def _find_memories_by_entity(beam: "BeamMemory", entity_name: str, threshold: float = 0.8) -> List[str]:
    """
    Find memory IDs that mention an entity (or similar entity via fuzzy match).
    Returns list of memory_id strings.
    """
    try:
        from mnemosyne.core.entities import find_similar_entities
        from mnemosyne.core.triples import TripleStore
        
        triples = TripleStore(db_path=beam.db_path)
        
        # Get all known entities
        known_entities = triples.get_distinct_objects("mentions")
        if not known_entities:
            return []
        
        # Find similar entities
        matches = find_similar_entities(entity_name, known_entities, threshold=threshold)
        
        # Collect memory IDs for all matched entities
        memory_ids: Set[str] = set()
        for matched_entity, _ in matches:
            results = triples.query_by_predicate("mentions", object=matched_entity)
            for row in results:
                memory_ids.add(row["subject"])
        
        return list(memory_ids)
    except Exception:
        return []


def _find_memories_by_fact(beam: "BeamMemory", query: str) -> List[str]:
    """
    Find memory IDs that have extracted facts matching the query.
    Does simple keyword matching against stored fact triples.
    Returns list of memory_id strings.
    """
    try:
        from mnemosyne.core.triples import TripleStore
        
        triples = TripleStore(db_path=beam.db_path)
        
        # Get all fact triples
        all_facts = triples.query_by_predicate("fact")
        if not all_facts:
            return []
        
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Simple keyword matching against fact text
        memory_ids: Set[str] = set()
        for fact_row in all_facts:
            fact_text = fact_row.get("object", "").lower()
            # Check if any query word appears in the fact
            if any(word in fact_text for word in query_words):
                memory_ids.add(fact_row["subject"])
            # Also check if the full query is a substring of the fact
            elif query_lower in fact_text:
                memory_ids.add(fact_row["subject"])
        
        return list(memory_ids)
    except Exception:
        return []


def _in_memory_vec_search(conn: sqlite3.Connection, query_embedding: np.ndarray, k: int = 20) -> List[Dict]:
    """Fallback vector search using memory_embeddings table + numpy cosine similarity."""
    cursor = conn.cursor()
    # Join with episodic_memory (not memories) since that's where BEAM stores consolidated data
    cursor.execute("""
        SELECT em.rowid, me.memory_id, me.embedding_json
        FROM memory_embeddings me
        JOIN episodic_memory em ON me.memory_id = em.id
        LIMIT 10000
    """)
    rows = cursor.fetchall()
    if not rows:
        return []

    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []
    query_unit = query_embedding / query_norm

    results = []
    for row in rows:
        try:
            vec = np.array(json.loads(row["embedding_json"]), dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = float(np.dot(query_unit, vec / vec_norm))
            # Convert similarity to distance-like metric (1 - sim) for consistent ranking
            results.append({"rowid": row["rowid"], "distance": 1.0 - sim})
        except Exception:
            continue

    results.sort(key=lambda x: x["distance"])
    return results[:k]


def _effective_vec_type(conn: sqlite3.Connection) -> str:
    """Re-detect the actual vector type used by vec_episodes."""
    if not _vec_available(conn):
        return "float32"
    try:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='vec_episodes'"
        ).fetchone()
        if row and "int8" in row[0]:
            return "int8"
        if row and "bit" in row[0]:
            return "bit"
    except Exception:
        pass
    return "float32"


def _vec_insert(conn: sqlite3.Connection, rowid: int, embedding: List[float]):
    """Insert embedding into sqlite-vec table with quantization via SQL functions."""
    vec_type = _effective_vec_type(conn)
    emb_json = json.dumps(embedding)
    if vec_type == "bit":
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_binary(?))",
            (rowid, emb_json)
        )
    elif vec_type == "int8":
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, vec_quantize_int8(?, 'unit'))",
            (rowid, emb_json)
        )
    else:
        conn.execute(
            "INSERT INTO vec_episodes(rowid, embedding) VALUES (?, ?)",
            (rowid, emb_json)
        )


def _vec_search(conn: sqlite3.Connection, embedding: List[float], k: int = 20) -> List[Dict]:
    """Search sqlite-vec and return rowids with distances."""
    vec_type = _effective_vec_type(conn)
    emb_json = json.dumps(embedding)
    # NOTE: sqlite-vec requires the KNN limit to be known at query planning time.
    # Parameter binding (LIMIT ?) fails on some versions because xBestIndex
    # can't resolve the parameter value. We inline k safely since it's
    # always an integer computed internally.
    k = int(k)
    if vec_type == "bit":
        rows = conn.execute(
            f"SELECT rowid, distance FROM vec_episodes WHERE embedding MATCH vec_quantize_binary(?) ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    elif vec_type == "int8":
        rows = conn.execute(
            f"SELECT rowid, distance FROM vec_episodes WHERE embedding MATCH vec_quantize_int8(?, 'unit') ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT rowid, distance FROM vec_episodes WHERE embedding MATCH ? ORDER BY distance LIMIT {k}",
            (emb_json,)
        ).fetchall()
    return [{"rowid": r["rowid"], "distance": r["distance"]} for r in rows]


def _fts_search(conn: sqlite3.Connection, query: str, k: int = 20) -> List[Dict]:
    """Search FTS5 episodes and return rowids with ranks."""
    safe_query = " ".join(f'"{w}"' for w in query.split() if w)
    if not safe_query:
        return []
    rows = conn.execute(
        "SELECT rowid, rank FROM fts_episodes WHERE fts_episodes MATCH ? ORDER BY rank LIMIT ?",
        (safe_query, k)
    ).fetchall()
    return [{"rowid": r["rowid"], "rank": r["rank"]} for r in rows]


def _fts_search_working(conn: sqlite3.Connection, query: str, k: int = 20) -> List[Dict]:
    """Search FTS5 working memory and return ids with ranks."""
    safe_query = " ".join(f'"{w}"' for w in query.split() if w)
    if not safe_query:
        return []
    rows = conn.execute(
        "SELECT id, rank FROM fts_working WHERE fts_working MATCH ? ORDER BY rank LIMIT ?",
        (safe_query, k)
    ).fetchall()
    return [{"id": r["id"], "rank": r["rank"]} for r in rows]


class BeamMemory:
    """
    BEAM memory interface.
    """

    def __init__(self, session_id: str = "default", db_path: Path = None):
        self.session_id = session_id
        self.db_path = db_path or DEFAULT_DB_PATH
        self.conn = _get_connection(self.db_path)
        init_beam(self.db_path)

    # ------------------------------------------------------------------
    # Working Memory
    # ------------------------------------------------------------------
    def _find_duplicate(self, content: str) -> Optional[str]:
        """Check if exact same content already exists in working_memory for this session.
        Returns the existing memory_id if found, else None."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id FROM working_memory
            WHERE session_id = ? AND content = ?
            LIMIT 1
        """, (self.session_id, content))
        row = cursor.fetchone()
        return row["id"] if row else None

    def remember(self, content: str, source: str = "conversation",
                 importance: float = 0.5, metadata: Dict = None,
                 valid_until: str = None, scope: str = "session",
                 extract_entities: bool = False,
                 extract: bool = False) -> str:
        """Store into working_memory. Deduplicates exact content matches.

        Args:
            content: The text to remember
            source: Origin of the memory (e.g., "conversation", "document")
            importance: 0.0-1.0 relevance score
            metadata: Optional dict of additional fields
            valid_until: ISO timestamp when this memory expires
            scope: "session" or "global"
            extract_entities: If True, extract and store entity mentions as triples
            extract: If True, extract structured facts from content using LLM
                and store as triples. Default False.
        """
        # --- Deduplication: exact match ---
        existing_id = self._find_duplicate(content)
        if existing_id:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE working_memory
                SET importance = MAX(importance, ?), timestamp = ?, source = ?,
                    valid_until = COALESCE(?, valid_until),
                    scope = COALESCE(?, scope)
                WHERE id = ? AND session_id = ?
            """, (importance, datetime.now().isoformat(), source,
                  valid_until, scope, existing_id, self.session_id))
            self.conn.commit()
            return existing_id

        memory_id = _generate_id(content)
        timestamp = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO working_memory
            (id, content, source, timestamp, session_id, importance, metadata_json, valid_until, scope)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, content, source, timestamp, self.session_id, importance,
              json.dumps(metadata or {}), valid_until, scope))
        self.conn.commit()
        self._trim_working_memory()

        # Auto-generate temporal triple
        self._add_temporal_triple(memory_id, timestamp, source, content)

        # --- Entity extraction ---
        if extract_entities:
            _extract_and_store_entities(self, memory_id, content)

        # --- Structured fact extraction ---
        if extract:
            _extract_and_store_facts(self, memory_id, content, source)

        return memory_id

    def remember_batch(self, items: List[Dict]) -> List[str]:
        """
        Batch insert into working_memory for high-throughput ingestion.
        Each item dict should have keys: content, source, importance, metadata (optional).
        """
        cursor = self.conn.cursor()
        ids = []
        timestamp = datetime.now().isoformat()
        for item in items:
            memory_id = _generate_id(item["content"])
            ids.append(memory_id)
            cursor.execute("""
                INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                item["content"],
                item.get("source", "conversation"),
                timestamp,
                self.session_id,
                item.get("importance", 0.5),
                json.dumps(item.get("metadata") or {})
            ))
        self.conn.commit()
        self._trim_working_memory()
        return ids

    def _add_temporal_triple(self, memory_id: str, timestamp: str, source: str, content: str):
        """Auto-generate temporal triple for a memory. Bridges BEAM and TripleStore."""
        try:
            # Import triples module lazily to avoid circular dependency
            from mnemosyne.core.triples import TripleStore, init_triples
            date_str = timestamp[:10]  # YYYY-MM-DD
            # Ensure triples table exists
            init_triples(db_path=self.db_path)
            triple_store = TripleStore(db_path=self.db_path)
            triple_store.add(
                subject=memory_id,
                predicate="occurred_on",
                object=date_str,
                valid_from=date_str
            )
            # Also tag source type
            if source and source not in ("conversation", "user", "assistant"):
                triple_store.add(
                    subject=memory_id,
                    predicate="has_source",
                    object=source,
                    valid_from=date_str
                )
        except Exception:
            # TripleStore is optional; don't fail memory write if triples fail
            pass

    def _trim_working_memory(self):
        """Keep working_memory within size/time limits."""
        cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS)).isoformat()
        self.conn.execute("""
            DELETE FROM working_memory
            WHERE session_id = ? AND (
                timestamp < ? OR
                id NOT IN (
                    SELECT id FROM working_memory
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                )
            )
        """, (self.session_id, cutoff, self.session_id, WORKING_MEMORY_MAX_ITEMS))
        self.conn.commit()

    def get_context(self, limit: int = 10) -> List[Dict]:
        """Get recent working_memory for prompt injection.
        Global memories are returned first, then session memories."""
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        cursor.execute("""
            SELECT id, content, source, timestamp, importance, scope
            FROM working_memory
            WHERE (session_id = ? OR scope = 'global')
              AND (valid_until IS NULL OR valid_until > ?)
              AND superseded_by IS NULL
            ORDER BY
                CASE WHEN scope = 'global' THEN 0 ELSE 1 END,
                timestamp DESC
            LIMIT ?
        """, (self.session_id, now, limit))
        return [dict(row) for row in cursor.fetchall()]

    def invalidate(self, memory_id: str, replacement_id: str = None) -> bool:
        """
        Mark a memory as invalid/superseded.
        If replacement_id is provided, sets superseded_by.
        Otherwise sets valid_until to now (immediate expiry).
        """
        cursor = self.conn.cursor()
        now = datetime.now().isoformat()
        # Try working_memory first
        cursor.execute("""
            UPDATE working_memory
            SET valid_until = ?, superseded_by = ?
            WHERE id = ? AND (session_id = ? OR scope = 'global')
        """, (now, replacement_id, memory_id, self.session_id))
        if cursor.rowcount > 0:
            self.conn.commit()
            return True
        # Try episodic_memory
        cursor.execute("""
            UPDATE episodic_memory
            SET valid_until = ?, superseded_by = ?
            WHERE id = ? AND (session_id = ? OR scope = 'global')
        """, (now, replacement_id, memory_id, self.session_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_working_stats(self) -> Dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM working_memory")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT timestamp FROM working_memory ORDER BY timestamp DESC LIMIT 1")
        last = cursor.fetchone()
        return {"total": total, "last": last[0] if last else None}

    # DEPRECATED — kept for backward compatibility with hermes_memory_provider/cli.py
    def get_global_working_stats(self) -> Dict:
        """DEPRECATED: Use get_working_stats() instead. Kept for backward compatibility."""
        return self.get_working_stats()

    def update_working(self, memory_id: str, content: str = None,
                       importance: float = None) -> bool:
        """Update a working_memory entry."""
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
            f"UPDATE working_memory SET {', '.join(updates)} WHERE id = ? AND session_id = ?",
            params
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def forget_working(self, memory_id: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM working_memory WHERE id = ? AND session_id = ?", (memory_id, self.session_id))
        self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Episodic Memory
    # ------------------------------------------------------------------
    def consolidate_to_episodic(self, summary: str, source_wm_ids: List[str],
                                source: str = "consolidation", importance: float = 0.6,
                                metadata: Dict = None, valid_until: str = None,
                                scope: str = "session") -> str:
        """
        Store a consolidated summary into episodic_memory with optional embedding.
        """
        memory_id = _generate_id(summary)
        timestamp = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO episodic_memory
            (id, content, source, timestamp, session_id, importance, metadata_json, summary_of, valid_until, scope)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, summary, source, timestamp, self.session_id, importance,
              json.dumps(metadata or {}), ",".join(source_wm_ids), valid_until, scope))
        rowid = cursor.lastrowid

        if _embeddings.available():
            vec = _embeddings.embed([summary])
            if vec is not None:
                if _vec_available(self.conn):
                    _vec_insert(self.conn, rowid, vec[0].tolist())
                else:
                    # Fallback: store in memory_embeddings table for in-memory search
                    cursor.execute("""
                        INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model)
                        VALUES (?, ?, ?)
                    """, (memory_id, _embeddings.serialize(vec[0]), _embeddings._DEFAULT_MODEL))

        self.conn.commit()
        return memory_id

    def recall(self, query: str, top_k: int = 5, *, from_date: Optional[str] = None, to_date: Optional[str] = None, source: Optional[str] = None, topic: Optional[str] = None) -> List[Dict]:
        """
        Hybrid recall across working_memory + episodic_memory.
        Uses sqlite-vec + FTS5 for episodic, FTS5 for working.
        Falls back to recency-only for working memory if FTS5 unavailable.
        
        Temporal filtering:
            from_date/to_date: ISO date strings (YYYY-MM-DD) to filter by timestamp.
            source: Filter by memory source (e.g., 'cron', 'user', 'conversation').
            topic: Filter by topic tag (stored in source field for now, pending dedicated column).
        """
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        # ---- Working memory (FTS5 fast path) ----
        try:
            wm_fts = _fts_search_working(self.conn, query, k=max(top_k * 3, 50))
        except Exception:
            wm_fts = []

        wm_ids = {r["id"] for r in wm_fts}
        wm_ranks = {r["id"]: r["rank"] for r in wm_fts}

        # Build temporal filter clause for working memory
        wm_where_clauses = [
            "(session_id = ? OR scope = 'global')",
            "(valid_until IS NULL OR valid_until > ?)",
            "superseded_by IS NULL"
        ]
        wm_params = [self.session_id, datetime.now().isoformat()]
        
        if from_date:
            wm_where_clauses.append("timestamp >= ?")
            wm_params.append(f"{from_date}T00:00:00")
        if to_date:
            wm_where_clauses.append("timestamp <= ?")
            wm_params.append(f"{to_date}T23:59:59")
        if source:
            wm_where_clauses.append("source = ?")
            wm_params.append(source)
        if topic:
            # Topic stored in source field for now (pending dedicated topic column)
            wm_where_clauses.append("source = ?")
            wm_params.append(topic)
        
        wm_where = " AND ".join(wm_where_clauses)

        if wm_ids:
            placeholders = ",".join("?" * len(wm_ids))
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope
                FROM working_memory
                WHERE id IN ({placeholders})
                  AND {wm_where}
            """, (*tuple(wm_ids), *wm_params))
            rows = cursor.fetchall()
        else:
            # Fallback: fetch recent items and score in Python (old path)
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope
                FROM working_memory
                WHERE {wm_where}
                ORDER BY timestamp DESC
                LIMIT {min(EPISODIC_RECALL_LIMIT, 2000)}
            """, wm_params)
            rows = cursor.fetchall()

        # Precompute min_rank/rng for wm_ranks normalization
        if wm_ranks:
            min_rank = min(wm_ranks.values())
            max_rank = max(wm_ranks.values())
            rng = max_rank - min_rank if max_rank != min_rank else 1.0
        else:
            min_rank = 0.0
            rng = 1.0

        for row in rows:
            content_lower = row["content"].lower()
            if wm_ranks and row["id"] in wm_ranks:
                normalized = 1.0 - ((wm_ranks[row["id"]] - min_rank) / rng)
                relevance = normalized
            else:
                exact = sum(1 for w in query_words if w in content_lower)
                partial = sum(1 for w in query_words for cw in content_lower.split() if w in cw or cw in w)
                # Cross-substring match: if any query word is a substring of content, or vice versa
                cross = sum(1 for w in query_words if len(w) >= 2 for cw in content_lower.split() if len(cw) >= 2 and (w in cw or cw in w))
                # Also check if the full query is a substring of content (handles spaceless languages)
                full_match = 1.0 if query_lower in content_lower else 0.0
                if not full_match and content_lower in query_lower:
                    full_match = 0.5
                # Character-level overlap for spaceless languages (e.g. Chinese)
                query_chars = set(query_lower)
                content_chars = set(content_lower)
                char_overlap = len(query_chars & content_chars) / max(len(query_chars), 1) if query_chars else 0.0
                relevance = (exact * 1.0 + partial * 0.3 + cross * 0.5 + full_match + char_overlap * 0.8) / max(len(query_words), 1)
            if relevance > 0.02 or wm_ranks:
                decay = _recency_decay(row["timestamp"])
                base_score = relevance * 0.35 + row["importance"] * 0.2
                score = base_score * (0.7 + 0.3 * decay)
                results.append({
                    "id": row["id"],
                    "content": row["content"][:500],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                    "tier": "working",
                    "score": round(score, 4),
                    "keyword_score": round(relevance, 4),
                    "dense_score": 0.0,
                    "fts_score": round(relevance, 4) if wm_ranks else 0.0,
                    "importance": row["importance"],
                    "recall_count": row["recall_count"] or 0,
                    "last_recalled": row["last_recalled"],
                    "recency_decay": round(decay, 4),
                    "scope": row["scope"] if "scope" in row.keys() else "session",
                    "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                    "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None
                })

        # ---- Entity-aware recall ----
        entity_memory_ids = _find_memories_by_entity(self, query)
        if entity_memory_ids:
            # Fetch entity-matched memories and boost their scores
            placeholders = ",".join("?" * len(entity_memory_ids))
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope
                FROM working_memory
                WHERE id IN ({placeholders})
                  AND {wm_where}
            """, (*tuple(entity_memory_ids), *wm_params))
            entity_rows = cursor.fetchall()
            
            # Add entity-matched memories with boosted scores
            existing_ids = {r["id"] for r in results}
            for row in entity_rows:
                if row["id"] in existing_ids:
                    # Boost existing result
                    for r in results:
                        if r["id"] == row["id"]:
                            r["score"] = round(min(r["score"] * 1.3, 1.0), 4)
                            r["entity_match"] = True
                            break
                else:
                    decay = _recency_decay(row["timestamp"])
                    score = (0.6 + row["importance"] * 0.2) * (0.7 + 0.3 * decay)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "working",
                        "score": round(score, 4),
                        "keyword_score": 0.0,
                        "dense_score": 0.0,
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "entity_match": True
                    })
            
            # Also check episodic memory for entity matches
            em_placeholders = ",".join("?" * len(entity_memory_ids))
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope
                FROM episodic_memory
                WHERE id IN ({em_placeholders})
                  AND (session_id = ? OR scope = 'global')
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND superseded_by IS NULL
            """, (*tuple(entity_memory_ids), self.session_id, datetime.now().isoformat()))
            em_entity_rows = cursor.fetchall()
            
            em_existing_ids = {r["id"] for r in results}
            for row in em_entity_rows:
                if row["id"] in em_existing_ids:
                    for r in results:
                        if r["id"] == row["id"]:
                            r["score"] = round(min(r["score"] * 1.3, 1.0), 4)
                            r["entity_match"] = True
                            break
                else:
                    decay = _recency_decay(row["timestamp"])
                    score = (0.6 + row["importance"] * 0.2) * (0.7 + 0.3 * decay)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "episodic",
                        "score": round(score, 4),
                        "keyword_score": 0.0,
                        "dense_score": 0.0,
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "entity_match": True
                    })

        # ---- Fact-aware recall ----
        fact_memory_ids = _find_memories_by_fact(self, query)
        if fact_memory_ids:
            placeholders = ",".join("?" * len(fact_memory_ids))
            cursor = self.conn.cursor()
            # Check working_memory for fact matches
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope
                FROM working_memory
                WHERE id IN ({placeholders})
                  AND {wm_where}
            """, (*tuple(fact_memory_ids), *wm_params))
            fact_rows = cursor.fetchall()
            
            existing_ids = {r["id"] for r in results}
            for row in fact_rows:
                if row["id"] in existing_ids:
                    for r in results:
                        if r["id"] == row["id"]:
                            r["score"] = round(min(r["score"] * 1.2, 1.0), 4)
                            r["fact_match"] = True
                            break
                else:
                    decay = _recency_decay(row["timestamp"])
                    score = (0.5 + row["importance"] * 0.2) * (0.7 + 0.3 * decay)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "working",
                        "score": round(score, 4),
                        "keyword_score": 0.0,
                        "dense_score": 0.0,
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "fact_match": True
                    })
            
            # Also check episodic memory for fact matches
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope
                FROM episodic_memory
                WHERE id IN ({placeholders})
                  AND (session_id = ? OR scope = 'global')
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND superseded_by IS NULL
            """, (*tuple(fact_memory_ids), self.session_id, datetime.now().isoformat()))
            em_fact_rows = cursor.fetchall()
            
            em_existing_ids = {r["id"] for r in results}
            for row in em_fact_rows:
                if row["id"] in em_existing_ids:
                    for r in results:
                        if r["id"] == row["id"]:
                            r["score"] = round(min(r["score"] * 1.2, 1.0), 4)
                            r["fact_match"] = True
                            break
                else:
                    decay = _recency_decay(row["timestamp"])
                    score = (0.5 + row["importance"] * 0.2) * (0.7 + 0.3 * decay)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "episodic",
                        "score": round(score, 4),
                        "keyword_score": 0.0,
                        "dense_score": 0.0,
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "fact_match": True
                    })

        # ---- Episodic memory (vec + FTS5 hybrid) ----
        vec_results = {}
        max_distance = 0.0
        if _embeddings.available():
            emb_result = _embeddings.embed_query(query)
            if emb_result is not None:
                if _vec_available(self.conn):
                    vec_rows = _vec_search(self.conn, emb_result.tolist(), k=max(top_k * 3, 20))
                else:
                    # Fallback: in-memory cosine similarity search
                    vec_rows = _in_memory_vec_search(self.conn, emb_result, k=max(top_k * 3, 20))
                if vec_rows:
                    max_distance = max(vr["distance"] for vr in vec_rows)
                    for vr in vec_rows:
                        sim = max(0.0, 1.0 - (vr["distance"] / max_distance)) if max_distance > 0 else 1.0
                        vec_results[vr["rowid"]] = sim

        fts_results = {}
        fts_rows = _fts_search(self.conn, query, k=max(top_k * 3, 20))
        if fts_rows:
            min_rank = min(r["rank"] for r in fts_rows)
            max_rank = max(r["rank"] for r in fts_rows)
            rng = max_rank - min_rank if max_rank != min_rank else 1.0
            for fr in fts_rows:
                normalized = 1.0 - ((fr["rank"] - min_rank) / rng)
                fts_results[fr["rowid"]] = normalized

        episodic_rowids = set(vec_results.keys()) | set(fts_results.keys())
        
        # Build temporal filter for episodic memory
        em_where_clauses = [
            "(session_id = ? OR scope = 'global')",
            "(valid_until IS NULL OR valid_until > ?)",
            "superseded_by IS NULL"
        ]
        em_params = [self.session_id, datetime.now().isoformat()]
        
        if from_date:
            em_where_clauses.append("timestamp >= ?")
            em_params.append(f"{from_date}T00:00:00")
        if to_date:
            em_where_clauses.append("timestamp <= ?")
            em_params.append(f"{to_date}T23:59:59")
        if source:
            em_where_clauses.append("source = ?")
            em_params.append(source)
        if topic:
            em_where_clauses.append("source = ?")
            em_params.append(topic)
        
        em_where = " AND ".join(em_where_clauses)
        
        if episodic_rowids:
            placeholders = ",".join("?" * len(episodic_rowids))
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT rowid, id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope
                FROM episodic_memory
                WHERE rowid IN ({placeholders})
                  AND {em_where}
            """, (*tuple(episodic_rowids), *em_params))
            for row in cursor.fetchall():
                rid = row["rowid"]
                sim = vec_results.get(rid, 0.0)
                fts = fts_results.get(rid, 0.0)
                decay = _recency_decay(row["timestamp"])
                base_score = sim * 0.5 + fts * 0.3 + row["importance"] * 0.2
                score = base_score * (0.7 + 0.3 * decay)
                results.append({
                    "id": row["id"],
                    "content": row["content"][:500],
                    "source": row["source"],
                    "timestamp": row["timestamp"],
                    "tier": "episodic",
                    "score": round(score, 4),
                    "keyword_score": 0.0,
                    "dense_score": round(sim, 4),
                    "fts_score": round(fts, 4),
                    "importance": row["importance"],
                    "recall_count": row["recall_count"] or 0,
                    "last_recalled": row["last_recalled"],
                    "recency_decay": round(decay, 4),
                    "scope": row["scope"] if "scope" in row.keys() else "session",
                    "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                    "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None
                })

        # Fallback: if no episodic matches from vec/FTS, scan recent episodic entries
        if not episodic_rowids:
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT rowid, id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope
                FROM episodic_memory
                WHERE {em_where}
                ORDER BY timestamp DESC
                LIMIT {min(EPISODIC_RECALL_LIMIT, 500)}
            """, em_params)
            for row in cursor.fetchall():
                content_lower = row["content"].lower()
                exact = sum(1 for w in query_words if w in content_lower)
                partial = sum(1 for w in query_words for cw in content_lower.split() if w in cw or cw in w)
                cross = sum(1 for w in query_words if len(w) >= 2 for cw in content_lower.split() if len(cw) >= 2 and (w in cw or cw in w))
                full_match = 1.0 if query_lower in content_lower else 0.0
                if not full_match and content_lower in query_lower:
                    full_match = 0.5
                # Character-level overlap for spaceless languages (e.g. Chinese)
                query_chars = set(query_lower)
                content_chars = set(content_lower)
                char_overlap = len(query_chars & content_chars) / max(len(query_chars), 1) if query_chars else 0.0
                relevance = (exact * 1.0 + partial * 0.3 + cross * 0.5 + full_match + char_overlap * 0.8) / max(len(query_words), 1)
                if relevance > 0.02:
                    decay = _recency_decay(row["timestamp"])
                    base_score = relevance * 0.35 + row["importance"] * 0.2
                    score = base_score * (0.7 + 0.3 * decay)
                    results.append({
                        "id": row["id"],
                        "content": row["content"][:500],
                        "source": row["source"],
                        "timestamp": row["timestamp"],
                        "tier": "episodic",
                        "score": round(score, 4),
                        "keyword_score": round(relevance, 4),
                        "dense_score": 0.0,
                        "fts_score": 0.0,
                        "importance": row["importance"],
                        "recall_count": row["recall_count"] or 0,
                        "last_recalled": row["last_recalled"],
                        "recency_decay": round(decay, 4),
                        "scope": row["scope"] if "scope" in row.keys() else "session",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        final_results = results[:top_k]

        # --- Recall tracking: increment counts + set last_recalled ---
        now_iso = datetime.now().isoformat()
        wm_ids = [r["id"] for r in final_results if r.get("tier") == "working"]
        em_ids = [r["id"] for r in final_results if r.get("tier") == "episodic"]
        cursor = self.conn.cursor()
        if wm_ids:
            placeholders = ",".join("?" * len(wm_ids))
            cursor.execute(f"""
                UPDATE working_memory
                SET recall_count = recall_count + 1, last_recalled = ?
                WHERE id IN ({placeholders}) AND (session_id = ? OR scope = 'global')
            """, (now_iso, *tuple(wm_ids), self.session_id))
        if em_ids:
            placeholders = ",".join("?" * len(em_ids))
            cursor.execute(f"""
                UPDATE episodic_memory
                SET recall_count = recall_count + 1, last_recalled = ?
                WHERE id IN ({placeholders}) AND (session_id = ? OR scope = 'global')
            """, (now_iso, *tuple(em_ids), self.session_id))
        self.conn.commit()

        return final_results

    def get_episodic_stats(self) -> Dict:
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodic_memory")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT timestamp FROM episodic_memory ORDER BY timestamp DESC LIMIT 1")
        last = cursor.fetchone()
        vec_count = 0
        vec_type = "none"
        if _vec_available(self.conn):
            try:
                vec_count = cursor.execute("SELECT COUNT(*) FROM vec_episodes").fetchone()[0]
                vec_type = _effective_vec_type(self.conn)
            except Exception:
                pass
        return {"total": total, "last": last[0] if last else None, "vectors": vec_count, "vec_type": vec_type}

    # ------------------------------------------------------------------
    # Scratchpad
    # ------------------------------------------------------------------
    def scratchpad_write(self, content: str) -> str:
        pad_id = _generate_id(content)
        ts = datetime.now().isoformat()
        self.conn.execute("""
            INSERT INTO scratchpad (id, content, session_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET content=excluded.content, updated_at=excluded.updated_at
        """, (pad_id, content, self.session_id, ts, ts))
        self.conn.commit()
        return pad_id

    def scratchpad_read(self) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT id, content, created_at, updated_at
            FROM scratchpad
            WHERE session_id = ?
            ORDER BY updated_at DESC
            LIMIT {SCRATCHPAD_MAX_ITEMS}
        """, (self.session_id,))
        return [dict(row) for row in cursor.fetchall()]

    def scratchpad_clear(self):
        self.conn.execute("DELETE FROM scratchpad WHERE session_id = ?", (self.session_id,))
        self.conn.commit()

    # ------------------------------------------------------------------
    # Consolidation / Sleep
    # ------------------------------------------------------------------
    def sleep(self, dry_run: bool = False) -> Dict:
        """
        Consolidate old working_memory into episodic_memory summaries.
        Uses a local lightweight LLM when available; falls back to aaak
        compression if the model is missing or inference fails.
        Returns summary of what was done.
        """
        from mnemosyne.core.aaak import encode as aaak_encode
        from mnemosyne.core import local_llm

        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)).isoformat()
        cursor.execute(f"""
            SELECT id, content, source, timestamp, importance, metadata_json, scope, valid_until
            FROM working_memory
            WHERE session_id = ? AND timestamp < ?
            ORDER BY timestamp ASC
            LIMIT {SLEEP_BATCH_SIZE}
        """, (self.session_id, cutoff))
        rows = cursor.fetchall()
        if not rows:
            return {"status": "no_op", "message": "No old working memories to consolidate"}

        grouped: Dict[str, List[Dict]] = {}
        for row in rows:
            grouped.setdefault(row["source"], []).append(dict(row))

        consolidated_ids = []
        summaries_created = 0
        llm_used_count = 0
        for source, items in grouped.items():
            lines = [item["content"] for item in items]
            ids = [item["id"] for item in items]

            # Aggregate scope: if ANY item is global, the summary is global
            aggregated_scope = "session"
            aggregated_valid_until = None
            for item in items:
                if item.get("scope") == "global":
                    aggregated_scope = "global"
                if item.get("valid_until"):
                    if aggregated_valid_until is None or item["valid_until"] < aggregated_valid_until:
                        aggregated_valid_until = item["valid_until"]

            # --- Try LLM summarization (chunked to fit context) ---
            summary = None
            if local_llm.llm_available():
                chunks = local_llm.chunk_memories_by_budget(lines, source=source)
                if chunks:
                    if len(chunks) == 1:
                        # All memories fit in one prompt
                        summary = local_llm.summarize_memories(chunks[0], source=source)
                    else:
                        # Multi-chunk: summarize each chunk, then summarize the summaries
                        chunk_summaries = []
                        for chunk in chunks:
                            chunk_summary = local_llm.summarize_memories(chunk, source=source)
                            if chunk_summary:
                                chunk_summaries.append(chunk_summary)
                        if chunk_summaries:
                            # Second-pass: summarize the chunk summaries
                            if len(chunk_summaries) == 1:
                                summary = chunk_summaries[0]
                            else:
                                summary = local_llm.summarize_memories(
                                    chunk_summaries,
                                    source=f"{source} (consolidated)"
                                )
                                # If second-pass also overflows, concatenate
                                if not summary:
                                    summary = " | ".join(chunk_summaries)
                    if summary:
                        llm_used_count += 1

            # --- Fallback to aaak encoding ---
            if summary is None:
                combined = " | ".join(lines)
                compressed = aaak_encode(combined)
                summary = f"[{source}] {compressed}"

            if not dry_run:
                self.consolidate_to_episodic(
                    summary=summary,
                    source_wm_ids=ids,
                    source="sleep_consolidation",
                    importance=0.6,
                    scope=aggregated_scope,
                    valid_until=aggregated_valid_until,
                    metadata={
                        "original_count": len(items),
                        "source": source,
                        "llm_used": summary != f"[{source}] {aaak_encode(' | '.join(lines))}"
                    }
                )
                placeholders = ",".join("?" * len(ids))
                cursor.execute(f"DELETE FROM working_memory WHERE id IN ({placeholders})", ids)
                self.conn.commit()
            consolidated_ids.extend(ids)
            summaries_created += 1

        method = "llm" if llm_used_count == summaries_created else ("llm+aaak" if llm_used_count > 0 else "aaak")
        if not dry_run:
            cursor.execute("""
                INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview)
                VALUES (?, ?, ?)
            """, (self.session_id, len(consolidated_ids), f"{summaries_created} summaries ({method}) from {len(consolidated_ids)} items"))
            self.conn.commit()

        return {
            "status": "dry_run" if dry_run else "consolidated",
            "items_consolidated": len(consolidated_ids),
            "summaries_created": summaries_created,
            "llm_used": llm_used_count,
            "method": method,
            "consolidated_ids": consolidated_ids
        }

    def get_consolidation_log(self, limit: int = 10) -> List[Dict]:
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, session_id, items_consolidated, summary_preview, created_at
            FROM consolidation_log
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (self.session_id, limit))
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------
    def export_to_dict(self) -> Dict:
        """
        Export all BEAM data to a portable dictionary.
        Includes working_memory, episodic_memory, embeddings, scratchpad,
        and consolidation_log across ALL sessions (not just current).
        """
        cursor = self.conn.cursor()
        export = {
            "mnemosyne_export": {
                "version": "1.0",
                "export_date": datetime.now().isoformat(),
                "source_db": str(self.db_path),
                "component": "beam"
            }
        }

        # Working memory (all sessions)
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id, importance,
                   metadata_json, valid_until, superseded_by, scope,
                   recall_count, last_recalled, created_at
            FROM working_memory
            ORDER BY session_id, timestamp
        """)
        export["working_memory"] = [dict(row) for row in cursor.fetchall()]

        # Episodic memory (all sessions)
        cursor.execute("""
            SELECT rowid, id, content, source, timestamp, session_id, importance,
                   metadata_json, summary_of, valid_until, superseded_by, scope,
                   recall_count, last_recalled, created_at
            FROM episodic_memory
            ORDER BY session_id, timestamp
        """)
        export["episodic_memory"] = [dict(row) for row in cursor.fetchall()]

        # Episodic embeddings from vec_episodes
        export["episodic_embeddings"] = []
        if _vec_available(self.conn):
            try:
                cursor.execute("SELECT rowid, embedding FROM vec_episodes")
                for row in cursor.fetchall():
                    emb = row["embedding"]
                    if isinstance(emb, bytes):
                        emb = list(emb)
                    elif isinstance(emb, str):
                        try:
                            emb = json.loads(emb)
                        except Exception:
                            pass
                    export["episodic_embeddings"].append({
                        "rowid": row["rowid"],
                        "embedding": emb
                    })
            except Exception:
                pass

        # Scratchpad (all sessions)
        cursor.execute("""
            SELECT id, content, session_id, created_at, updated_at
            FROM scratchpad
            ORDER BY session_id, updated_at
        """)
        export["scratchpad"] = [dict(row) for row in cursor.fetchall()]

        # Consolidation log (all sessions)
        cursor.execute("""
            SELECT id, session_id, items_consolidated, summary_preview, created_at
            FROM consolidation_log
            ORDER BY session_id, created_at
        """)
        export["consolidation_log"] = [dict(row) for row in cursor.fetchall()]

        return export

    def import_from_dict(self, data: Dict, force: bool = False) -> Dict:
        """
        Import BEAM data from a dictionary produced by export_to_dict().
        Idempotent by default: skips records whose id already exists.
        Set force=True to overwrite existing records.
        Returns import statistics.
        """
        stats = {
            "working_memory": {"inserted": 0, "skipped": 0, "overwritten": 0},
            "episodic_memory": {"inserted": 0, "skipped": 0, "overwritten": 0, "embeddings_inserted": 0},
            "scratchpad": {"inserted": 0, "updated": 0},
            "consolidation_log": {"inserted": 0},
        }
        cursor = self.conn.cursor()

        # -- Working memory --
        for item in data.get("working_memory", []):
            mid = item.get("id")
            cursor.execute("SELECT 1 FROM working_memory WHERE id = ?", (mid,))
            exists = cursor.fetchone() is not None
            if exists and not force:
                stats["working_memory"]["skipped"] += 1
                continue
            if exists and force:
                cursor.execute("DELETE FROM working_memory WHERE id = ?", (mid,))
                stats["working_memory"]["overwritten"] += 1
            else:
                stats["working_memory"]["inserted"] += 1
            cursor.execute("""
                INSERT INTO working_memory
                (id, content, source, timestamp, session_id, importance, metadata_json,
                 valid_until, superseded_by, scope, recall_count, last_recalled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mid, item.get("content"), item.get("source"), item.get("timestamp"),
                item.get("session_id", "default"), item.get("importance", 0.5),
                item.get("metadata_json", "{}"), item.get("valid_until"),
                item.get("superseded_by"), item.get("scope", "session"),
                item.get("recall_count", 0), item.get("last_recalled"), item.get("created_at")
            ))
        self.conn.commit()

        # -- Episodic memory --
        old_to_new_rowid = {}
        for item in data.get("episodic_memory", []):
            mid = item.get("id")
            cursor.execute("SELECT rowid FROM episodic_memory WHERE id = ?", (mid,))
            existing = cursor.fetchone()
            if existing and not force:
                stats["episodic_memory"]["skipped"] += 1
                old_to_new_rowid[item.get("rowid")] = existing["rowid"]
                continue
            if existing and force:
                cursor.execute("DELETE FROM episodic_memory WHERE id = ?", (mid,))
                stats["episodic_memory"]["overwritten"] += 1
            else:
                stats["episodic_memory"]["inserted"] += 1
            cursor.execute("""
                INSERT INTO episodic_memory
                (id, content, source, timestamp, session_id, importance, metadata_json,
                 summary_of, valid_until, superseded_by, scope, recall_count, last_recalled, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mid, item.get("content"), item.get("source"), item.get("timestamp"),
                item.get("session_id", "default"), item.get("importance", 0.5),
                item.get("metadata_json", "{}"), item.get("summary_of", ""),
                item.get("valid_until"), item.get("superseded_by"),
                item.get("scope", "session"), item.get("recall_count", 0),
                item.get("last_recalled"), item.get("created_at")
            ))
            new_rowid = cursor.lastrowid
            old_to_new_rowid[item.get("rowid")] = new_rowid
        self.conn.commit()

        # -- Episodic embeddings --
        vec_ok = _vec_available(self.conn)
        for emb_item in data.get("episodic_embeddings", []):
            old_rowid = emb_item.get("rowid")
            new_rowid = old_to_new_rowid.get(old_rowid)
            if not new_rowid:
                continue
            embedding = emb_item.get("embedding")
            if not embedding:
                continue
            if vec_ok:
                try:
                    _vec_insert(self.conn, new_rowid, embedding)
                    stats["episodic_memory"]["embeddings_inserted"] += 1
                except Exception:
                    pass
        if vec_ok:
            self.conn.commit()

        # -- Scratchpad --
        for item in data.get("scratchpad", []):
            pid = item.get("id")
            cursor.execute("SELECT 1 FROM scratchpad WHERE id = ?", (pid,))
            exists = cursor.fetchone() is not None
            if exists:
                cursor.execute("""
                    UPDATE scratchpad SET content=?, session_id=?, created_at=?, updated_at=?
                    WHERE id=?
                """, (item.get("content"), item.get("session_id", "default"),
                      item.get("created_at"), item.get("updated_at"), pid))
                stats["scratchpad"]["updated"] += 1
            else:
                cursor.execute("""
                    INSERT INTO scratchpad (id, content, session_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (pid, item.get("content"), item.get("session_id", "default"),
                      item.get("created_at"), item.get("updated_at")))
                stats["scratchpad"]["inserted"] += 1
        self.conn.commit()

        # -- Consolidation log --
        for item in data.get("consolidation_log", []):
            cursor.execute("""
                INSERT INTO consolidation_log (session_id, items_consolidated, summary_preview, created_at)
                VALUES (?, ?, ?, ?)
            """, (item.get("session_id", "default"), item.get("items_consolidated", 0),
                  item.get("summary_preview", ""), item.get("created_at")))
            stats["consolidation_log"]["inserted"] += 1
        self.conn.commit()

        return stats
