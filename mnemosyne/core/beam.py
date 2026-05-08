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
from typing import List, Dict, Optional, Any, Set, Union
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

# Tiered episodic degradation
TIER2_DAYS = int(os.environ.get("MNEMOSYNE_TIER2_DAYS", "30"))
TIER3_DAYS = int(os.environ.get("MNEMOSYNE_TIER3_DAYS", "180"))
TIER1_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER1_WEIGHT", "1.0"))
TIER2_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER2_WEIGHT", "0.5"))
TIER3_WEIGHT = float(os.environ.get("MNEMOSYNE_TIER3_WEIGHT", "0.25"))
DEGRADE_BATCH_SIZE = int(os.environ.get("MNEMOSYNE_DEGRADE_BATCH", "100"))
SMART_COMPRESS = os.environ.get("MNEMOSYNE_SMART_COMPRESS", "1") not in ("0", "false", "no")
TIER3_MAX_CHARS = int(os.environ.get("MNEMOSYNE_TIER3_MAX_CHARS", "300"))

# Veracity weighting (memory confidence)
STATED_WEIGHT = float(os.environ.get("MNEMOSYNE_STATED_WEIGHT", "1.0"))
INFERRED_WEIGHT = float(os.environ.get("MNEMOSYNE_INFERRED_WEIGHT", "0.7"))
TOOL_WEIGHT = float(os.environ.get("MNEMOSYNE_TOOL_WEIGHT", "0.5"))
IMPORTED_WEIGHT = float(os.environ.get("MNEMOSYNE_IMPORTED_WEIGHT", "0.6"))
UNKNOWN_WEIGHT = float(os.environ.get("MNEMOSYNE_UNKNOWN_WEIGHT", "0.8"))

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
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
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
            veracity TEXT DEFAULT 'unknown',
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
            veracity TEXT DEFAULT 'unknown',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_session ON episodic_memory(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_timestamp ON episodic_memory(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_source ON episodic_memory(source)")

    # --- Tiered degradation migration (v2.3) ---
    try:
        cursor.execute("ALTER TABLE episodic_memory ADD COLUMN tier INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE episodic_memory ADD COLUMN degraded_at TEXT")
    except sqlite3.OperationalError:
        pass
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_tier ON episodic_memory(tier)")

    # --- Veracity migration (v2.4) ---
    try:
        cursor.execute("ALTER TABLE working_memory ADD COLUMN veracity TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE episodic_memory ADD COLUMN veracity TEXT DEFAULT 'unknown'")
    except sqlite3.OperationalError:
        pass

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

    # --- Migration: multi-agent identity layer (v2.1) ---
    _add_column_if_missing(conn, "working_memory", "author_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "author_type", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "working_memory", "channel_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "author_id", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "author_type", "TEXT DEFAULT NULL")
    _add_column_if_missing(conn, "episodic_memory", "channel_id", "TEXT DEFAULT NULL")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_author ON working_memory(author_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_wm_channel ON working_memory(channel_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_author ON episodic_memory(author_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_em_channel ON episodic_memory(channel_id)")

    # --- FACTS (LLM-extracted structured knowledge) ---
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            fact_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            timestamp TEXT,
            source_msg_id TEXT,
            confidence REAL DEFAULT 1.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_source ON facts(source_msg_id)")

    # FTS5 for full-text search on facts
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_facts USING fts5(
            subject, predicate, object, content='facts'
        )
    """)
    # Triggers to keep FTS5 in sync
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
            INSERT INTO fts_facts(rowid, subject, predicate, object)
            VALUES (new.rowid, new.subject, new.predicate, new.object);
        END
    """)
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
            INSERT INTO fts_facts(fts_facts, rowid, subject, predicate, object)
            VALUES ('delete', old.rowid, old.subject, old.predicate, old.object);
        END
    """)

    # Vector table for facts (sqlite-vec)
    try:
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_facts USING vec0(
                embedding {effective_vec_type}[{EMBEDDING_DIM}]
            )
        """)
    except (sqlite3.OperationalError, RuntimeError):
        pass  # sqlite-vec not available


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


def _normalize_weights(vec_weight: Optional[float], fts_weight: Optional[float],
                       importance_weight: Optional[float]) -> tuple[float, float, float]:
    """
    Normalize hybrid scoring weights to sum to 1.0.

    Falls back to env vars, then defaults:
        vec_weight      -> MNEMOSYNE_VEC_WEIGHT      -> 0.5
        fts_weight      -> MNEMOSYNE_FTS_WEIGHT      -> 0.3
        importance_weight -> MNEMOSYNE_IMPORTANCE_WEIGHT -> 0.2

    After normalization: vw + fw + iw == 1.0
    """
    vw = vec_weight if vec_weight is not None else float(os.environ.get("MNEMOSYNE_VEC_WEIGHT", "0.5"))
    fw = fts_weight if fts_weight is not None else float(os.environ.get("MNEMOSYNE_FTS_WEIGHT", "0.3"))
    iw = importance_weight if importance_weight is not None else float(os.environ.get("MNEMOSYNE_IMPORTANCE_WEIGHT", "0.2"))

    # Clamp to non-negative
    vw = max(0.0, vw)
    fw = max(0.0, fw)
    iw = max(0.0, iw)

    total = vw + fw + iw
    if total == 0.0:
        # All zero = revert to defaults
        return (0.5, 0.3, 0.2)

    return (vw / total, fw / total, iw / total)


def _recency_decay(timestamp_str: str, halflife_hours: float = RECENCY_HALFLIFE_HOURS) -> float:
    """Calculate recency decay factor. 1.0 = brand new, ~0.5 = one halflife old.
    
    Exponential decay based on age. Returns 0.5 for unknown/invalid timestamps.
    """
    if not timestamp_str:
        return 0.5  # Unknown age = neutral
    try:
        ts = datetime.fromisoformat(timestamp_str)
        age_hours = (datetime.now() - ts).total_seconds() / 3600.0
        return math.exp(-age_hours / halflife_hours)
    except Exception:
        return 0.5


def _parse_query_time(query_time: Optional[Union[str, datetime]]) -> datetime:
    """Parse query_time parameter into a datetime object.

    - None -> datetime.now()
    - str  -> parsed from ISO format
    - datetime -> returned as-is
    """
    if query_time is None:
        return datetime.now()
    if isinstance(query_time, datetime):
        return query_time
    if isinstance(query_time, str):
        # Try ISO format with various precisions
        try:
            return datetime.fromisoformat(query_time)
        except ValueError:
            # Try appending time if only date provided
            try:
                return datetime.fromisoformat(f"{query_time}T00:00:00")
            except ValueError:
                raise ValueError(f"Invalid query_time format: {query_time!r}. Expected ISO datetime string.")
    raise TypeError(f"query_time must be str, datetime, or None; got {type(query_time).__name__}")


# Fast-path timestamp parsing cache
_TS_CACHE: Dict[str, datetime] = {}
_TS_CACHE_MAX = 2000


def _parse_ts_fast(ts: str) -> Optional[datetime]:
    """Parse ISO timestamp with LRU-style cache for performance."""
    if not ts:
        return None
    cached = _TS_CACHE.get(ts)
    if cached is not None:
        return cached
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if len(_TS_CACHE) >= _TS_CACHE_MAX:
        _TS_CACHE.clear()
    _TS_CACHE[ts] = dt
    return dt


def _temporal_boost(memory_timestamp_str: str, query_time: datetime,
                    halflife_hours: float = 24.0) -> float:
    """Temporal boost factor based on proximity to query_time.

    Formula: exp(-hours_delta / halflife)
    - memory at query_time -> boost = 1.0
    - memory 1 halflife away -> boost = exp(-1) ≈ 0.368
    - memory 3 halflives away -> boost = exp(-3) ≈ 0.050

    Returns 0.0 for invalid timestamps or future timestamps (clamped to now).
    """
    ts = _parse_ts_fast(memory_timestamp_str)
    if ts is None:
        return 0.0

    # Clamp future timestamps to query_time (no negative deltas)
    if ts > query_time:
        ts = query_time

    hours_delta = (query_time - ts).total_seconds() / 3600.0
    return math.exp(-hours_delta / halflife_hours)


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
    Extract structured facts from content using LLM and store as triples + facts table.
    Called internally by remember() when extract=True.

    Stores in TWO places:
    1. TripleStore (entity-level triples, backward compat)
    2. facts table (structured SPO facts for fact_recall())
    """
    try:
        from mnemosyne.core.extraction import extract_facts_safe
        from mnemosyne.core.triples import TripleStore
        
        facts = extract_facts_safe(content)
        if not facts:
            return
        
        # Store in triples (existing behavior)
        triples = TripleStore(db_path=beam.db_path)
        triples.add_facts(memory_id, facts, source=source, confidence=0.7)

        # ALSO store in facts table (new cloud extraction path)
        _store_facts_in_table(beam, memory_id, content, source, facts)

    except Exception:
        # Fact extraction is best-effort; never fail remember() because of it
        pass


def _store_facts_in_table(beam: "BeamMemory", memory_id: str,
                          content: str, source: str, facts: list):
    """Store extracted free-text facts as simple SPO entries in the facts table."""
    import hashlib
    cursor = beam.conn.cursor()
    timestamp = __import__('datetime').datetime.now().isoformat()
    
    for i, fact_text in enumerate(facts):
        # Derive subject from source, predicate = "stated", object = fact text
        subject = source or "user"
        fact_id = hashlib.sha256(
            f"{memory_id}:fact:{i}:{fact_text[:50]}".encode()
        ).hexdigest()[:24]

        try:
            cursor.execute("""
                INSERT OR IGNORE INTO facts
                (fact_id, session_id, subject, predicate, object,
                 timestamp, source_msg_id, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fact_id,
                beam.session_id,
                subject,
                "stated",
                fact_text,
                timestamp,
                memory_id,
                0.7,
            ))
        except Exception:
            continue  # Best-effort per fact
    
    beam.conn.commit()


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

    def __init__(self, session_id: str = "default", db_path: Path = None,
                 author_id: str = None, author_type: str = None,
                 channel_id: str = None, use_cloud: bool = False):
        self.session_id = session_id
        self.author_id = author_id
        self.author_type = author_type
        self.channel_id = channel_id or session_id  # default channel = session
        self.db_path = db_path or DEFAULT_DB_PATH
        self.use_cloud = use_cloud  # Enable LLM fact extraction during remember()
        self._extraction_client = None  # Lazy-loaded ExtractionClient
        self._extraction_buffer = []  # Buffer for batch extraction
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
                 memory_id: str = None,
                 extract_entities: bool = False,
                 extract: bool = False,
                 veracity: str = "unknown") -> str:
        """Store into working_memory. Deduplicates exact content matches.

        When called from the legacy-compatible Mnemosyne.remember() path,
        memory_id is passed through so the legacy memories row and BEAM
        working_memory row stay addressable by the same ID. Direct BEAM calls
        still generate their own deterministic ID.

        Args:
            content: The text to remember
            source: Origin of the memory (e.g., "conversation", "document")
            importance: 0.0-1.0 relevance score
            metadata: Optional dict of additional fields
            valid_until: ISO timestamp when this memory expires
            scope: "session" or "global"
            memory_id: Optional pre-generated ID from legacy layer
            extract_entities: If True, extract and store entity mentions as triples
            extract: If True, extract structured facts from content using LLM
                and store as triples. Default False.
            veracity: Confidence level — 'stated', 'inferred', 'tool', 'imported', 'unknown'
        """
        # --- Deduplication: exact match ---
        existing_id = self._find_duplicate(content)
        if existing_id:
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE working_memory
                SET importance = MAX(importance, ?), timestamp = ?, source = ?,
                    valid_until = COALESCE(?, valid_until),
                    scope = COALESCE(?, scope),
                    author_id = COALESCE(?, author_id),
                    author_type = COALESCE(?, author_type),
                    channel_id = COALESCE(?, channel_id)
                WHERE id = ? AND session_id = ?
            """, (importance, datetime.now().isoformat(), source,
                  valid_until, scope,
                  self.author_id, self.author_type, self.channel_id,
                  existing_id, self.session_id))
            self.conn.commit()
            return existing_id

        memory_id = memory_id or _generate_id(content)
        timestamp = datetime.now().isoformat()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO working_memory
            (id, content, source, timestamp, session_id, importance, metadata_json, valid_until, scope,
             author_id, author_type, channel_id, veracity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, content, source, timestamp, self.session_id, importance,
              json.dumps(metadata or {}), valid_until, scope,
              self.author_id, self.author_type, self.channel_id, veracity))
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
                INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, metadata_json,
                author_id, author_type, channel_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                item["content"],
                item.get("source", "conversation"),
                timestamp,
                self.session_id,
                item.get("importance", 0.5),
                json.dumps(item.get("metadata") or {}),
                item.get("author_id", self.author_id),
                item.get("author_type", self.author_type),
                item.get("channel_id", self.channel_id)
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
        """Get working_memory for prompt injection.
        Global memories first, then sorted by importance (high first),
        then by recency. High-importance rules/bans surface reliably."""
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
                importance DESC,
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

    def get_working_stats(self, author_id: str = None, author_type: str = None,
                          channel_id: str = None) -> Dict:
        cursor = self.conn.cursor()
        where_clauses = []
        params = []
        if author_id:
            where_clauses.append("author_id = ?")
            params.append(author_id)
        if author_type:
            where_clauses.append("author_type = ?")
            params.append(author_type)
        if channel_id:
            where_clauses.append("channel_id = ?")
            params.append(channel_id)
        where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        cursor.execute(f"SELECT COUNT(*) FROM working_memory{where_str}", params)
        total = cursor.fetchone()[0]
        cursor.execute(f"SELECT timestamp FROM working_memory{where_str} ORDER BY timestamp DESC LIMIT 1", params)
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
            (id, content, source, timestamp, session_id, importance, metadata_json, summary_of, valid_until, scope,
             author_id, author_type, channel_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (memory_id, summary, source, timestamp, self.session_id, importance,
              json.dumps(metadata or {}), ",".join(source_wm_ids), valid_until, scope,
              self.author_id, self.author_type, self.channel_id))
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

    def recall(self, query: str, top_k: int = 5, *,
               from_date: Optional[str] = None, to_date: Optional[str] = None,
               source: Optional[str] = None, topic: Optional[str] = None,
               author_id: Optional[str] = None,
               author_type: Optional[str] = None,
               channel_id: Optional[str] = None,
               veracity: Optional[str] = None,
               temporal_weight: float = 0.0,
               query_time: Optional[Any] = None,
               temporal_halflife: Optional[float] = None,
               vec_weight: float = None,
               fts_weight: float = None,
               importance_weight: float = None) -> List[Dict]:
        """
        Hybrid recall across working_memory + episodic_memory.
        Uses sqlite-vec + FTS5 for episodic, FTS5 for working.
        Falls back to recency-only for working memory if FTS5 unavailable.

        Temporal filtering:
            from_date/to_date: ISO date strings (YYYY-MM-DD) to filter by timestamp.
            source: Filter by memory source (e.g., 'cron', 'user', 'conversation').
            topic: Filter by topic tag (stored in source field for now, pending dedicated column).

        Multi-agent identity filtering (v2.1):
            author_id: Filter by author (e.g., 'abdias', 'codex-agent').
            author_type: Filter by author type ('human', 'agent', 'system').
            channel_id: Filter by channel/group (e.g., 'fluxspeak-team').

        Temporal scoring (Phase 3):
            temporal_weight: Float 0.0-1.0. Soft boost for memories near query_time.
                0.0 = no temporal boost (default, backward compatible).
            query_time: Target time for temporal scoring. None = now().
            temporal_halflife: Hours for temporal decay. None = env var or 24h default.

        Temporal scoring (Phase 3):
            temporal_weight: Float 0.0-1.0. Soft boost for memories near query_time.
                0.0 = no temporal boost (default, backward compatible).
            query_time: Target time for temporal scoring. None = now().
            temporal_halflife: Hours for temporal decay. None = env var or 24h default.

        Configurable hybrid scoring (Phase 4):
            vec_weight: Weight for vector (dense) similarity in episodic scoring.
                None = use env var MNEMOSYNE_VEC_WEIGHT or default 0.5.
            fts_weight: Weight for FTS5 text relevance in episodic scoring.
                None = use env var MNEMOSYNE_FTS_WEIGHT or default 0.3.
            importance_weight: Weight for importance score in all scoring.
                None = use env var MNEMOSYNE_IMPORTANCE_WEIGHT or default 0.2.

            The three episodic weights are automatically normalized to sum to 1.0.
            Working memory uses a derived split: keyword gets (1 - importance_weight) * 0.6,
            recency gets (1 - importance_weight) * 0.4.
        """
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()

        # ---- Configurable hybrid scoring setup (Phase 4) ----
        vw, fw, iw = _normalize_weights(vec_weight, fts_weight, importance_weight)

        # ---- Temporal scoring setup ----
        parsed_query_time = _parse_query_time(query_time)
        if temporal_halflife is not None:
            th_halflife = temporal_halflife
        else:
            th_halflife = float(os.environ.get("MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS", "24"))

        # ---- Working memory (FTS5 fast path) ----
        try:
            wm_fts = _fts_search_working(self.conn, query, k=max(top_k * 3, 50))
        except Exception:
            wm_fts = []

        wm_ids = {r["id"] for r in wm_fts}
        wm_ranks = {r["id"]: r["rank"] for r in wm_fts}

        # Build temporal filter clause for working memory
        wm_where_clauses = [
            "(valid_until IS NULL OR valid_until > ?)",
            "superseded_by IS NULL"
        ]
        wm_params = [datetime.now().isoformat()]
        
        # Session scope: channel filter only when explicitly specified.
        # Author-only searches have no session/channel restriction.
        if channel_id:
            wm_where_clauses.append("(session_id = ? OR scope = 'global' OR channel_id = ?)")
            wm_params.extend([self.session_id, channel_id])
        elif author_id or author_type:
            wm_where_clauses.append("(1=1)")
        else:
            wm_where_clauses.append("(session_id = ? OR scope = 'global')")
            wm_params.append(self.session_id)
        
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
        if veracity:
            wm_where_clauses.append("veracity = ?")
            wm_params.append(veracity)
        if author_id:
            wm_where_clauses.append("author_id = ?")
            wm_params.append(author_id)
        if author_type:
            wm_where_clauses.append("author_type = ?")
            wm_params.append(author_type)
        if channel_id:
            wm_where_clauses.append("channel_id = ?")
            wm_params.append(channel_id)
        
        wm_where = " AND ".join(wm_where_clauses)

        if wm_ids:
            placeholders = ",".join("?" * len(wm_ids))
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity
                FROM working_memory
                WHERE id IN ({placeholders})
                  AND {wm_where}
            """, (*tuple(wm_ids), *wm_params))
            rows = cursor.fetchall()
        else:
            # Fallback: fetch recent items and score in Python (old path)
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity
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
                # Phase 4: configurable scoring for working memory
                # keyword_share = (1 - importance_weight) * 0.6, recency_share = (1 - importance_weight) * 0.4
                kw_share = (1.0 - iw) * 0.6
                rc_share = (1.0 - iw) * 0.4
                base_score = relevance * kw_share + row["importance"] * iw
                score = base_score * (rc_share + (1.0 - rc_share) * decay)
                # Temporal boost (Phase 3)
                if temporal_weight > 0.0:
                    t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                    score *= (1.0 + temporal_weight * t_boost)
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
                    "author_id": row["author_id"] if "author_id" in row.keys() else None,
                    "author_type": row["author_type"] if "author_type" in row.keys() else None,
                    "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                    "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
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
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity
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
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
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
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "entity_match": True
                    })
            
            # Also check episodic memory for entity matches
            em_placeholders = ",".join("?" * len(entity_memory_ids))
            if channel_id:
                em_entity_scope = "(session_id = ? OR scope = 'global' OR channel_id = ?)"
                em_entity_params = [*tuple(entity_memory_ids), self.session_id, channel_id]
            elif author_id or author_type:
                em_entity_scope = "(1=1)"
                em_entity_params = [*tuple(entity_memory_ids)]
            else:
                em_entity_scope = "(session_id = ? OR scope = 'global')"
                em_entity_params = [*tuple(entity_memory_ids), self.session_id]
            em_entity_params.extend([datetime.now().isoformat()])
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity
                FROM episodic_memory
                WHERE id IN ({em_placeholders})
                  AND {em_entity_scope}
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND superseded_by IS NULL
            """, (*em_entity_params,))
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
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
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
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
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
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity
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
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
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
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "fact_match": True
                    })
            
            # Also check episodic memory for fact matches
            if channel_id:
                fact_em_scope = "(session_id = ? OR scope = 'global' OR channel_id = ?)"
                fact_em_params = [*tuple(fact_memory_ids), self.session_id, channel_id]
            elif author_id or author_type:
                fact_em_scope = "(1=1)"
                fact_em_params = [*tuple(fact_memory_ids)]
            else:
                fact_em_scope = "(session_id = ? OR scope = 'global')"
                fact_em_params = [*tuple(fact_memory_ids), self.session_id]
            fact_em_params.extend([datetime.now().isoformat()])
            cursor.execute(f"""
                SELECT id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id, veracity
                FROM episodic_memory
                WHERE id IN ({placeholders})
                  AND {fact_em_scope}
                  AND (valid_until IS NULL OR valid_until > ?)
                  AND superseded_by IS NULL
            """, (*fact_em_params,))
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
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
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
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
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
            "(valid_until IS NULL OR valid_until > ?)",
            "superseded_by IS NULL"
        ]
        em_params = [datetime.now().isoformat()]
        
        # Session scope: channel filter only when explicitly specified.
        # Author-only searches have no session/channel restriction.
        if channel_id:
            em_where_clauses.append("(session_id = ? OR scope = 'global' OR channel_id = ?)")
            em_params.extend([self.session_id, channel_id])
        elif author_id or author_type:
            em_where_clauses.append("(1=1)")
        else:
            em_where_clauses.append("(session_id = ? OR scope = 'global')")
            em_params.append(self.session_id)
        
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
        if veracity:
            em_where_clauses.append("veracity = ?")
            em_params.append(veracity)
        if author_id:
            em_where_clauses.append("author_id = ?")
            em_params.append(author_id)
        if author_type:
            em_where_clauses.append("author_type = ?")
            em_params.append(author_type)
        if channel_id:
            em_where_clauses.append("channel_id = ?")
            em_params.append(channel_id)
        
        em_where = " AND ".join(em_where_clauses)
        
        if episodic_rowids:
            placeholders = ",".join("?" * len(episodic_rowids))
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT rowid, id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id
                FROM episodic_memory
                WHERE rowid IN ({placeholders})
                  AND {em_where}
            """, (*tuple(episodic_rowids), *em_params))
        for row in cursor.fetchall():
            rid = row["rowid"]
            sim = vec_results.get(rid, 0.0)
            fts = fts_results.get(rid, 0.0)
            decay = _recency_decay(row["timestamp"])
            # Phase 4: configurable hybrid scoring for episodic memory
            # vec_weight + fts_weight + importance_weight are normalized to sum to 1.0
            base_score = sim * vw + fts * fw + row["importance"] * iw
            score = base_score * (0.7 + 0.3 * decay)
            # Temporal boost (Phase 3)
            if temporal_weight > 0.0:
                t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                score *= (1.0 + temporal_weight * t_boost)
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
                "author_id": row["author_id"] if "author_id" in row.keys() else None,
                "author_type": row["author_type"] if "author_type" in row.keys() else None,
                "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None
            })

        # Fallback: if no episodic matches from vec/FTS, scan recent episodic entries
        if not episodic_rowids:
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT rowid, id, content, source, timestamp, importance, recall_count, last_recalled, valid_until, superseded_by, scope, author_id, author_type, channel_id
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
                    # Phase 4: configurable scoring for episodic fallback
                    kw_share = (1.0 - iw) * 0.6
                    rc_share = (1.0 - iw) * 0.4
                    base_score = relevance * kw_share + row["importance"] * iw
                    score = base_score * (rc_share + (1.0 - rc_share) * decay)
                    # Temporal boost (Phase 3)
                    if temporal_weight > 0.0:
                        t_boost = _temporal_boost(row["timestamp"], parsed_query_time, th_halflife)
                        score *= (1.0 + temporal_weight * t_boost)
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
                        "author_id": row["author_id"] if "author_id" in row.keys() else None,
                        "author_type": row["author_type"] if "author_type" in row.keys() else None,
                        "channel_id": row["channel_id"] if "channel_id" in row.keys() else None,
                        "veracity": row["veracity"] if "veracity" in row.keys() else "unknown",
                        "valid_until": row["valid_until"] if "valid_until" in row.keys() else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None
                    })

        # --- Tiered degradation weighting: apply tier multiplier to episodic scores ---
        weight_map = {1: TIER1_WEIGHT, 2: TIER2_WEIGHT, 3: TIER3_WEIGHT}
        veracity_map = {"stated": STATED_WEIGHT, "inferred": INFERRED_WEIGHT,
                        "tool": TOOL_WEIGHT, "imported": IMPORTED_WEIGHT,
                        "unknown": UNKNOWN_WEIGHT}
        em_ids_for_tier = [r["id"] for r in results if r.get("tier") == "episodic"]
        if em_ids_for_tier:
            placeholders = ",".join("?" * len(em_ids_for_tier))
            tier_rows = cursor.execute(
                f"SELECT id, tier, veracity FROM episodic_memory WHERE id IN ({placeholders})",
                em_ids_for_tier
            ).fetchall()
            tier_lookup = {r["id"]: (r["tier"] or 1) for r in tier_rows}
            veracity_lookup = {r["id"]: (r["veracity"] or "unknown") for r in tier_rows}
            for r in results:
                if r.get("tier") == "episodic":
                    ep_tier = tier_lookup.get(r["id"], 1)
                    ep_veracity = veracity_lookup.get(r["id"], "unknown")
                    r["degradation_tier"] = ep_tier
                    r["veracity"] = ep_veracity
                    r["score"] *= weight_map.get(ep_tier, 1.0)
                    r["score"] *= veracity_map.get(ep_veracity, UNKNOWN_WEIGHT)

        results.sort(key=lambda x: x["score"], reverse=True)
        final_results = results[:top_k]

        # --- Recall tracking: increment counts + set last_recalled ---
        now_iso = datetime.now().isoformat()
        wm_ids = [r["id"] for r in final_results if r.get("tier") == "working"]
        em_ids = [r["id"] for r in final_results if r.get("tier") == "episodic"]
        cursor = self.conn.cursor()
        if channel_id:
            rec_scope = "(session_id = ? OR scope = 'global' OR channel_id = ?)"
        elif author_id or author_type:
            rec_scope = "(1=1)"
        else:
            rec_scope = "(session_id = ? OR scope = 'global')"
        if wm_ids:
            placeholders = ",".join("?" * len(wm_ids))
            rec_params = [now_iso, *tuple(wm_ids)]
            if channel_id:
                rec_params.extend([self.session_id, channel_id])
            elif not (author_id or author_type):
                rec_params.append(self.session_id)
            cursor.execute(f"""
                UPDATE working_memory
                SET recall_count = recall_count + 1, last_recalled = ?
                WHERE id IN ({placeholders}) AND {rec_scope}
            """, (*rec_params,))
        if em_ids:
            placeholders = ",".join("?" * len(em_ids))
            rec_params = [now_iso, *tuple(em_ids)]
            if channel_id:
                rec_params.extend([self.session_id, channel_id])
            elif not (author_id or author_type):
                rec_params.append(self.session_id)
            cursor.execute(f"""
                UPDATE episodic_memory
                SET recall_count = recall_count + 1, last_recalled = ?
                WHERE id IN ({placeholders}) AND {rec_scope}
            """, (*rec_params,))
        self.conn.commit()

        return final_results

    def fact_recall(self, query: str, top_k: int = 30) -> List[Dict]:
        """Search the facts table (LLM-extracted structured knowledge).

        Returns facts as list of dicts with: content, score, fact_id, subject, predicate.

        Falls back gracefully if facts table is empty or sqlite-vec unavailable.
        """
        cursor = self.conn.cursor()
        results = []
        query_lower = query.lower()

        # Try FTS5 search first
        try:
            fts_rows = cursor.execute(
                "SELECT rowid, rank FROM fts_facts WHERE fts_facts MATCH ? ORDER BY rank LIMIT ?",
                (query, top_k * 3)
            ).fetchall()
        except Exception:
            fts_rows = []

        if not fts_rows:
            # Fallback: simple LIKE scan
            for word in query_lower.split()[:6]:
                if len(word) < 3:
                    continue
                try:
                    like_rows = cursor.execute(
                        "SELECT rowid FROM facts WHERE subject LIKE ? OR predicate LIKE ? OR object LIKE ? LIMIT ?",
                        (f"%{word}%", f"%{word}%", f"%{word}%", top_k)
                    ).fetchall()
                except Exception:
                    continue
                for row in like_rows:
                    if row["rowid"] not in {r["rowid"] for r in fts_rows}:
                        fts_rows.append({"rowid": row["rowid"], "rank": 0})

        if not fts_rows:
            return []

        # Get full rows for matched fact IDs
        fact_ids = [r["rowid"] for r in fts_rows[:top_k]]
        placeholders = ",".join("?" * len(fact_ids))

        try:
            cursor.execute(f"""
                SELECT fact_id, subject, predicate, object,
                       timestamp, confidence
                FROM facts
                WHERE rowid IN ({placeholders})
                ORDER BY confidence DESC
                LIMIT ?
            """, (*fact_ids, top_k))
            fact_rows = cursor.fetchall()
        except Exception:
            return []

        for row in fact_rows:
            fact_text = row["object"] if row["object"] else f"{row['subject']} {row['predicate']} {row['object']}"
            results.append({
                "content": fact_text,
                "score": row.get("confidence", 0.5),
                "fact_id": row["fact_id"],
                "subject": row.get("subject", ""),
                "predicate": row.get("predicate", ""),
            })

        return results

    def get_episodic_stats(self, author_id: str = None, author_type: str = None,
                           channel_id: str = None) -> Dict:
        cursor = self.conn.cursor()
        where_clauses = []
        params = []
        if author_id:
            where_clauses.append("author_id = ?")
            params.append(author_id)
        if author_type:
            where_clauses.append("author_type = ?")
            params.append(author_type)
        if channel_id:
            where_clauses.append("channel_id = ?")
            params.append(channel_id)
        where_str = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        
        cursor.execute(f"SELECT COUNT(*) FROM episodic_memory{where_str}", params)
        total = cursor.fetchone()[0]
        cursor.execute(f"SELECT timestamp FROM episodic_memory{where_str} ORDER BY timestamp DESC LIMIT 1", params)
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
    # Tiered Episodic Degradation
    # ------------------------------------------------------------------
    def _extract_key_signal(self, content: str, max_chars: int = 300) -> str:
        """Extract the highest-signal sentences from content for tier 3 compression.

        Scores each sentence by entity/keyword density (proper nouns, technical
        terms, preference indicators) and keeps top-scoring sentences until the
        character budget is reached. Falls back to first-N-chars if content has
        no clear sentence boundaries.
        """
        import re
        if len(content) <= max_chars:
            return content

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', content)
        if len(sentences) <= 1:
            # No sentence boundaries — take first max_chars
            return content[:max_chars] + " [...]"

        # Scoring patterns
        signal_patterns = [
            (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', 3),     # Proper nouns: "GitHub Actions", "Docker Compose"
            (r'\b[A-Z]{2,}\b', 3),                            # Acronyms: "XKCD", "CI/CD", "API", "AWS"
            (r'\b(Docker|Kubernetes|AWS|GCP|Azure|Terraform|Python|Rust|Go|TypeScript|React|Next\.?js|Node\.?js|SQLite|Postgres|Redis|nginx|systemd|Linux|macOS|Windows)\b', 4),
            (r'\b(prefers?|uses?|likes?|loves?|hates?|dislikes?|wants?|needs?)\b', 2),  # Preference indicators
            (r'\b(password|token|secret|key|credential|auth|encrypt|decrypt|private)\b', 3),  # Security terms
            (r'\b(production|staging|deploy|database|backup|migration)\b', 2),  # Infra terms
            (r'\b(critical|urgent|important|breaking|incident|outage|down)\b', 3),  # Urgency
            (r'\b(always|never|every|must|should)\b', 1),  # Emphasis words
            (r'\b(\d{1,3}\.\d{1,3}\.\d{1,3})\b', 3),  # Version numbers
            (r'\b(https?://|www\.|[a-z]+\.[a-z]{2,})\b', 2),  # URLs / domains
            (r'["\'].*?["\']', 1),  # Quoted strings
        ]

        scored = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            score = 0
            # Bonus for shorter sentences (signal density)
            if len(sentence) < 120:
                score += 1
            for pattern, weight in signal_patterns:
                score += len(re.findall(pattern, sentence)) * weight
            scored.append((score, sentence))

        # Sort by score descending, keep top sentences up to max_chars
        scored.sort(key=lambda x: x[0], reverse=True)
        result = []
        used = 0
        for _, sentence in scored:
            if used + len(sentence) + 1 > max_chars:
                break
            result.append(sentence)
            used += len(sentence) + 1  # +1 for space

        if not result:
            return content[:max_chars] + " [...]"

        compressed = " ".join(result)
        if len(content) > len(compressed):
            compressed += " [...]"
        return compressed

    def degrade_episodic(self, dry_run: bool = False) -> Dict:
        """Degrade old episodic memories through tier 1→2→3 compression.

        Tier 1 (0-TIER2_DAYS): Full detail, 1.0x recall weight
        Tier 2 (TIER2_DAYS-TIER3_DAYS): LLM-summarized, 0.5x weight
        Tier 3 (TIER3_DAYS+): Text extraction compressed, 0.25x weight

        Returns summary of tier transitions performed.
        """
        cursor = self.conn.cursor()
        now = datetime.now()
        results = {"status": "dry_run" if dry_run else "degraded",
                   "tier1_to_tier2": 0, "tier2_to_tier3": 0}

        # --- Find candidates for degradation ---
        tier2_cutoff = (now - timedelta(days=TIER2_DAYS)).isoformat()
        tier3_cutoff = (now - timedelta(days=TIER3_DAYS)).isoformat()

        # Tier 1 → Tier 2: old enough, still at tier 1
        cursor.execute("""
            SELECT id, content, importance FROM episodic_memory
            WHERE tier = 1 AND created_at < ?
            ORDER BY created_at ASC LIMIT ?
        """, (tier2_cutoff, DEGRADE_BATCH_SIZE))
        tier1_rows = cursor.fetchall()

        # Tier 2 → Tier 3: very old, at tier 2
        cursor.execute("""
            SELECT id, content FROM episodic_memory
            WHERE tier = 2 AND created_at < ?
            ORDER BY created_at ASC LIMIT ?
        """, (tier3_cutoff, DEGRADE_BATCH_SIZE // 2))
        tier2_rows = cursor.fetchall()

        if dry_run:
            results["tier1_to_tier2"] = len(tier1_rows)
            results["tier2_to_tier3"] = len(tier2_rows)
            return results

        # --- Degrade tier 1 → tier 2: LLM summarization ---
        from mnemosyne.core import local_llm
        for row in tier1_rows:
            try:
                compressed = row["content"]
                if local_llm.llm_available() and len(row["content"]) > 300:
                    summary = local_llm.summarize_memories([row["content"]])
                    if summary:
                        compressed = summary[:400]
                cursor.execute(
                    "UPDATE episodic_memory SET content = ?, tier = 2, degraded_at = ? WHERE id = ?",
                    (compressed[:800], now.isoformat(), row["id"])
                )
                results["tier1_to_tier2"] += 1
            except Exception:
                pass

        # --- Degrade tier 2 → tier 3: smart extraction (keep key entities) ---
        for row in tier2_rows:
            try:
                content = row["content"]
                if SMART_COMPRESS and len(content) > TIER3_MAX_CHARS:
                    compressed = self._extract_key_signal(content, max_chars=TIER3_MAX_CHARS)
                else:
                    compressed = content[:TIER3_MAX_CHARS]
                    if len(content) > TIER3_MAX_CHARS:
                        compressed += " [...]"
                cursor.execute(
                    "UPDATE episodic_memory SET content = ?, tier = 3, degraded_at = ? WHERE id = ?",
                    (compressed, now.isoformat(), row["id"])
                )
                results["tier2_to_tier3"] += 1
            except Exception:
                pass

        self.conn.commit()
        return results

    def get_contaminated(self, limit: int = 50, min_importance: float = 0.0) -> List[Dict]:
        """Return potentially contaminated memories for review.

        Contaminated = veracity in ('inferred', 'tool', 'imported', 'unknown')
        — i.e., anything not explicitly stated by the user. Sorted by
        importance descending so the highest-stakes items surface first.

        Args:
            limit: Max memories to return
            min_importance: Only return memories with importance >= this
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, source, veracity, tier, importance,
                   created_at, degraded_at, session_id
            FROM episodic_memory
            WHERE veracity IN ('inferred', 'tool', 'imported', 'unknown')
              AND importance >= ?
            ORDER BY importance DESC, created_at DESC
            LIMIT ?
        """, (min_importance, limit))
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Consolidation / Sleep
    # ------------------------------------------------------------------
    def sleep(self, dry_run: bool = False) -> Dict:
        """
        Consolidate old working_memory for this session into episodic summaries.
        Uses a local lightweight LLM when available; falls back to aaak
        compression if the model is missing or inference fails.
        Returns summary of what was done.

        Note: this method intentionally remains session-scoped. Use
        sleep_all_sessions() for maintenance that consolidates eligible old
        working memories across inactive sessions.
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
            llm_succeeded = False
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
                        llm_succeeded = True

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
                        "llm_used": llm_succeeded
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

        # Run tiered degradation after consolidation
        degrade_result = self.degrade_episodic(dry_run=dry_run)

        return {
            "status": "dry_run" if dry_run else "consolidated",
            "items_consolidated": len(consolidated_ids),
            "summaries_created": summaries_created,
            "llm_used": llm_used_count,
            "method": method,
            "consolidated_ids": consolidated_ids,
            "degradation": degrade_result
        }

    def sleep_all_sessions(self, dry_run: bool = False) -> Dict:
        """
        Consolidate eligible old working memories across all sessions.

        This is the maintenance-oriented counterpart to sleep(), which remains
        scoped to self.session_id. It prevents inactive sessions from leaving
        old working_memory rows stranded after they pass the sleep cutoff.
        """
        cursor = self.conn.cursor()
        cutoff = (datetime.now() - timedelta(hours=WORKING_MEMORY_TTL_HOURS // 2)).isoformat()
        cursor.execute("""
            SELECT session_id, COUNT(*) AS eligible
            FROM working_memory
            WHERE timestamp < ?
            GROUP BY session_id
            ORDER BY MIN(timestamp) ASC
        """, (cutoff,))
        session_rows = cursor.fetchall()
        if not session_rows:
            return {
                "status": "no_op",
                "message": "No old working memories to consolidate",
                "sessions_scanned": 0,
                "sessions_consolidated": 0,
                "items_consolidated": 0,
                "summaries_created": 0,
                "llm_used": 0,
                "errors": 0,
                "session_results": [],
            }

        session_results = []
        sessions_consolidated = 0
        items_consolidated = 0
        summaries_created = 0
        llm_used = 0
        errors = []

        for row in session_rows:
            session_id = row["session_id"] if hasattr(row, "keys") else row[0]
            if session_id is None:
                session_id = "default"
            try:
                beam = self if session_id == self.session_id else BeamMemory(
                    session_id=session_id,
                    db_path=self.db_path,
                )
                result = beam.sleep(dry_run=dry_run)
                result = dict(result)
                result["session_id"] = session_id
                result["eligible"] = row["eligible"] if hasattr(row, "keys") else row[1]
                session_results.append(result)

                if result.get("status") in ("consolidated", "dry_run"):
                    sessions_consolidated += 1
                    items_consolidated += int(result.get("items_consolidated", 0) or 0)
                    summaries_created += int(result.get("summaries_created", 0) or 0)
                    llm_used += int(result.get("llm_used", 0) or 0)
            except Exception as exc:
                errors.append({"session_id": session_id, "error": repr(exc)})

        # Run tiered degradation after all-sessions consolidation
        degrade_result = self.degrade_episodic(dry_run=dry_run)

        return {
            "status": "dry_run" if dry_run else ("consolidated" if items_consolidated else "no_op"),
            "sessions_scanned": len(session_rows),
            "sessions_consolidated": sessions_consolidated,
            "items_consolidated": items_consolidated,
            "summaries_created": summaries_created,
            "llm_used": llm_used,
            "errors": len(errors),
            "error_details": errors,
            "session_results": session_results,
            "degradation": degrade_result
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
