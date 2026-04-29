"""
Mnemosyne Core - Direct SQLite Integration
No HTTP, no server, just pure Python + SQLite

This is the heart of Mnemosyne — a zero-dependency memory system
that delivers sub-millisecond performance through direct SQLite access.

Now upgraded with BEAM architecture:
- working_memory: hot context auto-injected into prompts
- episodic_memory: long-term storage with sqlite-vec + FTS5
- scratchpad: temporary agent reasoning workspace
"""

import sqlite3
import json
import hashlib
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path

from mnemosyne.core import embeddings as _embeddings
from mnemosyne.core.beam import BeamMemory, init_beam, _get_connection as _beam_get_connection

# Single shared connection per thread (legacy path)
_thread_local = threading.local()

# Default data directory
# NOTE: On Fly.io and ephemeral VMs, only ~/.hermes is persisted.
# This MUST match beam.py's DEFAULT_DATA_DIR to avoid split-brain.
DEFAULT_DATA_DIR = Path.home() / ".hermes" / "mnemosyne" / "data"
DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"

# Allow override via environment
import os
if os.environ.get("MNEMOSYNE_DATA_DIR"):
    DEFAULT_DATA_DIR = Path(os.environ.get("MNEMOSYNE_DATA_DIR"))
    DEFAULT_DB_PATH = DEFAULT_DATA_DIR / "mnemosyne.db"


def _get_connection(db_path = None) -> sqlite3.Connection:
    """Get thread-local database connection"""
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not hasattr(_thread_local, 'conn') or _thread_local.conn is None or getattr(_thread_local, 'db_path', None) != str(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        _thread_local.conn = sqlite3.connect(str(path), check_same_thread=False)
        _thread_local.conn.row_factory = sqlite3.Row
        _thread_local.db_path = str(path)
    return _thread_local.conn


def init_db(db_path: Path = None):
    """Initialize legacy database schema + BEAM schema"""
    conn = _get_connection(db_path)
    cursor = conn.cursor()

    # Legacy memories table (kept for backward compatibility)
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_source ON memories(source)")

    # Legacy embeddings table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id TEXT PRIMARY KEY,
            embedding_json TEXT NOT NULL,
            model TEXT DEFAULT 'bge-small-en-v1.5',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)

    conn.commit()

    # Initialize BEAM schema on same DB
    init_beam(db_path)


# Initialize on module load
init_db()


def generate_id(content: str) -> str:
    """Generate unique ID for memory"""
    return hashlib.sha256(f"{content}{datetime.now().isoformat()}".encode()).hexdigest()[:16]


class Mnemosyne:
    """
    Native memory interface - no HTTP, direct SQLite.
    Now backed by BEAM architecture for scalable retrieval.
    """

    def __init__(self, session_id: str = "default", db_path: Path = None):
        self.session_id = session_id
        self.db_path = db_path or DEFAULT_DB_PATH
        self.conn = _get_connection(self.db_path)
        init_db(self.db_path)
        self.beam = BeamMemory(session_id=session_id, db_path=db_path)

    def remember(self, content: str, source: str = "conversation",
                 importance: float = 0.5, metadata: Dict = None,
                 valid_until: str = None, scope: str = "session",
                 extract_entities: bool = False,
                 extract: bool = False) -> str:
        """
        Store a memory directly to SQLite.
        Writes to both BEAM working_memory and legacy memories table.

        Args:
            extract_entities: If True, extract entities from content and store
                as triples for fuzzy entity-aware recall. Default False.
            extract: If True, extract structured facts from content using LLM
                and store as triples. Default False.
        """
        # BEAM write first (generates its own ID)
        memory_id = self.beam.remember(content, source=source, importance=importance, metadata=metadata,
                           valid_until=valid_until, scope=scope)
        timestamp = datetime.now().isoformat()

        # Legacy dual-write with same ID
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO memories (id, content, source, timestamp, session_id, importance, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, content, source, timestamp, self.session_id,
            importance, json.dumps(metadata or {})
        ))

        # Legacy embedding store
        if _embeddings.available():
            vec = _embeddings.embed([content])
            if vec is not None:
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model)
                    VALUES (?, ?, ?)
                """, (memory_id, _embeddings.serialize(vec[0]), _embeddings._DEFAULT_MODEL))

        self.conn.commit()

        # Entity extraction (best-effort, never fails the memory write)
        if extract_entities:
            try:
                from mnemosyne.core.entities import extract_entities_regex
                from mnemosyne.core.triples import TripleStore
                entities = extract_entities_regex(content)
                if entities:
                    triples = TripleStore(db_path=self.db_path)
                    for entity in entities:
                        triples.add(
                            subject=memory_id,
                            predicate="mentions",
                            object=entity,
                            source=source,
                            confidence=0.8
                        )
            except Exception:
                pass  # Entity extraction is best-effort

        # Structured fact extraction (best-effort, never fails the memory write)
        if extract:
            try:
                from mnemosyne.core.extraction import extract_facts_safe
                from mnemosyne.core.triples import TripleStore
                facts = extract_facts_safe(content)
                if facts:
                    triples = TripleStore(db_path=self.db_path)
                    triples.add_facts(memory_id, facts, source=source, confidence=0.7)
            except Exception:
                pass  # Fact extraction is best-effort

        return memory_id

    def recall(self, query: str, top_k: int = 5, *, from_date: Optional[str] = None, to_date: Optional[str] = None, source: Optional[str] = None, topic: Optional[str] = None) -> List[Dict]:
        """
        Search memories with hybrid relevance scoring.
        Uses BEAM episodic + working memory retrieval (sqlite-vec + FTS5).
        Supports temporal filtering: from_date, to_date, source, topic.
        """
        return self.beam.recall(query, top_k=top_k, from_date=from_date, to_date=to_date, source=source, topic=topic)

    def get_context(self, limit: int = 10) -> List[Dict]:
        """
        Get recent memories from current session for context injection.
        Pulls from BEAM working_memory.
        """
        return self.beam.get_context(limit=limit)

    def get_stats(self) -> Dict:
        """Get memory system statistics (legacy + BEAM)."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total_legacy = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM memories")
        sessions = cursor.fetchone()[0]

        cursor.execute("SELECT source, COUNT(*) FROM memories GROUP BY source")
        sources = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT timestamp FROM memories ORDER BY timestamp DESC LIMIT 1")
        last = cursor.fetchone()

        beam_wm = self.beam.get_working_stats()
        beam_ep = self.beam.get_episodic_stats()

        return {
            "total_memories": total_legacy,
            "total_sessions": sessions,
            "sources": sources,
            "last_memory": last[0] if last else None,
            "database": str(self.db_path),
            "mode": "beam",
            "beam": {
                "working_memory": beam_wm,
                "episodic_memory": beam_ep
            }
        }

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID from legacy table and working_memory."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ? AND session_id = ?",
                      (memory_id, self.session_id))
        self.conn.commit()
        self.beam.forget_working(memory_id)
        return cursor.rowcount > 0

    def update(self, memory_id: str, content: str = None,
               importance: float = None) -> bool:
        """Update an existing memory in legacy table and BEAM."""
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

        # Sync BEAM working_memory
        self.beam.update_working(memory_id, content=content, importance=importance)

        return cursor.rowcount > 0

    def invalidate(self, memory_id: str, replacement_id: str = None) -> bool:
        """Mark a memory as expired or superseded. Delegates to BEAM."""
        return self.beam.invalidate(memory_id, replacement_id=replacement_id)

    # ------------------------------------------------------------------
    # BEAM-specific public methods
    # ------------------------------------------------------------------
    def sleep(self, dry_run: bool = False) -> Dict:
        """Run consolidation sleep cycle."""
        return self.beam.sleep(dry_run=dry_run)

    def scratchpad_write(self, content: str) -> str:
        """Write to scratchpad."""
        return self.beam.scratchpad_write(content)

    def scratchpad_read(self) -> List[Dict]:
        """Read scratchpad entries."""
        return self.beam.scratchpad_read()

    def scratchpad_clear(self):
        """Clear scratchpad."""
        self.beam.scratchpad_clear()

    def consolidation_log(self, limit: int = 10) -> List[Dict]:
        """Get consolidation history."""
        return self.beam.get_consolidation_log(limit=limit)

    def export_to_file(self, output_path: str) -> Dict:
        """
        Export all Mnemosyne data (legacy + BEAM + triples) to a JSON file.
        Returns export metadata.
        """
        from mnemosyne.core.triples import TripleStore
        import json as _json

        export = {
            "mnemosyne_export": {
                "version": "1.0",
                "export_date": datetime.now().isoformat(),
                "source_db": str(self.db_path),
            }
        }

        # BEAM data
        beam_data = self.beam.export_to_dict()
        export["working_memory"] = beam_data.get("working_memory", [])
        export["episodic_memory"] = beam_data.get("episodic_memory", [])
        export["episodic_embeddings"] = beam_data.get("episodic_embeddings", [])
        export["scratchpad"] = beam_data.get("scratchpad", [])
        export["consolidation_log"] = beam_data.get("consolidation_log", [])

        # Legacy memories
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id, importance,
                   metadata_json, created_at
            FROM memories
            ORDER BY session_id, timestamp
        """)
        export["legacy_memories"] = [dict(row) for row in cursor.fetchall()]

        # Legacy embeddings
        cursor.execute("""
            SELECT memory_id, embedding_json, model, created_at
            FROM memory_embeddings
            ORDER BY memory_id
        """)
        export["legacy_embeddings"] = [dict(row) for row in cursor.fetchall()]

        # Triples
        triples = TripleStore(db_path=self.db_path)
        export["triples"] = triples.export_all()

        with open(output_path, "w", encoding="utf-8") as f:
            _json.dump(export, f, indent=2, ensure_ascii=False, default=str)

        return {
            "status": "exported",
            "path": output_path,
            "working_memory_count": len(export["working_memory"]),
            "episodic_memory_count": len(export["episodic_memory"]),
            "scratchpad_count": len(export["scratchpad"]),
            "legacy_memories_count": len(export["legacy_memories"]),
            "triples_count": len(export["triples"]),
        }

    def import_from_file(self, input_path: str, force: bool = False) -> Dict:
        """
        Import Mnemosyne data from a JSON file produced by export_to_file().
        Idempotent by default: skips existing records.
        Set force=True to overwrite.
        Returns import statistics.
        """
        from mnemosyne.core.triples import TripleStore
        import json as _json

        with open(input_path, "r", encoding="utf-8") as f:
            data = _json.load(f)

        # Validate
        meta = data.get("mnemosyne_export", {})
        if meta.get("version") != "1.0":
            raise ValueError(f"Unsupported export version: {meta.get('version')}")

        stats = {"beam": {}, "legacy": {}, "triples": {}}

        # BEAM import
        beam_stats = self.beam.import_from_dict(data, force=force)
        stats["beam"] = beam_stats

        # Legacy memories
        l_stats = {"inserted": 0, "skipped": 0, "overwritten": 0}
        cursor = self.conn.cursor()
        for item in data.get("legacy_memories", []):
            mid = item.get("id")
            cursor.execute("SELECT 1 FROM memories WHERE id = ?", (mid,))
            exists = cursor.fetchone() is not None
            if exists and not force:
                l_stats["skipped"] += 1
                continue
            if exists and force:
                cursor.execute("DELETE FROM memories WHERE id = ?", (mid,))
                l_stats["overwritten"] += 1
            else:
                l_stats["inserted"] += 1
            cursor.execute("""
                INSERT INTO memories (id, content, source, timestamp, session_id,
                                      importance, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mid, item.get("content"), item.get("source"), item.get("timestamp"),
                item.get("session_id", "default"), item.get("importance", 0.5),
                item.get("metadata_json", "{}"), item.get("created_at")
            ))
        self.conn.commit()

        # Legacy embeddings
        for item in data.get("legacy_embeddings", []):
            mid = item.get("memory_id")
            cursor.execute("SELECT 1 FROM memory_embeddings WHERE memory_id = ?", (mid,))
            exists = cursor.fetchone() is not None
            if exists and not force:
                continue
            if exists and force:
                cursor.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (mid,))
            cursor.execute("""
                INSERT INTO memory_embeddings (memory_id, embedding_json, model, created_at)
                VALUES (?, ?, ?, ?)
            """, (mid, item.get("embedding_json"), item.get("model", "bge-small-en-v1.5"), item.get("created_at")))
        self.conn.commit()
        stats["legacy"] = l_stats

        # Triples
        triples = TripleStore(db_path=self.db_path)
        t_stats = triples.import_all(data.get("triples", []), force=force)
        stats["triples"] = t_stats

        return stats


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
             importance: float = 0.5, metadata: Dict = None,
             scope: str = "session", valid_until: str = None,
             extract_entities: bool = False,
             extract: bool = False) -> str:
    """Store a memory using the global instance"""
    return _get_default().remember(content, source, importance, metadata,
                                   scope=scope, valid_until=valid_until,
                                   extract_entities=extract_entities,
                                   extract=extract)


def recall(query: str, top_k: int = 5, *, from_date: Optional[str] = None, to_date: Optional[str] = None, source: Optional[str] = None, topic: Optional[str] = None) -> List[Dict]:
    """Search memories using the global instance with temporal filtering"""
    return _get_default().recall(query, top_k, from_date=from_date, to_date=to_date, source=source, topic=topic)


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


def sleep(dry_run: bool = False) -> Dict:
    """Run consolidation sleep cycle using the global instance"""
    return _get_default().sleep(dry_run=dry_run)


def scratchpad_write(content: str) -> str:
    """Write to scratchpad using the global instance"""
    return _get_default().scratchpad_write(content)


def scratchpad_read() -> List[Dict]:
    """Read scratchpad using the global instance"""
    return _get_default().scratchpad_read()


def scratchpad_clear():
    """Clear scratchpad using the global instance"""
    return _get_default().scratchpad_clear()
