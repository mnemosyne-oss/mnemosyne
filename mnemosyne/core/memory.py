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


def _default_data_dir() -> Path:
    """Return the current default data directory, honoring runtime env changes."""
    if os.environ.get("MNEMOSYNE_DATA_DIR"):
        return Path(os.environ["MNEMOSYNE_DATA_DIR"])
    return DEFAULT_DATA_DIR


def _default_db_path() -> Path:
    """Return the current default DB path, honoring runtime env changes."""
    return _default_data_dir() / "mnemosyne.db"


def _get_connection(db_path = None) -> sqlite3.Connection:
    """Get thread-local database connection"""
    path = Path(db_path) if db_path else _default_db_path()
    if not hasattr(_thread_local, 'conn') or _thread_local.conn is None or getattr(_thread_local, 'db_path', None) != str(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        _thread_local.conn = sqlite3.connect(str(path), check_same_thread=False)
        _thread_local.conn.row_factory = sqlite3.Row
        _thread_local.conn.execute("PRAGMA journal_mode=WAL")
        _thread_local.conn.execute("PRAGMA busy_timeout=5000")
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

    Supports memory bank isolation via the `bank` parameter.
    Each bank is a separate SQLite database for complete isolation.
    """

    def __init__(self, session_id: str = "default", db_path: Path = None, bank: str = None,
                 author_id: str = None, author_type: str = None,
                 channel_id: str = None):
        self.session_id = session_id
        self.bank = bank or "default"
        self.author_id = author_id
        self.author_type = author_type
        self.channel_id = channel_id or session_id  # default channel = session

        # Resolve database path based on bank
        if db_path:
            self.db_path = db_path
        elif bank and bank != "default":
            from mnemosyne.core.banks import BankManager
            self.db_path = BankManager().get_bank_db_path(bank)
        else:
            self.db_path = _default_db_path()

        self.conn = _get_connection(self.db_path)
        init_db(self.db_path)

        # Phase 8: Streaming + Patterns + Plugins (lazy init)
        self._stream = None
        self._compressor = None
        self._pattern_detector = None
        self._delta_sync = None
        self._plugin_manager = None

        # Create beam with streaming emitter wired
        self.beam = BeamMemory(session_id=session_id, db_path=self.db_path,
                               author_id=author_id, author_type=author_type,
                               channel_id=channel_id,
                               event_emitter=self._stream_emit)

    # ─── Phase 8: Streaming ─────────────────────────────────────────

    @property
    def stream(self):
        """Lazy-initialized memory event stream."""
        if self._stream is None:
            from mnemosyne.core.streaming import MemoryStream
            self._stream = MemoryStream()
        return self._stream

    def enable_streaming(self) -> "Mnemosyne":
        """Enable event streaming for this memory instance.

        Wires the stream into BeamMemory so all write operations emit events.
        Call once after construction to activate the streaming subsystem.
        """
        _ = self.stream  # Force init
        # Retroactively wire emitter into existing beam (handles lazy init case)
        if self.beam._event_emitter is None:
            self.beam._event_emitter = self._stream_emit
        return self

    def _stream_emit(self, event) -> None:
        """Callback passed to BeamMemory; routes events to the lazy-init stream."""
        if self._stream is not None:
            self._stream.emit(event)

    # ─── Phase 8: Compression ───────────────────────────────────────

    @property
    def compressor(self):
        """Lazy-initialized memory compressor."""
        if self._compressor is None:
            from mnemosyne.core.patterns import MemoryCompressor
            self._compressor = MemoryCompressor()
        return self._compressor

    def compress(self, content: str, method: str = "auto"):
        """Compress memory content. Returns (compressed, stats)."""
        return self.compressor.compress(content, method=method)

    def decompress(self, content: str, method: str = "dict") -> str:
        """Decompress memory content."""
        return self.compressor.decompress(content, method=method)

    def compress_memories(self, memories: list, method: str = "auto"):
        """Compress a batch of memories. Returns (compressed_memories, stats)."""
        return self.compressor.compress_batch(memories, method=method)

    # ─── Phase 8: Pattern Detection ─────────────────────────────────

    @property
    def patterns(self):
        """Lazy-initialized pattern detector."""
        if self._pattern_detector is None:
            from mnemosyne.core.patterns import PatternDetector
            self._pattern_detector = PatternDetector()
        return self._pattern_detector

    def detect_patterns(self, memories: list = None) -> list:
        """Detect patterns in memories. Uses all working+episodic if none provided."""
        if memories is None:
            memories = self.get_all_memories()
        return self.patterns.detect_all(memories)

    def summarize_patterns(self, memories: list = None) -> dict:
        """Generate a summary of detected patterns."""
        if memories is None:
            memories = self.get_all_memories()
        return self.patterns.summarize_patterns(memories)

    def get_all_memories(self) -> List[Dict]:
        """Return all working + episodic rows for pattern analysis.

        Scoped to the active session (and global memories), with the same
        validity filters that get_context() and recall() apply: invalidated
        and expired memories are excluded so retracted notes do not skew
        pattern detection.
        """
        now = datetime.now().isoformat()
        cursor = self.beam.conn.cursor()
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id, importance
            FROM working_memory
            WHERE (session_id = ? OR scope = 'global')
              AND (valid_until IS NULL OR valid_until > ?)
              AND superseded_by IS NULL
        """, (self.session_id, now))
        rows = [dict(row) for row in cursor.fetchall()]
        seen_ids = {r["id"] for r in rows}
        cursor.execute("""
            SELECT id, content, source, timestamp, session_id, importance
            FROM episodic_memory
            WHERE (session_id = ? OR scope = 'global')
              AND (valid_until IS NULL OR valid_until > ?)
              AND superseded_by IS NULL
        """, (self.session_id, now))
        for row in cursor.fetchall():
            if row["id"] not in seen_ids:
                rows.append(dict(row))
        return rows

    # ─── Phase 8: Delta Sync ──────────────────────────────────────

    @property
    def delta_sync(self):
        """Lazy-initialized delta sync."""
        if self._delta_sync is None:
            from mnemosyne.core.streaming import DeltaSync
            self._delta_sync = DeltaSync(self)
        return self._delta_sync

    def sync_to(self, peer_id: str, table: str = "working_memory") -> dict:
        """Compute delta for a peer. Returns {peer_id, table, delta, count}."""
        return self.delta_sync.sync_to(peer_id, table)

    def sync_from(self, peer_id: str, delta: list, table: str = "working_memory") -> dict:
        """Apply delta from a peer. Returns {peer_id, table, stats, checkpoint}."""
        return self.delta_sync.sync_from(peer_id, delta, table)

    # ─── Phase 8: Plugins ───────────────────────────────────────────

    @property
    def plugins(self):
        """Lazy-initialized plugin manager."""
        if self._plugin_manager is None:
            from mnemosyne.core.plugins import PluginManager
            self._plugin_manager = PluginManager()
        return self._plugin_manager

    @plugins.setter
    def plugins(self, manager):
        """Attach an external PluginManager."""
        self._plugin_manager = manager

    def remember(self, content: str, source: str = "conversation",
                 importance: float = 0.5, metadata: Dict = None,
                 valid_until: str = None, scope: str = "session",
                 extract_entities: bool = False,
                 extract: bool = False,
                 trust_tier: str = None) -> str:
        """
        Store a memory directly to SQLite.
        Writes to both BEAM working_memory and legacy memories table.

        Args:
            extract_entities: If True, extract entities from content and store
                in the AnnotationStore (kind='mentions') for fuzzy entity-aware
                recall. Default False. (Pre-E6 wrote to TripleStore; the storage
                target moved as part of E6 — see mnemosyne.core.annotations.)
            extract: If True, extract structured facts from content using LLM
                and store in the AnnotationStore (kind='fact'). Default False.
            trust_tier: Trust classification for prompt-injection defense.
                None = use beam default ('STATED'). 'EXTERNAL_WRITE' for MCP
                tool calls, 'IMPORTED' for bulk imports.
        """
        # BEAM write first (generates its own ID). Extract flags are passed
        # through so BeamMemory's canonical _extract_and_store_entities and
        # _extract_and_store_facts helpers run — these populate the `facts`
        # table that fact_recall() queries (the wrapper used to reimplement
        # only the triples half of extraction inline, leaving facts table
        # writes silently skipped — see C12.a).

        # Content sanitization: extract binary payloads to blob storage.
        # Applied here so the legacy memories table row also gets the
        # sanitized content (not just the BEAM working_memory row).
        from mnemosyne.core.content_sanitizer import sanitize_content as _sanitize
        sanitized_content, blob_meta = _sanitize(content)
        if blob_meta:
            metadata = (metadata or {}).copy()
            metadata["_blob"] = blob_meta

        _content = sanitized_content if blob_meta else content

        memory_id = self.beam.remember(
            _content, source=source,
            importance=importance, metadata=metadata,
            valid_until=valid_until, scope=scope,
            extract_entities=extract_entities, extract=extract,
            trust_tier=trust_tier,
        )
        timestamp = datetime.now().isoformat()

        # Legacy dual-write with same ID (INSERT OR REPLACE for dedup safety)
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO memories (id, content, source, timestamp, session_id, importance, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id, _content, source, timestamp, self.session_id,
            importance, json.dumps(metadata or {})
        ))

        # Legacy embedding store
        if _embeddings.available():
            vec = _embeddings.embed([_content])
            if vec is not None:
                cursor.execute("""
                    INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json, model)
                    VALUES (?, ?, ?)
                """, (memory_id, _embeddings.serialize(vec[0]), _embeddings._DEFAULT_MODEL))

        self.conn.commit()

        # The first BEAM write already inserted the working_memory row with
        # the correct memory_id (we used it for the legacy dual-write above)
        # and, because we passed extract_entities/extract through, BeamMemory
        # already ran _extract_and_store_entities / _extract_and_store_facts
        # to populate annotations (post-E6) and the facts table. A second
        # beam.remember call would only re-run the dedup branch and
        # _ingest_graph_and_veracity — duplicating gist/fact graph edges and
        # bumping mention_count for what is a single user-level remember. So
        # this function returns directly after the legacy write.
        return memory_id

    def recall(self, query: str, top_k: int = 5, *,
               from_date: Optional[str] = None, to_date: Optional[str] = None,
               source: Optional[str] = None, topic: Optional[str] = None,
               author_id: Optional[str] = None,
               author_type: Optional[str] = None,
               channel_id: Optional[str] = None,
               temporal_weight: float = 0.0,
               query_time: Optional[Any] = None,
               temporal_halflife: Optional[float] = None,
               vec_weight: float = None,
               fts_weight: float = None,
               importance_weight: float = None) -> List[Dict]:
        """
        Search memories with hybrid relevance scoring.
        Uses BEAM episodic + working memory retrieval (sqlite-vec + FTS5).
        Supports temporal filtering: from_date, to_date, source, topic.
        Supports multi-agent identity filtering: author_id, author_type, channel_id.
        Supports temporal scoring: temporal_weight, query_time, temporal_halflife.
        Supports scoring weight overrides: vec_weight, fts_weight, importance_weight.
        """
        return self.beam.recall(query, top_k=top_k,
                                from_date=from_date, to_date=to_date,
                                source=source, topic=topic,
                                author_id=author_id, author_type=author_type,
                                channel_id=channel_id,
                                temporal_weight=temporal_weight,
                                query_time=query_time,
                                temporal_halflife=temporal_halflife,
                                vec_weight=vec_weight,
                                fts_weight=fts_weight,
                                importance_weight=importance_weight)

    def _emit_wrapper(self, event_type: str, memory_id: str, **kwargs) -> None:
        """Emit a streaming event through the Mnemosyne wrapper layer."""
        if self._stream is not None:
            try:
                from mnemosyne.core.streaming import MemoryEvent, EventType
                evt = EventType[event_type]
                event = MemoryEvent(
                    event_type=evt,
                    memory_id=memory_id,
                    session_id=self.session_id,
                    **kwargs,
                )
                self._stream.emit(event)
            except Exception:
                pass

    def get_context(self, limit: int = 10) -> List[Dict]:
        """
        Get recent memories from current session for context injection.
        Pulls from BEAM working_memory.
        """
        return self.beam.get_context(limit=limit)

    def get_stats(self, author_id: str = None, author_type: str = None,
                  channel_id: str = None) -> Dict:
        """Get memory system statistics (legacy + BEAM). Supports multi-agent identity filters."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memories")
        total_legacy = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM memories")
        sessions = cursor.fetchone()[0]

        cursor.execute("SELECT source, COUNT(*) FROM memories GROUP BY source")
        sources = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("SELECT timestamp FROM memories ORDER BY timestamp DESC LIMIT 1")
        last = cursor.fetchone()

        beam_wm = self.beam.get_working_stats(author_id=author_id, author_type=author_type,
                                               channel_id=channel_id)
        beam_ep = self.beam.get_episodic_stats(author_id=author_id, author_type=author_type,
                                                channel_id=channel_id)

        # Triples count — table is created lazily by TripleStore.init_triples;
        # if it does not exist yet (no triple has ever been written), report 0.
        # Narrow the suppression to the missing-table case so DB locks, I/O
        # errors, and corruption are not silently turned into "0 triples".
        triple_total = 0
        try:
            cursor.execute("SELECT COUNT(*) FROM triples")
            triple_total = cursor.fetchone()[0]
        except sqlite3.OperationalError as e:
            if "no such table" not in str(e).lower():
                raise

        # Bank list — scoped to the same data dir as this Mnemosyne instance so
        # a per-bank or per-tmp-dir caller does not get bank names from the
        # default ~/.hermes tree. Banks live at <data_dir>/banks/, where
        # data_dir is the parent of self.db_path.
        try:
            from mnemosyne.core.banks import BankManager
            banks = BankManager(data_dir=Path(self.db_path).parent).list_banks()
        except Exception:
            banks = ["default"]

        return {
            "total_memories": total_legacy,
            "total_sessions": sessions,
            "sources": sources,
            "last_memory": last[0] if last else None,
            "database": str(self.db_path),
            "mode": "beam",
            "banks": banks,
            "beam": {
                "working_memory": beam_wm,
                "episodic_memory": beam_ep,
                "triples": {"total": triple_total},
            }
        }

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID from legacy table and working_memory."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ? AND session_id = ?",
                      (memory_id, self.session_id))
        self.conn.commit()
        result = self.beam.forget_working(memory_id)
        self._emit_wrapper("MEMORY_INVALIDATED", memory_id)
        return result

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

        self._emit_wrapper("MEMORY_UPDATED", memory_id, content=content, importance=importance)
        return cursor.rowcount > 0

    def invalidate(self, memory_id: str, replacement_id: str = None) -> bool:
        """Mark a memory as expired or superseded. Delegates to BEAM."""
        result = self.beam.invalidate(memory_id, replacement_id=replacement_id)
        self._emit_wrapper("MEMORY_INVALIDATED", memory_id, replacement_id=replacement_id)
        return result

    # ------------------------------------------------------------------
    # BEAM-specific public methods
    # ------------------------------------------------------------------
    def sleep(self, dry_run: bool = False) -> Dict:
        """Run consolidation sleep cycle for the current session."""
        return self.beam.sleep(dry_run=dry_run)

    def sleep_all_sessions(self, dry_run: bool = False) -> Dict:
        """Run consolidation sleep cycle across all sessions with eligible old working memories."""
        return self.beam.sleep_all_sessions(dry_run=dry_run)

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
        Export all Mnemosyne data (legacy + BEAM + triples + annotations) to
        a JSON file. Returns export metadata.

        Schema version 1.1 (post-E6) adds an `annotations` section alongside
        the existing `triples` section. Imports of 1.0 backups still work —
        they simply restore zero annotations, and the auto-migrate hook will
        relocate any annotation-flavored triples rows on next BeamMemory init.
        """
        from mnemosyne.core.triples import TripleStore
        from mnemosyne.core.annotations import AnnotationStore
        import json as _json

        export = {
            "mnemosyne_export": {
                "version": "1.1",
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

        # Triples (current-truth temporal facts; post-E6 scope)
        triples = TripleStore(db_path=self.db_path)
        export["triples"] = triples.export_all()

        # Annotations (post-E6: multi-valued mentions, facts, occurred_on,
        # has_source). Pre-E6 backups won't have this key — the import path
        # handles that gracefully.
        annotations = AnnotationStore(db_path=self.db_path)
        export["annotations"] = annotations.export_all()

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
            "annotations_count": len(export["annotations"]),
        }

    def import_from_file(self, input_path: str, force: bool = False) -> Dict:
        """
        Import Mnemosyne data from a JSON file produced by export_to_file().
        Idempotent by default: skips existing records.
        Set force=True to overwrite.
        Returns import statistics.

        Accepts both schema version 1.0 (pre-E6) and 1.1 (post-E6). When a
        1.0 backup is imported, the `annotations` section is treated as empty
        and any annotation-flavored rows in the imported `triples` will be
        relocated by the auto-migrate hook on the next BeamMemory init.
        """
        from mnemosyne.core.triples import TripleStore
        from mnemosyne.core.annotations import AnnotationStore
        import json as _json

        with open(input_path, "r", encoding="utf-8") as f:
            data = _json.load(f)

        if not isinstance(data, dict):
            raise ValueError("Import file must contain a Mnemosyne export object")

        # Validate — accept the two known schema versions.
        meta = data.get("mnemosyne_export", {})
        version = meta.get("version")
        if version not in ("1.0", "1.1"):
            raise ValueError(f"Unsupported export version: {version}")

        stats = {"beam": {}, "legacy": {}, "triples": {}, "annotations": {}}

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

        # Triples (current-truth temporal facts)
        triples = TripleStore(db_path=self.db_path)
        t_stats = triples.import_all(data.get("triples", []), force=force)
        stats["triples"] = t_stats

        # Annotations (post-E6 schema 1.1; absent from 1.0 backups)
        annotations = AnnotationStore(db_path=self.db_path)
        a_stats = annotations.import_all(data.get("annotations", []), force=force)
        stats["annotations"] = a_stats

        return stats


# Global instance for module-level convenience functions
_default_instance = None
_default_bank = "default"


def _get_default(bank: str = None):
    """Get or create the default Mnemosyne instance. Supports bank switching."""
    global _default_instance, _default_bank
    target_bank = bank or _default_bank or "default"
    if _default_instance is None or _default_instance.bank != target_bank:
        _default_bank = target_bank
        _default_instance = Mnemosyne(bank=target_bank)
    return _default_instance


def set_bank(bank: str):
    """
    Switch the global default memory bank.
    All subsequent module-level calls (remember, recall, etc.) will use this bank.
    """
    global _default_bank, _default_instance
    _default_bank = bank
    _default_instance = None  # Force re-creation on next access


def get_bank() -> str:
    """Get the current default bank name."""
    return _default_bank or "default"


# Module-level convenience functions
def remember(content: str, source: str = "conversation",
             importance: float = 0.5, metadata: Dict = None,
             scope: str = "session", valid_until: str = None,
             extract_entities: bool = False,
             extract: bool = False, bank: str = None,
             trust_tier: str = None) -> str:
    """Store a memory using the global instance"""
    return _get_default(bank).remember(content, source, importance, metadata,
                                       scope=scope, valid_until=valid_until,
                                       extract_entities=extract_entities,
                                       extract=extract, trust_tier=trust_tier)


def recall(query: str, top_k: int = 5, *,
           from_date: Optional[str] = None, to_date: Optional[str] = None,
           source: Optional[str] = None, topic: Optional[str] = None,
           temporal_weight: float = 0.0,
           query_time: Optional[Any] = None,
           temporal_halflife: Optional[float] = None,
           vec_weight: float = None,
           fts_weight: float = None,
           importance_weight: float = None,
           bank: str = None) -> List[Dict]:
    """Search memories using the global instance with temporal filtering and scoring"""
    return _get_default(bank).recall(query, top_k,
                                     from_date=from_date, to_date=to_date,
                                     source=source, topic=topic,
                                     temporal_weight=temporal_weight,
                                     query_time=query_time,
                                     temporal_halflife=temporal_halflife,
                                     vec_weight=vec_weight,
                                     fts_weight=fts_weight,
                                     importance_weight=importance_weight)


def get_context(limit: int = 10, bank: str = None) -> List[Dict]:
    """Get session context using the global instance"""
    return _get_default(bank).get_context(limit)


def get_stats(bank: str = None) -> Dict:
    """Get stats using the global instance"""
    return _get_default(bank).get_stats()


def forget(memory_id: str, bank: str = None) -> bool:
    """Delete memory using the global instance"""
    return _get_default(bank).forget(memory_id)


def update(memory_id: str, content: str = None, importance: float = None, bank: str = None) -> bool:
    """Update memory using the global instance"""
    return _get_default(bank).update(memory_id, content, importance)


def sleep(dry_run: bool = False, bank: str = None) -> Dict:
    """Run consolidation sleep cycle for the global instance's current session"""
    return _get_default(bank).sleep(dry_run=dry_run)


def sleep_all_sessions(dry_run: bool = False, bank: str = None) -> Dict:
    """Run consolidation sleep cycle across all sessions using the global instance"""
    return _get_default(bank).sleep_all_sessions(dry_run=dry_run)


def scratchpad_write(content: str, bank: str = None) -> str:
    """Write to scratchpad using the global instance"""
    return _get_default(bank).scratchpad_write(content)


def scratchpad_read(bank: str = None) -> List[Dict]:
    """Read scratchpad using the global instance"""
    return _get_default(bank).scratchpad_read()


def scratchpad_clear():
    """Clear scratchpad using the global instance"""
    return _get_default().scratchpad_clear()
