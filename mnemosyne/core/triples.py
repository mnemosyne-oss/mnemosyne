"""
Mnemosyne Temporal Triples
Time-aware knowledge graph on top of SQLite.
Tracks when facts were true, enabling contradiction detection and historical queries.

Post-E6 scope
-------------
TripleStore is the canonical home for **single-current-truth temporal facts**.
Its `add()` auto-invalidates prior rows with the same `(subject, predicate)`
on every write — correct for facts like "user prefers X" later superseded
by "user prefers Y", wrong for multi-valued annotations where many objects
should coexist for the same `(subject, predicate)` key.

Multi-valued annotation use cases (`(memory_id, "mentions", entity)`,
`(memory_id, "fact", text)`, `(memory_id, "occurred_on", date)`, etc.)
have moved to `mnemosyne.core.annotations.AnnotationStore`, which is
append-only and preserves all values. See the E6 migration:

- `mnemosyne/core/annotations.py` — the new append-only store
- `scripts/migrate_triplestore_split.py` — moves existing annotation rows
- `.hermes/ledger/memory-contract.md` (E6) — ledger row + audit trail

Legacy callers of `TripleStore.add_facts()` continue to work — the method
now routes writes to `AnnotationStore` and emits a DeprecationWarning so
new code uses the right store directly.
"""

import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

LEGACY_DATA_DIR = Path.home() / ".hermes" / "mnemosyne" / "data"
DEFAULT_DATA_DIR = Path(os.environ.get("MNEMOSYNE_DATA_DIR", LEGACY_DATA_DIR))
DEFAULT_DB = DEFAULT_DATA_DIR / "triples.db"
LEGACY_DB = LEGACY_DATA_DIR / "triples.db"


def _copy_legacy_db(source: Path, destination: Path) -> None:
    """Copy a SQLite DB using SQLite's backup API for a consistent snapshot."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        source_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
        try:
            dest_conn = sqlite3.connect(str(temp_path))
            try:
                source_conn.backup(dest_conn)
            finally:
                dest_conn.close()
        finally:
            source_conn.close()

        if not destination.exists():
            temp_path.replace(destination)
        else:
            temp_path.unlink(missing_ok=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _resolve_default_db() -> Path:
    """Return the default triples DB, copying legacy data into place if needed."""
    if DEFAULT_DATA_DIR != LEGACY_DATA_DIR and not DEFAULT_DB.exists() and LEGACY_DB.exists():
        _copy_legacy_db(LEGACY_DB, DEFAULT_DB)
    return DEFAULT_DB


def _get_conn(db_path = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _resolve_default_db()
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
    Temporal knowledge graph for Mnemosyne — single-current-truth semantics.

    `add()` auto-invalidates prior rows with the same `(subject, predicate)`.
    This is the right shape for facts that change over time, where only one
    value should be "currently true" at any moment:

        >>> kg = TripleStore()
        >>> kg.add("Maya", "assigned_to", "auth-migration", valid_from="2026-01-15")
        >>> kg.add("Maya", "assigned_to", "billing", valid_from="2026-03-01")
        >>> kg.query("Maya")                 # → "billing" (current)
        >>> kg.query("Maya", as_of="2026-02-01")  # → "auth-migration" (historical)

    Do NOT use TripleStore for multi-valued annotations like entity mentions
    or extracted facts on a single memory — those belong in
    `mnemosyne.core.annotations.AnnotationStore`, which is append-only:

        >>> from mnemosyne.core.annotations import AnnotationStore
        >>> ann = AnnotationStore()
        >>> ann.add("mem-1", "mentions", "Alice")
        >>> ann.add("mem-1", "mentions", "Bob")  # both preserved
    """
    
    def __init__(self, db_path: Path = None):
        self.db_path = Path(db_path) if db_path else _resolve_default_db()
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
        [DEPRECATED post-E6] Use AnnotationStore.add_many(memory_id, "fact", facts).

        Multi-fact storage is an annotation use case — multiple values per
        `(memory_id, "fact")` key should coexist. The pre-E6 implementation
        called `TripleStore.add()` per fact, which silently invalidated each
        prior fact on the next write because the invalidation key is
        `(subject, predicate)` regardless of object.

        Post-E6, this shim routes writes to `AnnotationStore` so external
        callers' facts land in the table the new recall path reads from
        (`_find_memories_by_fact`). Without this redirect, deprecation-period
        callers would get a successful return code but their facts would be
        invisible to `Mnemosyne.recall()` until the next BeamMemory init
        auto-migrated them out of the triples table — a real silent
        behavior change. Routing through AnnotationStore makes the shim
        compatibility-correct.

        Args:
            memory_id: The subject memory ID
            facts: List of fact strings to store
            source: Source identifier
            confidence: Confidence score for extracted facts (default 0.7)

        Returns:
            Number of facts stored (matches legacy filtering: drops empty
            and shorter-than-10-char entries). With INSERT OR IGNORE on the
            UNIQUE(memory_id, kind, value) index, duplicate facts are
            silently de-duped — the count reflects facts kept after both
            length filtering and uniqueness.
        """
        import warnings
        warnings.warn(
            "TripleStore.add_facts is deprecated post-E6. Use "
            "AnnotationStore.add_many(memory_id, 'fact', facts) directly. "
            "This shim routes writes to AnnotationStore so the data lands "
            "where the post-E6 recall path looks for it; it will be "
            "removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )
        from mnemosyne.core.annotations import AnnotationStore, filter_facts
        kept = filter_facts(facts)
        if not kept:
            return 0
        store = AnnotationStore(db_path=self.db_path)
        store.add_many(memory_id, "fact", kept, source=source, confidence=confidence)
        return len(kept)

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
        """Import triples from a list of dictionaries.

        Default behavior (``force=False``):
          - **No id collision** in the destination: insert with the
            imported ``id``. Counted in ``stats["inserted"]``.
          - **Id collision + identical content**: skip (legitimate
            round-trip idempotency -- re-importing the same export
            stays a no-op). Counted in ``stats["skipped"]``.
          - **Id collision + DIFFERENT content**: insert with a fresh
            auto-assigned id, preserving the imported row. Counted in
            ``stats["imported_renumbered"]``. Pre-C28 this case was
            silently bucketed into ``stats["skipped"]`` and the row
            was discarded -- backup-restore silently lost data
            whenever the destination already had a row at the same
            autoincrement id slot.
          - **No id supplied** (``id`` missing or ``None``): insert
            with a fresh auto-assigned id. Counted in
            ``stats["inserted"]`` (no renumbering needed since the
            caller didn't ask for a specific id).

        ``force=True`` keeps the explicit-overwrite contract: on id
        collision the existing row is replaced regardless of content.

        Content comparison uses the full set of stored columns
        (subject / predicate / object / valid_from / valid_until /
        source / confidence / created_at) so a row that differs in
        any field is treated as new.

        Returns import statistics with keys ``inserted``, ``skipped``,
        ``overwritten``, ``imported_renumbered``. Sum of values equals
        ``len(triples)``.
        """
        stats = {"inserted": 0, "skipped": 0, "overwritten": 0,
                 "imported_renumbered": 0}
        cursor = self.conn.cursor()

        # Fields compared to decide identical-content vs different-content
        # on an id collision. Excludes ``id`` itself.
        _CONTENT_FIELDS = ("subject", "predicate", "object", "valid_from",
                           "valid_until", "source", "confidence",
                           "created_at")
        # Defaults applied at INSERT time -- mirror these when normalizing
        # imported items for content comparison, so a partial dict round-
        # trips idempotently (codex review #2).
        _INSERT_DEFAULTS = {"source": "imported", "confidence": 1.0}

        def _normalized(item):
            return {
                f: item.get(f) if item.get(f) is not None else _INSERT_DEFAULTS.get(f)
                for f in _CONTENT_FIELDS
            }

        # Detect within-batch duplicate ids before touching the DB. Two
        # rows with the same explicit id in a single import is malformed
        # input -- pre-fix this produced an IntegrityError mid-stream
        # that left a partially-committed transaction (codex review #5).
        seen_ids = set()
        for item in triples:
            tid = item.get("id")
            if tid is not None:
                if tid in seen_ids:
                    raise ValueError(
                        f"import_all: duplicate id {tid!r} in the imported "
                        f"batch. Deduplicate the input before calling."
                    )
                seen_ids.add(tid)

        # BEGIN IMMEDIATE acquires the write lock now, so the snapshot
        # SELECT and the subsequent INSERTs see a consistent view
        # (codex review #6). Without this, Python sqlite3's implicit
        # transaction starts on the first DML, leaving SELECT in
        # autocommit -- another writer could change state between
        # snapshot and insert.
        cursor.execute("BEGIN IMMEDIATE")
        try:
            cursor.execute(
                "SELECT id, subject, predicate, object, valid_from, valid_until, "
                "source, confidence, created_at FROM triples"
            )
            existing_snapshot = {
                row[0]: dict(zip(_CONTENT_FIELDS, row[1:]))
                for row in cursor.fetchall()
            }

            def _insert_with_id(item, tid):
                cursor.execute("""
                    INSERT INTO triples (id, subject, predicate, object, valid_from,
                                         valid_until, source, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tid, item.get("subject"), item.get("predicate"),
                    item.get("object"), item.get("valid_from"),
                    item.get("valid_until"), item.get("source", "imported"),
                    item.get("confidence", 1.0), item.get("created_at"),
                ))

            def _insert_without_id(item):
                cursor.execute("""
                    INSERT INTO triples (subject, predicate, object, valid_from,
                                         valid_until, source, confidence, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item.get("subject"), item.get("predicate"),
                    item.get("object"), item.get("valid_from"),
                    item.get("valid_until"), item.get("source", "imported"),
                    item.get("confidence", 1.0), item.get("created_at"),
                ))

            # Three-bucket split to avoid cascading collisions (codex
            # review #1). Single-snapshot two-phase still has a hole:
            # if Phase 1 contains a no-id row processed before a later
            # explicit-id row, the no-id row's auto-assigned id can
            # claim that later explicit id. Order:
            #   1. Explicit non-colliding ids -- claim specific slots.
            #   2. No-id rows -- auto-assign past Phase 1's high water.
            #   3. Collisions -- renumber inserts land past Phase 2's max.
            explicit_no_collision = []
            no_id = []
            collisions = []
            for item in triples:
                tid = item.get("id")
                if tid is None:
                    no_id.append(item)
                elif tid in existing_snapshot:
                    collisions.append(item)
                else:
                    explicit_no_collision.append(item)

            for item in explicit_no_collision:
                _insert_with_id(item, item["id"])
                stats["inserted"] += 1
            for item in no_id:
                _insert_without_id(item)
                stats["inserted"] += 1

            for item in collisions:
                tid = item["id"]
                existing_content = existing_snapshot[tid]
                if force:
                    cursor.execute("DELETE FROM triples WHERE id = ?", (tid,))
                    _insert_with_id(item, tid)
                    stats["overwritten"] += 1
                    continue
                if _normalized(item) == existing_content:
                    stats["skipped"] += 1
                    continue
                # Different content: renumber with a fresh id. Catch
                # IntegrityError (codex review #3) -- another unique
                # constraint may make this row a semantic duplicate
                # despite differing in metadata (triples has no
                # secondary unique index today; the catch is defensive
                # and symmetric with annotations.import_all).
                try:
                    _insert_without_id(item)
                    stats["imported_renumbered"] += 1
                except sqlite3.IntegrityError:
                    stats["skipped"] += 1

            self.conn.commit()
        except Exception:
            # Roll back partial inserts so a mid-stream failure doesn't
            # leave the connection in a state where a later commit
            # could accidentally persist orphans (codex review #4).
            self.conn.rollback()
            raise
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
