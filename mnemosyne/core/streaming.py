"""
Mnemosyne Streaming Memory + Delta Sync
========================================

Real-time memory event streaming and incremental synchronization
between Mnemosyne instances.

Event-driven architecture:
- Push: callbacks registered on the stream
- Pull: iterate over events as they occur

Delta sync:
- Diff-based: only changed memories since last sync
- Incremental: track sync checkpoints per peer
"""

import json
import logging
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Iterator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path


logger = logging.getLogger(__name__)


# [C25] Tables that DeltaSync is permitted to operate on. Pre-fix, the
# `table` kwarg was interpolated directly into f-string SQL via
# `f"SELECT * FROM {table}"`, `f"INSERT INTO {table} ..."`, etc — a
# real SQL injection vector. The allowlist gates that surface at the
# public method boundary. Adding a new syncable table is a deliberate
# change to this set, not a silent ride-along via the kwarg.
ALLOWED_DELTA_TABLES = frozenset({"working_memory", "episodic_memory"})

# [C25 /review hardening] Qualify table names with the main schema to
# defeat temp-table shadowing. SQLite resolves unqualified names to
# temp schema first; a same-connection `CREATE TEMP TABLE working_memory`
# would make subsequent unqualified SQL target the temp shadow.
# `main.working_memory` always resolves to the real table.
_QUALIFIED_TABLE_NAMES = {
    "working_memory": '"main"."working_memory"',
    "episodic_memory": '"main"."episodic_memory"',
}

# [C25 /review hardening — opt-in allowlist for peer mutations]
# Pre-fix the reserved-column set was "deny known routing keys, accept
# everything else." That accepts `session_id`, `superseded_by`,
# `scope`, `valid_until`, etc — letting a peer reroute another
# session's row, soft-delete victim rows, etc.
#
# Flipped to opt-in: only the columns a peer is allowed to set / mutate
# are accepted. Everything else (identity, scope, lifecycle, audit) is
# destination-controlled. Sync is content-mirror; routing is local.
#
# UPDATE-mutable: peer can mutate row content + sync-relevant metadata
# fields on an EXISTING row matched by id. Identity (id), scope
# (session_id, scope), lifecycle (valid_until, superseded_by,
# created_at, timestamp, recall_count, last_recalled, consolidated_at,
# degraded_at, tier), and authorship (author_id, author_type,
# channel_id) are all destination-controlled.
_DELTA_UPDATABLE_COLUMNS = frozenset({
    "content",
    "importance",
    "metadata_json",
    "veracity",
    "memory_type",
    "binary_vector",  # episodic only; harmless filter on working_memory
    "source",
    "summary_of",     # episodic only
})

# INSERT-acceptable: peer creates a row with content + sync-relevant
# fields. id is the row identity (peer-supplied). Everything else
# (lifecycle / scope / routing / audit) is filled by destination
# defaults — a peer cannot land a row directly inside the
# destination's local session, claim authorship, or pre-tombstone
# a future legitimate write.
_DELTA_INSERTABLE_COLUMNS = frozenset({
    "id",
    "content",
    "importance",
    "metadata_json",
    "veracity",
    "memory_type",
    "binary_vector",
    "source",
    "summary_of",
    "timestamp",  # peer's original creation time — preserves history
})


class EventType(Enum):
    MEMORY_ADDED = auto()
    MEMORY_RECALLED = auto()
    MEMORY_INVALIDATED = auto()
    MEMORY_CONSOLIDATED = auto()
    MEMORY_UPDATED = auto()


@dataclass
class MemoryEvent:
    """A memory system event."""
    event_type: EventType
    memory_id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    session_id: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    importance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    delta: Optional[Dict[str, Any]] = None  # Only changed fields for updates

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_type"] = self.event_type.name
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEvent":
        data["event_type"] = EventType[data["event_type"]]
        return cls(**{k: v for k, v in data.items() if k in [f.name for f in cls.__dataclass_fields__.values()]})


class MemoryStream:
    """
    Real-time event stream for memory operations.

    Supports both push (callbacks) and pull (iterator) patterns.
    Thread-safe. Events are buffered for iterators that connect
    after the event fired.
    """

    def __init__(self, max_buffer: int = 1000):
        self._callbacks: Dict[EventType, List[Callable[[MemoryEvent], None]]] = {
            et: [] for et in EventType
        }
        self._any_callbacks: List[Callable[[MemoryEvent], None]] = []
        self._buffer: List[MemoryEvent] = []
        self._max_buffer = max_buffer
        self._lock = threading.Lock()
        self._iterators: List["_StreamIterator"] = []

    def on(self, event_type: EventType, callback: Callable[[MemoryEvent], None]) -> None:
        """Register a callback for a specific event type."""
        with self._lock:
            self._callbacks[event_type].append(callback)

    def on_any(self, callback: Callable[[MemoryEvent], None]) -> None:
        """Register a callback for all event types."""
        with self._lock:
            self._any_callbacks.append(callback)

    def off(self, event_type: EventType, callback: Callable[[MemoryEvent], None]) -> None:
        """Remove a callback for a specific event type."""
        with self._lock:
            if callback in self._callbacks[event_type]:
                self._callbacks[event_type].remove(callback)

    def off_any(self, callback: Callable[[MemoryEvent], None]) -> None:
        """Remove an any-event callback."""
        with self._lock:
            if callback in self._any_callbacks:
                self._any_callbacks.remove(callback)

    def emit(self, event: MemoryEvent) -> None:
        """Emit an event to all registered callbacks and iterators."""
        with self._lock:
            # Buffer for late-joining iterators
            self._buffer.append(event)
            if len(self._buffer) > self._max_buffer:
                self._buffer = self._buffer[-self._max_buffer:]

            # Notify type-specific callbacks
            callbacks = list(self._callbacks[event.event_type])
            any_callbacks = list(self._any_callbacks)
            iterators = list(self._iterators)

        # Call outside lock to avoid blocking
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass  # Never let a callback break the stream
        for cb in any_callbacks:
            try:
                cb(event)
            except Exception:
                pass
        for it in iterators:
            it._push(event)

    def listen(self, event_types: Optional[List[EventType]] = None) -> Iterator[MemoryEvent]:
        """Return an iterator that yields events as they occur."""
        it = _StreamIterator(self, event_types)
        with self._lock:
            self._iterators.append(it)
        return iter(it)

    def _remove_iterator(self, it: "_StreamIterator") -> None:
        with self._lock:
            if it in self._iterators:
                self._iterators.remove(it)

    def get_buffer(self, event_types: Optional[List[EventType]] = None,
                   since: Optional[str] = None) -> List[MemoryEvent]:
        """Get buffered events, optionally filtered."""
        with self._lock:
            events = list(self._buffer)
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        if since:
            events = [e for e in events if e.timestamp >= since]
        return events

    def clear_buffer(self) -> None:
        """Clear the event buffer."""
        with self._lock:
            self._buffer.clear()


class _StreamIterator:
    """Internal iterator that buffers events from the stream."""

    def __init__(self, stream: MemoryStream, event_types: Optional[List[EventType]] = None):
        self._stream = stream
        self._event_types = event_types
        self._queue: List[MemoryEvent] = []
        self._lock = threading.Lock()
        self._index = 0

    def _push(self, event: MemoryEvent) -> None:
        if self._event_types is None or event.event_type in self._event_types:
            with self._lock:
                self._queue.append(event)

    def __iter__(self):
        return self

    def __next__(self) -> MemoryEvent:
        while True:
            with self._lock:
                if self._index < len(self._queue):
                    event = self._queue[self._index]
                    self._index += 1
                    return event
            # Small sleep to avoid busy-waiting
            import time
            time.sleep(0.01)

    def __del__(self):
        self._stream._remove_iterator(self)


@dataclass
class SyncCheckpoint:
    """Checkpoint for incremental delta sync."""
    peer_id: str
    last_sync_at: str
    last_memory_id: Optional[str] = None
    last_rowid: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class DeltaSync:
    """
    Incremental memory synchronization between two Mnemosyne instances.

    Only transfers memories that have changed since the last sync checkpoint.
    Uses delta encoding: only changed fields, not full objects.
    """

    def __init__(self, mnemosyne_instance, checkpoint_dir: Optional[Path] = None):
        from mnemosyne.core.memory import Mnemosyne
        if not isinstance(mnemosyne_instance, Mnemosyne):
            raise TypeError("DeltaSync requires a Mnemosyne instance")
        self.mnemosyne = mnemosyne_instance
        self.checkpoint_dir = checkpoint_dir or (Path.home() / ".hermes" / "mnemosyne" / "sync")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Keyed by (peer_id, table) — see _checkpoint_path docstring.
        self._checkpoints: Dict[Any, SyncCheckpoint] = {}
        self._lock = threading.Lock()
        # [C25] Per-table column allowlist, lazily populated from the
        # live schema via PRAGMA table_info on first use. Schema-
        # driven so future column additions track automatically. Cached
        # because PRAGMA per row would dominate apply_delta latency.
        self._column_cache: Dict[str, frozenset] = {}
        self._load_checkpoints()

    @staticmethod
    def _validate_table(table, method: str) -> str:
        """[C25] Reject any table not in ALLOWED_DELTA_TABLES.

        Returns the qualified `main.table` form for downstream SQL —
        callers should use the returned string, NOT the raw input.
        Returning the qualified form forces every callsite to go
        through validation (no opportunity to forget the `main.`
        prefix elsewhere).

        Strict `type(table) is str` check (not `isinstance`) — a
        subclass of `str` with overridden `__eq__`/`__hash__` can
        compare-equal to an allowlisted value while carrying a
        different actual string content. The f-string then uses the
        carrier's content, bypassing the gate. Strict type check
        closes this. (Caught by /review adversarial pass — empirical
        exploit demonstrated with MyStr(str).)
        """
        if type(table) is not str or table not in ALLOWED_DELTA_TABLES:
            raise ValueError(
                f"DeltaSync.{method}: table {table!r} is not in the "
                f"allowlist {sorted(ALLOWED_DELTA_TABLES)!r}. To sync a "
                f"new table, add it to ALLOWED_DELTA_TABLES in "
                f"mnemosyne/core/streaming.py — silently accepting "
                f"arbitrary table names is a security regression. "
                f"(Note: strict type check; str subclasses are rejected.)"
            )
        return _QUALIFIED_TABLE_NAMES[table]

    def _allowed_columns(self, table: str) -> frozenset:
        """[C25] Return the schema-derived column allowlist for `table`.

        Defense-in-depth filter on top of the opt-in
        `_DELTA_INSERTABLE_COLUMNS` / `_DELTA_UPDATABLE_COLUMNS` sets:
        an attacker who somehow lands a column name that is in both
        the static allowlist AND the live schema still passes; that
        intersection is small and trustworthy.

        Validates `table` defensively in case a caller reaches this
        method without going through `_validate_table` first
        (`mnemosyne.delta_sync._allowed_columns(...)` etc). /review
        flagged the direct-call path as a defense-in-depth gap.

        Uses `PRAGMA main.table_info("<table>")` to defeat temp-table
        shadowing — SQLite resolves unqualified names to temp schema
        first, so a same-connection `CREATE TEMP TABLE working_memory`
        would otherwise make the PRAGMA return the temp shadow's
        columns. The `main.` prefix + quoted identifier is the fix.
        """
        self._validate_table(table, "_allowed_columns")
        if table in self._column_cache:
            return self._column_cache[table]
        cursor = self.mnemosyne.conn.cursor()
        # Both qualifier and identifier quoted — table is provably one
        # of the small allowlist literals so this is safe.
        cursor.execute(f'PRAGMA main.table_info("{table}")')
        cols = frozenset(row[1] for row in cursor.fetchall())
        if not cols:
            raise ValueError(
                f"DeltaSync._allowed_columns: PRAGMA main.table_info({table!r}) "
                f"returned no columns. The table is in the allowlist "
                f"but the schema is missing — was BeamMemory initialized?"
            )
        self._column_cache[table] = cols
        return cols

    # [C25 /review hardening] Checkpoints are scoped by (peer_id, table).
    # rowid namespaces are table-local; a single per-peer checkpoint
    # produced silent skip-rows on cross-table sync (peer syncs
    # working_memory to rowid=100, then compute_delta(peer, table=
    # episodic_memory) skipped episodic rows at rowid < 100).
    #
    # New filename: checkpoint_<peer>__<table>.json (double-underscore
    # separator avoids collision with peer_id strings containing single
    # underscores). Legacy format (checkpoint_<peer>.json without a
    # table suffix) is loaded as the working_memory checkpoint for
    # backward compat.

    def _checkpoint_path(self, peer_id: str, table: str) -> Path:
        return self.checkpoint_dir / f"checkpoint_{peer_id}__{table}.json"

    @staticmethod
    def _parse_checkpoint_filename(stem: str):
        """Parse 'checkpoint_<peer>__<table>' or legacy 'checkpoint_<peer>'.
        Returns (peer_id, table) — legacy maps to working_memory.
        Returns None if the filename doesn't match either shape."""
        if not stem.startswith("checkpoint_"):
            return None
        body = stem[len("checkpoint_"):]
        if "__" in body:
            peer_id, _, table = body.rpartition("__")
            if not peer_id or not table:
                return None
            return peer_id, table
        # Legacy: pre-/review-hardening files only carry peer_id.
        return body, "working_memory"

    def _load_checkpoints(self) -> None:
        """Load all saved checkpoints (both new and legacy filenames)."""
        if not self.checkpoint_dir.exists():
            return
        for f in self.checkpoint_dir.glob("checkpoint_*.json"):
            parsed = self._parse_checkpoint_filename(f.stem)
            if parsed is None:
                continue
            peer_id, table = parsed
            try:
                with open(f, "r") as fh:
                    data = json.load(fh)
                self._checkpoints[(peer_id, table)] = SyncCheckpoint(**data)
            except Exception:
                pass

    def _save_checkpoint(self, peer_id: str, table: str) -> None:
        """Save checkpoint to disk."""
        cp = self._checkpoints.get((peer_id, table))
        if cp:
            path = self._checkpoint_path(peer_id, table)
            with open(path, "w") as f:
                f.write(cp.to_json())

    def get_checkpoint(self, peer_id: str, table: str = "working_memory") -> Optional[SyncCheckpoint]:
        """Get the current checkpoint for a (peer, table) pair."""
        with self._lock:
            return self._checkpoints.get((peer_id, table))

    def set_checkpoint(self, peer_id: str, checkpoint: SyncCheckpoint,
                       table: str = "working_memory") -> None:
        """Set and save a checkpoint for a (peer, table) pair."""
        with self._lock:
            self._checkpoints[(peer_id, table)] = checkpoint
        self._save_checkpoint(peer_id, table)

    def compute_delta(self, peer_id: str, table: str = "working_memory") -> List[Dict[str, Any]]:
        """
        Compute the delta of changed memories since last sync with peer.

        Returns list of memory dicts with only changed fields if possible,
        or full memory objects for new memories.

        Only `working_memory` and `episodic_memory` are accepted as
        `table` values. Other strings raise ValueError. See C25 in
        the memory-contract ledger.

        Checkpoints are scoped by (peer_id, table). Pre-/review, a
        single checkpoint per peer covered all tables — peer syncing
        working_memory to rowid=100, then later compute_delta for
        episodic_memory with the same peer, would skip episodic rows
        at rowid < 100 because rowid namespaces are table-local.
        """
        qualified = self._validate_table(table, "compute_delta")
        checkpoint = self.get_checkpoint(peer_id, table)
        conn = self.mnemosyne.conn
        cursor = conn.cursor()

        if checkpoint:
            # Get memories modified since last sync
            cursor.execute(f"""
                SELECT * FROM {qualified}
                WHERE rowid > ? OR timestamp > ?
                ORDER BY rowid ASC
            """, (checkpoint.last_rowid, checkpoint.last_sync_at))
        else:
            # First sync: send everything
            cursor.execute(f"""
                SELECT * FROM {qualified}
                ORDER BY rowid ASC
            """)

        rows = cursor.fetchall()
        delta = []
        for row in rows:
            mem = dict(row)
            # Strip internal fields
            mem.pop("embedding", None)
            delta.append(mem)

        return delta

    def apply_delta(self, peer_id: str, delta: List[Dict[str, Any]],
                    table: str = "working_memory") -> Dict[str, int]:
        """
        Apply an incoming delta from a peer.

        Returns stats: {inserted: N, updated: N, skipped: N, filtered_keys: N}.
        `filtered_keys` counts peer-supplied keys that didn't pass the
        column allowlist (typo'd or malicious column names). Pre-C25
        those keys would have crashed the apply (OperationalError) or
        been used directly in SQL. Post-C25 they're silently dropped
        and counted; the rest of the row still applies.

        Only `working_memory` and `episodic_memory` are accepted as
        `table` values. Other strings raise ValueError. See C25 in
        the memory-contract ledger.
        """
        qualified = self._validate_table(table, "apply_delta")
        schema_cols = self._allowed_columns(table)
        # Defense-in-depth: only columns that are BOTH in the static
        # opt-in allowlist AND in the live schema can be touched.
        # Static set protects against schema-poisoning (temp tables,
        # attached DBs, future ALTER TABLE adding sensitive columns);
        # schema set protects against allowlist staleness if a column
        # gets renamed.
        updatable_cols = _DELTA_UPDATABLE_COLUMNS & schema_cols
        insertable_cols = _DELTA_INSERTABLE_COLUMNS & schema_cols
        conn = self.mnemosyne.conn
        cursor = conn.cursor()
        stats = {"inserted": 0, "updated": 0, "skipped": 0, "filtered_keys": 0}

        for mem in delta:
            mid = mem.get("id")
            if not mid:
                stats["skipped"] += 1
                continue

            # Check if exists
            cursor.execute(f"SELECT 1 FROM {qualified} WHERE id = ?", (mid,))
            exists = cursor.fetchone() is not None

            if exists:
                # UPDATE: opt-in allowlist of mutable columns. Identity
                # (id), scope (session_id, scope), lifecycle (valid_until,
                # superseded_by, created_at, timestamp, recall_count,
                # last_recalled, consolidated_at, degraded_at, tier),
                # and authorship (author_id, author_type, channel_id)
                # are all destination-controlled; pre-/review they
                # leaked into UPDATE because the reserved-set was
                # opt-out and only covered the obvious four.
                updatable = {}
                for k, v in mem.items():
                    if k == "id":
                        # match key, not a mutation target
                        continue
                    if k not in updatable_cols:
                        stats["filtered_keys"] += 1
                        continue
                    if v is None:
                        continue
                    updatable[k] = v
                if updatable:
                    # Identifier-quote each column for defense-in-depth
                    # against schema poisoning (temp shadows, etc.).
                    sets = ", ".join(f'"{k}" = ?' for k in updatable.keys())
                    cursor.execute(
                        f"UPDATE {qualified} SET {sets} WHERE id = ?",
                        list(updatable.values()) + [mid]
                    )
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
            else:
                # INSERT: opt-in allowlist. id must be present (peer
                # supplies row identity). Destination defaults fill
                # session_id ('default'), scope ('session'), audit
                # columns, etc — a peer cannot land a row directly
                # inside the destination's local session, claim
                # authorship, or pre-tombstone via superseded_by.
                cols = []
                for k in mem.keys():
                    if k not in insertable_cols:
                        stats["filtered_keys"] += 1
                        continue
                    cols.append(k)
                if not cols or "id" not in cols:
                    stats["skipped"] += 1
                    continue
                quoted_cols = ", ".join(f'"{c}"' for c in cols)
                placeholders = ", ".join("?" for _ in cols)
                cursor.execute(
                    f"INSERT INTO {qualified} ({quoted_cols}) VALUES ({placeholders})",
                    [mem.get(c) for c in cols]
                )
                stats["inserted"] += 1

        conn.commit()

        # Update checkpoint scoped by (peer_id, table) — rowid
        # namespaces are table-local; a single checkpoint per peer
        # across tables produces silent skip-rows on cross-table sync.
        cursor.execute(f"SELECT MAX(rowid) FROM {qualified}")
        max_rowid = cursor.fetchone()[0] or 0
        self.set_checkpoint(peer_id, SyncCheckpoint(
            peer_id=peer_id,
            last_sync_at=datetime.now().isoformat(),
            last_rowid=max_rowid
        ), table=table)

        return stats

    def sync_to(self, peer_id: str, table: str = "working_memory") -> Dict[str, Any]:
        """
        Full sync cycle: compute delta for peer, return it.
        The caller is responsible for sending the delta to the peer.
        """
        delta = self.compute_delta(peer_id, table)
        cp = self.get_checkpoint(peer_id, table)
        return {
            "peer_id": peer_id,
            "table": table,
            "delta": delta,
            "count": len(delta),
            "checkpoint": cp.to_dict() if cp else None
        }

    def sync_from(self, peer_id: str, delta: List[Dict[str, Any]],
                  table: str = "working_memory") -> Dict[str, Any]:
        """
        Full sync cycle: apply delta from peer.
        """
        stats = self.apply_delta(peer_id, delta, table)
        cp = self.get_checkpoint(peer_id, table)
        return {
            "peer_id": peer_id,
            "table": table,
            "stats": stats,
            "checkpoint": cp.to_dict() if cp else None
        }
