"""
Mnemosyne Sync Engine
=====================
Event-log-based memory synchronization with conflict resolution,
optional encryption, and HTTP transport.

Designed to work standalone (no Hermes dependency) on top of
Mnemosyne's BEAM architecture.
"""

import json
import hashlib
import logging
import sys
import uuid
import os
import base64
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple, Union

logger = logging.getLogger(__name__)


def _parse_sync_timestamp(value: str) -> datetime:
    """Parse sync timestamps consistently across supported Python versions.

    Python 3.10's ``datetime.fromisoformat`` does not accept a trailing
    ``Z`` UTC designator, while newer versions do. Normalize it so conflict
    detection behaves the same on every CI Python.
    """
    if isinstance(value, str) and value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


# ---------------------------------------------------------------------------
# SyncEvent dataclass
# ---------------------------------------------------------------------------

@dataclass
class SyncEvent:
    """A tracked sync event representing a memory mutation."""
    event_id: str
    memory_id: str
    operation: str  # 'CREATE' | 'UPDATE' | 'DELETE' | 'CONSOLIDATE'
    timestamp: str
    device_id: str
    payload: Optional[str] = None
    parent_event_ids: str = "[]"
    importance: float = 0.5
    expiry: Optional[str] = None
    event_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncEvent":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**clean)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "SyncEvent":
        """Build from a sqlite3.Row / dict returned by the DB."""
        return cls(
            event_id=row.get("event_id", ""),
            memory_id=row.get("memory_id", ""),
            operation=row.get("operation", ""),
            timestamp=row.get("timestamp", ""),
            device_id=row.get("device_id", ""),
            payload=row.get("payload"),
            parent_event_ids=row.get("parent_event_ids", "[]"),
            importance=row.get("importance", 0.5),
            expiry=row.get("expiry"),
            event_hash=row.get("event_hash"),
        )


# ---------------------------------------------------------------------------
# SyncEncryption — optional encryption layer
# ---------------------------------------------------------------------------

class SyncEncryption:
    """Encryption for sync payloads.

    Uses cryptography.fernet.Fernet if available, falling back to
    PyNaCl secretbox. Key derivation uses PBKDF2HMAC (SHA256, 600K
    iterations) or Argon2id if argon2-cffi is installed.
    """

    @staticmethod
    def derive_key(passphrase: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Derive a 32-byte key from *passphrase*.

        Returns (key, salt) — salt is random if not provided, so
        callers should store it alongside the ciphertext.
        """
        import hashlib as _hlib

        if salt is None:
            salt = os.urandom(16)

        # Try Argon2id first
        try:
            import argon2.low_level as _argon2
            key = _argon2.hash_secret_raw(
                secret=passphrase.encode("utf-8"),
                salt=salt,
                time_cost=2,
                memory_cost=19456,   # 19 MB
                parallelism=1,
                hash_len=32,
                type=_argon2.Type.ID,
            )
            return key, salt
        except ImportError:
            pass

        # Fallback: PBKDF2HMAC (SHA256, 600K iterations)
        try:
            from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC as _PBKDF2
            from cryptography.hazmat.primitives import hashes as _hashes
            kdf = _PBKDF2(
                algorithm=_hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=600_000,
            )
            key = kdf.derive(passphrase.encode("utf-8"))
            return key, salt
        except ImportError:
            raise ImportError(
                "SyncEncryption requires either 'cryptography>=41.0' or "
                "'argon2-cffi' for key derivation. "
                "Install with: pip install mnemosyne-memory[sync]"
            )

    @staticmethod
    def generate_key() -> str:
        """Generate a random 32-byte key, base64-encoded."""
        return base64.urlsafe_b64encode(os.urandom(32)).decode("ascii")

    @staticmethod
    def encrypt_payload(payload: dict, key: bytes) -> str:
        """Serialize *payload* to JSON, encrypt, and return base64 string."""
        try:
            from cryptography.fernet import Fernet
            f = Fernet(base64.urlsafe_b64encode(key))
            data = json.dumps(payload, default=str).encode("utf-8")
            return f.encrypt(data).decode("utf-8")
        except ImportError:
            pass

        try:
            import nacl.secret as _secret
            import nacl.utils as _utils
            box = _secret.SecretBox(key)
            data = json.dumps(payload, default=str).encode("utf-8")
            encrypted = box.encrypt(data)
            return base64.b64encode(encrypted).decode("ascii")
        except ImportError:
            raise ImportError(
                "SyncEncryption.encrypt_payload requires 'cryptography>=41.0' "
                "or 'PyNaCl>=1.5'. Install with: pip install mnemosyne-memory[sync]"
            )

    @staticmethod
    def decrypt_payload(encrypted: str, key: bytes) -> dict:
        """Decrypt a base64-encoded encrypted payload back to a dict."""
        try:
            from cryptography.fernet import Fernet
            f = Fernet(base64.urlsafe_b64encode(key))
            data = f.decrypt(encrypted.encode("utf-8"))
            return json.loads(data.decode("utf-8"))
        except ImportError:
            pass

        try:
            import nacl.secret as _secret
            box = _secret.SecretBox(key)
            raw = base64.b64decode(encrypted)
            decrypted = box.decrypt(raw)
            return json.loads(decrypted.decode("utf-8"))
        except ImportError:
            raise ImportError(
                "SyncEncryption.decrypt_payload requires 'cryptography>=41.0' "
                "or 'PyNaCl>=1.5'. Install with: pip install mnemosyne-memory[sync]"
            )

    @classmethod
    def from_config(cls, key_source: Optional[str] = None, **kwargs) -> Optional["SyncEncryption"]:
        """Attempt to load an encryption key from environment, keyring, or a file.

        Returns a SyncEncryption instance or None if no key is configured.
        """
        key: Optional[bytes] = None

        if key_source:
            # key_source could be a file path or raw key
            if os.path.isfile(key_source):
                with open(key_source, "r") as fh:
                    raw = fh.read().strip()
                key = base64.urlsafe_b64decode(raw)
            else:
                # Treat as raw base64-encoded key
                try:
                    key = base64.urlsafe_b64decode(key_source)
                except Exception:
                    try:
                        key = base64.urlsafe_b64decode(key_source + "==")
                    except Exception:
                        raise ValueError(
                            f"key_source is neither a file path nor a valid "
                            f"base64-encoded key"
                        )
        elif "MNEMOSYNE_SYNC_KEY" in os.environ:
            raw = os.environ["MNEMOSYNE_SYNC_KEY"].strip()
            key = base64.urlsafe_b64decode(raw)

        if key is None:
            return None

        # Wrap in a lightweight object that exposes encrypt/decrypt
        instance = cls.__new__(cls)
        instance._key = key
        return instance

    def encrypt(self, payload: dict) -> str:
        return self.encrypt_payload(payload, self._key)

    def decrypt(self, encrypted: str) -> dict:
        return self.decrypt_payload(encrypted, self._key)


# ---------------------------------------------------------------------------
# ConflictResolution — simple last-writer-wins + tiebreaker
# ---------------------------------------------------------------------------

class ConflictResolution:
    """v1/v2 conflict resolution strategy.

    v1 (``resolve``): simple last-writer-wins with tiebreakers.
        1. Latest timestamp wins
        2. Higher importance breaks ties
        3. Deterministic device_id comparison as final tiebreaker

    v2 (``resolve_with_chain``): version-chain-aware resolution.
        Uses parent_event_ids to detect causal relationships. If event B's
        parent_event_ids contain event A's event_id, B is a strictly-later
        version of A and wins by default. Falls back to the v1 strategy when
        no causal relationship is detected.
    """

    # ------------------------------------------------------------------
    # v1 -- simple last-writer-wins
    # ------------------------------------------------------------------

    @staticmethod
    def resolve(events: List[SyncEvent]) -> SyncEvent:
        """Pick the winning event from a group of conflicting events."""
        if not events:
            raise ValueError("Cannot resolve empty event list")
        if len(events) == 1:
            return events[0]

        def _sort_key(ev: SyncEvent):
            ts = ev.timestamp
            imp = ev.importance if ev.importance is not None else 0.0
            dev = ev.device_id or ""
            return (ts, imp, dev)

        # Sort descending: latest timestamp, highest importance,
        # then deterministic device_id
        sorted_events = sorted(events, key=_sort_key, reverse=True)
        return sorted_events[0]

    # ------------------------------------------------------------------
    # v2 -- version-chain-aware resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_parent_ids(event: SyncEvent) -> List[str]:
        """Parse the parent_event_ids JSON field into a Python list.

        Handles both already-parsed lists and JSON-encoded strings.
        """
        raw = event.parent_event_ids
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, list) else []
            except (json.JSONDecodeError, TypeError):
                return []
        return []

    @staticmethod
    def _build_parent_map(events: List[SyncEvent]) -> Dict[str, List[str]]:
        """Build a mapping of event_id -> list of parent_event_ids.

        Returns a dict where each key is an event_id and each value is
        the list of event_ids that this event declares as its direct
        causal parents.
        """
        parent_map: Dict[str, List[str]] = {}
        for ev in events:
            parent_map[ev.event_id] = ConflictResolution._parse_parent_ids(ev)
        return parent_map

    @staticmethod
    def resolve_with_chain(
        events: List[SyncEvent],
        parent_map: Optional[Dict[str, List[str]]] = None,
    ) -> SyncEvent:
        """Resolve conflicts using version-chain information (v2 strategy).

        **Causal relationship detection**: If event B lists event A's
        event_id in its ``parent_event_ids`` field, then B is a strictly-
        later version of A and wins the conflict by default.  This works
        transitively: B → A and C → B means C wins over both.

        **Fallback**: When no causal relationship exists between any pair
        of conflicting events, falls back to the v1 strategy (latest
        timestamp → higher importance → deterministic device_id).

        Args:
            events: Conflicting events for the same memory_id.
            parent_map: Optional pre-computed mapping of event_id →
                parent_event_ids.  Built from the events themselves if
                not provided.

        Returns:
            The winning SyncEvent.

        Raises:
            ValueError: If *events* is empty.
        """
        if not events:
            raise ValueError("Cannot resolve empty event list")
        if len(events) == 1:
            return events[0]

        # Build parent map if not provided
        if parent_map is None:
            parent_map = ConflictResolution._build_parent_map(events)

        # Phase 1 — build a "descendants" lookup: for each event,
        # collect all events that transitively descend from it.
        event_ids = {ev.event_id for ev in events}

        # Compute transitive ancestors for each event (BFS / fixed-point)
        ancestors: Dict[str, set] = {}
        for ev in events:
            ancestors[ev.event_id] = set(ConflictResolution._parse_parent_ids(ev))

        # Expand transitively until stable
        changed = True
        while changed:
            changed = False
            for ev in events:
                current = ancestors[ev.event_id]
                expanded = set(current)
                for pid in current:
                    if pid in ancestors:
                        expanded |= ancestors[pid]
                if expanded != current:
                    ancestors[ev.event_id] = expanded
                    changed = True

        # Phase 2 — determine if any event is a strict descendant of another
        # within the conflict group.  B > A if A's event_id is in B's
        # transitive ancestors.
        dominated: set = set()
        for ev_a in events:
            for ev_b in events:
                if ev_a.event_id == ev_b.event_id:
                    continue
                # If ev_b has ev_a in its ancestors, ev_a is dominated
                if ev_a.event_id in ancestors.get(ev_b.event_id, set()):
                    dominated.add(ev_a.event_id)

        # Phase 3 — collect undominated events
        undominated = [ev for ev in events if ev.event_id not in dominated]

        if len(undominated) == 1:
            return undominated[0]

        # Phase 4 — fallback to v1 for remaining undominated events
        return ConflictResolution.resolve(undominated)

    @staticmethod
    def detect_conflicts(
        local_events: List[SyncEvent],
        remote_events: List[SyncEvent],
        window_seconds: float = 5.0,
    ) -> List[List[SyncEvent]]:
        """Find groups of events that conflict.

        Two events conflict if they share the same memory_id and
        their timestamps differ by at most *window_seconds*.
        Returns a list of conflict groups (each group is list of events).
        """
        from collections import defaultdict

        # Index local events by memory_id
        local_by_mid: Dict[str, List[SyncEvent]] = defaultdict(list)
        for ev in local_events:
            local_by_mid[ev.memory_id].append(ev)

        remote_by_mid: Dict[str, List[SyncEvent]] = defaultdict(list)
        for ev in remote_events:
            remote_by_mid[ev.memory_id].append(ev)

        conflicts: List[List[SyncEvent]] = []

        # Check all memory_ids present in either set
        all_mids = set(local_by_mid.keys()) | set(remote_by_mid.keys())

        for mid in all_mids:
            local_for_mid = local_by_mid.get(mid, [])
            remote_for_mid = remote_by_mid.get(mid, [])

            if not local_for_mid or not remote_for_mid:
                continue

            # Compare each local vs each remote for this memory_id
            for lev in local_for_mid:
                for rev in remote_for_mid:
                    try:
                        lts = _parse_sync_timestamp(lev.timestamp)
                        rts = _parse_sync_timestamp(rev.timestamp)
                    except (ValueError, TypeError):
                        continue

                    diff = abs((lts - rts).total_seconds())
                    if diff <= window_seconds:
                        # All events in this conflict group
                        group = [lev, rev]
                        # Add any other remote events in window
                        for rev2 in remote_for_mid:
                            if rev2.event_id != rev.event_id:
                                try:
                                    rts2 = _parse_sync_timestamp(rev2.timestamp)
                                    if abs((lts - rts2).total_seconds()) <= window_seconds:
                                        group.append(rev2)
                                except (ValueError, TypeError):
                                    pass
                        # Deduplicate by event_id
                        seen_ids = set()
                        deduped = []
                        for ev in group:
                            if ev.event_id not in seen_ids:
                                seen_ids.add(ev.event_id)
                                deduped.append(ev)
                        if len(deduped) > 1:
                            conflicts.append(deduped)

        return conflicts

    # ------------------------------------------------------------------
    # Agent-assisted merge proposal (stub)
    # ------------------------------------------------------------------

    @staticmethod
    def propose_merge(
        conflict_groups: List[List[SyncEvent]],
        full_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Build a merge-proposal data structure suitable for LLM consumption.

        This is a **stub** that returns a structured dict for each
        conflict group.  It does *not* call any LLM itself; a Hermes
        plugin or other agent system consumes the output, applies its
        own reasoning, and returns a resolution.

        **How a Hermes plugin would consume this**:

        1. The plugin calls ``propose_merge()`` to obtain a list of
           conflict proposals.
        2. It serialises each proposal into a prompt, e.g.::

               "You are resolving conflicting memory updates for
                memory {memory_id}.  Here are the candidates..."

        3. The LLM responds with an action string:
           ``"keep_latest"``, ``"merge"``, or ``"keep_both"`` and
           (optionally) a merged content string and favoured
           candidate index.
        4. The plugin feeds the LLM's decision back into
           :meth:`ConflictResolution.resolve_with_chain` or
           a custom reconciliation routine.

        Args:
            conflict_groups: List of conflict groups, each a list of
                conflicting SyncEvents (as returned by
                :meth:`detect_conflicts`).
            full_context: Optional dict with additional context the
                LLM agent might need (e.g. ``{"memory_bank": "...",
                "user_identity": "...", "recent_decisions": [...]}``).

        Returns:
            A list of merge proposals, one per conflict group.  Each
            proposal is a dict with the following keys:

            * ``memory_id`` (str) — the memory being conflicted
            * ``candidates`` (list[dict]) — each candidate has keys
              ``device``, ``content``, ``importance``, ``timestamp``
            * ``suggested_action`` (str) — pre-computed suggestion:
              ``"keep_latest"``, ``"merge"``, or ``"keep_both"``
            * ``suggested_winner_index`` (int | None) — index into
              *candidates* of the suggested winner, if applicable
            * ``context`` (dict | None) — the *full_context* passed
              by the caller (may be augmented by the stub)
        """
        proposals: List[Dict[str, Any]] = []

        for group in conflict_groups:
            if len(group) < 2:
                continue

            memory_id = group[0].memory_id

            # Build candidate summaries
            candidates: List[Dict[str, Any]] = []
            for ev in group:
                content = ""
                if ev.payload:
                    try:
                        content = json.loads(ev.payload).get("content", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                candidates.append({
                    "device": ev.device_id,
                    "content": content,
                    "importance": ev.importance or 0.5,
                    "timestamp": ev.timestamp,
                    "event_id": ev.event_id,
                })

            # Pre-compute a simple heuristic suggestion: favour the
            # candidate with the highest importance.  The LLM agent
            # can override this.
            best_idx = max(
                range(len(candidates)),
                key=lambda i: (
                    candidates[i]["importance"],
                    candidates[i]["timestamp"],
                ),
            )
            suggested_action = "keep_latest"

            proposals.append({
                "memory_id": memory_id,
                "candidates": candidates,
                "suggested_action": suggested_action,
                "suggested_winner_index": best_idx,
                "context": full_context or {},
            })

        return proposals


# ---------------------------------------------------------------------------
# SyncEngine — main sync orchestrator
# ---------------------------------------------------------------------------

class SyncEngine:
    """Orchestrates memory synchronization between Mnemosyne instances.

    Uses the memory_events table as an append-only event log and
    DeltaSync for applying memory mutations.

    Usage:
        engine = SyncEngine(mnemosyne_instance, device_id="my-device")
        engine.log_event("mem-123", "UPDATE", payload={"content": "new"})
        changes = engine.pull_changes(since_cursor="2024-01-01T00:00:00")
        result = engine.push_changes(changes["events"])
    """

    def __init__(
        self,
        beam_instance,
        device_id: Optional[str] = None,
        encryption: Optional[SyncEncryption] = None,
    ):
        # Accept either a Mnemosyne or a BeamMemory instance.
        # Store both the outer (Mnemosyne) and inner (BeamMemory) so
        # push_changes can route through the full memory pipeline
        # (FTS5, embeddings, entity extraction) via remember().
        self._mnemosyne: Any = None
        self._beam: Any = beam_instance
        if hasattr(beam_instance, "beam"):
            self._mnemosyne = beam_instance
            self._beam = beam_instance.beam
        if not hasattr(self._beam, "conn"):
            pass

        self.conn = self._beam.conn
        self.encryption = encryption

        # Lazy import DeltaSync (avoid circular at module level)
        # DeltaSync requires a full Mnemosyne instance; if we only have
        # a raw connection, the engine still works for event logging and
        # pull_changes — push_changes will degrade gracefully.
        self._delta_sync: Any = None
        try:
            from mnemosyne.core.streaming import DeltaSync
            self._delta_sync = DeltaSync(
                self._beam if hasattr(self._beam, "sleep") else beam_instance
            )
        except (TypeError, ImportError) as _ds_err:
            logger.debug("DeltaSync not available: %s", _ds_err)
            self._delta_sync = None

        self._lock = threading.Lock()

        self._init_events_table()

        # Device identity: explicit arg wins, then load from DB, then generate
        # new. Persisted in sync_meta so `mnemosyne sync-status` reports a
        # stable device_id across restarts.
        if device_id:
            self.device_id = device_id
        else:
            stored = self._meta_get("device_id")
            if stored:
                self.device_id = stored
            else:
                self.device_id = f"device-{uuid.uuid4().hex[:8]}"
                self._meta_set("device_id", self.device_id)

    def _init_events_table(self) -> None:
        """Safely ensure the memory_events and sync_meta tables exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                operation TEXT NOT NULL CHECK(operation IN ('CREATE','UPDATE','DELETE','CONSOLIDATE')),
                timestamp TEXT NOT NULL,
                device_id TEXT NOT NULL,
                payload TEXT,
                parent_event_ids TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                expiry TEXT,
                event_hash TEXT,
                synced_at TEXT
            )
        """)
        # Persist device identity and sync state across engine restarts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Indices (IF NOT EXISTS is not supported for indices in all
        # SQLite versions, so we try/except)
        for index_ddl in [
            "CREATE INDEX IF NOT EXISTS idx_me_timestamp ON memory_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_me_memory_id ON memory_events(memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_me_device_id ON memory_events(device_id)",
        ]:
            try:
                cursor.execute(index_ddl)
            except Exception:
                pass
        self.conn.commit()

    def _meta_get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM sync_meta WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row[0] if row else default

    def _meta_set(self, key: str, value: str) -> None:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT OR REPLACE INTO sync_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def _compute_event_hash(self, event: SyncEvent) -> str:
        """Compute a deterministic hash for an event (for dedup)."""
        raw = (
            f"{event.memory_id}|{event.operation}|{event.timestamp}|"
            f"{event.device_id}|{event.payload or ''}|"
            f"{event.parent_event_ids}|{event.importance}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def log_event(
        self,
        memory_id: str,
        operation: str,
        payload: Optional[dict] = None,
        importance: float = 0.5,
        parent_event_ids: Optional[List[str]] = None,
    ) -> SyncEvent:
        """Create and persist a sync event.

        This is the primary method to record memory mutations for
        replication to peers.
        """
        if operation not in ("CREATE", "UPDATE", "DELETE", "CONSOLIDATE"):
            raise ValueError(f"Invalid operation: {operation!r}")

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Serialize payload if present
        payload_str: Optional[str] = None
        if payload is not None:
            if self.encryption:
                payload_str = self.encryption.encrypt(payload)
            else:
                payload_str = json.dumps(payload, default=str)

        parent_ids_json = json.dumps(parent_event_ids or [])

        event = SyncEvent(
            event_id=event_id,
            memory_id=memory_id,
            operation=operation,
            timestamp=now,
            device_id=self.device_id,
            payload=payload_str,
            parent_event_ids=parent_ids_json,
            importance=importance,
        )
        event.event_hash = self._compute_event_hash(event)

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO memory_events (
                event_id, memory_id, operation, timestamp, device_id,
                payload, parent_event_ids, importance, expiry, event_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id,
                event.memory_id,
                event.operation,
                event.timestamp,
                event.device_id,
                event.payload,
                event.parent_event_ids,
                event.importance,
                event.expiry,
                event.event_hash,
            ),
        )
        self.conn.commit()

        logger.debug(
            "Logged sync event %s: %s %s", event_id, operation, memory_id
        )
        return event

    def _find_unlogged_memories(self, limit: int = 5000) -> List[dict]:
        """Scan working_memory for entries not yet logged as sync events.

        Creates events for unlogged memories so they can be synced to peers.
        Returns the events created.
        """
        cursor = self.conn.cursor()
        # Find memory IDs already in the event log
        cursor.execute("SELECT DISTINCT memory_id FROM memory_events")
        logged_ids = {row[0] for row in cursor.fetchall()}

        cursor.execute(
            """SELECT id, content, source, timestamp, importance, metadata_json, memory_type, veracity
               FROM main.working_memory
               ORDER BY timestamp ASC
               LIMIT ?""",
            (limit,),
        )
        rows = cursor.fetchall()
        created = []
        for row in rows:
            mem_id = row["id"] if isinstance(row, dict) else row[0]
            if mem_id in logged_ids:
                continue
            if isinstance(row, dict):
                content = row.get("content", "")
                source = row.get("source", "conversation")
                timestamp = row.get("timestamp", "")
                importance = row.get("importance", 0.5)
                metadata_json = row.get("metadata_json")
            else:
                content = row[1] or ""
                source = row[2] or "conversation"
                timestamp = row[3] or ""
                importance = row[4] if row[4] is not None else 0.5
                metadata_json = row[6] if len(row) > 6 else None

            payload = {"content": content, "source": source}
            if metadata_json:
                try:
                    payload["metadata_json"] = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                except (json.JSONDecodeError, TypeError):
                    payload["metadata_json"] = metadata_json

            ev = self.log_event(
                memory_id=mem_id,
                operation="CREATE",
                payload=payload,
                importance=float(importance) if importance else 0.5,
            )
            created.append(ev)
        return created

    def pull_changes(
        self,
        since_cursor: Optional[str] = None,
        limit: int = 1000,
        device_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pull events from the local event log since a cursor.

        Returns:
            {
                "events": [SyncEvent, ...],
                "next_cursor": str | None,
                "has_more": bool,
                "total": int
            }
        """
        cursor = self.conn.cursor()

        if since_cursor:
            cursor.execute(
                """SELECT * FROM memory_events
                   WHERE timestamp > ?
                   ORDER BY timestamp ASC, event_id ASC
                   LIMIT ?""",
                (since_cursor, limit + 1),
            )
        else:
            cursor.execute(
                """SELECT * FROM memory_events
                   ORDER BY timestamp ASC, event_id ASC
                   LIMIT ?""",
                (limit + 1,),
            )

        rows = cursor.fetchall()
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        events = [SyncEvent.from_row(dict(row)) for row in rows]

        next_cursor = None
        if events:
            next_cursor = events[-1].timestamp
            # Ensure cursor is past the last returned event
            if has_more:
                import uuid as _uuid
                next_cursor = events[-1].timestamp

        return {
            "events": [ev.to_dict() for ev in events],
            "next_cursor": next_cursor,
            "has_more": has_more,
            "total": len(events),
        }

    def push_changes(
        self,
        events: List[dict],
    ) -> Dict[str, Any]:
        """Validate, deduplicate, and apply incoming events.

        Uses DeltaSync.apply_delta to apply memory mutations.
        Returns stats: {accepted, duplicates, conflicts, errors, next_cursor}.

        Events with an event_hash already present in the local log are
        silently skipped (idempotent push).
        """
        cursor = self.conn.cursor()
        stats: Dict[str, Any] = {
            "accepted": 0,
            "duplicates": 0,
            "conflicts": 0,
            "errors": 0,
            "details": [],
        }

        # Build set of known event_hashes for dedup
        cursor.execute("SELECT event_hash FROM memory_events WHERE event_hash IS NOT NULL")
        known_hashes = {row[0] for row in cursor.fetchall()}

        incoming: List[SyncEvent] = []
        for raw in events:
            try:
                ev = SyncEvent.from_dict(raw)
                if ev.event_hash and ev.event_hash in known_hashes:
                    stats["duplicates"] += 1
                    continue
                incoming.append(ev)
            except Exception as exc:
                stats["errors"] += 1
                stats["details"].append(f"invalid event: {exc}")
                continue

        # Detect conflicts with local events in the same time window
        if incoming:
            local_raw = self.pull_changes(
                since_cursor=None, limit=5000
            )
            local_events = [SyncEvent.from_dict(d) for d in local_raw["events"]]
            conflict_groups = ConflictResolution.detect_conflicts(
                local_events, incoming
            )

            # Resolve each conflict group
            resolved_ids: set = set()
            for group in conflict_groups:
                winner = ConflictResolution.resolve(group)
                for ev in group:
                    if ev.event_id != winner.event_id:
                        resolved_ids.add(ev.event_id)
                stats["conflicts"] += len(group) - 1

            # Filter out losing events, keep winners
            incoming = [ev for ev in incoming if ev.event_id not in resolved_ids]

        # Apply memory mutations through the full Mnemosyne pipeline
        # (FTS5 indexing, embeddings, entity extraction, callbacks).
        _total = len(incoming)
        _progress_interval = max(1, _total // 50) if _total > 100 else 100
        for idx, ev in enumerate(incoming):
            try:
                if _total > 100 and idx > 0 and idx % _progress_interval == 0:
                    pct = int(idx / _total * 100)
                    sys.stderr.write(f"\r  Progress: {idx}/{_total} ({pct}%)  \r")
                    sys.stderr.flush()
            except KeyboardInterrupt:
                stats["interrupted"] = True
                break
            try:
                payload_dict: Optional[dict] = None
                if ev.payload:
                    # Detect encrypted payloads (Fernet base64 prefix)
                    _is_encrypted = ev.payload.startswith("gAAAAA")
                    if self.encryption and _is_encrypted:
                        payload_dict = self.encryption.decrypt(ev.payload)
                    elif not _is_encrypted:
                        payload_dict = json.loads(ev.payload)
                    else:
                        # Encrypted but no key -- store opaque; memory
                        # mutation skipped, but event is logged for relay
                        pass

                content = ""
                source = "sync"
                importance = ev.importance or 0.5
                metadata: Optional[dict] = None
                veracity = "unknown"

                if payload_dict:
                    content = payload_dict.get("content", "")
                    source = payload_dict.get("source", "sync")
                    importance = payload_dict.get("importance", importance)
                    metadata = payload_dict.get("metadata_json")
                    if metadata and isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except (json.JSONDecodeError, TypeError):
                            pass
                    veracity = payload_dict.get("veracity", "unknown")

                if ev.operation == "DELETE":
                    # Call forget() through the Mnemosyne API pipeline
                    if self._mnemosyne is not None and hasattr(self._mnemosyne, "forget"):
                        self._mnemosyne.forget(ev.memory_id)
                    else:
                        cursor.execute(
                            "DELETE FROM main.working_memory WHERE id = ?",
                            (ev.memory_id,),
                        )
                    stats["accepted"] += 1

                elif ev.operation in ("CREATE", "UPDATE", "CONSOLIDATE"):
                    if content:
                        # Route through remember() for full pipeline
                        try:
                            if self._beam is not None and hasattr(self._beam, "remember"):
                                self._beam.remember(
                                    content=content,
                                    source=source,
                                    importance=importance,
                                    metadata=metadata,
                                    memory_id=ev.memory_id,
                                )
                                stats["accepted"] += 1
                            else:
                                # Fallback: direct DeltaSync
                                delta_item: Dict[str, Any] = {"id": ev.memory_id}
                                for key in ("content", "importance", "source", "memory_type", "veracity"):
                                    if payload_dict and key in payload_dict:
                                        delta_item[key] = payload_dict[key]
                                if self._delta_sync is not None:
                                    apply_stats = self._delta_sync.apply_delta(
                                        peer_id=ev.device_id,
                                        delta=[delta_item],
                                        table="working_memory",
                                    )
                                    if apply_stats.get("inserted") or apply_stats.get("updated"):
                                        stats["accepted"] += 1
                                    else:
                                        stats["details"].append(
                                            f"event {ev.event_id}: no rows affected"
                                        )
                                else:
                                    stats["accepted"] += 1
                        except KeyboardInterrupt:
                            stats["interrupted"] = True
                            break
                    else:
                        # No content — just log the event
                        stats["accepted"] += 1

                # Log the event locally for future syncs
                cursor.execute(
                    """INSERT OR IGNORE INTO memory_events (
                        event_id, memory_id, operation, timestamp, device_id,
                        payload, parent_event_ids, importance, expiry, event_hash, synced_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        ev.event_id,
                        ev.memory_id,
                        ev.operation,
                        ev.timestamp,
                        ev.device_id,
                        ev.payload,
                        ev.parent_event_ids,
                        ev.importance,
                        ev.expiry,
                        ev.event_hash,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
            except Exception as exc:
                stats["errors"] += 1
                stats["details"].append(f"event {ev.event_id}: {exc}")
                logger.warning("Failed to apply event %s: %s", ev.event_id, exc)

        self.conn.commit()

        return stats

    def sync_with(
        self,
        remote_url: str,
        mode: str = "bidirectional",
        api_key: Optional[str] = None,
        encryption_key: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Run a full sync cycle with a remote sync server.

        *mode* can be 'push', 'pull', or 'bidirectional' (default).

        Returns a summary dict with stats for each phase.
        """
        import urllib.request as _request
        import urllib.error as _error

        result: Dict[str, Any] = {
            "remote": remote_url,
            "mode": mode,
            "push": None,
            "pull": None,
            "errors": [],
        }

        headers: Dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def _post(endpoint: str, body: dict) -> Optional[dict]:
            url = f"{remote_url.rstrip('/')}{endpoint}"
            data = json.dumps(body, default=str).encode("utf-8")
            req = _request.Request(url, data=data, headers=headers, method="POST")
            try:
                with _request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except _error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
                result["errors"].append(f"HTTP {e.code} on {endpoint}: {err_body}")
                return None
            except Exception as e:
                result["errors"].append(f"{endpoint}: {e}")
                return None

        def _get(endpoint: str) -> Optional[dict]:
            url = f"{remote_url.rstrip('/')}{endpoint}"
            req = _request.Request(url, headers=headers, method="GET")
            try:
                with _request.urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                result["errors"].append(f"{endpoint}: {e}")
                return None

        # Phase 1: Push (send local changes to remote)
        if mode in ("push", "bidirectional"):
            # Scan for memories that haven't been logged as events yet
            new_events = self._find_unlogged_memories(limit=5000)
            if new_events:
                logger.debug("Found %d unlogged memories to sync", len(new_events))

            pull_status = _post("/sync/pull", {
                "since": None,
                "device_id": self.device_id,
                "limit": 1000,
            })
            remote_since = None
            if pull_status and "next_cursor" in pull_status:
                remote_since = pull_status["next_cursor"]

            local_changes = self.pull_changes(since_cursor=remote_since, limit=5000)
            if local_changes["events"]:
                push_resp = _post("/sync/push", {
                    "events": local_changes["events"],
                    "device_id": self.device_id,
                })
                result["push"] = push_resp
            else:
                result["push"] = {"accepted": 0, "duplicates": 0, "conflicts": 0}

        # Phase 2: Pull (fetch remote changes)
        if mode in ("pull", "bidirectional"):
            # Load persisted cursor, then fall back to DB-derived max
            since_cursor = self._meta_get(
                f"last_sync_cursor_{remote_url}"
            )
            if not since_cursor:
                cur = self.conn.cursor()
                cur.execute(
                    "SELECT MAX(timestamp) FROM memory_events WHERE device_id != ?",
                    (self.device_id,),
                )
                row = cur.fetchone()
                since_cursor = row[0] if row and row[0] else None

            pull_resp = _post("/sync/pull", {
                "since": since_cursor,
                "device_id": self.device_id,
                "limit": 5000,
            })

            if pull_resp and pull_resp.get("events"):
                push_result = self.push_changes(pull_resp["events"])
                result["pull"] = {
                    "events_fetched": len(pull_resp["events"]),
                    "accepted": push_result.get("accepted", 0),
                    "duplicates": push_result.get("duplicates", 0),
                    "conflicts": push_result.get("conflicts", 0),
                    "errors": push_result.get("errors", 0),
                }
                if push_result.get("interrupted"):
                    result["interrupted"] = True
                # Persist cursor so next sync picks up where we left off
                if pull_resp.get("next_cursor"):
                    self._meta_set(
                        f"last_sync_cursor_{remote_url}",
                        pull_resp["next_cursor"],
                    )
                # Also mark synced_at on just-accepted events
                if push_result.get("accepted", 0) > 0:
                    self._meta_set(
                        f"last_sync_at_{remote_url}",
                        datetime.now(timezone.utc).isoformat(),
                    )
            else:
                result["pull"] = {"events_fetched": 0}

        return result

    def get_status(self, remote_url: Optional[str] = None) -> Dict[str, Any]:
        """Return sync status and statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM memory_events")
        total_events = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT device_id) FROM memory_events")
        device_count = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(timestamp) FROM memory_events")
        last_event_time = cursor.fetchone()[0]

        cursor.execute("""
            SELECT operation, COUNT(*) as cnt
            FROM memory_events
            GROUP BY operation
            ORDER BY cnt DESC
        """)
        operation_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute(
            "SELECT COUNT(*) FROM memory_events WHERE synced_at IS NOT NULL"
        )
        synced_count = cursor.fetchone()[0]

        result: Dict[str, Any] = {
            "device_id": self.device_id,
            "total_events": total_events,
            "device_count": device_count,
            "last_event_time": last_event_time,
            "operation_breakdown": operation_breakdown,
            "synced_events": synced_count,
        }

        if remote_url:
            result["remote"] = remote_url
            last_sync = self._meta_get(f"last_sync_at_{remote_url}")
            if last_sync:
                result["last_sync"] = last_sync
            try:
                remote_status = self.sync_with(remote_url, mode="pull")
                result["remote_status"] = remote_status
            except Exception as e:
                result["remote_error"] = str(e)

        return result
