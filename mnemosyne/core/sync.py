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
import hmac
import ipaddress
import logging
import sys
import uuid
import os
import base64
import binascii
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Any, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ENCRYPTED_WIRE_PREFIX = "mne1:"
_PRIVATE_METADATA_KEYS = {
    "session_id",
    "source_profile_session",
    "profile_session",
    "author_id",
    "author_type",
    "channel_id",
    "trust_tier",
}
_SYNC_PAYLOAD_FIELDS = (
    "content",
    "source",
    "importance",
    "metadata_json",
    "memory_type",
    "veracity",
    "valid_until",
)


def _parse_sync_timestamp(value: str) -> datetime:
    """Parse sync timestamps consistently across supported Python versions.

    Python 3.10's ``datetime.fromisoformat`` does not accept a trailing
    ``Z`` UTC designator, while newer versions do. Normalize it so conflict
    detection behaves the same on every CI Python.
    """
    if isinstance(value, str) and value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _encode_cursor(timestamp: str, event_id: str) -> str:
    """Return an opaque cursor that cannot skip equal-timestamp events."""
    return json.dumps(
        {"timestamp": timestamp, "event_id": event_id},
        separators=(",", ":"),
        sort_keys=True,
    )


def _decode_cursor(value: Optional[str]) -> Tuple[Optional[str], str]:
    """Decode a v2 cursor while accepting legacy timestamp-only cursors."""
    if not value:
        return None, ""
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return str(value), ""
    if isinstance(decoded, dict) and decoded.get("timestamp"):
        return str(decoded["timestamp"]), str(decoded.get("event_id") or "")
    return str(value), ""


def _event_sort_key(event: "SyncEvent") -> Tuple[datetime, str, str]:
    """Deterministic total order used for last-writer-wins resolution."""
    try:
        timestamp = _parse_sync_timestamp(event.timestamp)
    except (TypeError, ValueError):
        timestamp = datetime.min.replace(tzinfo=timezone.utc)
    return (
        timestamp,
        event.device_id or "",
        event.event_id or "",
    )


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
    surface_id: Optional[str] = None

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
            surface_id=row.get("surface_id"),
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
                            "key_source is neither a file path nor a valid "
                            "base64-encoded key"
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
        require_encryption: Optional[bool] = None,
        relay_mode: bool = False,
        surface_only: bool = False,
        surface_id: str = "shared-surface-v1",
        initialize_surface: bool = False,
        claim_surface_rows: bool = True,
        claim_existing_surface: bool = False,
        allow_unscoped_sync: bool = False,
        max_future_skew_seconds: int = 300,
        max_response_bytes: int = 10 * 1024 * 1024,
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
        self.surface_only = bool(surface_only)
        self.surface_id = str(surface_id).strip() if self.surface_only else None
        self.initialize_surface = bool(initialize_surface)
        self.claim_surface_rows = bool(claim_surface_rows)
        self.claim_existing_surface = bool(claim_existing_surface)
        self.allow_unscoped_sync = bool(allow_unscoped_sync)
        if self.surface_only and not self.surface_id:
            raise ValueError("surface_id is required in surface-only mode")
        self.surface_session_id = getattr(self._beam, "session_id", None) or "sync"
        self.encryption = encryption
        self.relay_mode = bool(relay_mode)
        self.require_encryption = (
            encryption is not None if require_encryption is None else bool(require_encryption)
        )
        if self.require_encryption and encryption is None and not self.relay_mode:
            raise ValueError("encryption is required but no encryption key is configured")
        if max_future_skew_seconds < 0:
            raise ValueError("max_future_skew_seconds must be non-negative")
        self.max_future_skew_seconds = int(max_future_skew_seconds)
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.max_response_bytes = int(max_response_bytes)

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

        if self.surface_only and not self.initialize_surface:
            meta_exists = self.conn.execute(
                """SELECT 1 FROM sqlite_master
                   WHERE type = 'table' AND name = 'sync_meta'"""
            ).fetchone()
            if meta_exists is None:
                raise ValueError(
                    "surface DB is not initialized; use explicit surface initialization"
                )
            marker = self.conn.execute(
                "SELECT value FROM sync_meta WHERE key = 'surface_db_id'"
            ).fetchone()
            if marker is None or marker[0] != self.surface_id:
                raise ValueError("surface DB marker is missing or does not match")

        self._init_events_table()
        if self.surface_only:
            try:
                self._init_surface_namespace()
            except Exception:
                self.conn.rollback()
                raise

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

        if self.surface_only:
            legacy_count = self.conn.execute(
                "SELECT COUNT(*) FROM memory_events WHERE surface_id IS NULL"
            ).fetchone()[0]
            if legacy_count:
                raise ValueError(
                    "surface-only sync found legacy unscoped events; migrate them explicitly"
                )
            foreign_count = self.conn.execute(
                "SELECT COUNT(*) FROM memory_events WHERE surface_id != ?",
                (self.surface_id,),
            ).fetchone()[0]
            if foreign_count:
                raise ValueError("surface-only sync requires a dedicated single-surface event log")
        if self.encryption is not None and self.require_encryption:
            self._migrate_legacy_encrypted_payloads()
        elif self.relay_mode and self.require_encryption:
            self._validate_relay_wire_payloads()

    def _init_surface_namespace(self) -> None:
        """Validate or explicitly initialize a durable shared-surface marker."""
        marker = self._meta_get("surface_db_id")
        initializing = marker is None
        if initializing:
            if not self.initialize_surface:
                raise ValueError(
                    "surface DB is not initialized; use explicit surface initialization"
                )
            existing_rows = self.conn.execute(
                "SELECT COUNT(*) FROM working_memory"
            ).fetchone()[0]
            if existing_rows and not self.claim_existing_surface:
                raise ValueError(
                    "first-time surface initialization requires an empty working_memory table"
                )
            if existing_rows:
                invalid_rows = self.conn.execute(
                    """SELECT COUNT(*) FROM working_memory
                       WHERE scope != 'global' OR session_id != ?""",
                    (self.surface_session_id,),
                ).fetchone()[0]
                if invalid_rows:
                    raise ValueError(
                        "existing surface migration found non-global or foreign-session rows"
                    )
        elif marker != self.surface_id:
            raise ValueError("surface DB marker does not match configured surface_id")

        columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(working_memory)").fetchall()
        }
        if "sync_surface_id" not in columns:
            if not self.initialize_surface:
                raise ValueError("surface DB schema has no durable row marker")
            self.conn.execute(
                "ALTER TABLE working_memory ADD COLUMN sync_surface_id TEXT"
            )
        if initializing:
            self.conn.execute(
                "INSERT INTO sync_meta (key, value) VALUES ('surface_db_id', ?)",
                (self.surface_id,),
            )
        if self.claim_surface_rows:
            self._claim_current_surface_rows()
        self._validate_surface_ownership()
        self.conn.commit()

    def _claim_current_surface_rows(self) -> None:
        """Claim new global rows written by the configured shared-surface Beam."""
        self.conn.execute(
            """UPDATE working_memory SET sync_surface_id = ?
               WHERE sync_surface_id IS NULL AND scope = 'global' AND session_id = ?""",
            (self.surface_id, self.surface_session_id),
        )

    def _validate_surface_ownership(self) -> None:
        """Require a physically dedicated working-memory surface."""
        unowned = self.conn.execute(
            """SELECT COUNT(*) FROM working_memory
               WHERE sync_surface_id IS NULL OR sync_surface_id != ?""",
            (self.surface_id,),
        ).fetchone()[0]
        if unowned:
            raise ValueError(
                "surface-only sync requires a dedicated DB with no unowned working rows"
            )

    def _migrate_legacy_encrypted_payloads(self) -> None:
        """Rewrap pre-mne1 events into the authenticated encrypted wire format."""
        if self.surface_only:
            rows = self.conn.execute(
                "SELECT * FROM memory_events WHERE payload IS NOT NULL AND surface_id = ?",
                (self.surface_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memory_events WHERE payload IS NOT NULL"
            ).fetchall()
        changed = False
        try:
            for row in rows:
                event = SyncEvent.from_row(dict(row))
                if self._payload_looks_encrypted(event.payload):
                    self._validate_encrypted_wire_payload(event.payload)
                    self._decode_payload(event)
                    continue
                try:
                    payload = json.loads(event.payload)
                except (TypeError, json.JSONDecodeError):
                    payload = self.encryption.decrypt(event.payload)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"legacy event {event.event_id} payload is not an object"
                    )
                envelope = {
                    "_sync": {
                        "version": 1,
                        "event_id": event.event_id,
                        "memory_id": event.memory_id,
                        "operation": event.operation,
                        "timestamp": event.timestamp,
                        "device_id": event.device_id,
                        "surface_id": event.surface_id,
                        "parent_event_ids": event.parent_event_ids,
                        "importance": event.importance,
                    },
                    "payload": payload,
                }
                event.payload = _ENCRYPTED_WIRE_PREFIX + self.encryption.encrypt(envelope)
                event.event_hash = self._compute_event_hash(event)
                self.conn.execute(
                    "UPDATE memory_events SET payload = ?, event_hash = ? WHERE event_id = ?",
                    (event.payload, event.event_hash, event.event_id),
                )
                changed = True
            if changed:
                self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def _validate_relay_wire_payloads(self) -> None:
        """Refuse startup when a blind relay contains legacy/plaintext events."""
        if self.surface_only:
            rows = self.conn.execute(
                """SELECT event_id, payload FROM memory_events
                   WHERE payload IS NOT NULL AND surface_id = ?""",
                (self.surface_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT event_id, payload FROM memory_events WHERE payload IS NOT NULL"
            ).fetchall()
        for row in rows:
            try:
                self._validate_encrypted_wire_payload(row["payload"])
            except ValueError as exc:
                raise ValueError(
                    f"relay event {row['event_id']} requires offline encrypted migration"
                ) from exc

    def _init_events_table(self) -> None:
        """Safely ensure the memory_events and sync_meta tables exist."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_events (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL,
                operation TEXT NOT NULL CHECK(operation IN ('CREATE','UPDATE','DELETE','CONSOLIDATE')),
                timestamp TEXT NOT NULL,
                timestamp_epoch REAL,
                device_id TEXT NOT NULL,
                surface_id TEXT,
                payload TEXT,
                parent_event_ids TEXT DEFAULT '[]',
                importance REAL DEFAULT 0.5,
                expiry TEXT,
                event_hash TEXT,
                synced_at TEXT,
                apply_state TEXT NOT NULL DEFAULT 'applied'
            )
        """)
        event_columns = {
            row[1] for row in cursor.execute("PRAGMA table_info(memory_events)").fetchall()
        }
        if "timestamp_epoch" not in event_columns:
            cursor.execute("ALTER TABLE memory_events ADD COLUMN timestamp_epoch REAL")
        if "surface_id" not in event_columns:
            cursor.execute("ALTER TABLE memory_events ADD COLUMN surface_id TEXT")
        if "apply_state" not in event_columns:
            cursor.execute(
                "ALTER TABLE memory_events ADD COLUMN apply_state "
                "TEXT NOT NULL DEFAULT 'applied'"
            )
        rows_without_epoch = cursor.execute(
            "SELECT event_id, timestamp FROM memory_events WHERE timestamp_epoch IS NULL"
        ).fetchall()
        for row in rows_without_epoch:
            try:
                epoch = _parse_sync_timestamp(row["timestamp"]).timestamp()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid stored sync timestamp for event {row['event_id']}"
                ) from exc
            cursor.execute(
                "UPDATE memory_events SET timestamp_epoch = ? WHERE event_id = ?",
                (epoch, row["event_id"]),
            )
        # Persist device identity and sync state across engine restarts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_outbox_ack (
                remote_url TEXT NOT NULL,
                event_id TEXT NOT NULL,
                acked_at TEXT NOT NULL,
                PRIMARY KEY (remote_url, event_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sync_memory_state (
                memory_id TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL DEFAULT '',
                last_operation TEXT NOT NULL DEFAULT 'CREATE',
                event_id TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        # Indices (IF NOT EXISTS is not supported for indices in all
        # SQLite versions, so we try/except)
        for index_ddl in [
            "CREATE INDEX IF NOT EXISTS idx_me_timestamp ON memory_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_me_epoch_event ON memory_events(timestamp_epoch, event_id)",
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
        commit: bool = True,
    ) -> SyncEvent:
        """Create and persist a sync event.

        This is the primary method to record memory mutations for
        replication to peers.
        """
        if operation not in ("CREATE", "UPDATE", "DELETE", "CONSOLIDATE"):
            raise ValueError(f"Invalid operation: {operation!r}")

        event_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        parent_ids_json = json.dumps(parent_event_ids or [])

        event = SyncEvent(
            event_id=event_id,
            memory_id=memory_id,
            operation=operation,
            timestamp=now,
            device_id=self.device_id,
            surface_id=self.surface_id,
            payload=None,
            parent_event_ids=parent_ids_json,
            importance=importance,
        )
        if payload is not None:
            payload = self._sanitize_sync_payload(payload)
            if self.encryption:
                envelope = {
                    "_sync": {
                        "version": 1,
                        "event_id": event.event_id,
                        "memory_id": event.memory_id,
                        "operation": event.operation,
                        "timestamp": event.timestamp,
                        "device_id": event.device_id,
                        "surface_id": event.surface_id,
                        "parent_event_ids": event.parent_event_ids,
                        "importance": event.importance,
                    },
                    "payload": payload,
                }
                event.payload = _ENCRYPTED_WIRE_PREFIX + self.encryption.encrypt(envelope)
            else:
                event.payload = json.dumps(payload, default=str)
        event.event_hash = self._compute_event_hash(event)

        cursor = self.conn.cursor()
        cursor.execute(
            """INSERT INTO memory_events (
                event_id, memory_id, operation, timestamp, timestamp_epoch, device_id,
                surface_id, payload, parent_event_ids, importance, expiry, event_hash, apply_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'applied')""",
            (
                event.event_id,
                event.memory_id,
                event.operation,
                event.timestamp,
                _parse_sync_timestamp(event.timestamp).timestamp(),
                event.device_id,
                event.surface_id,
                event.payload,
                event.parent_event_ids,
                event.importance,
                event.expiry,
                event.event_hash,
            ),
        )
        if commit:
            self.conn.commit()

        logger.debug(
            "Logged sync event %s: %s %s", event_id, operation, memory_id
        )
        return event

    @staticmethod
    def _scrub_private_metadata(value: Any) -> Any:
        if isinstance(value, dict):
            scrubbed = {}
            for key, item in value.items():
                normalized = str(key).lower()
                if (
                    normalized in _PRIVATE_METADATA_KEYS
                    or normalized.endswith("_session_id")
                    or normalized.startswith("author_")
                ):
                    continue
                scrubbed[key] = SyncEngine._scrub_private_metadata(item)
            return scrubbed
        if isinstance(value, list):
            return [SyncEngine._scrub_private_metadata(item) for item in value]
        return value

    @classmethod
    def _sanitize_sync_payload(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = {*_SYNC_PAYLOAD_FIELDS, "deleted"}
        sanitized = {key: value for key, value in payload.items() if key in allowed}
        metadata = sanitized.get("metadata_json")
        parsed = metadata
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, (dict, list)):
            sanitized["metadata_json"] = json.dumps(
                cls._scrub_private_metadata(parsed),
                sort_keys=True,
                separators=(",", ":"),
            )
        return sanitized

    @staticmethod
    def _canonical_payload(payload: Dict[str, Any]) -> str:
        """Canonical JSON used to detect semantic row changes."""
        normalized = {
            key: payload[key] for key in _SYNC_PAYLOAD_FIELDS if key in payload
        }
        metadata = normalized.get("metadata_json")
        if isinstance(metadata, str):
            try:
                normalized["metadata_json"] = json.loads(metadata)
            except json.JSONDecodeError:
                pass
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)

    @classmethod
    def _payload_fingerprint(cls, payload: Dict[str, Any]) -> str:
        return hashlib.sha256(cls._canonical_payload(payload).encode("utf-8")).hexdigest()

    def _working_columns(self) -> List[str]:
        cached = getattr(self, "_sync_working_columns", None)
        if cached is not None:
            return cached
        available = {
            row[1] for row in self.conn.execute("PRAGMA table_info(working_memory)").fetchall()
        }
        wanted = ["id", *_SYNC_PAYLOAD_FIELDS]
        columns = [column for column in wanted if column in available]
        if "id" not in columns or "content" not in columns:
            raise RuntimeError("working_memory schema is missing id/content")
        self._sync_working_columns = columns
        return columns

    @classmethod
    def _row_to_payload(cls, row: Any) -> Tuple[str, Dict[str, Any]]:
        values = dict(row)
        memory_id = str(values.pop("id"))
        return memory_id, cls._sanitize_sync_payload(values)

    def _working_payloads(self) -> Dict[str, Dict[str, Any]]:
        """Read every working-memory row without a fixed bootstrap cap."""
        columns = self._working_columns()
        if self.surface_only:
            rows = self.conn.execute(
                f"""SELECT {', '.join(columns)} FROM main.working_memory
                    WHERE sync_surface_id = ? ORDER BY id""",
                (self.surface_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                f"SELECT {', '.join(columns)} FROM main.working_memory ORDER BY id"
            ).fetchall()
        return dict(self._row_to_payload(row) for row in rows)

    def _working_payload(self, memory_id: str) -> Optional[Dict[str, Any]]:
        columns = self._working_columns()
        if self.surface_only:
            row = self.conn.execute(
                f"""SELECT {', '.join(columns)} FROM main.working_memory
                    WHERE id = ? AND sync_surface_id = ?""",
                (memory_id, self.surface_id),
            ).fetchone()
        else:
            row = self.conn.execute(
                f"SELECT {', '.join(columns)} FROM main.working_memory WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return self._row_to_payload(row)[1] if row else None

    def _state_set(
        self,
        memory_id: str,
        fingerprint: str,
        operation: str,
        event_id: Optional[str],
        timestamp: Optional[str] = None,
    ) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO sync_memory_state
               (memory_id, fingerprint, last_operation, event_id, timestamp)
               VALUES (?, ?, ?, ?, ?)""",
            (
                memory_id,
                fingerprint,
                operation,
                event_id,
                timestamp or datetime.now(timezone.utc).isoformat(),
            ),
        )

    def discover_local_mutations(self) -> Dict[str, Any]:
        """Discover mutations in one transaction, rolling back on any failure."""
        try:
            if self.conn.in_transaction:
                raise RuntimeError("mutation discovery requires a clean SQLite transaction")
            self.conn.execute("BEGIN IMMEDIATE")
            stats = self._discover_local_mutations()
            self.conn.commit()
            return stats
        except Exception:
            self.conn.rollback()
            raise

    def _discover_local_mutations(self) -> Dict[str, Any]:
        """Snapshot working memory and emit CREATE/UPDATE/DELETE events.

        All writers (Hermes, Codex MCP, CLI, direct Beam calls) converge through
        the SQLite table, so snapshot comparison is more reliable than requiring
        every caller to install a mutation hook. State is updated when an event
        is durably logged; failed network pushes leave that event in the outbox.
        """
        if self.surface_only:
            self._claim_current_surface_rows()
            self._validate_surface_ownership()
        current = self._working_payloads()
        state_rows = self.conn.execute(
            "SELECT memory_id, fingerprint, last_operation, event_id FROM sync_memory_state"
        ).fetchall()
        state = {row["memory_id"]: dict(row) for row in state_rows}
        stats: Dict[str, Any] = {
            "created": 0,
            "updated": 0,
            "deleted": 0,
            "events": [],
        }

        # Upgrade/bootstrap recovery: the legacy event log may contain the last
        # known row while sync_memory_state is still empty. Seed that shadow, or
        # emit the missing tombstone when the row was deleted before upgrade.
        latest_applied: Dict[str, SyncEvent] = {}
        if self.surface_only:
            applied_rows = self.conn.execute(
                """SELECT * FROM memory_events
                   WHERE apply_state = 'applied' AND surface_id = ?""",
                (self.surface_id,),
            ).fetchall()
        else:
            applied_rows = self.conn.execute(
                "SELECT * FROM memory_events WHERE apply_state = 'applied'"
            ).fetchall()
        for row in applied_rows:
            candidate = SyncEvent.from_row(dict(row))
            previous_event = latest_applied.get(candidate.memory_id)
            if previous_event is None or _event_sort_key(candidate) > _event_sort_key(previous_event):
                latest_applied[candidate.memory_id] = candidate
        for memory_id, latest_event in latest_applied.items():
            if memory_id in state or memory_id in current:
                continue
            if latest_event.operation == "DELETE":
                self._state_set(
                    memory_id,
                    "",
                    "DELETE",
                    latest_event.event_id,
                    latest_event.timestamp,
                )
                state[memory_id] = {
                    "memory_id": memory_id,
                    "fingerprint": "",
                    "last_operation": "DELETE",
                    "event_id": latest_event.event_id,
                }
                continue
            tombstone = self.log_event(
                memory_id=memory_id,
                operation="DELETE",
                payload={"deleted": True},
                parent_event_ids=[latest_event.event_id],
                commit=False,
            )
            self._state_set(
                memory_id, "", "DELETE", tombstone.event_id, tombstone.timestamp
            )
            state[memory_id] = {
                "memory_id": memory_id,
                "fingerprint": "",
                "last_operation": "DELETE",
                "event_id": tombstone.event_id,
            }
            stats["deleted"] += 1
            stats["events"].append(tombstone.to_dict())

        for memory_id, payload in current.items():
            fingerprint = self._payload_fingerprint(payload)
            previous = state.get(memory_id)
            if previous is not None and previous["fingerprint"] == fingerprint \
                    and previous["last_operation"] != "DELETE":
                continue

            latest = self._latest_event(memory_id)
            if latest is not None and latest.operation != "DELETE":
                try:
                    latest_payload = self._decode_payload(latest)
                except Exception:
                    latest_payload = None
                if latest_payload is not None and self._payload_fingerprint(latest_payload) == fingerprint:
                    self._state_set(
                        memory_id,
                        fingerprint,
                        latest.operation,
                        latest.event_id,
                        latest.timestamp,
                    )
                    continue

            if previous is None and latest is None:
                operation = "CREATE"
            elif previous is not None and previous["last_operation"] == "DELETE":
                operation = "CREATE"
            else:
                operation = "UPDATE"
            parent_event_id = (
                previous.get("event_id") if previous else (latest.event_id if latest else None)
            )
            parent_ids = [parent_event_id] if parent_event_id else []
            event = self.log_event(
                memory_id=memory_id,
                operation=operation,
                payload=payload,
                importance=float(payload.get("importance") or 0.5),
                parent_event_ids=parent_ids,
                commit=False,
            )
            self._state_set(memory_id, fingerprint, operation, event.event_id, event.timestamp)
            stats["created" if operation == "CREATE" else "updated"] += 1
            stats["events"].append(event.to_dict())

        for memory_id, previous in state.items():
            if memory_id in current or previous["last_operation"] == "DELETE":
                continue
            latest = self._latest_event(memory_id)
            if latest is not None and latest.operation == "DELETE":
                self._state_set(memory_id, "", "DELETE", latest.event_id, latest.timestamp)
                continue
            parent_ids = [previous["event_id"]] if previous.get("event_id") else []
            event = self.log_event(
                memory_id=memory_id,
                operation="DELETE",
                payload={"deleted": True},
                parent_event_ids=parent_ids,
                commit=False,
            )
            self._state_set(memory_id, "", "DELETE", event.event_id, event.timestamp)
            stats["deleted"] += 1
            stats["events"].append(event.to_dict())

        return stats

    def _find_unlogged_memories(self, limit: int = 5000) -> List[dict]:
        """Backward-compatible wrapper around full mutation discovery.

        ``limit`` is intentionally ignored: a fixed cap caused databases larger
        than 5000 rows to become permanently only partially synchronized.
        """
        del limit
        return self.discover_local_mutations()["events"]

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
        since_timestamp, since_event_id = _decode_cursor(since_cursor)
        clauses: List[str] = ["apply_state IN ('applied', 'relayed')"]
        params: List[Any] = []
        if self.surface_only:
            clauses.append("surface_id = ?")
            params.append(self.surface_id)

        if since_timestamp:
            since_epoch = _parse_sync_timestamp(since_timestamp).timestamp()
            clauses.append(
                "(timestamp_epoch > ? OR (timestamp_epoch = ? AND event_id > ?))"
            )
            params.extend([since_epoch, since_epoch, since_event_id])
        if device_id:
            clauses.append("device_id != ?")
            params.append(device_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit + 1)
        cursor.execute(
            f"""SELECT * FROM memory_events
                {where}
                ORDER BY timestamp_epoch ASC, event_id ASC
                LIMIT ?""",
            params,
        )

        rows = cursor.fetchall()
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        events = [SyncEvent.from_row(dict(row)) for row in rows]
        next_cursor = (
            _encode_cursor(events[-1].timestamp, events[-1].event_id)
            if events else since_cursor
        )

        return {
            "events": [ev.to_dict() for ev in events],
            "next_cursor": next_cursor,
            "has_more": has_more,
            "total": len(events),
        }

    @staticmethod
    def _payload_looks_encrypted(payload: str) -> bool:
        return isinstance(payload, str) and payload.startswith(_ENCRYPTED_WIRE_PREFIX)

    @staticmethod
    def _validate_encrypted_wire_payload(payload: str) -> str:
        if not SyncEngine._payload_looks_encrypted(payload):
            raise ValueError("encrypted sync payload is missing the mne1 wire marker")
        token = payload[len(_ENCRYPTED_WIRE_PREFIX):]
        if not token:
            raise ValueError("encrypted sync payload is empty")
        try:
            padding = "=" * (-len(token) % 4)
            decoded = base64.b64decode(
                (token + padding).encode("ascii"), altchars=b"-_", validate=True
            )
        except (ValueError, UnicodeEncodeError, binascii.Error) as exc:
            raise ValueError("encrypted sync payload is not valid base64") from exc
        if len(decoded) < 40:
            raise ValueError("encrypted sync payload is too short")
        return token

    def _decode_payload(self, event: SyncEvent) -> Optional[Dict[str, Any]]:
        if not event.payload:
            if self.require_encryption:
                raise ValueError("encrypted sync requires an authenticated payload")
            if event.operation == "DELETE":
                return {}
            raise ValueError("CREATE/UPDATE sync events require a payload")
        encrypted = self._payload_looks_encrypted(event.payload)
        if encrypted:
            token = self._validate_encrypted_wire_payload(event.payload)
            if self.encryption is None:
                if self.relay_mode:
                    return None
                raise ValueError("encrypted payload received without a decryption key")
            decoded = self.encryption.decrypt(token)
            if not isinstance(decoded, dict):
                raise ValueError("decrypted sync payload is not an object")
            envelope_meta = decoded.get("_sync")
            envelope_payload = decoded.get("payload")
            if not isinstance(envelope_meta, dict) or not isinstance(envelope_payload, dict):
                if self.require_encryption:
                    raise ValueError("encrypted payload has no authenticated sync envelope")
                return decoded
            expected_meta = {
                "version": 1,
                "event_id": event.event_id,
                "memory_id": event.memory_id,
                "operation": event.operation,
                "timestamp": event.timestamp,
                "device_id": event.device_id,
                "surface_id": event.surface_id,
                "parent_event_ids": event.parent_event_ids,
                "importance": event.importance,
            }
            if envelope_meta != expected_meta:
                raise ValueError("authenticated sync metadata does not match the event")
            return envelope_payload
        if self.require_encryption:
            raise ValueError("plaintext payload rejected because encryption is required")
        decoded = json.loads(event.payload)
        if not isinstance(decoded, dict):
            raise ValueError("sync payload is not an object")
        return decoded

    def _latest_event(
        self, memory_id: str, exclude_event_id: Optional[str] = None
    ) -> Optional[SyncEvent]:
        clauses = ["memory_id = ?"]
        params: List[Any] = [memory_id]
        if self.surface_only:
            clauses.append("surface_id = ?")
            params.append(self.surface_id)
        if exclude_event_id:
            clauses.append("event_id != ?")
            params.append(exclude_event_id)
        rows = self.conn.execute(
            f"SELECT * FROM memory_events WHERE {' AND '.join(clauses)}",
            params,
        ).fetchall()
        events = [SyncEvent.from_row(dict(row)) for row in rows]
        return max(events, key=_event_sort_key) if events else None

    def _insert_incoming_event(self, event: SyncEvent, apply_state: str) -> None:
        self.conn.execute(
            """INSERT OR IGNORE INTO memory_events (
                event_id, memory_id, operation, timestamp, timestamp_epoch, device_id,
                surface_id, payload, parent_event_ids, importance, expiry, event_hash, synced_at,
                apply_state
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event.event_id, event.memory_id, event.operation, event.timestamp,
                _parse_sync_timestamp(event.timestamp).timestamp(),
                event.device_id, event.surface_id, event.payload, event.parent_event_ids,
                event.importance, event.expiry, event.event_hash,
                datetime.now(timezone.utc).isoformat(), apply_state,
            ),
        )

    def _mark_event_state(self, event_id: str, apply_state: str) -> None:
        self.conn.execute(
            "UPDATE memory_events SET apply_state = ? WHERE event_id = ?",
            (apply_state, event_id),
        )

    def _prepare_embedding(self, payload: Optional[Dict[str, Any]]) -> Optional[Any]:
        """Compute derived vector state before opening the SQLite write transaction."""
        if not payload or not payload.get("content"):
            return None
        try:
            from mnemosyne.core.beam import _embeddings

            if not _embeddings.available():
                return None
            vectors = _embeddings.embed([str(payload["content"])])
            return vectors[0] if vectors is not None and len(vectors) else None
        except Exception as exc:
            logger.warning("sync embedding preparation failed: %s", exc)
            return None

    def _apply_memory_event(
        self,
        event: SyncEvent,
        payload: Optional[Dict[str, Any]],
        embedding: Optional[Any] = None,
    ) -> None:
        """Materialize one event without committing the active transaction."""
        if payload is None:  # blind relay: never materialize opaque operations
            return

        if event.operation == "DELETE":
            if self.surface_only:
                target = self.conn.execute(
                    """SELECT id FROM main.working_memory
                       WHERE id = ? AND sync_surface_id = ?""",
                    (event.memory_id, self.surface_id),
                ).fetchone()
            else:
                target = self.conn.execute(
                    "SELECT id FROM main.working_memory WHERE id = ?",
                    (event.memory_id,),
                ).fetchone()
            if target is not None:
                try:
                    from mnemosyne.core.beam import _wm_vec_delete

                    _wm_vec_delete(self.conn, event.memory_id)
                except Exception:
                    pass
                if self.surface_only:
                    self.conn.execute(
                        """DELETE FROM main.working_memory
                           WHERE id = ? AND sync_surface_id = ?""",
                        (event.memory_id, self.surface_id),
                    )
                else:
                    self.conn.execute(
                        "DELETE FROM main.working_memory WHERE id = ?", (event.memory_id,)
                    )
                self.conn.execute(
                    "DELETE FROM annotations WHERE memory_id = ?", (event.memory_id,)
                )
                self.conn.execute(
                    "DELETE FROM memory_embeddings WHERE memory_id = ?", (event.memory_id,)
                )
            self._state_set(event.memory_id, "", "DELETE", event.event_id, event.timestamp)
            return

        content = str(payload.get("content") or "")
        if not content:
            raise ValueError(f"event {event.event_id} has no content")
        importance = float(payload.get("importance", event.importance or 0.5))
        source = str(payload.get("source") or "sync")
        metadata = payload.get("metadata_json")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw": metadata}
        if metadata is not None and not isinstance(metadata, dict):
            metadata = {"value": metadata}

        available = {
            row[1] for row in self.conn.execute("PRAGMA table_info(working_memory)").fetchall()
        }
        values: Dict[str, Any] = {
            "content": content,
            "source": source,
            "timestamp": event.timestamp,
            "importance": importance,
            "metadata_json": json.dumps(metadata or {}, sort_keys=True),
            "memory_type": payload.get("memory_type"),
            "veracity": payload.get("veracity") or "unknown",
            "valid_until": payload.get("valid_until"),
            "scope": "global",
            "sync_surface_id": self.surface_id,
        }
        if self.surface_only:
            exists = self.conn.execute(
                """SELECT 1 FROM main.working_memory
                   WHERE id = ? AND sync_surface_id = ?""",
                (event.memory_id, self.surface_id),
            ).fetchone() is not None
        else:
            exists = self.conn.execute(
                "SELECT 1 FROM main.working_memory WHERE id = ?", (event.memory_id,)
            ).fetchone() is not None
        if exists:
            try:
                from mnemosyne.core.beam import _wm_vec_delete

                _wm_vec_delete(self.conn, event.memory_id)
            except Exception:
                pass
            present = {
                key: value
                for key, value in values.items()
                if key in available and value is not None
            }
            assignments = ", ".join(f"{key} = ?" for key in present)
            if self.surface_only:
                self.conn.execute(
                    f"""UPDATE main.working_memory SET {assignments}
                        WHERE id = ? AND sync_surface_id = ?""",
                    [*present.values(), event.memory_id, self.surface_id],
                )
            else:
                self.conn.execute(
                    f"UPDATE main.working_memory SET {assignments} WHERE id = ?",
                    [*present.values(), event.memory_id],
                )
            self.conn.execute(
                "DELETE FROM memory_embeddings WHERE memory_id = ?", (event.memory_id,)
            )
        else:
            insert_values: Dict[str, Any] = {
                "id": event.memory_id,
                **values,
                "session_id": self.surface_session_id,
            }
            present = {
                key: value
                for key, value in insert_values.items()
                if key in available and value is not None
            }
            columns = ", ".join(present)
            placeholders = ", ".join("?" for _ in present)
            self.conn.execute(
                f"INSERT INTO main.working_memory ({columns}) VALUES ({placeholders})",
                list(present.values()),
            )

        if embedding is not None:
            try:
                from mnemosyne.core.beam import _store_working_embedding

                _store_working_embedding(
                    self.conn, event.memory_id, embedding, commit_vec=False
                )
            except Exception as exc:
                logger.warning("sync embedding storage failed for %s: %s", event.memory_id, exc)

        normalized_payload = self._working_payload(event.memory_id) or payload
        fingerprint = self._payload_fingerprint(normalized_payload)
        self._state_set(
            event.memory_id, fingerprint, event.operation, event.event_id, event.timestamp
        )

    def push_changes(self, events: List[dict]) -> Dict[str, Any]:
        """Validate, order, deduplicate, and restart-safely apply events."""
        if self.surface_only:
            self._validate_surface_ownership()
        stats: Dict[str, Any] = {
            "accepted": 0,
            "duplicates": 0,
            "conflicts": 0,
            "errors": 0,
            "details": [],
            "acknowledged_event_ids": [],
        }
        if self.surface_only:
            known_query = self.conn.execute(
                "SELECT * FROM memory_events WHERE surface_id = ?", (self.surface_id,)
            ).fetchall()
        else:
            known_query = self.conn.execute("SELECT * FROM memory_events").fetchall()
        known_rows = {row["event_id"]: dict(row) for row in known_query}
        known_states = {
            event_id: row["apply_state"] for event_id, row in known_rows.items()
        }

        incoming: List[Tuple[SyncEvent, bool]] = []
        scheduled_ids = set()
        maximum_timestamp = datetime.now(timezone.utc) + timedelta(
            seconds=self.max_future_skew_seconds
        )
        for raw in events:
            try:
                if not isinstance(raw, dict):
                    raise ValueError("event must be an object")
                if (
                    self.relay_mode
                    and self.require_encryption
                    and raw.get("_transport_authenticated") is not True
                ):
                    raise ValueError("blind relay requires authenticated transport")
                event = SyncEvent.from_dict(raw)
                if not event.event_id or not event.memory_id or not event.device_id:
                    raise ValueError("event_id, memory_id, and device_id are required")
                if self.surface_only and event.surface_id != self.surface_id:
                    raise ValueError("event belongs to a different or unscoped surface")
                if event.operation not in {"CREATE", "UPDATE", "DELETE", "CONSOLIDATE"}:
                    raise ValueError(f"invalid operation: {event.operation!r}")
                event_timestamp = _parse_sync_timestamp(event.timestamp)
                if event_timestamp > maximum_timestamp:
                    raise ValueError("event timestamp exceeds allowed future clock skew")
                provided_hash = str(event.event_hash or "")
                computed_hash = self._compute_event_hash(event)
                if (
                    len(provided_hash) == 64
                    and all(char in "0123456789abcdefABCDEF" for char in provided_hash)
                    and provided_hash.lower() != computed_hash
                ):
                    raise ValueError("event_hash does not match event contents")
                event.event_hash = computed_hash
                if event.event_id in scheduled_ids:
                    continue
                scheduled_ids.add(event.event_id)
                existing_state = known_states.get(event.event_id)
                if existing_state is not None and existing_state != "pending":
                    stored_event = SyncEvent.from_row(known_rows[event.event_id])
                    if stored_event.to_dict() != event.to_dict():
                        raise ValueError("duplicate event_id does not match stored event")
                    stats["duplicates"] += 1
                    stats["acknowledged_event_ids"].append(event.event_id)
                    continue
                if existing_state == "pending":
                    stored_event = SyncEvent.from_row(known_rows[event.event_id])
                    if stored_event.to_dict() != event.to_dict():
                        raise ValueError("pending retry does not match the stored event")
                    event = stored_event
                incoming.append((event, existing_state == "pending"))
            except Exception as exc:
                stats["errors"] += 1
                stats["details"].append(f"invalid event: {exc}")

        incoming.sort(key=lambda item: _event_sort_key(item[0]))
        total = len(incoming)
        progress_interval = max(1, total // 50) if total > 100 else 100
        for index, (event, retry_pending) in enumerate(incoming):
            if total > 100 and index > 0 and index % progress_interval == 0:
                pct = int(index / total * 100)
                sys.stderr.write(f"\r  Progress: {index}/{total} ({pct}%)  \r")
                sys.stderr.flush()
            try:
                payload = self._decode_payload(event)
                if payload is not None:
                    payload = self._sanitize_sync_payload(payload)
                embedding = self._prepare_embedding(payload)
                if self.conn.in_transaction:
                    raise RuntimeError("sync apply requires a clean SQLite transaction")
                self.conn.execute("BEGIN IMMEDIATE")
                latest = self._latest_event(
                    event.memory_id,
                    exclude_event_id=event.event_id if retry_pending else None,
                )
                if latest is not None and _event_sort_key(latest) >= _event_sort_key(event):
                    if retry_pending:
                        self._mark_event_state(event.event_id, "conflict")
                    else:
                        self._insert_incoming_event(event, "conflict")
                    self.conn.commit()
                    stats["conflicts"] += 1
                    stats["acknowledged_event_ids"].append(event.event_id)
                    continue

                if payload is None:
                    if retry_pending:
                        self._mark_event_state(event.event_id, "relayed")
                    else:
                        self._insert_incoming_event(event, "relayed")
                    self.conn.commit()
                    stats["accepted"] += 1
                    stats["acknowledged_event_ids"].append(event.event_id)
                    continue

                if not retry_pending:
                    self._insert_incoming_event(event, "pending")
                self._apply_memory_event(event, payload, embedding=embedding)
                self._mark_event_state(event.event_id, "applied")
                self.conn.commit()
                stats["accepted"] += 1
                stats["acknowledged_event_ids"].append(event.event_id)
            except KeyboardInterrupt:
                self.conn.rollback()
                stats["interrupted"] = True
                break
            except Exception as exc:
                self.conn.rollback()
                stats["errors"] += 1
                stats["details"].append(f"event {event.event_id}: {exc}")
                logger.warning("Failed to apply event %s: %s", event.event_id, exc)

        return stats

    def _read_bounded_http_response(self, response: Any) -> bytes:
        """Read an HTTP response without allowing an unbounded allocation."""
        headers = getattr(response, "headers", None)
        content_length = headers.get("Content-Length") if headers is not None else None
        if content_length is not None:
            try:
                if int(content_length) > self.max_response_bytes:
                    raise ValueError("remote response exceeds configured size limit")
            except ValueError as exc:
                if "exceeds" in str(exc):
                    raise
                raise ValueError("remote response has invalid Content-Length") from exc
        data = response.read(self.max_response_bytes + 1)
        if len(data) > self.max_response_bytes:
            raise ValueError("remote response exceeds configured size limit")
        return data

    @staticmethod
    def _validate_sync_remote_url(remote_url: str) -> None:
        parsed = urlparse(remote_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("sync remote must be an absolute HTTP(S) URL")
        if parsed.scheme == "https":
            return
        hostname = parsed.hostname.lower()
        loopback = hostname == "localhost"
        if not loopback:
            try:
                loopback = ipaddress.ip_address(hostname).is_loopback
            except ValueError:
                loopback = False
        if not loopback:
            raise ValueError("non-loopback sync remotes require HTTPS")

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

        del encryption_key  # retained for API compatibility; engine owns encryption state
        if mode not in {"push", "pull", "bidirectional"}:
            raise ValueError(f"unsupported sync mode: {mode}")
        if not self.surface_only and not self.allow_unscoped_sync:
            raise ValueError(
                "network sync requires surface_only=True; private/unscoped DB sync is disabled"
            )
        remote_url = remote_url.rstrip("/")
        self._validate_sync_remote_url(remote_url)

        result: Dict[str, Any] = {
            "remote": remote_url,
            "mode": mode,
            "push": None,
            "pull": None,
            "errors": [],
        }
        configured_remote: Optional[str] = None
        if mode in ("push", "bidirectional"):
            configured_remote = self._meta_get("configured_push_remote")
            ack_remotes = {
                row[0]
                for row in self.conn.execute(
                    "SELECT DISTINCT remote_url FROM sync_outbox_ack"
                ).fetchall()
            }
            if not configured_remote and ack_remotes:
                if ack_remotes != {remote_url}:
                    result["errors"].append(
                        "existing outbox acknowledgements belong to a different relay"
                    )
                    return result
                configured_remote = remote_url
                self._meta_set("configured_push_remote", remote_url)
            if configured_remote and configured_remote != remote_url:
                result["errors"].append(
                    "this sync database is pinned to a different push relay"
                )
                return result
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        def _post(endpoint: str, body: dict) -> Optional[dict]:
            url = f"{remote_url.rstrip('/')}{endpoint}"
            data = json.dumps(body, default=str).encode("utf-8")
            request_headers = dict(headers)
            if api_key:
                request_headers["X-Mnemosyne-Body-MAC"] = hmac.new(
                    api_key.encode("utf-8"), data, hashlib.sha256
                ).hexdigest()
            request = _request.Request(
                url, data=data, headers=request_headers, method="POST"
            )
            try:
                with _request.urlopen(request, timeout=30) as response:
                    return json.loads(
                        self._read_bounded_http_response(response).decode("utf-8")
                    )
            except _error.HTTPError as exc:
                body_text = (
                    self._read_bounded_http_response(exc).decode(
                        "utf-8", errors="replace"
                    )
                    if exc.fp
                    else str(exc)
                )
                result["errors"].append(f"HTTP {exc.code} on {endpoint}: {body_text}")
            except Exception as exc:
                result["errors"].append(f"{endpoint}: {exc}")
            return None

        page_size = 1000
        discovered = self.discover_local_mutations()
        if mode in ("push", "bidirectional"):
            push_total = {
                "accepted": 0,
                "duplicates": 0,
                "conflicts": 0,
                "errors": 0,
                "batches": 0,
                "discovered": {
                    key: discovered[key] for key in ("created", "updated", "deleted")
                },
            }
            while True:
                if self.surface_only:
                    rows = self.conn.execute(
                        """SELECT me.* FROM memory_events AS me
                           WHERE me.device_id = ? AND me.surface_id = ?
                             AND NOT EXISTS (
                                SELECT 1 FROM sync_outbox_ack AS ack
                                WHERE ack.remote_url = ? AND ack.event_id = me.event_id
                             )
                           ORDER BY me.timestamp_epoch ASC, me.event_id ASC LIMIT ?""",
                        (self.device_id, self.surface_id, remote_url, page_size),
                    ).fetchall()
                else:
                    rows = self.conn.execute(
                        """SELECT me.* FROM memory_events AS me
                           WHERE me.device_id = ?
                             AND NOT EXISTS (
                                SELECT 1 FROM sync_outbox_ack AS ack
                                WHERE ack.remote_url = ? AND ack.event_id = me.event_id
                             )
                           ORDER BY me.timestamp_epoch ASC, me.event_id ASC LIMIT ?""",
                        (self.device_id, remote_url, page_size),
                    ).fetchall()
                if not rows:
                    break
                batch = [SyncEvent.from_row(dict(row)).to_dict() for row in rows]
                response = _post(
                    "/sync/push", {"events": batch, "device_id": self.device_id}
                )
                if response is None:
                    break
                push_total["batches"] += 1
                for key in ("accepted", "duplicates", "conflicts", "errors"):
                    push_total[key] += int(response.get(key, 0) or 0)

                batch_ids = {event["event_id"] for event in batch}
                acknowledged_raw = list(response.get("acknowledged_event_ids") or [])
                unknown_acknowledgements = set(acknowledged_raw) - batch_ids
                if unknown_acknowledgements:
                    result["errors"].append(
                        "remote acknowledged event IDs outside the current push batch"
                    )
                    break
                acknowledged_set = set(acknowledged_raw)
                if not acknowledged_set:
                    result["errors"].append("remote did not acknowledge pushed events")
                    break
                acknowledged = [
                    event["event_id"]
                    for event in batch
                    if event["event_id"] in acknowledged_set
                ]
                now = datetime.now(timezone.utc).isoformat()
                if configured_remote is None:
                    self.conn.execute(
                        """INSERT OR REPLACE INTO sync_meta (key, value)
                           VALUES ('configured_push_remote', ?)""",
                        (remote_url,),
                    )
                    configured_remote = remote_url
                self.conn.executemany(
                    """INSERT OR REPLACE INTO sync_outbox_ack
                       (remote_url, event_id, acked_at) VALUES (?, ?, ?)""",
                    [(remote_url, event_id, now) for event_id in acknowledged],
                )
                self.conn.commit()
                if len(acknowledged) < len(batch):
                    result["errors"].append("remote only partially acknowledged a push batch")
                    break
            result["push"] = push_total
            if not result["errors"]:
                self._meta_set("configured_push_remote", remote_url)

        if mode in ("pull", "bidirectional"):
            cursor_key = f"last_pull_cursor_{remote_url}"
            pull_cursor = self._meta_get(cursor_key)
            pull_total = {
                "events_fetched": 0,
                "accepted": 0,
                "duplicates": 0,
                "conflicts": 0,
                "errors": 0,
                "batches": 0,
            }
            while True:
                response = _post(
                    "/sync/pull",
                    {
                        "since": pull_cursor,
                        "device_id": self.device_id,
                        "limit": page_size,
                    },
                )
                if response is None:
                    break
                events = list(response.get("events") or [])
                if events:
                    authenticated_events = []
                    for event in events:
                        if not isinstance(event, dict):
                            result["errors"].append(
                                "pull page contains a non-object event"
                            )
                            authenticated_events = []
                            break
                        authenticated = dict(event)
                        authenticated["_transport_authenticated"] = True
                        authenticated_events.append(authenticated)
                    if not authenticated_events:
                        break
                    applied = self.push_changes(authenticated_events)
                    pull_total["batches"] += 1
                    pull_total["events_fetched"] += len(events)
                    for key in ("accepted", "duplicates", "conflicts", "errors"):
                        pull_total[key] += int(applied.get(key, 0) or 0)
                    if applied.get("interrupted"):
                        result["interrupted"] = True
                        break
                    if int(applied.get("errors", 0) or 0) > 0:
                        result["errors"].append("pull page failed validation; cursor not advanced")
                        break
                next_cursor = response.get("next_cursor")
                if next_cursor:
                    pull_cursor = str(next_cursor)
                    self._meta_set(cursor_key, pull_cursor)
                if not response.get("has_more"):
                    break
                if not events:
                    result["errors"].append("remote reported has_more without returning events")
                    break
            result["pull"] = pull_total

        if not result["errors"]:
            self._meta_set(
                f"last_sync_at_{remote_url}", datetime.now(timezone.utc).isoformat()
            )
        return result

    def get_status(
        self,
        remote_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return local sync statistics and optionally read remote status."""
        cursor = self.conn.cursor()

        where = " WHERE surface_id = ?" if self.surface_only else ""
        params: Tuple[Any, ...] = (self.surface_id,) if self.surface_only else ()

        cursor.execute(f"SELECT COUNT(*) FROM memory_events{where}", params)
        total_events = cursor.fetchone()[0]

        cursor.execute(
            f"SELECT COUNT(DISTINCT device_id) FROM memory_events{where}", params
        )
        device_count = cursor.fetchone()[0]

        cursor.execute(f"SELECT MAX(timestamp) FROM memory_events{where}", params)
        last_event_time = cursor.fetchone()[0]

        cursor.execute(
            f"""SELECT operation, COUNT(*) as cnt
                FROM memory_events{where}
                GROUP BY operation
                ORDER BY cnt DESC""",
            params,
        )
        operation_breakdown = {row[0]: row[1] for row in cursor.fetchall()}

        configured_remote = self._meta_get("configured_push_remote")
        if configured_remote:
            ack_where = " AND me.surface_id = ?" if self.surface_only else ""
            ack_params: Tuple[Any, ...] = (
                (configured_remote, self.surface_id)
                if self.surface_only
                else (configured_remote,)
            )
            cursor.execute(
                f"""SELECT COUNT(*) FROM sync_outbox_ack AS ack
                    JOIN memory_events AS me ON me.event_id = ack.event_id
                    WHERE ack.remote_url = ?{ack_where}""",
                ack_params,
            )
            synced_count = cursor.fetchone()[0]
        else:
            synced_count = 0

        result: Dict[str, Any] = {
            "device_id": self.device_id,
            "total_events": total_events,
            "device_count": device_count,
            "last_event_time": last_event_time,
            "operation_breakdown": operation_breakdown,
            "synced_events": synced_count,
        }

        if remote_url:
            remote_url = remote_url.rstrip("/")
            self._validate_sync_remote_url(remote_url)
            result["remote"] = remote_url
            last_sync = self._meta_get(f"last_sync_at_{remote_url}")
            if last_sync:
                result["last_sync"] = last_sync
            try:
                import urllib.request as _request

                headers = {"Accept": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                request = _request.Request(
                    f"{remote_url.rstrip('/')}/sync/status",
                    headers=headers,
                    method="GET",
                )
                with _request.urlopen(request, timeout=10) as response:
                    result["remote_status"] = json.loads(
                        self._read_bounded_http_response(response).decode("utf-8")
                    )
            except Exception as exc:
                result["remote_error"] = str(exc)

        return result
