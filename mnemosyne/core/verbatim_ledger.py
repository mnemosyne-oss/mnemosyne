"""Opt-in, best-effort compression-boundary self-echo release.

This is NOT a live-context membership oracle or a durable checkpoint (Hermes
v2 / issue #872). A v1 pre-hook releases ALL exclusions, even if compression
retains text, does nothing or fails. Without an observed callback, recall is
ordinary recall. Reconstruction loses that capability; reset disables affected
keys for the remaining provider lifetime.

New suppression needs a successful, freshly marked provider write, unchanged
row readback, and a unique source projection AFTER an unchanged boundary-tail
anchor in the sync transcript, never released at a boundary. Generation revocation handles in-flight writes; the
cumulative released-source set handles old calls ENTERING after a boundary.
Repeated text and unsupported/missing sync transcripts abstain. Unknown hook
projections or resource overflow permanently disable the affected ledger state,
not truncate its safety evidence. No clocks, turn ring or historical backfill.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field

MAX_CAPTURES = 1024
MAX_SOURCE_HASHES = 8192
MAX_SESSIONS = 32
MAX_SESSION_KEY_CHARS = 512
MAX_PAYLOAD_CHARS = 4_000_000
MAX_MESSAGES = 16384
CAPTURE_KEY = "_mnemosyne_self_echo_capture"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


def self_echo_enabled() -> bool:
    return os.environ.get("MNEMOSYNE_SELF_ECHO_ENABLED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }


def projected_sources(messages) -> list[str] | None:
    """Known direct-text projection only; no host import or introspection.

    Multimodal text parts are newline-joined. Image parts do not contribute
    text. Unknown direct-message shapes abstain rather than pretending the
    payload covered queued sync source text. Tool/system payloads aren't sync
    sources. Limits bound work as well as the retained hash evidence.
    """
    if not isinstance(messages, (list, tuple)) or len(messages) > MAX_MESSAGES:
        return None
    hashes = []
    chars = 0
    for message in messages:
        if not isinstance(message, dict):
            return None
        if message.get("role") not in ("user", "assistant"):
            continue
        content = message.get("content")
        if isinstance(content, list):
            if len(content) > MAX_MESSAGES:
                return None
            texts = []
            for part in content:
                if not isinstance(part, dict):
                    return None
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    texts.append(part["text"])
                elif part.get("type") != "image_url":
                    return None
            if sum(map(len, texts)) > MAX_PAYLOAD_CHARS:
                return None
            content = "\n".join(texts)
        # Assistant tool-call placeholders contain no directly synced text.
        if content is None and message.get("role") == "assistant" and message.get("tool_calls"):
            continue
        if not isinstance(content, str):
            return None
        chars += len(content)
        if chars > MAX_PAYLOAD_CHARS:
            return None
        if content:
            hashes.append(content_hash(content))
        if len(hashes) > MAX_SOURCE_HASHES:
            return None
    return hashes


@dataclass
class _Generation:
    valid: bool = True


@dataclass(frozen=True)
class CaptureProof:
    memory_id: str
    session_id: str
    source_hash: str
    stored_hash: str
    nonce: str


@dataclass(frozen=True)
class ExclusionSnapshot:
    generation: _Generation
    captures: tuple[CaptureProof, ...]


@dataclass
class _Session:
    generation: _Generation = field(default_factory=_Generation)
    observed: bool = False
    anchor: str | None = None
    disabled: bool = False
    released: set[str] = field(default_factory=set)
    captures: dict[str, CaptureProof] = field(default_factory=dict)

    def disable(self):
        self.generation.valid = False
        self.disabled = True
        self.captures.clear()
        self.released.clear()


class VerbatimLedger:
    """Bounded provider-instance state; caps fail open, never age evidence out."""

    def __init__(self, enabled: bool | None = None):
        self.enabled = self_echo_enabled() if enabled is None else enabled
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.RLock()

    def _session(self, key):
        if not self.enabled or not isinstance(key, str) or not key or len(key) > MAX_SESSION_KEY_CHARS:
            return None
        if key not in self._sessions:
            if len(self._sessions) >= MAX_SESSIONS:
                for state in self._sessions.values():
                    state.disable()
                self._sessions.clear()
                self.enabled = False
                return None
            self._sessions[key] = _Session()
        return self._sessions[key]

    def release(self, key, messages):
        with self._lock:
            state = self._session(key)
            if state is None or state.disabled:
                return
            # Revoke snapshots and tickets BEFORE parsing any hook payload.
            state.generation.valid = False
            state.generation = _Generation()
            state.released.update(p.source_hash for p in state.captures.values())
            state.captures.clear()
            state.observed = True
            hashes = projected_sources(messages)
            if hashes is None or len(state.released | set(hashes)) > MAX_SOURCE_HASHES:
                state.disable()
            else:
                state.released.update(hashes)
                state.anchor = hashes[-1] if hashes else None

    def begin(self, key, messages):
        """Take ordering evidence before waiting for the provider's Beam lock."""
        with self._lock:
            state = self._session(key)
            if state is None or state.disabled or not state.observed:
                return None
            hashes = projected_sources(messages)
            if hashes is None or state.anchor is None or hashes.count(state.anchor) != 1:
                return None
            # An unchanged, unique boundary-tail anchor must PRECEDE the
            # source in this sync's transcript. Negative hash membership
            # alone cannot distinguish a rewritten/skill-expanded old source
            # from a new turn. Missing/rewritten anchors conservatively abstain.
            boundary = hashes.index(state.anchor)
            from collections import Counter
            counts = Counter(hashes)
            eligible = {h for h in hashes[boundary + 1:] if counts[h] == 1}
            return (state.generation, eligible)

    def capture(self, key, ticket, beam, raw, **kwargs):
        """Ordinary remember, with fresh metadata ONLY for a provable capture.

        Dedup preserves the previous metadata, so a colliding existing row
        cannot acquire this fresh nonce. Failed/unreadable writes never get a
        proof; do not fall back to guessing the stored content.
        """
        if ticket is None:
            return beam.remember(**kwargs)
        nonce = None
        raw_hash = content_hash(raw)
        with self._lock:
            state = self._sessions.get(key)
            if (ticket is not None and state is not None and not state.disabled
                    and ticket[0] is state.generation and ticket[0].valid
                    and raw_hash in ticket[1] and raw_hash not in state.released):
                repeats = [mid for mid, p in state.captures.items() if p.source_hash == raw_hash]
                if repeats:
                    # Same-text repeats lack distinct source identity. Revoke
                    # already handed-out snapshots too, not just this mapping.
                    state.generation.valid = False
                    state.generation = _Generation()
                    for mid in repeats:
                        state.captures.pop(mid)
                    state.released.add(raw_hash)
                    if len(state.released) > MAX_SOURCE_HASHES:
                        state.disable()
                else:
                    nonce = uuid.uuid4().hex
        if nonce:
            kwargs["metadata"] = {CAPTURE_KEY: nonce}
        memory_id = beam.remember(**kwargs)
        if not nonce or ticket is None or not isinstance(memory_id, str):
            return memory_id
        try:
            row = beam.conn.execute(
                "SELECT session_id, source, content, metadata_json FROM working_memory WHERE id = ?",
                (memory_id,),
            ).fetchone()
            session_id, source, stored, metadata = row
            if (not isinstance(session_id, str) or not session_id
                    or len(session_id) > MAX_SESSION_KEY_CHARS
                    or session_id != beam.session_id or source != "conversation"
                    or not isinstance(stored, str)
                    or json.loads(metadata or "{}").get(CAPTURE_KEY) != nonce):
                return memory_id
            proof = CaptureProof(memory_id, session_id, raw_hash, content_hash(stored), nonce)
        except (AttributeError, TypeError, ValueError, sqlite3.Error):
            return memory_id
        with self._lock:
            state = self._sessions.get(key)
            if (state is not None and not state.disabled and ticket[0].valid
                    and ticket[0] is state.generation and raw_hash not in state.released):
                if len(state.captures) >= MAX_CAPTURES:
                    state.disable()
                else:
                    state.captures[memory_id] = proof
        return memory_id

    def snapshot_for(self, key):
        with self._lock:
            if not isinstance(key, str) or not key or len(key) > MAX_SESSION_KEY_CHARS:
                return None
            state = self._sessions.get(key)
            if (not self.enabled or state is None or state.disabled
                    or not state.observed or not state.captures):
                return None
            return ExclusionSnapshot(state.generation, tuple(state.captures.values()))

    def reset_session(self, key):
        with self._lock:
            # Keep a bounded tombstone: forgetting released-source evidence
            # could let queued old calls entering after reset re-arm this key.
            # Only a fresh provider/session lifetime can gain capability again.
            state = self._session(key)
            if state is not None:
                state.observed = False
                state.anchor = None
                state.disable()


def resolve_exclusions(conn, snapshot) -> set[str]:
    """Shared bounded PK readback; equality of content is NOT ownership.

    A revoked cached snapshot becomes ordinary recall. No SQL UDF/full-bank
    scan, migrations, imported/NULL-session inference, or sorted-prefix cap.
    """
    if (not isinstance(snapshot, ExclusionSnapshot) or not snapshot.generation.valid
            or len(snapshot.captures) > MAX_CAPTURES):
        return set()
    try:
        # Leave room for candidate IDs and ordinary recall filters as well
        # as exclusions on older SQLite builds. Python 3.10 lacks getlimit;
        # use the conservative historical default there, not a claimed exact
        # runtime limit. A lower custom limit still fails open on SQL error.
        getlimit = getattr(conn, "getlimit", None)
        limit = getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER) if callable(getlimit) else 999
        if len(snapshot.captures) * 3 + 500 > limit:
            return set()
        proofs = {p.memory_id: p for p in snapshot.captures}
        excluded = set()
        ids = list(proofs)
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            rows = conn.execute(
                "SELECT id, session_id, source, content, metadata_json FROM working_memory WHERE id IN ("
                + ",".join("?" for _ in chunk) + ")", chunk,
            ).fetchall()
            for mid, session, source, content, metadata in rows:
                proof = proofs[mid]
                try:
                    owned = json.loads(metadata or "{}").get(CAPTURE_KEY) == proof.nonce
                except (ValueError, TypeError, AttributeError):
                    owned = False
                if (owned and session and session == proof.session_id and source == "conversation"
                        and isinstance(content, str) and content_hash(content) == proof.stored_hash):
                    excluded.add(mid)
        return excluded if snapshot.generation.valid else set()
    except (AttributeError, TypeError, ValueError, sqlite3.Error):
        return set()


def exclusion_sql(ids, column="id"):
    """Bounded IDs resolved from proofs, for WM-only WHERE predicates."""
    if not ids:
        return "", []
    values = sorted(ids)
    return f" AND {column} NOT IN ({','.join('?' for _ in values)})", values
