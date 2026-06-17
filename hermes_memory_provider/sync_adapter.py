"""
Mnemosyne Sync Adapter for Hermes
==================================
Wraps Mnemosyne's SyncEngine with encryption, HTTP transport, and Hermes
MemoryProvider lifecycle integration.

Exposes three tools to Hermes agents:
- mnemosyne_sync_push   — push local changes to a remote
- mnemosyne_sync_pull   — pull remote changes to local
- mnemosyne_sync_status — show sync state (device, cursor, event count)

Lifecycle:
    from hermes_memory_provider.sync_adapter import SyncAdapter
    adapter = SyncAdapter(beam, config={...})
    # ... adapter runs in background, tools auto-registered
    adapter.shutdown()

Config (env vars, then config.yaml, then defaults):
    MNEMOSYNE_SYNC_REMOTE     — remote sync server URL (https://host:port)
    MNEMOSYNE_SYNC_ENCRYPT    — '1'/'true' to enable client-side encryption
    MNEMOSYNE_SYNC_KEY        — raw key string (Fernet-compatible, 32 bytes base64)
    MNEMOSYNE_SYNC_KEY_SOURCE — 'env' | 'keyring' | 'prompt' | 'file:<path>'
    MNEMOSYNE_SYNC_TOKEN      — auth token for the remote server
    MNEMOSYNE_SYNC_MODE        — 'bidirectional' | 'pull' | 'push'
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

SYNC_PUSH_SCHEMA = {
    "name": "mnemosyne_sync_push",
    "description": (
        "Push local memory changes to a remote Mnemosyne sync server. "
        "Only events created since the last sync are sent. Requires a "
        "configured remote sync server (configured via config.yaml or "
        "MNEMOSYNE_SYNC_REMOTE env var)."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

SYNC_PULL_SCHEMA = {
    "name": "mnemosyne_sync_pull",
    "description": (
        "Pull remote memory changes from the configured Mnemosyne sync server. "
        "Applies incoming events locally with timestamp + importance conflict "
        "resolution."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

SYNC_STATUS_SCHEMA = {
    "name": "mnemosyne_sync_status",
    "description": (
        "Show Mnemosyne sync status: device ID, last cursor, event count, "
        "remote URL, and encryption state."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

ALL_SYNC_TOOL_SCHEMAS = [SYNC_PUSH_SCHEMA, SYNC_PULL_SCHEMA, SYNC_STATUS_SCHEMA]


# ---------------------------------------------------------------------------
# SyncAdapter
# ---------------------------------------------------------------------------

class SyncAdapter:
    """Hermes-side adapter wrapping Mnemosyne's SyncEngine.

    Call start() after construction. Call shutdown() on agent exit.
    Failure to construct is non-fatal — tools are simply not registered.
    """

    def __init__(self, beam_instance, config: Optional[Dict[str, Any]] = None):
        """Initialize the adapter.

        Args:
            beam_instance: A BeamMemory instance (already initialized).
            config: Optional dict of overrides. Keys map to env vars:
                remote, encrypt, key, key_source, token, mode.
        """
        self._beam = beam_instance
        self._config = config or {}
        self._engine: Any = None
        self._error: Optional[str] = None
        self._lock = threading.Lock()

        # Resolve configuration: explicit config > env vars > defaults
        self.remote = self._resolve_remote()
        self.encrypt_enabled = self._resolve_bool("encrypt", False)
        self.encryption_key = self._resolve_key()
        self.auth_token = self._string("token", "")
        self.mode = self._string("mode", "bidirectional")

        # Build the engine
        self._build_engine()

    # --- Config resolution ------------------------------------------------

    def _resolve_remote(self) -> str:
        remote = self._string("remote", "")
        if remote:
            return remote
        # Fallback: check MNEMOSYNE_SYNC_HOST + MNEMOSYNE_SYNC_PORT
        host = os.environ.get("MNEMOSYNE_SYNC_HOST", "").strip()
        port = os.environ.get("MNEMOSYNE_SYNC_PORT", "").strip()
        if host and port:
            return f"http://{host}:{port}"
        return ""

    def _resolve_key(self) -> str:
        """Resolve encryption key from config > env > keyring > file."""
        # 1. Explicit config
        raw = self._string("key", "")
        if raw:
            return raw

        # 2. Key source routing
        source = self._string("key_source", "env").lower()

        if source == "env":
            return os.environ.get("MNEMOSYNE_SYNC_KEY", "").strip()
        elif source.startswith("file:"):
            path = source[5:]
            try:
                return Path(os.path.expanduser(path)).read_text().strip()
            except Exception as exc:
                logger.warning("Sync key file %s unreadable: %s", path, exc)
                return ""
        elif source == "keyring":
            try:
                import keyring
                return keyring.get_password("mnemosyne-sync", "encryption-key") or ""
            except Exception:
                return ""
        elif source == "prompt":
            return ""  # Caller must supply via config at construction time

        return ""

    def _string(self, key: str, default: str = "") -> str:
        env_key = f"MNEMOSYNE_SYNC_{key.upper()}"
        env_val = os.environ.get(env_key, "").strip()
        if env_val:
            return env_val
        return str(self._config.get(key, default)).strip()

    def _resolve_bool(self, key: str, default: bool = False) -> bool:
        val = self._string(key, str(default)).lower()
        return val in ("1", "true", "yes", "on")

    # --- Engine construction -----------------------------------------------

    def _build_engine(self) -> None:
        """Construct the SyncEngine, encryption layer, and verify readiness."""
        try:
            from mnemosyne.core.sync import SyncEngine, SyncEncryption

            encryption = None
            if self.encrypt_enabled and self.encryption_key:
                encryption = SyncEncryption(key=self.encryption_key)
                logger.info("Sync encryption enabled (key length: %d)", len(self.encryption_key))
            elif self.encrypt_enabled and not self.encryption_key:
                logger.warning(
                    "Sync encryption enabled but no key configured. "
                    "Set MNEMOSYNE_SYNC_KEY or use key_source=keyring/file."
                )
                # Don't fail — engine still works plaintext

            self._engine = SyncEngine(
                beam_instance=self._beam,
                encryption=encryption,
            )
            logger.info(
                "SyncAdapter initialized: device=%s, remote=%s, encrypt=%s",
                getattr(self._engine, "device_id", "?"),
                self.remote or "(unconfigured)",
                self.encrypt_enabled,
            )

        except Exception as exc:
            self._error = str(exc)
            logger.debug("SyncAdapter init failed: %s", exc)

    # --- Lifecycle ---------------------------------------------------------

    def start(self) -> bool:
        """Called after construction. Returns True if ready."""
        if self._engine is None:
            logger.debug("SyncAdapter not started: %s", self._error or "no engine")
            return False
        return True

    def shutdown(self) -> None:
        """Called on agent exit. No-op for now (engine is connection-scoped)."""
        self._engine = None
        logger.debug("SyncAdapter shut down")

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        if self.is_ready:
            return list(ALL_SYNC_TOOL_SCHEMAS)
        return []

    # --- Tool dispatch -----------------------------------------------------

    def handle_tool_call(self, tool_name: str, args: dict) -> str:
        if not self.is_ready:
            return json.dumps({
                "status": "error",
                "error": f"Sync adapter not available: {self._error or 'not initialized'}",
            })

        try:
            if tool_name == "mnemosyne_sync_push":
                return self._handle_push()
            elif tool_name == "mnemosyne_sync_pull":
                return self._handle_pull()
            elif tool_name == "mnemosyne_sync_status":
                return self._handle_status()
            else:
                return json.dumps({"status": "error", "error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            logger.debug("Sync tool %s failed: %s", tool_name, exc)
            return json.dumps({"status": "error", "error": str(exc)})

    # --- Push --------------------------------------------------------------

    def _handle_push(self) -> str:
        if not self.remote:
            return json.dumps({
                "status": "error",
                "error": "No remote configured. Set MNEMOSYNE_SYNC_REMOTE env var.",
            })

        # Last cursor from sync_meta
        cursor = self._engine._meta_get("last_sync_cursor") or ""
        changes = self._engine.pull_changes(since_cursor=cursor or None, limit=500)

        events = changes.get("events", [])
        if not events:
            return json.dumps({
                "status": "ok",
                "pushed": 0,
                "message": "No local changes to push.",
            })

        # Push to remote
        result = self._http_post("/sync/push", {"events": events})
        if result.get("status") != "ok":
            return json.dumps(result)

        accepted = result.get("accepted", 0)
        cursor = result.get("next_cursor") or changes.get("next_cursor", "")

        if cursor:
            self._engine._meta_set("last_sync_cursor", cursor)

        return json.dumps({
            "status": "ok",
            "pushed": accepted,
            "duplicates": result.get("duplicates", 0),
            "conflicts": result.get("conflicts", 0),
            "next_cursor": cursor[:30] + "..." if len(cursor) > 30 else cursor,
        })

    # --- Pull --------------------------------------------------------------

    def _handle_pull(self) -> str:
        if not self.remote:
            return json.dumps({
                "status": "error",
                "error": "No remote configured. Set MNEMOSYNE_SYNC_REMOTE env var.",
            })

        cursor = self._engine._meta_get("last_sync_cursor") or ""
        result = self._http_post("/sync/pull", {"since_token": cursor or None})

        if result.get("status") != "ok":
            return json.dumps(result)

        incoming = result.get("events", [])
        if not incoming:
            return json.dumps({
                "status": "ok",
                "pulled": 0,
                "message": "No remote changes to pull.",
            })

        # Apply locally
        push_result = self._engine.push_changes(incoming)
        accepted = push_result.get("accepted", 0)
        cursor = result.get("next_cursor", "")

        if cursor:
            self._engine._meta_set("last_sync_cursor", cursor)

        return json.dumps({
            "status": "ok",
            "pulled": accepted,
            "duplicates": push_result.get("duplicates", 0),
            "conflicts": push_result.get("conflicts", 0),
            "next_cursor": cursor[:30] + "..." if len(cursor) > 30 else cursor,
        })

    # --- Status ------------------------------------------------------------

    def _handle_status(self) -> str:
        engine = self._engine
        if not engine:
            return json.dumps({"status": "error", "error": "No engine"})

        cursor = engine._meta_get("last_sync_cursor") or ""
        device_id = getattr(engine, "device_id", "unknown")

        # Count local events
        try:
            row = engine.conn.execute(
                "SELECT COUNT(*) FROM memory_events"
            ).fetchone()
            event_count = row[0] if row else 0
        except Exception:
            event_count = 0

        return json.dumps({
            "status": "ok",
            "device_id": device_id,
            "remote": self.remote or "(unconfigured)",
            "encryption": "enabled" if self.encrypt_enabled else "disabled",
            "mode": self.mode,
            "local_events": event_count,
            "last_cursor": cursor[:30] + "..." if len(cursor) > 30 else (cursor or "none"),
        })

    # --- HTTP transport (stdlib-only, no external deps) --------------------

    def _http_post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON to the remote sync server. Never raises."""
        url = self.remote.rstrip("/") + path
        data = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"mnemosyne-sync/{getattr(self._engine, 'device_id', 'hermes')}",
        }
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            logger.debug("Sync HTTP %d from %s: %s", exc.code, url, exc.reason)
            body = exc.read().decode("utf-8", errors="replace")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"status": "error", "error": f"HTTP {exc.code}: {exc.reason}"}
        except Exception as exc:
            logger.debug("Sync request failed: %s", exc)
            return {"status": "error", "error": str(exc)}
