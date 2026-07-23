"""
Mnemosyne MCP Server — Model Context Protocol for cross-agent sharing.

This module provides MCP tool definitions and handlers for Mnemosyne,
enabling any MCP-compatible client (Claude Desktop, etc.) to interact
with the memory system.

Usage:
    from mnemosyne.mcp_tools import TOOLS, handle_tool_call

All imports are guarded — this module loads safely even if mcp is not installed.
"""

from typing import Dict, Any, List
import json
import math
import os
import sqlite3
from pathlib import Path

# Guarded import — MCP is optional
try:
    from mcp.types import Tool, TextContent, CallToolResult, ErrorData
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    Tool = None
    TextContent = None
    CallToolResult = None
    ErrorData = None

from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.beam import BeamMemory, _guarded_transaction

from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS
from mnemosyne.batch_tool import (
    BatchValidationError,
    apply_beam_batch,
    batch_validation_error_payload,
    dry_run_batch,
    validate_batch_operations,
)

# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = []
for _s in ALL_TOOL_SCHEMAS:
    _t = dict(_s)
    if "parameters" in _t:
        _t["inputSchema"] = _t.pop("parameters")
    TOOLS.append(_t)

# ---------------------------------------------------------------------------
# Individual tool schemas (lazy - computed on first access)
# ---------------------------------------------------------------------------

def _get_schema(name: str) -> Dict[str, Any]:
    """Extract inputSchema from TOOLS by tool name."""
    for tool in TOOLS:
        if tool["name"] == name:
            return tool["inputSchema"]
    raise KeyError(f"Tool not found: {name}")

class _SchemaProxy:
    """Lazy proxy to access tool schemas after TOOLS is populated."""
    def __init__(self, name: str):
        self._name = name
        self._schema = None
    
    def __getattr__(self, attr):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return getattr(self._schema, attr)
    
    def __getitem__(self, key):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return self._schema[key]
    
    def __contains__(self, key):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return key in self._schema
    
    def get(self, key, default=None):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return self._schema.get(key, default)
    
    def __iter__(self):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return iter(self._schema)
    
    def __len__(self):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return len(self._schema)
    
    def __repr__(self):
        if self._schema is None:
            self._schema = _get_schema(self._name)
        return repr(self._schema)

_REMEMBER_SCHEMA = _SchemaProxy("mnemosyne_remember")
_RECALL_SCHEMA = _SchemaProxy("mnemosyne_recall")
_SLEEP_SCHEMA = _SchemaProxy("mnemosyne_sleep")
_SCRATCHPAD_READ_SCHEMA = _SchemaProxy("mnemosyne_scratchpad_read")
_SCRATCHPAD_WRITE_SCHEMA = _SchemaProxy("mnemosyne_scratchpad_write")
_GET_STATS_SCHEMA = _SchemaProxy("mnemosyne_stats")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_HERMES_HOME = os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
_MNEMOSYNE_HOME = os.environ.get("MNEMOSYNE_HOME", str(Path(_HERMES_HOME) / "mnemosyne"))


def _shared_db_path() -> Path:
    """Return the shared surface DB path."""
    return Path(os.environ.get("MNEMOSYNE_SHARED_DB_PATH", str(Path(_MNEMOSYNE_HOME) / "data" / "shared" / "mnemosyne.db")))


def _create_instance(session_id: str = None, author_id: str = None,
                     author_type: str = None, channel_id: str = None,
                     bank: str = "default") -> Mnemosyne:
    """Create a fresh Mnemosyne instance for each MCP connection.

    Identity is resolved from:
    1. Explicit args (from tool call or constructor)
    2. Environment variables (MNEMOSYNE_AUTHOR_ID, etc.)
    3. None (backward compatible, no identity tracking)
    """
    auth = author_id or os.environ.get("MNEMOSYNE_AUTHOR_ID")
    auth_type = author_type or os.environ.get("MNEMOSYNE_AUTHOR_TYPE")
    chan = channel_id or os.environ.get("MNEMOSYNE_CHANNEL_ID") or session_id or "default"
    sess = session_id or f"mcp_{bank}"

    return Mnemosyne(
        session_id=sess,
        author_id=auth,
        author_type=auth_type,
        channel_id=chan,
        bank=bank
    )


def _create_surface_instance() -> BeamMemory:
    """Create a BeamMemory instance for the shared surface DB."""
    shared_path = _shared_db_path()
    shared_path.parent.mkdir(parents=True, exist_ok=True)
    return BeamMemory(session_id="mcp_shared_surface", db_path=shared_path)


def _resolve_bank(arguments: Dict[str, Any]) -> str:
    """Resolve per-call bank, falling back to MCP server default bank."""
    return arguments.get("bank") or os.environ.get("MNEMOSYNE_MCP_BANK") or "default"


def _resolve_default_scope() -> str:
    """Resolve default scope for remember() calls.
    
    Precedence: MNEMOSYNE_DEFAULT_SCOPE env var, falling back to 'session'.
    Only 'session' and 'global' are accepted; unrecognized values fall through
    to the hardcoded default."""
    raw = os.environ.get("MNEMOSYNE_DEFAULT_SCOPE", "").strip().lower()
    if raw in ("session", "global"):
        return raw
    return "session"


def _serialize(obj):
    """Recursively convert non-serializable objects (datetime, etc.) to strings."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    return obj


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

class _WrapperBatchAdapter:
    """Expose Mnemosyne wrapper mutations through the Beam batch interface."""

    def __init__(self, mem: Mnemosyne):
        self._mem = mem
        self.conn = mem.beam.conn
        self.wrapper_events = []

    def remember(self, **kwargs):
        return self._call_wrapper("remember", **kwargs)

    def update_working(self, memory_id: str, *, content=None, importance=None):
        wrapper_ok = self._call_wrapper("update", memory_id, content=content, importance=importance)
        if wrapper_ok:
            return True
        return self._mem.beam.update_working(memory_id, content=content, importance=importance)

    def forget_working(self, memory_id: str):
        return self._call_wrapper("forget", memory_id)

    def invalidate(self, memory_id: str, *, replacement_id=None):
        return self._call_wrapper("invalidate", memory_id, replacement_id=replacement_id)

    def replay_wrapper_events(self) -> None:
        for event_type, memory_id, kwargs in self.wrapper_events:
            self._mem._emit_wrapper(event_type, memory_id, **kwargs)

    def _call_wrapper(self, method_name: str, *args, **kwargs):
        original_conn = self._mem.conn
        original_emit = self._mem._emit_wrapper
        self._mem.conn = self.conn
        self._mem._emit_wrapper = lambda event_type, memory_id, **event_kwargs: self.wrapper_events.append(
            (event_type, memory_id, event_kwargs)
        )
        try:
            return getattr(self._mem, method_name)(*args, **kwargs)
        finally:
            self._mem.conn = original_conn
            self._mem._emit_wrapper = original_emit


def _handle_remember(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_remember tool call."""
    content = arguments["content"]
    source = arguments.get("source", "mcp")
    importance = arguments.get("importance", 0.5)
    metadata = arguments.get("metadata", {})
    extract_entities = arguments.get("extract_entities", False)
    extract = arguments.get("extract", False)
    scope = arguments.get("scope", _resolve_default_scope())
    valid_until = arguments.get("valid_until") or None
    veracity = arguments.get("veracity", "unknown")
    bank = _resolve_bank(arguments)

    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    memory_id = mem.remember(
        content=content,
        source=source,
        importance=importance,
        metadata=metadata,
        extract_entities=extract_entities,
        extract=extract,
        scope=scope,
        valid_until=valid_until,
        veracity=veracity,
    )

    return {
        "status": "stored",
        "memory_id": memory_id,
        "content_preview": content[:100],
        "bank": bank
    }


def _handle_batch(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_batch tool call."""
    try:
        normalized = validate_batch_operations(arguments.get("operations"))
    except BatchValidationError as exc:
        return batch_validation_error_payload(exc)

    if bool(arguments.get("dry_run", False)):
        return dry_run_batch(normalized)

    bank = _resolve_bank(arguments)
    mem = _create_instance(
        author_id=arguments.get("author_id"),
        author_type=arguments.get("author_type"),
        channel_id=arguments.get("channel_id"),
        bank=bank,
    )
    audit_events = []
    adapter = _WrapperBatchAdapter(mem)
    result = apply_beam_batch(
        adapter,
        normalized,
        default_scope=_resolve_default_scope(),
        remember_source_default="mcp",
        audit_event=lambda name, **kwargs: audit_events.append({"event": name, **kwargs}),
    )
    if result.get("status") == "ok":
        adapter.replay_wrapper_events()
    result["bank"] = bank
    if audit_events:
        result["audit_events"] = audit_events
    return result


def _handle_recall(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_recall tool call."""
    query = arguments["query"]
    top_k = int(arguments.get("limit", arguments.get("top_k", 5)))
    bank = _resolve_bank(arguments)
    temporal_weight = arguments.get("temporal_weight", 0.0)
    query_time = arguments.get("query_time")
    temporal_halflife = arguments.get("temporal_halflife", 24)
    vec_weight = arguments.get("vec_weight")
    fts_weight = arguments.get("fts_weight")
    importance_weight = arguments.get("importance_weight")
    explain = bool(arguments.get("explain", False))

    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    recall_payload = mem.recall(
        query=query,
        top_k=top_k,
        temporal_weight=temporal_weight,
        query_time=query_time,
        temporal_halflife=temporal_halflife,
        vec_weight=vec_weight,
        fts_weight=fts_weight,
        importance_weight=importance_weight,
        explain=explain,
    )
    if explain:
        results = recall_payload.get("results", [])
        explain_payload = recall_payload.get("explain", {})
    else:
        results = recall_payload
        explain_payload = None

    serializable = []
    for r in results:
        item = dict(r) if hasattr(r, "keys") else r
        for key in ["timestamp", "created_at", "valid_until", "last_recalled"]:
            if key in item and item[key] is not None:
                if hasattr(item[key], "isoformat"):
                    item[key] = item[key].isoformat()
        serializable.append(item)

    response = {
        "status": "ok",
        "count": len(serializable),
        "results": serializable,
        "bank": bank
    }
    if explain_payload is not None:
        response.update({"query": query, "top_k": top_k, "explain": explain_payload})
    return response


def _handle_shared_remember(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_shared_remember tool call."""
    content = (arguments.get("content") or "").strip()
    if not content:
        return {"error": "content is required"}
    if content.startswith("[USER]") or content.startswith("[ASSISTANT]"):
        return {"error": "raw conversation content is not allowed in shared memory"}
    kind = (arguments.get("kind") or "meta").strip().lower()
    if kind not in {"meta", "preference", "correction", "identity"}:
        return {"error": "kind must be one of: meta, preference, correction, identity"}
    importance = max(0.0, min(float(arguments.get("importance", 0.8)), 1.0))
    metadata = arguments.get("metadata") or {}
    if not isinstance(metadata, dict):
        return {"error": "metadata must be an object"}

    surface_beam = _create_surface_instance()
    import hashlib
    normalized = " ".join(str(content).lower().split())
    content_hash = hashlib.sha256(f"surface:v1:{normalized}".encode("utf-8")).hexdigest()[:24]
    prefixes = ("surface meta:", "surface preference:", "surface correction:", "surface identity:", "surface fact:")
    if content.lower().startswith(prefixes):
        surface_content = content
    else:
        label_map = {"meta": "Surface meta", "preference": "Surface preference",
                     "correction": "Surface correction", "identity": "Surface identity"}
        surface_content = f"{label_map.get(kind, 'Surface meta')}: {content}"
    stable_id = "sf_" + content_hash
    meta = dict(metadata)
    meta.update({"shared_memory": True, "surface_kind": kind, "write_path": "mcp_tool"})
    memory_id = surface_beam.remember(
        content=surface_content,
        source="surface_manual",
        importance=importance,
        metadata=meta,
        scope="global",
        memory_id=stable_id,
    )

    return {
        "status": "stored_shared",
        "memory_id": memory_id,
        "content_preview": surface_content[:120],
        "shared_db": str(_shared_db_path()),
        "kind": kind,
    }


def _handle_shared_recall(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_shared_recall tool call."""
    query = arguments.get("query", "")
    if not query:
        return {"error": "query is required"}
    top_k = int(arguments.get("limit", 5))
    surface_beam = _create_surface_instance()
    results = []
    for r in surface_beam.recall(query, top_k=top_k):
        r = dict(r)
        r["shared_surface"] = True
        results.append(r)
    return {"query": query, "count": len(results), "shared_db": str(_shared_db_path()), "results": results}


def _handle_shared_forget(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_shared_forget tool call."""
    memory_id = (arguments.get("memory_id") or "").strip()
    if not memory_id:
        return {"error": "memory_id is required"}
    surface_beam = _create_surface_instance()
    ok = surface_beam.forget_working(memory_id)
    return {"status": "deleted" if ok else "not_found", "memory_id": memory_id, "shared_db": str(_shared_db_path())}


def _handle_shared_stats(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_shared_stats tool call."""
    surface_beam = _create_surface_instance()
    return {
        "provider": "mnemosyne_shared",
        "shared_db": str(_shared_db_path()),
        "working": _serialize(surface_beam.get_working_stats()),
        "episodic": _serialize(surface_beam.get_episodic_stats()),
    }


def _handle_sleep(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_sleep tool call."""
    dry_run = arguments.get("dry_run", False)
    force = arguments.get("force", False)
    all_sessions = arguments.get("all_sessions", False)
    bank = _resolve_bank(arguments)

    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    if all_sessions and hasattr(mem, "sleep_all_sessions"):
        result = mem.sleep_all_sessions(dry_run=dry_run, force=force)
    else:
        result = mem.sleep(dry_run=dry_run, force=force)

    working = _serialize(mem.beam.get_working_stats()) if hasattr(mem, "beam") else {}
    episodic = _serialize(mem.beam.get_episodic_stats()) if hasattr(mem, "beam") else {}

    return {
        "status": result.get("status", "consolidated"),
        "result": result,
        "working": working,
        "episodic": episodic,
        "bank": bank
    }


def _handle_stats(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_stats tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    stats = mem.get_stats()
    return {"provider": "mnemosyne", "session_id": mem._session_id if hasattr(mem, "_session_id") else None, "stats": _serialize(stats)}


def _handle_invalidate(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_invalidate tool call."""
    memory_id = arguments.get("memory_id", "")
    replacement_id = arguments.get("replacement_id") or None
    if not memory_id:
        return {"error": "memory_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    mem.invalidate(memory_id, replacement_id=replacement_id)
    return {"status": "invalidated", "memory_id": memory_id}


def _handle_validate(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_validate tool call."""
    memory_id = arguments.get("memory_id", "")
    action = arguments.get("action", "")
    bank = arguments.get("bank", "private")
    validator = arguments.get("validator") or os.environ.get("MNEMOSYNE_AUTHOR_ID") or "mcp"
    new_content = arguments.get("new_content", "")
    note = arguments.get("note", "")

    if not memory_id:
        return {"error": "memory_id is required"}
    if action not in ("attest", "update", "invalidate", "delete"):
        return {"error": f"unknown action: {action}"}
    if bank not in ("private", "surface"):
        return {"error": f"unknown bank: {bank}"}
    if action == "update" and not new_content:
        return {"error": "new_content is required for action='update'"}

    if bank == "surface":
        target_beam = _create_surface_instance()
    else:
        mem = _create_instance()
        target_beam = mem.beam

    conn = target_beam.conn
    existing = conn.execute(
        "SELECT id, author_id, content FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()
    if not existing:
        return {"error": "memory_not_found", "memory_id": memory_id, "bank": bank}

    author_id = existing[1]
    prev_content = existing[2]

    try:
        with _guarded_transaction(conn):
            if action == "delete":
                # Cascade delete child rows before removing parent.
                conn.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
                conn.execute("DELETE FROM annotations WHERE memory_id = ?", (memory_id,))
                # vec_working is optional (sqlite-vec may be unavailable).
                row = conn.execute("SELECT rowid FROM working_memory WHERE id = ?", (memory_id,)).fetchone()
                if row is not None:
                    try:
                        conn.execute("DELETE FROM vec_working WHERE rowid = ?", (row["rowid"],))
                    except sqlite3.OperationalError as vec_err:
                        if "no such table" not in str(vec_err).lower():
                            raise
                conn.execute("DELETE FROM working_memory WHERE id = ?", (memory_id,))
            elif action == "update":
                conn.execute(
                    "UPDATE working_memory SET content = ?, validator = ?, "
                    "validated_at = CURRENT_TIMESTAMP, "
                    "validation_count = COALESCE(validation_count, 0) + 1 "
                    "WHERE id = ?",
                    (new_content, validator, memory_id),
                )
            elif action == "invalidate":
                conn.execute(
                    "UPDATE working_memory SET valid_until = CURRENT_TIMESTAMP, "
                    "validator = ?, validated_at = CURRENT_TIMESTAMP, "
                    "validation_count = COALESCE(validation_count, 0) + 1 "
                    "WHERE id = ?",
                    (validator, memory_id),
                )
            else:
                conn.execute(
                    "UPDATE working_memory SET validator = ?, "
                    "validated_at = CURRENT_TIMESTAMP, "
                    "validation_count = COALESCE(validation_count, 0) + 1 "
                    "WHERE id = ?",
                    (validator, memory_id),
                )
            conn.execute(
                "INSERT INTO memory_validations "
                "(memory_id, validator, action, new_content, note) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, validator, action,
                 new_content if action == "update" else None,
                 note or None),
            )
    except Exception as exc:
        return {"error": "validation_failed", "reason": str(exc), "memory_id": memory_id}

    return {
        "status": f"validation_{action}",
        "memory_id": memory_id,
        "bank": bank,
        "validator": validator,
        "author_id": author_id,
        "previous_content": prev_content[:200] if prev_content else None,
    }


def _handle_get(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_get tool call."""
    memory_id = arguments.get("memory_id", "")
    if not memory_id:
        return {"error": "memory_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    result = mem.get(memory_id)
    if result is None:
        return {"status": "not_found", "memory_id": memory_id}
    return {"status": "ok", "memory": _serialize(result)}


def _handle_triple_add(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_triple_add tool call.

    Routes annotation-flavored predicates (mentions, fact, occurred_on,
    has_source) to AnnotationStore; everything else to TripleStore.
    For occurred_on, valid_from is forwarded to AnnotationStore (issue #111).
    """
    import logging
    _log = logging.getLogger("mnemosyne.mcp.triple_add")

    from mnemosyne.core.annotations import ANNOTATION_KINDS, AnnotationStore
    from mnemosyne.core.triples import TripleStore

    predicate = arguments["predicate"]

    if isinstance(predicate, str) and predicate in ANNOTATION_KINDS:
        bank = _resolve_bank(arguments)
        mem = _create_instance(bank=bank)
        db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
        store = getattr(mem.beam, "annotations", None)
        if store is None:
            store = AnnotationStore(db_path=db_path, conn=mem.beam.conn)
        valid_from = arguments.get("valid_from")
        if predicate == "occurred_on" and valid_from:
            row_id = store.add(
                memory_id=arguments["subject"],
                kind=predicate,
                value=arguments["object"],
                source=arguments.get("source", "conversation"),
                confidence=arguments.get("confidence", 1.0),
                valid_from=valid_from,
            )
        else:
            if valid_from:
                _log.warning(
                    "mnemosyne_triple_add: valid_from=%r provided with "
                    "predicate=%r (not occurred_on); valid_from discarded.",
                    valid_from, predicate,
                )
            row_id = store.add(
                memory_id=arguments["subject"],
                kind=predicate,
                value=arguments["object"],
                source=arguments.get("source", "conversation"),
                confidence=arguments.get("confidence", 1.0),
            )
        return {"status": "added", "annotation_id": row_id, "store": "annotations"}

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
    kg = TripleStore(db_path=db_path)
    triple_id = kg.add(
        subject=arguments["subject"],
        predicate=predicate,
        object=arguments["object"],
        valid_from=arguments.get("valid_from"),
        source=arguments.get("source", "conversation"),
        confidence=arguments.get("confidence", 1.0),
    )
    return {"status": "added", "triple_id": triple_id, "store": "triples"}


def _handle_triple_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_triple_query tool call.

    Mirrors the write-side routing: annotation predicates query
    AnnotationStore; others query TripleStore.
    """
    from mnemosyne.core.annotations import ANNOTATION_KINDS, AnnotationStore
    from mnemosyne.core.triples import TripleStore

    predicate = arguments.get("predicate")
    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path

    if isinstance(predicate, str) and predicate in ANNOTATION_KINDS:
        store = getattr(mem.beam, "annotations", None)
        if store is None:
            store = AnnotationStore(db_path=db_path, conn=mem.beam.conn)
        results = store.query_by_kind(
            kind=predicate,
            value=arguments.get("object"),
            memory_id=arguments.get("subject"),
        )
        return {"results_count": len(results), "results": results, "store": "annotations"}

    kg = TripleStore(db_path=db_path)
    results = kg.query(
        subject=arguments.get("subject"),
        predicate=predicate,
        object=arguments.get("object"),
        as_of=arguments.get("as_of"),
    )
    return {"results_count": len(results), "results": results, "store": "triples"}


def _canonical_owner(arguments: Dict[str, Any]) -> str:
    """Owner id for canonical ops over the MCP surface.

    The shared tool schema does not expose owner_id (so an LLM can't target
    another owner's bank); over MCP the owner is a deployment-level setting via
    MNEMOSYNE_DEFAULT_OWNER, defaulting to "default" for single-owner use."""
    return (os.environ.get("MNEMOSYNE_DEFAULT_OWNER") or "default").strip() or "default"


def _handle_remember_canonical(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_remember_canonical tool call."""
    from mnemosyne.core.canonical import CanonicalStore

    category = (arguments.get("category") or "").strip()
    name = (arguments.get("name") or "").strip()
    body = (arguments.get("body") or "").strip()
    if not category or not name:
        return {"error": "category and name are required"}
    if not body:
        return {"error": "body is required"}

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    store = getattr(mem.beam, "canonical", None)
    if store is None:
        db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
        store = CanonicalStore(db_path=db_path, conn=mem.beam.conn)

    owner_id = _canonical_owner(arguments)
    row = store.remember(
        owner_id, category, name, body,
        source=arguments.get("source", "canonical_tool"),
        confidence=arguments.get("confidence", 1.0),
    )
    status = row.pop("status", "stored")
    return {"status": status, "owner_id": owner_id, "category": category,
            "name": name, "version": row.get("version"), "store": "canonical"}


def _handle_recall_canonical(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_recall_canonical tool call."""
    from mnemosyne.core.canonical import CanonicalStore

    category = (arguments.get("category") or "").strip()
    name = (arguments.get("name") or "").strip()
    query = (arguments.get("query") or "").strip()
    include_history = bool(arguments.get("include_history", False))
    try:
        limit = int(arguments.get("limit", 10))
    except (TypeError, ValueError):
        limit = 10

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    store = getattr(mem.beam, "canonical", None)
    if store is None:
        db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path
        store = CanonicalStore(db_path=db_path, conn=mem.beam.conn)
    owner_id = _canonical_owner(arguments)

    if query:
        results = store.search(owner_id, query, limit=limit)
        return {"mode": "search", "owner_id": owner_id, "query": query,
                "results_count": len(results), "results": results, "store": "canonical"}
    if category and name:
        if include_history:
            results = store.history(owner_id, category, name)
            return {"mode": "history", "owner_id": owner_id, "category": category,
                    "name": name, "results_count": len(results),
                    "results": results, "store": "canonical"}
        row = store.recall(owner_id, category, name)
        result = {"mode": "recall", "owner_id": owner_id, "category": category,
                "name": name, "found": row is not None, "result": row,
                "store": "canonical"}
        if row is None:
            # Diagnostic: check if the row exists under a different owner_id
            try:
                conn = store.conn if hasattr(store, "conn") else None
                if conn is not None:
                    cur = conn.execute(
                        "SELECT owner_id FROM canonical_facts "
                        "WHERE category=? AND name=? AND valid_until IS NULL LIMIT 1",
                        (category, name),
                    )
                    alt = cur.fetchone()
                    if alt:
                        result["hint"] = (
                            f"Row exists under owner_id '{alt[0]}' but you queried "
                            f"with '{owner_id}'. Set MNEMOSYNE_DEFAULT_OWNER={alt[0]} "
                            f"or check your profile/provider config."
                        )
            except Exception:
                pass
        return result
    results = store.list(owner_id, category=category or None)
    return {"mode": "list", "owner_id": owner_id, "category": category or None,
            "results_count": len(results), "results": results, "store": "canonical"}


def _handle_scratchpad_write(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_scratchpad_write tool call."""
    content = arguments.get("content", "").strip()
    if not content:
        return {"error": "Content is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    entry_id = mem.scratchpad_write(content)
    return {"status": "written", "id": entry_id}


def _handle_scratchpad_read(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_scratchpad_read tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    entries = mem.scratchpad_read()
    return {"entries_count": len(entries), "entries": entries}


def _handle_scratchpad_clear(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_scratchpad_clear tool call."""
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    mem.scratchpad_clear()
    return {"status": "cleared"}


def _handle_export(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_export tool call."""
    output_path = arguments.get("output_path", "").strip()
    if not output_path:
        return {"error": "output_path is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    result = mem.export_to_file(output_path)
    return _serialize(result)


def _handle_update(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_update tool call."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return {"error": "memory_id is required"}
    content = arguments.get("content")
    importance = arguments.get("importance")
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    ok = mem.update(memory_id, content=content, importance=importance)
    return {"status": "updated" if ok else "not_found", "memory_id": memory_id}


def _handle_forget(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_forget tool call."""
    memory_id = arguments.get("memory_id", "").strip()
    if not memory_id:
        return {"error": "memory_id is required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    ok = mem.forget(memory_id)
    return {"status": "deleted" if ok else "not_found", "memory_id": memory_id}


def _handle_import(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_import tool call."""
    provider = (arguments.get("provider") or "").strip().lower()
    input_path = arguments.get("input_path", "").strip()
    dry_run = bool(arguments.get("dry_run", False))
    force = bool(arguments.get("force", False))
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)

    if provider:
        api_key = arguments.get("api_key", "").strip()
        user_id = arguments.get("user_id", "").strip() or None
        agent_id = arguments.get("agent_id", "").strip() or None
        base_url = arguments.get("base_url", "").strip() or None
        channel_id = arguments.get("channel_id")
        if not api_key:
            env_key = f"{provider.upper()}_API_KEY"
            api_key = os.environ.get(env_key, "")
        if not api_key:
            return {"error": f"api_key required for {provider} import. Set {provider.upper()}_API_KEY env var or pass api_key parameter."}
        from mnemosyne.core.importers import import_from_provider
        result = import_from_provider(
            provider, mem,
            api_key=api_key,
            user_id=user_id,
            agent_id=agent_id,
            base_url=base_url,
            dry_run=dry_run,
            channel_id=channel_id,
        )
        return _serialize(result.to_dict() if hasattr(result, "to_dict") else result)

    if not input_path:
        return {"error": "Either input_path (for file import) or provider (for cross-provider import) is required"}
    stats = mem.import_from_file(input_path, force=force)
    return {"status": "imported", "stats": stats}


def _handle_diagnose(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_diagnose tool call."""
    from mnemosyne.diagnose import run_diagnostics
    result = run_diagnostics(
        repair_vec_working=bool(arguments.get("repair_vec_working", False)),
        dry_run=bool(arguments.get("dry_run", False)),
    )
    db_path = None
    try:
        mem = _create_instance()
        if hasattr(mem, "beam") and hasattr(mem.beam, "db_path"):
            db_path = str(mem.beam.db_path)
    except Exception:
        pass
    if db_path:
        result["active_provider_db_path"] = db_path
    return _serialize(result)


def _handle_graph_query(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_graph_query tool call."""
    seed_id = arguments.get("seed_memory_id", "").strip()
    if not seed_id:
        return {"error": "seed_memory_id is required"}
    depth = int(arguments.get("max_hops", 2))
    if depth < 1:
        return {"error": "max_hops must be greater than 0"}
    edge_type = arguments.get("edge_type", "") or ""
    min_weight = float(arguments.get("min_weight", 0.0))
    if not (0.0 <= min_weight <= 1.0):
        return {"error": "min_weight must be between 0.0 and 1.0"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    if mem.beam.episodic_graph is None:
        return {"error": "Episodic graph not available"}
    related = mem.beam.episodic_graph.find_related_memories(
        seed_id, depth=depth, edge_type=edge_type, min_weight=min_weight
    )
    return {
        "seed_memory_id": seed_id,
        "max_hops": depth,
        "edge_type": edge_type or "all",
        "min_weight": min_weight,
        "count": len(related),
        "results": related,
    }


def _handle_graph_link(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_graph_link tool call."""
    source_id = arguments.get("source_id", "").strip()
    target_id = arguments.get("target_id", "").strip()
    relationship = arguments.get("relationship", "").strip()
    weight = float(arguments.get("weight", 0.5))
    if not (0.0 <= weight <= 1.0):
        return {"error": "weight must be between 0.0 and 1.0"}
    if not all([source_id, target_id, relationship]):
        return {"error": "source_id, target_id, and relationship are required"}
    bank = _resolve_bank(arguments)
    mem = _create_instance(author_id=arguments.get("author_id"), author_type=arguments.get("author_type"), channel_id=arguments.get("channel_id"), bank=bank)
    if mem.beam.episodic_graph is None:
        return {"error": "Episodic graph not available"}
    from mnemosyne.core.episodic_graph import GraphEdge
    from datetime import datetime
    edge = GraphEdge(
        source=source_id,
        target=target_id,
        edge_type=relationship,
        weight=weight,
        timestamp=datetime.now().isoformat(),
    )
    mem.beam.episodic_graph.add_edge(edge)
    return {"status": "linked", "source": source_id, "target": target_id, "relationship": relationship}


# ---------------------------------------------------------------------------
# Hygiene handlers (issue #428)
# ---------------------------------------------------------------------------

def _handle_hygiene_audit(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_hygiene_audit tool call."""
    from mnemosyne.core.hygiene import audit_noise

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path

    limit = arguments.get("limit", 200)
    min_score = arguments.get("min_score", 0.3)
    tables = arguments.get("tables") or None
    offset = arguments.get("offset", 0)
    scan_all = arguments.get("scan_all", False)
    batch_size = arguments.get("batch_size", 1000)

    report = audit_noise(
        db_path=db_path,
        limit=limit,
        tables=tables,
        min_score=min_score,
        offset=offset,
        scan_all=scan_all,
        batch_size=batch_size,
    )
    return {
        "status": "audited",
        "report": report.to_dict(),
        "bank": bank,
    }


def _handle_hygiene_clean(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Handle mnemosyne_hygiene_clean tool call."""
    from mnemosyne.core.hygiene import NoiseCandidate, clean_noise

    candidates_json = arguments.get("candidates_json", "[]")
    try:
        raw_candidates = json.loads(candidates_json) if isinstance(candidates_json, str) else candidates_json
    except json.JSONDecodeError:
        return {"error": "candidates_json is not valid JSON"}

    if not isinstance(raw_candidates, list):
        return {"error": "candidates_json must be a list of valid hygiene candidates"}

    candidates = []
    for candidate_data in raw_candidates:
        if not isinstance(candidate_data, dict):
            return {"error": "candidates_json must be a list of valid hygiene candidates"}
        if not all(
            isinstance(candidate_data.get(key), str) and candidate_data[key].strip()
            for key in ("memory_id", "table_name")
        ):
            return {"error": "candidates_json must be a list of valid hygiene candidates"}
        if candidate_data["table_name"] not in {"working_memory", "memories", "episodic_memory"}:
            return {"error": "candidates_json must be a list of valid hygiene candidates"}
        noise_score = candidate_data.get("noise_score", 0.0)
        importance = candidate_data.get("importance", 0.5)
        content_length = candidate_data.get("content_length", 0)
        if (
            not isinstance(noise_score, (int, float))
            or isinstance(noise_score, bool)
            or not 0 <= noise_score <= 1
            or (isinstance(noise_score, float) and not math.isfinite(noise_score))
            or not isinstance(importance, (int, float))
            or isinstance(importance, bool)
            or not 0 <= importance <= 1
            or (isinstance(importance, float) and not math.isfinite(importance))
            or not isinstance(content_length, int)
            or isinstance(content_length, bool)
            or content_length < 0
        ):
            return {"error": "candidates_json must be a list of valid hygiene candidates"}
        if any(
            key in candidate_data
            and (not isinstance(candidate_data[key], list) or not all(isinstance(item, str) for item in candidate_data[key]))
            for key in ("noise_reasons", "secret_flags")
        ):
            return {"error": "candidates_json must be a list of valid hygiene candidates"}
        if any(
            key in candidate_data and not isinstance(candidate_data[key], str)
            for key in ("content_preview", "source", "timestamp", "suggested_action")
        ):
            return {"error": "candidates_json must be a list of valid hygiene candidates"}
        if candidate_data.get("suggested_action", "keep") not in {"delete", "archive", "keep", "flag"}:
            return {"error": "candidates_json must be a list of valid hygiene candidates"}
        candidates.append(
            NoiseCandidate(
                memory_id=candidate_data["memory_id"],
                table_name=candidate_data["table_name"],
                content_preview=candidate_data.get("content_preview", ""),
                noise_score=candidate_data.get("noise_score", 0.0),
                noise_reasons=candidate_data.get("noise_reasons", []),
                secret_flags=candidate_data.get("secret_flags", []),
                importance=candidate_data.get("importance", 0.5),
                source=candidate_data.get("source", ""),
                timestamp=candidate_data.get("timestamp", ""),
                suggested_action=candidate_data.get("suggested_action", "keep"),
                content_length=candidate_data.get("content_length", 0),
            )
        )

    bank = _resolve_bank(arguments)
    mem = _create_instance(bank=bank)
    db_path = mem.beam.db_path if hasattr(mem.beam, "db_path") else mem.db_path

    action = arguments.get("action", "keep")
    confirm = arguments.get("confirm", False)
    dry_run = not confirm

    result = clean_noise(
        db_path=db_path,
        candidates=candidates,
        action=action,
        confirm=confirm,
        dry_run=dry_run,
    )
    return {
        "status": "dry_run" if dry_run else "applied",
        "result": result.to_dict(),
        "bank": bank,
    }


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_TOOL_HANDLERS = {
    "mnemosyne_remember": _handle_remember,
    "mnemosyne_batch": _handle_batch,
    "mnemosyne_recall": _handle_recall,
    "mnemosyne_shared_remember": _handle_shared_remember,
    "mnemosyne_shared_recall": _handle_shared_recall,
    "mnemosyne_shared_forget": _handle_shared_forget,
    "mnemosyne_shared_stats": _handle_shared_stats,
    "mnemosyne_sleep": _handle_sleep,
    "mnemosyne_stats": _handle_stats,
    "mnemosyne_invalidate": _handle_invalidate,
    "mnemosyne_validate": _handle_validate,
    "mnemosyne_get": _handle_get,
    "mnemosyne_triple_add": _handle_triple_add,
    "mnemosyne_triple_query": _handle_triple_query,
    "mnemosyne_remember_canonical": _handle_remember_canonical,
    "mnemosyne_recall_canonical": _handle_recall_canonical,
    "mnemosyne_scratchpad_write": _handle_scratchpad_write,
    "mnemosyne_scratchpad_read": _handle_scratchpad_read,
    "mnemosyne_scratchpad_clear": _handle_scratchpad_clear,
    "mnemosyne_export": _handle_export,
    "mnemosyne_update": _handle_update,
    "mnemosyne_forget": _handle_forget,
    "mnemosyne_import": _handle_import,
    "mnemosyne_diagnose": _handle_diagnose,
    "mnemosyne_graph_query": _handle_graph_query,
    "mnemosyne_graph_link": _handle_graph_link,
    "mnemosyne_hygiene_audit": _handle_hygiene_audit,
    "mnemosyne_hygiene_clean": _handle_hygiene_clean,
}


def handle_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch an MCP tool call to the correct handler.

    Args:
        name: Tool name (e.g., "mnemosyne_remember")
        arguments: Parsed JSON arguments

    Returns:
        JSON-serializable result dict

    Raises:
        ValueError: If tool name is unknown
    """
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}. Available: {list(_TOOL_HANDLERS.keys())}")

    return handler(arguments)


def get_tool_definitions() -> List[Dict[str, Any]]:
    """Return all tool definitions for MCP server registration."""
    return TOOLS
