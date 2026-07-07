"""Shared v1 implementation for the narrow mnemosyne_batch mutation tool."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from mnemosyne.core.beam import _deferred_commits
from mnemosyne.core.veracity_consolidation import clamp_veracity


logger = logging.getLogger(__name__)

_ALLOWED_BATCH_ACTIONS = {"remember", "update", "forget", "invalidate"}
_FUZZY_BATCH_FIELDS = {"old_text", "query", "content_match"}
_BATCH_MAX_OPS = 50


@dataclass(frozen=True)
class BatchValidationError(ValueError):
    message: str
    failed_index: int | None = None
    action: str | None = None

    def __str__(self) -> str:
        return self.message


class BatchOperationError(RuntimeError):
    pass


def validate_batch_operations(operations: Any) -> list[dict[str, Any]]:
    if not isinstance(operations, list) or not operations:
        raise BatchValidationError("operations must be a non-empty list")
    if len(operations) > _BATCH_MAX_OPS:
        raise BatchValidationError(f"operations must contain at most {_BATCH_MAX_OPS} items")

    normalized: list[dict[str, Any]] = []
    for index, op in enumerate(operations):
        if not isinstance(op, dict):
            raise BatchValidationError("operation must be an object", index)
        action = op.get("action")
        if not isinstance(action, str) or not action:
            raise BatchValidationError("action is required", index)
        if action not in _ALLOWED_BATCH_ACTIONS:
            raise BatchValidationError(f"unknown action: {action}", index, action)
        fuzzy = sorted(_FUZZY_BATCH_FIELDS.intersection(op))
        if fuzzy:
            raise BatchValidationError(
                f"fuzzy fields are not supported in mnemosyne_batch v1: {', '.join(fuzzy)}",
                index,
                action,
            )

        _VALIDATORS[action](op, index, action)
        normalized.append({"index": index, "action": action, "payload": dict(op)})
    return normalized


def _validate_remember(op: dict[str, Any], index: int, action: str) -> None:
    content = op.get("content")
    if not isinstance(content, str) or not content.strip():
        raise BatchValidationError("content is required", index, action)


def _validate_update(op: dict[str, Any], index: int, action: str) -> None:
    if not _non_empty_string(op.get("memory_id")):
        raise BatchValidationError("memory_id is required", index, action)
    if op.get("content") is None and op.get("importance") is None:
        raise BatchValidationError("content or importance is required", index, action)


def _validate_by_memory_id(op: dict[str, Any], index: int, action: str) -> None:
    if not _non_empty_string(op.get("memory_id")):
        raise BatchValidationError("memory_id is required", index, action)


_VALIDATORS: dict[str, Callable[[dict[str, Any], int, str], None]] = {
    "remember": _validate_remember,
    "update": _validate_update,
    "forget": _validate_by_memory_id,
    "invalidate": _validate_by_memory_id,
}


def dry_run_batch(normalized: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "dry_run",
        "operations_count": len(normalized),
        "results": [_dry_run_result(op) for op in normalized],
    }


def batch_validation_error_payload(exc: BatchValidationError) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": "error", "error": str(exc)}
    if exc.failed_index is not None:
        payload["failed_index"] = exc.failed_index
    if exc.action:
        payload["action"] = exc.action
    return payload


def apply_beam_batch(
    beam: Any,
    normalized: list[dict[str, Any]],
    *,
    default_scope: str = "session",
    remember_source_default: str = "user",
    remember_source_tool: str = "mnemosyne_batch",
    audit_event: Callable[..., Any] | None = None,
    extract_defaults_global: bool = False,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    audit_events: list[tuple[str, dict[str, Any]]] = []
    current = {"index": None, "action": None}
    try:
        with _deferred_commits(beam.conn):
            for current in normalized:
                results.append(_apply_one(
                    beam,
                    current,
                    default_scope=default_scope,
                    remember_source_default=remember_source_default,
                    remember_source_tool=remember_source_tool,
                    audit_events=audit_events,
                    extract_defaults_global=extract_defaults_global,
                ))
    except Exception as exc:
        logger.exception(
            "mnemosyne_batch failed at index=%s action=%s",
            current.get("index"),
            current.get("action"),
        )
        return {
            "status": "error",
            "failed_index": current.get("index"),
            "action": current.get("action"),
            "error": f"{type(exc).__name__}: {exc}",
        }
    if audit_event:
        for event_name, event_kwargs in audit_events:
            audit_event(event_name, **event_kwargs)
    return {"status": "ok", "operations_count": len(results), "results": results}


def _apply_one(
    beam: Any,
    op: dict[str, Any],
    *,
    default_scope: str,
    remember_source_default: str,
    remember_source_tool: str,
    audit_events: list[tuple[str, dict[str, Any]]],
    extract_defaults_global: bool,
) -> dict[str, Any]:
    index = op["index"]
    action = op["action"]
    payload = op["payload"]

    if action == "remember":
        extract = bool(payload.get("extract", False))
        scope = payload.get("scope", "global" if extract_defaults_global and extract else default_scope)
        metadata = payload.get("metadata") or None
        veracity = clamp_veracity(payload.get("veracity"), context="mnemosyne_batch")
        content = payload["content"]
        memory_id = beam.remember(
            content=content,
            importance=float(payload.get("importance", 0.5)),
            source=payload.get("source", remember_source_default),
            scope=scope,
            valid_until=payload.get("valid_until") or None,
            extract_entities=bool(payload.get("extract_entities", False)),
            extract=extract,
            metadata=metadata,
            veracity=veracity,
        )
        audit_events.append((
            "remember",
            {"memory_id": memory_id, "bank": "private", "scope": scope, "source_tool": remember_source_tool},
        ))
        return {"index": index, "action": action, "status": "stored", "memory_id": memory_id, "content_preview": content[:100]}

    memory_id = str(payload["memory_id"]).strip()
    if action == "update":
        importance = payload.get("importance")
        ok = beam.update_working(
            memory_id,
            content=payload.get("content"),
            importance=float(importance) if importance is not None else None,
        )
        if not ok:
            raise BatchOperationError("memory_not_found")
        audit_events.append(("update", {"memory_id": memory_id, "bank": "private", "source_tool": remember_source_tool}))
        return {"index": index, "action": action, "status": "updated", "memory_id": memory_id}
    if action == "forget":
        ok = beam.forget_working(memory_id)
        if not ok:
            raise BatchOperationError("memory_not_found")
        audit_events.append(("forget", {"memory_id": memory_id, "bank": "private", "source_tool": remember_source_tool}))
        return {"index": index, "action": action, "status": "deleted", "memory_id": memory_id}

    ok = beam.invalidate(memory_id, replacement_id=payload.get("replacement_id") or None)
    if not ok:
        raise BatchOperationError("memory_not_found")
    audit_events.append(("invalidate", {"memory_id": memory_id, "bank": "private", "source_tool": remember_source_tool}))
    return {"index": index, "action": action, "status": "invalidated", "memory_id": memory_id}


def _dry_run_result(op: dict[str, Any]) -> dict[str, Any]:
    action = op["action"]
    result: dict[str, Any] = {"index": op["index"], "action": action}
    if action == "remember":
        result["status"] = "would_store"
    elif action == "update":
        result.update({"status": "would_update", "memory_id": op["payload"]["memory_id"]})
    elif action == "forget":
        result.update({"status": "would_delete", "memory_id": op["payload"]["memory_id"]})
    else:
        result.update({"status": "would_invalidate", "memory_id": op["payload"]["memory_id"]})
    return result


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
