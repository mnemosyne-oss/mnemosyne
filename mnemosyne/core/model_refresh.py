"""LLM-assisted canonical model refresh for Mnemosyne sleep.

This module deliberately does not render or overwrite free-form mental-model
blobs. It asks the sleep LLM for structured candidate updates to canonical
model slots, validates the response, and lets the caller decide whether to
store proposals or apply them.
"""

from __future__ import annotations

import json
import math
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from mnemosyne.core.filters import _SYSTEM_DERIVED_WRITE_CAPABILITY


DEFAULT_MODEL_CATEGORIES: Set[str] = {
    "model:user",
    "model:workflow",
    "model:project",
    "model:agent",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def sleep_model_refresh_enabled() -> bool:
    """Return whether sleep should run model-refresh inference.

    The default follows sleep itself: if a sleep cycle runs, model refresh runs
    too when an LLM path is available. Deployments can set
    MNEMOSYNE_SLEEP_MODEL_REFRESH_ENABLED=false as an emergency brake.
    """

    return _env_bool("MNEMOSYNE_SLEEP_MODEL_REFRESH_ENABLED", True)


def _allowed_categories_from_env() -> Set[str]:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_CATEGORIES", "").strip()
    if not raw:
        return set(DEFAULT_MODEL_CATEGORIES)
    categories = {part.strip() for part in re.split(r"[,\s]+", raw) if part.strip()}
    return categories or set(DEFAULT_MODEL_CATEGORIES)


def _strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # Some models wrap prose around the JSON. Prefer the outermost array/object.
    start_candidates = [i for i in (text.find("["), text.find("{")) if i >= 0]
    if start_candidates:
        start = min(start_candidates)
        end = max(text.rfind("]"), text.rfind("}"))
        if end > start:
            return text[start : end + 1].strip()
    return text


def _coerce_evidence_ids(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    ids: List[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            ids.append(text)
    return ids


def coerce_confidence(value: Any, default: float) -> float:
    """Return value as a finite float clamped to [0.0, 1.0], or default.

    Confidence reaches this module from two unhardened directions: LLM
    JSON (json.loads round-trips NaN and Infinity, so a model can emit
    them as literals) and persisted metadata_json on legacy banks (any
    JSON type, including strings). Non-numeric and non-finite values
    degrade to the caller's default instead of raising. NaN in
    particular must never survive as a float: it compares False against
    every threshold, so downstream gates of the form
    ``confidence < minimum`` silently pass it. bool is rejected before
    the float attempt: it subclasses int, so ``float(True)`` would
    silently read a JSON ``true`` as full confidence. Finite values are
    clamped to the confidence domain: a persisted ``2.0`` must not
    outrank every in-range proposal or reach the canonical store
    unbounded.
    """
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def parse_model_update_proposals(
    raw: str,
    *,
    allowed_categories: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Parse and validate LLM candidate updates for canonical model slots."""

    allowed = allowed_categories or _allowed_categories_from_env()
    try:
        payload = json.loads(_strip_json_fence(raw))
    except Exception:
        return []
    if isinstance(payload, dict):
        payload = payload.get("proposals") or payload.get("updates") or [payload]
    if not isinstance(payload, list):
        return []

    proposals: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        category = str(item.get("category") or "").strip()
        name = str(item.get("name") or "").strip()
        body = str(item.get("body") or "").strip()
        if not category or not name or not body:
            continue
        if category not in allowed:
            continue
        confidence = coerce_confidence(item.get("confidence", 0.0), 0.0)
        if confidence <= 0.0:
            continue
        evidence_ids = _coerce_evidence_ids(item.get("evidence_ids") or item.get("evidence") or [])
        if not evidence_ids:
            continue
        action = str(item.get("action") or "update").strip().lower()
        if action not in {"update", "keep", "ignore"}:
            action = "update"
        proposals.append({
            "category": category,
            "name": name,
            "body": body,
            "confidence": confidence,
            "evidence_ids": evidence_ids,
            "action": action,
            "reason": str(item.get("reason") or "").strip(),
        })
    return proposals


def _format_memory_lines(items: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in items:
        memory_id = str(item.get("id") or "").strip()
        content = str(item.get("content") or "").strip().replace("\n", " ")
        if memory_id and content:
            lines.append(f"- id={memory_id}: {content}")
    return "\n".join(lines)


def build_model_refresh_prompt(
    items: Sequence[Dict[str, Any]],
    *,
    allowed_categories: Optional[Iterable[str]] = None,
) -> str:
    categories = sorted(set(allowed_categories or _allowed_categories_from_env()))
    memories = _format_memory_lines(items)
    return f"""You infer durable canonical model-slot updates during a memory sleep cycle.

Return ONLY JSON. Do not include markdown unless using a JSON code fence.
Return an array of objects with keys: category, name, body, confidence, evidence_ids, action, reason.
Allowed categories: {', '.join(categories)}
Allowed actions: update, keep, ignore.

Rules:
- Propose only stable preferences, identity/profile facts, workflow rules, project models, or agent operating models.
- Do not propose one-off task progress, temporary debugging state, deadlines, issue numbers, commit hashes, or secrets.
- Each proposal must cite at least one evidence id from the input.
- Prefer compact slot bodies that can be stored as canonical facts.
- If nothing is durable enough, return [].

Memories:
{memories}
""".strip()


def infer_model_update_proposals(
    items: Sequence[Dict[str, Any]],
    *,
    allowed_categories: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    """Ask the configured sleep LLM for canonical model-slot proposals."""

    if not sleep_model_refresh_enabled() or not items:
        return []
    allowed = allowed_categories or _allowed_categories_from_env()
    prompt = build_model_refresh_prompt(items, allowed_categories=allowed)

    try:
        from mnemosyne.core import local_llm
    except Exception:
        return []

    raw = None
    attempted_host = False
    try:
        attempted_host, raw = local_llm._try_host_llm(  # internal sibling API used by sleep too
            prompt,
            max_tokens=int(os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_MAX_TOKENS", "2048") or "2048"),
            temperature=float(os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_TEMPERATURE", "0.1") or "0.1"),
        )
    except Exception:
        attempted_host = False
        raw = None

    if raw is None and not attempted_host:
        try:
            raw = local_llm._call_remote_llm(prompt, temperature=0.1)
        except Exception:
            raw = None
    if not raw:
        return []
    return parse_model_update_proposals(raw, allowed_categories=allowed)


def proposal_to_memory_content(proposal: Dict[str, Any]) -> str:
    """Render a pending model-refresh proposal as a compact working-memory row."""

    return (
        "[MODEL_REFRESH_PROPOSAL] "
        f"{proposal.get('category')}::{proposal.get('name')} "
        f"confidence={proposal.get('confidence')}: {proposal.get('body')}"
    )


PROPOSAL_SOURCE = "sleep_model_refresh_proposal"

# Set on the row while a resolution is in flight; see _claim_pending_proposal.
RESOLUTION_TOKEN_KEY = "resolution_token"
RESOLUTION_CLAIMED_AT_KEY = "resolution_claimed_at"


def auto_apply_enabled() -> bool:
    """Whether sleep may apply validated proposals immediately.

    This defaults ON because model refresh is a sleep-time automation, not a
    human approval queue. Operators can disable it as an emergency brake with
    MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY=false.

    Resolved through the central hot-reload config (config.yaml > env > default)
    so a maintenance operation observes one consistent value at its boundary.
    When verified Dream mode is active (``dream_active``), auto-apply is forced
    off: Dream owns canonical mutations and the sleep path must not race it by
    applying model-refresh proposals directly.
    """
    from mnemosyne.core.config import get_config

    config = get_config()
    if config.get_bool("dream_active", False):
        return False
    return config.get_bool("sleep_model_refresh_auto_apply", True)


def auto_apply_min_confidence() -> float:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_AUTO_APPLY_MIN_CONFIDENCE", "0.90")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.90


def auto_apply_min_evidence() -> int:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_MIN_EVIDENCE", "2")
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2


def auto_apply_conflict_min_confidence() -> float:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_CONFIDENCE", "0.98")
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return 0.98


def auto_apply_conflict_min_evidence() -> int:
    raw = os.environ.get("MNEMOSYNE_SLEEP_MODEL_REFRESH_CONFLICT_MIN_EVIDENCE", "3")
    try:
        return max(auto_apply_min_evidence(), int(raw))
    except (TypeError, ValueError):
        return 3


_EPHEMERAL_RE = re.compile(
    r"(\bpr\s*#?\d+\b|\bissue\s*#?\d+\b|\bcommit\s+[0-9a-f]{7,}\b|"
    r"\b[0-9a-f]{12,}\b|\btemporary\b|\btransient\b|\bone[- ]off\b|"
    r"\bdebugging state\b|\btask progress\b|\bphase\s+\d+\s+done\b|"
    r"\bapi[_-]?key\b|\bpassword\b|\bsecret\b|\btoken\b)",
    re.IGNORECASE,
)


def _is_ephemeral_or_sensitive(metadata: Dict[str, Any]) -> bool:
    haystack = " ".join(
        str(metadata.get(key) or "")
        for key in ("category", "name", "body", "reason")
    )
    return bool(_EPHEMERAL_RE.search(haystack))


def prepare_proposal_metadata(proposal: Dict[str, Any], *, source_wm_ids: Sequence[str]) -> Dict[str, Any]:
    """Return persisted metadata for a newly inferred proposal."""

    metadata = dict(proposal)
    metadata.setdefault("action", "update")
    metadata["status"] = "pending"
    metadata["source_wm_ids"] = list(source_wm_ids)
    return metadata


def _proposal_row(beam, proposal_id: str) -> Optional[Dict[str, Any]]:
    row = beam.conn.execute(
        "SELECT id, content, source, metadata_json, timestamp, consolidated_at "
        "FROM working_memory WHERE id = ? AND source = ?",
        (proposal_id, PROPOSAL_SOURCE),
    ).fetchone()
    return dict(row) if row is not None else None


def _parse_metadata(raw: Any) -> Dict[str, Any]:
    try:
        metadata = json.loads(raw or "{}")
    except Exception:
        metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def _load_metadata(row: Dict[str, Any]) -> Dict[str, Any]:
    return _parse_metadata(row.get("metadata_json"))


def _is_resolved_proposal_metadata(raw: Any) -> bool:
    return _parse_metadata(raw).get("status") in ("applied", "rejected")


def _dump_metadata(metadata: Dict[str, Any]) -> str:
    """Serialise metadata to JSON that SQLite's json_valid() accepts.

    Proposal metadata can carry non-finite floats (a NaN confidence is exactly
    what the stored-confidence hardening path must persist rather than drop),
    and ``json.dumps`` emits those as bare ``NaN``. SQLite treats that as
    invalid JSON, which would make the claim guard unable to see its own token.
    Non-finite floats therefore become null.
    """

    def _finite(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: _finite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_finite(item) for item in value]
        return value

    return json.dumps(_finite(metadata), sort_keys=True)


def _save_metadata(
    beam,
    proposal_id: str,
    metadata: Dict[str, Any],
    *,
    token: Optional[str] = None,
) -> bool:
    """Persist the proposal outcome and invalidate its old recall eligibility.

    When ``token`` is given the write only lands if this caller still holds the
    claim it took in ``_claim_pending_proposal``. The canonical write and the
    terminal status then belong to one guarded transition: a concurrent
    resolver cannot overwrite the outcome that was recorded first.
    """
    sql = "UPDATE working_memory SET metadata_json = ? WHERE id = ? AND source = ?"
    params: List[Any] = [_dump_metadata(metadata), proposal_id, PROPOSAL_SOURCE]
    if token is not None:
        sql += (
            " AND json_valid(metadata_json)"
            " AND json_extract(metadata_json, '$.resolution_token') = ?"
        )
        params.append(token)
    cursor = beam.conn.execute(sql, params)
    beam.conn.commit()
    invalidate = getattr(beam, "_invalidate_query_cache_after_commit", None)
    if callable(invalidate):
        invalidate("model_refresh")
    return cursor.rowcount == 1


# A resolution may only start from `pending` and must be claimed exactly once.
# Malformed or NULL metadata is read as pending everywhere else in this module
# (``_parse_metadata`` -> {}), so the SQL guard mirrors that instead of raising.
_PENDING_CLAIM_GUARD_SQL = (
    " AND (metadata_json IS NULL"
    " OR NOT json_valid(metadata_json)"
    " OR COALESCE(json_extract(metadata_json, '$.status'), 'pending') = 'pending')"
    " AND (metadata_json IS NULL"
    " OR NOT json_valid(metadata_json)"
    " OR json_extract(metadata_json, '$.resolution_token') IS NULL"
    " OR julianday(COALESCE(json_extract(metadata_json,"
    " '$.resolution_claimed_at'), '1970-01-01'))"
    " < julianday('now', '-15 minutes'))"
)


def _claim_pending_proposal(
    beam,
    proposal_id: str,
    metadata: Dict[str, Any],
) -> Optional[str]:
    """Atomically claim one still-pending proposal for resolution.

    Returns the claim token, or ``None`` when the row was resolved (or claimed)
    between the read and this call. The claim keeps ``status`` at ``pending`` so
    every existing reader — recall eligibility, listings, audits — is unchanged
    while the side effect is in flight.
    """
    token = uuid.uuid4().hex
    claimed = dict(metadata)
    claimed[RESOLUTION_TOKEN_KEY] = token
    claimed[RESOLUTION_CLAIMED_AT_KEY] = datetime.now(timezone.utc).isoformat()
    cursor = beam.conn.execute(
        "UPDATE working_memory SET metadata_json = ? WHERE id = ? AND source = ?"
        + _PENDING_CLAIM_GUARD_SQL,
        (_dump_metadata(claimed), proposal_id, PROPOSAL_SOURCE),
    )
    beam.conn.commit()
    return token if cursor.rowcount == 1 else None


def _release_claim(
    beam,
    proposal_id: str,
    metadata: Dict[str, Any],
    token: str,
) -> None:
    """Drop a claim after the side effect failed, so the row can be retried."""
    beam.conn.execute(
        "UPDATE working_memory SET metadata_json = ? WHERE id = ? AND source = ?"
        " AND json_valid(metadata_json)"
        " AND json_extract(metadata_json, '$.resolution_token') = ?",
        (_dump_metadata(metadata), proposal_id, PROPOSAL_SOURCE, token),
    )
    beam.conn.commit()


def list_model_refresh_proposals(
    beam,
    *,
    status: str = "pending",
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """List sleep model-refresh proposals stored in working memory."""

    rows = beam.conn.execute(
        "SELECT id, content, source, metadata_json, timestamp, consolidated_at "
        "FROM working_memory WHERE source = ? ORDER BY timestamp DESC LIMIT ?",
        (PROPOSAL_SOURCE, int(limit)),
    ).fetchall()
    out: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        metadata = _load_metadata(item)
        item["metadata"] = metadata
        if status and status != "all" and metadata.get("status", "pending") != status:
            continue
        out.append(item)
    return out


def apply_model_refresh_proposal(
    beam,
    proposal_id: str,
    *,
    owner_id: str = "default",
    validator: str = "",
    auto_applied: bool = False,
) -> Dict[str, Any]:
    """Apply one pending proposal to CanonicalStore and mark it applied."""

    row = _proposal_row(beam, proposal_id)
    if row is None:
        raise ValueError(f"model refresh proposal not found: {proposal_id}")
    metadata = _load_metadata(row)
    if metadata.get("status", "pending") != "pending":
        return {"status": metadata.get("status"), "proposal_id": proposal_id, "metadata": metadata}
    if metadata.get("action", "update") != "update":
        raise ValueError("only update proposals can be applied")
    for key in ("category", "name", "body"):
        if not str(metadata.get(key) or "").strip():
            raise ValueError(f"proposal metadata missing {key}")

    from mnemosyne.core.canonical import CanonicalStore

    token = _claim_pending_proposal(beam, proposal_id, metadata)
    if token is None:
        # Another connection resolved (or claimed) the row after our read.
        # Report its outcome instead of overwriting it.
        current = _load_metadata(_proposal_row(beam, proposal_id) or row)
        return {"status": current.get("status"), "proposal_id": proposal_id, "metadata": current}

    store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
    try:
        canonical = store.remember(
            owner_id,
            metadata["category"],
            metadata["name"],
            metadata["body"],
            source="sleep_model_refresh",
            confidence=coerce_confidence(metadata.get("confidence"), 0.5),
            _write_kind=_SYSTEM_DERIVED_WRITE_CAPABILITY,
        )
    except Exception:
        _release_claim(beam, proposal_id, metadata, token)
        raise
    metadata["status"] = "applied"
    metadata["applied_by"] = validator or "system"
    metadata["applied_owner_id"] = owner_id
    metadata["canonical_id"] = canonical.get("id")
    metadata["auto_applied"] = bool(auto_applied)
    if not _save_metadata(beam, proposal_id, metadata, token=token):
        raise RuntimeError(
            f"model refresh proposal {proposal_id} lost its resolution claim "
            "after the canonical write"
        )
    return {"status": "applied", "proposal_id": proposal_id, "canonical": canonical, "metadata": metadata}


def reject_model_refresh_proposal(
    beam,
    proposal_id: str,
    *,
    reason: str = "",
    validator: str = "",
) -> Dict[str, Any]:
    """Reject one pending proposal without touching canonical facts."""

    row = _proposal_row(beam, proposal_id)
    if row is None:
        raise ValueError(f"model refresh proposal not found: {proposal_id}")
    metadata = _load_metadata(row)
    if metadata.get("status", "pending") != "pending":
        # A resolved proposal keeps its outcome: rejecting an applied one
        # must not rewrite the recorded outcome.
        return {"status": metadata.get("status"), "proposal_id": proposal_id, "metadata": metadata}
    token = _claim_pending_proposal(beam, proposal_id, metadata)
    if token is None:
        current = _load_metadata(_proposal_row(beam, proposal_id) or row)
        return {"status": current.get("status"), "proposal_id": proposal_id, "metadata": current}
    metadata["status"] = "rejected"
    metadata["rejected_by"] = validator or "system"
    metadata["rejection_reason"] = reason
    if not _save_metadata(beam, proposal_id, metadata, token=token):
        raise RuntimeError(
            f"model refresh proposal {proposal_id} lost its resolution claim"
        )
    return {"status": "rejected", "proposal_id": proposal_id, "metadata": metadata}


def maybe_auto_apply_model_refresh_proposal(
    beam,
    proposal_id: str,
    *,
    owner_id: str = "default",
) -> bool:
    """Validate and automatically resolve one sleep model-refresh proposal.

    Normal product behavior is automated: strong durable candidates are applied;
    weak, ephemeral, unsupported, or unsafe candidates are rejected. Pending rows
    remain only when auto-apply is explicitly disabled by deployment config.
    """

    if not auto_apply_enabled():
        return False
    row = _proposal_row(beam, proposal_id)
    if row is None:
        return False
    metadata = _load_metadata(row)
    if metadata.get("status", "pending") != "pending":
        return False
    if metadata.get("action", "update") != "update":
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="non-update model-refresh action is not durable canonical truth",
            validator="sleep_model_refresh_auto_validation",
        )
        return False
    confidence = coerce_confidence(metadata.get("confidence"), 0.0)

    evidence_ids = [str(x) for x in (metadata.get("evidence_ids") or []) if str(x).strip()]
    source_wm_ids = {str(x) for x in (metadata.get("source_wm_ids") or []) if str(x).strip()}
    if len(evidence_ids) < auto_apply_min_evidence():
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="insufficient evidence for automated model refresh",
            validator="sleep_model_refresh_auto_validation",
        )
        return False
    if source_wm_ids and not set(evidence_ids).issubset(source_wm_ids):
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="proposal cites evidence outside the sleep batch",
            validator="sleep_model_refresh_auto_validation",
        )
        return False
    if _is_ephemeral_or_sensitive(metadata):
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="ephemeral or sensitive content is not eligible for canonical model refresh",
            validator="sleep_model_refresh_auto_validation",
        )
        return False
    if confidence < auto_apply_min_confidence():
        reject_model_refresh_proposal(
            beam, proposal_id,
            reason="confidence below automated model-refresh threshold",
            validator="sleep_model_refresh_auto_validation",
        )
        return False

    try:
        from mnemosyne.core.canonical import CanonicalStore
        store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
        current = store.recall(
            owner_id,
            str(metadata.get("category") or ""),
            str(metadata.get("name") or ""),
        )
    except Exception:
        current = None
    if current is not None and str(current.get("body") or "").strip() != str(metadata.get("body") or "").strip():
        if confidence < auto_apply_conflict_min_confidence() or len(evidence_ids) < auto_apply_conflict_min_evidence():
            reject_model_refresh_proposal(
                beam, proposal_id,
                reason="conflicts with current canonical slot without enough supersession evidence",
                validator="sleep_model_refresh_auto_validation",
            )
            return False

    apply_model_refresh_proposal(
        beam,
        proposal_id,
        owner_id=owner_id,
        validator="sleep_model_refresh_auto_apply",
        auto_applied=True,
    )
    return True
