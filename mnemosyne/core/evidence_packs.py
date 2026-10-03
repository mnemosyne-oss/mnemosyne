"""Provenance-preserving assembly of supplemental retrieval evidence.

This module intentionally does not query storage or change `BeamMemory.recall()`.
Callers provide two already-scoped result lists:

* `primary`: the normal Top-K ranking shown to the caller;
* `candidates`: a wider, separately retrieved pool.

The primary ranking is copied unchanged. The pack contains only candidates from
sessions absent from primary, at most one per explicit session/source group, and
is ordered chronologically for consumption. A caller must annotate every row
with a stable `session_id` (or choose a different `group_key`). Rows without a
resolvable group are dropped: provenance is required for supplemental evidence.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from datetime import datetime, timezone
from numbers import Real
from typing import Any


Result = Mapping[str, Any]


def _row_identity(row: Result) -> tuple[str, str]:
    """Return a tier-qualified identity to avoid cross-table ID collisions."""
    return (str(row.get("tier", "")), str(row["id"]))


def _chronological_key(row: Mapping[str, Any]) -> tuple[int, float, int]:
    """Normalize ISO-8601 or numeric timestamps without raising on bad input."""
    value = row.get("timestamp")
    if isinstance(value, Real) and not isinstance(value, bool):
        return (0, float(value), row["evidence_rank"])
    if isinstance(value, str) and value:
        try:
            normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return (0, parsed.timestamp(), row["evidence_rank"])
        except ValueError:
            pass
    # Unknown timestamps remain deterministic, after known chronology.
    return (1, 0.0, row["evidence_rank"])


def build_evidence_pack(
    primary: Iterable[Result],
    candidates: Iterable[Result],
    *,
    max_items: int = 5,
    group_key: str = "session_id",
) -> dict[str, list[dict[str, Any]]]:
    """Return an unchanged primary ranking plus compact supplemental evidence.

    `candidates` must be in retrieval-rank order.  The first candidate from
    each group wins selection; selected rows retain that rank as
    ``evidence_rank`` and are then ordered by timestamp for chronological
    consumption.  The function has no database side effects and does not
    mutate either input list or its row dictionaries.
    """
    if max_items < 0:
        raise ValueError("max_items must be non-negative")

    primary_rows = [deepcopy(dict(row)) for row in primary]
    primary_ids = {_row_identity(row) for row in primary_rows if row.get("id") is not None}
    seen_ids = set(primary_ids)
    seen_groups = {
        ("group", str(row[group_key]))
        for row in primary_rows
        if row.get(group_key) not in (None, "")
    }
    selected: list[dict[str, Any]] = []

    for candidate_rank, raw_row in enumerate(candidates, start=1):
        if len(selected) >= max_items:
            break
        row_id = raw_row.get("id")
        # A pack needs a stable provenance handle. Do not invent one.
        if row_id in (None, ""):
            continue
        row_identity = _row_identity(raw_row)
        if row_identity in seen_ids:
            continue
        group_value = raw_row.get(group_key)
        # Evidence packs must preserve session provenance. Unknown tiers or
        # incomplete rows fail closed rather than appearing as unique groups.
        if group_value in (None, ""):
            continue
        group = ("group", str(group_value))
        if group in seen_groups:
            continue
        row = deepcopy(dict(raw_row))
        seen_ids.add(row_identity)
        seen_groups.add(group)
        row["evidence_rank"] = candidate_rank
        selected.append(row)

    selected.sort(key=_chronological_key)
    return {"primary": primary_rows, "evidence_pack": selected}
