"""Contract tests for provenance-preserving supplemental evidence packs."""
from __future__ import annotations

import pytest

from mnemosyne.core.evidence_packs import build_evidence_pack


def _row(memory_id: str, session_id: str | None, timestamp: object, *, score: float = 0.5) -> dict:
    row = {"id": memory_id, "timestamp": timestamp, "score": score, "content": memory_id}
    if session_id is not None:
        row["session_id"] = session_id
    return row


def test_primary_ranking_is_preserved_and_candidates_do_not_mutate_inputs():
    primary = [_row("p1", "session-primary", "2024-05-03", score=0.9)]
    candidates = [
        _row("p1", "session-primary", "2024-05-03", score=0.9),
        _row("c2", "session-2", "2024-05-02", score=0.8),
        _row("c3", "session-3", "2024-05-01", score=0.7),
    ]

    packed = build_evidence_pack(primary, candidates, max_items=2)

    assert packed["primary"] == primary
    assert [row["id"] for row in packed["evidence_pack"]] == ["c3", "c2"]
    assert [row["evidence_rank"] for row in packed["evidence_pack"]] == [3, 2]
    assert "evidence_rank" not in candidates[1]


def test_only_first_candidate_per_session_is_selected():
    packed = build_evidence_pack(
        [],
        [
            _row("first", "same-session", "2024-05-03"),
            _row("later", "same-session", "2024-05-04"),
            _row("other", "other-session", "2024-05-05"),
        ],
        max_items=5,
    )

    assert [row["id"] for row in packed["evidence_pack"]] == ["first", "other"]
    assert [row["evidence_rank"] for row in packed["evidence_pack"]] == [1, 3]


def test_missing_session_id_is_dropped_to_preserve_provenance():
    packed = build_evidence_pack(
        [],
        [_row("a", None, "2024-05-01"), _row("b", None, "2024-05-02")],
        max_items=2,
    )

    assert packed["evidence_pack"] == []


def test_primary_session_and_duplicate_candidate_ids_are_not_repeated():
    packed = build_evidence_pack(
        [_row("primary", "already-present", "2024-05-01")],
        [
            _row("same-session", "already-present", "2024-05-02"),
            _row("duplicate", "new-session", "2024-05-03"),
            _row("duplicate", "other-session", "2024-05-04"),
            _row("kept", "kept-session", "2024-05-05"),
        ],
    )

    assert [row["id"] for row in packed["evidence_pack"]] == ["duplicate", "kept"]


def test_custom_group_key_and_numeric_timestamp_are_supported():
    rows = [
        {"id": "later", "source": "same", "timestamp": 10},
        {"id": "first", "source": "same", "timestamp": 9},
        {"id": "other", "source": "other", "timestamp": 8},
    ]
    packed = build_evidence_pack([], rows, group_key="source")

    assert [row["id"] for row in packed["evidence_pack"]] == ["other", "later"]


def test_unknown_timestamps_sort_after_known_chronology_deterministically():
    packed = build_evidence_pack(
        [],
        [
            _row("iso", "iso", "2024-05-01T00:00:00Z"),
            _row("bad", "bad", "not-a-date"),
            {"id": "missing", "session_id": "missing"},
            _row("epoch", "epoch", 10),
        ],
    )

    assert [row["id"] for row in packed["evidence_pack"]] == ["epoch", "iso", "bad", "missing"]


def test_candidate_without_stable_id_is_not_emitted():
    packed = build_evidence_pack([], [{"session_id": "s", "timestamp": "2024-05-01"}])
    assert packed["evidence_pack"] == []


def test_zero_capacity_returns_no_supplemental_evidence():
    packed = build_evidence_pack([], [_row("c", "s", "2024-05-01")], max_items=0)
    assert packed == {"primary": [], "evidence_pack": []}


def test_negative_capacity_is_rejected():
    with pytest.raises(ValueError, match="max_items"):
        build_evidence_pack([], [], max_items=-1)
