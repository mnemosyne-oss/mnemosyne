"""Regression tests for opt-in recall explain traces (#322)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def test_recall_default_return_shape_unchanged(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam.remember("Alice prefers Vim editor", source="pref", importance=0.7)

    results = beam.recall("Alice Vim", top_k=5)

    assert isinstance(results, list)
    assert results
    assert all(isinstance(row, dict) for row in results)


def test_recall_explain_returns_structured_payload(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam.remember("Alice prefers Vim editor", source="pref", importance=0.7)

    payload = beam.recall("Alice Vim", top_k=5, explain=True)

    assert payload["query"] == "Alice Vim"
    assert payload["top_k"] == 5
    assert payload["engine"] == "linear"
    assert isinstance(payload["results"], list)
    explain = payload["explain"]
    assert set(explain) >= {"filters", "weights", "embedding", "stages", "candidates", "truncation"}
    json.dumps(payload)


def test_explain_contains_stage_counts_and_topk_truncation(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    for i in range(4):
        beam.remember(f"Alice project note {i}", source="note", importance=0.5)

    payload = beam.recall("Alice project", top_k=1, explain=True)
    explain = payload["explain"]

    assert explain["stages"]
    assert explain["truncation"]["top_k"] == 1
    assert explain["truncation"]["dropped_count"] >= 1
    assert any(c["drop_reason"] == "top_k_truncated" for c in explain["candidates"])


def test_explain_candidate_has_source_path_and_scores(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam.remember("Alice keeps architecture notes", source="note", importance=0.6)

    payload = beam.recall("Alice architecture", top_k=5, explain=True)
    candidates = payload["explain"]["candidates"]

    assert candidates
    candidate = candidates[0]
    assert candidate["source_path"] in {"wm_fts", "wm_vec", "wm_fallback", "em_fts", "em_vec", "em_fallback", "memoria", "memoria_source"}
    assert set(candidate["scores"]) >= {"keyword", "fts", "dense", "importance", "recency_decay", "final"}


def test_explain_does_not_leak_cross_session_filtered_rows(temp_db):
    foreign = BeamMemory(session_id="foreign", db_path=temp_db)
    foreign.remember("SECRET foreign Alice content", source="note", importance=0.9)
    local = BeamMemory(session_id="local", db_path=temp_db)
    local.remember("Alice local visible content", source="note", importance=0.5)

    payload = local.recall("Alice", top_k=5, explain=True)
    serialized_explain = json.dumps(payload["explain"])

    assert "SECRET foreign" not in serialized_explain


def test_explain_has_no_raw_sql_or_embedding_values(temp_db):
    beam = BeamMemory(session_id="s1", db_path=temp_db)
    beam.remember("Alice prefers Vim editor", source="pref", importance=0.7)

    payload = beam.recall("Alice Vim", top_k=5, explain=True)
    serialized_explain = json.dumps(payload["explain"])

    assert "SELECT " not in serialized_explain.upper()
    assert " MATCH " not in serialized_explain.upper()
    assert "embedding" in payload["explain"]
    assert "values" not in payload["explain"]["embedding"]
