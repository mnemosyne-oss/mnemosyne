"""Deterministic Polish-language recall acceptance harness."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory


CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "polish_recall_cases.json").read_text()
)


def test_polish_recall_topk_and_abstention(tmp_path, monkeypatch):
    if os.environ.get("MNEMOSYNE_RUN_POLISH_RECALL_HARNESS") != "1":
        pytest.skip("set MNEMOSYNE_RUN_POLISH_RECALL_HARNESS=1 for multilingual embedding evaluation")
    if not beam_module._embeddings.available():
        pytest.skip("configured embedding backend is unavailable")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_CROSS_SESSION", "0")
    memory = BeamMemory(session_id="polish-eval", db_path=tmp_path / "memory.db")

    ids = {}
    for record in CASES["records"]:
        ids[record["key"]] = memory.remember(
            record["content"], source="polish-harness", importance=0.8
        )

    top1_hits = 0
    top3_hits = 0
    misses = []
    for case in CASES["queries"]:
        results = memory.recall(case["query"], top_k=3)
        result_ids = [row["id"] for row in results]
        expected_id = ids[case["expected"]]
        top1_hits += bool(result_ids and result_ids[0] == expected_id)
        top3_hits += expected_id in result_ids
        if expected_id not in result_ids:
            misses.append({"query": case["query"], "returned": result_ids})

    false_positives = []
    for query in CASES["negative_queries"]:
        results = memory.recall(query, top_k=3)
        if results:
            false_positives.append({"query": query, "returned": [row["id"] for row in results]})

    positives = len(CASES["queries"])
    print(json.dumps({
        "positive_total": positives,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "negative_total": len(CASES["negative_queries"]),
        "false_positives": len(false_positives),
    }, sort_keys=True))
    assert top3_hits == positives, {"top3_hits": top3_hits, "misses": misses}
    assert top1_hits / positives >= 0.90, {"top1_hits": top1_hits, "total": positives}
    assert false_positives == []
