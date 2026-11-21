"""Opt-in relevance gate on the recall canonical/surface merge arms.

Default off: explicit recall stays the broad merge the pinned contract
preserves. With MNEMOSYNE_RECALL_CANONICAL_GATE=1 each arm demands
distinctive lexical evidence and contributes at most one row, because
canonical rows score 0.72..1.0 absolutely while the beam arm returns
RRF-scale scores of ~0.016 per voice: ungated, an unrelated canonical row
takes a top slot in every query by scale alone (measured: hit@3 4% vs 38%
on a 43-question set).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_prefetch_hardening import (  # noqa: E402
    FakeBeam,
    FakeCanonicalStore,
    _provider,
)

from mnemosyne_hermes import MnemosyneMemoryProvider  # noqa: E402

GATE = "MNEMOSYNE_RECALL_CANONICAL_GATE"


def _recall(p, query):
    return json.loads(p.handle_tool_call("mnemosyne_recall", {"query": query, "limit": 5}))


def test_default_keeps_the_broad_merge(monkeypatch):
    monkeypatch.delenv(GATE, raising=False)
    p = _provider([])
    p._beam.canonical = FakeCanonicalStore([
        {"name": "geology-a", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
        {"name": "geology-b", "body": "Caldrin studies coastal erosion.", "category": "model:user"},
    ])
    response = _recall(p, "Caldrin research lens")
    assert {r.get("canonical_name") for r in response["results"]} == {"geology-a", "geology-b"}


def test_gate_drops_unrelated_and_caps_at_one(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    p = _provider([])
    p._beam.canonical = FakeCanonicalStore([
        {"name": "coffee", "body": "The user drinks filter coffee black.", "category": "model:user"},
        {"name": "dac", "body": "User uses the Topping D91SE DAC for measurements.", "category": "model:user"},
    ])
    response = _recall(p, "which Topping D91SE DAC measures")
    names = [r.get("canonical_name") for r in response["results"] if r.get("canonical_name")]
    assert names == ["dac"]


def test_gate_on_surface_arm_and_beam_rows_untouched(monkeypatch):
    monkeypatch.setenv(GATE, "1")
    p = _provider([
        {"id": "wm1", "content": "the DAC was calibrated in may", "score": 0.016},
        {"id": "wm2", "content": "unrelated grocery list", "score": 0.015},
    ])
    p._beam.canonical = FakeCanonicalStore([])
    surface = FakeBeam([
        {"id": "surf1", "content": "Topping D91SE DAC has a THD of 0.00008", "score": 0.9},
        {"id": "surf2", "content": "surface noise about weather", "score": 0.8},
    ])
    p._shared_surface_read = True
    p._surface_beam = surface
    p._shared_surface_bank = "surface"
    monkeypatch.setattr(p, "_ensure_surface_beam", lambda: None)
    response = _recall(p, "Topping D91SE THD measurement")
    surface_ids = [r["id"] for r in response["results"] if r.get("shared_surface")]
    assert surface_ids == ["surf1"]
    assert {r["id"] for r in response["results"] if not r.get("shared_surface")} == {"wm1", "wm2"}
