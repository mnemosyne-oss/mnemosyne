"""Opt-in relevance gate on the recall canonical/surface merge arms.

Default off: explicit recall stays the broad merge the pinned contract
preserves. With MNEMOSYNE_RECALL_CANONICAL_GATE=1 each arm demands
distinctive lexical evidence and contributes at most one row, because
canonical rows score 0.72..1.0 absolutely while the beam arm returns
RRF-scale scores of ~0.016 per voice: ungated, an unrelated canonical row
takes a top slot in every query by scale alone (measured: hit@3 4% vs 38%
on a 43-question set).

Fixtures here DISCRIMINATE filtering from capping: each gated case carries
(a) a row the producer admits but the gate rejects (its only overlap is on
tokens the gate treats as generic: 'preferences'/'recommendations'), and
(b) two distinct rows that qualify, so exactly-one-survives assertions fail
if the one-row cap is removed. Threshold variables are pinned per test so
nothing depends on the ambient process environment.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_prefetch_hardening import (  # noqa: E402
    FakeBeam,
    FakeCanonicalStore,
    _provider,
)

from mnemosyne_hermes import MnemosyneMemoryProvider  # noqa: E402

GATE = "MNEMOSYNE_RECALL_CANONICAL_GATE"


@pytest.fixture(autouse=True)
def _pinned_thresholds(monkeypatch):
    """Pin every variable the gate/producer thresholds resolve through."""
    for key in (
        GATE,
        "MNEMOSYNE_RECALL_CANONICAL_MIN_COVERAGE",
        "MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE",
        "MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS",
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS",
    ):
        monkeypatch.delenv(key, raising=False)


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
    """Canonical arm: an admitted row fails the gate, two qualify, one survives.

    'prefs' passes the producer (its tokens are distinctive under the
    explicit-recall generic set) yet fails the GATE (they are generic under
    the lexical set), so the case pins filtering, not just capping.
    'dac-top' and 'dac-thd' are two DISTINCT qualifying rows: without the
    one-row cap both would survive.
    """
    monkeypatch.setenv(GATE, "1")
    p = _provider([])
    p._beam.canonical = FakeCanonicalStore([
        {"name": "prefs", "body": "Preferences about recommendations for coffee brewing.",
         "category": "model:user"},
        {"name": "dac-top", "body": "Topping D91SE DAC measurements show low noise.",
         "category": "model:user"},
        {"name": "dac-thd", "body": "The D91SE DAC THD report was published.",
         "category": "model:user"},
        {"name": "coffee", "body": "The user drinks filter coffee black.",
         "category": "model:user"},
    ])
    response = _recall(
        p, "which Topping D91SE DAC measures preferences recommendations"
    )
    names = [r.get("canonical_name") for r in response["results"] if r.get("canonical_name")]
    assert names == ["dac-top"]  # exactly one survivor, the higher-scoring qualifier


def test_gate_selects_before_the_producer_shortlist_cap(monkeypatch):
    """A gated row ranked BELOW the ungated cap must still be selectable.

    Five distractors carry only gate-generic overlap ('preferences',
    'recommendations') and rank ahead of the eligible row in the producer's
    ordering, so the ungated top-5 cap buries it; the gated path harvests a
    full ranked window, filters first, and admits the one row that actually
    matches the query.
    """
    fillers = ["coffee brewing", "tea steeping", "bread proofing",
               "vinyl cleaning", "tire pressure"]
    rows = [
        {"name": f"distractor-{i}",
         "body": f"Preferences about recommendations for {filler}.",
         "category": "model:user"}
        for i, filler in enumerate(fillers)
    ]
    # Equal producer score to the distractors (same overlap count and
    # coverage) but listed last, so the stable ranking keeps it at position
    # 6 - beyond the ungated top-5 cap.
    rows.append(
        {"name": "target", "body": "D91SE DAC in balanced mode.", "category": "model:user"}
    )
    query = "Topping D91SE DAC measurements preferences recommendations"
    p = _provider([])
    p._beam.canonical = FakeCanonicalStore(rows)

    # Gate off: the pinned broad merge keeps its shortlist behaviour - the
    # distractors answer, the eligible row is invisible under the cap.
    monkeypatch.delenv(GATE, raising=False)
    names = [r.get("canonical_name") for r in _recall(p, query)["results"]
             if r.get("canonical_name")]
    assert names, "the ungated merge must still answer with producer candidates"
    assert set(names).issubset({f"distractor-{i}" for i in range(5)})
    assert "target" not in names

    # Gate on: the filter runs before any cap, and exactly the eligible row survives.
    monkeypatch.setenv(GATE, "1")
    names = [r.get("canonical_name") for r in _recall(p, query)["results"]
             if r.get("canonical_name")]
    assert names == ["target"]


def test_gate_on_surface_arm_and_beam_rows_untouched(monkeypatch):
    """Surface arm: two qualifying rows survive the producer, one survives the cap."""
    monkeypatch.setenv(GATE, "1")
    p = _provider([
        {"id": "wm1", "content": "the DAC was calibrated in may", "score": 0.016},
        {"id": "wm2", "content": "unrelated grocery list", "score": 0.015},
    ])
    p._beam.canonical = FakeCanonicalStore([])
    surface = FakeBeam([
        {"id": "surf1", "content": "Topping D91SE DAC has a THD of 0.00008", "score": 0.9},
        # Second DISTINCT qualifying row: without the cap both would merge.
        {"id": "surf3", "content": "D91SE THD measured with a DAC adapter", "score": 0.85},
        {"id": "surf2", "content": "surface noise about weather", "score": 0.8},
    ])
    p._shared_surface_read = True
    p._surface_beam = surface
    p._shared_surface_bank = "surface"
    monkeypatch.setattr(p, "_ensure_surface_beam", lambda: None)
    response = _recall(p, "Topping D91SE THD measurement")
    surface_ids = [r["id"] for r in response["results"] if r.get("shared_surface")]
    assert surface_ids == ["surf1"]  # one-row cap on the gated arm
    assert {r["id"] for r in response["results"] if not r.get("shared_surface")} == {"wm1", "wm2"}

    # Gate off: both qualifying surface rows merge - the cap belongs to the
    # gate, not to the surface arm in general.
    monkeypatch.delenv(GATE, raising=False)
    p2 = _provider([])
    p2._beam.canonical = FakeCanonicalStore([])
    p2._shared_surface_read = True
    p2._surface_beam = FakeBeam([
        {"id": "surf1", "content": "Topping D91SE DAC has a THD of 0.00008", "score": 0.9},
        {"id": "surf3", "content": "D91SE THD measured with a DAC adapter", "score": 0.85},
    ])
    p2._shared_surface_bank = "surface"
    monkeypatch.setattr(p2, "_ensure_surface_beam", lambda: None)
    response = _recall(p2, "Topping D91SE THD measurement")
    surface_ids = [r["id"] for r in response["results"] if r.get("shared_surface")]
    assert surface_ids == ["surf1", "surf3"]
