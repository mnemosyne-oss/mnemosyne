"""Golden check: the shared prefetch module and the Hermes provider render
identical context blocks, pinned to an exact expected string."""

from __future__ import annotations

import sqlite3
import types

import pytest

import hermes_memory_provider
from hermes_memory_provider import MnemosyneMemoryProvider, PrefetchProfile
from mnemosyne.core.prefetch import identity_rows, render_bank_source, render_identity


class FakeLedger:
    enabled = False


class FakeBeam:
    def __init__(self, rows):
        self.rows = rows
        self.recall_calls = []

    def recall(self, **kwargs):
        self.recall_calls.append(kwargs)
        return self.rows


ROWS = [
    {"content": "User prefers dark mode interfaces", "source": "preference",
     "score": 0.9, "importance": 0.9, "fts_score": 0.8,
     "timestamp": "2026-09-01T10:00:00"},
    {"content": "User prefers dark mode interfaces for IDE work", "source": "preference",
     "score": 0.85, "importance": 0.8, "fts_score": 0.7,
     "timestamp": "2026-09-02T11:00:00"},
    {"content": "still", "source": "fact", "score": 0.8, "importance": 0.7,
     "fts_score": 0.6},
    {"content": "[ASSISTANT] I will use dark theme", "source": "conversation",
     "score": 0.8, "importance": 0.5, "fts_score": 0.6},
    {"content": "[USER] hi there", "source": "conversation",
     "score": 0.5, "importance": 0.5, "fts_score": 0.05},
    {"content": "Deploys happen Fridays", "source": "insight",
     "score": 0.7, "importance": 0.8, "fts_score": 0.5,
     "timestamp": "2026-09-03T09:30:00", "trust_tier": "INFERRED"},
]

GOLDEN_BLOCK = (
    "## Mnemosyne Context\n"
    "  [2026-09-01T10:00] (importance 0.90, source preference) "
    "User prefers dark mode interfaces\n"
    "  [2026-09-03T09:30] (importance 0.80, source insight) "
    "[INFERRED] Deploys happen Fridays"
)


@pytest.fixture(autouse=True)
def _no_char_limit(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", raising=False)


def test_render_bank_source_golden():
    beam = FakeBeam(ROWS)
    out = render_bank_source(beam, "dark mode deploys", "s1", PrefetchProfile(name="golden"))
    assert out == GOLDEN_BLOCK
    (kwargs,) = beam.recall_calls
    assert kwargs["top_k"] == 16
    assert kwargs["temporal_weight"] == 0.2
    assert kwargs["temporal_halflife"] == 48


def test_provider_delegation_matches_shared_module():
    beam = FakeBeam(ROWS)
    provider = object.__new__(MnemosyneMemoryProvider)
    provider._beam = beam
    provider._verbatim_ledger = FakeLedger()
    provider._active_session_id = "s1"
    profile = PrefetchProfile(name="golden")

    delegated = provider._prefetch_bank("dark mode deploys", "s1", profile)
    direct = render_bank_source(beam, "dark mode deploys", "s1", profile)

    assert delegated == direct == GOLDEN_BLOCK


def test_moved_names_reexported_from_provider():
    assert hermes_memory_provider._resolve_profile("general").name == "general"
    assert hermes_memory_provider._is_low_quality_prefetch("still")
    assert hermes_memory_provider._format_prefetch_content("a b c", 0) == "a b c"
    assert hermes_memory_provider._prefetch_tokens("[USER] prefers dark mode")
    assert hermes_memory_provider._semantic_dedup_prefetch([]) == []
    assert hermes_memory_provider._prefetch_content_char_limit() == 0
    assert hermes_memory_provider._PREFETCH_TOP_K == 5
    assert "general" in hermes_memory_provider._BUILTIN_PROFILES


def test_identity_roundtrip():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE working_memory (content TEXT, importance REAL, "
        "timestamp TEXT, source TEXT, session_id TEXT)"
    )
    conn.execute(
        "INSERT INTO working_memory VALUES "
        "('Likes oolong tea', 0.9, '2026-09-01T10:00:00', 'identity', 's1')"
    )
    conn.execute(
        "INSERT INTO working_memory VALUES "
        "('Other session identity', 0.9, '2026-09-01T10:00:00', 'identity', 's2')"
    )
    beam = types.SimpleNamespace(conn=conn, session_id="s1")

    rows = identity_rows(beam)
    assert [r["content"] for r in rows] == ["Likes oolong tea"]

    block = render_identity(rows, [], PrefetchProfile(name="golden"))
    assert "[IDENTITY] Likes oolong tea" in block
    assert "Other session" not in block
