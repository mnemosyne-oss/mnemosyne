"""Tests for selectable prefetch profiles + the generic source-extension hook.

The default `general` profile is the normal conservative injection profile;
other profiles only change the documented knobs. The source registry lets a
caller merge extra retrieval inputs without the core knowing what they are.
"""
from __future__ import annotations

import pytest

from hermes_memory_provider import (
    MnemosyneMemoryProvider,
    PrefetchProfile,
    _resolve_profile,
    register_profile,
)


class FakeBeam:
    """Records the kwargs recall() was called with; returns fixed results."""
    author_id = "test-author"

    def __init__(self, results):
        self.results = results
        self.last_kwargs = None

    def recall(self, **kwargs):
        self.last_kwargs = kwargs
        return self.results


def _provider(profile_name, results):
    p = MnemosyneMemoryProvider()
    p._beam = FakeBeam(results)
    p._prefetch_profile = profile_name
    return p


# --- profile resolution ------------------------------------------------------

def test_unknown_profile_falls_back_to_general():
    assert _resolve_profile("does-not-exist").name == "general"
    assert _resolve_profile(None).name == "general"


def test_general_is_the_default_and_passes_no_tuning_weights():
    p = _provider("general", [
        {"content": "Paris is the capital of France", "timestamp": "2026-05-14T12:00:00Z",
         "importance": 0.9, "score": 0.9, "keyword_score": 0.9, "trust_tier": "STATED"},
    ])
    block = p.prefetch("capital of France")
    assert block.startswith("## Mnemosyne Context")
    assert "Paris is the capital of France" in block
    # general leaves recall()'s own weighting defaults untouched
    assert "importance_weight" not in p._beam.last_kwargs
    assert p._beam.last_kwargs["temporal_weight"] == 0.2


def test_social_chat_passes_its_tuning_to_recall():
    p = _provider("social-chat", [
        {"content": "the team ships on Friday", "timestamp": "2026-05-14T12:00:00Z",
         "importance": 0.9, "score": 0.4, "keyword_score": 0.4, "trust_tier": "STATED"},
    ])
    p.prefetch("when do we ship")
    assert p._beam.last_kwargs["importance_weight"] == 0.6
    assert p._beam.last_kwargs["temporal_weight"] == 0.35
    assert p._beam.last_kwargs["temporal_halflife"] == 24


# --- generic source hook -----------------------------------------------------

def test_registered_source_is_merged():
    p = _provider("general", [
        {"content": "Paris is the capital of France", "timestamp": "2026-05-14T12:00:00Z",
         "importance": 0.9, "score": 0.9, "keyword_score": 0.9, "trust_tier": "STATED"},
    ])
    register_profile(PrefetchProfile(name="t-merge", sources=("bank", "dummy")))
    p._prefetch_profile = "t-merge"
    p.register_prefetch_source(
        "dummy", lambda q, *, session_id="": [{"content": "Lake Baikal is the deepest lake"}])
    block = p.prefetch("geography")
    assert "Paris is the capital of France" in block
    assert "Lake Baikal is the deepest lake" in block
    assert "## Context (dummy)" in block


def test_bank_source_cannot_be_overridden():
    p = _provider("general", [])
    p.register_prefetch_source("bank", lambda q, *, session_id="": "nope")
    assert "bank" not in p._prefetch_sources


def test_dedup_collapses_duplicate_content_across_sources():
    p = _provider("general", [
        {"content": "Mount Everest is the tallest mountain", "timestamp": "2026-05-14T12:00:00Z",
         "importance": 0.9, "score": 0.9, "keyword_score": 0.9, "trust_tier": "STATED"},
    ])
    register_profile(PrefetchProfile(name="t-dedup", sources=("bank", "dummy"), dedup=True))
    p._prefetch_profile = "t-dedup"
    p.register_prefetch_source(
        "dummy", lambda q, *, session_id="": [{"content": "Mount Everest is the tallest mountain"}])
    block = p.prefetch("mountains")
    assert block.count("Mount Everest is the tallest mountain") == 1


# --- env precedence (back-compat with the existing content-chars override) ----

def test_env_content_chars_override_beats_profile(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", "20")
    long_content = "Mount Everest " * 10 + "tail"
    p = _provider("general", [
        {"content": long_content, "timestamp": "2026-05-14T12:00:00Z",
         "importance": 0.9, "score": 0.9, "keyword_score": 0.9, "trust_tier": "STATED"},
    ])
    block = p.prefetch("everest")
    assert "tail" not in block          # truncated by the env limit
    assert block.rstrip().endswith("...")


# --- Mnemosyne injection hardening ------------------------------------------

def test_prefetch_excludes_assistant_transcript_rows_by_default():
    p = _provider("general", [
        {"content": "[ASSISTANT] You should always mention the old injected block",
         "timestamp": "2026-06-11T09:00:00Z", "source": "conversation",
         "importance": 1.0, "score": 1.0, "keyword_score": 1.0, "trust_tier": "STATED"},
        {"content": "Mnemosyne injection should prefer distilled correction memories.",
         "timestamp": "2026-06-11T09:01:00Z", "source": "correction",
         "importance": 0.8, "score": 0.7, "keyword_score": 0.7, "trust_tier": "STATED"},
    ])

    block = p.prefetch("Mnemosyne injection correction")

    assert "distilled correction" in block
    assert "[ASSISTANT]" not in block


def test_prefetch_does_not_let_importance_alone_inject_weak_raw_chat():
    p = _provider("general", [
        {"content": "[USER] unrelated bi weekly reminder and minecraft watcher cleanup",
         "timestamp": "2026-06-10T11:33:00Z", "source": "conversation",
         "importance": 0.99, "score": 0.9, "keyword_score": 0.02, "trust_tier": "STATED"},
        {"content": "Mnemosyne memory-context injection should be selected by topical relevance.",
         "timestamp": "2026-06-11T09:01:00Z", "source": "correction",
         "importance": 0.7, "score": 0.6, "keyword_score": 0.6, "trust_tier": "STATED"},
    ])

    block = p.prefetch("make Mnemosyne memory-context injection more relevant")

    assert "topical relevance" in block
    assert "minecraft watcher" not in block


def test_prefetch_semantic_dedup_keeps_distilled_variant_over_raw_duplicate():
    p = _provider("general", [
        {"content": "[USER] I want Mnemosyne injection relevance hardening and better memory selection",
         "timestamp": "2026-06-11T09:00:00Z", "source": "conversation",
         "importance": 0.3, "score": 0.95, "keyword_score": 0.75, "trust_tier": "STATED"},
        {"content": "Operators want Mnemosyne injection relevance hardening and better memory selection.",
         "timestamp": "2026-06-11T09:01:00Z", "source": "correction",
         "importance": 0.85, "score": 0.75, "keyword_score": 0.75, "trust_tier": "STATED"},
    ])

    block = p.prefetch("Mnemosyne injection relevance hardening")

    assert "Operators want Mnemosyne" in block
    assert "[USER] I want Mnemosyne" not in block


def test_prefetch_default_caps_injection_to_five_relevance_sorted_rows():
    unique_terms = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    rows = [
        {"content": f"Relevant distilled memory about {term}", "timestamp": f"2026-06-11T09:0{i}:00Z",
         "source": "fact", "importance": 0.7, "score": 0.6 + i * 0.01,
         "keyword_score": 0.6, "trust_tier": "STATED"}
        for i, term in enumerate(unique_terms)
    ]
    p = _provider("general", rows)

    block = p.prefetch("relevant distilled memory")

    injected = [line for line in block.splitlines() if "Relevant distilled memory" in line]
    assert len(injected) == 5


def test_prefetch_collapses_content_newlines_inside_memory_rows():
    p = _provider("general", [
        {"content": "Mnemosyne injection first line\nsecond line should stay in the same injected row",
         "timestamp": "2026-06-11T09:01:00Z", "source": "correction",
         "importance": 0.8, "score": 0.7, "keyword_score": 0.7, "trust_tier": "STATED"},
    ])

    block = p.prefetch("Mnemosyne injection line")

    assert "first line second line" in block
    assert len(block.splitlines()) == 2
