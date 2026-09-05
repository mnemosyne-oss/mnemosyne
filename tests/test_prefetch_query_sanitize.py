"""Tests for prefetch query sanitization (speaker-stamp stripping).

Multi-user gateways may stamp the speaker's display name onto message
content before it reaches the memory layer ("[Alice] what time ...").
The stamp is envelope metadata: kept in the recall query, it makes every
row mentioning the same speaker score as lexically relevant regardless of
subject. prefetch() strips leading stamps from the QUERY only; captured
rows and consolidated summaries keep stamped text so attribution
survives distillation.

The grammar is NAME-shaped, not bracket-shaped: only a short human-name
token (Unicode letters/spaces/apostrophes/hyphens/periods, 1-3 words) is
treated as a stamp. Bracketed topical content -- song titles ([Untitled]),
tags ([TODO]), timestamps ([2026-05-14 12:00]), markdown links
([API docs](url)) -- must reach recall() untouched.

End-to-end coverage is parametrized over BOTH plugin copies
(hermes_memory_provider and integrations.hermes.src.mnemosyne_hermes):
a regression that drops the sanitize call from either copy must fail.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_copy(module_path: str):
    """Import a plugin copy by filesystem path (copy 2 is not a package
    on sys.path in the repo checkout)."""
    if module_path == "hermes_memory_provider":
        return importlib.import_module("hermes_memory_provider")
    src = _REPO_ROOT / "integrations" / "hermes" / "src" / "mnemosyne_hermes" / "__init__.py"
    name = "_mnemosyne_hermes_copy2_for_sanitize_test"
    spec = importlib.util.spec_from_file_location(name, src)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_PLUGIN_COPIES = [
    "hermes_memory_provider",
    "integrations.hermes.src.mnemosyne_hermes",
]

import importlib.util  # noqa: E402


@pytest.mark.parametrize("module_path", _PLUGIN_COPIES)
def test_copy_sanitize_helper_matches_spec(module_path):
    mod = _load_copy(module_path)
    sanitize = mod._sanitize_prefetch_query
    assert sanitize("[Alice] what time does the bakery open") == "what time does the bakery open"
    assert sanitize("[TODO] fix the login bug") == "[TODO] fix the login bug"
    assert sanitize("no stamp here") == "no stamp here"


# Import copy 1 at module level: the plugin-discovery pollution bug this
# file's meta-tests guard against (sys.modules['logging'] clobbered by a
# plugin named logging.py) would otherwise break the lazy import inside
# a test run after that pollution. Importing early pins a healthy module.
from hermes_memory_provider import _sanitize_prefetch_query as _copy1_sanitize


def _sanitize():
    return _copy1_sanitize


@pytest.mark.parametrize("stamped,clean", [
    # name-shaped stamps: stripped
    ("[Alice] what time does the bakery open", "what time does the bakery open"),
    ("[Alice] [Bob] who forgot the tickets", "who forgot the tickets"),
    ("[Alice]\tmuseum plans for Sunday", "museum plans for Sunday"),
    ("[Jean-Luc] library plans", "library plans"),
    ("[Mary Ann] doctor appointment", "doctor appointment"),
    ("[O'Brien] cafe visit", "cafe visit"),
    ("[Anne] cafe plans", "cafe plans"),
    # multi-word surnames up to 4 words + particles (round-2: cap 4)
    ("[Jose Maria Aznar Lopez] what did he say", "what did he say"),
    ("[Maria de la Cruz] says hello", "says hello"),
    # caseless-script multiword names (round-2: caseless passes the gate)
    ("[王 明] 你好", "你好"),
    ("[محمد أحمد] where are we meeting", "where are we meeting"),
    # long compound name within the 48-char inner window (round-2)
    ("[Mary-Jane Anne van Dijk-Visser] lunch", "lunch"),
    # NOT name-shaped: kept (the auditor's false-positive matrix)
    ("[Untitled] live version lyrics meaning", "[Untitled] live version lyrics meaning"),
    ("[TODO] fix the login bug", "[TODO] fix the login bug"),
    ("[Note] buy milk", "[Note] buy milk"),
    # blocklist survives inflection and multiword (round-2: per-word + fold)
    ("[TODOs] remaining work items", "[TODOs] remaining work items"),
    ("[Noted] meeting follow-ups", "[Noted] meeting follow-ups"),
    ("[Annotation] margin comment", "[Annotation] margin comment"),
    ("[TODO Now] fix login", "[TODO Now] fix login"),
    ("[API Docs] how to authenticate", "[API Docs] how to authenticate"),
    ("[Release Notes] v2 launch details", "[Release Notes] v2 launch details"),
    ("[API docs](http://x) how do I auth", "[API docs](http://x) how do I auth"),
    ("[2026-05-14 12:00] meeting notes", "[2026-05-14 12:00] meeting notes"),
    ("[untitled_track] lyrics", "[untitled_track] lyrics"),
    # possessive: no whitespace boundary after ']', so kept
    ("[Alice]'s bakery order", "[Alice]'s bakery order"),
    # punctuation residue lstrip (round-4)
    ("[Alice] , what did we decide", "what did we decide"),
    ("[Alice] - status update", "status update"),
    # exotic-casefold forms now hit the blocklist via NFKC → KEPT
    # (round-4: without NFKC they bypassed the blocklist and were stripped)
    ("[\uff34\uff2f\uff24\uff2f] fix login", "[TODO] fix login"),
    # digit display names still kept (regex-side rejection, documented)
    ("[Neo1999] what did they say", "[Neo1999] what did they say"),
    # passthrough / degenerate cases
    ("no stamp here", "no stamp here"),
    ("", ""),
    ("[] empty stamp is kept", "[] empty stamp is kept"),
    ("[" + "x" * 61 + "] overlong stamp kept", "[" + "x" * 61 + "] overlong stamp kept"),
])
def test_stamp_stripping(stamped, clean):
    assert _sanitize()(stamped) == clean


def test_all_stamps_stripped_falls_back_to_original():
    # A message that is ONLY stamps carries no topical query; recall must
    # NOT run on the stamp tokens (speaker-name recall — the exact bug).
    # Contract (round-4): return "" so prefetch skips the recall call;
    # falling back to the stamped original re-introduced the bug one layer
    # downstream (r4 audit F1).
    assert _sanitize()("[Alice] [Bob]") == ""


class FakeBeam:
    """Serves rows for both plugin copies' filter chains.

    Copy 2's lexical-evidence gate needs the row to share enough query
    tokens for coverage, so the content deliberately echoes the query's
    topical words."""

    author_id = "test-author"

    def __init__(self, seen):
        self._seen = seen

    def recall(self, query, top_k, temporal_weight, temporal_halflife, author_id):
        self._seen.append(query)
        return [
            {"content": "The user usually orders spicy ramen; ramen is the "
                        "user favorite order at the Ginza corner shop",
             "timestamp": "2026-05-14T12:00:00Z", "importance": 0.6,
             "score": 0.9, "keyword_score": 0.9, "trust_tier": "STATED"},
        ]


@pytest.mark.parametrize("module_path", _PLUGIN_COPIES)
def test_prefetch_recalls_with_sanitized_query(module_path, monkeypatch):
    # End-to-end per copy: if either copy's prefetch() drops or bypasses
    # the sanitize call, THIS test fails (the auditor proved the suite
    # was blind to copy 2 when only copy 1 was exercised).
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CONTENT_CHARS", raising=False)
    mod = _load_copy(module_path)

    seen = []
    provider = mod.MnemosyneMemoryProvider()
    provider._beam = FakeBeam(seen)

    block = provider.prefetch("[Alice] what ramen does user order")

    assert seen == ["what ramen does user order"]
    assert "ramen" in block
