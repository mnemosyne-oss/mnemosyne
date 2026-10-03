"""Contract edges for canonical CJK matching, pinned separately from #1022/#1025.

Three behaviours are covered here, all of them currently intentional:

* a one-character CJK query (``人``, ``木``) can still reach the iteration-mark
  slot it belongs to (``人々``, ``木々``),
* recall and automatic prefetch agree on how they order several matching slots,
* the middle-dot body ``猫・犬`` has explicit behaviour for the queries ``猫`` and
  ``犬``.

The fixtures are the generic examples from the issue: nothing here mirrors a real
memory, and no Unicode normalisation or ranking rule is generalised by this file.
"""

from __future__ import annotations

import pytest

from mnemosyne_hermes import (
    _canonical_match_tokens,
    _canonical_prefetch_rows,
    _canonical_recall_rows,
)


class FakeCanonicalStore:
    """Minimal stand-in for mnemocore's canonical store."""

    def __init__(self, rows):
        self._rows = rows

    def list(self, owner_id):
        return self._rows


def _row(body: str, name: str) -> dict:
    return {
        "body": body,
        "category": "preference",
        "name": name,
        "valid_from": "2026-01-01T00:00:00Z",
    }


@pytest.fixture(autouse=True)
def _clear_prefetch_configuration(monkeypatch):
    """Keep the documented prefetch defaults in force for every case here."""
    for key in (
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE",
        "MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY",
    ):
        monkeypatch.delenv(key, raising=False)


def _names(rows):
    return [row.get("canonical_name") for row in rows]


# --- one-character CJK compatibility -------------------------------------
#
# The n-gram width is chosen from the *query*, so a single-character query is
# tokenised as that one character and can match a character sitting inside a
# longer body. That is what keeps ``人`` reachable for a ``人々`` slot. It is a
# deliberate part of the contract: expanding the mark deletes the very character
# the user typed.


def test_single_character_query_reaches_its_iteration_mark_slot():
    store = FakeCanonicalStore([_row("人々", "crowd")])

    assert _names(_canonical_recall_rows(store, "default", "人")) == ["crowd"]
    assert _names(_canonical_prefetch_rows(store, "default", "人")) == ["crowd"]


def test_single_character_query_reaches_the_second_iteration_mark_slot():
    store = FakeCanonicalStore([_row("木々", "grove")])

    assert _names(_canonical_recall_rows(store, "default", "木")) == ["grove"]
    assert _names(_canonical_prefetch_rows(store, "default", "木")) == ["grove"]


def test_expanded_form_query_does_not_reach_a_single_character_slot():
    # The converse is intentionally asymmetric: ``人々`` is a two-character
    # query tokenised as its own bigram, which a bare ``人`` body cannot supply.
    store = FakeCanonicalStore([_row("人", "person")])

    assert _names(_canonical_recall_rows(store, "default", "人々")) == []
    assert _names(_canonical_prefetch_rows(store, "default", "人々")) == []


def test_single_character_query_also_matches_a_raw_duplicate_body():
    # ``人人`` carries no iteration mark, so it does not expand; the single
    # character still matches it through the same query-driven n-gram.
    store = FakeCanonicalStore([_row("人人", "duplicated")])

    assert _names(_canonical_recall_rows(store, "default", "人")) == ["duplicated"]
    assert _names(_canonical_prefetch_rows(store, "default", "人")) == ["duplicated"]


# --- middle dot (U+30FB) --------------------------------------------------


def test_middle_dot_body_is_reachable_from_each_side_term():
    store = FakeCanonicalStore([_row("猫・犬", "pets")])

    assert _names(_canonical_recall_rows(store, "default", "猫")) == ["pets"]
    assert _names(_canonical_prefetch_rows(store, "default", "猫")) == ["pets"]
    assert _names(_canonical_recall_rows(store, "default", "犬")) == ["pets"]
    assert _names(_canonical_prefetch_rows(store, "default", "犬")) == ["pets"]


def test_middle_dot_exact_query_matches_the_body():
    store = FakeCanonicalStore([_row("猫・犬", "pets")])

    assert _names(_canonical_recall_rows(store, "default", "猫・犬")) == ["pets"]
    assert _names(_canonical_prefetch_rows(store, "default", "猫・犬")) == ["pets"]


def test_middle_dot_query_does_not_reach_a_single_term_body():
    # A two-character query is tokenised as bigrams, and ``猫・犬`` yields
    # ``猫・`` / ``・犬`` — never a bare ``猫`` — so a ``猫`` body is not a match.
    store = FakeCanonicalStore([_row("猫", "cat")])

    assert _names(_canonical_recall_rows(store, "default", "猫・犬")) == []
    assert _names(_canonical_prefetch_rows(store, "default", "猫・犬")) == []


def test_middle_dot_stays_inside_its_own_bigrams():
    # Documents the mechanism behind the behaviour above.
    assert _canonical_match_tokens("猫・犬", cjk_ngram_size=2) == {"猫・", "・犬"}
    assert _canonical_match_tokens("猫", cjk_ngram_size=1) == {"猫"}


# --- ordering when several slots match ------------------------------------


def test_both_paths_rank_by_coverage_regardless_of_store_order():
    # Query has three CJK bigrams; the narrower body covers two of them and the
    # wider body all three. Both paths score by coverage, so the wider body
    # leads in either insertion order.
    query = "猫犬鳥魚"
    narrower = _row("猫犬鳥", "narrower")
    wider = _row("猫犬鳥魚", "wider")

    for rows in ([narrower, wider], [wider, narrower]):
        store = FakeCanonicalStore(list(rows))
        recall = _canonical_recall_rows(store, "default", query)
        prefetch = _canonical_prefetch_rows(store, "default", query)

        assert _names(recall) == ["wider", "narrower"]
        assert _names(prefetch) == ["wider", "narrower"]
        assert recall[0]["score"] > recall[1]["score"]
        assert prefetch[0]["score"] > prefetch[1]["score"]
