"""Regressions for #1023: prefetch must apply the iteration-mark predicate too.

With a store holding `佐々木`, a `佐々野` query correctly returned no explicit-recall
result but automatic prefetch could still inject the sibling `佐々木` — the worse of
the two paths, because prefetch content is written into the prompt unasked. Both
paths now run the same narrow, run-local U+3005 predicate.

Recall and prefetch are asserted separately, as the issue asks, and the cases that
must not change (exact match, expanded kana suffix, Hangul, Latin, run boundary)
are pinned next to them. Only synthetic rows are used.
"""

from __future__ import annotations

import pytest

from mnemosyne_hermes import (
    _canonical_prefetch_rows,
    _canonical_recall_rows,
)


class FakeCanonicalStore:
    def __init__(self, rows_or_owners):
        if isinstance(rows_or_owners, dict):
            self.rows_by_owner = rows_or_owners
        else:
            self.rows_by_owner = {"default": rows_or_owners}
        self.requested_owner_ids = []

    def list(self, owner_id):
        self.requested_owner_ids.append(owner_id)
        return self.rows_by_owner.get(owner_id, [])


@pytest.fixture(autouse=True)
def clear_prefetch_configuration(monkeypatch):
    for key in (
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS",
        "MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE",
        "MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY",
    ):
        monkeypatch.delenv(key, raising=False)


def _row(body, *, category="model:user"):
    return {
        "body": body,
        "category": category,
        "name": "slot-" + body,
        "valid_from": "2026-01-01T00:00:00Z",
    }


def _names(rows):
    return [row["canonical_name"] for row in rows]


def test_prefetch_no_longer_injects_the_iteration_mark_sibling():
    # The #1023 report: recall rejects this, prefetch used to inject it anyway.
    store = FakeCanonicalStore([_row("佐々木")])

    assert _canonical_recall_rows(store, "default", "佐々野") == []
    assert _canonical_prefetch_rows(store, "default", "佐々野") == []


def test_exact_iteration_mark_match_is_preserved_on_both_paths():
    store = FakeCanonicalStore([_row("佐々木")])

    assert _names(_canonical_recall_rows(store, "default", "佐々木")) == ["slot-佐々木"]
    assert _names(_canonical_prefetch_rows(store, "default", "佐々木")) == ["slot-佐々木"]


def test_kana_suffix_sibling_is_rejected_on_both_paths():
    # `佐々あ` and `佐々い` share the 々 anchor; the following kana distinguishes them.
    store = FakeCanonicalStore([_row("佐々あ")])

    assert _canonical_recall_rows(store, "default", "佐々い") == []
    assert _canonical_prefetch_rows(store, "default", "佐々い") == []
    assert _names(_canonical_recall_rows(store, "default", "佐々あ")) == ["slot-佐々あ"]
    assert _names(_canonical_prefetch_rows(store, "default", "佐々あ")) == ["slot-佐々あ"]


def test_expanded_iteration_query_stays_recall_only():
    # #1022 contract, unchanged here: iteration-normalized evidence is added for
    # explicit recall only, while prefetch keeps its raw bigrams. Applying the
    # predicate to prefetch must not widen that tokenization.
    store = FakeCanonicalStore([_row("佐々あ")])

    assert _names(_canonical_recall_rows(store, "default", "佐佐あ")) == ["slot-佐々あ"]
    assert _canonical_prefetch_rows(store, "default", "佐佐あ") == []


def test_run_boundary_mark_agrees_on_both_paths():
    # `の々木` has no Han antecedent, so the mark never expands into a repeat; the
    # exact string is still its own fact on both paths, and the expanded spelling
    # stays a different token set.
    store = FakeCanonicalStore([_row("の々木")])
    assert _canonical_recall_rows(store, "default", "の々木") != []
    assert _names(_canonical_prefetch_rows(store, "default", "の々木")) == ["slot-の々木"]

    expanded = FakeCanonicalStore([_row("のの木")])
    assert _canonical_recall_rows(expanded, "default", "の々木") == []
    assert _canonical_prefetch_rows(expanded, "default", "の々木") == []


def test_run_start_mark_sharing_a_raw_bigram_is_rejected_on_both_paths():
    # `々木` has no antecedent, so the mark never expands — but its raw bigram is
    # also present in `佐々木`, which is where the sibling leak would reappear if
    # only the expansion anchors were checked. Recall already rejected it; the
    # prefetch path must agree.
    store = FakeCanonicalStore([_row("佐々木")])

    assert _canonical_recall_rows(store, "default", "々木") == []
    assert _canonical_prefetch_rows(store, "default", "々木") == []


def test_punctuation_boundary_is_preserved():
    store = FakeCanonicalStore([_row("佐々木。")])

    assert _names(_canonical_prefetch_rows(store, "default", "佐々木")) == ["slot-佐々木。"]
    assert _canonical_prefetch_rows(store, "default", "佐々野") == []


def test_hangul_and_latin_matches_are_unaffected():
    hangul = FakeCanonicalStore([_row("서울 숙소", category="fact")])
    assert _names(_canonical_prefetch_rows(hangul, "default", "서울 숙소 알려줘")) == ["slot-서울 숙소"]

    latin = FakeCanonicalStore([_row("deploy pipeline", category="procedure")])
    assert _names(
        _canonical_prefetch_rows(latin, "default", "deploy pipeline config")
    ) == ["slot-deploy pipeline"]


def test_whole_body_unit_from_1025_still_qualifies():
    # The predicate must not undo the single-unit rule that #1025 added.
    store = FakeCanonicalStore([_row("部署", category="task")])

    assert _names(_canonical_prefetch_rows(store, "default", "什么时候部署？")) == ["slot-部署"]
    assert _names(_canonical_recall_rows(store, "default", "什么时候部署？")) == ["slot-部署"]


def test_owner_isolation_is_preserved():
    store = FakeCanonicalStore({"default": [_row("佐々木")], "other-owner": []})

    assert _canonical_prefetch_rows(store, "other-owner", "佐々木") == []
    assert store.requested_owner_ids == ["other-owner"]
