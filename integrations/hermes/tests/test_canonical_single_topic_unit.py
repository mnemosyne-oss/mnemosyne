"""Regressions for #1025: single topical CJK units must stay recallable.

A short canonical fact such as ``部署`` is answerable from ``什么时候部署？``, but the
surrounding context makes the query's lexical coverage of that fact low, so the
ordinary evidence rules rejected it. The fix is a minimum-evidence rule rather
than a lower global threshold: a single distinctive unit is enough when it *is*
the whole canonical body, which a long unrelated fact can never claim. These
tests use the public helpers (not the store internals) and only synthetic rows.
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


def _row(body, *, category="fact", name=None):
    return {
        "body": body,
        "category": category,
        "name": name or f"slot-{body}",
        "valid_from": "2026-01-01T00:00:00Z",
    }


def _contents(rows):
    return [row["content"] for row in rows]


@pytest.mark.parametrize(
    ("body", "query"),
    [
        ("部署", "什么时候部署？"),
        ("予約", "いつ予約するの？"),
        ("배포", "언제 배포해요?"),
    ],
)
def test_explicit_recall_finds_a_whole_body_unit(body, query):
    store = FakeCanonicalStore([_row(body, category="task")])

    rows = _canonical_recall_rows(store, "default", query)

    assert _contents(rows) == [body]


@pytest.mark.parametrize(
    ("body", "query"),
    [
        ("部署", "什么时候部署？"),
        ("予約", "いつ予約するの？"),
        ("배포", "언제 배포해요?"),
    ],
)
def test_prefetch_finds_a_whole_body_unit(body, query):
    store = FakeCanonicalStore([_row(body, category="task")])

    rows = _canonical_prefetch_rows(store, "default", query)

    assert _contents(rows) == [body]


def test_unrelated_long_slot_is_still_rejected_by_recall():
    store = FakeCanonicalStore([
        _row("サーバーのバックアップ手順を確認すること", category="procedure"),
    ])

    assert _canonical_recall_rows(store, "default", "いつ予約するの？") == []


def test_unrelated_long_slot_is_still_rejected_by_prefetch():
    store = FakeCanonicalStore([
        _row("サーバーのバックアップ手順を確認すること", category="procedure"),
    ])

    assert _canonical_prefetch_rows(store, "default", "いつ予約するの？") == []


def test_body_unit_absent_from_the_query_is_rejected():
    store = FakeCanonicalStore([_row("予約", category="task")])

    assert _canonical_recall_rows(store, "default", "いつ出発するの？") == []
    assert _canonical_prefetch_rows(store, "default", "いつ出発するの？") == []


def test_iteration_mark_siblings_are_still_distinguished():
    # #1013/#1022 contract: the sibling body is a two-token candidate, so the
    # whole-body shortcut must not apply and the explicit recall stays empty.
    store = FakeCanonicalStore([_row("佐々木")])

    assert _canonical_recall_rows(store, "default", "佐々野") == []


def test_owner_isolation_is_preserved():
    store = FakeCanonicalStore({
        "default": [_row("部署", category="task")],
        "other-owner": [],
    })

    assert _canonical_recall_rows(store, "other-owner", "什么时候部署？") == []
    assert store.requested_owner_ids == ["other-owner"]


def test_prefetch_keeps_the_rarity_guard(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY", "1")
    store = FakeCanonicalStore([
        _row("部署", category="task"),
        _row("部署", category="fact"),
    ])

    assert _canonical_prefetch_rows(store, "default", "什么时候部署？") == []


def test_non_cjk_single_token_body_is_not_a_whole_body_unit():
    # ``cjk_ngram_size == 2`` is also true for ordinary non-CJK queries, so a bare
    # ``deploy`` body must not be able to borrow the CJK bypass. The query is long
    # enough that its coverage stays under the bar, which is the only reason the
    # bypass would have mattered.
    query = "how do I deploy this service to production tomorrow?"

    store = FakeCanonicalStore([_row("deploy", category="procedure")])
    assert _canonical_recall_rows(store, "default", query) == []
    assert _canonical_prefetch_rows(store, "default", query) == []


def test_repeated_cjk_units_are_not_a_whole_body_unit():
    # Tokenization returns a set, so ``部署 部署`` collapses to one entry and would
    # claim the bypass without this raw-body check.
    query = "什么时候部署？"

    store = FakeCanonicalStore([_row("部署 部署", category="task")])
    assert _canonical_recall_rows(store, "default", query) == []
    assert _canonical_prefetch_rows(store, "default", query) == []


def test_separators_around_one_unit_still_qualify():
    query = "什么时候部署？"

    store = FakeCanonicalStore([_row("部署。", category="task")])
    assert _contents(_canonical_recall_rows(store, "default", query)) == ["部署。"]
    assert _contents(_canonical_prefetch_rows(store, "default", query)) == ["部署。"]


def test_fullwidth_punctuation_around_one_unit_still_qualifies():
    # The tokenizer drops fullwidth/halfwidth punctuation, so the predicate has to
    # agree: `部署／` tokenizes to one unit, and without the exemption it would be
    # unreachable even though nothing else is in the body.
    query = "什么时候部署？"

    for body in ("部署／", "部署｡"):
        store = FakeCanonicalStore([_row(body, category="task")])
        assert _contents(_canonical_recall_rows(store, "default", query)) == [body], body
        assert _contents(_canonical_prefetch_rows(store, "default", query)) == [body], body

    # Punctuation around two units is still two units.
    two_units = FakeCanonicalStore([_row("部署／計画", category="task")])
    assert _canonical_recall_rows(two_units, "default", query) == []
    assert _canonical_prefetch_rows(two_units, "default", query) == []


def test_whole_body_unit_helper_contract():
    # Imported here (not at module scope) so the behaviour tests above fail on
    # assertions rather than on a collection error when the rule is missing.
    import mnemosyne_hermes as hermes

    whole_body_unit = hermes._canonical_whole_body_unit
    query = {"什么", "么时", "时候", "候部", "部署"}
    assert whole_body_unit("部署", query, cjk_ngram_size=2) is True
    assert whole_body_unit("部署。", query, cjk_ngram_size=2) is True
    assert whole_body_unit("部署／", query, cjk_ngram_size=2) is True
    # Only separators around one CJK bigram qualify.
    assert whole_body_unit("部署 部署", query, cjk_ngram_size=2) is False
    assert whole_body_unit("部署計画", query, cjk_ngram_size=2) is False
    assert whole_body_unit("deploy", query, cjk_ngram_size=2) is False
    assert whole_body_unit("部", query, cjk_ngram_size=2) is False
    # The one-character compatibility path has its own contract.
    assert whole_body_unit("部署", query, cjk_ngram_size=1) is False
    assert whole_body_unit("部署", {"什么"}, cjk_ngram_size=2) is False
