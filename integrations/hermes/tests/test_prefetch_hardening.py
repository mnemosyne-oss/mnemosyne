from __future__ import annotations

import json

import pytest
from mnemosyne.core.canonical import CanonicalStore

from mnemosyne_hermes import (
    MnemosyneMemoryProvider,
    _canonical_cjk_ngram_size,
    _canonical_match_tokens,
    _canonical_prefetch_rows,
    _canonical_recall_rows,
    _prefetch_canonical_generic_tokens,
    _prefetch_min_query_coverage,
    _prefetch_tokens,
)


class FakeBeam:
    author_id = "test-author"

    def __init__(self, results=None):
        self.results = results or []
        self.writes = []

    def recall(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs
        return self.results

    def remember(self, **kwargs):
        self.writes.append(kwargs)


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


def _provider(results):
    p = MnemosyneMemoryProvider()
    p._beam = FakeBeam(results)
    p._agent_context = "primary"
    return p


def test_prefetch_excludes_assistant_transcript_rows():
    p = _provider([
        {"content": "[ASSISTANT] stale answer that should not inject", "source": "conversation",
         "timestamp": "2026-06-11T09:00:00Z", "importance": 1.0, "score": 1.0,
         "keyword_score": 1.0, "trust_tier": "STATED"},
        {"content": "Mnemosyne injection should prefer distilled correction memories.",
         "source": "correction", "timestamp": "2026-06-11T09:01:00Z",
         "importance": 0.8, "score": 0.7, "keyword_score": 0.7, "trust_tier": "STATED"},
    ])

    block = p.prefetch("Mnemosyne injection correction")

    assert "distilled correction" in block
    assert "[ASSISTANT]" not in block


def test_prefetch_requires_topic_signal_not_importance_only():
    p = _provider([
        {"content": "[USER] unrelated minecraft watcher cleanup", "source": "conversation",
         "timestamp": "2026-06-10T11:33:00Z", "importance": 0.99, "score": 0.9,
         "keyword_score": 0.02, "trust_tier": "STATED"},
        {"content": "Mnemosyne memory-context injection should be selected by topical relevance.",
         "source": "correction", "timestamp": "2026-06-11T09:01:00Z",
         "importance": 0.7, "score": 0.6, "keyword_score": 0.6, "trust_tier": "STATED"},
    ])

    block = p.prefetch("make Mnemosyne memory-context injection more relevant")

    assert "topical relevance" in block
    assert "minecraft watcher" not in block


def test_prefetch_requires_two_distinctive_lexical_terms(monkeypatch):
    monkeypatch.setenv(
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "sampleowner,assistant",
    )
    p = _provider([
        {"content": "A lightweight workflow unrelated to tea.", "source": "workflow",
         "timestamp": "2026-06-11T09:00:00Z", "importance": 0.95, "score": 0.9,
         "keyword_score": 0.5, "trust_tier": "STATED"},
        {"content": "SampleOwner prefers Cedar Bakery over Harbor Bakery.", "source": "preference",
         "timestamp": "2026-06-11T09:01:00Z", "importance": 0.9, "score": 0.8,
         "keyword_score": 0.8, "trust_tier": "STATED"},
    ])

    tea = p.prefetch("tea preference jasmine oolong floral citrus clear infusion")
    bakery = p.prefetch("Cedar Bakery preference")

    assert tea == ""
    assert "Cedar Bakery" in bakery


def test_prefetch_supports_cjk_lexical_evidence():
    p = _provider([
        {
            "content": "東京では静かな喫茶店を好む。",
            "source": "preference",
            "timestamp": "2026-08-08T00:00:00Z",
            "importance": 0.9,
            "score": 0.8,
            "keyword_score": 0.8,
            "trust_tier": "STATED",
        },
    ])

    block = p.prefetch("東京で静かな喫茶店を探している")

    assert "静かな喫茶店" in block


def test_prefetch_supports_cyrillic_lexical_evidence():
    p = _provider([
        {
            "content": "Пользователь предпочитает тёмную резервную копию.",
            "source": "preference",
            "timestamp": "2026-08-08T00:00:00Z",
            "importance": 0.9,
            "score": 0.8,
            "keyword_score": 0.8,
            "trust_tier": "STATED",
        },
    ])

    block = p.prefetch("Найди тёмную резервную копию")

    assert "тёмную резервную копию" in block


def test_canonical_generic_override_does_not_change_normal_prefetch(monkeypatch):
    monkeypatch.setenv(
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "cedar,bakery",
    )
    p = _provider([
        {"content": "SampleOwner prefers Cedar Bakery over Harbor Bakery.", "source": "preference",
         "timestamp": "2026-06-11T09:01:00Z", "importance": 0.9, "score": 0.8,
         "keyword_score": 0.8, "trust_tier": "STATED"},
    ])

    block = p.prefetch("Cedar Bakery preference")

    assert "Cedar Bakery" in block


def test_prefetch_generic_token_configuration_replaces_defaults(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", "sampleowner,assistant")

    tokens = _prefetch_canonical_generic_tokens()

    assert tokens == {"sampleowner", "assistant"}


def test_prefetch_extra_generic_token_configuration_is_additive(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", "sampleowner,assistant")
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_EXTRA_GENERIC_TOKENS", "local,ownername")

    tokens = _prefetch_canonical_generic_tokens()

    assert tokens == {"sampleowner", "assistant", "local", "ownername"}


def test_prefetch_lexical_thresholds_are_configurable(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS", "1")
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE", "0.10")
    p = _provider([
        {"content": "A lightweight workflow unrelated to tea.", "source": "workflow",
         "timestamp": "2026-06-11T09:00:00Z", "importance": 0.95, "score": 0.9,
         "keyword_score": 0.5, "trust_tier": "STATED"},
    ])

    block = p.prefetch("tea floral jasmine clear infusion")

    assert "lightweight workflow" in block


def test_prefetch_nonfinite_query_coverage_uses_conservative_default(monkeypatch, caplog):
    for value in ("NaN", "inf", "-inf"):
        monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE", value)

        assert _prefetch_min_query_coverage() == 0.30

    assert caplog.text.count("MNEMOSYNE_PREFETCH_MIN_QUERY_COVERAGE") == 3


def test_sync_roles_can_disable_assistant_autosave():
    p = MnemosyneMemoryProvider()
    p._beam = FakeBeam()
    p._agent_context = "primary"
    p._sync_roles = {"user"}

    p.sync_turn("please remember user side", "assistant side should not be stored")

    written = [w["content"] for w in p._beam.writes]
    assert any(c.startswith("[USER]") for c in written)
    assert not any(c.startswith("[ASSISTANT]") for c in written)


def test_canonical_prefetch_rejects_common_single_token_matches(monkeypatch):
    monkeypatch.setenv(
        "MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS",
        "user,owner,assistant,agent,system,profile,identity,default,sampleowner,prefers,preference",
    )
    store = FakeCanonicalStore([
        {"name": "tea", "body": "SampleOwner likes fragrant jasmine tea.", "category": "model:user"},
        {"name": "unrelated-a", "body": "SampleOwner studies orbital mechanics.", "category": "model:user"},
        {"name": "unrelated-b", "body": "SampleOwner collects antique maps.", "category": "model:user"},
    ])

    rows = _canonical_prefetch_rows(store, "default", "SampleOwner tea preference")

    assert [row["canonical_name"] for row in rows] == ["tea"]


def test_canonical_prefetch_allows_rare_single_token_and_two_token_matches(monkeypatch):
    monkeypatch.delenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", raising=False)
    store = FakeCanonicalStore([
        {"name": "archive", "body": "Archive boundaries apply to expired records.", "category": "model:user"},
        {"name": "archive-noise", "body": "External indexes require archiving.", "category": "model:workflow"},
        {"name": "geology", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
        {"name": "other", "body": "Copper alloys resist corrosion.", "category": "model:user"},
    ])

    archive = _canonical_prefetch_rows(store, "default", "archive boundaries")
    caldrin = _canonical_prefetch_rows(store, "default", "Caldrin research lens")

    assert [row["canonical_name"] for row in archive] == ["archive"]
    assert [row["canonical_name"] for row in caldrin] == ["geology"]


def test_canonical_prefetch_rarity_is_scoped_to_requested_owner(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY", "1")
    store = FakeCanonicalStore({
        "default": [
            {"name": "geology", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
        ],
        "other-owner": [
            {"name": "geology-a", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
            {"name": "geology-b", "body": "Caldrin studies coastal erosion.", "category": "model:user"},
        ],
    })

    default_rows = _canonical_prefetch_rows(store, "default", "Caldrin research lens")
    other_rows = _canonical_prefetch_rows(store, "other-owner", "Caldrin research lens")

    assert store.requested_owner_ids == ["default", "other-owner"]
    assert [row["canonical_name"] for row in default_rows] == ["geology"]
    assert other_rows == []


def test_explicit_recall_preserves_broad_canonical_merge():
    p = _provider([])
    p._beam.canonical = FakeCanonicalStore([
        {"name": "geology-a", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
        {"name": "geology-b", "body": "Caldrin studies coastal erosion.", "category": "model:user"},
    ])

    response = json.loads(p.handle_tool_call(
        "mnemosyne_recall",
        {"query": "Caldrin research lens", "limit": 5},
    ))

    assert p._beam.last_args == ("Caldrin research lens",)
    assert {row["canonical_name"] for row in response["results"]} == {"geology-a", "geology-b"}


def test_explicit_recall_does_not_apply_prefetch_specific_generic_tokens():
    p = _provider([])
    store = FakeCanonicalStore([
        {
            "name": "selection-lens",
            "body": "SampleOwner recommendations emphasize reliability.",
            "category": "model:user",
        },
    ])
    p._beam.canonical = store

    assert _canonical_prefetch_rows(store, "default", "recommendations") == []

    response = json.loads(p.handle_tool_call(
        "mnemosyne_recall",
        {"query": "recommendations", "limit": 5},
    ))

    assert [row["canonical_name"] for row in response["results"]] == ["selection-lens"]


def test_explicit_recall_honors_configured_canonical_generic_tokens(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", "sampleowner")
    p = _provider([])
    p._beam.canonical = FakeCanonicalStore([
        {
            "name": "budget-lens",
            "body": "SampleOwner budget approach.",
            "category": "model:user",
        },
    ])

    response = json.loads(p.handle_tool_call(
        "mnemosyne_recall",
        {"query": "SampleOwner", "limit": 5},
    ))

    assert response["results"] == []


def test_explicit_recall_override_can_include_prefetch_specific_generic_token(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_GENERIC_TOKENS", "recommendations")
    p = _provider([])
    p._beam.canonical = FakeCanonicalStore([
        {
            "name": "selection-lens",
            "body": "SampleOwner recommendations emphasize reliability.",
            "category": "model:user",
        },
    ])

    response = json.loads(p.handle_tool_call(
        "mnemosyne_recall",
        {"query": "recommendations", "limit": 5},
    ))

    assert response["results"] == []


def test_canonical_prefetch_can_disable_single_token_exception(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CANONICAL_RARE_TOKEN_MAX_FREQUENCY", "0")
    store = FakeCanonicalStore([
        {"name": "geology", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
    ])

    rows = _canonical_prefetch_rows(store, "default", "Caldrin research lens")

    assert rows == []


def test_canonical_prefetch_keeps_singleton_coverage_guard_when_minimum_is_one(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS", "1")
    store = FakeCanonicalStore([
        {"name": "geology", "body": "Caldrin catalogs basalt formations.", "category": "model:user"},
    ])

    rows = _canonical_prefetch_rows(
        store,
        "default",
        "Caldrin research archive survey analysis history",
    )

    assert rows == []


def test_canonical_prefetch_respects_higher_distinctive_token_minimum(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_PREFETCH_MIN_DISTINCTIVE_TOKENS", "3")
    store = FakeCanonicalStore([
        {"name": "tea", "body": "Fragrant jasmine tea is preferred.", "category": "model:user"},
    ])

    rows = _canonical_prefetch_rows(store, "default", "fragrant jasmine soup")

    assert rows == []


@pytest.mark.parametrize(
    ("script", "query", "ordinary", "unrelated"),
    [
        (
            "japanese",
            "ArchiveBoxの運用方針はどうなってる？",
            "ArchiveBoxの運用方針は週次バックアップです。",
            "家族の健康方針と旅行予定を優先する。",
        ),
        (
            "korean",
            "서울 여행 숙소는 어디야?",
            "서울 여행 숙소는 한강 근처다.",
            "업무 회의 장소는 조용한 방이다.",
        ),
        (
            "chinese",
            "上海旅行计划是什么？",
            "上海旅行计划包括外滩和博物馆。",
            "用户喜欢海边和计划性工作。",
        ),
    ],
)
def test_cjk_canonical_matching_rejects_noise_without_displacing_public_results(
    script, query, ordinary, unrelated
):
    unrelated_store = FakeCanonicalStore([
        {"name": f"unrelated-{script}", "body": unrelated, "category": "model:user"},
    ])
    assert _canonical_recall_rows(unrelated_store, "default", query, limit=5) == []
    assert _canonical_prefetch_rows(unrelated_store, "default", query, limit=5) == []

    provider = _provider([{
        "content": ordinary,
        "source": "fact",
        "timestamp": "2026-09-17T00:00:00Z",
        "importance": 0.7,
        "score": 0.60,
        "keyword_score": 0.60,
        "trust_tier": "STATED",
    }])
    assert provider._beam is not None
    provider._beam.canonical = unrelated_store
    block = provider.prefetch(query)
    response = json.loads(provider.handle_tool_call(
        "mnemosyne_recall", {"query": query, "limit": 1},
    ))
    assert ordinary in block
    assert unrelated not in block
    assert [row["content"] for row in response["results"]] == [ordinary]

    positive_provider = _provider([])
    assert positive_provider._beam is not None
    positive_provider._beam.canonical = FakeCanonicalStore([{
        "name": f"matching-{script}", "body": ordinary, "category": "model:user",
    }])
    assert ordinary in positive_provider.prefetch(query)
    positive_response = json.loads(positive_provider.handle_tool_call(
        "mnemosyne_recall", {"query": query, "limit": 1},
    ))
    assert [row["content"] for row in positive_response["results"]] == [ordinary]


@pytest.mark.parametrize(
    ("query", "body", "expected_token"),
    [
        ("東京", "東京の喫茶店を好む。", "東京"),
        ("서울", "서울의 박물관을 좋아한다.", "서울"),
        ("上海", "上海的博物馆很安静。", "上海"),
        ("茶", "用户喜欢茶。", "茶"),
        ("茶？", "用户喜欢茶。", "茶"),
    ],
)
def test_canonical_matching_preserves_short_cjk_terms(query, body, expected_token):
    ngram_size = _canonical_cjk_ngram_size(query)
    store = FakeCanonicalStore([
        {"name": "short-term", "body": body, "category": "model:user"},
    ])
    assert expected_token in _canonical_match_tokens(query, cjk_ngram_size=ngram_size)
    assert [row["canonical_name"] for row in _canonical_recall_rows(store, "default", query)] == [
        "short-term"
    ]
    assert [row["canonical_name"] for row in _canonical_prefetch_rows(store, "default", query)] == [
        "short-term"
    ]


def test_canonical_matching_keeps_japanese_iteration_mark_in_cjk_run():
    store = FakeCanonicalStore([
        {"name": "japanese-name", "body": "佐々木", "category": "model:user"},
    ])

    assert _canonical_match_tokens("佐々木") == {"佐々", "々木"}
    assert [row["canonical_name"] for row in _canonical_recall_rows(store, "default", "佐々木")] == [
        "japanese-name"
    ]
    assert [
        row["canonical_name"] for row in _canonical_prefetch_rows(store, "default", "佐々木")
    ] == ["japanese-name"]


def test_iteration_mark_canonical_slots_recall_only_the_exact_name(tmp_path):
    store = CanonicalStore(db_path=tmp_path / "canonical.db")
    store.remember("default", "model:user", "sasaki", "佐々木")
    store.remember("default", "model:user", "sasano", "佐々野")
    provider = _provider([])
    provider._beam.canonical = store

    sasaki = json.loads(provider.handle_tool_call(
        "mnemosyne_recall", {"query": "佐々木", "limit": 5},
    ))
    sasano = json.loads(provider.handle_tool_call(
        "mnemosyne_recall", {"query": "佐々野", "limit": 5},
    ))

    assert [row["canonical_name"] for row in sasaki["results"]] == ["sasaki"]
    assert [row["canonical_name"] for row in sasano["results"]] == ["sasano"]
    assert [
        row["canonical_name"] for row in _canonical_prefetch_rows(store, "default", "佐々木")
    ] == ["sasaki"]
    assert [
        row["canonical_name"] for row in _canonical_prefetch_rows(store, "default", "佐々野")
    ] == ["sasano"]


@pytest.mark.parametrize(
    ("query", "expected_name"),
    [
        ("佐々あ", "saa"),
        ("佐佐あ", "saa"),
        ("佐々い", "sai"),
        ("佐佐い", "sai"),
    ],
)
def test_iteration_mark_canonical_slots_keep_kana_suffix_discriminator(
    tmp_path, query, expected_name
):
    store = CanonicalStore(db_path=tmp_path / "canonical.db")
    store.remember("default", "model:user", "saa", "佐々あ")
    store.remember("default", "model:user", "sai", "佐々い")
    provider = _provider([])
    provider._beam.canonical = store

    response = json.loads(provider.handle_tool_call(
        "mnemosyne_recall", {"query": query, "limit": 5},
    ))

    assert [row["canonical_name"] for row in response["results"]] == [expected_name]


@pytest.mark.parametrize(
    ("query", "body", "sibling_body"),
    [
        ("佐々木", "佐佐木", "佐佐野"),
        ("佐佐木", "佐々木", "佐々野"),
        ("佐々々木", "佐佐佐木", "佐佐佐野"),
        ("人々", "人人", "人海"),
        ("ID-佐々木-A", "ID-佐佐木-A", "ID-佐佐野-A"),
    ],
)
def test_explicit_recall_treats_iteration_marks_as_literal_han_repetition(
    query, body, sibling_body
):
    store = FakeCanonicalStore([
        {"name": "matching", "body": body, "category": "model:user"},
        {"name": "sibling", "body": sibling_body, "category": "model:user"},
    ])

    assert [
        row["canonical_name"] for row in _canonical_recall_rows(store, "default", query, limit=5)
    ] == ["matching"]


@pytest.mark.parametrize(
    ("query", "body"),
    [
        ("々木", "佐々木"),
        ("佐 々木", "佐々木"),
        ("佐、々木", "佐々木"),
        ("佐々木", "佐 々木"),
        ("の々木", "のの木"),
    ],
)
def test_explicit_recall_does_not_expand_iteration_marks_across_boundaries(query, body):
    store = FakeCanonicalStore([
        {"name": "non-matching", "body": body, "category": "model:user"},
    ])

    assert _canonical_recall_rows(store, "default", query, limit=5) == []


def test_start_of_run_iteration_marks_keep_literal_compatibility():
    store = FakeCanonicalStore([
        {"name": "matching", "body": "々木", "category": "model:user"},
        {"name": "sibling", "body": "々野", "category": "model:user"},
    ])

    assert [
        row["canonical_name"] for row in _canonical_recall_rows(store, "default", "々木", limit=5)
    ] == ["matching"]


def test_start_of_run_literal_match_survives_unrelated_iteration_run():
    store = FakeCanonicalStore([
        {"name": "matching", "body": "々木 佐々野", "category": "model:user"},
    ])

    assert [
        row["canonical_name"] for row in _canonical_recall_rows(store, "default", "々木", limit=5)
    ] == ["matching"]


@pytest.mark.parametrize(
    ("query", "body"),
    [
        ("Caldrin basalt", "Caldrin catalogs basalt formations."),
        ("тёмную копию", "Пользователь предпочитает тёмную резервную копию."),
    ],
)
def test_canonical_matching_preserves_latin_and_cyrillic_terms(query, body):
    store = FakeCanonicalStore([
        {"name": "matching", "body": body, "category": "model:user"},
    ])
    assert [row["canonical_name"] for row in _canonical_recall_rows(store, "default", query)] == [
        "matching"
    ]
    assert [row["canonical_name"] for row in _canonical_prefetch_rows(store, "default", query)] == [
        "matching"
    ]


def test_canonical_recall_owner_isolation_survives_cjk_matching():
    store = FakeCanonicalStore({
        "default": [
            {"name": "tokyo", "body": "東京の喫茶店を好む。", "category": "model:user"},
        ],
        "other-owner": [
            {"name": "seoul", "body": "서울의 박물관을 좋아한다.", "category": "model:user"},
        ],
    })
    assert [
        row["canonical_name"] for row in _canonical_recall_rows(store, "default", "東京")
    ] == ["tokyo"]
    assert _canonical_recall_rows(store, "other-owner", "東京") == []


@pytest.mark.parametrize(
    "content",
    [
        "Caldrin catalogs basalt formations.",
        "Пользователь предпочитает тёмную резервную копию.",
        "https://example.test/archive_path",
    ],
)
def test_canonical_matching_preserves_non_cjk_tokenization(content):
    assert _canonical_match_tokens(content) == _prefetch_tokens(content)


def test_prefixed_single_character_cjk_query_uses_exact_compatibility_path():
    store = FakeCanonicalStore([
        {"name": "tea", "body": "用户喜欢茶。", "category": "model:user"},
    ])
    assert [
        row["canonical_name"] for row in _canonical_recall_rows(store, "default", "[USER] 茶")
    ] == ["tea"]


@pytest.mark.parametrize(
    ("query", "body"),
    [
        ("納豆は？", "納豆が苦手です。"),
        ("숙소가?", "숙소는 한강 근처다."),
        ("茶叶呢？", "用户喜欢茶叶。"),
    ],
)
def test_canonical_matching_keeps_one_strong_bigram_for_short_queries(query, body):
    store = FakeCanonicalStore([
        {"name": "matching", "body": body, "category": "model:user"},
    ])
    assert [row["canonical_name"] for row in _canonical_recall_rows(store, "default", query)] == [
        "matching"
    ]
    assert [row["canonical_name"] for row in _canonical_prefetch_rows(store, "default", query)] == [
        "matching"
    ]
