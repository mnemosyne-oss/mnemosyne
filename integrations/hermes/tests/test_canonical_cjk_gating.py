"""CJK canonical-slot gating regressions for #971.

Spaceless CJK text used to be tokenized one character at a time, so any two
characters shared with a canonical slot satisfied the "two distinctive tokens"
gate. Unrelated single-source-of-truth facts therefore scored 0.94-0.98 and
were injected into the memory context of every turn, crowding out precise
recall.

These tests pin the 2-gram behaviour on the public canonical paths:

* unrelated CJK queries return no canonical rows,
* genuine overlaps are preserved for Japanese, Korean and Chinese,
* a short question whose only topical unit is its last 2-gram still matches
  (the Chinese "什么时候部署？" case that a plain unit-count threshold dropped),
* a one-character CJK run still matches, so very short slots keep working,
* Latin behaviour is untouched.
"""

from __future__ import annotations

import json

import pytest

from mnemosyne_hermes import (
    MnemosyneMemoryProvider,
    _canonical_prefetch_rows,
    _canonical_recall_rows,
    _prefetch_lexical_units,
    _prefetch_tokens,
    _semantic_dedup_prefetch,
)

JAPANESE = {
    "slots": [
        ("identity", "name_pronoun", "利用者は「タナカ」と呼び、一人称は「私」を使う。"),
        ("preference", "tea", "静かな喫茶店と日本茶（緑茶・ほうじ茶）を好む。"),
        ("procedure", "backup", "週次のバックアップ手順を実行し、保存先は外付けディスクとする。"),
        ("environment", "shared_server", "社内の共有サーバーは業務時間内のみ稼働する。"),
        ("fact", "inventory", "棚卸しは月末の金曜日にまとめて数える。"),
        ("procedure", "deploy", "配備は金曜日を避け、火曜日の午前中に行う。"),
    ],
    "unrelated": [
        "会議室の予約ルールはどこにある？",
        "ノートパソコンの在庫は足りてる？",
        "新入社員の研修はいつから？",
        "備品の注文は誰が承認する？",
        "印刷機のトナーはどこに置いてある？",
        "駐車場の契約を更新したい",
    ],
    "related": [
        ("静かな喫茶店を探している", "tea"),
        ("バックアップの保存先は？", "backup"),
        ("共有サーバーの稼働時間は？", "shared_server"),
        ("棚卸しの日程は？", "inventory"),
        ("配備はいつ実施する？", "deploy"),
    ],
}

KOREAN = {
    "slots": [
        ("identity", "name_pronoun", "사용자를 철수라고 부르고 일인칭은 저를 쓴다."),
        ("preference", "tea", "조용한 찻집과 녹차를 좋아한다."),
        ("procedure", "backup", "매주 백업을 수행하고 저장 위치는 외장 디스크로 한다."),
        ("environment", "shared_server", "사내 공용 서버는 업무 시간에만 운영된다."),
        ("fact", "inventory", "재고 조사는 매월 마지막 금요일에 한다."),
        ("procedure", "deploy", "배포는 금요일을 피하고 화요일 오전에 실행한다."),
    ],
    "unrelated": [
        "회의실 예약 규칙은 어디에 있나요?",
        "노트북 재고는 충분한가요?",
        "신입 사원 교육은 언제 시작하나요?",
        "비품 주문은 누가 승인하나요?",
        "사무실 프린터 토너는 어디에 있나요?",
        "주차장 계약을 갱신하고 싶습니다",
    ],
    "related": [
        ("조용한 찻집을 찾고 있어요", "tea"),
        ("백업 저장 위치는?", "backup"),
        ("공용 서버 운영 시간은?", "shared_server"),
        ("재고 조사는 언제 하나요?", "inventory"),
        ("배포는 언제 하나요?", "deploy"),
    ],
}

CHINESE = {
    "slots": [
        ("identity", "name_pronoun", "把用户称为小李，第一人称使用我。"),
        ("preference", "tea", "喜欢安静的茶馆和绿茶。"),
        ("procedure", "backup", "每周执行备份，保存位置是外接硬盘。"),
        ("environment", "shared_server", "内部共享服务器只在工作时间内运行。"),
        ("fact", "inventory", "盘点在每月最后一个周五进行。"),
        ("procedure", "deploy", "部署避开周五，安排在周二上午。"),
    ],
    "unrelated": [
        "会议室的预订规则在哪里？",
        "办公电脑的库存够用吗？",
        "新员工培训什么时候开始？",
        "办公用品由谁审批？",
        "办公室打印机的碳粉放在哪里？",
        "停车场的合同需要续签",
    ],
    "related": [
        ("想找安静的茶馆", "tea"),
        ("备份保存在哪里？", "backup"),
        ("共享服务器什么时候运行？", "shared_server"),
        ("盘点什么时候进行？", "inventory"),
        ("什么时候部署？", "deploy"),
    ],
}

FIXTURES = {"ja": JAPANESE, "ko": KOREAN, "zh": CHINESE}

CANONICAL_PATHS = (
    ("recall", _canonical_recall_rows),
    ("prefetch", _canonical_prefetch_rows),
)
CANONICAL_PATH_IDS = [name for name, _ in CANONICAL_PATHS]


class FakeCanonicalStore:
    """Minimal canonical store: the gates only need ``list(owner_id)``."""

    def __init__(self, rows):
        self._rows = rows

    def list(self, owner_id):  # noqa: ARG002 - owner scoping is core-provided
        return self._rows


def _store(language):
    return FakeCanonicalStore([
        {
            "body": body,
            "category": category,
            "name": name,
            "created_at": "2026-01-01T00:00:00Z",
        }
        for category, name, body in FIXTURES[language]["slots"]
    ])


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
@pytest.mark.parametrize("language", sorted(FIXTURES))
def test_unrelated_cjk_query_returns_no_canonical_rows(language, path_name, path):
    store = _store(language)

    matched = {
        query: [row.get("canonical_name") for row in path(store, "default", query)]
        for query in FIXTURES[language]["unrelated"]
    }

    assert matched == {query: [] for query in FIXTURES[language]["unrelated"]}, (
        f"{language}/{path_name} injected unrelated canonical slots"
    )


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
@pytest.mark.parametrize("language", sorted(FIXTURES))
def test_cjk_true_positives_are_preserved_and_exclusive(language, path_name, path):
    store = _store(language)
    related = dict(FIXTURES[language]["related"])

    results = {
        query: [row.get("canonical_name") for row in path(store, "default", query)]
        for query in related
    }

    assert results == {query: [name] for query, name in related.items()}, (
        f"{language}/{path_name} did not return exactly the matching slot"
    )


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_single_topical_unit_query_still_matches(path_name, path):
    """Chinese questions often carry one meaningful 2-gram plus function words.

    "什么时候部署？" shares only "部署" with the deploy slot. Function-word units
    must not dilute query coverage into a miss.
    """

    store = _store("zh")

    names = [row.get("canonical_name") for row in path(store, "default", "什么时候部署？")]

    assert "deploy" in names, f"{path_name} dropped a one-topical-unit match: {names}"


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_single_character_cjk_runs_still_match(path_name, path):
    """Comma-separated short CJK entries produce one-character runs."""

    store = FakeCanonicalStore([
        {"body": "猫、犬", "category": "fact", "name": "pets", "created_at": "2026-01-01T00:00:00Z"},
    ])

    names = [row.get("canonical_name") for row in path(store, "default", "猫")]

    assert names == ["pets"]


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_single_character_cjk_run_does_not_match_a_different_character(path_name, path):
    store = FakeCanonicalStore([
        {"body": "猫、犬", "category": "fact", "name": "pets", "created_at": "2026-01-01T00:00:00Z"},
    ])

    assert path(store, "default", "猿") == []


def test_cjk_stop_units_are_operator_configurable(monkeypatch):
    """MNEMOSYNE_PREFETCH_CJK_STOP_UNITS replaces the built-in defaults.

    Setting it to a topical unit proves the knob is live: the Chinese
    "什么时候部署？" question then loses the single unit that made it match.
    """

    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CJK_STOP_UNITS", "什么,么时,时候,部署")

    assert _canonical_recall_rows(_store("zh"), "default", "什么时候部署？") == []


class FakeBeam:
    """Beam stub for the public paths: recall results plus a canonical store."""

    author_id = "test-author"

    def __init__(self, canonical=None, results=None):
        self.canonical = canonical
        self.results = results or []
        self.writes = []

    def recall(self, *args, **kwargs):  # noqa: ARG002 - provider passes through
        self.last_kwargs = kwargs
        return self.results

    def remember(self, **kwargs):
        self.writes.append(kwargs)


class OwnerScopedCanonicalStore:
    """Canonical store that honours ``list(owner_id)`` scoping."""

    def __init__(self, rows_by_owner):
        self._rows_by_owner = rows_by_owner
        self.requested_owner_ids = []

    def list(self, owner_id):
        self.requested_owner_ids.append(owner_id)
        return self._rows_by_owner.get(owner_id, [])


def _provider(canonical=None, results=None, identity=None):
    provider = MnemosyneMemoryProvider()
    provider._beam = FakeBeam(canonical, results)
    provider._agent_context = "primary"
    if identity is not None:
        provider._agent_identity = identity
    return provider


def _working_row(content, source="preference"):
    return {
        "content": content,
        "source": source,
        "timestamp": "2026-09-01T00:00:00Z",
        "importance": 0.9,
        "score": 0.8,
        "keyword_score": 0.8,
        "trust_tier": "STATED",
    }


UNRELATED_QUERY = "会議室の予約ルールはどこにある？"
RELATED_QUERY = "静かな喫茶店を探している"
ORDINARY_MEMORY = "会議室の予約ルールは総務が管理していて、変更は申請が必要。"


def test_prefetch_public_path_keeps_ordinary_results_and_drops_unrelated_canonical_rows():
    """The per-turn injection path must not let unrelated slots displace recall.

    This is the user-visible half of #971: automatic prefetch appends canonical
    rows to ordinary results, so character-level noise put unrelated
    single-source-of-truth facts ahead of the memory that actually matched.
    """

    provider = _provider(canonical=_store("ja"), results=[_working_row(ORDINARY_MEMORY)])

    block = provider.prefetch(UNRELATED_QUERY)

    assert "総務" in block, "the ordinary matching memory must still be injected"
    for _, _, body in FIXTURES["ja"]["slots"]:
        assert body not in block, f"unrelated canonical slot injected: {body}"


def test_prefetch_public_path_injects_the_matching_canonical_row():
    provider = _provider(canonical=_store("ja"))

    block = provider.prefetch(RELATED_QUERY)

    assert FIXTURES["ja"]["slots"][1][2] in block


def test_recall_tool_path_keeps_ordinary_results_and_drops_unrelated_canonical_rows():
    """mnemosyne_recall merges canonical rows into normal recall results."""

    provider = _provider(canonical=_store("ja"), results=[_working_row(ORDINARY_MEMORY)])

    payload = json.loads(provider._handle_recall({"query": UNRELATED_QUERY}))

    contents = [row.get("content") for row in payload["results"]]
    assert ORDINARY_MEMORY in contents
    for _, name, body in FIXTURES["ja"]["slots"]:
        assert body not in contents, f"unrelated canonical slot merged: {name}"


def test_recall_tool_path_merges_the_matching_canonical_row():
    provider = _provider(canonical=_store("ja"))

    payload = json.loads(provider._handle_recall({"query": RELATED_QUERY}))

    assert FIXTURES["ja"]["slots"][1][2] in [row.get("content") for row in payload["results"]]


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_canonical_matching_stays_owner_scoped(path_name, path):
    """Rows stored for another owner must not match, in either path."""

    store = OwnerScopedCanonicalStore({
        "other-profile": [
            {
                "body": FIXTURES["ja"]["slots"][1][2],
                "category": "preference",
                "name": "tea",
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]
    })

    assert list(path(store, "default", RELATED_QUERY)) == []
    assert store.requested_owner_ids == ["default"]


def test_prefetch_public_path_stays_owner_scoped():
    provider = _provider(
        canonical=OwnerScopedCanonicalStore({
            "other-profile": [
                {
                    "body": FIXTURES["ja"]["slots"][1][2],
                    "category": "preference",
                    "name": "tea",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        }),
        identity="default",
    )

    assert FIXTURES["ja"]["slots"][1][2] not in provider.prefetch(RELATED_QUERY)


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_cyrillic_canonical_matching_is_unchanged(path_name, path):
    """Non-CJK scripts keep the pre-#971 word-token behaviour."""

    store = FakeCanonicalStore([
        {
            "body": "Пользователь предпочитает тёмную резервную копию.",
            "category": "preference",
            "name": "backup_style",
            "created_at": "2026-01-01T00:00:00Z",
        }
    ])

    assert [row.get("canonical_name") for row in path(store, "default", "Найди тёмную резервную копию")] == ["backup_style"]



def test_identical_cjk_span_does_not_bypass_the_stop_unit_filter(monkeypatch):
    """A shared run must not match through the raw word token.

    _PREFETCH_TOKEN_RE uses Unicode ``\w``, so before this was masked the whole
    CJK run also arrived as one word token and an identical query/row pair
    matched even when every 2-gram of that run was configured as a stop unit.
    """

    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CJK_STOP_UNITS", "状況,況確,確認")
    body = "状況確認"

    store = FakeCanonicalStore([
        {"body": body, "category": "procedure", "name": "status_check", "created_at": "2026-01-01T00:00:00Z"}
    ])

    for path_name, path in CANONICAL_PATHS:
        assert path(store, "default", body) == [], f"{path_name} matched a fully filtered run"


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_latin_canonical_matching_is_unchanged(path_name, path):
    """Latin words keep their pre-#971 behaviour: two shared words still match."""

    store = FakeCanonicalStore([
        {
            "body": "SampleOwner prefers Cedar Bakery over Harbor Bakery.",
            "category": "preference",
            "name": "bakery",
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "body": "A lightweight workflow unrelated to tea.",
            "category": "workflow",
            "name": "workflow",
            "created_at": "2026-01-01T00:00:00Z",
        },
    ])

    names = [row.get("canonical_name") for row in path(store, "default", "Cedar Bakery preference")]

    assert "bakery" in names
    assert "workflow" not in names


# --- dedup boundary ------------------------------------------------------------------
# Review (coderabbitai on #975): the dedup step compared per-character tokens, so two
# rows that merely shared characters were treated as duplicates and one of them was
# dropped before injection / before the recall result list was returned.


# ---------------------------------------------------------------------------
# Review round on #975 (dplush, CHANGES_REQUESTED): iteration marks, bridge
# units inside function-word spans, and the Japanese middle dot.
# ---------------------------------------------------------------------------

def _rows_store(rows):
    return FakeCanonicalStore([
        {
            "body": body,
            "category": category,
            "name": name,
            "created_at": "2026-01-01T00:00:00Z",
        }
        for category, name, body in rows
    ])


def test_iteration_mark_folds_into_one_unit():
    assert _prefetch_lexical_units("佐々木") == {"佐木"}
    assert _prefetch_lexical_units("佐々野") == {"佐野"}
    assert not (_prefetch_lexical_units("佐々木") & _prefetch_lexical_units("佐々野"))


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_iteration_mark_does_not_collapse_two_names(path_name, path):
    store = _rows_store([
        ("identity", "sasaki", "佐々木"),
        ("identity", "sasano", "佐々野"),
    ])

    # Before the fix both queries produced one character per unit, so 佐々野
    # shared 佐 with 佐々木 and reached the slot.
    assert [row.get("canonical_name") for row in path(store, "default", "佐々木")] == ["sasaki"]
    assert [row.get("canonical_name") for row in path(store, "default", "佐々野")] == ["sasano"]


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_iteration_mark_query_does_not_reach_a_longer_sentence(path_name, path):
    store = _rows_store([("identity", "family_name", "利用者の姓は「佐々木」である。")])

    assert path(store, "default", "佐々野") == []
    assert path(store, "default", "佐々野の予定は？") == []
    assert [row.get("canonical_name") for row in path(store, "default", "佐々木")] == ["family_name"]


def test_function_word_spans_leave_no_bridge_units():
    assert _prefetch_lexical_units("すること") == set()

    units = _prefetch_lexical_units("確認すること")
    assert "るこ" not in units
    assert "確認" in units


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_bridge_units_inside_function_words_do_not_match_unrelated_slots(path_name, path):
    store = _rows_store([
        ("procedure", "backup", "バックアップを確認すること。"),
        ("procedure", "booking", "予約は受付で行うこと。"),
    ])

    matched = [row.get("canonical_name") for row in path(store, "default", "予約すること")]
    assert matched == ["booking"]


def test_dedup_signature_ignores_the_cjk_stop_unit_configuration(monkeypatch):
    """Canonical stop units tune slot matching, not ordinary deduplication.

    With ``部署`` configured as a stop unit the canonical units for 部署計画
    change (部署 disappears), which is the documented contract. The dedup
    signature must stay identical, so the same pair still collapses to one row
    (review: coderabbitai on #975).
    """

    rows = [{"content": "部署"}, {"content": "部署計画"}]
    assert len(_semantic_dedup_prefetch([dict(row) for row in rows])) == 1

    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CJK_STOP_UNITS", "部署")

    assert _prefetch_lexical_units("部署計画") == {"署計", "計画"}, "canonical units must follow the config"
    assert len(_semantic_dedup_prefetch([dict(row) for row in rows])) == 1, "ordinary dedup must not"


BACKUP_A = "バックアップの保存先は外付けディスク。"
BACKUP_B = "バックアップの保存先は外付けディスクとする。"


def test_prefetch_dedup_ignores_the_cjk_stop_unit_configuration(monkeypatch):
    """The public prefetch path deduplicates ordinary rows the same way."""

    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CJK_STOP_UNITS", "保存先")
    provider = _provider(results=[_working_row(BACKUP_A), _working_row(BACKUP_B)])

    block = provider.prefetch("バックアップの保存先は？")

    assert BACKUP_A in block, "the kept ordinary row must still be injected"
    assert BACKUP_B not in block, "a duplicate ordinary row must still be dropped"


def test_recall_dedup_ignores_the_cjk_stop_unit_configuration(monkeypatch):
    """mnemosyne_recall merges canonical rows without changing ordinary dedup."""

    monkeypatch.setenv("MNEMOSYNE_PREFETCH_CJK_STOP_UNITS", "保存先")
    canonical = FakeCanonicalStore([
        {
            "body": "週次のバックアップ手順を実行し、保存先は外付けディスクとする。",
            "category": "procedure",
            "name": "backup",
            "created_at": "2026-01-01T00:00:00Z",
        },
    ])
    server_a = "社内の共有サーバーは業務時間内のみ稼働する。"
    server_b = "社内の共有サーバーは業務時間内のみ稼働する運用です。"
    provider = _provider(canonical=canonical, results=[_working_row(server_a), _working_row(server_b)])

    payload = json.loads(provider._handle_recall({"query": "バックアップの保存先は？"}))
    contents = [row.get("content") for row in payload["results"]]

    assert canonical._rows[0]["body"] in contents, "the canonical row must still be merged"
    assert server_a in contents, "the kept ordinary row must still be returned"
    assert server_b not in contents, "a duplicate ordinary row must still be dropped"


def test_middle_dot_separates_cjk_runs():
    assert _prefetch_lexical_units("猫・犬") == {"猫", "犬"}
    assert _prefetch_lexical_units("猫・犬") == _prefetch_lexical_units("猫、犬")


@pytest.mark.parametrize("path_name,path", CANONICAL_PATHS, ids=CANONICAL_PATH_IDS)
def test_middle_dot_separated_entry_matches_as_separate_runs(path_name, path):
    store = _rows_store([
        ("preference", "pets", "猫・犬"),
        ("preference", "birds", "鳥・犬"),
    ])

    # Before the fix 猫・犬 produced {"猫・", "・犬"}, so neither entry matched.
    assert [row.get("canonical_name") for row in path(store, "default", "猫")] == ["pets"]
    assert [row.get("canonical_name") for row in path(store, "default", "鳥")] == ["birds"]


def _shared_character_store():
    return FakeCanonicalStore([
        {
            "body": SHARED_TEXT,
            "category": "fact",
            "name": "meeting_room",
            "created_at": "2026-01-01T00:00:00Z",
        },
    ])


SHARED_TEXT = "会議室の予約ルール"
SPACED_TEXT = "会 議 室 の 予 約 ル ー ル"


def test_dedup_keeps_rows_that_only_share_characters():
    """Character overlap is not lexical overlap: 2-grams vs single characters."""

    shared = _prefetch_tokens(SHARED_TEXT) & _prefetch_tokens(SPACED_TEXT)
    assert len(shared) == 8, f"fixture no longer reproduces the old collision: {shared}"
    assert not (_prefetch_lexical_units(SHARED_TEXT) & _prefetch_lexical_units(SPACED_TEXT))

    kept = _semantic_dedup_prefetch([{"content": SHARED_TEXT}, {"content": SPACED_TEXT}])

    assert [row["content"] for row in kept] == [SHARED_TEXT, SPACED_TEXT]


def test_dedup_still_collapses_true_duplicates():
    kept = _semantic_dedup_prefetch([
        {"content": "会議室の予約ルールは総務が管理している。"},
        {"content": "会議室の予約ルールは総務が管理している"},
    ])

    assert len(kept) == 1


def test_dedup_keeps_a_row_whose_units_are_all_function_words():
    """A row with no topical unit must survive instead of losing its signature."""

    kept = _semantic_dedup_prefetch([{"content": "すること"}])

    assert len(kept) == 1


def test_prefetch_public_path_keeps_both_rows_that_only_share_characters():
    provider = _provider(canonical=_shared_character_store(), results=[_working_row(SPACED_TEXT)])

    block = provider.prefetch(SHARED_TEXT)

    assert SHARED_TEXT in block, "the canonical row was dropped by dedup"
    assert SPACED_TEXT in block, "the ordinary recall result was dropped by dedup"


def test_recall_tool_path_keeps_both_rows_that_only_share_characters():
    provider = _provider(canonical=_shared_character_store(), results=[_working_row(SPACED_TEXT)])

    payload = json.loads(provider._handle_recall({"query": SHARED_TEXT}))

    contents = [row.get("content") for row in payload["results"]]
    assert SHARED_TEXT in contents, "the canonical row was dropped by dedup"
    assert SPACED_TEXT in contents, "the ordinary recall result was dropped by dedup"
