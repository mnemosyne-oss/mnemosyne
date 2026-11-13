"""Lexical relevance prepares the query once per recall pass (#1124).

``_lexical_relevance()`` used to rebuild every query-side input (augmented
tokens, components, weights, CJK set, Hangul and Cyrillic flags) for each
candidate row, and the working-memory pre-pass tokenized each row a second
time. These tests pin the refactor to the exact floats of the scorer it
replaced and count the preparations ``recall()`` performs.
"""

import random
import re

import pytest

from mnemosyne.core import beam
from mnemosyne.core.beam import (
    BeamMemory,
    _HANGUL_PREFIX_MATCH_WEIGHT,
    _RECALL_SYNONYMS,
    _RECALL_TOKEN_RE,
    _component_unit_weight,
    _cyrillic_score,
    _has_cyrillic,
    _has_hangul,
    _hyphen_components,
    _hyphen_fragment_tokens,
    _is_meaningful_recall_token,
    _leading_hyphen_fragments,
    _lexical_relevance,
    _prepare_lexical_content,
    _prepare_lexical_query,
    _prepared_lexical_relevance,
    _recall_tokens,
    _symbolic_code_tokens,
)


def _reference_lexical_relevance(query_tokens, content, query_lower=""):
    """Per-row scorer as it stood before #1124, comments removed."""
    content_lower = content.lower()
    query_cjk = {
        ch for ch in query_lower
        if "\u4e00" <= ch <= "\u9fff"
        or "\u3040" <= ch <= "\u30ff"
        or "\uac00" <= ch <= "\ud7af"
    }
    if query_lower:
        query_tokens = [*query_tokens, *_hyphen_fragment_tokens(query_lower)]
        query_tokens = [*query_tokens, *_symbolic_code_tokens(query_lower)]
        query_tokens = [*query_tokens, *_leading_hyphen_fragments(query_lower)]
        query_tokens = list(dict.fromkeys(query_tokens))
    if query_tokens:
        token_chars = {ch for token in query_tokens for ch in token}
        query_cjk &= token_chars
    if not query_tokens and not query_cjk:
        return 0.0
    component_groups = [_hyphen_components(token) for token in query_tokens]
    lexical_unit_count = sum(
        _component_unit_weight(components) for components in component_groups
    )
    content_tokens = set(_recall_tokens(content_lower))
    content_tokens.update(_hyphen_fragment_tokens(content_lower))
    content_tokens.update(_symbolic_code_tokens(content_lower))
    content_tokens.update(_leading_hyphen_fragments(content_lower))
    expanded_content_tokens = set(content_tokens)
    for token in list(content_tokens):
        expanded_content_tokens.update(
            part for part in re.split(r"[_:/.-]+", token)
            if _is_meaningful_recall_token(part)
        )
    content_tokens = expanded_content_tokens
    short_stems = {
        token for token in query_tokens
        if len(token) < 3 and not _has_hangul(token)
    }
    if short_stems:
        content_tokens.update(
            short_stems.intersection(_RECALL_TOKEN_RE.findall(content_lower))
        )
    if not content_tokens and not query_cjk:
        return 0.0
    exact = 0.0
    partial = 0.0
    for token, components in zip(query_tokens, component_groups, strict=True):
        if token in content_tokens:
            exact += _component_unit_weight(components)
            continue
        if _has_hangul(token) and any(
            ctoken.startswith(token) for ctoken in content_tokens
        ):
            exact += _component_unit_weight(components) * _HANGUL_PREFIX_MATCH_WEIGHT
            continue
        component_hits = sum(part in content_tokens for part in components)
        if len(components) >= 2 and component_hits >= 2:
            exact += component_hits
            continue
        if components:
            continue
        synonyms = _RECALL_SYNONYMS.get(token, ())
        if synonyms and any(syn in content_tokens for syn in synonyms):
            partial += 0.75
            continue
        if (
            len(token) >= 4
            and not (_has_hangul(query_lower) and not _has_hangul(token))
            and any(
                token in ctoken or ctoken in token
                for ctoken in content_tokens
                if len(ctoken) >= 4
            )
        ):
            partial += 0.4
    if _has_hangul(query_lower):
        full_match = (
            1.0 if query_tokens and set(query_tokens) <= content_tokens else 0.0
        )
    else:
        full_match = 1.0 if query_lower and query_lower in content_lower else 0.0
    score = (exact + partial + full_match) / max(lexical_unit_count, 1)
    if score == 0.0:
        if query_cjk:
            content_cjk = {
                ch for ch in content_lower
                if "\u4e00" <= ch <= "\u9fff"
                or "\u3040" <= ch <= "\u30ff"
                or "\uac00" <= ch <= "\ud7af"
            }
            score = len(query_cjk & content_cjk) / len(query_cjk)
        elif _has_cyrillic(query_lower):
            score = _cyrillic_score(query_lower, content_lower)
    return min(score, 1.0)


_VOCAB = [
    "orion-gateway", "orion-telemetrie", "telemetrie", "gateway", "port", "4831",
    "branding", "brand", "positioning", "preference", "prefers", "current", "now",
    "lifecycle", "lifecycle.log", "telemetry_api_latency_ms", "latency", "api",
    "deploy", "deployment", "deployed", "redeploy", "modelforge", "modelforgexyz",
    "--force", "force", "forceful", "foo--force", "rm", "-rf", "-v", "python",
    "c++", "c#", "g++", "node_modules", "git-rebase", "ai", "AI가", "ok", "db",
    "the", "and", "use", "백업", "백업을", "캐시", "보관", "보관한다", "바나나",
    "바나나우유", "설정을", "가", "東京", "数据库", "备份", "ポート", "設定",
    "тёмная", "тёмную", "резервная", "копия", "база", "stoßlüften", "mensa-plan",
    "supercalifragilisticexpialidocious-configuration-management-subsystem",
]
_SEPARATORS = [" ", " ", " ", ", ", ". ", "\n", " (", ") ", ": "]


def _random_text(rnd, low, high):
    words = []
    for _ in range(rnd.randint(low, high)):
        words.append(rnd.choice(_VOCAB))
        words.append(rnd.choice(_SEPARATORS))
    text = "".join(words).strip()
    if rnd.random() < 0.3:
        text = text.upper() if rnd.random() < 0.5 else text.title()
    return text


def _corpus():
    rnd = random.Random(1124)
    contents = [_random_text(rnd, 1, 30) for _ in range(60)] + [
        "", "The user does not like rm -rf.", "The user codes in C++.",
        "We deploy with --force after review.", "바나나우유를 좋아한다",
        "Der Orion-Gateway nutzt Port 4831 für interne Telemetrie.",
        "東京の設定", "резервная копия базы", "Rotated lifecycle.log at noon.",
    ]
    queries = [_random_text(rnd, 1, 10) for _ in range(50)] + [
        "", "the and use", "--force", "rm -rf", "c++", "AI가 target meaning",
        "바나나", "보관", "orion-telemetrie", "ModelForge가", "lifecycle",
        "тёмная тема", "東京", "branding preference", "deploymentpipeline",
    ]
    return queries, contents


def test_prepared_scorer_returns_the_exact_floats_of_the_per_row_scorer():
    queries, contents = _corpus()
    compared = 0
    for query in queries:
        query_lower = query.lower()
        tokens = _recall_tokens(query_lower)
        for query_tokens, scorer_lower in (
            (tokens, query_lower),
            (tokens, ""),
            ([], query_lower),
            (tokens[:3], query_lower),
        ):
            prepared = _prepare_lexical_query(query_tokens, scorer_lower)
            for content in contents:
                expected = _reference_lexical_relevance(
                    list(query_tokens), content, scorer_lower
                )
                assert _lexical_relevance(list(query_tokens), content, scorer_lower) == expected
                if prepared.matchable:
                    assert _prepared_lexical_relevance(
                        prepared, _prepare_lexical_content(content)
                    ) == expected
                compared += 1
    assert compared == len(queries) * 4 * len(contents)


def test_prepared_query_is_reusable_across_rows():
    """A substring cache filled by one row must not change the next row's score."""
    query_lower = "deploymentpipeline telemetryservice"
    prepared = _prepare_lexical_query(_recall_tokens(query_lower), query_lower)
    rows = [
        "The deployment finished.",
        "Nothing relevant here.",
        "telemetry and service logs",
        "The deployment finished.",
    ]
    scores = [
        _prepared_lexical_relevance(prepared, _prepare_lexical_content(row))
        for row in rows
    ]
    assert scores == [
        _reference_lexical_relevance(_recall_tokens(query_lower), row, query_lower)
        for row in rows
    ]
    assert scores[0] == scores[3]


class _Counter:
    def __init__(self, monkeypatch, name):
        self.calls = []
        original = getattr(beam, name)

        def wrapper(*args, **kwargs):
            self.calls.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(beam, name, wrapper)


def _seed(tmp_path, rows=12):
    memory = BeamMemory(session_id="issue-1124", db_path=tmp_path / "memory.db")
    for i in range(rows):
        memory.remember(
            f"Row {i} mentions gateway telemetry and deploy notes.",
            source="test",
            importance=0.5,
        )
    return memory


def test_recall_prepares_the_query_once_and_each_working_row_once(tmp_path, monkeypatch):
    memory = _seed(tmp_path)
    query_prep = _Counter(monkeypatch, "_prepare_lexical_query")
    content_prep = _Counter(monkeypatch, "_prepare_lexical_content")
    tokenizer = _Counter(monkeypatch, "_recall_tokens")

    # Four or more query tokens enable the working-memory multi-hit pre-pass,
    # which tokenized every row a second time before #1124.
    results = memory.recall("gateway telemetry deploy notes", top_k=20)

    assert results
    assert len(query_prep.calls) == 1
    wm_contents = [args[0] for args in content_prep.calls]
    assert len(wm_contents) == len(set(wm_contents)) == 12
    row_tokenizations = [
        args for args in tokenizer.calls if args and args[0].startswith("row ")
    ]
    assert len(row_tokenizations) == 12


def test_unmatchable_query_tokenizes_no_rows(tmp_path, monkeypatch):
    memory = _seed(tmp_path)
    content_prep = _Counter(monkeypatch, "_prepare_lexical_content")

    assert memory.recall("the and", top_k=20) == []
    assert content_prep.calls == []


@pytest.mark.parametrize(
    "query",
    [
        "gateway telemetry deploy notes",
        "deploy --force",
        "orion-telemetrie gateway",
        "백업 캐시 gateway",
    ],
)
def test_recall_orders_results_as_the_per_row_scorer_does(tmp_path, monkeypatch, query):
    rnd = random.Random(7)
    memory = BeamMemory(session_id="issue-1124", db_path=tmp_path / "memory.db")
    for _ in range(40):
        memory.remember(_random_text(rnd, 3, 25), source="test", importance=round(rnd.random(), 3))
    # Recency would differ by microseconds between the two calls.
    monkeypatch.setattr(beam, "_recency_decay", lambda timestamp: 1.0)

    prepared = memory.recall(query, top_k=20)

    query_lower = query.lower()
    query_words = _recall_tokens(query_lower)
    monkeypatch.setattr(
        beam,
        "_prepared_lexical_relevance",
        lambda _query, content: _reference_lexical_relevance(
            query_words, content.content_lower, query_lower
        ),
    )
    monkeypatch.setattr(
        beam,
        "_lexical_relevance_for_query",
        lambda _query, content: _reference_lexical_relevance(
            query_words, content, query_lower
        ),
    )
    reference = memory.recall(query, top_k=20)

    assert [(r["id"], r["score"]) for r in prepared] == [
        (r["id"], r["score"]) for r in reference
    ]
