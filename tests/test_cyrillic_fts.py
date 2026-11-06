"""
Tests for the Cyrillic (Russian) FTS5 fallback added alongside this file.

Background: the default unicode61 FTS5 tokenizer has no built-in stemmer
for inflected Russian surface forms (тёмная/тёмную/тёмной, встреча/
встречи/встречу), so a Russian query with one inflection matches only
that exact surface form. The fallback in ``_cyrillic_like_search`` is
triggered when FTS5 returns zero rows for a query that contains at least
one Cyrillic character. It scans ``working_memory``/``episodic_memory``
with LIKE and re-ranks candidates by trigram Jaccard similarity.

These tests pin the contract:

- _has_cyrillic detects Russian and other Cyrillic scripts covered by
  [а-яёА-ЯЁ] (Ukrainian, Bulgarian, etc. — but NOT Serbian/Mongolian)
- _ngrams produces the expected set for short and long strings
- _cyrillic_score is 0 for non-overlapping content and high for
  matching inflections of the same lemma
- _cyrillic_like_search returns rows in score-descending order and
  reaches across grammatical cases (nominative vs accusative vs
  prepositional, etc.)
- _fts_search / _fts_search_working route Cyrillic queries to the
  fallback when FTS5 returns nothing
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import (
    BeamMemory,
    init_beam,
    _has_cyrillic,
    _ngrams,
    _cyrillic_score,
    _cyrillic_like_search,
    _fts_search,
    _fts_search_working,
    _lexical_relevance,
    _recall_tokens,
)


# ---------------------------------------------------------------------------
# Detection and primitives
# ---------------------------------------------------------------------------

class TestHasCyrillic:
    def test_empty(self):
        assert _has_cyrillic("") is False

    def test_pure_ascii(self):
        assert _has_cyrillic("hello world") is False
        assert _has_cyrillic("User prefers dark mode") is False

    def test_pure_latin_with_accents(self):
        # Latin diacritics are not Cyrillic
        assert _has_cyrillic("café déjà vu") is False

    def test_pure_cyrillic(self):
        assert _has_cyrillic("тёмная") is True
        assert _has_cyrillic("встреча") is True
        assert _has_cyrillic("українська") is True
        assert _has_cyrillic("български") is True

    def test_mixed_ru_en(self):
        assert _has_cyrillic("backend Python — это бэкенд") is True
        assert _has_cyrillic("dark mode тёмная тема") is True

    def test_case_insensitive(self):
        # Uppercase Cyrillic letters must also trigger
        assert _has_cyrillic("ТЁМНАЯ") is True
        assert _has_cyrillic("Русский") is True

    def test_yo_letter(self):
        # ё / Ё are explicitly included in the regex
        assert _has_cyrillic("ёжик") is True
        assert _has_cyrillic("ЁЖИК") is True


class TestNgrams:
    def test_short_string_returns_whole(self):
        # Strings shorter than n are returned as a single n-gram
        assert _ngrams("ab", 3) == {"ab"}
        assert _ngrams("кот", 3) == {"кот"}

    def test_exact_length(self):
        # Length == n: exactly one n-gram
        assert _ngrams("тём", 3) == {"тём"}

    def test_longer_string(self):
        # Sliding window over "тёмная" (6 chars, n=3): 4 trigrams
        assert _ngrams("тёмная", 3) == {"тём", "ёмн", "мна", "ная"}

    def test_unique_ngrams(self):
        # The set deduplicates repeated n-grams
        assert _ngrams("аааа", 3) == {"ааа"}


# ---------------------------------------------------------------------------
# Trigram Jaccard scoring
# ---------------------------------------------------------------------------

class TestCyrillicScore:
    def test_empty_inputs(self):
        assert _cyrillic_score("", "") == 0.0
        assert _cyrillic_score("тёмная", "") == 0.0
        assert _cyrillic_score("", "тёмная") == 0.0

    def test_identical_strings(self):
        # Trigram Jaccard of a string with itself is 1.0
        assert _cyrillic_score("тёмная", "тёмная") == pytest.approx(1.0)

    def test_inflection_overlap(self):
        # Different cases of the same lemma should overlap heavily:
        # "тёмная" trigrams {тём, ёмн, мна, ная}
        # "тёмную"  trigrams {тём, ёмн, мну, ную}
        # shared = {тём, ёмн} -> 2 / (4 + 4 - 2) = 0.333
        s = _cyrillic_score("тёмная", "тёмную")
        assert 0.25 <= s <= 0.4

    def test_unrelated_words_score_low(self):
        # "тёмная" vs "встреча" share no trigrams at all -> 0
        assert _cyrillic_score("тёмная", "встреча") == 0.0

    @pytest.mark.parametrize("query,content,expected", [
        ("ежик", "ёжик", 1.0),
        ("ЁЖИК", "ежик", 1.0),
        ("свеклы", "свёкла", 0.6),
        ("СВЁКЛЫ", "свекла", 0.6),
        ("тёмная", "темную", 1 / 3),
    ])
    def test_yo_vs_e_collapses(self, query, content, expected):
        # Cross-spelling pairs must work, including uppercase and inflection.
        assert _cyrillic_score(query, content) == pytest.approx(expected)

    def test_yo_normalization_preserves_query_word_mean(self):
        # Folding spelling does not make unrelated query words disappear from
        # the denominator or change the aggregation policy discussed in #1295.
        assert _cyrillic_score("большая свеклы", "свёкла") == pytest.approx(0.3)
        assert _cyrillic_score(
            "какая у него любимая клавиатура и буквы на ней",
            "он рассказал про механическую клавиатуру с подсветкой вчера",
        ) == pytest.approx((7 / 9) / 6)

    @pytest.mark.parametrize("query,content", [
        ("жёлтый", "автобус"),
        ("автобус", "ёлочка"),
    ])
    def test_yo_normalization_does_not_admit_unrelated_words(self, query, content):
        assert _cyrillic_score(query, content) == 0.0

    def test_short_query_words_ignored(self):
        # 1-2 char words are skipped to avoid stop-word noise
        # "а" alone: 0 valid query words -> 0
        assert _cyrillic_score("а", "автобус") == 0.0
        # "он" is 2 chars: still skipped
        assert _cyrillic_score("он", "автобус") == 0.0

    def test_mixed_script_query(self):
        # Latin words also pass through the regex [а-яёa-z0-9]
        # and participate in the Jaccard computation
        s = _cyrillic_score("backend Python", "Backend на Python")
        assert s > 0.5  # strong overlap

    def test_punctuation_does_not_disturb(self):
        # Punctuation between words must not affect scoring
        s1 = _cyrillic_score("тёмная, тема!", "Тёмная тема оформления.")
        s2 = _cyrillic_score("тёмная тема", "Тёмная тема оформления")
        assert s1 == pytest.approx(s2)


# ---------------------------------------------------------------------------
# End-to-end: fallback through _fts_search
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_cyrillic.db"
        init_beam(db_path)
        yield db_path


def _seed_working(db_path: Path, rows: list[tuple[str, str, float]]) -> None:
    """Insert a few rows into working_memory (and fts_working mirror).

    rows: list of (id, content, importance) tuples
    """
    conn = sqlite3.connect(db_path)
    try:
        for mid, content, importance in rows:
            conn.execute(
                "INSERT INTO working_memory (id, content, importance) VALUES (?, ?, ?)",
                (mid, content, importance),
            )
            conn.execute(
                "INSERT INTO fts_working (id, content) VALUES (?, ?)",
                (mid, content),
            )
        conn.commit()
    finally:
        conn.close()


def _seed_episodic(db_path: Path, rows: list[tuple[str, str, float]]) -> None:
    """Insert a few rows into episodic_memory (and fts_episodes mirror).

    rows: list of (id, content, importance) tuples. The id is a TEXT
    unique identifier (separate from the implicit rowid).
    """
    conn = sqlite3.connect(db_path)
    try:
        for mid, content, importance in rows:
            conn.execute(
                "INSERT INTO episodic_memory (id, content, importance) VALUES (?, ?, ?)",
                (mid, content, importance),
            )
            conn.execute(
                "INSERT INTO fts_episodes (rowid, content) VALUES ((SELECT rowid FROM episodic_memory WHERE id = ?), ?)",
                (mid, content),
            )
        conn.commit()
    finally:
        conn.close()


def _open(db_path: Path) -> sqlite3.Connection:
    """Open a connection with dict-like row access.

    ``_fts_search`` and friends index rows by column name (``r["id"]``),
    so tests must use ``sqlite3.Row`` factory.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.mark.parametrize("working", [True, False])
@pytest.mark.parametrize("query,content", [
    ("свеклы", "Свёкла тушёная хранится отдельно"),
    ("свёклы", "Свекла тушеная хранится отдельно"),
    ("СВЕКЛЫ", "СВЁКЛА ТУШЁНАЯ ХРАНИТСЯ ОТДЕЛЬНО"),
    ("СВЁКЛЫ", "СВЕКЛА ТУШЕНАЯ ХРАНИТСЯ ОТДЕЛЬНО"),
])
def test_yo_spelling_fallback_reaches_both_memory_tiers(temp_db, working, query, content):
    seed = _seed_working if working else _seed_episodic
    seed(temp_db, [
        ("target", content, 0.8),
        ("unrelated", "Автобус отправляется утром", 0.8),
    ])
    conn = _open(temp_db)
    try:
        # No exact FTS match: the query and stored word have different endings.
        fts_table = "fts_working" if working else "fts_episodes"
        assert conn.execute(
            f"SELECT count(*) FROM {fts_table} WHERE {fts_table} MATCH ?", (query,)
        ).fetchone()[0] == 0
        direct = _cyrillic_like_search(conn, query, k=5, working=working)
        routed = (_fts_search_working if working else _fts_search)(conn, query, k=5)
        table = "working_memory" if working else "episodic_memory"
        assert conn.execute(
            f"SELECT content FROM {table} WHERE id = ?", ("target",)
        ).fetchone()[0] == content
    finally:
        conn.close()
    result_key, expected_id = ("id", "target") if working else ("rowid", 1)
    for results in (direct, routed):
        assert [row[result_key] for row in results] == [expected_id]
        assert results[0]["rank"] == pytest.approx(-0.6)


@pytest.mark.parametrize("query,content", [
    ("свеклы", "Свёкла тушёная хранится отдельно"),
    ("свёклы", "Свекла тушеная хранится отдельно"),
])
def test_yo_spelling_passes_linear_recall_gate(temp_db, monkeypatch, query, content):
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.delenv("MNEMOSYNE_LEXICAL_GATE_MIN", raising=False)
    memory = BeamMemory(session_id="yo-normalization", db_path=temp_db)
    target = memory.remember(content, source="test", importance=0.8)
    memory.remember("Автобус отправляется утром", source="test", importance=0.8)
    relevance = _lexical_relevance(_recall_tokens(query), content, query)
    assert relevance == pytest.approx(0.6)
    results = memory.recall(query, top_k=5)
    assert [row["id"] for row in results] == [target]


class TestCyrillicLikeSearchWorking:
    """Verify _cyrillic_like_search against working_memory."""

    def test_finds_inflected_form(self, temp_db):
        # Stored: "тёмную" (accusative) and "тёмной" (prepositional).
        # Query: "тёмная" (nominative) — the inflection NOT in the DB.
        # Direct FTS5 would return zero. Fallback must find both.
        _seed_working(temp_db, [
            ("m1", "Пользователь предпочитает тёмную тему оформления", 0.9),
            ("m2", "Тёмная комната была неуютной", 0.5),
            ("m3", "Совершенно посторонний текст про погоду", 0.5),
        ])
        conn = _open(temp_db)
        try:
            results = _cyrillic_like_search(conn, "тёмная", k=5, working=True)
        finally:
            conn.close()
        ids = [r["id"] for r in results]
        # Both m1 and m2 must surface (their surface forms are inflections
        # of the same lemma as the query).
        assert "m1" in ids
        assert "m2" in ids
        # m3 has no overlap with the query -> must NOT appear.
        assert "m3" not in ids

    def test_returns_empty_for_latin_query(self, temp_db):
        # Cyrillic fallback must not engage on non-Cyrillic text
        _seed_working(temp_db, [
            ("m1", "User prefers dark mode", 0.9),
        ])
        conn = _open(temp_db)
        try:
            results = _cyrillic_like_search(conn, "dark mode", k=5, working=True)
        finally:
            conn.close()
        assert results == []

    def test_returns_empty_when_no_candidates(self, temp_db):
        # Cyrillic query against Cyrillic-free corpus -> 0
        _seed_working(temp_db, [
            ("m1", "User prefers dark mode", 0.9),
        ])
        conn = _open(temp_db)
        try:
            results = _cyrillic_like_search(conn, "тёмная", k=5, working=True)
        finally:
            conn.close()
        assert results == []

    def test_ranking_is_descending(self, temp_db):
        _seed_working(temp_db, [
            ("m1", "Встреча с клиентом в пятницу", 0.5),  # exact match
            ("m2", "Встречи бывают продуктивными", 0.5),  # only "встреч"
            ("m3", "Шум в коридоре", 0.5),                # no overlap
        ])
        conn = _open(temp_db)
        try:
            results = _cyrillic_like_search(conn, "встреча пятница", k=5, working=True)
        finally:
            conn.close()
        # m1 should beat m2; m3 should not appear (score 0)
        ids = [r["id"] for r in results]
        assert "m1" in ids
        if "m2" in ids:
            assert ids.index("m1") < ids.index("m2")
        assert "m3" not in ids


class TestCyrillicLikeSearchEpisodic:
    """Verify _cyrillic_like_search against episodic_memory."""

    def test_finds_across_grammatical_cases(self, temp_db):
        # Stored accusative, queried with prepositional: "в тёмной"
        _seed_episodic(temp_db, [
            ("ep1", "Пользователь предпочитает тёмную тему", 0.9),
            ("ep2", "Светлая комната была уютной", 0.5),
        ])
        conn = _open(temp_db)
        try:
            results = _cyrillic_like_search(
                conn, "тёмной", k=5, working=False,
            )
        finally:
            conn.close()
        rowids = [r["rowid"] for r in results]
        assert 1 in rowids
        assert 2 not in rowids  # irrelevant


class TestFtsSearchRoutesCyrillic:
    """End-to-end: _fts_search routes Cyrillic queries to the fallback."""

    def test_working_memory_fallback(self, temp_db):
        # Seed only via the FTS5 mirror so that FTS5's MATCH must be the
        # path that returns zero. We use a stored surface form different
        # from the query surface form to provoke the fallback.
        _seed_working(temp_db, [
            ("m1", "Пользователь предпочитает тёмную тему", 0.9),
        ])
        conn = _open(temp_db)
        try:
            rows = _fts_search_working(conn, "тёмная", k=5)
        finally:
            conn.close()
        assert any(r["id"] == "m1" for r in rows)

    def test_episodic_fallback(self, temp_db):
        _seed_episodic(temp_db, [
            ("ep1", "Пользователь предпочитает тёмную тему", 0.9),
        ])
        conn = _open(temp_db)
        try:
            rows = _fts_search(conn, "тёмная", k=5)
        finally:
            conn.close()
        assert any(r["rowid"] == 1 for r in rows)

    def test_latin_query_does_not_trigger_cyrillic_fallback(self, temp_db):
        # Sanity: a Latin-only query against a Cyrillic-free corpus
        # should return rows via the normal FTS5 path, not via our
        # Cyrillic fallback.
        _seed_working(temp_db, [
            ("m1", "User prefers dark mode interfaces", 0.9),
        ])
        conn = _open(temp_db)
        try:
            rows = _fts_search_working(conn, "dark mode", k=5)
        finally:
            conn.close()
        assert any(r["id"] == "m1" for r in rows)


class TestLexicalRelevanceCyrillicGate:
    """Verify the relevance gate admits inflected Cyrillic forms.

    Before this fix, `_lexical_relevance` returned 0.0 for short inflected
    Cyrillic queries (тёмная vs тёмную) because exact token matching
    cannot span Russian case endings. The scoring pipeline generated the
    candidate via `_cyrillic_like_search` (LIKE + trigram Jaccard) but
    then dropped it at the relevance gate, so the recall returned 0
    results for the very queries Cyrillic users actually type.
    """

    def test_nonzero_prior_score_still_suppresses_fallback(self, monkeypatch):
        def unexpected_fallback(*args):
            pytest.fail("a nonzero lexical score must not invoke Cyrillic fallback")

        monkeypatch.setattr(beam_module, "_cyrillic_score", unexpected_fallback)
        query = "рецепт блюдо кухня свеклы"
        assert _lexical_relevance(
            _recall_tokens(query), "кухня свёкла", query
        ) == pytest.approx(0.25)

    def test_inflected_query_passes_gate(self):
        # Nominative "тёмная тема" vs accusative "тёмную тему":
        # exact match returns 0.0; Cyrillic fallback returns trigram
        # Jaccard, which should be > the 0.15 threshold for a 2-token
        # query.
        q = "тёмная тема"
        content = "Пользователь предпочитает тёмную тему интерфейса"
        rel = _lexical_relevance(_recall_tokens(q), content, q)
        cyr_score = _cyrillic_score(q, content)
        assert rel > 0.0, f"gate returned 0.0 for inflected Cyrillic (cyr_score={cyr_score})"
        assert rel == pytest.approx(cyr_score), "fallback should mirror _cyrillic_score"

    def test_exact_form_still_full_score(self):
        # When the query and content use the same surface form, the
        # exact-token path should still return 1.0 and not get diluted
        # by the Cyrillic fallback.
        q = "тёмную тему"
        content = "Пользователь предпочитает тёмную тему интерфейса"
        rel = _lexical_relevance(_recall_tokens(q), content, q)
        assert rel == pytest.approx(1.0)

    def test_english_query_unaffected(self):
        # Non-Cyrillic queries must not engage the Cyrillic fallback.
        q = "dark mode"
        content = "User prefers dark mode interfaces"
        rel = _lexical_relevance(_recall_tokens(q), content, q)
        assert rel == pytest.approx(1.0)

    def test_latin_query_against_cyrillic_content_returns_zero(self):
        # Latin query vs Cyrillic content: no token overlap, no Cyrillic
        # in query, so neither path fires. The fallback would be wrong
        # if it engaged here.
        q = "dark mode"
        content = "Пользователь предпочитает тёмную тему"
        rel = _lexical_relevance(_recall_tokens(q), content, q)
        assert rel == 0.0

    def test_second_decline_inflection_passes_gate(self):
        # "резервная копия" vs "Резервное копирование": the second
        # word's inflection (копия -> копирование) also defeats exact
        # matching. The Cyrillic fallback should still admit this.
        q = "резервная копия"
        content = "Резервное копирование Яндекс Диска включено по расписанию"
        rel = _lexical_relevance(_recall_tokens(q), content, q)
        assert rel > 0.0, f"gate returned 0.0 for inflected Cyrillic (rel={rel})"
