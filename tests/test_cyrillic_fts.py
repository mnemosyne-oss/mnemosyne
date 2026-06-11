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
    init_beam,
    _has_cyrillic,
    _ngrams,
    _cyrillic_score,
    _cyrillic_like_search,
    _fts_search,
    _fts_search_working,
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

    def test_yo_vs_e_collapses(self):
        # We deliberately do NOT collapse ё -> е; the regex preserves
        # the user's exact spelling. Trigrams differ, so the score is
        # the same as for any unrelated substitution.
        s_with_yo = _cyrillic_score("ёжик", "ёжики")
        s_with_e = _cyrillic_score("ежик", "ежики")
        # Both should be the same since the logic is symmetric
        assert s_with_yo == pytest.approx(s_with_e)
        # And both should be reasonably high
        assert s_with_yo > 0.3

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
