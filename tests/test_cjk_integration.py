"""
CJK bigram integration tests for integration/v3.15.1-cjk.

Strategy: ``init_beam`` owns its connection via ``_get_connection`` / thread-local,
so tests set ``MNEMOSYNE_DB_PATH`` to a temp file, call ``init_beam()``, then open
an independent ``_BeamConnection`` for assertions.  UDF is registered by
``_BeamConnection.__init__``, so every independently opened connection has it.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mnemosyne.core.beam import (
    _cjk_bigram,
    _register_cjk_bigram,
    _fts_query_terms,
    _has_cjk,
    _BeamConnection,
    init_beam,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open(path: str | Path) -> sqlite3.Connection:
    """Open a _BeamConnection (UDF registered via __init__)."""
    conn = sqlite3.connect(str(path), factory=_BeamConnection, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@pytest.fixture()
def db(tmp_path, monkeypatch):
    """Fresh BEAM DB in a temp dir; returns path as Path."""
    p = tmp_path / "test.db"
    monkeypatch.setenv("MNEMOSYNE_DB_PATH", str(p))
    # Reset thread-local so _get_connection picks up the new path
    import mnemosyne.core.beam as _beam
    if hasattr(_beam._thread_local, "conn") and _beam._thread_local.conn is not None:
        try:
            _beam._thread_local.conn.close()
        except Exception:
            pass
        _beam._thread_local.conn = None
    init_beam(p)
    return p


def _wm_insert(conn, wid, content, session="s1"):
    conn.execute(
        "INSERT INTO working_memory(id, content, source, timestamp, session_id) "
        "VALUES(?,?,'test','2026-01-01T00:00:00',?)",
        (wid, content, session),
    )
    conn.commit()


def _ep_insert(conn, content, session="s1", eid=None):
    import hashlib
    eid = eid or hashlib.sha1(content.encode()).hexdigest()[:16]
    conn.execute(
        "INSERT INTO episodic_memory(id, content, source, timestamp, session_id) "
        "VALUES(?,?,'test','2026-01-01T00:00:00',?)",
        (eid, content, session),
    )
    conn.commit()


def _fts_hit_working(conn, query: str) -> list[str]:
    terms = _fts_query_terms(query)
    fts_q = " OR ".join(terms)
    rows = conn.execute(
        "SELECT content FROM fts_working WHERE fts_working MATCH ?", (fts_q,)
    ).fetchall()
    return [r["content"] for r in rows]


def _fts_hit_episodic(conn, query: str) -> list[str]:
    terms = _fts_query_terms(query)
    fts_q = " OR ".join(terms)
    rows = conn.execute(
        "SELECT content FROM fts_episodes WHERE fts_episodes MATCH ?", (fts_q,)
    ).fetchall()
    return [r["content"] for r in rows]


# ---------------------------------------------------------------------------
# 1. Tokenizer unit tests
# ---------------------------------------------------------------------------

class TestCjkBigram:
    def test_pure_chinese(self):
        out = _cjk_bigram("腾讯云")
        tokens = out.split()
        assert "腾" in tokens
        assert "腾讯" in tokens
        assert "讯" in tokens
        assert "讯云" in tokens
        assert "云" in tokens

    def test_two_chars(self):
        tokens = _cjk_bigram("然然").split()
        assert "然" in tokens
        assert "然然" in tokens

    def test_mixed_latin_cjk(self):
        tokens = _cjk_bigram("Docker容器").split()
        assert "Docker" in tokens
        assert "容" in tokens
        assert "容器" in tokens
        assert "器" in tokens

    def test_mixed_number_cjk(self):
        tokens = _cjk_bigram("5美元").split()
        assert "美" in tokens
        assert "美元" in tokens
        assert "元" in tokens

    def test_pure_latin_unchanged(self):
        assert _cjk_bigram("English") == "English"

    def test_pure_latin_sentence(self):
        tokens = _cjk_bigram("hello world").split()
        assert "hello" in tokens
        assert "world" in tokens

    def test_cyrillic_preserved(self):
        tokens = _cjk_bigram("Привет мир").split()
        assert "Привет" in tokens
        assert "мир" in tokens

    def test_empty(self):
        assert _cjk_bigram("") == ""

    def test_non_string_passthrough(self):
        assert _cjk_bigram(None) is None  # type: ignore[arg-type]

    def test_extension_a(self):
        char = "\u3400"  # CJK Extension A
        tokens = _cjk_bigram(char).split()
        assert char in tokens


# ---------------------------------------------------------------------------
# 2. Fresh DB: trigger writes & FTS MATCH
# ---------------------------------------------------------------------------

class TestFreshDbTriggers:
    def test_insert_working_stores_bigram(self, db):
        conn = _open(db)
        _wm_insert(conn, "w1", "腾讯云VPS")
        raw = conn.execute("SELECT content FROM fts_working WHERE id='w1'").fetchone()
        assert raw is not None
        assert "腾讯" in raw["content"]

    def test_fts_match_chinese_working(self, db):
        conn = _open(db)
        _wm_insert(conn, "w1", "腾讯云VPS")
        hits = _fts_hit_working(conn, "腾讯云")
        assert hits, f"No FTS hit; terms={_fts_query_terms('腾讯云')}"

    def test_fts_match_chinese_episodic(self, db):
        conn = _open(db)
        _ep_insert(conn, "然然是AI助手")
        hits = _fts_hit_episodic(conn, "然然")
        assert hits, f"No episodic FTS hit; terms={_fts_query_terms('然然')}"

    def test_content_update_working(self, db):
        conn = _open(db)
        _wm_insert(conn, "w1", "旧内容")
        conn.execute("UPDATE working_memory SET content='新内容腾讯云' WHERE id='w1'")
        conn.commit()
        hits = _fts_hit_working(conn, "腾讯云")
        assert hits, "Updated content should be findable"

    def test_metadata_only_update_no_fts_churn(self, db):
        conn = _open(db)
        _wm_insert(conn, "w1", "腾讯云")
        conn.execute("UPDATE working_memory SET importance=0.9 WHERE id='w1'")
        conn.commit()
        count = conn.execute(
            "SELECT COUNT(*) FROM fts_working WHERE id='w1'"
        ).fetchone()[0]
        assert count == 1, f"Expected 1 FTS row, got {count}"

    def test_delete_working(self, db):
        conn = _open(db)
        _wm_insert(conn, "w1", "腾讯云")
        conn.execute("DELETE FROM working_memory WHERE id='w1'")
        conn.commit()
        hits = _fts_hit_working(conn, "腾讯云")
        assert not hits, "Deleted row should not be findable"

    def test_latin_still_searchable(self, db):
        conn = _open(db)
        _wm_insert(conn, "w2", "Docker container setup")
        hits = _fts_hit_working(conn, "Docker")
        assert hits, "Latin terms should still be searchable"

    def test_cyrillic_still_searchable(self, db):
        conn = _open(db)
        _wm_insert(conn, "w3", "Привет мир Docker")
        # Direct FTS query for Cyrillic term
        rows = conn.execute(
            "SELECT content FROM fts_working WHERE fts_working MATCH ?",
            ('"Привет"',),
        ).fetchall()
        assert rows, "Cyrillic should be searchable"


# ---------------------------------------------------------------------------
# 3. Legacy DB: plain-text FTS — new rows use bigram after init
# ---------------------------------------------------------------------------

class TestLegacyDb:
    def _make_legacy(self, tmp_path: Path) -> Path:
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS working_memory (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source TEXT,
                timestamp TEXT,
                session_id TEXT,
                importance REAL DEFAULT 0.5,
                metadata_json TEXT DEFAULT '{}',
                consolidated_at TEXT,
                scope TEXT DEFAULT 'session',
                valid_until TEXT,
                summary_of TEXT,
                veracity TEXT DEFAULT 'unknown',
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_working USING fts5(
                id UNINDEXED, content
            );
            INSERT INTO working_memory(id, content, source, timestamp, session_id)
                VALUES('old1','腾讯云VPS旧数据','legacy','2026-01-01T00:00:00','s0');
            INSERT INTO fts_working(id, content)
                VALUES('old1','腾讯云VPS旧数据');
        """)
        conn.commit()
        conn.close()
        return db

    def test_init_on_legacy_db_does_not_crash(self, tmp_path, monkeypatch):
        db = self._make_legacy(tmp_path)
        monkeypatch.setenv("MNEMOSYNE_DB_PATH", str(db))
        import mnemosyne.core.beam as _beam
        if hasattr(_beam._thread_local, "conn") and _beam._thread_local.conn:
            try:
                _beam._thread_local.conn.close()
            except Exception:
                pass
            _beam._thread_local.conn = None
        init_beam(db)  # Must not raise

    def test_new_rows_after_init_use_bigram(self, tmp_path, monkeypatch):
        db = self._make_legacy(tmp_path)
        monkeypatch.setenv("MNEMOSYNE_DB_PATH", str(db))
        import mnemosyne.core.beam as _beam
        if hasattr(_beam._thread_local, "conn") and _beam._thread_local.conn:
            try:
                _beam._thread_local.conn.close()
            except Exception:
                pass
            _beam._thread_local.conn = None
        init_beam(db)
        conn = _open(db)
        _wm_insert(conn, "new1", "腾讯云新数据")
        hits = _fts_hit_working(conn, "腾讯云")
        assert any("腾讯" in h for h in hits), f"New row not found; hits={hits}"


# ---------------------------------------------------------------------------
# 4. Copy-DB: production-like schema (vec_working present, persona absent)
# ---------------------------------------------------------------------------

class TestCopyDb:
    def test_init_on_production_snapshot(self, db):
        conn = _open(db)
        # Add tables that exist in production but not in base schema
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS vec_working (
                rowid INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding_json TEXT NOT NULL,
                model TEXT
            );
            CREATE TABLE IF NOT EXISTS memory_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT,
                event_type TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)
        conn.commit()
        conn.close()
        # Second init_beam on this snapshot must not raise
        import mnemosyne.core.beam as _beam
        if hasattr(_beam._thread_local, "conn") and _beam._thread_local.conn:
            try:
                _beam._thread_local.conn.close()
            except Exception:
                pass
            _beam._thread_local.conn = None
        init_beam(db)

    def test_missing_persona_sync_meta_ok(self, db):
        conn = _open(db)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert "working_memory" in tables
        assert "episodic_memory" in tables
        # Production may lack these; they should be created by init_beam
        # but their absence before init must not be an error (handled above).


# ---------------------------------------------------------------------------
# 5. Thread: worker-factory connection has UDF
# ---------------------------------------------------------------------------

class TestThreadConnection:
    def test_worker_thread_has_cjk_bigram_udf(self, db):
        result: list = []
        errors: list = []

        def worker():
            try:
                c = sqlite3.connect(str(db), factory=_BeamConnection)
                row = c.execute("SELECT cjk_bigram('腾讯云')").fetchone()
                result.append(row[0])
                c.close()
            except Exception as exc:
                errors.append(str(exc))

        t = threading.Thread(target=worker)
        t.start()
        t.join(timeout=10)
        assert not errors, f"Worker thread error: {errors}"
        assert result and "腾讯" in result[0].split()


# ---------------------------------------------------------------------------
# 6. Idempotent init
# ---------------------------------------------------------------------------

class TestIdempotentInit:
    def test_double_init_does_not_raise(self, db):
        conn = _open(db)
        _wm_insert(conn, "w1", "腾讯云")
        conn.close()
        import mnemosyne.core.beam as _beam
        if hasattr(_beam._thread_local, "conn") and _beam._thread_local.conn:
            try:
                _beam._thread_local.conn.close()
            except Exception:
                pass
            _beam._thread_local.conn = None
        init_beam(db)  # Second call
        conn2 = _open(db)
        count = conn2.execute(
            "SELECT COUNT(*) FROM fts_working WHERE id='w1'"
        ).fetchone()[0]
        assert count == 1, f"Expected 1 FTS row after double init, got {count}"


class TestRegexExtractionToggle:
    """Verify MNEMOSYNE_DISABLE_REGEX_EXTRACTION opt-in toggle."""

    def test_disable_regex_extraction_toggle(self, tmp_path, monkeypatch):
        from mnemosyne.core.beam import BeamMemory

        db = tmp_path / "toggle.db"
        bm = BeamMemory(db_path=db)
        text = "I prefer you need a closer look at this configuration."

        monkeypatch.setenv("MNEMOSYNE_DISABLE_REGEX_EXTRACTION", "1")
        bm.extract_and_store_facts(text, message_idx=1)
        count_disabled = bm.conn.execute("SELECT COUNT(*) FROM memoria_preferences").fetchone()[0]
        assert count_disabled == 0

        monkeypatch.setenv("MNEMOSYNE_DISABLE_REGEX_EXTRACTION", "0")
        bm.extract_and_store_facts(text, message_idx=2)
        count_enabled = bm.conn.execute("SELECT COUNT(*) FROM memoria_preferences").fetchone()[0]
        assert count_enabled > 0
