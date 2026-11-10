"""Regression coverage for schema-free supplemental CJK working-memory recall."""

import sqlite3

from mnemosyne.core import beam


QUERY = "鹽田參觀名額，夜間導覽時間"
TARGET = "鹽田的參觀名額為每日三十人，夜間的導覽時間是晚上七點。"


def test_supplement_preserves_ranked_fts_pool_and_recovers_split_cjk():
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            "CREATE TABLE working_memory(id TEXT PRIMARY KEY, content TEXT);"
            "CREATE VIRTUAL TABLE fts_working USING fts5(id UNINDEXED, content);"
        )
        fts = [(f"layout-{i:03}", "鹽田參觀名額 。見出しの例。") for i in range(64)]
        # Keep a duplicate FTS row outside the original ranked pool. It is
        # still an FTS hit and cannot enter the CJK-only lane.
        conn.executemany("INSERT INTO fts_working VALUES(?,?)", fts + [fts[-1], ("gold", TARGET)])
        conn.executemany("INSERT INTO working_memory VALUES(?,?)", fts + [("gold", TARGET)])
        schema = list(conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name"))
        terms = beam._fts_query_terms(QUERY)
        original = beam._fts_staged_rows(conn, "fts_working", "id", QUERY, 50, terms)
        got = beam._fts_search_working(conn, QUERY, 50)
        assert got[:50] == original
        assert got[50:] == [{"id": "gold", "rank": None, "cjk_supplement": True}]
        assert len({row["id"] for row in got}) == len(got)
        assert list(conn.execute("SELECT name, sql FROM sqlite_master ORDER BY name")) == schema


def test_zero_match_uses_original_fallback():
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            "CREATE TABLE working_memory(id TEXT PRIMARY KEY, content TEXT);"
            "CREATE VIRTUAL TABLE fts_working USING fts5(id UNINDEXED, content);"
        )
        conn.execute("INSERT INTO working_memory VALUES('gold', ?)", (TARGET,))
        conn.execute("INSERT INTO fts_working VALUES('gold', ?)", (TARGET,))
        expected = beam._cjk_like_search(conn, QUERY, 50, working=True)
        assert expected and beam._fts_search_working(conn, QUERY, 50) == expected


def test_addition_needs_evidence_for_each_cjk_run():
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            "CREATE TABLE working_memory(id TEXT PRIMARY KEY, content TEXT);"
            "CREATE VIRTUAL TABLE fts_working USING fts5(id UNINDEXED, content);"
        )
        rows = [("layout", "鹽田參觀名額 。見出しの例。"), ("gold", TARGET)]
        conn.executemany("INSERT INTO fts_working VALUES(?,?)", rows)
        conn.executemany("INSERT INTO working_memory VALUES(?,?)", rows)
        query = "鹽田參觀名額，火山灰化驗結果"
        terms = beam._fts_query_terms(query)
        original = beam._fts_staged_rows(conn, "fts_working", "id", query, 50, terms)
        assert original and beam._fts_search_working(conn, query, 50) == original


def test_shared_pair_does_not_count_for_an_unanswered_query_part():
    query = "屋上花壇の散水、屋上改修の工費"
    rows = [
        ("literal", "屋上花壇の散水 。見出し例。"),
        ("gardening", "屋上の花壇に散水する順番は、入口の壁に掛けた当番表で確認します。"),
    ]
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            "CREATE TABLE working_memory(id TEXT PRIMARY KEY, content TEXT);"
            "CREATE VIRTUAL TABLE fts_working USING fts5(id UNINDEXED, content);"
        )
        conn.executemany("INSERT INTO working_memory VALUES(?,?)", rows)
        conn.executemany("INSERT INTO fts_working VALUES(?,?)", rows)
        terms = beam._fts_query_terms(query)
        ranked = beam._fts_staged_rows(conn, "fts_working", "id", query, 50, terms)
        assert [row["id"] for row in ranked] == ["literal"]
        assert beam._fts_search_working(conn, query, 50) == ranked


def test_recall_explain_counts_supplement_and_preserves_fts_score(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    memory = beam.BeamMemory(session_id="cjk", db_path=tmp_path / "cjk.db")
    with memory.conn:
        memory.conn.executemany(
            "INSERT INTO working_memory(id,content,source,timestamp,scope)"
            " VALUES(?,?,'preference','2026-09-15T10:00:00','global')",
            [("literal", "鹽田參觀名額 。見出しの例。"), ("gold", TARGET)],
        )

    def recall():
        return memory.recall(QUERY, top_k=3, explain=True, _track_recall=False)

    with monkeypatch.context() as baseline:
        baseline.setattr(beam, "_supplement_cjk_working", lambda conn, query, rows, k, terms: rows)
        original = {row["id"]: row for row in recall()["results"]}

    payload = recall()
    recovered = {row["id"]: row for row in payload["results"]}
    assert set(original) == {"literal"}
    assert set(recovered) == {"literal", "gold"}
    for key in ("score", "keyword_score", "fts_score"):
        assert recovered["literal"][key] == original["literal"][key]
    stage = next(stage for stage in payload["explain"]["stages"] if stage["name"] == "wm_primary")
    assert stage["raw_count"] == stage["after_filter_count"] == stage["kept_count"] == 2
