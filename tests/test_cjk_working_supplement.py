"""Regression coverage for schema-free supplemental CJK working-memory recall."""

import sqlite3

import pytest

from mnemosyne.core import beam


QUERY = "鹽田參觀名額，夜間導覽時間"
TARGET = "鹽田的參觀名額為每日三十人，夜間的導覽時間是晚上七點。"


def test_supplement_preserves_ranked_fts_pool_and_recovers_split_cjk():
    """Append split CJK matches without changing the ranked FTS prefix or schema."""
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
    """Retain the original CJK fallback when FTS finds no rows."""
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
    """Reject candidates lacking evidence for a separate query part."""
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
    """A shared topic pair cannot satisfy an unanswered query part."""
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


@pytest.mark.parametrize(
    ("query", "literal", "target", "expected_relevance"),
    [
        (QUERY, "鹽田參觀名額", "鹽田的名額有限，夜間活動另行通知。", 0.5455),
        ("カメラ保管、レンズ清掃", "カメラ保管", "カメラの保管場所でレンズを清掃します。", 1.0),
        ("전시장안내, 주차장요금", "전시장안내", "전시장 안내는 입구에서 받고 주차장 요금은 출구에서 냅니다.", 1.0),
    ],
    ids=["han", "kana", "hangul"],
)
def test_recall_explain_counts_supplement_and_preserves_fts_score(
    tmp_path, monkeypatch, query, literal, target, expected_relevance,
):
    """Credit supplemental rows in explain while preserving existing FTS scores."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    memory = beam.BeamMemory(session_id="cjk", db_path=tmp_path / "cjk.db")
    with memory.conn:
        memory.conn.executemany(
            "INSERT INTO working_memory(id,content,source,timestamp,scope)"
            " VALUES(?,?,'preference','2026-09-15T10:00:00','global')",
            [("literal", literal + " 。"), ("gold", target)],
        )

    def recall():
        """Run the same public recall without updating usage counters."""
        return memory.recall(query, top_k=3, explain=True, _track_recall=False)

    with monkeypatch.context() as baseline:
        baseline.setattr(beam, "_supplement_cjk_working", lambda conn, query, rows, k, terms, **kwargs: rows)
        original = {row["id"]: row for row in recall()["results"]}

    payload = recall()
    recovered = {row["id"]: row for row in payload["results"]}
    assert set(original) == {"literal"}
    assert set(recovered) == {"literal", "gold"}
    # Each target matches both query parts lexically despite being outside MATCH.
    assert recovered["gold"]["keyword_score"] == recovered["gold"]["fts_score"] == expected_relevance
    for key in ("score", "keyword_score", "fts_score"):
        assert recovered["literal"][key] == original["literal"][key]
    stage = next(stage for stage in payload["explain"]["stages"] if stage["name"] == "wm_primary")
    assert stage["raw_count"] == stage["after_filter_count"] == stage["kept_count"] == 2


@pytest.fixture
def supplement_conn():
    """Provide a disposable SQLite store with real FTS5 for bounded-search controls."""
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(
            "CREATE TABLE working_memory(id TEXT PRIMARY KEY, content TEXT);"
            "CREATE VIRTUAL TABLE fts_working USING fts5(id UNINDEXED, content);"
        )
        yield conn


@pytest.mark.parametrize(("raw_hits", "supplements"), [(256, ["gold"]), (257, [])])
def test_supplement_raw_match_boundary(supplement_conn, raw_hits, supplements):
    """Supplement at 256 raw matches and decline the saturated 257-match pool."""
    conn = supplement_conn
    conn.executemany(
        "INSERT INTO fts_working VALUES(?, '鹽田參觀名額')",
        [(f"literal-{i:03}",) for i in range(raw_hits)],
    )
    conn.execute("INSERT INTO working_memory VALUES('gold', ?)", (TARGET,))
    terms = beam._fts_query_terms(QUERY)
    ranked = beam._fts_staged_rows(conn, "fts_working", "id", QUERY, 50, terms)
    assert len(ranked) == 50
    result = beam._fts_search_working(conn, QUERY, 50)
    assert result[:50] == ranked
    assert [row["id"] for row in result[50:]] == supplements


@pytest.mark.parametrize(("k", "expected"), [(3, 3), (50, 16)])
def test_supplement_candidate_cap(supplement_conn, k, expected):
    """Apply both the requested candidate limit and the sixteen-row supplement cap."""
    conn = supplement_conn
    conn.execute("INSERT INTO fts_working VALUES('literal', '鹽田參觀名額')")
    conn.executemany(
        "INSERT INTO working_memory VALUES(?, ?)",
        [(f"gold-{19 - i:02}", TARGET) for i in range(20)],
    )
    result = beam._fts_search_working(conn, QUERY, k)
    assert result[0]["id"] == "literal"
    assert [row["id"] for row in result[1:]] == [f"gold-{19 - i:02}" for i in range(expected)]


def test_supplement_bounded_rowid_windows(supplement_conn):
    """Include boundary rows from the head and tail while excluding the middle."""
    conn = supplement_conn
    conn.execute("INSERT INTO fts_working VALUES('literal', '鹽田參觀名額')")
    edge_rows = {1, 384, 385, 387, 388, 515}
    conn.executemany(
        "INSERT INTO working_memory(rowid,id,content) VALUES(?,?,?)",
        [(i, str(i), TARGET if i in edge_rows else "background") for i in range(1, 516)],
    )
    result = beam._fts_search_working(conn, QUERY, 50)
    assert [row["id"] for row in result] == ["literal", "1", "384", "388", "515"]


@pytest.mark.parametrize(
    ("query", "allowed"),
    [
        ("天地，玄黃，宇宙，洪荒", True),
        ("天地，玄黃，宇宙，洪荒，日月", False),
        ("天地玄黃宇宙洪荒日月盈昃辰宿列張寒來暑往秋收冬藏閏餘成歲律呂調陽雲", True),
        ("天地玄黃宇宙洪荒日月盈昃辰宿列張寒來暑往秋收冬藏閏餘成歲律呂調陽雲騰", False),
    ],
)
def test_supplement_query_limits(supplement_conn, query, allowed):
    """Bound query work by the permitted number of CJK runs and pairs."""
    conn = supplement_conn
    conn.execute("INSERT INTO working_memory VALUES('gold', ?)", (query,))
    # Supply a ranked hit directly to isolate the supplement's query guards.
    ranked = [{"id": "literal", "rank": -1.0}]
    terms = beam._fts_query_terms(query)
    result = beam._supplement_cjk_working(conn, query, ranked, 50, terms)
    assert result[:1] == ranked
    assert [row["id"] for row in result[1:]] == (["gold"] if allowed else [])


def test_supplement_sqlite_error_preserves_ranked_pool(supplement_conn):
    """Keep the original ranked candidates when supplemental SQL fails."""
    conn = supplement_conn
    conn.execute("INSERT INTO fts_working VALUES('literal', '鹽田參觀名額')")
    terms = beam._fts_query_terms(QUERY)
    ranked = beam._fts_staged_rows(conn, "fts_working", "id", QUERY, 50, terms)
    assert ranked
    conn.execute("DROP TABLE working_memory")
    assert beam._fts_search_working(conn, QUERY, 50) == ranked


@pytest.mark.parametrize(
    ("field", "blocked", "kwargs"),
    [
        ("session_id", "foreign", {}),
        ("valid_until", "2000-01-01T00:00:00Z", {}),
        ("superseded_by", "replacement", {}),
        ("source", "other", {"source": "preference"}),
        ("timestamp", "2000-01-01T00:00:00Z", {"from_date": "2026-01-01"}),
        ("author_id", "other", {"author_id": "requested"}),
        ("channel_id", "other", {"channel_id": "requested"}),
        ("veracity", "false", {"veracity": "confirmed"}),
        ("memory_type", "other", {"memory_type": "fact"}),
    ],
)
def test_filtered_supplements_do_not_consume_visible_slots(
    tmp_path, monkeypatch, field, blocked, kwargs,
):
    """Apply recall visibility filters before filling supplemental slots."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    memory = beam.BeamMemory(session_id="local", db_path=tmp_path / "visible.db")
    with memory.conn:
        memory.conn.executemany(
            "INSERT INTO working_memory(id,content,source,timestamp,scope,session_id)"
            " VALUES(?,?,'preference','2026-09-15T10:00:00',?,?)",
            [("literal", "鹽田參觀名額", "session", "local")]
            + [(f"foreign-{i}", TARGET, "session", "local") for i in range(20)]
            + [("gold", "鹽田的名額有限，夜間活動另行通知。", "session", "local")],
        )
        for column, value in (
            ("author_id", "requested"), ("channel_id", "requested"),
            ("veracity", "confirmed"), ("memory_type", "fact"),
        ):
            memory.conn.execute(f"UPDATE working_memory SET {column} = ?", (value,))
        memory.conn.execute(
            f"UPDATE working_memory SET {field} = ? WHERE id LIKE 'foreign-%'", (blocked,),
        )
    result = memory.recall(QUERY, top_k=3, _cross_session=False, _track_recall=False, **kwargs)
    assert {row["id"] for row in result} == {"literal", "gold"}


def test_supplement_rejects_insufficient_pair_evidence(supplement_conn):
    """Reject a candidate matching too few independent character pairs."""
    conn = supplement_conn
    for table in ("working_memory", "fts_working"):
        conn.executemany(
            f"INSERT INTO {table} VALUES(?,?)",
            [("literal", "天地玄黃"), ("weak", "天地")],
        )
    assert [row["id"] for row in beam._fts_search_working(conn, "天地玄黃", 50)] == ["literal"]


def test_supplement_rejects_positive_score_below_floor(supplement_conn, monkeypatch):
    """Enforce the relevance floor even when lexical similarity is positive."""
    monkeypatch.setenv("MNEMOSYNE_LEXICAL_GATE_MIN", "0.9")
    conn = supplement_conn
    conn.execute("INSERT INTO fts_working VALUES('literal', '鹽田參觀名額')")
    conn.execute("INSERT INTO working_memory VALUES('weak', '鹽田的名額有限，夜間活動另行通知。')")
    assert [row["id"] for row in beam._fts_search_working(conn, QUERY, 50)] == ["literal"]


def test_staged_short_pool_excludes_raw_hits(supplement_conn):
    """Exclude raw MATCH hits omitted from a shorter staged FTS pool."""
    conn = supplement_conn
    query = "전시장안내에서, 주차장요금은"
    target = "전시장 안내는 입구에서 받고 주차장 요금은 출구에서 냅니다."
    conn.execute("INSERT INTO fts_working VALUES('literal', '전시장안내')")
    conn.execute("INSERT INTO fts_working VALUES('outside', '전시장안내가 주차장요금은')")
    conn.execute("INSERT INTO working_memory VALUES('outside', '전시장안내가 주차장요금은')")
    conn.execute("INSERT INTO working_memory VALUES('gold', ?)", (target,))
    terms = beam._fts_query_terms(query)
    assert beam._fts_precise_terms(query) != terms
    ranked = [{"id": "literal", "rank": -1.0}]
    result = beam._supplement_cjk_working(conn, query, ranked, 50, terms)
    assert result == ranked + [{"id": "gold", "rank": None, "cjk_supplement": True}]


@pytest.mark.parametrize(
    ("query", "allowed"),
    [("天地玄，宇宙洪，日月盈，寒來暑", True), ("天地玄，宇宙洪，日月盈，寒來暑，秋收冬", False)],
)
def test_query_run_limit_through_fts(supplement_conn, query, allowed):
    """Exercise the run-count guard through the actual FTS entry point."""
    conn = supplement_conn
    literal = query.split("，")[0]
    target = query.replace("，", "的")
    conn.execute("INSERT INTO fts_working VALUES('literal', ?)", (literal,))
    conn.execute("INSERT INTO working_memory VALUES('gold', ?)", (target,))
    result = beam._fts_search_working(conn, query, 50)
    assert result[0]["id"] == "literal"
    assert [row["id"] for row in result[1:]] == (["gold"] if allowed else [])


def test_supplement_preserves_multiple_bm25_scores(tmp_path, monkeypatch):
    """Leave distinct existing BM25 scores unchanged when appending a candidate."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    memory = beam.BeamMemory(session_id="cjk", db_path=tmp_path / "ranks.db")
    with memory.conn:
        memory.conn.executemany(
            "INSERT INTO working_memory(id,content,source,timestamp,scope)"
            " VALUES(?,?,'preference','2026-09-15T10:00:00','global')",
            [
                ("short", "鹽田參觀名額"),
                ("long", "鹽田參觀名額 " + "background " * 30),
                ("gold", "鹽田的名額有限，夜間活動另行通知。"),
            ],
        )
    ranked = beam._fts_staged_rows(
        memory.conn, "fts_working", "id", QUERY, 50, beam._fts_query_terms(QUERY),
    )
    assert len({row["rank"] for row in ranked}) == 2
    with monkeypatch.context() as baseline:
        baseline.setattr(beam, "_supplement_cjk_working", lambda conn, query, rows, k, terms, **kw: rows)
        original = {row["id"]: row for row in memory.recall(QUERY, top_k=5, _track_recall=False)}
    result = {row["id"]: row for row in memory.recall(QUERY, top_k=5, _track_recall=False)}
    assert set(result) == set(original) | {"gold"}
    assert result["gold"]["keyword_score"] == result["gold"]["fts_score"] == 0.5455
    for mid in original:
        assert result[mid]["fts_score"] == original[mid]["fts_score"]


@pytest.mark.parametrize("target_position", [1, 599], ids=["head", "tail"])
def test_filtered_cjk_recall_at_31_variable_limit(tmp_path, monkeypatch, target_position):
    """Preserve filtered recall in both row windows under a low SQLite bind limit."""
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    memory = beam.BeamMemory(session_id="local", db_path=tmp_path / "limits.db")
    if not hasattr(memory.conn, "setlimit"):
        pytest.skip("runtime SQLite limits require Python 3.11+")
    rows = [("literal", "鹽田參觀名額")]
    rows.extend((f"background-{i}", "background") for i in range(598))
    rows.insert(target_position, ("gold", TARGET))
    with memory.conn:
        memory.conn.executemany(
            "INSERT INTO working_memory"
            "(id,content,source,timestamp,scope,session_id,author_id,author_type,"
            "channel_id,veracity,memory_type)"
            " VALUES(?,?,'preference','2026-09-15T10:00:00','session','local',"
            "'alice','human','channel','confirmed','fact')",
            rows,
        )
    kwargs = dict(
        top_k=3, _cross_session=False, _track_recall=False, source="preference",
        author_id="alice", author_type="human", channel_id="channel",
        veracity="confirmed", memory_type="fact", from_date="2026-01-01", to_date="2026-12-31",
    )
    expected = memory.recall(QUERY, **kwargs)
    assert {row["id"] for row in expected} == {"literal", "gold"}
    memory.conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 31)
    # Flush prepared statements so the lowered limit applies to every query.
    for i in range(160):
        memory.conn.execute(f"SELECT {i}").fetchall()
    assert memory.recall(QUERY, **kwargs) == expected


@pytest.mark.parametrize("k", [1, 50])
def test_supplement_orders_by_score_before_rowid_and_cap(supplement_conn, k):
    """Prefer stronger evidence before rowid tie-breaking and candidate truncation."""
    conn = supplement_conn
    conn.execute("INSERT INTO fts_working VALUES('literal', '鹽田參觀名額')")
    conn.executemany(
        "INSERT INTO working_memory VALUES(?, ?)",
        [(f"weak-{i:02}", "鹽田的名額有限，夜間活動另行通知。") for i in range(20)]
        + [("gold", TARGET)],
    )
    result = beam._fts_search_working(conn, QUERY, k)
    assert result[0]["id"] == "literal"
    assert result[1]["id"] == "gold"
    assert len(result[1:]) == min(k, 16)
    assert [row["id"] for row in result[2:]] == [f"weak-{i:02}" for i in range(min(k, 16) - 1)]
