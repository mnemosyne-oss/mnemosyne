"""Polyphonic recall must read the store populated by episodic vector writes."""
import json
import sqlite3

import numpy as np
import pytest

import mnemosyne.core.beam as bm
from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine, RecallResult

pytest.importorskip("sqlite_vec")


@pytest.fixture(params=["float32", "int8", "bit"])
def store(request, tmp_path, monkeypatch):
    import sqlite_vec

    beam = bm.BeamMemory(session_id="vector-session", db_path=tmp_path / "memory.db")
    conn = beam.conn
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("DROP TABLE IF EXISTS vec_episodes")
    kind = request.param
    ddl = "float" if kind == "float32" else kind
    conn.execute(f"CREATE VIRTUAL TABLE vec_episodes USING vec0(embedding {ddl}[64])")
    conn.execute("PRAGMA user_version = 0")
    conn.commit()
    query = np.array([1.0, -1.0] * 32, dtype=np.float32)
    query /= np.linalg.norm(query)
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(bm._embeddings, "embed_query", lambda text: query)
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: [query for _ in texts])
    monkeypatch.delenv("MNEMOSYNE_VOICE_VECTOR", raising=False)
    return beam, query, kind


def _write(beam, monkeypatch, vector, label, source="selected"):
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: [vector for _ in texts])
    mid = beam.consolidate_to_episodic(label, [], source=source)
    rowid = beam.conn.execute("SELECT rowid FROM episodic_memory WHERE id=?", (mid,)).fetchone()[0]
    assert beam.conn.execute("SELECT embedding FROM vec_episodes WHERE rowid=?", (rowid,)).fetchone()
    assert beam.conn.execute("SELECT 1 FROM memory_embeddings WHERE memory_id=?", (mid,)).fetchone() is None
    return mid, rowid


def _stored_cosine(beam, query, rowid, kind):
    blob = beam.conn.execute("SELECT embedding FROM vec_episodes WHERE rowid=?", (rowid,)).fetchone()[0]
    if kind == "bit":
        qblob = beam.conn.execute("SELECT vec_quantize_binary(?)", (json.dumps(query.tolist()),)).fetchone()[0]
        differences = np.unpackbits(np.bitwise_xor(np.frombuffer(qblob, dtype=np.uint8), np.frombuffer(blob, dtype=np.uint8))).sum()
        return float(np.cos(np.pi * differences / query.size))
    if kind == "int8":
        qblob = beam.conn.execute("SELECT vec_quantize_int8(?, 'unit')", (json.dumps(query.tolist()),)).fetchone()[0]
        q = np.frombuffer(qblob, dtype=np.int8).astype(np.float64)
        row = np.frombuffer(blob, dtype=np.int8).astype(np.float64)
    else:
        q = query.astype(np.float64)
        row = np.frombuffer(blob, dtype=np.float32).astype(np.float64)
    return float(np.dot(q, row) / (np.linalg.norm(q) * np.linalg.norm(row)))


def _voice(beam, query, **kwargs):
    return PolyphonicRecallEngine(db_path=beam.db_path, conn=beam.conn)._vector_voice(query, **kwargs)


def test_normal_episodic_write_without_json_recalled(store, monkeypatch):
    beam, query, kind = store
    mid, _ = _write(beam, monkeypatch, query, "persisted directional record")
    assert bm._classify_vec_store_regime(beam.conn) == "legacy"
    statements = []
    beam.conn.set_trace_callback(statements.append)
    results = _voice(beam, query)
    hit = next((r for r in results if r.memory_id == mid), None)
    assert hit is not None, "normal episodic writes must not disappear from the vector voice"
    assert hit.score == pytest.approx(1.0)
    assert hit.metadata["backend"] == "sqlite-vec"
    assert hit.metadata["vec_type"] == kind
    assert not any("MATCH" in sql and "vec_episodes" in sql for sql in statements)
    assert not any(sql.lstrip().upper().startswith(("INSERT", "UPDATE", "DELETE")) for sql in statements)
    beam.conn.set_trace_callback(None)
    for voice in ("TEMPORAL", "GRAPH", "FACT"):
        monkeypatch.setenv(f"MNEMOSYNE_VOICE_{voice}", "0")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: [query for _ in texts])
    public = beam.recall("unrelated opaque query", top_k=5)
    hit = next((r for r in public if r.get("id") == mid), None)
    assert hit is not None
    assert hit["voice_scores"]["vector"] > 0
    assert not any(hit["voice_scores"].get(voice) for voice in ("temporal", "graph", "fact"))


@pytest.mark.parametrize("filter_arg", ["source", "topic"])
def test_legacy_scan_filters_before_bounded_selection(store, monkeypatch, filter_arg):
    beam, query, kind = store
    # More perfect but ineligible hits than the normalized path's KNN budget.
    for i in range(65):
        mid, _ = _write(beam, monkeypatch, query, f"ineligible record {i}", source="other")
        if i % 3 == 0:
            beam.conn.execute("UPDATE episodic_memory SET source='selected', superseded_by='replacement' WHERE id=?", (mid,))
        elif i % 3 == 1:
            beam.conn.execute("UPDATE episodic_memory SET source='selected', valid_until='2000-01-01' WHERE id=?", (mid,))
        beam.conn.commit()
    target = query.copy()
    target[:4] *= -1
    mid, rowid = _write(beam, monkeypatch, target, "eligible survivor")
    results = _voice(beam, query, **{filter_arg: "selected"})
    assert [r.memory_id for r in results] == [mid]
    expected = _stored_cosine(beam, query, rowid, kind)
    assert results[0].metadata["cosine_similarity"] == pytest.approx(expected, abs=1e-6)
    assert results[0].score == pytest.approx((expected + 1) / 2, abs=1e-6)


@pytest.mark.parametrize("store", ["float32", "int8"], indirect=True)
def test_legacy_low_norm_target_beyond_knn_and_admission(store, monkeypatch):
    beam, query, kind = store
    # This fixture is calibrated against a 0.80 floor: the distractors sit at
    # cosine 0.75 (8 of 64 signs flipped) and are deliberately nearer in L2
    # than the 0.1-magnitude target, which is what pushes the target outside
    # the KNN budget. At the shipped 0.62 default those distractors would be
    # admitted instead and the target would no longer be the only survivor,
    # so pin the floor this geometry was built for.
    monkeypatch.setattr(bm, "EM_VEC_ADMIT", 0.80)
    for i in range(75):
        distractor = query.copy()
        distractor[:8] *= -1  # cosine .75: below the pinned .80 admission boundary
        _write(beam, monkeypatch, distractor, f"directional distractor {i}")
    mid, rowid = _write(beam, monkeypatch, query, "small magnitude legacy target")
    # Ordinary writes normalize. Replace only this blob to model a pre-normalization row.
    low = query * 0.1
    value = json.dumps(low.tolist())
    expression = "vec_quantize_int8(?, 'unit')" if kind == "int8" else "?"
    beam.conn.execute("DELETE FROM vec_episodes WHERE rowid=?", (rowid,))
    beam.conn.execute(f"INSERT INTO vec_episodes(rowid, embedding) VALUES (?, {expression})", (rowid, value))
    beam.conn.commit()
    for i in range(75):
        _write(beam, monkeypatch, distractor, f"later directional distractor {i}")
    raw = beam.conn.execute("SELECT embedding FROM vec_episodes WHERE rowid=?", (rowid,)).fetchone()[0]
    dtype = np.int8 if kind == "int8" else np.float32
    assert 0 < np.linalg.norm(np.frombuffer(raw, dtype=dtype)) < (80 if kind == "int8" else 0.2)
    knn = beam.conn.execute(f"SELECT rowid FROM vec_episodes WHERE embedding MATCH {expression} AND k=60 ORDER BY distance", (json.dumps(query.tolist()),)).fetchall()
    assert rowid not in {r[0] for r in knn}
    assert beam.conn.execute("SELECT count(*) FROM memory_embeddings").fetchone()[0] == 0
    results = _voice(beam, query)
    assert [r.memory_id for r in results] == [mid]
    expected = _stored_cosine(beam, query, rowid, kind)
    assert results[0].score == pytest.approx((expected + 1) / 2, abs=1e-6)
    # Public routing/fusion/hydration, without another voice rescuing the target.
    for voice in ("TEMPORAL", "GRAPH", "FACT"):
        monkeypatch.setenv(f"MNEMOSYNE_VOICE_{voice}", "0")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: [query for _ in texts])
    public = beam.recall("unrelated opaque query", top_k=5)
    hit = next((r for r in public if r.get("id") == mid), None)
    assert hit is not None
    assert hit["voice_scores"]["vector"] > 0
    assert not any(hit["voice_scores"].get(voice) for voice in ("temporal", "graph", "fact"))


def test_linear_legacy_scan_filters_scope_before_bounded_selection(
        store, monkeypatch):
    beam, query, kind = store
    # More perfect but ineligible rows than linear recall's candidate budget.
    for i in range(25):
        mid, _ = _write(
            beam, monkeypatch, query, f"other-channel record {i}", source="selected"
        )
        beam.conn.execute(
            "UPDATE episodic_memory SET scope='session', session_id='other-session', "
            "channel_id='other-channel' WHERE id=?",
            (mid,),
        )
    target = query.copy()
    target[:4] *= -1
    mid, rowid = _write(beam, monkeypatch, target, "eligible channel survivor")
    beam.conn.execute(
        "UPDATE episodic_memory SET scope='session', channel_id='wanted-channel' "
        "WHERE id=?",
        (mid,),
    )
    beam.conn.commit()
    assert bm._classify_vec_store_regime(beam.conn) == "legacy"
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setattr(bm._embeddings, "embed_query", lambda text: query)

    results = beam.recall(
        "opaque query with no lexical match", top_k=1, channel_id="wanted-channel"
    )
    hit = next((r for r in results if r.get("id") == mid), None)
    assert hit is not None, "ineligible perfect rows starved the eligible legacy row"
    expected = _stored_cosine(beam, query, rowid, kind)
    assert hit["dense_score"] == pytest.approx(expected, abs=1e-4)


def test_pure_polyphonic_uses_shared_blob_admission(store, monkeypatch):
    beam, query, kind = store
    # 24 of 64 flipped signs stays below admission in every representation:
    # cosine 0.25 for float32/int8, and cos(pi*24/64) = 0.383 for the bit arm,
    # whose Hamming distance maps onto the same monotone 0..1 scale. The
    # count is chosen to hold at any floor the constant can reasonably take,
    # rather than tracking one default.
    below = query.copy()
    below[:24] *= -1
    rejected, rejected_rowid = _write(
        beam, monkeypatch, below, "below-admission pure record"
    )
    middle = query.copy()
    middle[:4] *= -1
    scored, scored_rowid = _write(
        beam, monkeypatch, middle, "nontrivial admitted pure record"
    )
    accepted, accepted_rowid = _write(
        beam, monkeypatch, query, "accepted pure record"
    )
    bm._mark_vec_store_norm_bit(beam.conn)
    assert bm._classify_vec_store_regime(beam.conn) == "pure"
    assert _stored_cosine(beam, query, rejected_rowid, kind) < bm.EM_VEC_ADMIT
    scored_expected = _stored_cosine(beam, query, scored_rowid, kind)
    assert scored_expected >= bm.EM_VEC_ADMIT
    assert _stored_cosine(beam, query, accepted_rowid, kind) >= bm.EM_VEC_ADMIT

    statements = []
    beam.conn.set_trace_callback(statements.append)
    results = _voice(beam, query)
    beam.conn.set_trace_callback(None)
    assert any("vec_episodes" in sql and " MATCH " in sql for sql in statements)
    assert [r.memory_id for r in results] == [accepted, scored]
    assert rejected not in {r.memory_id for r in results}
    hit = results[0]
    expected = _stored_cosine(beam, query, accepted_rowid, kind)
    assert hit.metadata["cosine_similarity"] == pytest.approx(expected, abs=1e-6)
    assert hit.score == pytest.approx((expected + 1) / 2, abs=1e-6)
    scored_hit = results[1]
    assert scored_hit.metadata["cosine_similarity"] == pytest.approx(
        scored_expected, abs=1e-6
    )
    assert scored_hit.score == pytest.approx((scored_expected + 1) / 2, abs=1e-6)


def test_legacy_vec_authority_without_json_only_fusion(store, monkeypatch):
    beam, query, kind = store
    target = query.copy()
    target[:4] *= -1
    mid, rowid = _write(beam, monkeypatch, target, "dual representation record")
    # 24 of 64 flipped signs stays below admission in every representation:
    # cosine 0.25 for float32/int8, and cos(pi*24/64) = 0.383 for the bit arm,
    # whose Hamming distance maps onto the same monotone 0..1 scale. The
    # count is chosen to hold at any floor the constant can reasonably take,
    # rather than tracking one default.
    below = query.copy()
    below[:24] *= -1
    rejected, _ = _write(beam, monkeypatch, below, "below admission record")
    json_mid, json_rowid = _write(beam, monkeypatch, query, "JSON only record")
    beam.conn.execute("DELETE FROM vec_episodes WHERE rowid=?", (json_rowid,))
    for key in (mid, rejected, json_mid):
        beam.conn.execute("INSERT INTO memory_embeddings(memory_id, embedding_json) VALUES (?,?)", (key, json.dumps(query.tolist())))
    beam.conn.commit()
    results = _voice(beam, query)
    ids = [r.memory_id for r in results]
    assert ids == [mid]
    assert rejected not in ids
    assert json_mid not in ids
    hit = results[0]
    expected = _stored_cosine(beam, query, rowid, kind)
    assert hit.score == pytest.approx((expected + 1) / 2, abs=1e-6)
    assert hit.metadata["backend"] == "sqlite-vec"


@pytest.mark.parametrize("regime", ["unknown", "legacy-sign", "read-error"])
def test_uncertain_marker_still_reads_vec_only_rows(store, monkeypatch, regime):
    beam, query, _ = store
    mid, _ = _write(beam, monkeypatch, query, "uncertain marker record")

    def classify(*args, **kwargs):
        if regime == "read-error":
            raise RuntimeError("marker unavailable")
        return regime

    monkeypatch.setattr(bm, "_classify_vec_store_regime", classify)
    assert [r.memory_id for r in _voice(beam, query)] == [mid]


def test_legacy_scan_failure_keeps_json_fallback(store, monkeypatch):
    import sqlite3

    beam, query, _ = store
    mid, _ = _write(beam, monkeypatch, query, "fallback availability record")
    beam.conn.execute("INSERT INTO memory_embeddings(memory_id, embedding_json) VALUES (?,?)", (mid, json.dumps(query.tolist())))
    beam.conn.commit()

    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("unavailable scan")

    monkeypatch.setattr(PolyphonicRecallEngine, "_legacy_episodic_vector_voice", fail)
    results = _voice(beam, query)
    assert [r.memory_id for r in results] == [mid]
    assert results[0].metadata["backend"] == "memory_embeddings"
    assert results[0].score == pytest.approx(1.0)


def test_polyphonic_legacy_filters_scope_before_voice_boundary(store, monkeypatch):
    beam, query, _ = store
    foreign_ids = []
    for index in range(25):
        mid, _ = _write(
            beam, monkeypatch, query, f"foreign legacy candidate {index}"
        )
        beam.conn.execute(
            "UPDATE episodic_memory SET scope='session', session_id='foreign' "
            "WHERE id=?",
            (mid,),
        )
        foreign_ids.append(mid)
    target_vec = query.copy()
    target_vec[:2] *= -1
    target, _ = _write(
        beam, monkeypatch, target_vec, "eligible legacy vector candidate"
    )
    beam.conn.execute(
        "UPDATE episodic_memory SET scope='session', session_id=? WHERE id=?",
        (beam.session_id, target),
    )
    beam.conn.execute("PRAGMA user_version = 0")
    beam.conn.commit()

    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_VOICE_TEMPORAL", "0")
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "0")
    monkeypatch.setenv("MNEMOSYNE_VOICE_FACT", "0")
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: [query for _ in texts])
    results = beam.recall("opaque-no-lexical-rescue", top_k=5)
    hit = next((row for row in results if row.get("id") == target), None)
    assert hit is not None
    assert hit["voice_scores"]["vector"] > 0
    assert not ({row.get("id") for row in results} & set(foreign_ids))


def test_pure_polyphonic_refills_after_source_filtered_knn(store, monkeypatch):
    beam, query, _ = store
    for index in range(65):
        _write(
            beam,
            monkeypatch,
            query,
            f"foreign pure candidate {index}",
            source="foreign",
        )
    target_vec = query.copy()
    target_vec[:2] *= -1
    target, _ = _write(
        beam,
        monkeypatch,
        target_vec,
        "eligible pure source candidate",
        source="wanted",
    )
    bm._mark_vec_store_norm_bit(beam.conn)
    assert bm._classify_vec_store_regime(beam.conn) == "pure"

    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_VOICE_TEMPORAL", "0")
    monkeypatch.setenv("MNEMOSYNE_VOICE_GRAPH", "0")
    monkeypatch.setenv("MNEMOSYNE_VOICE_FACT", "0")
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: [query for _ in texts])
    results = beam.recall(
        "opaque-no-lexical-rescue", top_k=5, source="wanted"
    )
    hit = next((row for row in results if row.get("id") == target), None)
    assert hit is not None
    assert hit["voice_scores"]["vector"] > 0
    assert all(row.get("source") == "wanted" for row in results)


def test_pure_polyphonic_uses_exact_eligible_scan_at_knn_ceiling(
    store, monkeypatch
):
    beam, query, _ = store
    target, _ = _write(
        beam, monkeypatch, query, "eligible ceiling survivor", source="wanted"
    )
    bm._mark_vec_store_norm_bit(beam.conn)
    knn_limits = []

    def full_ineligible_knn(conn, embedding, k):
        knn_limits.append(k)
        return [
            {"rowid": 1_000_000 + index, "distance": 0.0, "blob": None}
            for index in range(k)
        ], None

    exact_calls = []
    engine = PolyphonicRecallEngine(db_path=beam.db_path, conn=beam.conn)

    def exact_eligible_scan(*args, **kwargs):
        exact_calls.append(kwargs)
        return [
            RecallResult(
                memory_id=target,
                score=0.95,
                voice="vector",
                metadata={"embedding_tier": "episodic"},
            )
        ]

    monkeypatch.setattr(bm, "_vec_search_with_blobs", full_ineligible_knn)
    monkeypatch.setattr(
        engine, "_legacy_episodic_vector_voice", exact_eligible_scan
    )
    results = engine._vector_voice(
        query, source="wanted", default_dense_source_filter=False
    )
    assert [result.memory_id for result in results] == [target]
    assert knn_limits[-1] == 4096
    assert len(exact_calls) == 1
    assert "source = ?" in exact_calls[0]["episodic_where"]
    assert "wanted" in exact_calls[0]["episodic_params"]


@pytest.mark.parametrize(
    ("filter_arg", "column", "value"),
    [
        ("author_id", "author_id", "author-7"),
        ("author_type", "author_type", "human"),
        ("channel_id", "channel_id", "channel-7"),
    ],
)
@pytest.mark.parametrize("regime", ["legacy", "pure"])
def test_polyphonic_post_filter_preserves_cross_session_filter_scope(
        store, monkeypatch, filter_arg, column, value, regime):
    beam, query, _ = store
    target, _ = _write(beam, monkeypatch, query, f"cross-session {filter_arg}")
    beam.conn.execute(
        f"UPDATE episodic_memory SET scope='session', session_id='foreign', "
        f"{column}=? WHERE id=?",
        (value, target),
    )
    if regime == "pure":
        bm._mark_vec_store_norm_bit(beam.conn)
    beam.conn.commit()

    for voice in ("TEMPORAL", "GRAPH", "FACT"):
        monkeypatch.setenv(f"MNEMOSYNE_VOICE_{voice}", "0")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.setattr(bm._embeddings, "embed", lambda texts: [query for _ in texts])
    results = beam.recall(
        "opaque-no-lexical-rescue", top_k=5, **{filter_arg: value}
    )

    hit = next((row for row in results if row.get("id") == target), None)
    assert hit is not None
    assert hit["voice_scores"]["vector"] > 0


def test_pure_polyphonic_stops_after_twenty_eligible_below_admission(
        store, monkeypatch):
    beam, query, kind = store
    # 24 of 64 flipped signs stays below admission in every representation:
    # cosine 0.25 for float32/int8, and cos(pi*24/64) = 0.383 for the bit arm,
    # whose Hamming distance maps onto the same monotone 0..1 scale. The
    # count is chosen to hold at any floor the constant can reasonably take,
    # rather than tracking one default.
    below = query.copy()
    below[:24] *= -1
    stored = []
    for index in range(20):
        _, rowid = _write(
            beam, monkeypatch, below, f"eligible below-admission record {index}"
        )
        blob = beam.conn.execute(
            "SELECT embedding FROM vec_episodes WHERE rowid=?", (rowid,)
        ).fetchone()[0]
        stored.append((rowid, blob))
    if kind == "int8":
        query_blob = beam.conn.execute(
            "SELECT vec_quantize_int8(?, 'unit')", (json.dumps(query.tolist()),)
        ).fetchone()[0]
        query_values = np.frombuffer(query_blob, dtype=np.int8).astype(np.float64)
        row_values = np.frombuffer(stored[0][1], dtype=np.int8).astype(np.float64)
        boundary_distance = float(np.linalg.norm(query_values - row_values))
    elif kind == "bit":
        query_blob = beam.conn.execute(
            "SELECT vec_quantize_binary(?)", (json.dumps(query.tolist()),)
        ).fetchone()[0]
        boundary_distance = float(np.unpackbits(np.bitwise_xor(
            np.frombuffer(query_blob, dtype=np.uint8),
            np.frombuffer(stored[0][1], dtype=np.uint8),
        )).sum())
    else:
        query_blob = None
        row_values = np.frombuffer(stored[0][1], dtype=np.float32)
        boundary_distance = float(np.linalg.norm(query - row_values))
    candidates = [
        {"rowid": rowid, "distance": boundary_distance, "blob": blob}
        for rowid, blob in stored
    ]
    for index in range(40):
        candidates.append({
            "rowid": 1_000_000 + index,
            "distance": boundary_distance,
            "blob": None,
        })
    bm._mark_vec_store_norm_bit(beam.conn)
    knn_limits = []

    def eligible_knn(conn, embedding, k):
        knn_limits.append(k)
        return candidates, query_blob

    monkeypatch.setattr(bm, "_vec_search_with_blobs", eligible_knn)
    results = _voice(beam, query)

    assert results == []
    assert knn_limits == [60]


@pytest.mark.parametrize("store", ["int8"], indirect=True)
def test_pure_int8_refills_when_l2_boundary_cannot_prove_admission(
        store, monkeypatch):
    beam, _, _ = store
    # The literal vectors below encode a 0.80 boundary: the decoy measures
    # 0.7999 cosine against the query. They cannot be re-derived for a lower
    # floor, because the fixture's whole point is a decoy that is *nearer in
    # L2* than the target while scoring *lower* on cosine, and on a 64d
    # sign-flip fixture no granularity satisfies both below 0.62. Pin the
    # floor the vectors were tuned to.
    monkeypatch.setattr(bm, "EM_VEC_ADMIT", 0.80)
    query = np.ones(64, dtype=np.float32)
    query /= np.linalg.norm(query)
    below = np.array([
        0.0748693827, 0.1998435841, 0.0303676347, 0.1081579615,
        0.0729309524, 0.0863162611, 0.0864339336, 0.0382564627,
        0.0449808186, 0.0591658035, 0.2963587631, 0.1963280561,
        0.1657702801, 0.0812071853, 0.0896175538, 0.0766627460,
        0.0961151586, 0.1656131349, 0.1418973785, 0.0464476387,
        0.1448727752, 0.0945564747, 0.1089152551, 0.0642133798,
        0.1611412244, 0.2255107033, -0.0818155098, 0.1616931379,
        0.1725217321, 0.0662236029, 0.0172707407, 0.0352113762,
        0.1370367122, 0.0681096610, 0.1443356322, -0.0341960208,
        0.0807283771, 0.1347416447, 0.1521449973, 0.0710631363,
        0.0693365889, 0.0242280123, 0.0886170169, -0.0314712215,
        0.0224094258, 0.1188225433, 0.1675492088, 0.0622913949,
        0.2970065600, 0.1712158385, 0.1823561829, 0.0801119426,
        0.1602980592, 0.1590544020, 0.1285176709, 0.1524441124,
        0.1040303651, 0.1447252961, 0.1137004586, 0.0906786512,
        -0.0072694569, 0.1642734180, -0.0474488030, 0.0277973044,
    ], dtype=np.float32)
    above = np.array([
        0.1137352567, 0.0860509492, 0.0734865907, 0.0763267207,
        0.2244558994, 0.1403199904, 0.1393966953, 0.1394609800,
        0.1809900510, 0.1700873651, 0.0622119024, 0.1987253012,
        0.1403657861, 0.0536367727, 0.0519703951, 0.0443799643,
        0.0303934981, 0.1276043169, 0.0416006011, 0.0939725678,
        0.1617753998, 0.1471825293, 0.0500132247, -0.0061378102,
        0.1474620362, -0.0174417407, 0.0785364274, 0.0911392358,
        0.1721282737, -0.0874936694, 0.0403134959, -0.0794073242,
        0.1771377650, 0.0714552515, 0.0826892300, 0.1640696232,
        0.0388992197, 0.1350406960, 0.1567464983, 0.1796580806,
        0.1191425478, 0.1587461859, 0.1018961386, 0.2863708233,
        0.0175756843, 0.1388724947, 0.1189117505, 0.0011560168,
        0.0748748454, 0.0958382387, 0.0546492221, 0.1976421969,
        0.0849608110, 0.1457623401, 0.0984357943, 0.2648693868,
        0.1203139910, 0.0427435142, -0.0285329813, 0.0661529994,
        0.1957902840, 0.0427114772, 0.1286199734, 0.1077141864,
    ], dtype=np.float32)
    below /= np.linalg.norm(below)
    above /= np.linalg.norm(above)
    decoy_rowids = []
    for index in range(60):
        _, rowid = _write(
            beam, monkeypatch, below, f"below-admission closer L2 row {index}"
        )
        decoy_rowids.append(rowid)
    target, target_rowid = _write(
        beam, monkeypatch, above, "admitted farther L2 row"
    )
    bm._mark_vec_store_norm_bit(beam.conn)

    assert _stored_cosine(beam, query, decoy_rowids[0], "int8") < bm.EM_VEC_ADMIT
    assert _stored_cosine(beam, query, target_rowid, "int8") >= bm.EM_VEC_ADMIT
    knn = bm._vec_search_with_blobs(beam.conn, query.tolist(), k=60)[0]
    assert target_rowid not in {row["rowid"] for row in knn}

    results = _voice(beam, query)
    assert [result.memory_id for result in results] == [target]


def test_pure_blob_projection_failure_uses_json_fallback(store, monkeypatch):
    beam, query, _ = store
    target, _ = _write(beam, monkeypatch, query, "recoverable projection failure")
    beam.conn.execute(
        "INSERT INTO memory_embeddings(memory_id, embedding_json) VALUES (?,?)",
        (target, json.dumps(query.tolist())),
    )
    bm._mark_vec_store_norm_bit(beam.conn)
    beam.conn.commit()
    real_execute = beam.conn.execute
    failures = 0

    def fail_blob_projection_once(sql, params=()):
        nonlocal failures
        normalized = " ".join(str(sql).split())
        if (
            failures == 0
            and "SELECT rowid, distance, embedding" in normalized
            and "vec_episodes" in normalized
            and " MATCH " in normalized
        ):
            failures += 1
            raise sqlite3.OperationalError("injected blob projection failure")
        return real_execute(sql, params)

    monkeypatch.setattr(beam.conn, "execute", fail_blob_projection_once)
    engine = PolyphonicRecallEngine(db_path=beam.db_path, conn=beam.conn)
    results = engine._vector_voice(query)

    assert failures == 1
    assert [result.memory_id for result in results] == [target]
    assert results[0].metadata["backend"] == "memory_embeddings"
    assert engine.last_call_fallback["em"] is True



def test_no_vec_reindex_preserves_unmarked_store(store, monkeypatch):

    beam, query, _ = store
    _, rowid = _write(beam, monkeypatch, query, "legacy marker sentinel")
    beam.conn.execute("PRAGMA user_version = 0")
    beam.conn.commit()
    blob_before = beam.conn.execute(
        "SELECT embedding FROM vec_episodes WHERE rowid=?", (rowid,)
    ).fetchone()[0]
    db_path = beam.db_path
    beam.conn.close()

    monkeypatch.setattr(bm._embeddings, "EMBEDDING_DIM", 64)
    monkeypatch.setattr(bm._embeddings, "available", lambda: True)
    monkeypatch.setattr(
        bm._embeddings, "embed", lambda texts: [query for _ in texts]
    )
    offline = sqlite3.connect(db_path)
    offline.row_factory = sqlite3.Row
    assert bm._vec_available(offline) is False
    plan = bm.reindex_vectors(offline)
    assert plan["sqlite_vec"] is False
    assert not (
        offline.execute("PRAGMA user_version").fetchone()[0]
        & bm._VEC_NORM_BIT
    )
    offline.close()

    reopened = bm.BeamMemory(session_id=beam.session_id, db_path=db_path)
    assert bm._classify_vec_store_regime(reopened.conn) == "legacy"
    blob_after = reopened.conn.execute(
        "SELECT embedding FROM vec_episodes WHERE rowid=?", (rowid,)
    ).fetchone()[0]
    assert blob_after == blob_before
    reopened.conn.close()
