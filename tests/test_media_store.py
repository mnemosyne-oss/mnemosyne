"""Behavioural tests for MediaStore (RFC 0003 §2.1, §2.2, §2.4)."""

import json
import sqlite3

import pytest

from mnemosyne.core.media import MediaStore, MomentDraft


@pytest.fixture
def store(tmp_path):
    s = MediaStore(db_path=tmp_path / "media.db")
    yield s
    try:
        s.conn.close()
    except Exception:
        pass


def _asset(store, **kwargs):
    params = {"ref_kind": "url", "ref_value": "http://x/1", "modality": "image"}
    params.update(kwargs)
    return store.upsert_asset(**params)


# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------

def test_upsert_is_idempotent_by_reference(store):
    first = _asset(store)
    second = _asset(store)
    assert first.asset_id == second.asset_id
    assert (first.status, second.status) == ("created", "unchanged")
    assert store.conn.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0] == 1


def test_equivalent_references_resolve_to_one_asset(store):
    a = _asset(store, ref_value="HTTP://X.COM/a")
    b = _asset(store, ref_value="http://x.com/a")
    assert a.asset_id == b.asset_id


def test_same_ref_value_under_two_kinds_is_two_assets(store):
    a = _asset(store, ref_kind="url", ref_value="abc")
    b = _asset(store, ref_kind="file", ref_value="abc")
    assert a.asset_id != b.asset_id


def test_asset_id_is_stable_when_the_hash_arrives_later(store):
    """An asset first seen by reference and later by bytes keeps its id.
    Recomputing it would orphan every moment already written against it."""
    first = _asset(store)
    assert first.asset_id.startswith("ref:")

    store.add_moments(first.asset_id, [MomentDraft(kind="caption", text="a cat")])

    second = _asset(store, content_hash="DEADBEEF")
    assert second.asset_id == first.asset_id
    assert second.status == "updated"
    assert store.get_asset(first.asset_id)["content_hash"] == "deadbeef"
    # The moment is still bound to a live asset.
    assert len(store.get_moments(first.asset_id)) == 1


def test_a_second_reference_to_known_bytes_aliases_rather_than_raising(store):
    """One file reachable by both a URL and a local path is one asset -- not
    two rows, and certainly not an IntegrityError on the partial hash index."""
    first = _asset(store, content_hash="abc123")
    second = _asset(store, ref_kind="file", ref_value="/tmp/a.png", content_hash="abc123")

    assert second.asset_id == first.asset_id
    assert second.status == "aliased"
    assert store.conn.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0] == 1

    meta = json.loads(store.get_asset(first.asset_id)["metadata"])
    assert meta["alt_refs"] == [{"ref_kind": "file", "ref_value": "/tmp/a.png"}]


def test_alt_refs_are_deduped(store):
    _asset(store, content_hash="abc123")
    for _ in range(3):
        _asset(store, ref_kind="file", ref_value="/tmp/a.png", content_hash="abc123")
    meta = json.loads(store.find_asset_by_ref("url", "http://x/1")["metadata"])
    assert len(meta["alt_refs"]) == 1


def test_re_registration_adds_knowledge_but_never_removes_it(store):
    first = _asset(store, title="Original", width=100)
    _asset(store, title="Different", height=200)
    row = store.get_asset(first.asset_id)
    assert row["title"] == "Original", "a second sighting must not overwrite"
    assert row["width"] == 100
    assert row["height"] == 200, "but it may fill in what was NULL"


def test_captured_at_precision_default_is_not_treated_as_knowledge(store):
    first = _asset(store, captured_at="2026-01-01", captured_at_precision="exact")
    _asset(store)  # defaults to 'unknown'
    assert store.get_asset(first.asset_id)["captured_at_precision"] == "exact"


@pytest.mark.parametrize("kwargs", [
    {"modality": "hologram"},
    {"ref_kind": "carrier-pigeon"},
    {"understanding_status": "vibes"},
])
def test_upsert_rejects_unknown_vocabulary(store, kwargs):
    with pytest.raises(ValueError):
        _asset(store, **kwargs)


def test_understanding_status_transitions(store):
    a = _asset(store)
    assert store.get_asset(a.asset_id)["understanding_status"] == "pending"
    assert store.set_understanding_status(
        a.asset_id, "ok", provider="openai_compat", provider_model="qwen-vl"
    )
    row = store.get_asset(a.asset_id)
    assert row["understanding_status"] == "ok"
    assert row["provider_model"] == "qwen-vl"
    assert not store.set_understanding_status("nope", "ok")
    with pytest.raises(ValueError):
        store.set_understanding_status(a.asset_id, "vibes")


def test_unavailable_is_a_success_state_not_an_error(store):
    """Rung 4 of the RFC 0002 §3.3 degradation ladder: the asset is registered
    and recallable by reference even though nothing described it."""
    a = _asset(store)
    assert store.set_understanding_status(a.asset_id, "unavailable")
    assert store.get_asset(a.asset_id)["understanding_status"] == "unavailable"


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def test_moments_write_and_read_back(store):
    a = _asset(store, modality="video")
    result = store.add_moments(a.asset_id, [
        MomentDraft(kind="shot", text="a wide shot", span_kind="time",
                    t_start_ms=0, t_end_ms=5000, ordinal=0),
        MomentDraft(kind="shot", text="a close-up", span_kind="time",
                    t_start_ms=5000, t_end_ms=9000, ordinal=1),
    ])
    assert (result.inserted, result.duplicate, result.unbound) == (2, 0, 0)
    moments = store.get_moments(a.asset_id)
    assert [m["text"] for m in moments] == ["a wide shot", "a close-up"]
    assert [m["span_key"] for m in moments] == ["v1:time:0-5000", "v1:time:5000-9000"]


def test_re_ingest_is_a_no_op(store):
    """The idempotency contract: an archive re-push writes nothing new."""
    a = _asset(store)
    drafts = [MomentDraft(kind="caption", text="a terminal window")]
    first = store.add_moments(a.asset_id, drafts)
    second = store.add_moments(a.asset_id, drafts)

    assert (first.inserted, first.duplicate) == (1, 0)
    assert (second.inserted, second.duplicate) == (0, 1)
    assert second.moment_ids == first.moment_ids
    assert len(store.get_moments(a.asset_id)) == 1


def test_two_speakers_over_one_window_are_two_moments(store):
    """The correction that motivated folding speaker into span_key. Without it
    INSERT OR IGNORE drops the second -- silently, and precisely in the case
    diarization exists to handle."""
    a = _asset(store, modality="audio")
    result = store.add_moments(a.asset_id, [
        MomentDraft(kind="transcript", text="so I said", span_kind="time",
                    t_start_ms=1000, t_end_ms=2000, speaker="Alice"),
        MomentDraft(kind="transcript", text="no you didn't", span_kind="time",
                    t_start_ms=1000, t_end_ms=2000, speaker="Bob"),
    ])
    assert result.inserted == 2
    assert len(store.get_moments(a.asset_id)) == 2


def test_intra_batch_duplicates_are_rejected_not_swallowed(store):
    """INSERT OR IGNORE is right for a *repeat ingest* and wrong for a single
    batch that believes it is writing two distinct moments."""
    a = _asset(store)
    with pytest.raises(ValueError, match=r"drafts\[1\] duplicates drafts\[0\]"):
        store.add_moments(a.asset_id, [
            MomentDraft(kind="caption", text="one"),
            MomentDraft(kind="caption", text="two"),  # same span, same kind
        ])
    assert store.get_moments(a.asset_id) == []


def test_validation_writes_nothing_at_all(store):
    """A batch with one bad draft must not leave a partial span index behind
    that nothing knows is partial."""
    a = _asset(store)
    with pytest.raises(ValueError, match=r"drafts\[2\]"):
        store.add_moments(a.asset_id, [
            MomentDraft(kind="caption", text="good", span_kind="whole"),
            MomentDraft(kind="ocr", text="also good", span_kind="char",
                        char_start=0, char_end=10),
            MomentDraft(kind="shot", text="bad", span_kind="time"),  # no t_start_ms
        ])
    assert store.get_moments(a.asset_id) == []


@pytest.mark.parametrize("draft,match", [
    (MomentDraft(kind="vibe", text="x"), "unknown kind"),
    (MomentDraft(kind="caption", text="   "), "non-empty"),
    (MomentDraft(kind="shot", text="x", span_kind="time", t_start_ms=0, page_start=1),
     "does not own"),
    (MomentDraft(kind="caption", text="x", span_kind="whole", speaker="alice"),
     "only valid on span_kind='time'"),
])
def test_draft_validation_names_the_offending_index(store, draft, match):
    a = _asset(store)
    with pytest.raises(ValueError, match=match) as exc:
        store.add_moments(a.asset_id, [draft])
    assert "drafts[0]" in str(exc.value)


def test_moments_on_a_nonexistent_asset_are_refused(store):
    """The one orphan kind that is real corruption -- refuse to create it."""
    with pytest.raises(ValueError, match="no media_assets row"):
        store.add_moments("ghost", [MomentDraft(kind="caption", text="x")])


def test_bbox_is_stored_in_the_same_precision_the_span_key_uses(store):
    a = _asset(store)
    store.add_moments(a.asset_id, [
        MomentDraft(kind="ocr", text="LOGIN", span_kind="box",
                    bbox=[0.1 + 0.2, 0.2, 0.3, 0.4]),
    ])
    moment = store.get_moments(a.asset_id)[0]
    assert json.loads(moment["bbox"]) == [0.3, 0.2, 0.3, 0.4]
    assert moment["span_key"] == "v1:box:0.300000,0.200000,0.300000,0.400000"


def test_empty_batch_is_a_no_op(store):
    a = _asset(store)
    result = store.add_moments(a.asset_id, [])
    assert (result.inserted, result.duplicate, result.moment_ids) == (0, 0, [])


# ---------------------------------------------------------------------------
# Soft references (§2.4)
# ---------------------------------------------------------------------------

def _with_working_memory(store):
    store.conn.execute("CREATE TABLE working_memory (id TEXT PRIMARY KEY)")
    store.conn.execute("INSERT INTO working_memory VALUES ('mem-live')")
    store.conn.commit()


def test_a_dead_memory_id_is_nulled_and_counted_not_raised(store):
    """Eviction (_trim_working_memory) and consolidation both legitimately
    remove the row between drafting and insert."""
    _with_working_memory(store)
    a = _asset(store)
    result = store.add_moments(a.asset_id, [
        MomentDraft(kind="caption", text="bound", memory_id="mem-live"),
        MomentDraft(kind="ocr", text="unbound", memory_id="mem-gone"),
    ])
    assert (result.inserted, result.unbound) == (2, 1)
    by_text = {m["text"]: m["memory_id"] for m in store.get_moments(a.asset_id)}
    assert by_text == {"bound": "mem-live", "unbound": None}


def test_binding_is_guarded_by_existence_rather_than_a_foreign_key(store):
    _with_working_memory(store)
    a = _asset(store)
    moment_id = store.add_moments(
        a.asset_id, [MomentDraft(kind="caption", text="x")]
    ).moment_ids[0]

    assert not store.bind_moment_memory(moment_id, "mem-gone")
    assert store.get_moments(a.asset_id)[0]["memory_id"] is None
    assert store.bind_moment_memory(moment_id, "mem-live")
    assert store.get_moments(a.asset_id)[0]["memory_id"] == "mem-live"
    assert [m["text"] for m in store.moments_for_memory("mem-live")] == ["x"]


def test_a_standalone_store_preserves_bindings_it_cannot_verify(store):
    """Without a working_memory table there is nothing to check against;
    nulling the binding would be a silent data loss, so it is kept and left for
    doctor to count."""
    a = _asset(store)
    result = store.add_moments(
        a.asset_id, [MomentDraft(kind="caption", text="x", memory_id="mem-elsewhere")]
    )
    assert result.unbound == 0
    assert store.get_moments(a.asset_id)[0]["memory_id"] == "mem-elsewhere"


def test_count_orphans_separates_the_two_kinds(store):
    _with_working_memory(store)
    a = _asset(store)
    store.add_moments(a.asset_id, [
        MomentDraft(kind="caption", text="ok", memory_id="mem-live"),
    ])
    assert store.count_orphans() == {"asset_orphans": 0, "memory_orphans": 0}

    # Consolidation summarized the memory row away -- the expected steady state.
    store.conn.execute("DELETE FROM working_memory WHERE id = 'mem-live'")
    # ...and a genuinely corrupt row, which nothing in the design produces.
    store.conn.execute(
        "INSERT INTO media_moments "
        "(moment_id, asset_id, kind, text, span_kind, span_key) "
        "VALUES ('m-orphan', 'ghost-asset', 'caption', 'x', 'whole', 'v1:whole')"
    )
    store.conn.commit()
    assert store.count_orphans() == {"asset_orphans": 1, "memory_orphans": 1}


# ---------------------------------------------------------------------------
# No bytes, ever
# ---------------------------------------------------------------------------

def test_the_store_never_reaches_for_the_blob_writer(store, monkeypatch):
    """MediaStore holds references and text. Byte handling belongs to
    content_sanitizer (write) and resolvers (read), never here."""
    from mnemosyne.core import content_sanitizer

    def _explode(*args, **kwargs):
        raise AssertionError("MediaStore must never store blobs")

    monkeypatch.setattr(content_sanitizer, "_store_blob", _explode)

    a = _asset(store, ref_kind="blob", ref_value="blob://sha256/" + "ab" * 32)
    store.add_moments(a.asset_id, [MomentDraft(kind="caption", text="a chart")])
    assert store.get_moments(a.asset_id)[0]["text"] == "a chart"


def test_stored_values_are_text_not_bytes(store):
    a = _asset(store)
    store.add_moments(a.asset_id, [MomentDraft(kind="caption", text="x")])
    for table in ("media_assets", "media_moments"):
        for row in store.conn.execute(f"SELECT * FROM {table}").fetchall():
            assert not any(isinstance(v, (bytes, bytearray)) for v in tuple(row))


def test_store_survives_a_shared_connection_being_reused(tmp_path):
    """The BeamMemory wiring shape: one connection, several stores."""
    conn = sqlite3.connect(tmp_path / "shared.db")
    conn.row_factory = sqlite3.Row
    try:
        one = MediaStore(db_path=tmp_path / "shared.db", conn=conn)
        two = MediaStore(db_path=tmp_path / "shared.db", conn=conn)
        a = one.upsert_asset(ref_kind="url", ref_value="http://x/1", modality="image")
        assert two.get_asset(a.asset_id) is not None
    finally:
        conn.close()
