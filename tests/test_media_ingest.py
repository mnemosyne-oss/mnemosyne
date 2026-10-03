"""End-to-end ingest tests for remember_media (RFC 0003 §3.4, §5.1).

Three groups, in order of how much they matter:

1. **Privacy invariants**, as executable assertions: zero outbound traffic with
   the flag unset, no bytes read for a reference we never describe, and no
   `data:` URI reaching any column.
2. **Ordering**, which is what makes a crash mid-ingest recoverable.
3. **The degradation ladder**, where rung 4 is a success state.
"""

import base64
import json

import pytest

from mnemosyne.core import media
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.media import MediaIngestResult, remember_media
from mnemosyne.core.modality_backends import (
    CallableModalityBackend,
    DescribedMoment,
    DescribeResult,
    set_modality_backend,
)

_PNG = b"\x89PNG\r\n\x1a\n" + b"fake pixel data" * 4
_DATA_URI = "data:image/png;base64," + base64.b64encode(_PNG).decode()


@pytest.fixture
def beam(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
    b = BeamMemory(session_id="ingest", db_path=tmp_path / "ingest.db")
    yield b
    try:
        b.conn.close()
    except Exception:
        pass


@pytest.fixture
def enabled(monkeypatch, tmp_path):
    """Opt in the way a real operator must.

    Setting only ``MNEMOSYNE_MODALITY_ENABLED`` is not enough. ``config.yaml``
    wins over env vars, and presence beats value: once a key exists in the
    file, the variable is never consulted. A seeded config carries
    ``modality_enabled: false``, so an env-only fixture passes on a machine
    whose config predates the key and fails on a fresh one, which is exactly
    how this slipped through locally and failed in CI.

    Pointing ``MNEMOSYNE_DATA_DIR`` at a fresh directory makes the seed honour
    the variables, which is the same path a new install takes.
    """
    from mnemosyne.core.config import MnemosyneConfig
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg-enabled"))
    monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
    MnemosyneConfig.reset_instance()
    yield
    MnemosyneConfig.reset_instance()


def _provider(result=None, modalities=frozenset({"image", "video", "audio", "document"})):
    calls = []

    def _describe(request):
        calls.append(request)
        if callable(result):
            return result(request)
        return result

    set_modality_backend(
        CallableModalityBackend(name="stub", func=_describe, modalities=modalities)
    )
    return calls


_CAPTION = DescribeResult(
    summary="a screenshot of a terminal",
    moments=[
        DescribedMoment(kind="caption", text="a terminal window running tests"),
        DescribedMoment(kind="ocr", text="2478 passed", bbox=[0.1, 0.2, 0.3, 0.4]),
    ],
    provider="stub",
    model="stub/model",
)


def _png_file(tmp_path, name="shot.png"):
    path = tmp_path / name
    path.write_bytes(_PNG)
    return path


# ---------------------------------------------------------------------------
# Privacy invariants
# ---------------------------------------------------------------------------

def test_no_socket_is_opened_when_disabled(beam, tmp_path, monkeypatch):
    """The whole privacy claim in one assertion."""
    import socket

    def _forbidden(*args, **kwargs):
        raise AssertionError("ingest opened a socket with modality support off")

    _provider(_CAPTION)
    monkeypatch.setattr(socket, "socket", _forbidden)

    result = remember_media(beam, ref=str(_png_file(tmp_path)))
    assert result.status == "unavailable"


def test_the_provider_is_never_called_when_disabled(beam, tmp_path):
    calls = _provider(_CAPTION)
    remember_media(beam, ref=str(_png_file(tmp_path)))
    assert calls == []


def test_no_bytes_are_read_when_disabled(beam, tmp_path, monkeypatch):
    """Not just 'nothing is sent' -- the file is not even opened."""
    from pathlib import Path

    original = Path.read_bytes

    def _watched(self, *args, **kwargs):
        assert self.suffix != ".png", "ingest read media bytes with the flag off"
        return original(self, *args, **kwargs)

    _provider(_CAPTION)
    monkeypatch.setattr(Path, "read_bytes", _watched)
    remember_media(beam, ref=str(_png_file(tmp_path)))


def test_a_data_uri_never_reaches_any_column(beam):
    """The regression test for the prerequisite. `sanitize_content` only ever
    inspects `content`; `metadata_json` and `ref_value` are uncapped TEXT
    columns it never sees, so a `data:` URI arriving as a *reference* would
    write megabytes of base64 straight past every guard."""
    result = remember_media(beam, ref=_DATA_URI)

    rows = beam.conn.execute(
        "SELECT content, metadata_json FROM working_memory"
    ).fetchall()
    assert rows, "ingest wrote no memory row at all"

    for content, metadata_json in rows:
        assert "data:" not in (content or "")
        assert "base64" not in (metadata_json or "")
        assert len(metadata_json or "") < 4096, "metadata_json is not small"

    asset = beam.media.get_asset(result.asset_id)
    assert asset["ref_value"].startswith("blob://sha256/")
    assert "base64" not in asset["metadata"]
    assert "data:" not in asset["ref_value"]


def test_the_data_uri_bytes_land_in_the_blob_store(beam, tmp_path):
    """Converted at the door, not discarded: the bytes are still retrievable
    through the resolver registry."""
    from mnemosyne.core.resolvers import resolve_open

    result = remember_media(beam, ref=_DATA_URI)
    asset = beam.media.get_asset(result.asset_id)

    stream = resolve_open(asset["ref_value"])
    assert stream is not None
    with stream:
        assert stream.read() == _PNG


def test_a_reference_is_not_copied_into_content(beam, tmp_path):
    path = _png_file(tmp_path, "vacation-photo.png")
    remember_media(beam, ref=str(path))
    contents = [r[0] for r in beam.conn.execute("SELECT content FROM working_memory")]
    assert contents == ["Referenced image: vacation-photo.png"]
    assert not any(str(path) in c for c in contents)


def test_an_oversized_reference_is_refused(beam):
    with pytest.raises(ValueError, match="over the .* limit"):
        remember_media(beam, ref="x" * (media.MAX_REF_LENGTH + 1))


def test_ingest_never_stores_a_blob_for_an_ordinary_reference(beam, tmp_path, monkeypatch):
    from mnemosyne.core import content_sanitizer

    monkeypatch.setattr(
        content_sanitizer, "_store_blob",
        lambda *a, **k: pytest.fail("a non-data ref must not be copied into the blob store"),
    )
    remember_media(beam, ref="https://example.test/a.png")
    remember_media(beam, ref=str(_png_file(tmp_path)))


def test_recall_calls_no_resolver(beam, tmp_path, enabled, monkeypatch):
    """Recall must never generate outbound traffic or read bytes."""
    _provider(_CAPTION)
    remember_media(beam, ref=str(_png_file(tmp_path)))

    from mnemosyne.core import resolvers

    monkeypatch.setattr(
        resolvers, "resolve_open",
        lambda *a, **k: pytest.fail("recall opened media bytes"),
    )
    monkeypatch.setattr(
        resolvers, "resolve_head",
        lambda *a, **k: pytest.fail("recall stat-ed media bytes"),
    )
    beam.recall("terminal window")


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_moments_become_recallable_memories(beam, tmp_path, enabled):
    _provider(_CAPTION)
    result = remember_media(beam, ref=str(_png_file(tmp_path)), title="test run")

    assert result.status == "ok" and result.described
    assert len(result.moment_ids) == 2
    assert len(result.memory_ids) == 2

    contents = {
        r[0] for r in beam.conn.execute("SELECT content FROM working_memory")
    }
    assert "a terminal window running tests" in contents
    assert "2478 passed" in contents
    assert "Referenced image: test run" in contents


def test_moment_memories_are_typed_as_artifacts(beam, tmp_path, enabled):
    """`classify_memory` would call a caption an 'observation' (RFC 0003 §3.3)."""
    _provider(_CAPTION)
    result = remember_media(beam, ref=str(_png_file(tmp_path)))

    placeholders = ",".join("?" for _ in result.memory_ids)
    types = {
        r[0] for r in beam.conn.execute(
            f"SELECT memory_type FROM working_memory WHERE id IN ({placeholders})",
            tuple(result.memory_ids),
        )
    }
    assert types == {"artifact"}


def test_identical_captions_from_different_assets_stay_distinct(beam, tmp_path, enabled):
    """The highest silent-wrong-answer risk in the whole design. Without
    `dedupe=False`, `_find_duplicate` collapses these into one row and the
    second moment binds to the first asset's memory -- so recall answers
    'which image showed X' with the wrong image, with no error and no log."""
    _provider(DescribeResult(
        moments=[DescribedMoment(kind="caption", text="a black frame")],
        provider="stub",
    ))

    first = remember_media(beam, ref=str(_png_file(tmp_path, "a.png")))
    second = remember_media(beam, ref=str(_png_file(tmp_path, "b.png")))

    assert first.asset_id != second.asset_id
    assert first.memory_ids != second.memory_ids
    assert len(first.memory_ids) == len(second.memory_ids) == 1

    count = beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = 'a black frame'"
    ).fetchone()[0]
    assert count == 2, "the two captions collapsed into one memory row"

    for result in (first, second):
        moment = beam.media.get_moments(result.asset_id)[0]
        assert moment["memory_id"] == result.memory_ids[0]


def test_span_kinds_are_taken_from_what_the_provider_populated(beam, tmp_path, enabled):
    _provider(DescribeResult(provider="stub", moments=[
        DescribedMoment(kind="shot", text="a wide shot", t_start_ms=0, t_end_ms=5000),
        DescribedMoment(kind="ocr", text="LOGIN", bbox=[0.1, 0.2, 0.3, 0.4]),
        DescribedMoment(kind="page", text="page three", page_start=3),
        DescribedMoment(kind="caption", text="the whole thing"),
    ]))
    result = remember_media(beam, ref=str(_png_file(tmp_path)), modality="video")

    kinds = {m["kind"]: m["span_kind"] for m in beam.media.get_moments(result.asset_id)}
    assert kinds == {"shot": "time", "ocr": "box", "page": "page", "caption": "whole"}


def test_re_ingest_writes_nothing_new(beam, tmp_path, enabled):
    """Idempotency end to end: same asset, same moments, same anchor."""
    _provider(_CAPTION)
    path = str(_png_file(tmp_path))

    first = remember_media(beam, ref=path)
    before = beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]
    second = remember_media(beam, ref=path)
    after = beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0]

    assert second.asset_id == first.asset_id
    assert second.anchor_memory_id == first.anchor_memory_id
    assert second.memory_ids == first.memory_ids
    assert after == before, "a repeat ingest created new memory rows"
    assert len(beam.media.get_moments(first.asset_id)) == 2


def test_provider_receives_bytes_for_a_local_file(beam, tmp_path, enabled):
    calls = _provider(_CAPTION)
    remember_media(beam, ref=str(_png_file(tmp_path)))

    assert len(calls) == 1
    assert calls[0].fetch is not None
    assert calls[0].fetch() == _PNG, "the lazy fetcher must reach the file"


def test_provider_receives_bytes_for_a_blob_reference(beam, enabled):
    calls = _provider(_CAPTION)
    remember_media(beam, ref=_DATA_URI)
    assert calls[0].fetch() == _PNG, "the fetcher must go through the resolver registry"


def test_the_fetcher_refuses_an_oversized_file(beam, tmp_path, enabled, monkeypatch):
    monkeypatch.setattr(media, "MAX_FETCH_BYTES", 4)
    calls = _provider(_CAPTION)
    remember_media(beam, ref=str(_png_file(tmp_path)))
    assert calls[0].fetch() is None


def test_a_missing_local_file_still_registers(beam, enabled):
    """The reference is worth recording even when the bytes are gone."""
    calls = _provider(_CAPTION)
    result = remember_media(beam, ref="/nonexistent/photo.png")
    assert calls[0].fetch() is None
    assert beam.media.get_asset(result.asset_id) is not None


# ---------------------------------------------------------------------------
# The degradation ladder
# ---------------------------------------------------------------------------

def test_rung_four_is_a_success_state(beam, tmp_path):
    """No provider configured. The user still gets a searchable record that
    they referenced this file, which is strictly more than they had."""
    result = remember_media(beam, ref=str(_png_file(tmp_path)))

    assert isinstance(result, MediaIngestResult)
    assert result.status == "unavailable"
    assert result.described is False
    assert result.anchor_memory_id is not None
    assert result.moment_ids == []
    assert beam.media.get_asset(result.asset_id)["understanding_status"] == "unavailable"


def test_the_anchor_is_written_before_the_gate(beam, tmp_path, enabled):
    """A provider that explodes must not cost the user the reference row."""
    def _explode(request):
        raise RuntimeError("provider on fire")

    set_modality_backend(CallableModalityBackend(name="broken", func=_explode))
    result = remember_media(beam, ref=str(_png_file(tmp_path)))

    assert result.anchor_memory_id is not None
    assert result.status == "unavailable"


def test_a_refusal_is_recorded_as_refused(beam, tmp_path, enabled):
    _provider(DescribeResult(provider="stub", refused=True,
                             warnings=["declined on policy grounds"]))
    result = remember_media(beam, ref=str(_png_file(tmp_path)))

    assert result.status == "refused"
    assert beam.media.get_asset(result.asset_id)["understanding_status"] == "refused"
    assert result.warnings == ["declined on policy grounds"]
    assert result.moment_ids == []


def test_unwritable_provider_moments_are_dropped_and_counted(beam, tmp_path, enabled):
    """One bad moment must not cost the batch: `add_moments` is deliberately
    all-or-nothing, so the filtering happens before it."""
    _provider(DescribeResult(provider="stub", moments=[
        DescribedMoment(kind="caption", text="good"),
        DescribedMoment(kind="caption", text=""),               # no text
        DescribedMoment(kind="shot", text="bad span", t_start_ms=-5),
    ]))
    result = remember_media(beam, ref=str(_png_file(tmp_path)))

    assert result.status == "partial"
    assert len(result.moment_ids) == 1
    assert any("dropped" in w for w in result.warnings)


def test_repeated_provider_moments_are_deduped_before_the_strict_batch_check(beam, tmp_path, enabled):
    _provider(DescribeResult(provider="stub", moments=[
        DescribedMoment(kind="caption", text="a chart"),
        DescribedMoment(kind="caption", text="a chart"),
    ]))
    result = remember_media(beam, ref=str(_png_file(tmp_path)))
    assert len(result.moment_ids) == 1
    assert result.status == "partial"


def test_a_summary_with_no_moments_still_produces_one(beam, tmp_path, enabled):
    _provider(DescribeResult(provider="stub", summary="a photo of a dog"))
    result = remember_media(beam, ref=str(_png_file(tmp_path)))

    assert result.status == "ok"
    moments = beam.media.get_moments(result.asset_id)
    assert [(m["kind"], m["text"]) for m in moments] == [("summary", "a photo of a dog")]


def test_an_empty_result_leaves_the_asset_unavailable(beam, tmp_path, enabled):
    _provider(DescribeResult(provider="stub"))
    result = remember_media(beam, ref=str(_png_file(tmp_path)))
    assert result.status == "unavailable"
    assert result.anchor_memory_id is not None


def test_a_backend_that_does_not_serve_the_modality_is_not_called(beam, tmp_path, enabled):
    calls = _provider(_CAPTION, modalities=frozenset({"audio"}))
    result = remember_media(beam, ref=str(_png_file(tmp_path)))
    assert calls == []
    assert result.status == "unavailable"


def test_status_is_never_left_pending(beam, tmp_path, enabled, monkeypatch):
    """The `finally` exists because nothing retries a stranded `pending`, and
    doctor would not flag it either."""
    _provider(_CAPTION)
    monkeypatch.setattr(
        media, "_bind_moment_memories",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("constraint error")),
    )
    with pytest.raises(RuntimeError):
        remember_media(beam, ref=str(_png_file(tmp_path)))

    statuses = [
        r[0] for r in beam.conn.execute("SELECT understanding_status FROM media_assets")
    ]
    assert "pending" not in statuses


# ---------------------------------------------------------------------------
# Reference handling and inference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ref,expected_kind", [
    ("https://example.test/a.png", "url"),
    ("https://www.youtube.com/watch?v=abc", "youtube"),
    ("https://youtu.be/abc", "youtube"),
    ("blob://sha256/" + "ab" * 32, "blob"),
    ("/tmp/a.png", "file"),
    ("ab" * 32, "sha256"),
])
def test_ref_kind_inference(ref, expected_kind):
    kind, _, _, _ = media.normalize_media_input(ref)
    assert kind == expected_kind


def test_an_explicit_ref_kind_wins():
    kind, _, _, _ = media.normalize_media_input("/tmp/a.png", ref_kind="archive")
    assert kind == "archive"


@pytest.mark.parametrize("ref,mime,expected", [
    ("/tmp/a.png", None, "image"),
    ("/tmp/a.mp4", None, "video"),
    ("/tmp/a.mp3", None, "audio"),
    ("/tmp/a.pdf", None, "document"),
    ("/tmp/a.unknown", None, "document"),
    ("/tmp/a.unknown", "image/png", "image"),
    ("https://example.test/photo.jpg?v=2", None, "image"),
])
def test_modality_inference(ref, mime, expected):
    kind, value, mime_out, _ = media.normalize_media_input(ref, mime=mime)
    assert media.infer_modality(kind, value, mime_out) == expected


def test_an_explicit_mime_beats_the_extension():
    assert media.infer_modality("file", "/tmp/a.png", "audio/mpeg") == "audio"


def test_youtube_urls_are_video_without_an_extension():
    assert media.infer_modality("youtube", "https://youtu.be/abc") == "video"


def test_an_unknown_modality_is_refused(beam):
    with pytest.raises(ValueError, match="unknown modality"):
        remember_media(beam, ref="/tmp/a.png", modality="hologram")


def test_an_empty_ref_is_refused(beam):
    with pytest.raises(ValueError, match="non-empty"):
        remember_media(beam, ref="   ")


def test_the_same_bytes_twice_are_one_asset(beam):
    """Content addressing end to end: the same payload under two ingests lands
    on one asset, because the blob hash is the reference."""
    first = remember_media(beam, ref=_DATA_URI)
    second = remember_media(beam, ref=_DATA_URI)
    assert first.asset_id == second.asset_id
    assert beam.conn.execute("SELECT COUNT(*) FROM media_assets").fetchone()[0] == 1


def test_a_second_reference_to_known_bytes_aliases(beam):
    """A URL later found to hold bytes we already have is the same asset, not
    an IntegrityError on the partial content-hash index."""
    import hashlib

    first = remember_media(beam, ref=_DATA_URI)
    digest = hashlib.sha256(_PNG).hexdigest()
    second = beam.media.upsert_asset(
        ref_kind="url", ref_value="https://example.test/a.png",
        modality="image", content_hash=digest,
    )
    assert second.asset_id == first.asset_id
    assert second.status == "aliased"


# ---------------------------------------------------------------------------
# The BeamMemory method
# ---------------------------------------------------------------------------

def test_the_beam_method_delegates(beam, tmp_path, enabled):
    _provider(_CAPTION)
    result = beam.remember_media(str(_png_file(tmp_path)), title="via the method")
    assert isinstance(result, MediaIngestResult)
    assert result.status == "ok"
    assert beam.media.get_asset(result.asset_id)["title"] == "via the method"


def test_metadata_carries_the_asset_link(beam, tmp_path, enabled):
    _provider(_CAPTION)
    result = beam.remember_media(str(_png_file(tmp_path)))

    row = beam.conn.execute(
        "SELECT metadata_json FROM working_memory WHERE id = ?",
        (result.memory_ids[0],),
    ).fetchone()
    meta = json.loads(row[0])["media"]
    assert meta["asset_id"] == result.asset_id
    assert meta["moment_id"] in result.moment_ids
    assert meta["provider"] == "stub"


# ---------------------------------------------------------------------------
# End to end through the real adapter
# ---------------------------------------------------------------------------

def test_ingest_through_the_real_openai_compat_adapter(beam, tmp_path, monkeypatch):
    """The whole ladder with nothing stubbed but the endpoint itself.

    This is the test that proves "working ingest": a local PNG, the real
    `modality_openai_compat` adapter, a real HTTP round trip to a localhost
    stub, and recallable memories at the end. Everything below the socket is
    production code.
    """
    from mnemosyne.core.config import MnemosyneConfig
    from mnemosyne.core.modality_openai_compat import register_if_configured
    from tests.test_modality_openai_compat import _Stub

    reply = json.dumps({
        "summary": "a screenshot of a terminal",
        "moments": [{"kind": "caption", "text": "a terminal running the test suite"}],
    })
    server = _Stub([(200, reply)])
    try:
        # A fresh data directory, because the endpoint keys have to be set
        # before the config is seeded: config.yaml wins over the environment
        # and presence beats value, so the empty base_url seeded by the
        # `enabled` fixture would shadow the variable set here.
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg-live-adapter"))
        monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
        monkeypatch.setenv("MNEMOSYNE_MODALITY_BASE_URL", server.base_url)
        monkeypatch.setenv("MNEMOSYNE_MODALITY_API_KEY", "test-key-not-real")
        monkeypatch.setenv("MNEMOSYNE_MODALITY_VISION_MODEL", "some/vision-model")
        MnemosyneConfig.reset_instance()

        assert register_if_configured() is True
        result = beam.remember_media(str(_png_file(tmp_path)))

        assert result.status == "ok"
        assert len(result.memory_ids) == 1

        # The bytes went out as a data part, since a local path is not fetchable
        # by the provider -- this is the DescribeRequest.fetch path, live.
        part = server.requests[0]["messages"][0]["content"][1]
        url = part["image_url"]["url"]
        assert url.startswith("data:")
        assert base64.b64decode(url.split(",", 1)[1]) == _PNG

        # ...and none of that reached the database.
        for content, metadata_json in beam.conn.execute(
            "SELECT content, metadata_json FROM working_memory"
        ):
            assert "data:" not in (content or "")
            assert "base64" not in (metadata_json or "")

        hits = beam.recall("terminal test suite", top_k=5)
        assert any("terminal running the test suite" in h.get("content", "")
                   for h in hits)
    finally:
        server.close()
        MnemosyneConfig.reset_instance()
