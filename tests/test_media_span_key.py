"""Pure-function tests for span identity (RFC 0003 §2.2).

``span_key`` is the idempotency contract for media ingest: it is what the
``(asset_id, kind, span_key)`` unique index compares, so its format is a
persisted data format, not an implementation detail.
"""

import pytest

from mnemosyne.core.media import (
    SPAN_KEY_VERSION,
    compute_asset_id,
    compute_moment_id,
    normalize_ref,
    span_key,
    validate_span,
)


# ---------------------------------------------------------------------------
# Golden strings
# ---------------------------------------------------------------------------

_GOLDEN = [
    ("whole", {}, "v1:whole"),
    ("time", {"t_start_ms": 90000, "t_end_ms": 96000}, "v1:time:90000-96000"),
    ("time", {"t_start_ms": 0, "t_end_ms": None}, "v1:time:0-na"),
    ("page", {"page_start": 3, "page_end": 3}, "v1:page:3-3"),
    ("page", {"page_start": 3, "page_end": None}, "v1:page:3-3"),
    ("char", {"char_start": 0, "char_end": 512}, "v1:char:0-512"),
    ("box", {"bbox": [0.1, 0.2, 0.3, 0.4]},
     "v1:box:0.100000,0.200000,0.300000,0.400000"),
    ("time", {"t_start_ms": 1000, "t_end_ms": 2000, "speaker": "Alice Chen"},
     "v1:time:1000-2000:spk=alice chen"),
]


@pytest.mark.parametrize("kind,kwargs,expected", _GOLDEN)
def test_span_key_golden_strings(kind, kwargs, expected):
    """These strings are written into ``media_moments.span_key`` and compared
    by a UNIQUE index, so changing any of them changes stored data.

    If this test fails you have changed a persisted format. Every existing row
    keeps its old key, the unique index cannot tell the two forms apart, and
    re-ingest silently produces duplicate moments. Bump ``SPAN_KEY_VERSION``
    and ship a rewrite migration -- do not just update the expected string.
    """
    assert span_key(kind, **kwargs) == expected, (
        "span_key format changed: this is a persisted data format. "
        "Bump SPAN_KEY_VERSION and write a rewrite migration."
    )


def test_version_prefix_is_present_and_current():
    """Without a version tag, a future format change produces both old- and
    new-form rows for the same span with no way to tell them apart."""
    assert SPAN_KEY_VERSION == "v1"
    assert span_key("whole").startswith(f"{SPAN_KEY_VERSION}:")


# ---------------------------------------------------------------------------
# Speaker is part of the identity
# ---------------------------------------------------------------------------

def test_speaker_distinguishes_overlapping_time_spans():
    """The correction that motivated folding speaker into the key: two
    diarized speakers over one window must not collapse to one row."""
    a = span_key("time", t_start_ms=1000, t_end_ms=2000, speaker="alice")
    b = span_key("time", t_start_ms=1000, t_end_ms=2000, speaker="bob")
    assert a != b


def test_no_speaker_is_byte_identical_to_pre_correction_format():
    """Appending the speaker *last* is what keeps existing speaker-less rows
    valid under the corrected format."""
    assert span_key("time", t_start_ms=1000, t_end_ms=2000) == "v1:time:1000-2000"
    assert ":spk=" not in span_key("time", t_start_ms=1000, t_end_ms=2000)


@pytest.mark.parametrize("speaker", ["  Alice   Chen  ", "ALICE CHEN", "alice chen"])
def test_speaker_slug_is_collapsed_and_casefolded(speaker):
    assert span_key("time", t_start_ms=0, speaker=speaker).endswith(":spk=alice chen")


def test_speaker_slug_is_truncated_to_64_chars():
    slug = span_key("time", t_start_ms=0, speaker="x" * 200).split(":spk=")[1]
    assert len(slug) == 64


def test_whitespace_only_speaker_is_no_speaker():
    assert span_key("time", t_start_ms=0) == span_key("time", t_start_ms=0, speaker="   ")


# ---------------------------------------------------------------------------
# Coercion rules
# ---------------------------------------------------------------------------

def test_bool_is_rejected_as_an_offset_and_falls_back_to_a_digest():
    """``True`` would otherwise render as ``1`` and produce a plausible key for
    a nonsense span. The function stays total, so the rejection shows up as the
    digest form rather than an exception."""
    key = span_key("time", t_start_ms=True)
    assert key.startswith("v1:time:h")
    assert key != "v1:time:1-na"


def test_numeric_strings_are_coerced():
    assert span_key("time", t_start_ms="90000", t_end_ms="96000") == "v1:time:90000-96000"


def test_bbox_floats_are_fixed_point_not_repr():
    """``str(0.1 + 0.2)`` is locale- and repr-sensitive; fixed-point is not."""
    assert span_key("box", bbox=[0.1 + 0.2, 0, 0, 0]).startswith("v1:box:0.300000,")


def test_bbox_accepts_its_json_string_form():
    assert span_key("box", bbox="[0.1, 0.2, 0.3, 0.4]") == span_key(
        "box", bbox=[0.1, 0.2, 0.3, 0.4]
    )


# ---------------------------------------------------------------------------
# Totality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, [], [1, 2], "not json", {"x": 1}, object()])
def test_unparseable_bbox_falls_back_to_a_digest_and_never_raises(bad):
    key = span_key("box", bbox=bad)
    assert key.startswith("v1:box:h")


def test_unknown_span_kind_still_produces_a_stable_key():
    a = span_key("teleport", t_start_ms=5)
    b = span_key("teleport", t_start_ms=5)
    assert a == b and a.startswith("v1:teleport:h")


def test_digest_fallback_is_stable_and_discriminating():
    assert span_key("box", bbox=[1, 2]) == span_key("box", bbox=[1, 2])
    assert span_key("box", bbox=[1, 2]) != span_key("box", bbox=[1, 3])


# ---------------------------------------------------------------------------
# validate_span
# ---------------------------------------------------------------------------

def _values(**kwargs):
    base = {
        "t_start_ms": None, "t_end_ms": None, "page_start": None,
        "page_end": None, "char_start": None, "char_end": None, "bbox": None,
    }
    base.update(kwargs)
    return base


def test_validate_accepts_the_canonical_shapes():
    validate_span("whole", _values())
    validate_span("time", _values(t_start_ms=0, t_end_ms=10))
    validate_span("page", _values(page_start=1))
    validate_span("char", _values(char_start=0, char_end=10))
    validate_span("box", _values(bbox=[0.0, 0.0, 1.0, 1.0]))


def test_foreign_span_columns_are_rejected_not_dropped():
    """A time moment carrying page_start yields a span_key blind to the page,
    so two such moments differing only by page collapse to one key and the
    second is silently swallowed by INSERT OR IGNORE."""
    with pytest.raises(ValueError, match="does not own"):
        validate_span("time", _values(t_start_ms=0, page_start=3))


def test_whole_rejects_every_span_column():
    with pytest.raises(ValueError, match="does not own"):
        validate_span("whole", _values(t_start_ms=0))


def test_missing_required_column_is_rejected():
    with pytest.raises(ValueError, match="requires 't_start_ms'"):
        validate_span("time", _values())
    with pytest.raises(ValueError, match="requires 'char_end'"):
        validate_span("char", _values(char_start=0))


def test_unknown_span_kind_is_rejected_by_the_validator():
    """span_key stays total for a weird kind; the validator does not."""
    with pytest.raises(ValueError, match="unknown span_kind"):
        validate_span("teleport", _values())


@pytest.mark.parametrize("values,message", [
    (_values(t_start_ms=-1), "t_start_ms must be >= 0"),
    (_values(t_start_ms=100, t_end_ms=50), "precedes"),
    (_values(page_start=5, page_end=1), "precedes"),
    (_values(char_start=10, char_end=1), "precedes"),
])
def test_ordering_and_range_are_enforced(values, message):
    kind = "time" if values["t_start_ms"] is not None else (
        "page" if values["page_start"] is not None else "char"
    )
    with pytest.raises(ValueError, match=message):
        validate_span(kind, values)


def test_bbox_must_be_normalized():
    with pytest.raises(ValueError, match="normalized"):
        validate_span("box", _values(bbox=[0, 0, 1920, 1080]))


def test_speaker_is_time_only():
    """RFC 0003 §2.3 gives speaker to time spans; anywhere else there is no
    time axis to attach it to."""
    validate_span("time", _values(t_start_ms=0), speaker="alice")
    with pytest.raises(ValueError, match="only valid on span_kind='time'"):
        validate_span("whole", _values(), speaker="alice")


# ---------------------------------------------------------------------------
# Reference normalization and asset identity
# ---------------------------------------------------------------------------

def test_normalize_lowercases_scheme_and_host_only():
    kind, value = normalize_ref("url", "HTTPS://Example.COM/Path?Q=1#T=90")
    assert (kind, value) == ("url", "https://example.com/Path?Q=1#T=90")


def test_normalize_lowercases_hex_digests():
    assert normalize_ref("sha256", "ABCDEF")[1] == "abcdef"
    assert normalize_ref("blob", "blob://sha256/ABC")[1] == "blob://sha256/abc"


def test_normalize_leaves_file_paths_case_intact():
    """POSIX paths are case-sensitive; folding them would collide two files."""
    assert normalize_ref("file", "  /tmp/Photo.PNG  ")[1] == "/tmp/Photo.PNG"


@pytest.mark.parametrize("kind,value", [("nope", "x"), ("url", ""), ("url", "   ")])
def test_normalize_rejects_bad_input(kind, value):
    with pytest.raises(ValueError):
        normalize_ref(kind, value)


def test_asset_id_includes_ref_kind_in_the_digest():
    """The same ref_value under two kinds is two legal rows (uniqueness is
    ``(ref_kind, ref_value)``); hashing the value alone would collide them on
    the PRIMARY KEY."""
    assert compute_asset_id("url", "abc") != compute_asset_id("file", "abc")


def test_asset_id_prefers_the_hash_form_when_bytes_are_known():
    assert compute_asset_id("url", "http://x/y", content_hash="AB12") == "sha256:ab12"
    assert compute_asset_id("url", "http://x/y").startswith("ref:")


def test_asset_id_is_stable_across_equivalent_references():
    assert compute_asset_id("url", "HTTP://X.COM/a") == compute_asset_id("url", "http://x.com/a")


def test_moment_id_derives_from_the_same_triple_as_the_unique_index():
    a = compute_moment_id("asset-1", "caption", "v1:whole")
    assert a == compute_moment_id("asset-1", "caption", "v1:whole")
    assert a != compute_moment_id("asset-2", "caption", "v1:whole")
    assert a != compute_moment_id("asset-1", "ocr", "v1:whole")
    assert a != compute_moment_id("asset-1", "caption", "v1:time:0-na")
