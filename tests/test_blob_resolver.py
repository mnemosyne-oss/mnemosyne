"""BlobResolver: the first reader the content-addressed blob store ever had.

The round-trip tests are the real regression tests. Before this resolver
existed, `sanitize_content` would extract oversized and high-entropy payloads
to disk and nothing in the repository could ever read them back -- the bytes
were written, referenced in metadata, and permanently unreachable.
"""

import base64
import os

import pytest

from mnemosyne.core import content_sanitizer
from mnemosyne.core.resolvers import BlobResolver, resolve_head, resolve_open


@pytest.fixture(autouse=True)
def blob_dir(tmp_path, monkeypatch):
    """Point the blob store at a temp dir. Read at call time by _blob_root()."""
    root = tmp_path / "blobs"
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(root))
    return root


def _extract(content: str) -> str:
    """Run content through the sanitizer and return its blob:// reference."""
    _stub, meta = content_sanitizer.sanitize_content(content)
    assert meta, "expected sanitize_content to extract this payload"
    return meta["blob_ref"]


# --- round trips through all three extraction branches ----------------------


def test_round_trip_size_cap_branch():
    """Rule 2: content over 1 MB. Previously unreachable once written."""
    original = "x" * (content_sanitizer.SIZE_HARD_CAP + 1)
    ref = _extract(original)

    with resolve_open(ref) as fh:
        assert fh.read().decode("utf-8") == original


def test_round_trip_high_entropy_branch():
    """Rule 3: base64-shaped payload over 100 KB. Also previously unreachable."""
    original = base64.b64encode(os.urandom(200_000)).decode("ascii")
    ref = _extract(original)

    with resolve_open(ref) as fh:
        assert fh.read().decode("utf-8") == original


def test_round_trip_data_uri_branch():
    """Rule 1: a data: URI. Here the blob holds the *decoded* bytes."""
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    ref = _extract("data:image/png;base64," + base64.b64encode(raw).decode("ascii"))

    with resolve_open(ref) as fh:
        assert fh.read() == raw


# --- head() ----------------------------------------------------------------


def test_head_reports_size_hash_and_etag():
    original = "y" * (content_sanitizer.SIZE_HARD_CAP + 1)
    ref = _extract(original)

    meta = resolve_head(ref)
    assert meta is not None
    assert meta.exists is True
    assert meta.byte_size == len(original.encode("utf-8"))
    assert meta.content_hash == ref.rsplit("/", 1)[-1]
    # Content-addressed storage makes the hash the strongest available etag.
    assert meta.etag == meta.content_hash


def test_head_mime_is_none_for_text_blobs():
    """The 'must not guess' rule. A size-cap blob is UTF-8 text with no magic
    bytes and no filename; the honest answer is None, not
    application/octet-stream."""
    ref = _extract("z" * (content_sanitizer.SIZE_HARD_CAP + 1))
    assert resolve_head(ref).mime is None


def test_head_mime_is_sniffed_from_magic_bytes():
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    ref = _extract("data:image/png;base64," + base64.b64encode(raw).decode("ascii"))

    # Sniffed from the bytes on disk -- NOT read back from the mime that
    # sanitize_content recorded in the memory row's metadata, which the blob
    # store has no access to. See the RFC 0004 §2.3 correction.
    assert resolve_head(ref).mime == "image/png"


@pytest.mark.parametrize(
    "raw,expected",
    [
        (b"\xff\xd8\xff\xe0" + b"\x00" * 16, "image/jpeg"),
        (b"GIF89a" + b"\x00" * 16, "image/gif"),
        (b"%PDF-1.7\n" + b"\x00" * 16, "application/pdf"),
        (b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 8, "image/webp"),
        (b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 8, "audio/wav"),
        (b"\x00\x00\x00\x20ftypisom" + b"\x00" * 8, "video/mp4"),
        (b"OggS" + b"\x00" * 16, "audio/ogg"),
        (b"not a known magic number at all", None),
    ],
)
def test_head_mime_sniffing_table(raw, expected):
    ref = "blob://sha256/" + content_sanitizer._store_blob(raw)
    assert resolve_head(ref).mime == expected


def test_head_missing_blob_reports_exists_false_not_none():
    """'Mine, and gone' must be distinguishable from 'not my scheme'.
    mnemosyne doctor needs that distinction to report a dead reference."""
    meta = resolve_head("blob://sha256/" + "a" * 64)
    assert meta is not None
    assert meta.exists is False
    assert meta.content_hash == "a" * 64
    assert meta.byte_size is None


@pytest.mark.parametrize(
    "uri",
    [
        "blob://sha256/tooshort",
        "blob://md5/" + "a" * 64,
        "blob://sha256/" + "A" * 64,  # uppercase hex is not what we write
        "blob://sha256/" + "a" * 63,
        "blob://sha256/" + "g" * 64,  # not hex
        "http://example.com/x",
        "",
    ],
)
def test_head_returns_none_for_uris_that_are_not_ours(uri):
    assert BlobResolver().head(uri) is None
    assert BlobResolver().open(uri) is None


def test_traversal_is_rejected():
    hostile = "blob://sha256/../../../../../../etc/passwd"
    assert BlobResolver().head(hostile) is None
    assert BlobResolver().open(hostile) is None


# --- presign ---------------------------------------------------------------


def test_presign_returns_none():
    """A local path has no presignable URL. Returning a file:// URL would hand
    a remote provider something it cannot fetch, while looking like success."""
    ref = _extract("q" * (content_sanitizer.SIZE_HARD_CAP + 1))
    assert BlobResolver().presign(ref) is None
    assert BlobResolver().presign(ref, ttl_s=900) is None


# --- root resolution -------------------------------------------------------


def test_root_is_read_at_call_time_not_construction(tmp_path, monkeypatch):
    """Constructing the resolver must not freeze MNEMOSYNE_BLOB_DIR. Tests set
    that variable after import, and a cached root would resolve against the
    wrong directory for the life of the process."""
    resolver = BlobResolver()

    later = tmp_path / "later-blobs"
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(later))
    digest = content_sanitizer._store_blob(b"stored after construction")

    with resolver.open("blob://sha256/" + digest) as fh:
        assert fh.read() == b"stored after construction"


def test_explicit_root_overrides_the_environment(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(explicit))
    digest = content_sanitizer._store_blob(b"in the explicit root")

    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "somewhere-else"))
    with BlobResolver(root=explicit).open("blob://sha256/" + digest) as fh:
        assert fh.read() == b"in the explicit root"


def test_layout_matches_store_blob_exactly():
    """If _store_blob's sharding ever changes, this fails loudly rather than
    the resolver quietly returning 'gone' for every blob."""
    digest = content_sanitizer._store_blob(b"layout check")
    expected = content_sanitizer._blob_root() / digest[:2] / digest[:4] / digest
    assert expected.exists()
    assert BlobResolver()._path_for("blob://sha256/" + digest) == expected
