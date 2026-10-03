"""
Mnemosyne Content Resolver Registry
===================================
Resolves an opaque content URI to metadata or bytes.

Mnemosyne stores *references* to heavy content, never the content itself:
``core/content_sanitizer.py`` extracts oversized or binary-shaped payloads to
a content-addressed blob store and leaves a ``blob://sha256/<hash>`` reference
behind. Until this module existed, that store had no reader at all -- bytes
extracted by the size-cap and high-entropy rules were unreachable by any code
path in the repository.

This module gives references a read side, and generalizes the idea: a resolver
claims one or more URI *schemes* and answers three questions about a URI --
does it exist and what is it (``head``), give me the bytes (``open``), and give
me a short-lived URL a third party can fetch (``presign``).

``presign`` is the load-bearing one. It is how a remote vision provider reaches
bytes that Mnemosyne does not want to read, buffer, or proxy through its own
process. It is also what makes "the archive is a separate product" structurally
true rather than aspirational: the archive serves bytes directly to whoever
needs them, and Mnemosyne only ever passes URIs around.

Absence is a supported state, not an error. When no resolver handles a scheme,
``get_resolver`` returns None and every caller degrades: a client shows a
reference without a preview, and memories recall perfectly, because their text
is local. Nothing raises. This mirrors ``embeddings.available()``.

    Losing the archive costs you previews, not memories.

See RFC 0004 (docs/rfc/0004-archive-boundary.md) for the full contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Iterable, Optional, Protocol

__all__ = [
    "BlobResolver",
    "CallableContentResolver",
    "ContentResolver",
    "ResolvedMeta",
    "clear_content_resolvers",
    "get_resolver",
    "resolve_head",
    "resolve_open",
    "resolve_presign",
    "scheme_of",
    "set_content_resolver",
]


# --- Shapes -----------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedMeta:
    """What a resolver knows about a URI without transferring its bytes.

    ``exists=False`` is a meaningful answer: the resolver recognized the URI
    and the content is gone. That is different from ``head()`` returning None,
    which means "not my scheme". ``mnemosyne doctor`` needs to tell a dead
    reference apart from an unconfigured one.

    Every field except ``exists`` is optional, because a resolver must never
    guess. A blob on disk has no filename and no sidecar, so its mime type is
    either sniffed from magic bytes or left as None.
    """

    exists: bool
    byte_size: Optional[int] = None
    mime: Optional[str] = None
    content_hash: Optional[str] = None
    etag: Optional[str] = None


class ContentResolver(Protocol):
    """Resolves an opaque content URI to metadata or bytes.

    Implementations claim one or more schemes and answer for those only.
    Like :class:`~mnemosyne.core.llm_backends.LLMBackend`, the interface is
    deliberately tiny and failure is a return value rather than an exception.
    """

    name: str
    schemes: frozenset  # e.g. {"blob"} | {"archive"} | {"file"}

    def head(self, uri: str) -> Optional[ResolvedMeta]:
        """Cheap existence and metadata probe.

        None means "not my scheme, or not a URI I can parse".
        ``ResolvedMeta(exists=False)`` means "mine, and gone".
        """
        ...

    def open(self, uri: str) -> Optional[BinaryIO]:
        """Lazy byte stream, or None if unavailable. The caller owns the handle."""
        ...

    def presign(self, uri: str, *, ttl_s: int = 300) -> Optional[str]:
        """A short-lived URL a third party can fetch, or None if unsupported."""
        ...


@dataclass
class CallableContentResolver:
    """Wrap plain functions as a :class:`ContentResolver`. Useful for tests.

    Any method whose function is None returns None, so a resolver that only
    knows how to ``head`` is a legal resolver.
    """

    name: str
    schemes: frozenset
    head_func: Optional[Callable[[str], Optional[ResolvedMeta]]] = None
    open_func: Optional[Callable[[str], Optional[BinaryIO]]] = None
    presign_func: Optional[Callable[..., Optional[str]]] = None

    def head(self, uri: str) -> Optional[ResolvedMeta]:
        return self.head_func(uri) if self.head_func is not None else None

    def open(self, uri: str) -> Optional[BinaryIO]:
        return self.open_func(uri) if self.open_func is not None else None

    def presign(self, uri: str, *, ttl_s: int = 300) -> Optional[str]:
        if self.presign_func is None:
            return None
        return self.presign_func(uri, ttl_s=ttl_s)


# --- URI helpers ------------------------------------------------------------

_SCHEME_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://")


def scheme_of(uri: str) -> Optional[str]:
    """Return the lowercased scheme of a ``scheme://rest`` URI, or None.

    Deliberately stricter than ``urllib.parse``: a bare word, an empty string,
    or a leading ``://`` is not a URI we will dispatch on.
    """
    if not uri or not isinstance(uri, str):
        return None
    m = _SCHEME_RE.match(uri)
    return m.group("scheme").lower() if m else None


# --- BlobResolver -----------------------------------------------------------

# blob://sha256/<64 lowercase hex>. Anything else is not ours.
_BLOB_URI_RE = re.compile(r"^blob://(?P<algo>sha256)/(?P<hash>[0-9a-f]{64})$")

# Magic-byte prefixes, longest-first where prefixes overlap. Deliberately
# short: this exists to answer "is this a PNG" for a preview, not to be a
# general-purpose file identifier.
_MAGIC: tuple = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"%PDF-", "application/pdf"),
    (b"OggS", "audio/ogg"),
    (b"ID3", "audio/mpeg"),
    (b"\xff\xfb", "audio/mpeg"),
    (b"fLaC", "audio/flac"),
    (b"PK\x03\x04", "application/zip"),
    (b"\x1a\x45\xdf\xa3", "video/webm"),
)

_SNIFF_BYTES = 512


def _sniff_mime(head_bytes: bytes) -> Optional[str]:
    """Identify a mime type from magic bytes, or return None.

    Returns None rather than ``application/octet-stream`` on no match. A blob
    path is a hash, so there is no filename to infer from and
    ``mimetypes.guess_type`` would be worse than useless here. An unrecognized
    type is unknown, and saying so is more useful to a caller than a default
    that looks like an answer.
    """
    if not head_bytes:
        return None
    for prefix, mime in _MAGIC:
        if head_bytes.startswith(prefix):
            return mime
    # RIFF containers carry their real type at offset 8.
    if head_bytes.startswith(b"RIFF") and len(head_bytes) >= 12:
        fourcc = head_bytes[8:12]
        if fourcc == b"WEBP":
            return "image/webp"
        if fourcc == b"WAVE":
            return "audio/wav"
        if fourcc == b"AVI ":
            return "video/x-msvideo"
    # ISO base media (mp4/mov/heic) puts 'ftyp' at offset 4.
    if len(head_bytes) >= 12 and head_bytes[4:8] == b"ftyp":
        brand = head_bytes[8:12]
        if brand in (b"heic", b"heix", b"hevc", b"heim"):
            return "image/heic"
        if brand == b"qt  ":
            return "video/quicktime"
        return "video/mp4"
    return None


class BlobResolver:
    """Reader for the content-addressed blob store written by ``content_sanitizer``.

    This is the first reader that store has ever had. Bytes extracted by the
    size-cap rule and the high-entropy rule were previously unreachable.

    ``presign`` always returns None: a local path is not a URL, and inventing a
    ``file://`` one would hand a remote provider something it cannot fetch
    while looking like success.
    """

    name = "blob"
    schemes = frozenset({"blob"})

    def __init__(self, root: Optional[Path] = None) -> None:
        # Stored, not resolved. _blob_root() reads MNEMOSYNE_BLOB_DIR at call
        # time, and tests set that variable after this module is imported --
        # caching the root here would silently resolve against the wrong
        # directory for the life of the process.
        self._root = Path(root) if root is not None else None

    def _blob_root(self) -> Path:
        if self._root is not None:
            return self._root
        from mnemosyne.core.content_sanitizer import _blob_root

        return _blob_root()

    def _path_for(self, uri: str) -> Optional[Path]:
        """Map a blob URI to its on-disk path, or None if the URI is not ours."""
        m = _BLOB_URI_RE.match(uri or "")
        if m is None:
            return None
        digest = m.group("hash")
        root = self._blob_root()
        # Layout must match content_sanitizer._store_blob exactly.
        path = root / digest[:2] / digest[:4] / digest
        # Defense in depth. The 64-hex pattern already makes traversal
        # unreachable; this makes that guarantee explicit rather than implied
        # by a regex somebody may later loosen.
        try:
            if not path.resolve().is_relative_to(root.resolve()):
                return None
        except (OSError, ValueError):
            return None
        return path

    def head(self, uri: str) -> Optional[ResolvedMeta]:
        path = self._path_for(uri)
        if path is None:
            return None
        digest = uri.rsplit("/", 1)[-1]
        try:
            size = path.stat().st_size
        except OSError:
            # Ours, and gone. Distinct from "not mine" above.
            return ResolvedMeta(exists=False, content_hash=digest)
        mime = None
        try:
            with path.open("rb") as fh:
                mime = _sniff_mime(fh.read(_SNIFF_BYTES))
        except OSError:
            pass
        # The store is content-addressed, so the hash IS the strongest possible
        # etag and it costs nothing to provide.
        return ResolvedMeta(
            exists=True,
            byte_size=size,
            mime=mime,
            content_hash=digest,
            etag=digest,
        )

    def open(self, uri: str) -> Optional[BinaryIO]:
        path = self._path_for(uri)
        if path is None:
            return None
        try:
            return path.open("rb")
        except OSError:
            return None

    def presign(self, uri: str, *, ttl_s: int = 300) -> Optional[str]:
        """Always None. A local filesystem path has no presignable URL.

        Callers handle None by falling back to ``open`` and sending bytes
        themselves, or by skipping the remote call entirely. Do not "fix" this
        by returning a ``file://`` URL -- no third party can fetch one.
        """
        return None


# --- Registry ---------------------------------------------------------------

_resolvers: Dict[str, ContentResolver] = {}

# Built-in fallbacks, consulted only when nothing is explicitly registered.
# Without this, nothing would ever construct a BlobResolver and the bug this
# module exists to fix would remain open. Keeping them out of `_resolvers`
# means clear_content_resolvers() restores byte access rather than removing it,
# which matters because tests/conftest.py calls it before every test.
_DEFAULT_RESOLVER_FACTORIES: Dict[str, Callable[[], ContentResolver]] = {
    "blob": BlobResolver,
}


def set_content_resolver(
    resolver: Optional[ContentResolver],
    *,
    schemes: Optional[Iterable[str]] = None,
) -> None:
    """Register (or unregister) a resolver for its declared schemes.

    Hosts call this from their initialize/shutdown hooks. By default the
    resolver claims every scheme in its own ``schemes`` attribute; pass
    ``schemes`` to override. Last registration wins per scheme.

    Pass ``resolver=None`` together with ``schemes`` to unregister those
    schemes, which restores the built-in default for any scheme that has one.
    """
    if resolver is None:
        if schemes is None:
            raise ValueError("set_content_resolver(None) requires schemes=")
        for s in schemes:
            _resolvers.pop(str(s).lower(), None)
        return

    claimed = schemes if schemes is not None else getattr(resolver, "schemes", ()) or ()
    for s in claimed:
        _resolvers[str(s).lower()] = resolver


def get_resolver(scheme_or_uri: str) -> Optional[ContentResolver]:
    """Return the resolver for a scheme (or for a URI's scheme), or None.

    Explicit registrations win; otherwise a built-in default is constructed on
    demand. Returns None for a scheme nobody handles -- the supported absence
    path, not an error.
    """
    if not scheme_or_uri:
        return None
    key = scheme_of(scheme_or_uri) or str(scheme_or_uri).lower()
    found = _resolvers.get(key)
    if found is not None:
        return found
    factory = _DEFAULT_RESOLVER_FACTORIES.get(key)
    return factory() if factory is not None else None


def clear_content_resolvers() -> None:
    """Drop all explicit registrations.

    Built-in defaults survive, so blob access still works afterwards. Tests
    call this between cases so a registration cannot bleed into the next one.
    """
    _resolvers.clear()


# --- Convenience wrappers ---------------------------------------------------
#
# These mirror call_host_llm (llm_backends.py:95): look up, call, swallow
# anything the resolver raises. A resolver talking to a network archive will
# raise eventually, and no caller of recall or doctor should have to care.


def resolve_head(uri: str) -> Optional[ResolvedMeta]:
    """Probe a URI's existence and metadata. None if unresolvable."""
    resolver = get_resolver(uri)
    if resolver is None:
        return None
    try:
        return resolver.head(uri)
    except Exception:
        return None


def resolve_open(uri: str) -> Optional[BinaryIO]:
    """Open a URI's bytes. None if unavailable. The caller owns the handle."""
    resolver = get_resolver(uri)
    if resolver is None:
        return None
    try:
        return resolver.open(uri)
    except Exception:
        return None


def resolve_presign(uri: str, *, ttl_s: int = 300) -> Optional[str]:
    """Get a short-lived third-party-fetchable URL. None if unsupported."""
    resolver = get_resolver(uri)
    if resolver is None:
        return None
    try:
        return resolver.presign(uri, ttl_s=ttl_s)
    except Exception:
        return None
