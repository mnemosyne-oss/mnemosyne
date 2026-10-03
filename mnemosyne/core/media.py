"""
Mnemosyne MediaStore — media assets and the moment index (RFC 0003 phase 0)
==========================================================================

Two sidecar tables and one store:

- ``media_assets``  — one row per referenced piece of media. **No BLOB column,
  by design.** Mnemosyne records the *reference* and the *text*; the bytes stay
  wherever they already live.
- ``media_moments`` — one row per semantically tagged span within an asset
  (a caption, a shot, a transcript segment, a page, an OCR region).

A moment is *not* a new kind of memory. Each retained moment is an ordinary
``working_memory`` row, with ``media_moments.memory_id`` bound to it, so it
inherits recall, decay, consolidation, sync, and reindex with no new code.
There is no ``vec_moments`` table and this module never creates one.

Three invariants this module exists to hold:

1. **Idempotency.** ``asset_id`` and ``span_key`` are deterministic, so
   re-ingesting the same media or re-pushing an archive writes nothing new.
   Paired with ``INSERT OR IGNORE`` and the unique indexes, a repeat is a
   no-op rather than a duplicate.
2. **Application-layer integrity.** No schema-level foreign keys (issue #503:
   ``PRAGMA foreign_keys=ON`` broke 22 tests that intentionally create orphan
   rows). ``add_moments`` validates; ``mnemosyne doctor`` reports.
3. **No bytes in the database.** The tables hold references and text. The byte
   path is ``core/content_sanitizer.py`` (write) and ``core/resolvers.py``
   (read); nothing here writes bytes into a column.

:func:`remember_media` at the bottom of this module is the ingest entry point
that ties the store to the provider seam.

See ``docs/rfc/0003-media-moments.md`` for the design and the corrections
this module implements.
"""

import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

#: Bumping this changes every ``span_key`` ever computed. Rows written under an
#: older version remain in the table under their old keys, and the unique index
#: cannot tell the two forms apart — so a bump REQUIRES a rewrite migration.
#: The golden-string tests in ``tests/test_media_span_key.py`` exist to make
#: that a deliberate act rather than an accident.
SPAN_KEY_VERSION = "v1"

REF_KINDS = frozenset({"sha256", "url", "youtube", "file", "blob", "archive"})
MODALITIES = frozenset({"image", "video", "audio", "document"})
MOMENT_KINDS = frozenset({
    "caption", "shot", "transcript", "style", "ocr", "page", "summary",
})
SPAN_KINDS = frozenset({"whole", "time", "page", "char", "box"})

#: ``unavailable`` is rung 4 of the RFC 0002 §3.3 degradation ladder and is a
#: *success* state: the asset is registered and recallable by reference, the
#: provider simply never described it.
UNDERSTANDING_STATUSES = frozenset({
    "pending", "ok", "partial", "unavailable", "refused",
})

#: Every span column, and which ``span_kind`` owns it. A moment carrying a
#: column its kind does not own is rejected rather than silently dropped — see
#: ``validate_span``.
_SPAN_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "whole": (),
    "time": ("t_start_ms", "t_end_ms"),
    "page": ("page_start", "page_end"),
    "char": ("char_start", "char_end"),
    "box": ("bbox",),
}

_ALL_SPAN_COLUMNS: Tuple[str, ...] = (
    "t_start_ms", "t_end_ms", "page_start", "page_end",
    "char_start", "char_end", "bbox",
)

#: Columns without which a span has no identity at all.
_REQUIRED_SPAN_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "whole": (),
    "time": ("t_start_ms",),
    "page": ("page_start",),
    "char": ("char_start", "char_end"),
    "box": ("bbox",),
}

_SPEAKER_MAX_LEN = 64


# ---------------------------------------------------------------------------
# Connection plumbing (mirrors core/canonical.py)
# ---------------------------------------------------------------------------

def _default_db_path() -> Path:
    """Resolved at call time so ``MNEMOSYNE_DATA_DIR`` is honoured without
    importing beam (which would be a circular import)."""
    data_dir = os.environ.get("MNEMOSYNE_DATA_DIR")
    base = Path(data_dir) if data_dir else (Path.home() / ".hermes" / "mnemosyne" / "data")
    return base / "mnemosyne.db"


def _get_conn(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_media(db_path: Optional[Path] = None) -> None:
    """Create the media tables and indexes if absent.

    Idempotent. Safe on databases that already have them. Opens and closes its
    own connection; does not leak file descriptors.
    """
    conn = _get_conn(db_path)
    try:
        _init_media_with_conn(conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _init_media_with_conn(conn: sqlite3.Connection) -> None:
    """Run schema DDL on an existing connection. Caller owns conn lifetime.

    All DDL is ``IF NOT EXISTS``, and ``BeamMemory.__init__`` runs on every
    open, so existing banks acquire these tables on next open with no migration
    module — exactly how ``canonical_facts`` reached existing databases.
    """
    cursor = conn.cursor()

    # No BLOB column anywhere in this table, and that is load-bearing: the
    # brain stores references and text, never media bytes (RFC 0004 invariant 3).
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS media_assets (
            asset_id                TEXT PRIMARY KEY,
            content_hash            TEXT,
            ref_kind                TEXT NOT NULL,
            ref_value               TEXT NOT NULL,
            modality                TEXT NOT NULL,
            mime                    TEXT,
            byte_size               INTEGER,
            duration_ms             INTEGER,
            width                   INTEGER,
            height                  INTEGER,
            page_count              INTEGER,
            title                   TEXT,
            source                  TEXT,
            session_id              TEXT,
            scope                   TEXT,
            captured_at             TEXT,
            captured_at_precision   TEXT DEFAULT 'unknown',
            understanding_status    TEXT DEFAULT 'pending',
            provider                TEXT,
            provider_model          TEXT,
            archive_locator         TEXT,
            metadata                TEXT DEFAULT '{}',
            created_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Identity is the reference...
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_media_ref "
        "ON media_assets(ref_kind, ref_value)"
    )
    # ...and, when the bytes have been seen, the bytes. A PARTIAL unique index
    # (the canonical.py:131-132 trick) is what lets many assets share a NULL
    # content_hash — reference-only assets whose bytes were never fetched —
    # while any *known* hash stays unique.
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_media_hash "
        "ON media_assets(content_hash) WHERE content_hash IS NOT NULL"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_media_modality ON media_assets(modality)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_media_session "
        "ON media_assets(session_id, created_at)"
    )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS media_moments (
            moment_id       TEXT PRIMARY KEY,
            asset_id        TEXT NOT NULL,
            memory_id       TEXT,
            kind            TEXT NOT NULL,
            text            TEXT NOT NULL,
            span_kind       TEXT NOT NULL,
            t_start_ms      INTEGER,
            t_end_ms        INTEGER,
            page_start      INTEGER,
            page_end        INTEGER,
            char_start      INTEGER,
            char_end        INTEGER,
            bbox            TEXT,
            speaker         TEXT,
            confidence      REAL DEFAULT 1.0,
            ordinal         INTEGER,
            span_key        TEXT NOT NULL,
            provider        TEXT,
            provider_model  TEXT,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_moment_asset ON media_moments(asset_id, ordinal)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_moment_time ON media_moments(asset_id, t_start_ms)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_moment_memory ON media_moments(memory_id)"
    )
    # The idempotency contract. Paired with INSERT OR IGNORE, this is what makes
    # re-ingest and archive re-push no-ops instead of duplicates. Same posture
    # as AnnotationStore's (memory_id, kind, value) unique index from #538.
    cursor.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_moment_span "
        "ON media_moments(asset_id, kind, span_key)"
    )

    conn.commit()


# ---------------------------------------------------------------------------
# Pure functions: reference normalization and asset identity
# ---------------------------------------------------------------------------

def normalize_ref(ref_kind: str, ref_value: str) -> Tuple[str, str]:
    """Canonicalize a media reference to its ``(ref_kind, ref_value)`` form.

    Normalization is deliberately conservative — it only folds differences that
    are *definitionally* insignificant, because anything more aggressive would
    make two genuinely different references collide on one ``asset_id``:

    - ``ref_kind`` is stripped and lower-cased, and must be in :data:`REF_KINDS`.
    - Hex digests (``sha256``, ``blob``) are lower-cased.
    - URLs get a lower-cased scheme and host; path, query, and fragment are
      left exactly as given. ``#t=90`` is meaningful for media and is kept.
    - ``file`` paths are only stripped — POSIX paths are case-sensitive, and
      the filesystem is never touched (no ``resolve()``, no ``stat()``).

    Raises ``ValueError`` on an unknown kind or an empty value.
    """
    kind = str(ref_kind or "").strip().lower()
    if kind not in REF_KINDS:
        raise ValueError(
            f"unknown ref_kind {ref_kind!r}; expected one of {sorted(REF_KINDS)}"
        )

    value = str(ref_value or "").strip()
    if not value:
        raise ValueError(f"ref_value must be a non-empty string for ref_kind={kind!r}")

    if kind in {"sha256", "blob"}:
        value = value.lower()
    elif kind == "url":
        parts = urlsplit(value)
        if parts.scheme or parts.netloc:
            value = urlunsplit((
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path,
                parts.query,
                parts.fragment,
            ))

    return kind, value


def compute_asset_id(
    ref_kind: str,
    ref_value: str,
    content_hash: Optional[str] = None,
) -> str:
    """Return the deterministic ``asset_id`` for a reference.

    ``sha256:<hash>`` when the bytes were seen at first registration, otherwise
    ``ref:<sha256(ref_kind + "\\x1f" + normalized_ref_value)>``. Determinism is
    what makes re-ingest and archive re-push idempotent without a lookup round
    trip.

    Two rules the first draft of RFC 0003 omitted, both load-bearing:

    1. **``ref_kind`` is inside the digest.** Uniqueness is ``(ref_kind,
       ref_value)``, so the same value under two kinds is two legal rows —
       hashing the value alone would collide them on the PRIMARY KEY.
    2. **The id is assigned at first registration and never rewritten.** An
       asset first seen by reference and later by bytes keeps its ``ref:`` id;
       the hash arriving late fills in ``content_hash`` via UPDATE. Recomputing
       would orphan every moment already written against the old id. This
       function is therefore only ever called for a *first* registration —
       :meth:`MediaStore.upsert_asset` looks up before it computes.
    """
    kind, value = normalize_ref(ref_kind, ref_value)
    if content_hash:
        return f"sha256:{str(content_hash).strip().lower()}"
    digest = hashlib.sha256(f"{kind}\x1f{value}".encode("utf-8")).hexdigest()
    return f"ref:{digest}"


def compute_moment_id(asset_id: str, kind: str, span_key_value: str) -> str:
    """Deterministic ``moment_id`` derived from the same triple as the unique
    index, so the PRIMARY KEY and ``idx_moment_span`` can never disagree about
    whether a moment is a duplicate."""
    digest = hashlib.sha256(
        f"{asset_id}\x1f{kind}\x1f{span_key_value}".encode("utf-8")
    ).hexdigest()
    return f"m:{digest[:32]}"


# ---------------------------------------------------------------------------
# Pure functions: span identity
# ---------------------------------------------------------------------------

def _as_int(value: Any) -> int:
    """``int()`` that rejects ``bool``.

    Without this, ``True`` renders as ``1`` and a boolean passed where an
    offset belongs produces a plausible-looking key for a nonsense span.
    """
    if isinstance(value, bool):
        raise TypeError("bool is not a valid span offset")
    return int(value)


def _parse_bbox(value: Any) -> List[float]:
    """Coerce a bbox to exactly four floats. Accepts a sequence or its JSON
    string form (the column is TEXT, so both shapes reach us in practice)."""
    raw = value
    if isinstance(raw, (str, bytes)):
        raw = json.loads(raw)
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise TypeError(f"bbox must be a sequence of four numbers, got {type(value)!r}")
    if len(raw) != 4:
        raise ValueError(f"bbox must have exactly four values, got {len(raw)}")
    out: List[float] = []
    for v in raw:
        if isinstance(v, bool):
            raise TypeError("bool is not a valid bbox coordinate")
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):
            raise ValueError("bbox coordinates must be finite")
        out.append(f)
    return out


def _digest_payload(values: Any) -> str:
    """Total fallback payload for anything that cannot be rendered in the
    canonical form. Keeps :func:`span_key` from ever raising — a span with a
    weird shape still gets a stable, collision-resistant identity."""
    try:
        blob = json.dumps(values, sort_keys=True, default=str)
    except Exception:
        blob = repr(values)
    return "h" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _speaker_slug(speaker: Optional[str]) -> str:
    """Whitespace-collapsed, case-folded, truncated to 64 chars. Empty in,
    empty out — an all-whitespace speaker is the same as no speaker."""
    if speaker is None:
        return ""
    return " ".join(str(speaker).split()).casefold()[:_SPEAKER_MAX_LEN]


def span_key(
    span_kind: str,
    *,
    t_start_ms: Optional[int] = None,
    t_end_ms: Optional[int] = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
    bbox: Any = None,
    speaker: Optional[str] = None,
) -> str:
    """Return the canonical **span identity** string for a moment.

    Format, normative::

        span_key := "v1:" <span_kind> [ ":" <payload> ] [ ":spk=" <speaker-slug> ]

    ==============  ===========================================  ==========================================
    ``span_kind``   payload                                      example
    ==============  ===========================================  ==========================================
    ``whole``       omitted                                      ``v1:whole``
    ``time``        ``{t_start_ms}-{t_end_ms or "na"}``          ``v1:time:90000-96000``
    ``page``        ``{page_start}-{page_end or page_start}``    ``v1:page:3-3``
    ``char``        ``{char_start}-{char_end}``                  ``v1:char:0-512``
    ``box``         four floats, ``f"{v:.6f}"``, comma-joined    ``v1:box:0.100000,0.200000,0.300000,0.400000``
    ==============  ===========================================  ==========================================

    Three rules that are not cosmetic:

    - **The ``v1:`` prefix is mandatory.** Without a version tag a future format
      change silently produces both old- and new-form rows for the same span,
      and the unique index cannot tell them apart. With it, the duplication is
      greppable and a rewrite migration can scope itself.
    - **``speaker`` is part of the identity, appended last.** RFC 0003 §2.3 puts
      a diarized audio segment at ``span_kind='time'`` *with* a speaker, so two
      speakers over one window would otherwise produce identical keys and
      ``INSERT OR IGNORE`` would drop the second — silently, and precisely in
      the case diarization exists to handle. Appending last keeps rows without a
      speaker byte-identical to the pre-correction format.
    - **This function is total.** It never raises. Anything unrenderable falls
      back to a ``h<digest>`` payload, so a malformed span still gets a stable
      identity instead of aborting a batch. Rejecting bad input is
      :func:`validate_span`'s job, not this one's.
    """
    kind = str(span_kind or "").strip().lower()

    try:
        if kind == "whole":
            payload = None
        elif kind == "time":
            start = _as_int(t_start_ms)
            end = "na" if t_end_ms is None else str(_as_int(t_end_ms))
            payload = f"{start}-{end}"
        elif kind == "page":
            start = _as_int(page_start)
            end = start if page_end is None else _as_int(page_end)
            payload = f"{start}-{end}"
        elif kind == "char":
            payload = f"{_as_int(char_start)}-{_as_int(char_end)}"
        elif kind == "box":
            # Never str(); it is repr- and locale-sensitive.
            payload = ",".join(f"{v:.6f}" for v in _parse_bbox(bbox))
        else:
            payload = _digest_payload({
                "span_kind": span_kind,
                "t_start_ms": t_start_ms, "t_end_ms": t_end_ms,
                "page_start": page_start, "page_end": page_end,
                "char_start": char_start, "char_end": char_end,
                "bbox": bbox,
            })
    except Exception:
        payload = _digest_payload({
            "span_kind": span_kind,
            "t_start_ms": t_start_ms, "t_end_ms": t_end_ms,
            "page_start": page_start, "page_end": page_end,
            "char_start": char_start, "char_end": char_end,
            "bbox": bbox,
        })

    key = f"{SPAN_KEY_VERSION}:{kind}" if payload is None else f"{SPAN_KEY_VERSION}:{kind}:{payload}"

    slug = _speaker_slug(speaker)
    if slug:
        key = f"{key}:spk={slug}"
    return key


def validate_span(
    span_kind: str,
    values: Dict[str, Any],
    speaker: Optional[str] = None,
) -> None:
    """Raise ``ValueError``/``TypeError`` if a span is not writable.

    SQLite gives no guarantee that a ``time`` moment has ``t_start_ms``
    populated (RFC 0003 §2.3 accepts this), so the guarantee lives here,
    consistent with the application-layer integrity posture in §1.6.

    The strict rule worth naming: **columns foreign to the span kind must be
    ``None``.** A ``time`` moment that also carries ``page_start`` produces a
    ``span_key`` blind to the page, so two such moments differing only by page
    collapse to one key and the second is silently dropped by ``INSERT OR
    IGNORE``. Rejecting is the only option that is not lossy.
    """
    kind = str(span_kind or "").strip().lower()
    if kind not in SPAN_KINDS:
        raise ValueError(
            f"unknown span_kind {span_kind!r}; expected one of {sorted(SPAN_KINDS)}"
        )

    owned = set(_SPAN_COLUMNS[kind])
    foreign = [
        col for col in _ALL_SPAN_COLUMNS
        if col not in owned and values.get(col) is not None
    ]
    if foreign:
        raise ValueError(
            f"span_kind={kind!r} does not own {sorted(foreign)}; "
            "a span carrying foreign columns yields a span_key blind to them, "
            "so near-duplicate moments would be silently dropped. Clear them "
            "or use the matching span_kind."
        )

    for col in _REQUIRED_SPAN_COLUMNS[kind]:
        if values.get(col) is None:
            raise ValueError(f"span_kind={kind!r} requires {col!r}")

    if kind == "time":
        start = _as_int(values["t_start_ms"])
        if start < 0:
            raise ValueError("t_start_ms must be >= 0")
        if values.get("t_end_ms") is not None:
            end = _as_int(values["t_end_ms"])
            if end < start:
                raise ValueError(f"t_end_ms ({end}) precedes t_start_ms ({start})")
    elif kind == "page":
        start = _as_int(values["page_start"])
        if start < 0:
            raise ValueError("page_start must be >= 0")
        if values.get("page_end") is not None:
            end = _as_int(values["page_end"])
            if end < start:
                raise ValueError(f"page_end ({end}) precedes page_start ({start})")
    elif kind == "char":
        start = _as_int(values["char_start"])
        end = _as_int(values["char_end"])
        if start < 0:
            raise ValueError("char_start must be >= 0")
        if end < start:
            raise ValueError(f"char_end ({end}) precedes char_start ({start})")
    elif kind == "box":
        coords = _parse_bbox(values["bbox"])
        if any(c < 0.0 or c > 1.0 for c in coords):
            raise ValueError(f"bbox coordinates must be normalized to [0, 1], got {coords}")

    # RFC 0003 §2.3 gives `speaker` to time spans only. Permitting it elsewhere
    # would put a speaker in the span_key of a span that has no time axis to
    # attach it to.
    if _speaker_slug(speaker) and kind != "time":
        raise ValueError(f"speaker is only valid on span_kind='time', not {kind!r}")


# ---------------------------------------------------------------------------
# Drafts and results
# ---------------------------------------------------------------------------

@dataclass
class MomentDraft:
    """An unwritten moment. Providers produce these; :meth:`MediaStore.add_moments`
    validates and writes them."""

    kind: str
    text: str
    span_kind: str = "whole"
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    bbox: Any = None
    speaker: Optional[str] = None
    confidence: float = 1.0
    ordinal: Optional[int] = None
    memory_id: Optional[str] = None
    provider: Optional[str] = None
    provider_model: Optional[str] = None

    def span_values(self) -> Dict[str, Any]:
        return {col: getattr(self, col) for col in _ALL_SPAN_COLUMNS}

    def span_key(self) -> str:
        return span_key(
            self.span_kind,
            t_start_ms=self.t_start_ms,
            t_end_ms=self.t_end_ms,
            page_start=self.page_start,
            page_end=self.page_end,
            char_start=self.char_start,
            char_end=self.char_end,
            bbox=self.bbox,
            speaker=self.speaker,
        )


@dataclass(frozen=True)
class AssetUpsert:
    """Result of :meth:`MediaStore.upsert_asset`.

    ``status`` is one of:

    - ``created``  — a new asset row
    - ``updated``  — an existing row matched by reference and gained new detail
    - ``unchanged``— an existing row matched by reference, nothing to add
    - ``aliased``  — an existing row matched by *content hash* under a different
      reference; the new reference was recorded in ``metadata.alt_refs``
    """

    asset_id: str
    status: str


@dataclass(frozen=True)
class MomentWriteResult:
    """Result of :meth:`MediaStore.add_moments`."""

    inserted: int = 0
    #: Moments whose (kind, span_key) already existed for this asset. A repeat
    #: ingest lands entirely here — that is the idempotency contract working.
    duplicate: int = 0
    #: Moments written with memory_id NULL because the referenced
    #: working_memory row was gone. Legitimate: eviction and consolidation both
    #: happen between drafting and insert.
    unbound: int = 0
    moment_ids: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_ASSET_DETAIL_COLUMNS: Tuple[str, ...] = (
    "content_hash", "mime", "byte_size", "duration_ms", "width", "height",
    "page_count", "title", "source", "session_id", "scope", "captured_at",
    "captured_at_precision", "provider", "provider_model", "archive_locator",
)


class MediaStore:
    """Handle over ``media_assets`` and ``media_moments``.

    Example::

        >>> store = MediaStore(db_path=path)
        >>> up = store.upsert_asset(ref_kind="file", ref_value="/tmp/a.png",
        ...                         modality="image")
        >>> store.add_moments(up.asset_id, [
        ...     MomentDraft(kind="caption", text="a terminal window",
        ...                 span_kind="whole"),
        ... ]).inserted
        1
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        conn: Optional[sqlite3.Connection] = None,
    ):
        """When ``conn`` is provided the store reuses it — this is how
        ``BeamMemory`` shares its thread-local connection, avoiding a per-call
        file-descriptor cost. The caller owns that connection's lifetime.
        When ``conn`` is None the store opens its own."""
        self.db_path = db_path or _default_db_path()
        if conn is not None:
            self.conn = conn
            _init_media_with_conn(conn)
        else:
            init_media(self.db_path)
            self.conn = _get_conn(self.db_path)

    # ------------------------------------------------------------------
    # Assets
    # ------------------------------------------------------------------

    def upsert_asset(
        self,
        *,
        ref_kind: str,
        ref_value: str,
        modality: str,
        content_hash: Optional[str] = None,
        mime: Optional[str] = None,
        byte_size: Optional[int] = None,
        duration_ms: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        page_count: Optional[int] = None,
        title: Optional[str] = None,
        source: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: Optional[str] = None,
        captured_at: Optional[str] = None,
        captured_at_precision: str = "unknown",
        understanding_status: str = "pending",
        provider: Optional[str] = None,
        provider_model: Optional[str] = None,
        archive_locator: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AssetUpsert:
        """Register (or re-register) a media asset. Idempotent.

        Resolution order, which is also the reason ``compute_asset_id`` is only
        ever reached for a genuinely new asset:

        1. **By reference.** ``(ref_kind, ref_value)`` already present → return
           that ``asset_id`` and fill in any detail column that is currently
           NULL. **Existing non-NULL values are never overwritten**, so a
           re-registration cannot downgrade what we already know.
        2. **By content hash.** A different reference resolving to the same
           known bytes → return the existing ``asset_id`` and record the new
           reference under ``metadata.alt_refs``. One file reachable by both a
           URL and a local path is one asset, not two, and certainly not an
           ``IntegrityError``.
        3. **New row**, with a deterministic id.

        ``asset_id`` is assigned once and never rewritten. A ``content_hash``
        arriving later fills the column via UPDATE and leaves the id alone —
        rewriting it would orphan every moment already bound to the asset.
        """
        kind, value = normalize_ref(ref_kind, ref_value)

        modality_norm = str(modality or "").strip().lower()
        if modality_norm not in MODALITIES:
            raise ValueError(
                f"unknown modality {modality!r}; expected one of {sorted(MODALITIES)}"
            )

        status_norm = str(understanding_status or "pending").strip().lower()
        if status_norm not in UNDERSTANDING_STATUSES:
            raise ValueError(
                f"unknown understanding_status {understanding_status!r}; "
                f"expected one of {sorted(UNDERSTANDING_STATUSES)}"
            )

        hash_norm = str(content_hash).strip().lower() if content_hash else None

        detail = {
            "content_hash": hash_norm,
            "mime": mime,
            "byte_size": byte_size,
            "duration_ms": duration_ms,
            "width": width,
            "height": height,
            "page_count": page_count,
            "title": title,
            "source": source,
            "session_id": session_id,
            "scope": scope,
            "captured_at": captured_at,
            "captured_at_precision": captured_at_precision,
            "provider": provider,
            "provider_model": provider_model,
            "archive_locator": archive_locator,
        }

        cursor = self.conn.cursor()

        # (1) Known reference.
        row = cursor.execute(
            "SELECT * FROM media_assets WHERE ref_kind = ? AND ref_value = ?",
            (kind, value),
        ).fetchone()
        if row is not None:
            changed = self._fill_asset_detail(row, detail)
            return AssetUpsert(asset_id=row["asset_id"],
                               status="updated" if changed else "unchanged")

        # (2) Known bytes under a new reference.
        if hash_norm:
            row = cursor.execute(
                "SELECT * FROM media_assets WHERE content_hash = ?", (hash_norm,)
            ).fetchone()
            if row is not None:
                self._record_alt_ref(row, kind, value)
                self._fill_asset_detail(row, detail)
                return AssetUpsert(asset_id=row["asset_id"], status="aliased")

        # (3) New asset.
        asset_id = compute_asset_id(kind, value, hash_norm)
        meta_json = json.dumps(metadata or {}, default=str)
        cursor.execute(
            """
            INSERT INTO media_assets (
                asset_id, content_hash, ref_kind, ref_value, modality, mime,
                byte_size, duration_ms, width, height, page_count, title,
                source, session_id, scope, captured_at, captured_at_precision,
                understanding_status, provider, provider_model, archive_locator,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id, hash_norm, kind, value, modality_norm, mime,
                byte_size, duration_ms, width, height, page_count, title,
                source, session_id, scope, captured_at, captured_at_precision,
                status_norm, provider, provider_model, archive_locator,
                meta_json,
            ),
        )
        self.conn.commit()
        return AssetUpsert(asset_id=asset_id, status="created")

    def _fill_asset_detail(self, row: sqlite3.Row, detail: Dict[str, Any]) -> bool:
        """Fill NULL detail columns on an existing asset. Returns whether
        anything changed. Non-NULL values are left alone: a second sighting of
        an asset can add knowledge but must never remove it."""
        updates = {
            col: val for col, val in detail.items()
            if val is not None and col in _ASSET_DETAIL_COLUMNS and row[col] is None
        }
        # 'unknown' is the column default, not real information.
        if updates.get("captured_at_precision") == "unknown":
            updates.pop("captured_at_precision", None)
        if not updates:
            return False
        assignments = ", ".join(f"{col} = ?" for col in updates)
        self.conn.execute(
            f"UPDATE media_assets SET {assignments} WHERE asset_id = ?",
            (*updates.values(), row["asset_id"]),
        )
        self.conn.commit()
        return True

    def _record_alt_ref(self, row: sqlite3.Row, kind: str, value: str) -> None:
        """Append a second reference to the same bytes under
        ``metadata.alt_refs``, deduped."""
        try:
            meta = json.loads(row["metadata"] or "{}")
            if not isinstance(meta, dict):
                meta = {}
        except Exception:
            meta = {}
        alts = meta.get("alt_refs")
        if not isinstance(alts, list):
            alts = []
        entry = {"ref_kind": kind, "ref_value": value}
        if entry not in alts:
            alts.append(entry)
        meta["alt_refs"] = alts
        self.conn.execute(
            "UPDATE media_assets SET metadata = ? WHERE asset_id = ?",
            (json.dumps(meta, default=str), row["asset_id"]),
        )
        self.conn.commit()

    def get_asset(self, asset_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM media_assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        return dict(row) if row else None

    def find_asset_by_ref(self, ref_kind: str, ref_value: str) -> Optional[Dict[str, Any]]:
        kind, value = normalize_ref(ref_kind, ref_value)
        row = self.conn.execute(
            "SELECT * FROM media_assets WHERE ref_kind = ? AND ref_value = ?",
            (kind, value),
        ).fetchone()
        return dict(row) if row else None

    def set_understanding_status(
        self,
        asset_id: str,
        status: str,
        *,
        provider: Optional[str] = None,
        provider_model: Optional[str] = None,
    ) -> bool:
        """Move an asset along the degradation ladder. Returns whether a row
        was updated."""
        status_norm = str(status or "").strip().lower()
        if status_norm not in UNDERSTANDING_STATUSES:
            raise ValueError(
                f"unknown understanding_status {status!r}; "
                f"expected one of {sorted(UNDERSTANDING_STATUSES)}"
            )
        assignments = ["understanding_status = ?"]
        params: List[Any] = [status_norm]
        if provider is not None:
            assignments.append("provider = ?")
            params.append(provider)
        if provider_model is not None:
            assignments.append("provider_model = ?")
            params.append(provider_model)
        params.append(asset_id)
        cursor = self.conn.execute(
            f"UPDATE media_assets SET {', '.join(assignments)} WHERE asset_id = ?",
            tuple(params),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Moments
    # ------------------------------------------------------------------

    def add_moments(
        self,
        asset_id: str,
        drafts: Sequence[MomentDraft],
    ) -> MomentWriteResult:
        """Write moments for an asset. Idempotent, all-or-nothing on validation.

        **Everything is validated before anything is written.** A batch with one
        bad draft writes zero rows and raises, rather than leaving a partial
        span index behind that nothing knows is partial.

        Three checks that are not obvious:

        - *Foreign span columns are rejected*, not dropped — see
          :func:`validate_span`.
        - *Intra-batch duplicates are rejected*, naming both indices. Two drafts
          with the same ``(kind, span_key)`` would otherwise have the second
          silently swallowed by ``INSERT OR IGNORE``, which is right for a
          *repeat ingest* and wrong for a *single batch* that believes it is
          writing two distinct moments.
        - *A ``memory_id`` with no live ``working_memory`` row is set NULL and
          counted, not raised.* Eviction (``_trim_working_memory``) and
          consolidation both legitimately remove the row between drafting and
          insert, and ``media_moments.text`` remains authoritative either way.

        Raises ``ValueError`` if the asset does not exist — a moment on a
        non-existent asset is the one orphan kind that is real corruption.
        """
        if not drafts:
            return MomentWriteResult()

        if self.get_asset(asset_id) is None:
            raise ValueError(
                f"no media_assets row for asset_id={asset_id!r}; "
                "register the asset before writing moments"
            )

        prepared: List[Tuple[MomentDraft, str, str]] = []
        seen: Dict[Tuple[str, str], int] = {}

        for index, draft in enumerate(drafts):
            kind = str(draft.kind or "").strip().lower()
            if kind not in MOMENT_KINDS:
                raise ValueError(
                    f"drafts[{index}]: unknown kind {draft.kind!r}; "
                    f"expected one of {sorted(MOMENT_KINDS)}"
                )
            text = str(draft.text or "").strip()
            if not text:
                raise ValueError(f"drafts[{index}]: text must be a non-empty string")

            try:
                validate_span(draft.span_kind, draft.span_values(), draft.speaker)
            except (ValueError, TypeError) as exc:
                raise type(exc)(f"drafts[{index}]: {exc}") from exc

            key = draft.span_key()
            if (kind, key) in seen:
                raise ValueError(
                    f"drafts[{index}] duplicates drafts[{seen[(kind, key)]}]: "
                    f"both resolve to kind={kind!r} span_key={key!r}. "
                    "The unique index would silently drop the second; "
                    "deduplicate the batch or give them distinct spans."
                )
            seen[(kind, key)] = index
            prepared.append((draft, kind, key))

        result_ids: List[str] = []
        inserted = duplicate = unbound = 0
        cursor = self.conn.cursor()

        for draft, kind, key in prepared:
            moment_id = compute_moment_id(asset_id, kind, key)
            memory_id = draft.memory_id
            if memory_id is not None and not self._memory_exists(memory_id):
                memory_id = None
                unbound += 1

            bbox_text = None
            if draft.bbox is not None:
                bbox_text = json.dumps(
                    [float(f"{v:.6f}") for v in _parse_bbox(draft.bbox)]
                )

            cursor.execute(
                """
                INSERT OR IGNORE INTO media_moments (
                    moment_id, asset_id, memory_id, kind, text, span_kind,
                    t_start_ms, t_end_ms, page_start, page_end,
                    char_start, char_end, bbox, speaker, confidence, ordinal,
                    span_key, provider, provider_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    moment_id, asset_id, memory_id, kind, draft.text,
                    str(draft.span_kind).strip().lower(),
                    draft.t_start_ms, draft.t_end_ms,
                    draft.page_start, draft.page_end,
                    draft.char_start, draft.char_end,
                    bbox_text, draft.speaker, draft.confidence, draft.ordinal,
                    key, draft.provider, draft.provider_model,
                ),
            )
            if cursor.rowcount > 0:
                inserted += 1
            else:
                duplicate += 1
            result_ids.append(moment_id)

        self.conn.commit()
        return MomentWriteResult(
            inserted=inserted,
            duplicate=duplicate,
            unbound=unbound,
            moment_ids=result_ids,
        )

    def bind_moment_memory(self, moment_id: str, memory_id: str) -> bool:
        """Bind a moment to its memory row, guarded by existence rather than an
        FK (issue #503). Returns whether the binding was written.

        This is the second half of the two-step ingest ordering: the moment row
        is written first, then the memory row, then this. A crash between them
        leaves a recoverable unbound moment that ``doctor`` counts; the reverse
        order would leak a memory row nothing can trace back.
        """
        if not self._memory_exists(memory_id):
            return False
        cursor = self.conn.execute(
            "UPDATE media_moments SET memory_id = ? WHERE moment_id = ?",
            (memory_id, moment_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def get_moments(
        self,
        asset_id: str,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM media_moments WHERE asset_id = ?"
        params: List[Any] = [asset_id]
        if kind:
            sql += " AND kind = ?"
            params.append(str(kind).strip().lower())
        sql += " ORDER BY ordinal IS NULL, ordinal, t_start_ms, moment_id"
        return [dict(r) for r in self.conn.execute(sql, tuple(params)).fetchall()]

    def moments_for_memory(self, memory_id: str) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM media_moments WHERE memory_id = ?", (memory_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Integrity
    # ------------------------------------------------------------------

    def count_orphans(self) -> Dict[str, int]:
        """Count both orphan kinds. **They are different health states.**

        ``asset_orphans`` — a moment whose asset is gone. Real corruption;
        nothing in the design should ever produce it.

        ``memory_orphans`` — a moment whose memory row no longer resolves. The
        *expected steady state*: consolidation summarizes the memory row away
        and ``media_moments.text`` remains authoritative, and working memory is
        trimmed on a cap. Reporting these as a problem would make the check cry
        wolf at every healthy media user after their first ``sleep()``, which
        trains people to ignore the check that catches the real corruption.
        """
        asset_orphans = self.conn.execute(
            "SELECT COUNT(*) FROM media_moments m "
            "WHERE NOT EXISTS (SELECT 1 FROM media_assets a WHERE a.asset_id = m.asset_id)"
        ).fetchone()[0]

        memory_orphans = 0
        if self._has_table("working_memory"):
            memory_orphans = self.conn.execute(
                "SELECT COUNT(*) FROM media_moments m WHERE m.memory_id IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM working_memory w WHERE w.id = m.memory_id)"
            ).fetchone()[0]

        return {"asset_orphans": asset_orphans, "memory_orphans": memory_orphans}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _has_table(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?", (name,)
        ).fetchone()
        return row is not None

    def _memory_exists(self, memory_id: str) -> bool:
        """Existence guard in place of an FK.

        When ``working_memory`` is absent entirely — a standalone MediaStore on
        a database that is not a Beam bank — the binding cannot be verified, so
        it is preserved as given rather than silently nulled. ``doctor`` will
        count it if it is genuinely dangling.
        """
        if not self._has_table("working_memory"):
            return True
        row = self.conn.execute(
            "SELECT 1 FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()
        return row is not None


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------
#
# ``remember_media`` is a free function taking ``beam`` rather than a method,
# following ``_extract_and_store_entities`` (``beam.py:1561``). ``beam.py`` is
# already ~8,000 lines; this keeps its growth to a short delegating method and
# makes the flow unit-testable against a stub.

#: Refuse to read more than this into memory for a describe call. A four-gigabyte
#: video is not something to base64 into a JSON body, and the failure should be a
#: clean "unavailable" rather than an OOM.
MAX_FETCH_BYTES = 20 * 1024 * 1024

#: A reference longer than this is not a reference. Chiefly a guard against a
#: caller routing an inline payload here after `data:` conversion has already
#: run -- ``ref_value`` and ``metadata`` are uncapped TEXT columns that
#: ``sanitize_content`` never inspects.
MAX_REF_LENGTH = 8192

_EXTENSION_MODALITY: Dict[str, str] = {
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".gif": "image",
    ".webp": "image", ".bmp": "image", ".tiff": "image", ".heic": "image",
    ".svg": "image",
    ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video",
    ".avi": "video", ".m4v": "video",
    ".mp3": "audio", ".wav": "audio", ".flac": "audio", ".ogg": "audio",
    ".m4a": "audio", ".aac": "audio", ".opus": "audio",
    ".pdf": "document", ".docx": "document", ".pptx": "document",
    ".epub": "document", ".txt": "document", ".md": "document",
}


@dataclass(frozen=True)
class MediaIngestResult:
    """Outcome of :func:`remember_media`.

    A bare ``str`` return would discard the thing the caller most needs: the
    degradation ladder produces a *status*, and ``unavailable`` is a success.
    """

    asset_id: str
    #: The reference memory for an admitted asset. Present even when no provider ran.
    anchor_memory_id: Optional[str] = None
    #: One of media.UNDERSTANDING_STATUSES, or ``filtered`` when admission
    #: rejects the caller's reference before an asset exists.
    status: str = "pending"
    moment_ids: List[str] = field(default_factory=list)
    #: Memory rows created for moments, in the same order.
    memory_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def described(self) -> bool:
        return self.status in {"ok", "partial"}


def normalize_media_input(
    ref: str,
    ref_kind: Optional[str] = None,
    mime: Optional[str] = None,
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """Resolve caller input to ``(ref_kind, ref_value, mime, content_hash)``.

    **A ``data:`` URI is converted here, at the door, before anything is
    stored.** This is not a tidiness preference. ``sanitize_content`` only ever
    inspects ``content`` (``beam.py:3251``, ``:3523``, ``memory.py:358``) --
    ``metadata_json`` is never sanitized, and neither is ``media_assets.
    ref_value``. A ``data:`` URI arriving as a *reference* would therefore write
    megabytes of base64 into two uncapped TEXT columns, with no size cap and no
    entropy check: exactly the outcome those branches exist to prevent, reached
    by a path that does not watch for it. Converting on entry means the bytes go
    to the blob store and only a short ``blob://`` reference is ever persisted.
    """
    if not ref or not str(ref).strip():
        raise ValueError("ref must be a non-empty string")
    ref = str(ref).strip()

    from mnemosyne.core import content_sanitizer

    if content_sanitizer._is_data_uri(ref):
        parsed = content_sanitizer._parse_data_uri(ref)
        if parsed is None:
            raise ValueError("ref looks like a data: URI but could not be parsed")
        data_mime, raw = parsed
        digest = content_sanitizer._store_blob(raw)
        return ("blob", f"blob://sha256/{digest}", mime or data_mime, digest)

    if len(ref) > MAX_REF_LENGTH:
        raise ValueError(
            f"ref is {len(ref)} characters, over the {MAX_REF_LENGTH} limit; "
            "inline payloads must arrive as a data: URI so they can be "
            "converted to a blob reference rather than stored as text"
        )

    if ref_kind:
        kind = str(ref_kind).strip().lower()
    elif ref.startswith("blob://"):
        kind = "blob"
    elif ref.startswith(("http://", "https://")):
        kind = "youtube" if _is_youtube(ref) else "url"
    elif len(ref) == 64 and all(c in "0123456789abcdefABCDEF" for c in ref):
        kind = "sha256"
    else:
        kind = "file"

    kind, value = normalize_ref(kind, ref)

    content_hash = None
    if kind == "sha256":
        content_hash = value
    elif kind == "blob":
        match = re.match(r"^blob://sha256/([0-9a-f]{64})$", value)
        if match:
            content_hash = match.group(1)

    return (kind, value, mime, content_hash)


def _is_youtube(ref: str) -> bool:
    host = (urlsplit(ref).netloc or "").lower()
    return host.endswith(("youtube.com", "youtu.be")) or host.endswith(".youtube.com")


def infer_modality(
    ref_kind: str,
    ref_value: str,
    mime: Optional[str] = None,
) -> str:
    """Explicit mime wins, then the file extension, then ``document``.

    ``document`` is the fallback rather than ``image`` because it is the least
    committal: a document moment is a page or a character range, which degrades
    to ``whole`` harmlessly, whereas guessing ``image`` invites a vision call on
    something that is not one.
    """
    if mime:
        top = str(mime).split("/", 1)[0].strip().lower()
        if top in {"image", "video", "audio"}:
            return top
        if top == "application" or top == "text":
            return "document"

    if ref_kind == "youtube":
        return "video"

    suffix = Path(urlsplit(ref_value).path or ref_value).suffix.lower()
    return _EXTENSION_MODALITY.get(suffix, "document")


def _short_ref(ref_kind: str, ref_value: str, limit: int = 120) -> str:
    """A human-readable stub for the anchor row's generated content.

    The full reference lives in ``media_assets.ref_value``; the memory row gets
    something a person would recognize in a recall result.
    """
    if ref_kind in {"blob", "sha256"}:
        return ref_value[-12:]
    name = Path(urlsplit(ref_value).path or ref_value).name
    return (name or ref_value)[:limit]


def _make_fetch(ref_kind: str, ref_value: str):
    """Build the lazy byte-reader handed to the provider.

    It is only ever *called* for content the provider cannot fetch itself, and
    it is wired from the ``core/resolvers.py`` registry so a host that registers
    an S3 or archive resolver gets remote media for free. A plain local path has
    no registered scheme, so it is read directly -- reading a file the user
    explicitly named is the operation they asked for, and it is still gated
    behind ``modality_enabled``.
    """
    def _fetch() -> Optional[bytes]:
        try:
            if ref_kind == "file":
                path = Path(ref_value)
                if not path.is_file():
                    return None
                if path.stat().st_size > MAX_FETCH_BYTES:
                    logger.info("media: %s exceeds the fetch cap; skipping", ref_kind)
                    return None
                return path.read_bytes()

            from mnemosyne.core.resolvers import resolve_open

            stream = resolve_open(ref_value)
            if stream is None:
                return None
            with stream:
                raw = stream.read(MAX_FETCH_BYTES + 1)
            if raw is not None and len(raw) > MAX_FETCH_BYTES:
                return None
            return raw
        except Exception:
            logger.info("media: fetch failed", exc_info=True)
            return None

    return _fetch


def remember_media(
    beam,
    *,
    ref: str,
    ref_kind: Optional[str] = None,
    modality: Optional[str] = None,
    mime: Optional[str] = None,
    title: Optional[str] = None,
    source: str = "media",
    hint: Optional[str] = None,
    max_moments: Optional[int] = None,
    importance: float = 0.5,
    scope: str = "session",
    captured_at: Optional[str] = None,
    captured_at_precision: str = "unknown",
    metadata: Optional[Dict[str, Any]] = None,
    _write_kind: object = "public",
    _write_policy=None,
) -> MediaIngestResult:
    """Register a piece of media and, if a provider is configured, describe it.

    Synchronous by design. RFC 0003 §5.1: ``core/`` has no access lock --
    ``_beam_access_lock`` lives only in the Hermes provider -- so a background
    thread here would run unserialized against SQLite, which is the shape of the
    WAL-checkpoint crash in #498/#520.

    The order of operations is the design, and each step is placed where it is
    for a stated reason:

    1. Admit the raw caller reference. A rejection returns before any blob or
       database write.
    2. Normalize the reference. ``data:`` becomes ``blob://`` here, before
       storage (see :func:`normalize_media_input`).
    3. Infer the modality.
    4. Upsert the asset as ``pending`` **and commit**, so a process killed
       mid-ingest leaves a retryable state rather than one nothing can be in.
    5. Write the anchor memory row **before the provider gate**, so rung 4 of
       the degradation ladder gives the user strictly more than they had. With
       no provider configured this function still succeeds and still leaves
       something recallable.
    6. Ask the provider, if and only if the operator has opted in.
    7. Per moment: **the moment row first**, then the memory row, then the bind.
       A crash between them leaves a recoverable unbound moment that ``doctor``
       counts; the reverse order leaks a memory row nothing can trace back.
    8. Set the final status in a ``finally``, so a constraint error mid-loop
       cannot strand the asset at ``pending``.

    Two rules keep the reference out of ``content``: ``ref`` is always a
    parameter and never content, and the anchor row's text is *generated*
    (``"Referenced image: photo.png"``), with the full reference living only in
    ``media_assets.ref_value``.
    """
    # Admit the caller's raw reference before data-URI normalization can write
    # a blob, and keep every derived write on this operation's immutable policy
    # snapshot. Internal exemptions require an opaque in-process capability;
    # caller-supplied strings such as "restore" cannot bypass admission.
    from mnemosyne.core.filters import admit_memory_write, current_write_policy

    write_policy = _write_policy or current_write_policy()
    if any(
        not admit_memory_write(
            value, write_kind=_write_kind, policy=write_policy
        )[0]
        for value in (ref, title)
        if value
    ):
        return MediaIngestResult(asset_id="", status="filtered")

    store = getattr(beam, "media", None)
    if store is None:
        store = MediaStore(db_path=getattr(beam, "db_path", None),
                           conn=getattr(beam, "conn", None))

    ref_kind_norm, ref_value, mime_norm, content_hash = normalize_media_input(
        ref, ref_kind, mime
    )
    modality_norm = (str(modality).strip().lower() if modality
                     else infer_modality(ref_kind_norm, ref_value, mime_norm))
    if modality_norm not in MODALITIES:
        raise ValueError(
            f"unknown modality {modality!r}; expected one of {sorted(MODALITIES)}"
        )

    upsert = store.upsert_asset(
        ref_kind=ref_kind_norm,
        ref_value=ref_value,
        modality=modality_norm,
        content_hash=content_hash,
        mime=mime_norm,
        title=title,
        source=source,
        session_id=getattr(beam, "session_id", None),
        scope=scope,
        captured_at=captured_at,
        captured_at_precision=captured_at_precision,
        understanding_status="pending",
        metadata=metadata,
    )
    asset_id = upsert.asset_id

    warnings: List[str] = []
    moment_ids: List[str] = []
    memory_ids: List[str] = []
    status = "unavailable"

    try:
        anchor_memory_id = _ensure_anchor_memory(
            beam, store, asset_id, ref_kind_norm, ref_value, modality_norm,
            title=title, source=source, importance=importance, scope=scope,
            write_kind=_write_kind, write_policy=write_policy,
        )

        result = _describe(
            asset_id, ref_kind_norm, ref_value, modality_norm, mime_norm,
            content_hash, hint, max_moments,
        )
        if result is None:
            return MediaIngestResult(
                asset_id=asset_id, anchor_memory_id=anchor_memory_id,
                status="unavailable", warnings=warnings,
            )

        warnings.extend(result.warnings or [])
        if result.refused:
            status = "refused"
            return MediaIngestResult(
                asset_id=asset_id, anchor_memory_id=anchor_memory_id,
                status=status, warnings=warnings,
            )

        drafts, skipped = _to_drafts(result, modality_norm)
        if skipped:
            warnings.append(f"{skipped} provider moment(s) dropped as unwritable")

        from mnemosyne.core.filters import admit_memory_write

        admitted_drafts = [
            draft
            for draft in drafts
            if admit_memory_write(
                draft.text, write_kind=_write_kind, policy=write_policy
            )[0]
        ]
        filtered = len(drafts) - len(admitted_drafts)
        if filtered:
            skipped += filtered
            warnings.append(
                f"{filtered} provider moment(s) dropped by write policy"
            )
        drafts = admitted_drafts

        if not drafts:
            status = "unavailable"
            return MediaIngestResult(
                asset_id=asset_id, anchor_memory_id=anchor_memory_id,
                status=status, warnings=warnings,
            )

        write = store.add_moments(asset_id, drafts)
        moment_ids = list(write.moment_ids)
        memory_ids = _bind_moment_memories(
            beam, store, asset_id, drafts, moment_ids,
            source=source, importance=importance, scope=scope,
            provider=result.provider, model=result.model,
            write_kind=_write_kind, write_policy=write_policy,
        )
        status = "partial" if skipped else "ok"

        return MediaIngestResult(
            asset_id=asset_id, anchor_memory_id=anchor_memory_id,
            status=status, moment_ids=moment_ids, memory_ids=memory_ids,
            warnings=warnings,
        )
    finally:
        # A constraint error mid-loop must not strand the asset at 'pending' --
        # nothing retries that state, and doctor would not flag it either.
        try:
            store.set_understanding_status(asset_id, status)
        except Exception:
            logger.info("media: could not finalize status for %s", asset_id, exc_info=True)


def _ensure_anchor_memory(
    beam, store: MediaStore, asset_id: str, ref_kind: str, ref_value: str,
    modality: str, *, title: Optional[str], source: str, importance: float,
    scope: str, write_kind: object, write_policy,
) -> Optional[str]:
    """Write (or reuse) the reference memory for an asset.

    Reused on re-ingest so a repeat does not accumulate anchor rows -- the
    asset upsert is idempotent and this has to be too. The recorded id is
    re-validated rather than trusted: consolidation may have taken the row.
    """
    asset = store.get_asset(asset_id) or {}
    try:
        meta = json.loads(asset.get("metadata") or "{}")
    except Exception:
        meta = {}
    existing = meta.get("anchor_memory_id") if isinstance(meta, dict) else None
    if existing and store._memory_exists(existing):
        return existing

    content = f"Referenced {modality}: {title or _short_ref(ref_kind, ref_value)}"
    memory_id = _remember_text(
        beam, content, source=source, importance=importance, scope=scope,
        metadata={"media": {"asset_id": asset_id, "modality": modality,
                            "ref_kind": ref_kind, "role": "anchor"}},
        write_kind=write_kind, write_policy=write_policy,
    )
    if memory_id and isinstance(meta, dict):
        meta["anchor_memory_id"] = memory_id
        try:
            store.conn.execute(
                "UPDATE media_assets SET metadata = ? WHERE asset_id = ?",
                (json.dumps(meta, default=str), asset_id),
            )
            store.conn.commit()
        except Exception:
            logger.info("media: could not record anchor id", exc_info=True)
    return memory_id


def _remember_text(
    beam, content: str, *, source: str, importance: float, scope: str,
    metadata: Dict[str, Any], write_kind: object, write_policy,
) -> Optional[str]:
    """Write one memory row for media.

    ``memory_type="artifact"`` because ``classify_memory`` would label a caption
    an "observation" (RFC 0003 §3.3 cost 3), and ``dedupe=False`` because
    ``_find_duplicate`` matches on ``(session_id, content)`` -- two identical
    captions from *different* assets would otherwise collapse into one row and
    the second moment would bind to the wrong asset, silently. It also avoids
    the ``COALESCE`` retype at ``beam.py:3301`` converting a user's
    conversational note into an artifact on a content collision.

    ``BeamMemory.remember`` is called directly rather than the ``core/memory.py``
    facade: ``should_remember`` can veto a write outright, which would leave a
    moment row bound to nothing.
    """
    try:
        from mnemosyne.core.filters import write_policy_operation

        with write_policy_operation(write_policy):
            return beam.remember(
                content,
                source=source,
                importance=importance,
                metadata=metadata,
                scope=scope,
                memory_type="artifact",
                dedupe=False,
                _write_kind=write_kind,
                _write_policy=write_policy,
            )
    except Exception:
        logger.info("media: memory write failed", exc_info=True)
        return None


def _describe(
    asset_id: str, ref_kind: str, ref_value: str, modality: str,
    mime: Optional[str], content_hash: Optional[str],
    hint: Optional[str], max_moments: Optional[int],
):
    """Ask the provider seam. Returns a DescribeResult or None.

    Never raises: ``call_modality_describe`` already returns None when the
    operator has not opted in, when nothing serves the modality, and when the
    backend fails. The import is local so ``core/media.py`` stays usable with no
    provider machinery loaded at all.
    """
    try:
        from mnemosyne.core.modality_backends import (
            DescribeRequest, call_modality_describe,
        )
    except Exception:
        return None

    try:
        cap = max_moments
        if cap is None:
            from mnemosyne.core import config as _config
            cap = _config.get_int("modality_max_moments", 12)
        request = DescribeRequest(
            modality=modality,
            uri=ref_value,
            mime=mime,
            content_hash=content_hash,
            hint=hint,
            max_moments=int(cap),
            fetch=_make_fetch(ref_kind, ref_value),
        )
        return call_modality_describe(request)
    except Exception:
        logger.info("media: describe failed for %s", asset_id, exc_info=True)
        return None


def _to_drafts(result, modality: str) -> Tuple[List[MomentDraft], int]:
    """Map provider output onto storage drafts, dropping what cannot be written.

    A provider's span guesses are validated here rather than trusted. Anything
    that fails is dropped and counted -- one bad moment must not cost the batch,
    since ``add_moments`` is deliberately all-or-nothing.
    """
    drafts: List[MomentDraft] = []
    seen = set()
    skipped = 0

    described = list(getattr(result, "moments", None) or [])
    if not described and getattr(result, "summary", None):
        described = [_summary_moment(result.summary)]

    for ordinal, item in enumerate(described):
        span_kind = _span_kind_for(item, modality)
        draft = MomentDraft(
            kind=_moment_kind_for(item, modality),
            text=str(getattr(item, "text", "") or "").strip(),
            span_kind=span_kind,
            t_start_ms=getattr(item, "t_start_ms", None) if span_kind == "time" else None,
            t_end_ms=getattr(item, "t_end_ms", None) if span_kind == "time" else None,
            page_start=getattr(item, "page_start", None) if span_kind == "page" else None,
            page_end=getattr(item, "page_end", None) if span_kind == "page" else None,
            char_start=getattr(item, "char_start", None) if span_kind == "char" else None,
            char_end=getattr(item, "char_end", None) if span_kind == "char" else None,
            bbox=getattr(item, "bbox", None) if span_kind == "box" else None,
            speaker=getattr(item, "speaker", None) if span_kind == "time" else None,
            confidence=float(getattr(item, "confidence", 1.0) or 1.0),
            ordinal=ordinal,
            provider=getattr(result, "provider", None),
            provider_model=getattr(result, "model", None),
        )
        if not draft.text or draft.kind not in MOMENT_KINDS:
            skipped += 1
            continue
        try:
            validate_span(draft.span_kind, draft.span_values(), draft.speaker)
        except (ValueError, TypeError):
            skipped += 1
            continue

        key = (draft.kind, draft.span_key())
        if key in seen:
            # Providers repeat themselves. The unique index would collapse these
            # anyway; dropping here keeps add_moments' strict batch check honest.
            skipped += 1
            continue
        seen.add(key)
        drafts.append(draft)

    return drafts, skipped


def _summary_moment(summary: str):
    from mnemosyne.core.modality_backends import DescribedMoment
    return DescribedMoment(kind="summary", text=str(summary))


def _moment_kind_for(item, modality: str) -> str:
    kind = str(getattr(item, "kind", "") or "").strip().lower()
    if kind in MOMENT_KINDS:
        return kind
    return {"audio": "transcript", "video": "shot", "document": "page"}.get(
        modality, "caption"
    )


def _span_kind_for(item, modality: str) -> str:
    """Pick the span kind from what the provider actually populated.

    Order matters: a provider that returns both a timestamp and a bbox gets the
    timestamp, because the time axis is the one a user queries. Everything that
    populates nothing is ``whole``, which is always writable.
    """
    if getattr(item, "t_start_ms", None) is not None:
        return "time"
    if getattr(item, "page_start", None) is not None:
        return "page"
    if getattr(item, "char_start", None) is not None and getattr(item, "char_end", None) is not None:
        return "char"
    if getattr(item, "bbox", None) is not None:
        return "box"
    return "whole"


def _bind_moment_memories(
    beam, store: MediaStore, asset_id: str, drafts: List[MomentDraft],
    moment_ids: List[str], *, source: str, importance: float, scope: str,
    provider: Optional[str], model: Optional[str], write_kind: object, write_policy,
) -> List[str]:
    """Write a memory row per *newly unbound* moment and bind it.

    Only unbound moments get a row: on a repeat ingest the moments already exist
    and are already bound, and writing a second memory row for each would
    reintroduce the duplication the unique index exists to prevent.
    """
    bound: List[str] = []
    existing = {m["moment_id"]: m for m in store.get_moments(asset_id)}

    for draft, moment_id in zip(drafts, moment_ids):
        row = existing.get(moment_id)
        if row is not None and row.get("memory_id"):
            bound.append(row["memory_id"])
            continue
        memory_id = _remember_text(
            beam, draft.text, source=source, importance=importance, scope=scope,
            metadata={"media": {
                "asset_id": asset_id, "moment_id": moment_id,
                "kind": draft.kind, "span_kind": draft.span_kind,
                "provider": provider, "provider_model": model,
            }},
            write_kind=write_kind, write_policy=write_policy,
        )
        if memory_id is None:
            continue
        store.bind_moment_memory(moment_id, memory_id)
        bound.append(memory_id)

    return bound
