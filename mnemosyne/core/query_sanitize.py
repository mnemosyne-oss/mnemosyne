"""Shared query-only speaker-stamp classification for memory providers."""

import re
import unicodedata

# Multi-user gateways (group chats, shared sessions) may stamp the speaker's
# display name onto message content before it reaches the memory layer, e.g.
# "[Alice] what time does the bakery open". The stamp is envelope
# metadata, not a topical token: left in the recall query, it makes every
# row mentioning the same speaker score as lexically relevant regardless of
# subject. Strip leading stamps from the QUERY only -- captured rows and
# consolidated summaries keep the stamped text so speaker attribution
# survives distillation.
#
# The grammar is deliberately NAME-shaped, not bracket-shaped: gateways
# stamp person names, so a stamp is a short human-name token (Unicode
# letters, spaces, apostrophes, hyphens, periods; 1-4 words) followed by
# whitespace or end-of-string. Bracketed tokens that carry topical meaning
# -- song titles like [Untitled], tags like [TODO], timestamps like
# [2026-05-14 12:00], markdown links like [API docs](http://x) -- contain
# digits/underscores or are not name-shaped and are kept in the query.
_PREFETCH_NAME_STAMP_RE = re.compile(
    r"(?:\[([^\[\]［］\r\n]{1,48})\]|［([^\[\]［］\r\n]{1,48})］)(?:\s+|$)"
)
_NAME_INNER_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|[ .'’\-]){0,47}")


# Bracketed tokens that look name-shaped but are common content tags, not
# speakers. Matched case-insensitively against the stamp's inner text.
_PREFETCH_NAME_STAMP_TAGS = frozenset({
    "todo", "untitled", "note", "notes", "notice", "draft", "wip", "api",
    "docs", "documentation", "link", "url", "tag", "important", "idea",
    "question", "answer", "update", "edit", "screenshot", "image", "photo",
    "video", "audio", "recording", "transcript", "log", "misc", "test",
    "annotation", "announcement", "summary", "minutes", "changelog",
    "release", "checklist", "reminder", "agenda", "minutes",
})

# Lowercase particles that occur inside real multi-word surnames
# (Spanish/French/Dutch/Arabic transliteration conventions). Allowed only
# BETWEEN capitalized words, never as the first or last word.
_NAME_PARTICLES = frozenset({
    "de", "del", "la", "las", "le", "les", "van", "von", "der", "den",
    "di", "da", "dos", "das", "du", "bin", "ibn", "al", "el", "y", "e",
})


def _looks_capitalized(w: str) -> bool:
    """Capitalized, or written in a script without letter case (CJK,
    Arabic, Hebrew...): caseless scripts must not fail the gate."""
    first = w[:1]
    if not first:
        return False
    return first.isupper() or first.lower() == first.upper()


def _blocklisted_word(w: str) -> bool:
    """A word is blocklisted when it (or its simple singular/past-inflected
    form) is a known content tag: kills [TODOs], [Noted], [API Docs]..."""
    w = unicodedata.normalize("NFKC", w).lower().rstrip(".,")
    if w in _PREFETCH_NAME_STAMP_TAGS:
        return True
    if w.endswith("s") and w[:-1] in _PREFETCH_NAME_STAMP_TAGS:
        return True
    if w.endswith("d") and w[:-1] in _PREFETCH_NAME_STAMP_TAGS:
        return True
    return False


def _is_name_stamp(inner: str) -> bool:
    """Whether a bracketed token's inner text plausibly names a speaker.

    Inner text is NFKC-normalized first: fullwidth lookalikes ([ＴＯＤＯ]),
    Turkish/combining variants, and other casefold-bypass forms must hit
    the blocklist exactly like their ASCII spellings.

    Note the accepted trade-off: a single lowercase Latin word passes the
    capitalization gate, so real lowercase display names ([alice]) are
    stripped AND lowercase topical tags ([recipe]) are stripped with them.
    The gateway stamps speaker display names verbatim, which includes
    lowercase names; grammar alone cannot separate the two without a
    per-message speaker signal, which deployments do not pass here.
    Multi-word stamps must look name-like; single words are accepted as
    the lesser evil (missing a speaker stamp pollutes every recall)."""
    inner = unicodedata.normalize("NFKC", inner)
    if not _NAME_INNER_RE.fullmatch(inner):
        return False
    words = inner.split()
    if not (1 <= len(words) <= 4):
        return False
    if any(w.isdigit() or "_" in w for w in words):
        return False
    if any(_blocklisted_word(w) for w in words):
        return False
    if len(words) > 1:
        for i, w in enumerate(words):
            if w.lower() in _NAME_PARTICLES:
                # particle: only between capitalized words
                if i == 0 or i == len(words) - 1:
                    return False
                continue
            if not _looks_capitalized(w):
                return False
    return True


def sanitize_prefetch_query(query: str) -> str:
    """Remove leading speaker stamps without rewriting any retained query text.

    Classification uses NFKC on bounded candidates, never on the query. The
    returned suffix is sliced once from original input; no-stamp input is
    returned unchanged. Separating whitespace belongs to the removed prefix.
    Punctuation-only residue after a stamp is treated as no topical query.
    """
    original = query or ""
    offset = 0
    while offset < len(original) and original[offset].isspace():
        offset += 1
    stripped = False
    while True:
        match = _PREFETCH_NAME_STAMP_RE.match(original, offset)
        if not match or not _is_name_stamp(match.group(1) or match.group(2)):
            break
        offset = match.end()
        stripped = True
    if not stripped:
        return original
    suffix = original[offset:]
    if not re.search(r"\w", suffix):
        return ""
    return suffix
