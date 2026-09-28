"""
Mnemosyne core prefetch rendering — host-agnostic.

"""

from __future__ import annotations

import logging
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def _prefetch_content_char_limit() -> int:
    """Return the per-memory prefetch content limit.

    ``0`` means no truncation. This is the default because the old hardcoded
    200-character cap often removed the actual fact from LLM-authored memories.
    Operators that need tighter prompt budgets can set
    ``MNEMOSYNE_PREFETCH_CONTENT_CHARS`` to a positive integer.
    """
    raw = os.environ.get("MNEMOSYNE_PREFETCH_CONTENT_CHARS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Invalid MNEMOSYNE_PREFETCH_CONTENT_CHARS=%r; disabling prefetch truncation",
            raw,
        )
        return 0


def _format_prefetch_content(content: str, limit: int) -> str:
    """Format recalled memory content for prompt injection.

    When a positive limit is configured, truncate on a word boundary instead of
    splitting mid-token. Without a positive limit, return the complete content.
    """
    if limit <= 0 or len(content) <= limit:
        return content

    cut = content[:limit].rstrip()
    # Prefer a word boundary when one exists reasonably close to the limit.
    boundary = cut.rfind(" ")
    if boundary >= max(1, limit // 2):
        cut = cut[:boundary].rstrip()
    return f"{cut}..."


# Low-quality fragment filter for prefetch.
#
# The regex fact extractor can emit bare single-token "facts" (a stray adverb, a
# particle, or a truncated word). Such tokens FTS-match common query words and can
# outrank real memories in the per-turn prefetch window. A real, injectable memory
# is a phrase, not a lone token, so drop lone short/stopword tokens. Exact- and
# length-based only, so genuine short multi-word facts are never affected.
_PREFETCH_FRAGMENT_STOPWORDS = frozenset({
    "still", "what", "most", "almost", "back", "now", "too", "right",
    "being", "going", "here", "there", "then", "just", "also", "only",
    "even", "very", "really", "again", "away", "off", "out", "up",
    "down", "over", "that", "this", "it", "so",
})
_PREFETCH_MIN_FRAGMENT_CHARS = 8   # lone tokens shorter than this are dropped
_PREFETCH_OVERFETCH = 16           # recall more, then filter junk and cap
_PREFETCH_TOP_K = 5                # final injected count: compact, relevance-first

# Prompt-usefulness filter for automatic memory-context injection. Manual recall
# tools can stay broad; prefetch is silently injected into every model call, so
# it should be conservative and favor distilled memories over raw transcript.
_PREFETCH_RAW_PREFIXES = ("[USER]", "[ASSISTANT]", "[IDENTITY]")
_PREFETCH_EXCLUDED_PREFIXES = ("[ASSISTANT]",)
_PREFETCH_RAW_SOURCES = {"conversation"}
_PREFETCH_DISTILLED_SOURCES = {
    "preference", "correction", "fact", "identity", "insight", "sleep_consolidation",
}
_PREFETCH_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./:-]*", re.IGNORECASE)
_PREFETCH_DEDUP_STOPWORDS = _PREFETCH_FRAGMENT_STOPWORDS | frozenset({
    "about", "after", "before", "because", "could", "from", "have", "into",
    "like", "more", "need", "needs", "than", "them", "they", "want", "wants",
    "when", "where", "which", "while", "would", "yourself",
})
_PREFETCH_MODEL_SLOT_STOPWORDS = _PREFETCH_DEDUP_STOPWORDS | frozenset({
    "and", "are", "for", "how", "should", "the", "with", "what", "why",
})


def _is_low_quality_prefetch(content: str) -> bool:
    """True if recalled content is a bare single-token fragment with no value as
    injected context. Multi-word phrases always pass."""
    c = (content or "").strip()
    if not c:
        return True
    if len(c.split()) <= 1 and (len(c) <= _PREFETCH_MIN_FRAGMENT_CHARS
                                or c.lower() in _PREFETCH_FRAGMENT_STOPWORDS):
        return True
    return False


def _strip_prefetch_prefix(content: str) -> str:
    c = (content or "").strip()
    upper = c.upper()
    for prefix in _PREFETCH_RAW_PREFIXES:
        if upper.startswith(prefix):
            return c[len(prefix):].strip()
    return c


def _prefetch_tokens(content: str) -> Set[str]:
    c = _strip_prefetch_prefix(content).lower()
    tokens: Set[str] = set()
    for token in _PREFETCH_TOKEN_RE.findall(c):
        if len(token) <= 2 or token in _PREFETCH_DEDUP_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _prefetch_model_slot_tokens(content: str) -> Set[str]:
    """Content-word tokens for selected canonical model-slot injection.

    Model-slot injection is more dangerous than dedup tokenization because a
    single overlap can silently inject a durable user/workflow model into the
    prompt. Ignore common function words so unrelated slots do not match on
    tokens like "and" or "the". Expand structured slot labels such as
    ``communication_style`` into useful lexical pieces so normal queries like
    "communication style" can match them.
    """

    tokens: Set[str] = set()
    for token in _prefetch_tokens(content):
        if token in _PREFETCH_MODEL_SLOT_STOPWORDS:
            continue
        tokens.add(token)
        for part in re.split(r"[_:/.-]+", token):
            if len(part) > 2 and part not in _PREFETCH_MODEL_SLOT_STOPWORDS:
                tokens.add(part)
    return tokens


def _prefetch_topic_signal(row: Dict[str, Any]) -> float:
    """Best available non-importance relevance signal for a recall row."""
    signal = max(
        float(row.get("keyword_score") or 0.0),
        float(row.get("fts_score") or 0.0),
        float(row.get("dense_score") or 0.0),
    )
    # Fact/entity matches are explicit relevance signals even when recall() did
    # not fill keyword/FTS scores for that path.
    if row.get("fact_match") or row.get("entity_match"):
        signal = max(signal, 0.20)
    return signal


def _prefetch_source_quality(row: Dict[str, Any]) -> float:
    """Relative usefulness multiplier for injected memory.

    Distilled memories are better prompt context than raw transcript snippets;
    assistant transcript snippets should not be injected at all by default.
    """
    content = (row.get("content") or "").strip()
    upper = content.upper()
    source = str(row.get("source") or "").lower()

    if upper.startswith(_PREFETCH_EXCLUDED_PREFIXES):
        return 0.0

    quality = 1.0
    if source in _PREFETCH_DISTILLED_SOURCES:
        quality *= 1.12
    if source in _PREFETCH_RAW_SOURCES:
        quality *= 0.72
    if upper.startswith("[USER]"):
        quality *= 0.68
    elif upper.startswith("[IDENTITY]"):
        quality *= 0.80
    elif source.startswith("memoria_source"):
        quality *= 0.90
    return quality


def _prefetch_is_raw(row: Dict[str, Any]) -> bool:
    content = (row.get("content") or "").strip().upper()
    source = str(row.get("source") or "").lower()
    return source in _PREFETCH_RAW_SOURCES or content.startswith("[USER]") or content.startswith("[IDENTITY]")


def _prefetch_adjusted_score(row: Dict[str, Any]) -> float:
    score = float(row.get("score") or 0.0)
    signal = _prefetch_topic_signal(row)
    importance = min(max(float(row.get("importance") or 0.0), 0.0), 1.0)
    return (score * 0.65 + signal * 0.35 + importance * 0.05) * _prefetch_source_quality(row)


def _semantic_dedup_prefetch(rows: List[Dict[str, Any]], threshold: float = 0.72) -> List[Dict[str, Any]]:
    """Collapse near-duplicate memory rows, keeping the best-ranked variant."""
    kept: List[Dict[str, Any]] = []
    kept_tokens: List[Set[str]] = []
    for row in rows:
        tokens = _prefetch_tokens(row.get("content", ""))
        if not tokens:
            continue
        duplicate = False
        for existing in kept_tokens:
            overlap = len(tokens & existing)
            if not overlap:
                continue
            jaccard = overlap / max(len(tokens | existing), 1)
            containment = overlap / max(min(len(tokens), len(existing)), 1)
            if jaccard >= threshold or containment >= 0.86:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(row)
        kept_tokens.append(tokens)
    return kept


# ---------------------------------------------------------------------------
# Prefetch profiles
#
# A profile is a named bundle of the prefetch knobs (recall breadth, weights,
# temporal decay, relevance thresholds, source quality filtering + dedup toggles,
# and which registered sources to merge). Operators select a profile via
# MNEMOSYNE_PREFETCH_PROFILE; libraries can register their own with
# register_profile().
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrefetchProfile:
    name: str
    top_k: int = _PREFETCH_TOP_K
    importance_weight: Optional[float] = None   # None -> recall() default
    vec_weight: Optional[float] = None
    fts_weight: Optional[float] = None
    temporal_weight: float = 0.2
    temporal_halflife: float = 48
    min_score: float = 0.20
    min_importance: float = 0.65
    min_topic_signal: float = 0.08
    raw_min_topic_signal: float = 0.18
    content_char_limit: int = 0                  # 0 -> use env / untruncated
    drop_low_quality: bool = True
    dedup: bool = True
    semantic_dedup: bool = True
    exclude_assistant: bool = True
    sources: Tuple[str, ...] = ("bank",)         # which registered sources to merge


_BUILTIN_PROFILES: Dict[str, PrefetchProfile] = {
    # Default per-turn injection: compact, relevance-first, and conservative
    # about raw transcript snippets.
    "general": PrefetchProfile(name="general"),
    # Favor recent, high-importance memories; same filter/dedup defaults.
    "social-chat": PrefetchProfile(
        name="social-chat", top_k=6,
        importance_weight=0.6, temporal_weight=0.35, temporal_halflife=24,
    ),
}


def register_profile(profile: "PrefetchProfile") -> None:
    """Register (or override) a named prefetch profile."""
    _BUILTIN_PROFILES[profile.name] = profile


def _resolve_profile(name: Optional[str]) -> PrefetchProfile:
    """Return the named profile, falling back to `general` for unknown/empty."""
    return _BUILTIN_PROFILES.get((name or "general"), _BUILTIN_PROFILES["general"])


def _norm_prefetch_line(line: str) -> str:
    """Normalize a content line for cross-source dedup: lowercase, collapse
    whitespace, drop a leading bracketed metadata prefix (e.g. timestamps)."""
    s = line.strip()
    while s.startswith("[") or s.startswith("("):
        close = s.find("]") if s.startswith("[") else s.find(")")
        if close == -1:
            break
        s = s[close + 1:].strip()
    return " ".join(s.lower().split())


def _dedup_blocks(blocks: List[str]) -> List[str]:
    """Collapse near-duplicate content lines across blocks, preserving each
    block's headers and order. A single block is returned unchanged."""
    seen: Set[str] = set()
    out: List[str] = []
    for block in blocks:
        kept: List[str] = []
        for line in block.split("\n"):
            is_header = line.lstrip().startswith("#") or not line.strip()
            if is_header:
                kept.append(line)
                continue
            norm = _norm_prefetch_line(line)
            if norm and norm in seen:
                continue
            if norm:
                seen.add(norm)
            kept.append(line)
        out.append("\n".join(kept))
    return out


def _coerce_source_output(out: Any, profile: "PrefetchProfile", header: str) -> str:
    """Turn a registered source's return value into an injectable block.

    A source may return a pre-formatted string (used verbatim) or a list of hit
    dicts ({"content", optional "timestamp"/"importance"}). Lists are formatted
    under `header`, low-quality-filtered + capped per the profile."""
    if not out:
        return ""
    if isinstance(out, str):
        return out
    try:
        hits = list(out)
    except TypeError:
        return ""
    lines = [header]
    limit = _prefetch_content_char_limit() or profile.content_char_limit
    for r in hits[: profile.top_k]:
        content = r.get("content", "") if isinstance(r, dict) else str(r)
        if profile.drop_low_quality and _is_low_quality_prefetch(content):
            continue
        content = _format_prefetch_content(content, limit)
        if isinstance(r, dict) and r.get("timestamp"):
            lines.append(f"  [{str(r['timestamp'])[:16]}] {content}")
        else:
            lines.append(f"  {content}")
    return "\n".join(lines) if len(lines) > 1 else ""


def identity_rows(beam: Any) -> List[Dict[str, Any]]:
    """Return ALL identity memories for the beam's ACTIVE session, deterministically.

    Identity memories (source='identity') answer "who am I talking to?" and
    must be injected on every turn regardless of the user's message. Routing
    them through semantic recall is a latent bug: a short/generic opener does
    not match the identity text, so it never enters recall's top_k window and
    the importance filter never sees it. This pulls them straight from the
    active session_id with a direct, query-independent SQL read. Strictly
    session-scoped, so there is zero cross-session leakage.
    """
    out: List[Dict[str, Any]] = []
    if beam is None:
        return out
    try:
        cur = beam.conn.cursor()
        cur.execute(
            "SELECT content, importance, timestamp FROM working_memory "
            "WHERE source='identity' AND session_id=? "
            "ORDER BY importance DESC, timestamp DESC",
            (beam.session_id,),
        )
        for content, importance, timestamp in cur.fetchall():
            if not content:
                continue
            out.append({
                "content": content,
                "importance": importance if importance is not None else 0.95,
                "timestamp": timestamp or "",
                "source": "identity",
                "_always_inject": True,
            })
    except Exception as e:
        logger.debug("Mnemosyne identity read failed (non-fatal): %s", e)
    return out


def render_identity(rows: List[Dict[str, Any]], existing_blocks: List[str], profile: "PrefetchProfile") -> str:
    """Render the always-inject identity block for the active session.

    Rows come from :func:`identity_rows`. They are rendered in the same format
    as the memory bank, tagged ``[IDENTITY]``. Rows whose content already
    appears in the blocks recall produced are dropped, so a query that *does*
    match the identity never yields a duplicate. Returns ``""`` when there is
    nothing to inject.
    """
    if not rows:
        return ""
    already = "\n".join(existing_blocks)
    content_limit = _prefetch_content_char_limit() or profile.content_char_limit
    lines: List[str] = ["## Mnemosyne Context"]
    seen: set = set()
    for r in rows:
        content = r.get("content", "")
        if not content or content in seen:
            continue
        disp = _format_prefetch_content(content, content_limit)
        # Dedup against anything recall already surfaced (raw or truncated).
        if content in already or disp in already:
            continue
        seen.add(content)
        ts = r.get("timestamp", "")[:16] if r.get("timestamp") else ""
        imp = r.get("importance", 0.95)
        lines.append(f"  [{ts}] (importance {imp:.2f}) [IDENTITY] {disp}")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def render_model_slots(beam: Any, query: str, profile: "PrefetchProfile", *, canonical_owner: str = "default") -> str:
    """Render relevant accepted canonical model slots for silent prefetch.

    Model cards are a display/debug view; normal prompt injection uses only
    selected canonical model slots with clear query overlap. This mirrors the
    useful part of Hindsight mental-model injection without globally
    injecting whole cards.
    """
    if beam is None:
        return ""
    query_tokens = _prefetch_model_slot_tokens(query)
    if not query_tokens:
        return ""
    try:
        max_slots = int(os.environ.get("MNEMOSYNE_PREFETCH_MODEL_SLOT_LIMIT", "3") or "3")
    except (TypeError, ValueError):
        max_slots = 3
    try:
        min_signal = int(os.environ.get("MNEMOSYNE_PREFETCH_MODEL_SLOT_MIN_OVERLAP", "1") or "1")
    except (TypeError, ValueError):
        min_signal = 1
    try:
        store = getattr(beam, "canonical", None)
        if store is None:
            from mnemosyne.core.canonical import CanonicalStore
            store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
            beam.canonical = store
        owner_id = canonical_owner
        rows = []
        for category in ("model:user", "model:workflow", "model:project", "model:agent"):
            rows.extend(store.list(owner_id, category=category))
    except Exception as e:
        logger.debug("Mnemosyne model-slot prefetch failed (non-fatal): %s", e)
        return ""
    scored: List[tuple] = []
    for row in rows:
        text = " ".join(str(row.get(k) or "") for k in ("category", "name", "body"))
        tokens = _prefetch_model_slot_tokens(text)
        overlap = len(query_tokens & tokens)
        if overlap < min_signal:
            continue
        confidence = float(row.get("confidence") or 0.0)
        scored.append((overlap, confidence, row))
    if not scored:
        return ""
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    content_limit = _prefetch_content_char_limit() or profile.content_char_limit
    lines = ["## Mnemosyne Model Context"]
    for _, _, row in scored[:max_slots]:
        body = _format_prefetch_content(str(row.get("body") or ""), content_limit)
        body = " ".join(body.split())
        if not body:
            continue
        category = str(row.get("category") or "model")
        name = str(row.get("name") or "slot").replace("_", " ")
        lines.append(f"  [{category}] {name}: {body}")
    return "\n".join(lines) if len(lines) > 1 else ""


def render_bank_source(
    beam: Any,
    query: str,
    session_id: str,
    profile: "PrefetchProfile",
    *,
    ledger: Any = None,
    active_session_id: str = "",
    lock: Any = None,
) -> str:
    """The built-in memory-bank source: hybrid recall with temporal weighting,
    relevance + low-quality filtering, strictly session-scoped.
    Parameterized by *profile*.

    ``ledger`` is the caller's verbatim ledger, keyed by the same session as
    the Beam (``session_id``, falling back to ``active_session_id``); pass
    ``None`` when the caller has no ledger. ``lock`` is a re-entrant lock the
    caller holds across the recall call (the Hermes provider passes its Beam
    access lock); direct callers own their own synchronization.
    """
    overfetch = max(profile.top_k * 2, _PREFETCH_OVERFETCH)  # over-fetch; junk filtered below
    recall_kwargs: Dict[str, Any] = dict(
        query=query, top_k=overfetch,
        temporal_weight=profile.temporal_weight,
        temporal_halflife=profile.temporal_halflife,
    )
    # Pass tuning weights only when the profile sets them, so the default
    # profile still lets recall() resolve its own weights.
    if profile.importance_weight is not None:
        recall_kwargs["importance_weight"] = profile.importance_weight
    if profile.vec_weight is not None:
        recall_kwargs["vec_weight"] = profile.vec_weight
    if profile.fts_weight is not None:
        recall_kwargs["fts_weight"] = profile.fts_weight
    # CWE-200 (#914 follow-up): author identity is NEVER injected into
    # the automatic prefetch recall path. A non-empty author_id makes
    # beam.recall() replace session/channel filtering with (1=1),
    # silently widening prefetch scope across gateway threads and
    # leaking memories across sessions. Author identity is applied
    # exclusively as a per-write stamp at the store site; the beam
    # read identity stays unset so recall keeps session scoping.
    # Revocable provider-owned capture proofs; explicit tools do not
    # pass this optimization to recall.
    _ledger_key = str(session_id or "").strip() or active_session_id
    if _ledger_key and ledger is not None and ledger.enabled:
        _echo_snapshot = ledger.snapshot_for(_ledger_key)
        if _echo_snapshot:
            recall_kwargs["exclude_captures"] = _echo_snapshot
    with (lock if lock is not None else nullcontext()):
        results = beam.recall(**recall_kwargs)
        snapshot = recall_kwargs.get("exclude_captures")
        if snapshot is not None and not snapshot.generation.valid:
            recall_kwargs.pop("exclude_captures", None)
            results = beam.recall(**recall_kwargs)
    if not results:
        return ""
    # Filter out low-relevance results to prevent context pollution.
    # Importance alone is not enough for silent injection: a memory must
    # also have a real topical signal. Raw transcript rows need a
    # stronger topical signal than distilled facts/preferences.
    filtered = []
    for r in results:
        if profile.drop_low_quality and _is_low_quality_prefetch(r.get("content", "")):
            continue
        if profile.exclude_assistant and _prefetch_source_quality(r) <= 0:
            continue
        signal = _prefetch_topic_signal(r)
        score = float(r.get("score") or 0.0)
        importance = float(r.get("importance") or 0.0)
        required_signal = profile.raw_min_topic_signal if _prefetch_is_raw(r) else profile.min_topic_signal
        if signal < required_signal:
            continue
        if score < profile.min_score and importance < profile.min_importance:
            continue
        filtered.append(r)

    filtered.sort(key=_prefetch_adjusted_score, reverse=True)
    if profile.semantic_dedup:
        filtered = _semantic_dedup_prefetch(filtered)
    # Cap back to the intended injection size after over-fetch+filter.
    filtered = filtered[:profile.top_k]
    if not filtered:
        return ""
    lines = ["## Mnemosyne Context"]
    content_limit = _prefetch_content_char_limit() or profile.content_char_limit
    for r in filtered:
        content = _format_prefetch_content(
            r.get("content", ""),
            content_limit,
        )
        content = " ".join(content.split())
        ts = r.get("timestamp", "")[:16] if r.get("timestamp") else ""
        imp = r.get("importance", 0.0)
        trust = r.get("trust_tier", "STATED")
        trust_tag = f" [{trust}]" if trust != "STATED" else ""
        source = str(r.get("source") or "").strip()
        source_tag = f", source {source}" if source and source != "conversation" else ""
        lines.append(f"  [{ts}] (importance {imp:.2f}{source_tag}){trust_tag} {content}")
    return "\n".join(lines)
