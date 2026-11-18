"""
Mnemosyne Modality Backend Registry
===================================
Pluggable adapter for turning non-text content into text.

A structural mirror of ``core/llm_backends.py``: one method, request-shaped,
returning result-or-None; the caller owns the prompt, the backend owns the
routing; failure is a return value, not an exception; and a callable test seam
ships with it.

Two things differ, both deliberately:

1. **The registry is keyed by modality.** A local Whisper install for audio
   alongside a remote endpoint for vision is a realistic configuration, and a
   single global slot cannot express it.
2. **There is an explicit opt-in gate, checked before the registry.** Mnemosyne
   must not make an outbound call to describe media until the operator has said
   yes, once, explicitly (RFC 0002 §3.4, restated as a testable invariant in
   RFC 0004 §3). Checking the gate *before* the lookup means a host that
   registered a backend for a user who never opted in makes no call at all.

Nothing here stores anything, and nothing here reads bytes on its own. Bytes,
when a provider needs them, arrive through :attr:`DescribeRequest.fetch`, which
the caller wires from ``core/resolvers.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol

logger = logging.getLogger(__name__)

MODALITIES: FrozenSet[str] = frozenset({"image", "video", "audio", "document"})


def modality_enabled() -> bool:
    """Whether the operator has opted in to outbound modality calls.

    **Read at call time, never cached in a module constant.** ``local_llm.py:54``
    reads ``MNEMOSYNE_HOST_LLM_ENABLED`` at import, which makes it impossible to
    monkeypatch and therefore impossible to test honestly. Repeating that here
    would fail in the worst direction: the privacy test passes (the flag was
    unset at import) while production dials out (the operator set it later).

    Defaults to **false**. A missing or unreadable config is a "no".
    """
    try:
        from mnemosyne.core import config as _config
        return _config.get_bool("modality_enabled", False)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Wire shapes
# ---------------------------------------------------------------------------

@dataclass
class DescribeRequest:
    """What the caller asks a provider to describe.

    ``fetch`` is the field that makes this usable for anything but a public URL.
    An OpenAI-compatible vision call needs a base64 data part for a local file
    or a ``blob://`` reference — which is to say, for essentially everything a
    privacy-first local memory system actually holds. It is **lazy and
    optional**: a backend describing a publicly fetchable ``https://`` URL never
    calls it, and no bytes are read. The caller supplies it from the
    ``core/resolvers.py`` registry.
    """

    modality: str
    uri: str
    mime: Optional[str] = None
    content_hash: Optional[str] = None
    #: The caller's prompt. Same division of labour as ``LLMBackend.complete``:
    #: the caller owns the prompt, the backend owns the routing.
    hint: Optional[str] = None
    max_moments: int = 12
    span_hint: Optional[str] = None
    timeout: float = 60.0
    detail: str = "auto"
    fetch: Optional[Callable[[], Optional[bytes]]] = None


@dataclass
class DescribedMoment:
    """One span a provider proposes.

    This is the *provider wire shape*: what a backend is allowed to say. It is
    deliberately **not** ``media.MomentDraft``, the storage shape — a provider
    must not be able to set ``memory_id``, and identity and binding belong to
    the store. ``remember_media`` maps one onto the other.

    (RFC 0002 §3.1 and RFC 0003 §2.2 both called their type ``MomentDraft``;
    the two names collided, so the provider-side one is renamed here.)
    """

    kind: str
    text: str
    t_start_ms: Optional[int] = None
    t_end_ms: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    bbox: Any = None
    speaker: Optional[str] = None
    confidence: float = 1.0
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DescribeResult:
    """What a provider returns.

    ``refused`` is distinct from failure. RFC 0003 §2.1 declares
    ``understanding_status='refused'`` as a valid asset state, and a provider
    that declines on safety grounds is meaningfully different from one that
    timed out — the first will decline again, the second is worth retrying. The
    distinction has to survive into the asset row or it is lost.
    """

    summary: Optional[str] = None
    moments: List[DescribedMoment] = field(default_factory=list)
    provider: Optional[str] = None
    model: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    refused: bool = False


class ModalityBackend(Protocol):
    """A provider that turns non-text content into text moments."""

    name: str
    modalities: FrozenSet[str]

    def describe(self, request: DescribeRequest) -> Optional[DescribeResult]:
        ...


@dataclass
class CallableModalityBackend:
    """Wrap a callable as a :class:`ModalityBackend`. For tests and one-off callers."""

    name: str
    func: Callable[[DescribeRequest], Optional[DescribeResult]]
    modalities: FrozenSet[str] = MODALITIES

    def describe(self, request: DescribeRequest) -> Optional[DescribeResult]:
        return self.func(request)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_backends: Dict[str, ModalityBackend] = {}
_default: Optional[ModalityBackend] = None


def set_modality_backend(
    backend: Optional[ModalityBackend],
    modalities: Optional[FrozenSet[str]] = None,
) -> None:
    """Register (or clear) a modality backend.

    With ``modalities``, the backend is registered for exactly those modalities.
    Without, it becomes the default for anything it declares it can serve.
    Passing ``backend=None`` clears the default; pass ``modalities`` alongside
    it to clear only those entries.
    """
    global _default
    if backend is None:
        if modalities:
            for modality in modalities:
                _backends.pop(str(modality).strip().lower(), None)
        else:
            _default = None
        return

    if modalities:
        for modality in modalities:
            _backends[str(modality).strip().lower()] = backend
    else:
        _default = backend


def get_modality_backend(modality: str) -> Optional[ModalityBackend]:
    """Return the backend that can serve ``modality``, or None.

    The default is only returned when it *declares* it serves this modality.
    Without that check, an image model gets handed an mp3 and asked to describe
    it — and the failure mode is not an error but a confident, wrong caption,
    which is worse than no caption at all.
    """
    key = str(modality or "").strip().lower()
    explicit = _backends.get(key)
    if explicit is not None:
        return explicit
    if _default is not None and key in getattr(_default, "modalities", frozenset()):
        return _default
    return None


def clear_modality_backends() -> None:
    """Reset the registry. Called from ``tests/conftest.py`` — the registry is a
    process-global, and a test that forgets to unregister would otherwise bleed
    into the next."""
    global _default
    _backends.clear()
    _default = None


def call_modality_describe(request: DescribeRequest) -> Optional[DescribeResult]:
    """Describe content via the registered backend, or return None.

    Returns ``None`` — never raises — when modality support is disabled, no
    backend serves the modality, or the backend itself fails. Rung 4 of the
    RFC 0002 §3.3 degradation ladder depends on this: the caller registers the
    asset by reference regardless, so a missing provider costs the user nothing
    they had before.

    The gate is checked **first**, before the registry is even consulted, so
    that "no outbound traffic when disabled" holds even on a host that
    registered a backend eagerly.
    """
    if not modality_enabled():
        return None

    backend = get_modality_backend(request.modality)
    if backend is None:
        # The documented setup is configuration only: MNEMOSYNE_MODALITY_*
        # (or config.yaml) and nothing else. Registration is deferred to here,
        # the first describe after the operator opted in, so importing a
        # module still never arms an outbound path. Explicit registrations
        # win: this only runs when nothing serves the modality.
        _autoregister_configured_backends()
        backend = get_modality_backend(request.modality)
    if backend is None:
        return None

    try:
        return backend.describe(request)
    except Exception:
        # Never raise into remember(). Provenance belongs at the call site, and
        # the request must not be logged — it can carry a file path.
        logger.info(
            "modality backend %r failed for modality %r",
            getattr(backend, "name", "?"), request.modality, exc_info=True,
        )
        return None


def _autoregister_configured_backends() -> None:
    """Register the built-in adapter as the default when it is configured.

    Called only behind the ``modality_enabled`` gate, and only when no backend
    serves the requested modality. A host that registered its own default is
    left alone.
    """
    if get_modality_backend("document") is None:
        # Documents are read locally: no endpoint, no model, nothing sent.
        try:
            from mnemosyne.core.modality_documents import LocalDocumentBackend
            set_modality_backend(LocalDocumentBackend(), frozenset({"document"}))
        except Exception:
            logger.info("modality: local document backend unavailable", exc_info=True)
    try:
        from mnemosyne.core.modality_openai_compat import is_configured, register_if_configured
    except Exception:
        logger.info("modality: built-in adapter unavailable", exc_info=True)
        return
    if not is_configured():
        return
    if get_modality_backend("video") is None:
        # Frames to the vision model and the soundtrack to transcription, via
        # the same endpoint; needs ffmpeg, and says so when it is missing.
        try:
            from mnemosyne.core.modality_video import VideoFrameBackend
            set_modality_backend(VideoFrameBackend(), frozenset({"video"}))
        except Exception:
            logger.info("modality: video backend unavailable", exc_info=True)
    if _default is None:
        try:
            register_if_configured()
        except Exception:
            logger.info("modality: built-in adapter registration failed", exc_info=True)

