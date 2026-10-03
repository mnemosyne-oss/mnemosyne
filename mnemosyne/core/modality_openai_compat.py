"""
OpenAI-Compatible Vision Adapter
================================
The reference :class:`~mnemosyne.core.modality_backends.ModalityBackend`.

This module speaks the OpenAI-compatible ``POST {base_url}/chat/completions``
shape with ``image_url`` content parts. That shape is served by Atlas Cloud,
OpenRouter, Together, Groq, vLLM, LM Studio, Ollama, and a local ``llama.cpp``
server alike, which is why the module is named after the **protocol** and not
after any provider. Switching endpoints must require changing environment
variables only; if it ever requires changing code, something in here has
acquired a vendor assumption and should be removed.

Three policies are borrowed rather than reinvented:

- Transport and timeout follow ``_call_remote_llm_with_model``
  (``core/local_llm.py:458``): httpx when present, ``urllib`` otherwise.
- Retry and backoff follow ``_embed_api`` (``core/embeddings.py:286-318``),
  with the rate-limit classifier ``_is_rate_limit_error``
  (``core/embeddings.py:247``). A third retry policy in one codebase is a
  third thing to get wrong.
- JSON tolerance follows ``core/extraction.py:80-105``: strict JSON, then
  partial salvage, then give up and return ``None``.

Nothing here raises into ``remember()``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from mnemosyne.core.modality_backends import (
    DescribedMoment,
    DescribeRequest,
    DescribeResult,
)

logger = logging.getLogger(__name__)

NAME = "openai_compat"

_MAX_ATTEMPTS = 3
_PUBLIC_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

DEFAULT_PROMPT = (
    "Describe this {modality} for a searchable memory index. "
    "Reply with JSON only, no prose and no code fences, in exactly this shape:\n"
    '{{"summary": "one or two sentences", "moments": '
    '[{{"kind": "caption", "text": "what is shown"}}]}}\n'
    "Use at most {max_moments} moments. Describe only what is actually present; "
    "do not guess at anything you cannot see."
)


# ---------------------------------------------------------------------------
# Configuration (read at call time, never cached at import)
# ---------------------------------------------------------------------------

def _cfg_str(key: str, default: str = "") -> str:
    try:
        from mnemosyne.core import config as _config
        return _config.get_str(key, default)
    except Exception:
        return default


def _cfg_int(key: str, default: int) -> int:
    try:
        from mnemosyne.core import config as _config
        return _config.get_int(key, default)
    except Exception:
        return default


def model_for(modality: str) -> str:
    """The configured model for a modality, or ``""``."""
    return _cfg_str({
        "image": "modality_vision_model",
        "document": "modality_vision_model",
        "video": "modality_video_model",
        "audio": "modality_audio_model",
    }.get(str(modality or "").strip().lower(), "modality_vision_model"))


def is_configured(modality: str = "image") -> bool:
    """Whether this adapter has what it needs to make a call.

    Both a base URL and an API key are required, per RFC 0002 §3.2 — a base URL
    alone is more likely a half-finished config than a deliberately
    unauthenticated endpoint, and guessing wrong means an outbound call the
    operator did not intend.
    """
    return bool(_cfg_str("modality_base_url")) and bool(_cfg_str("modality_api_key"))


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def _guess_mime(uri: Optional[str]) -> Optional[str]:
    """Type from the file name when the caller gave none.

    Vision endpoints validate the data URI's media type; OpenAI accepts only
    ``image/png``, ``image/jpeg``, ``image/gif`` and ``image/webp`` and rejects
    ``application/octet-stream`` outright, so an unlabeled local PNG would
    otherwise fail every call.
    """
    import mimetypes

    if not uri:
        return None
    guessed, _ = mimetypes.guess_type(str(uri), strict=False)
    return guessed


def _image_part(request: DescribeRequest) -> Optional[Dict[str, Any]]:
    """Build the ``image_url`` content part, fetching bytes only when needed.

    A publicly fetchable URL is passed through as-is and ``fetch`` is never
    called — no bytes are read and nothing leaves the machine that was not
    already public. Anything else (a local file, a ``blob://`` reference) needs
    a base64 data URI, which is the whole reason ``DescribeRequest.fetch``
    exists.
    """
    if _PUBLIC_URL_RE.match(request.uri or ""):
        return {
            "type": "image_url",
            "image_url": {"url": request.uri, "detail": request.detail},
        }

    if request.fetch is None:
        return None
    try:
        raw = request.fetch()
    except Exception:
        logger.info("modality fetch failed", exc_info=True)
        return None
    if not raw:
        return None

    mime = request.mime or _guess_mime(request.uri) or "application/octet-stream"
    encoded = base64.b64encode(raw).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{encoded}",
            "detail": request.detail,
        },
    }


def build_prompt(request: DescribeRequest) -> str:
    """The instruction sent alongside the content.

    ``request.hint`` wins when the caller supplies one — the caller owns the
    prompt, per ``llm_backends.py:30-33``. Otherwise
    ``MNEMOSYNE_MODALITY_PROMPT`` overrides the default, the same affordance
    ``core/extraction.py`` provides via ``MNEMOSYNE_EXTRACTION_PROMPT``.
    """
    if request.hint:
        return request.hint
    template = os.environ.get("MNEMOSYNE_MODALITY_PROMPT") or DEFAULT_PROMPT
    try:
        return template.format(
            modality=request.modality,
            max_moments=request.max_moments,
        )
    except Exception:
        # A malformed operator-supplied template must not break ingest.
        return template


# ---------------------------------------------------------------------------
# Response parsing — the extraction.py tolerance ladder
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return text


def parse_response(raw: str, max_moments: int) -> Optional[Tuple[Optional[str], List[DescribedMoment]]]:
    """Parse a provider reply into ``(summary, moments)``, or None.

    Rung 1: strict JSON. Rung 2: salvage — a truncated reply that still contains
    recognizable text is worth something, and a vision model that ran out of
    tokens mid-array is a routine occurrence rather than an error. Rung 3: give
    up. Returning ``None`` is a supported outcome all the way up the stack.
    """
    text = _strip_fences(raw)
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None

    if isinstance(parsed, dict):
        summary = parsed.get("summary")
        moments: List[DescribedMoment] = []
        raw_moments = parsed.get("moments")
        if isinstance(raw_moments, list):
            for item in raw_moments:
                moment = _coerce_moment(item)
                if moment is not None:
                    moments.append(moment)
                if len(moments) >= max_moments:
                    break
        if summary or moments:
            return (str(summary) if summary else None, moments)
        return None

    # Salvage: pull quoted strings long enough to be descriptions rather than
    # JSON keys. Deliberately crude -- it is a floor, not a parser.
    salvaged = [s for s in re.findall(r'"([^"]{16,})"', text)]
    if salvaged:
        return (salvaged[0], [
            DescribedMoment(kind="caption", text=s)
            for s in salvaged[:max_moments]
        ])
    return None


_MOMENT_INT_FIELDS = ("t_start_ms", "t_end_ms", "page_start", "page_end",
                      "char_start", "char_end")


def _coerce_moment(item: Any) -> Optional[DescribedMoment]:
    """Turn one provider-supplied dict into a moment, or drop it.

    Providers hallucinate span fields with some regularity. Anything that does
    not coerce cleanly is dropped rather than guessed at: a caption with no span
    is useful, a caption with a fabricated timestamp is a wrong answer that
    looks right.
    """
    if not isinstance(item, dict):
        if isinstance(item, str) and item.strip():
            return DescribedMoment(kind="caption", text=item.strip())
        return None

    text = str(item.get("text") or "").strip()
    if not text:
        return None

    moment = DescribedMoment(kind=str(item.get("kind") or "caption").strip().lower(), text=text)

    for field_name in _MOMENT_INT_FIELDS:
        value = item.get(field_name)
        if value is None or isinstance(value, bool):
            continue
        try:
            setattr(moment, field_name, int(value))
        except (TypeError, ValueError):
            continue

    bbox = item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            moment.bbox = [float(v) for v in bbox]
        except (TypeError, ValueError):
            moment.bbox = None

    speaker = item.get("speaker")
    if speaker:
        moment.speaker = str(speaker)

    try:
        moment.confidence = float(item.get("confidence", 1.0))
    except (TypeError, ValueError):
        moment.confidence = 1.0

    return moment


_REFUSAL_MARKERS = (
    "i can't help", "i cannot help", "i can't assist", "i cannot assist",
    "i'm unable to", "i am unable to", "content policy", "safety policy",
    "i won't", "i will not",
)


def _looks_refused(raw: str) -> bool:
    """A provider declining is not the same as a provider failing.

    The first will decline again; the second is worth retrying. The distinction
    survives into ``media_assets.understanding_status`` as ``refused``.
    """
    lowered = (raw or "").strip().lower()[:400]
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _retry_delay(attempt: int) -> float:
    """Same curve as ``embeddings._embed_api``."""
    return 0.5 * (2 ** attempt) + random.uniform(0, 0.5)


def _is_rate_limit_error(exc: BaseException) -> bool:
    """Delegate to the one classifier that already exists, with a local copy of
    its rule as a fallback so an import failure cannot turn a transient 429 into
    a permanent give-up."""
    try:
        from mnemosyne.core.embeddings import _is_rate_limit_error as _classify
        return _classify(exc)
    except Exception:
        msg = str(exc).lower()
        return ("429" in msg or "too many requests" in msg
                or "rate limit" in msg or "rate-limit" in msg)


def _post_chat(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
) -> Tuple[Optional[str], Optional[int], Optional[BaseException]]:
    """One HTTP round trip. Returns ``(text, status, exc)``, never raises.

    Same return contract as ``local_llm._call_remote_llm_with_model`` so the
    retry decision above can be made on status alone.
    """
    status: Optional[int] = None
    try:
        import httpx
        has_httpx = True
    except ImportError:
        has_httpx = False

    try:
        if has_httpx:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(url, json=payload, headers=headers)
                status = response.status_code
                if status >= 400:
                    try:
                        response.raise_for_status()
                    except Exception as exc:
                        return (None, status, exc)
                data = response.json()
        else:
            import urllib.error
            import urllib.request
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout) as resp:
                    status = getattr(resp, "status", 200)
                    data = json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                return (None, exc.code, exc)
            except Exception as exc:
                return (None, status, exc)

        choices = data.get("choices", []) if isinstance(data, dict) else []
        if choices:
            content = (choices[0] or {}).get("message", {}).get("content")
            if content:
                return (str(content), status, None)
        return (None, status, None)
    except Exception as exc:
        return (None, status, exc)


# ---------------------------------------------------------------------------
# Optional capability preflight
# ---------------------------------------------------------------------------

def probe_model_modalities(base_url: str, api_key: str, timeout: float = 10.0) -> Dict[str, List[str]]:
    """Best-effort map of model id → declared input modalities.

    ``GET {base_url}/models`` exposes ``input_modalities`` per model on Atlas and
    OpenRouter alike — an aggregator convention, not a vendor extension. It is
    **strictly optional**: an endpoint that omits it, or exposes a different
    shape, must still work. Returns ``{}`` on anything unexpected. Degrade to
    silence, never to failure.
    """
    try:
        import httpx
    except ImportError:
        return {}
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url.rstrip('/')}/models", headers=headers)
            if response.status_code >= 400:
                return {}
            data = response.json()
    except Exception:
        return {}

    out: Dict[str, List[str]] = {}
    entries = data.get("data") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        modalities = (
            entry.get("input_modalities")
            or (entry.get("architecture") or {}).get("input_modalities")
        )
        if model_id and isinstance(modalities, list):
            out[str(model_id)] = [str(m) for m in modalities]
    return out


def warn_if_model_cannot_see(model: str, base_url: str, api_key: str) -> Optional[str]:
    """Warn when the configured model declares no image input. Returns the
    warning text, or None when there is nothing to say — including when the
    endpoint does not publish capabilities at all."""
    capabilities = probe_model_modalities(base_url, api_key)
    declared = capabilities.get(model)
    if not declared:
        return None
    if "image" in {d.lower() for d in declared}:
        return None
    message = (
        f"configured modality model {model!r} declares input modalities "
        f"{sorted(declared)} and may not accept images"
    )
    logger.warning("%s", message)
    return message


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------

@dataclass
class OpenAICompatModalityBackend:
    """Describe content via any OpenAI-compatible vision endpoint."""

    name: str = NAME
    modalities: FrozenSet[str] = frozenset({"image", "audio"})

    def describe(self, request: DescribeRequest) -> Optional[DescribeResult]:
        base_url = _cfg_str("modality_base_url").rstrip("/")
        api_key = _cfg_str("modality_api_key")
        if not base_url or not api_key:
            return None

        model = model_for(request.modality)
        if not model:
            logger.info(
                "no modality model configured for %r; skipping", request.modality
            )
            return None

        if str(request.modality).strip().lower() == "audio":
            from mnemosyne.core.modality_openai_audio import describe_audio

            request.timeout = float(request.timeout or _cfg_int("modality_timeout", 60))
            request.max_moments = int(request.max_moments or _cfg_int("modality_max_moments", 12))
            return describe_audio(request, base_url=base_url, api_key=api_key,
                                  model=model, provider=self.name)

        part = _image_part(request)
        if part is None:
            # No bytes and no public URL: nothing to send. Rung 4 of the ladder
            # -- the caller still registers the asset by reference.
            return None

        timeout = float(request.timeout or _cfg_int("modality_timeout", 60))
        max_moments = int(request.max_moments or _cfg_int("modality_max_moments", 12))

        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [{"type": "text", "text": build_prompt(request)}, part],
            }],
            "temperature": 0.2,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        url = f"{base_url}/chat/completions"

        text = None
        for attempt in range(_MAX_ATTEMPTS):
            text, status, exc = _post_chat(url, headers, payload, timeout)
            if text is not None:
                break
            if status is None:
                # No HTTP response was received, so this was a transport
                # failure. Retry it regardless of how an exception is worded.
                transient = exc is not None
            else:
                # A received HTTP status is authoritative. In particular, do
                # not let an incidental "429" in the exception text turn a
                # terminal client error into a retry.
                transient = status == 429 or 500 <= status < 600
            if transient and attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_retry_delay(attempt))
                continue
            break

        if text is None:
            return None

        if _looks_refused(text):
            return DescribeResult(
                provider=self.name, model=model, refused=True,
                warnings=["provider declined to describe this content"],
            )

        parsed = parse_response(text, max_moments)
        if parsed is None:
            return None
        summary, moments = parsed
        return DescribeResult(
            summary=summary,
            moments=moments[:max_moments],
            provider=self.name,
            model=model,
        )


def register_if_configured() -> bool:
    """Register this adapter as the default backend when configured.

    Returns whether it registered. Deliberately **not** called at import: a
    module import must never have the side effect of arming an outbound path.
    """
    if not is_configured():
        return False
    from mnemosyne.core.modality_backends import set_modality_backend
    set_modality_backend(OpenAICompatModalityBackend())
    return True
