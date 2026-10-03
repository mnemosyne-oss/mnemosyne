"""Audio understanding over the OpenAI-compatible transcription endpoint.

``POST {base_url}/audio/transcriptions`` is the one audio call the protocol
standardizes, and every provider that speaks it (OpenAI, Groq, a local
whisper server, most gateways) accepts the same multipart form. With
``response_format=verbose_json`` it returns timed segments, which map straight
onto ``transcript`` moments with a time span -- so a recalled line of speech
comes back with where it was said.

Like the image path this module never raises into ``remember_media``: every
failure is ``None`` and the asset stays registered by reference.
"""

from __future__ import annotations

import json
import logging
import mimetypes
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from mnemosyne.core.modality_backends import DescribedMoment, DescribeRequest, DescribeResult

logger = logging.getLogger(__name__)

#: OpenAI's documented upload limit for the transcription endpoint. Larger
#: files are refused locally rather than uploaded and rejected remotely.
MAX_AUDIO_BYTES = 25 * 1024 * 1024
_MAX_ATTEMPTS = 3
_SUMMARY_CHARS = 280


def _retry_delay(attempt: int) -> float:
    return min(8.0, 0.5 * (2 ** attempt))


def _audio_bytes(request: DescribeRequest, timeout: float) -> Optional[bytes]:
    """The bytes to upload. Local and ``blob://`` content comes through
    ``request.fetch``; a public URL has to be downloaded, since the endpoint
    takes a file, not a link."""
    raw: Optional[bytes] = None
    if request.fetch is not None:
        try:
            raw = request.fetch()
        except Exception:
            logger.info("audio fetch failed", exc_info=True)
            raw = None
    if raw is None and (request.uri or "").lower().startswith(("http://", "https://")):
        raw = _download(request.uri, timeout)
    if raw is not None and len(raw) > MAX_AUDIO_BYTES:
        logger.info("audio exceeds the %d byte transcription limit; skipping", MAX_AUDIO_BYTES)
        return None
    return raw or None


def _download(url: str, timeout: float) -> Optional[bytes]:
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - caller-named public media
            raw = resp.read(MAX_AUDIO_BYTES + 1)
        return raw if len(raw) <= MAX_AUDIO_BYTES else None
    except Exception:
        logger.info("audio download failed", exc_info=True)
        return None


def _filename(request: DescribeRequest) -> str:
    """The upload needs a name whose extension matches the bytes; providers
    sniff the container from it."""
    uri = str(request.uri or "")
    tail = uri.rsplit("/", 1)[-1].split("?", 1)[0]
    if "." in tail:
        return tail
    ext = mimetypes.guess_extension(request.mime or "") or ".mp3"
    return f"audio{ext}"


def _multipart(fields: Dict[str, str], filename: str, data: bytes, mime: str) -> Tuple[bytes, str]:
    boundary = f"mnemosyne-{uuid.uuid4().hex}"
    out = bytearray()
    for name, value in fields.items():
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n").encode()
    out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n").encode()
    out += data + f"\r\n--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def _post_transcription(url: str, api_key: str, body: bytes, content_type: str,
                        timeout: float) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[BaseException]]:
    """One round trip. Returns ``(json, status, exc)``; never raises."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": content_type, "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-configured endpoint
            status = getattr(resp, "status", 200)
            payload = json.loads(resp.read().decode() or "{}")
            return (payload if isinstance(payload, dict) else None, status, None)
    except urllib.error.HTTPError as exc:
        return (None, exc.code, exc)
    except Exception as exc:
        return (None, None, exc)


def _group_segments(segments: List[Dict[str, Any]], max_moments: int) -> List[DescribedMoment]:
    """Fold timed segments into at most ``max_moments`` contiguous windows.

    Whisper-style segments are a few seconds each; one memory row per segment
    would flood recall with fragments. Contiguous windows keep every word and
    every timestamp while bounding the row count.
    """
    clean = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        try:
            start = int(round(float(seg.get("start")) * 1000))
            end = int(round(float(seg.get("end")) * 1000))
        except (TypeError, ValueError):
            continue
        if text and end >= start >= 0:
            clean.append((start, end, text))
    if not clean:
        return []
    cap = max(1, int(max_moments or 1))
    per = -(-len(clean) // cap)
    moments = []
    for i in range(0, len(clean), per):
        chunk = clean[i:i + per]
        moments.append(DescribedMoment(
            kind="transcript",
            text=" ".join(t for _, _, t in chunk),
            t_start_ms=chunk[0][0],
            t_end_ms=chunk[-1][1],
        ))
    return moments


def describe_audio(request: DescribeRequest, *, base_url: str, api_key: str,
                   model: str, provider: str) -> Optional[DescribeResult]:
    """Transcribe ``request`` into timed ``transcript`` moments."""
    timeout = float(request.timeout or 60)
    raw = _audio_bytes(request, timeout)
    if raw is None:
        return None

    mime = request.mime or mimetypes.guess_type(_filename(request))[0] or "application/octet-stream"
    fields = {"model": model, "response_format": "verbose_json"}
    if request.hint:
        # The endpoint's ``prompt`` biases vocabulary (names, jargon), which is
        # what a caller hint is for here.
        fields["prompt"] = str(request.hint)[:800]
    body, content_type = _multipart(fields, _filename(request), raw, mime)
    url = f"{base_url.rstrip('/')}/audio/transcriptions"

    payload = None
    for attempt in range(_MAX_ATTEMPTS):
        payload, status, exc = _post_transcription(url, api_key, body, content_type, timeout)
        if payload is not None:
            break
        transient = (exc is not None) if status is None else (status == 429 or 500 <= status < 600)
        if transient and attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_retry_delay(attempt))
            continue
        break
    if payload is None:
        return None

    text = str(payload.get("text") or "").strip()
    segments = payload.get("segments") if isinstance(payload.get("segments"), list) else []
    moments = _group_segments(segments, request.max_moments)
    if not moments and text:
        # Providers that ignore verbose_json still return the text; keep it as
        # one transcript spanning the whole clip rather than dropping it.
        moments = [DescribedMoment(kind="transcript", text=text)]
    if not moments:
        return None

    summary = text[:_SUMMARY_CHARS] if text else None
    return DescribeResult(
        summary=summary,
        moments=moments,
        provider=provider,
        model=model,
        warnings=[] if segments else ["provider returned no timed segments"],
    )
