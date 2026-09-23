"""Video understanding: sampled frames to the vision model, audio to transcription.

Chat completions have no portable video part. Each vendor that accepts video
does it differently, and the vendor-neutral rule says we do not special-case
one. What every OpenAI-compatible vision endpoint does accept is several
``image_url`` parts in one message, so video is handled as what it is: frames
over time, plus a soundtrack.

1. ``ffprobe`` reads the duration.
2. ``ffmpeg`` samples up to ``MAX_FRAMES`` evenly spaced frames as JPEG.
3. One chat call sends every frame, each labeled with its timestamp, and asks
   for ``shot`` moments with ``t_start_ms``/``t_end_ms``.
4. If an audio model is configured, ``ffmpeg`` extracts the soundtrack and it
   goes through :mod:`modality_openai_audio`, adding ``transcript`` moments.

``ffmpeg`` is an external binary, not a Python dependency. Without it the
asset is registered and the result says why nothing was described. Nothing
here raises into ``remember_media``.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import FrozenSet, List, Optional, Tuple

from mnemosyne.core.modality_backends import DescribedMoment, DescribeRequest, DescribeResult

logger = logging.getLogger(__name__)

NAME = "openai_compat_video"
#: Frames per video sent to the vision model. Bounds tokens and cost.
MAX_FRAMES = 8
#: Longest edge of a sampled frame, in pixels.
FRAME_EDGE = 768
_FFMPEG_TIMEOUT = 120

VIDEO_PROMPT = (
    "These are {n} frames sampled in order from one video, each labeled with its "
    "timestamp. Describe what happens. Reply with JSON only: "
    '{{"summary": "<one sentence>", "moments": [{{"kind": "shot", "t_start_ms": <int>, '
    '"t_end_ms": <int>, "text": "<what is on screen, including any readable text>"}}]}}. '
    "Use the frame timestamps for t_start_ms and t_end_ms; merge consecutive frames "
    "that show the same scene into one shot. At most {max_moments} moments."
)


def ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg")) and bool(shutil.which("ffprobe"))


def _stamp(ms: int) -> str:
    seconds = ms // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _duration_ms(source: str) -> Optional[int]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", source],
            capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT, check=False,
        )
        seconds = float(out.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None
    return int(seconds * 1000) if seconds > 0 else None


def _frame_jpeg(source: str, at_ms: int) -> Optional[bytes]:
    scale = f"scale='if(gt(iw,ih),min({FRAME_EDGE},iw),-2)':'if(gt(iw,ih),-2,min({FRAME_EDGE},ih))'"
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-ss", f"{at_ms / 1000:.3f}", "-i", source,
             "-frames:v", "1", "-vf", scale, "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "pipe:1"],
            capture_output=True, timeout=_FFMPEG_TIMEOUT, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 and out.stdout.startswith(b"\xff\xd8") else None


def _soundtrack_mp3(source: str, workdir: str) -> Optional[bytes]:
    """Mono 16 kHz MP3 of the audio track, or None when there is none."""
    target = os.path.join(workdir, "soundtrack.mp3")
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", source, "-vn", "-ac", "1", "-ar", "16000",
             "-b:a", "48k", target],
            capture_output=True, timeout=_FFMPEG_TIMEOUT * 5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not os.path.exists(target) or os.path.getsize(target) == 0:
        return None
    with open(target, "rb") as fh:
        return fh.read()


def sample_times(duration_ms: int, count: int) -> List[int]:
    """Frame centers of ``count`` equal windows across the clip."""
    count = max(1, count)
    return [int(duration_ms * (i + 0.5) / count) for i in range(count)]


def _describe_frames(frames: List[Tuple[int, bytes]], duration_ms: int, request: DescribeRequest,
                     *, base_url: str, api_key: str, model: str) -> Tuple[Optional[str], List[DescribedMoment], bool]:
    """One chat call over all frames. Returns ``(summary, shots, refused)``."""
    import base64

    from mnemosyne.core.modality_openai_compat import (
        _MAX_ATTEMPTS, _looks_refused, _post_chat, _retry_delay, parse_response,
    )

    prompt = request.hint or VIDEO_PROMPT.format(n=len(frames), max_moments=request.max_moments)
    content = [{"type": "text", "text": prompt}]
    for at_ms, jpeg in frames:
        content.append({"type": "text", "text": f"Frame at {_stamp(at_ms)} (t={at_ms} ms)"})
        content.append({"type": "image_url", "image_url": {
            "url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii"),
            "detail": request.detail,
        }})
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "temperature": 0.2, "stream": False}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    url = f"{base_url.rstrip('/')}/chat/completions"

    text = None
    for attempt in range(_MAX_ATTEMPTS):
        text, status, exc = _post_chat(url, headers, payload, float(request.timeout or 60))
        if text is not None:
            break
        transient = (exc is not None) if status is None else (status == 429 or 500 <= status < 600)
        if transient and attempt < _MAX_ATTEMPTS - 1:
            time.sleep(_retry_delay(attempt))
            continue
        break
    if text is None:
        return None, [], False
    if _looks_refused(text):
        return None, [], True
    parsed = parse_response(text, request.max_moments)
    if parsed is None:
        return None, [], False
    summary, moments = parsed

    # Anchor anything the model left untimed to the frame windows, in order,
    # so every shot is still locatable in the clip.
    window = duration_ms // max(1, len(frames))
    shots = []
    for index, moment in enumerate(moments):
        moment.kind = "shot"
        if moment.t_start_ms is None:
            slot = min(index, len(frames) - 1)
            moment.t_start_ms = slot * window
            moment.t_end_ms = min(duration_ms, (slot + 1) * window)
        elif moment.t_end_ms is None or moment.t_end_ms < moment.t_start_ms:
            moment.t_end_ms = min(duration_ms, moment.t_start_ms + window)
        moment.t_start_ms = max(0, min(moment.t_start_ms, duration_ms))
        moment.t_end_ms = max(moment.t_start_ms, min(moment.t_end_ms, duration_ms))
        shots.append(moment)
    return summary, shots, False


@dataclass
class VideoFrameBackend:
    """Frames to the configured vision model, soundtrack to transcription."""

    name: str = NAME
    modalities: FrozenSet[str] = field(default_factory=lambda: frozenset({"video"}))

    def describe(self, request: DescribeRequest) -> Optional[DescribeResult]:
        from mnemosyne.core.modality_openai_compat import _cfg_int, _cfg_str, model_for

        base_url = _cfg_str("modality_base_url")
        api_key = _cfg_str("modality_api_key")
        if not base_url or not api_key:
            return None
        frame_model = model_for("video") or model_for("image")
        audio_model = model_for("audio")
        if not frame_model and not audio_model:
            return None
        if not ffmpeg_available():
            return DescribeResult(provider=self.name,
                                  warnings=["video understanding needs ffmpeg and ffprobe on PATH"])

        request.timeout = float(request.timeout or _cfg_int("modality_timeout", 60))
        request.max_moments = int(request.max_moments or _cfg_int("modality_max_moments", 12))

        with tempfile.TemporaryDirectory(prefix="mnemosyne-video-") as workdir:
            source = self._local_source(request, workdir)
            if source is None:
                return None
            duration_ms = _duration_ms(source)
            if duration_ms is None:
                return DescribeResult(provider=self.name, warnings=["could not read the video duration"])

            warnings: List[str] = []
            summary: Optional[str] = None
            shots: List[DescribedMoment] = []
            refused = False
            if frame_model:
                count = min(MAX_FRAMES, max(1, request.max_moments))
                frames = [(t, jpeg) for t in sample_times(duration_ms, count)
                          if (jpeg := _frame_jpeg(source, t)) is not None]
                if frames:
                    summary, shots, refused = _describe_frames(
                        frames, duration_ms, request,
                        base_url=base_url, api_key=api_key, model=frame_model,
                    )
                else:
                    warnings.append("no frames could be decoded")

            transcript: List[DescribedMoment] = []
            if audio_model:
                audio = _soundtrack_mp3(source, workdir)
                if audio is not None:
                    from mnemosyne.core.modality_openai_audio import describe_audio

                    budget = max(1, request.max_moments - len(shots))
                    heard = describe_audio(
                        DescribeRequest(modality="audio", uri="soundtrack.mp3", mime="audio/mpeg",
                                        max_moments=budget, timeout=request.timeout,
                                        fetch=lambda: audio),
                        base_url=base_url, api_key=api_key, model=audio_model, provider=self.name,
                    )
                    if heard is not None:
                        transcript = heard.moments
                        summary = summary or heard.summary

        if refused and not transcript:
            return DescribeResult(provider=self.name, model=frame_model, refused=True,
                                  warnings=["provider declined to describe this content"])
        moments = sorted(shots + transcript, key=lambda m: (m.t_start_ms or 0, m.kind))
        if not moments and not summary:
            return DescribeResult(provider=self.name, warnings=warnings) if warnings else None
        return DescribeResult(
            summary=summary, moments=moments[:request.max_moments], provider=self.name,
            model=frame_model or audio_model, warnings=warnings,
        )

    @staticmethod
    def _local_source(request: DescribeRequest, workdir: str) -> Optional[str]:
        """A path or URL ffmpeg can read. Local files and public URLs are read
        in place, so a large video never passes through the byte-fetch cap;
        only ``blob://`` content is materialized."""
        uri = str(request.uri or "")
        if uri.lower().startswith(("http://", "https://")) or os.path.isfile(uri):
            return uri
        if request.fetch is None:
            return None
        try:
            raw = request.fetch()
        except Exception:
            logger.info("video fetch failed", exc_info=True)
            return None
        if not raw:
            return None
        path = os.path.join(workdir, "input")
        with open(path, "wb") as fh:
            fh.write(raw)
        return path

