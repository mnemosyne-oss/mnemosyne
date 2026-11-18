"""Video: sampled frames to the vision model, soundtrack to transcription.

Needs ffmpeg/ffprobe; generated clips are two seconds of testsrc and a sine.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mnemosyne.core.modality_backends import DescribeRequest
from mnemosyne.core.modality_video import VideoFrameBackend, sample_times

needs_ffmpeg = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg not installed"
)

SHOTS = json.dumps({
    "summary": "A colour test pattern with a counter.",
    "moments": [
        {"kind": "shot", "t_start_ms": 0, "t_end_ms": 1000, "text": "SMPTE colour bars, counter at 0"},
        {"kind": "shot", "t_start_ms": 1000, "t_end_ms": 2000, "text": "colour bars, counter at 1"},
    ],
})
TRANSCRIPT = {"text": "a steady tone", "segments": [{"start": 0.0, "end": 2.0, "text": "a steady tone"}]}


class _Stub:
    def __init__(self, chat_reply=SHOTS, audio_reply=None):
        self.chat, self.audio = [], []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path.endswith("/audio/transcriptions"):
                    outer.audio.append(raw)
                    payload = audio_reply or TRANSCRIPT
                else:
                    outer.chat.append(json.loads(raw))
                    payload = {"choices": [{"message": {"content": chat_reply}}]}
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    made = []

    def _make(**kw):
        s = _Stub(**kw)
        made.append(s)
        return s

    yield _make
    for s in made:
        s.close()


@pytest.fixture
def configure(monkeypatch, tmp_path):
    from mnemosyne.core.config import MnemosyneConfig

    def _configure(base_url, **extra):
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg"))
        monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
        monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
        monkeypatch.setenv("MNEMOSYNE_MODALITY_BASE_URL", base_url)
        monkeypatch.setenv("MNEMOSYNE_MODALITY_API_KEY", "test-key-not-real")
        for key, value in extra.items():
            monkeypatch.setenv(key, value)
        MnemosyneConfig.reset_instance()

    yield _configure
    MnemosyneConfig.reset_instance()


@pytest.fixture
def clip(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not installed")
    path = tmp_path / "pattern.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-shortest",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)],
        check=True, timeout=120,
    )
    return path


def test_sample_times_are_window_centers():
    assert sample_times(8000, 4) == [1000, 3000, 5000, 7000]
    assert sample_times(1000, 0) == [500]


@needs_ffmpeg
def test_frames_go_out_as_labeled_jpeg_parts_in_one_call(stub, configure, clip):
    s = stub()
    configure(s.base_url, MNEMOSYNE_MODALITY_VISION_MODEL="vision-m")
    result = VideoFrameBackend().describe(DescribeRequest(modality="video", uri=str(clip), max_moments=4, timeout=30))

    assert len(s.chat) == 1 and s.audio == []
    content = s.chat[0]["messages"][0]["content"]
    images = [p for p in content if p["type"] == "image_url"]
    labels = [p["text"] for p in content[1:] if p["type"] == "text"]
    assert len(images) == 4
    assert all(p["image_url"]["url"].startswith("data:image/jpeg;base64,/9j/") for p in images)
    assert labels[0].startswith("Frame at 00:00 (t=250 ms)")
    assert s.chat[0]["model"] == "vision-m"
    assert [(m.kind, m.t_start_ms, m.t_end_ms) for m in result.moments] == [("shot", 0, 1000), ("shot", 1000, 2000)]


@needs_ffmpeg
def test_soundtrack_is_transcribed_when_an_audio_model_is_set(stub, configure, clip):
    s = stub()
    configure(s.base_url, MNEMOSYNE_MODALITY_VISION_MODEL="v", MNEMOSYNE_MODALITY_AUDIO_MODEL="whisper-1")
    result = VideoFrameBackend().describe(DescribeRequest(modality="video", uri=str(clip), max_moments=6, timeout=30))

    assert len(s.audio) == 1 and b'filename="soundtrack.mp3"' in s.audio[0]
    kinds = [m.kind for m in result.moments]
    assert kinds.count("shot") == 2 and kinds.count("transcript") == 1
    assert [m.t_start_ms for m in result.moments] == sorted(m.t_start_ms for m in result.moments)


@needs_ffmpeg
def test_untimed_shots_are_anchored_to_frame_windows(stub, configure, clip):
    s = stub(chat_reply=json.dumps({"moments": [{"text": "bars"}, {"text": "more bars"}]}))
    configure(s.base_url, MNEMOSYNE_MODALITY_VISION_MODEL="v")
    result = VideoFrameBackend().describe(DescribeRequest(modality="video", uri=str(clip), max_moments=2, timeout=30))
    assert [(m.t_start_ms, m.t_end_ms) for m in result.moments] == [(0, 1000), (1000, 2000)]


def test_missing_ffmpeg_registers_and_says_why(stub, configure, monkeypatch):
    from mnemosyne.core import modality_video

    s = stub()
    configure(s.base_url, MNEMOSYNE_MODALITY_VISION_MODEL="v")
    monkeypatch.setattr(modality_video, "ffmpeg_available", lambda: False)
    result = VideoFrameBackend().describe(DescribeRequest(modality="video", uri="/x.mp4", max_moments=4))
    assert result.moments == [] and "ffmpeg" in result.warnings[0]
    assert s.chat == []


@needs_ffmpeg
def test_remember_media_on_a_video_stores_timed_shots_from_configuration(stub, configure, clip, tmp_path):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.media import MediaStore

    s = stub()
    configure(s.base_url, MNEMOSYNE_MODALITY_VISION_MODEL="v", MNEMOSYNE_MODALITY_AUDIO_MODEL="whisper-1")
    beam = BeamMemory(session_id="video", db_path=tmp_path / "m.db")
    result = beam.remember_media(str(clip))

    assert result.status == "ok", result.warnings
    moments = MediaStore(conn=beam.conn).get_moments(result.asset_id)
    assert {(m["kind"], m["span_kind"]) for m in moments} == {("shot", "time"), ("transcript", "time")}
    hits = beam.recall("colour bars counter", top_k=5)
    assert any("colour bars" in (h.get("content") or "") for h in hits)
