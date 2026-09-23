"""Audio understanding through ``POST /audio/transcriptions`` (verbose_json).

Every test runs against a localhost stub that records the multipart upload.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mnemosyne.core.modality_backends import DescribeRequest
from mnemosyne.core.modality_openai_audio import _group_segments, describe_audio

VERBOSE = {
    "text": "Welcome to the standup. Alice ships the parser. Bob fixes recall.",
    "segments": [
        {"start": 0.0, "end": 2.4, "text": "Welcome to the standup."},
        {"start": 2.4, "end": 5.1, "text": "Alice ships the parser."},
        {"start": 5.1, "end": 8.0, "text": "Bob fixes recall."},
    ],
}


class _Stub:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": raw})
                status, payload = outer.replies.pop(0) if outer.replies else (200, {})
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

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

    def _make(replies):
        s = _Stub(replies)
        made.append(s)
        return s

    yield _make
    for s in made:
        s.close()


def _req(data=b"ID3fakeaudio", **kw):
    kw.setdefault("modality", "audio")
    kw.setdefault("uri", "/music/standup.mp3")
    kw.setdefault("max_moments", 12)
    kw.setdefault("timeout", 10)
    return DescribeRequest(fetch=lambda: data, **kw)


def test_verbose_json_segments_become_timed_transcript_moments(stub):
    s = stub([(200, VERBOSE)])
    result = describe_audio(_req(), base_url=s.base_url, api_key="k", model="whisper-1", provider="p")

    assert result is not None and not result.refused
    assert [m.kind for m in result.moments] == ["transcript"] * 3
    assert (result.moments[1].t_start_ms, result.moments[1].t_end_ms) == (2400, 5100)
    assert result.moments[2].text == "Bob fixes recall."
    assert result.summary.startswith("Welcome to the standup.")

    sent = s.requests[0]
    assert sent["path"] == "/v1/audio/transcriptions"
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
    body = sent["body"]
    assert b'name="model"\r\n\r\nwhisper-1' in body
    assert b'name="response_format"\r\n\r\nverbose_json' in body
    assert b'filename="standup.mp3"' in body and b"Content-Type: audio/mpeg" in body
    assert b"ID3fakeaudio" in body


def test_segments_fold_into_at_most_max_moments_contiguous_windows():
    segs = [{"start": i, "end": i + 1, "text": f"w{i}"} for i in range(10)]
    moments = _group_segments(segs, 3)
    assert len(moments) == 3
    assert moments[0].t_start_ms == 0 and moments[-1].t_end_ms == 10000
    assert " ".join(m.text for m in moments) == " ".join(f"w{i}" for i in range(10))


def test_plain_text_response_is_kept_as_one_whole_clip_transcript(stub):
    s = stub([(200, {"text": "just the words"})])
    result = describe_audio(_req(), base_url=s.base_url, api_key="k", model="m", provider="p")
    assert [m.text for m in result.moments] == ["just the words"]
    assert result.moments[0].t_start_ms is None
    assert result.warnings == ["provider returned no timed segments"]


def test_hint_is_sent_as_the_vocabulary_prompt(stub):
    s = stub([(200, VERBOSE)])
    describe_audio(_req(hint="Mnemosyne, BEAM"), base_url=s.base_url, api_key="k", model="m", provider="p")
    assert b'name="prompt"\r\n\r\nMnemosyne, BEAM' in s.requests[0]["body"]


def test_oversize_audio_is_refused_locally(stub):
    from mnemosyne.core import modality_openai_audio as mod

    s = stub([(200, VERBOSE)])
    big = b"x" * (mod.MAX_AUDIO_BYTES + 1)
    assert describe_audio(_req(data=big), base_url=s.base_url, api_key="k", model="m", provider="p") is None
    assert s.requests == []


def test_terminal_client_error_is_not_retried(stub):
    s = stub([(400, {"error": "bad"}), (200, VERBOSE)])
    assert describe_audio(_req(), base_url=s.base_url, api_key="k", model="m", provider="p") is None
    assert len(s.requests) == 1


def test_server_error_is_retried(stub, monkeypatch):
    from mnemosyne.core import modality_openai_audio as mod

    monkeypatch.setattr(mod, "_retry_delay", lambda attempt: 0)
    s = stub([(503, {"error": "busy"}), (200, VERBOSE)])
    result = describe_audio(_req(), base_url=s.base_url, api_key="k", model="m", provider="p")
    assert result is not None and len(s.requests) == 2


def test_remember_media_stores_timed_transcript_moments_from_configuration(stub, tmp_path, monkeypatch):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.config import MnemosyneConfig

    s = stub([(200, VERBOSE)])
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
    monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
    monkeypatch.setenv("MNEMOSYNE_MODALITY_BASE_URL", s.base_url)
    monkeypatch.setenv("MNEMOSYNE_MODALITY_API_KEY", "test-key-not-real")
    monkeypatch.setenv("MNEMOSYNE_MODALITY_AUDIO_MODEL", "whisper-1")
    MnemosyneConfig.reset_instance()
    try:
        clip = tmp_path / "standup.mp3"
        clip.write_bytes(b"ID3" + b"\x00" * 64)
        beam = BeamMemory(session_id="audio", db_path=tmp_path / "m.db")

        result = beam.remember_media(str(clip))

        assert result.status == "ok", result.warnings
        assert len(result.memory_ids) == 3
        moments = beam.media.get_moments(result.asset_id) if getattr(beam, "media", None) else None
        if moments is None:
            from mnemosyne.core.media import MediaStore
            moments = MediaStore(conn=beam.conn).get_moments(result.asset_id)
        assert [(m["kind"], m["span_kind"], m["t_start_ms"]) for m in moments] == [
            ("transcript", "time", 0), ("transcript", "time", 2400), ("transcript", "time", 5100),
        ]
        hits = beam.recall("who ships the parser", top_k=5)
        assert any("Alice ships the parser" in (h.get("content") or "") for h in hits)
    finally:
        MnemosyneConfig.reset_instance()


def test_audio_without_an_audio_model_stays_unavailable(stub, tmp_path, monkeypatch):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.config import MnemosyneConfig

    s = stub([(200, VERBOSE)])
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
    monkeypatch.setenv("MNEMOSYNE_MODALITY_BASE_URL", s.base_url)
    monkeypatch.setenv("MNEMOSYNE_MODALITY_API_KEY", "k")
    monkeypatch.setenv("MNEMOSYNE_MODALITY_VISION_MODEL", "vision-only")
    MnemosyneConfig.reset_instance()
    try:
        clip = tmp_path / "a.wav"
        clip.write_bytes(b"RIFF" + b"\x00" * 64)
        result = BeamMemory(session_id="a", db_path=tmp_path / "m.db").remember_media(str(clip))
        assert result.status == "unavailable"
        assert s.requests == []
    finally:
        MnemosyneConfig.reset_instance()
