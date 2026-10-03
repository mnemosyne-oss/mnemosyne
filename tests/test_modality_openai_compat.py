"""Tests for the OpenAI-compatible vision adapter (RFC 0002 §3.2).

Every network test runs against a stub HTTP server on localhost. The live
endpoint is never contacted: verification against a real provider is a manual,
opt-in step gated on ``MNEMOSYNE_MODALITY_API_KEY`` being set in the operator's
own environment, and no key is ever committed.

The vendor-neutrality assertion lives here too, and it is mechanical rather than
aspirational: point the adapter at a second stub on a different port with only
environment variables changed, and it must work unmodified.
"""

import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from mnemosyne.core import modality_openai_compat as adapter
from mnemosyne.core.modality_backends import DescribeRequest


# ---------------------------------------------------------------------------
# Stub endpoint
# ---------------------------------------------------------------------------

class _Stub:
    """A minimal OpenAI-compatible endpoint that records what it was sent."""

    def __init__(self, replies, models_payload=None):
        self.replies = list(replies)
        self.models_payload = models_payload
        self.requests = []
        self.response_statuses = []
        self.model_probes = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if not self.path.endswith("/models") or outer.models_payload is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                outer.model_probes += 1
                body = json.dumps(outer.models_payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length)
                outer.requests.append(json.loads(raw.decode()))

                reply = outer.replies.pop(0) if outer.replies else (200, "{}")
                status, content, *raw_body = reply
                outer.response_statuses.append(status)
                if raw_body:
                    body = content.encode()
                elif status >= 400:
                    body = json.dumps({"error": content}).encode()
                else:
                    body = json.dumps(
                        {"choices": [{"message": {"content": content}}]}
                    ).encode()
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
        self.thread.join(timeout=5)


@pytest.fixture
def stub():
    made = []

    def _make(replies, models_payload=None):
        s = _Stub(replies, models_payload)
        made.append(s)
        return s

    yield _make
    for s in made:
        s.close()


@pytest.fixture
def configured(monkeypatch, tmp_path):
    """Point the adapter at a stub. Everything is an environment variable --
    that is the whole vendor-neutrality claim, made testable.

    The variables only reach the adapter through a config that was seeded
    while they were set: ``config.yaml`` wins over the environment, and
    presence beats value. Each call gets a fresh data directory so the seed
    picks them up, which is the path a new install takes.
    """
    from mnemosyne.core.config import MnemosyneConfig

    calls = {"n": 0}

    def _configure(base_url, **extra):
        calls["n"] += 1
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / f"cfg-{calls['n']}"))
        monkeypatch.setenv("MNEMOSYNE_MODALITY_ENABLED", "1")
        monkeypatch.setenv("MNEMOSYNE_MODALITY_BASE_URL", base_url)
        monkeypatch.setenv("MNEMOSYNE_MODALITY_API_KEY", "test-key-not-real")
        monkeypatch.setenv("MNEMOSYNE_MODALITY_VISION_MODEL", "some/vision-model")
        for key, value in extra.items():
            monkeypatch.setenv(key, value)
        MnemosyneConfig.reset_instance()

    yield _configure
    MnemosyneConfig.reset_instance()


def _request(**kwargs):
    kwargs.setdefault("modality", "image")
    kwargs.setdefault("uri", "https://example.test/a.png")
    return DescribeRequest(**kwargs)


_OK_REPLY = json.dumps({
    "summary": "a terminal window",
    "moments": [{"kind": "caption", "text": "a terminal window showing a test run"}],
})


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_describes_a_public_url(stub, configured):
    server = stub([(200, _OK_REPLY)])
    configured(server.base_url)

    result = adapter.OpenAICompatModalityBackend().describe(_request())

    assert result is not None
    assert result.summary == "a terminal window"
    assert [m.text for m in result.moments] == ["a terminal window showing a test run"]
    assert (result.provider, result.model) == ("openai_compat", "some/vision-model")

    sent = server.requests[0]
    assert sent["model"] == "some/vision-model"
    parts = sent["messages"][0]["content"]
    assert parts[1]["image_url"]["url"] == "https://example.test/a.png"


def test_a_public_url_never_reads_bytes(stub, configured):
    """Nothing leaves the machine that was not already public, and no local
    read happens for a URL the provider can fetch itself."""
    server = stub([(200, _OK_REPLY)])
    configured(server.base_url)
    fetched = []

    adapter.OpenAICompatModalityBackend().describe(
        _request(fetch=lambda: fetched.append(True) or b"bytes")
    )
    assert fetched == []


def test_local_content_is_sent_as_a_base64_data_part(stub, configured):
    """The reason ``DescribeRequest.fetch`` exists: without it the adapter could
    describe public URLs and nothing else -- which is to say none of the media a
    local-first memory system actually holds."""
    server = stub([(200, _OK_REPLY)])
    configured(server.base_url)

    result = adapter.OpenAICompatModalityBackend().describe(
        _request(uri="blob://sha256/" + "ab" * 32, mime="image/png",
                 fetch=lambda: b"\x89PNG fake")
    )
    assert result is not None

    url = server.requests[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG fake"


def test_local_content_without_a_fetcher_degrades_to_none(stub, configured):
    server = stub([(200, _OK_REPLY)])
    configured(server.base_url)
    assert adapter.OpenAICompatModalityBackend().describe(
        _request(uri="/tmp/local.png")
    ) is None
    assert server.requests == [], "no call should be attempted with nothing to send"


def test_a_failing_fetcher_degrades_to_none(stub, configured):
    server = stub([(200, _OK_REPLY)])
    configured(server.base_url)

    def _explode():
        raise OSError("disk gone")

    assert adapter.OpenAICompatModalityBackend().describe(
        _request(uri="/tmp/local.png", fetch=_explode)
    ) is None


# ---------------------------------------------------------------------------
# Vendor neutrality
# ---------------------------------------------------------------------------

def test_switching_endpoints_is_an_environment_change_only(stub, configured):
    """The acceptance test for vendor neutrality: same code, same call, second
    endpoint, only env vars changed."""
    first = stub([(200, _OK_REPLY)])
    second = stub([(200, json.dumps({"summary": "from the other provider"}))])
    backend = adapter.OpenAICompatModalityBackend()

    configured(first.base_url)
    assert backend.describe(_request()).summary == "a terminal window"

    configured(second.base_url, MNEMOSYNE_MODALITY_VISION_MODEL="other/model")
    assert backend.describe(_request()).summary == "from the other provider"

    assert len(first.requests) == 1 and len(second.requests) == 1
    assert second.requests[0]["model"] == "other/model"


def test_no_module_in_core_is_named_after_a_provider():
    """Sponsorship gets a recipe doc and a reference deployment; it does not get
    its name in ``mnemosyne/core/``."""
    import pathlib

    core = pathlib.Path(adapter.__file__).parent
    names = [p.name.lower() for p in core.glob("*.py")]
    for vendor in ("atlas", "openrouter", "together", "groq", "anthropic", "openai_only"):
        assert not any(vendor in n for n in names), f"core module named after {vendor}"


# ---------------------------------------------------------------------------
# Configuration gating
# ---------------------------------------------------------------------------

def test_unconfigured_makes_no_call(monkeypatch):
    from mnemosyne.core.config import MnemosyneConfig

    monkeypatch.setenv("MNEMOSYNE_MODALITY_BASE_URL", "")
    monkeypatch.setenv("MNEMOSYNE_MODALITY_API_KEY", "")
    MnemosyneConfig.reset_instance()
    try:
        assert adapter.is_configured() is False
        assert adapter.OpenAICompatModalityBackend().describe(_request()) is None
    finally:
        MnemosyneConfig.reset_instance()


def test_a_base_url_without_a_key_is_not_configured(monkeypatch, stub):
    """More likely a half-finished config than a deliberately unauthenticated
    endpoint; guessing wrong means an outbound call nobody asked for."""
    from mnemosyne.core.config import MnemosyneConfig

    server = stub([(200, _OK_REPLY)])
    monkeypatch.setenv("MNEMOSYNE_MODALITY_BASE_URL", server.base_url)
    monkeypatch.setenv("MNEMOSYNE_MODALITY_API_KEY", "")
    MnemosyneConfig.reset_instance()
    try:
        assert adapter.is_configured() is False
        assert adapter.OpenAICompatModalityBackend().describe(_request()) is None
    finally:
        MnemosyneConfig.reset_instance()
    assert server.requests == []


def test_a_missing_model_makes_no_call(stub, configured):
    server = stub([(200, _OK_REPLY)])
    configured(server.base_url, MNEMOSYNE_MODALITY_VISION_MODEL="")
    assert adapter.OpenAICompatModalityBackend().describe(_request()) is None
    assert server.requests == []


def test_registration_is_never_an_import_side_effect(monkeypatch, stub):
    """Importing a module must not arm an outbound path."""
    from mnemosyne.core.config import MnemosyneConfig
    from mnemosyne.core.modality_backends import get_modality_backend

    assert get_modality_backend("image") is None

    monkeypatch.setenv("MNEMOSYNE_MODALITY_BASE_URL", "")
    monkeypatch.setenv("MNEMOSYNE_MODALITY_API_KEY", "")
    MnemosyneConfig.reset_instance()
    try:
        assert adapter.register_if_configured() is False
        assert get_modality_backend("image") is None
    finally:
        MnemosyneConfig.reset_instance()


def test_registration_when_configured(stub, configured):
    from mnemosyne.core.modality_backends import get_modality_backend

    server = stub([(200, _OK_REPLY)])
    configured(server.base_url)
    assert adapter.register_if_configured() is True
    assert get_modality_backend("image") is not None
    assert get_modality_backend("audio") is not None, "audio goes to /audio/transcriptions"
    assert get_modality_backend("document") is None, "documents are read locally, not sent"
    assert get_modality_backend("video") is None, "video needs frame sampling, not this adapter alone"


def test_model_selection_is_per_modality(stub, configured):
    server = stub([(200, _OK_REPLY)])
    configured(server.base_url, MNEMOSYNE_MODALITY_VIDEO_MODEL="some/video-model")
    assert adapter.model_for("image") == "some/vision-model"
    assert adapter.model_for("document") == "some/vision-model"
    assert adapter.model_for("video") == "some/video-model"


# ---------------------------------------------------------------------------
# The JSON tolerance ladder
# ---------------------------------------------------------------------------

def test_strict_json_is_rung_one():
    parsed = adapter.parse_response(_OK_REPLY, max_moments=12)
    assert parsed is not None
    summary, moments = parsed
    assert summary == "a terminal window"
    assert len(moments) == 1


def test_code_fences_are_tolerated():
    parsed = adapter.parse_response(f"```json\n{_OK_REPLY}\n```", max_moments=12)
    assert parsed is not None and parsed[0] == "a terminal window"


def test_truncated_json_is_salvaged():
    """A vision model running out of tokens mid-array is routine, and a partial
    answer beats none."""
    truncated = '{"summary": "a long description of the scene", "moments": [{"text": "a'
    parsed = adapter.parse_response(truncated, max_moments=12)
    assert parsed is not None
    assert "a long description of the scene" in parsed[0]


@pytest.mark.parametrize("raw", ["", "   ", "no json here", "[]", "null"])
def test_unparseable_output_gives_up_cleanly(raw):
    assert adapter.parse_response(raw, max_moments=12) is None


def test_moments_are_capped():
    payload = json.dumps({
        "summary": "s",
        "moments": [{"kind": "caption", "text": f"moment {i}"} for i in range(50)],
    })
    _, moments = adapter.parse_response(payload, max_moments=3)
    assert len(moments) == 3


def test_span_fields_are_coerced_and_bad_ones_dropped():
    """A caption with no span is useful; a caption with a fabricated timestamp
    is a wrong answer that looks right."""
    payload = json.dumps({"moments": [{
        "kind": "shot", "text": "a shot",
        "t_start_ms": "1000", "t_end_ms": "not a number",
        "bbox": [0.1, 0.2, 0.3, 0.4], "speaker": "Alice", "confidence": "0.5",
    }]})
    _, moments = adapter.parse_response(payload, max_moments=12)
    moment = moments[0]
    assert moment.t_start_ms == 1000
    assert moment.t_end_ms is None
    assert moment.bbox == [0.1, 0.2, 0.3, 0.4]
    assert moment.speaker == "Alice"
    assert moment.confidence == 0.5


def test_a_malformed_bbox_is_dropped_not_guessed():
    payload = json.dumps({"moments": [
        {"text": "a", "bbox": [1, 2]},
        {"text": "b", "bbox": ["x", "y", "z", "w"]},
    ]})
    _, moments = adapter.parse_response(payload, max_moments=12)
    assert [m.bbox for m in moments] == [None, None]


def test_textless_moments_are_dropped():
    payload = json.dumps({"summary": "s", "moments": [{"kind": "caption"}, {"text": "  "}]})
    _, moments = adapter.parse_response(payload, max_moments=12)
    assert moments == []


def test_bare_strings_in_the_moments_array_become_captions():
    payload = json.dumps({"summary": "s", "moments": ["a plain string"]})
    _, moments = adapter.parse_response(payload, max_moments=12)
    assert [(m.kind, m.text) for m in moments] == [("caption", "a plain string")]


# ---------------------------------------------------------------------------
# Refusal, retry, and failure
# ---------------------------------------------------------------------------

def test_a_refusal_is_reported_as_refused_not_as_failure(stub, configured):
    server = stub([(200, "I can't help with identifying people in images.")])
    configured(server.base_url)

    result = adapter.OpenAICompatModalityBackend().describe(_request())
    assert result is not None
    assert result.refused is True
    assert result.moments == []
    assert result.warnings


def test_rate_limits_are_retried(stub, configured, monkeypatch):
    monkeypatch.setattr(adapter.time, "sleep", lambda _s: None)
    server = stub([(429, "slow down"), (200, _OK_REPLY)])
    configured(server.base_url)

    result = adapter.OpenAICompatModalityBackend().describe(_request())
    assert result is not None and result.summary == "a terminal window"
    assert len(server.requests) == 2


def test_server_errors_are_retried_then_given_up_on(stub, configured, monkeypatch):
    monkeypatch.setattr(adapter.time, "sleep", lambda _s: None)
    server = stub([(500, "boom"), (503, "boom"), (500, "boom")])
    configured(server.base_url)

    assert adapter.OpenAICompatModalityBackend().describe(_request()) is None
    assert len(server.requests) == 3, "bounded at three attempts"


@pytest.mark.parametrize("status", [401, 403])
def test_client_errors_are_not_retried(stub, configured, monkeypatch, status):
    """Terminal client errors are not retried."""
    monkeypatch.setattr(adapter.time, "sleep", lambda _s: None)
    server = stub([(status, "bad key"), (200, _OK_REPLY)])
    configured(server.base_url)

    assert adapter.OpenAICompatModalityBackend().describe(_request()) is None
    assert len(server.requests) == 1
    assert server.response_statuses == [status]
    assert server.replies == [(200, _OK_REPLY)]


def test_known_client_status_beats_429_in_exception_text(stub, configured, monkeypatch):
    server = stub([(403, "forbidden"), (200, _OK_REPLY)])
    configured(server.base_url)
    post_chat = adapter._post_chat

    def _post_with_misleading_exception(*args, **kwargs):
        text, status, _ = post_chat(*args, **kwargs)
        return text, status, RuntimeError("request URL included port 429")

    monkeypatch.setattr(adapter, "_post_chat", _post_with_misleading_exception)

    assert adapter.OpenAICompatModalityBackend().describe(_request()) is None
    assert len(server.requests) == 1
    assert server.response_statuses == [403]
    assert server.replies == [(200, _OK_REPLY)]


def test_received_malformed_json_is_not_retried(stub, configured, monkeypatch):
    monkeypatch.setattr(adapter.time, "sleep", lambda _s: None)
    server = stub([(200, "not JSON", True), (200, _OK_REPLY)])
    configured(server.base_url)

    assert adapter.OpenAICompatModalityBackend().describe(_request()) is None
    assert len(server.requests) == 1
    assert server.response_statuses == [200]
    assert server.replies == [(200, _OK_REPLY)]


def test_an_unreachable_endpoint_degrades_to_none(configured, monkeypatch):
    monkeypatch.setattr(adapter.time, "sleep", lambda _s: None)
    configured("http://127.0.0.1:1/v1")
    assert adapter.OpenAICompatModalityBackend().describe(_request()) is None


def test_the_adapter_never_raises(stub, configured, monkeypatch):
    """Nothing here may raise into remember()."""
    monkeypatch.setattr(adapter.time, "sleep", lambda _s: None)
    server = stub([(200, "not json at all"), (200, "{}")])
    configured(server.base_url)
    backend = adapter.OpenAICompatModalityBackend()

    assert backend.describe(_request()) is None
    assert backend.describe(_request()) is None


def test_rate_limit_classification_matches_the_existing_one():
    assert adapter._is_rate_limit_error(RuntimeError("HTTP 429")) is True
    assert adapter._is_rate_limit_error(RuntimeError("rate limit exceeded")) is True
    assert adapter._is_rate_limit_error(RuntimeError("rate limit detection failed")) is True
    assert adapter._is_rate_limit_error(RuntimeError("connection reset")) is False


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def test_the_caller_owns_the_prompt():
    assert adapter.build_prompt(_request(hint="describe the colours only")) == (
        "describe the colours only"
    )


def test_the_prompt_is_env_overridable(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_MODALITY_PROMPT", "custom for {modality}")
    assert adapter.build_prompt(_request()) == "custom for image"


def test_a_malformed_prompt_template_does_not_break_ingest(monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_MODALITY_PROMPT", "unbalanced {brace")
    assert adapter.build_prompt(_request()) == "unbalanced {brace"


def test_the_default_prompt_renders():
    prompt = adapter.build_prompt(_request(max_moments=4))
    assert "image" in prompt and "4" in prompt and "{" in prompt


# ---------------------------------------------------------------------------
# Optional capability preflight
# ---------------------------------------------------------------------------

_MODELS_PAYLOAD = {"data": [
    {"id": "some/vision-model", "input_modalities": ["text", "image"]},
    {"id": "some/text-model", "input_modalities": ["text"]},
    {"id": "nested/model", "architecture": {"input_modalities": ["text", "image"]}},
]}


def test_preflight_reads_input_modalities(stub):
    server = stub([], models_payload=_MODELS_PAYLOAD)
    caps = adapter.probe_model_modalities(server.base_url, "k")
    assert caps["some/vision-model"] == ["text", "image"]
    assert caps["nested/model"] == ["text", "image"]


def test_preflight_warns_only_about_a_text_only_model(stub):
    server = stub([], models_payload=_MODELS_PAYLOAD)
    assert adapter.warn_if_model_cannot_see("some/vision-model", server.base_url, "k") is None
    warning = adapter.warn_if_model_cannot_see("some/text-model", server.base_url, "k")
    assert warning is not None and "may not accept images" in warning


def test_preflight_is_silent_on_an_endpoint_that_does_not_publish_capabilities(stub):
    """It must stay strictly optional. Degrade to silence, never to failure."""
    server = stub([])  # no /models route at all
    assert adapter.probe_model_modalities(server.base_url, "k") == {}
    assert adapter.warn_if_model_cannot_see("anything", server.base_url, "k") is None


def test_preflight_tolerates_an_unexpected_shape(stub):
    server = stub([], models_payload={"models": "not the expected shape"})
    assert adapter.probe_model_modalities(server.base_url, "k") == {}


def test_preflight_tolerates_an_unreachable_endpoint():
    assert adapter.probe_model_modalities("http://127.0.0.1:1/v1", "k", timeout=1.0) == {}


def test_describe_does_not_preflight(stub, configured):
    """The preflight is an operator-facing diagnostic, not a per-call tax."""
    server = stub([(200, _OK_REPLY)], models_payload=_MODELS_PAYLOAD)
    configured(server.base_url)
    adapter.OpenAICompatModalityBackend().describe(_request())
    assert server.model_probes == 0


# ---------------------------------------------------------------------------
# Configuration alone is enough (no host registration)
# ---------------------------------------------------------------------------

def test_configuration_alone_registers_the_adapter(stub, configured):
    """The documented setup is env/config only. Before this, nothing ever
    called ``register_if_configured()``, so a fully configured install sent
    no request and every asset came back ``unavailable``."""
    from mnemosyne.core.modality_backends import call_modality_describe, get_modality_backend

    s = stub([(200, _OK_REPLY)])
    configured(s.base_url)
    assert get_modality_backend("image") is None

    result = call_modality_describe(_request())

    assert result is not None and result.summary == "a terminal window"
    assert len(s.requests) == 1


def test_remember_media_describes_a_local_image_from_configuration_alone(stub, configured, tmp_path, monkeypatch):
    from mnemosyne.core.beam import BeamMemory

    s = stub([(200, _OK_REPLY)])
    configured(s.base_url)
    monkeypatch.setenv("MNEMOSYNE_BLOB_DIR", str(tmp_path / "blobs"))
    image = tmp_path / "shot.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    beam = BeamMemory(session_id="autoreg", db_path=tmp_path / "m.db")

    result = beam.remember_media(str(image))

    assert result.status == "ok"
    assert len(result.moment_ids) == 1
    part = s.requests[0]["messages"][0]["content"][1]
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


def test_autoregistration_leaves_a_host_default_alone(stub, configured):
    from mnemosyne.core.modality_backends import (
        CallableModalityBackend, call_modality_describe, get_modality_backend,
        set_modality_backend,
    )

    s = stub([(200, _OK_REPLY)])
    configured(s.base_url)
    host = CallableModalityBackend(name="host", func=lambda r: None,
                                   modalities=frozenset({"audio"}))
    set_modality_backend(host)

    assert call_modality_describe(_request()) is None
    assert s.requests == []
    assert get_modality_backend("audio") is host


def test_autoregistration_never_runs_with_the_gate_off(stub, configured, monkeypatch):
    from mnemosyne.core.config import MnemosyneConfig
    from mnemosyne.core.modality_backends import call_modality_describe, get_modality_backend

    s = stub([(200, _OK_REPLY)])
    configured(s.base_url, MNEMOSYNE_MODALITY_ENABLED="0")
    MnemosyneConfig.reset_instance()

    assert call_modality_describe(_request()) is None
    assert get_modality_backend("image") is None
    assert s.requests == []
