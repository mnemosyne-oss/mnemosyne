"""Request-level coverage: every outbound HTTP path sends an application UA.

Each test drives the real product code and asserts on the *requests it
actually constructs* — not on source text. Two harnesses are used:

* a live ``ThreadingHTTPServer`` for the modules that accept a base URL, so
  the httpx and urllib transports run for real over a socket; and
* a recording ``urlopen`` stub with a path-aware router for the modules that
  hardcode their endpoint, which lets multi-node extractors (peers →
  sessions → messages) be walked to their last request.

Both harnesses assert the exact ``Mnemosyne/<version>`` value, and both
assert on *every* request they see, so a later request node cannot pass by
inheriting coverage from the first.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest

import mnemosyne
from mnemosyne.core import llm_conflict_detector, modality_openai_compat
from mnemosyne.core.importers import (
    cognee,
    hindsight,
    honcho,
    letta,
    mem0,
    supermemory,
    zep,
)
from mnemosyne.core.modality_backends import DescribeRequest
from mnemosyne.core.user_agent import application_user_agent
from mnemosyne.extraction.client import ExtractionClient

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "mnemosyne"

OUTBOUND_MARKERS = (
    "urllib.request.urlopen",
    "urllib.request.Request(",
    "httpx.Client(",
)


def expected_user_agent() -> str:
    return f"Mnemosyne/{mnemosyne.__version__}"


def assert_all_requests_carry_ua(requests):
    """Every recorded request must carry the application User-Agent."""
    assert requests, "no outbound request was made"
    for entry in requests:
        headers = {key.lower(): value for key, value in entry["headers"].items()}
        assert headers.get("user-agent") == expected_user_agent(), (
            f"{entry.get('url')} sent {headers.get('user-agent')!r}"
        )


# ---------------------------------------------------------------------------
# Harness 1: live HTTP server (real sockets, both transports)
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    recorder = None
    responder = None

    def _serve(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        type(self).recorder.append(
            {
                "method": self.command,
                "path": self.path,
                "url": self.path,
                "headers": dict(self.headers),
            }
        )
        status, body = type(self).responder(self.path)
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = _serve
    do_POST = _serve

    def log_message(self, *args):  # keep pytest output clean
        pass


class LiveServer:
    """A real HTTP endpoint that records every request it receives."""

    def __init__(self, responder):
        self.requests = []
        _Handler.recorder = self.requests
        _Handler.responder = responder
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def respond(self, responder):
        _Handler.responder = responder

    def close(self):
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


@pytest.fixture
def api_server():
    """A live OpenAI-compatible endpoint returning an empty JSON array."""
    server = LiveServer(lambda path: (200, {"choices": [{"message": {"content": "[]"}}]}))
    yield server
    server.close()


def _block_httpx():
    """Force the urllib fallback by making ``import httpx`` fail."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx" or name.startswith("httpx."):
            raise ImportError("httpx blocked for this test")
        return real_import(name, *args, **kwargs)

    return patch("builtins.__import__", fake_import)


# ---------------------------------------------------------------------------
# Harness 2: recording urlopen stub (multi-node extractors)
# ---------------------------------------------------------------------------


class _NoRoute:
    """Marker returned by a router for a path it does not expect."""


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class RecordingUrlopen:
    """Record every ``urlopen`` request and answer by path.

    An unrouted path raises, so a module cannot silently make a request
    that the test never asserts on.
    """

    def __init__(self, router=None):
        self.router = router or (lambda path: [])
        self.requests = []

    def __call__(self, req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        path = urlsplit(url).path
        self.requests.append({"url": url, "path": path, "headers": dict(req.headers)})
        payload = self.router(path)
        if payload is _NoRoute:
            raise AssertionError(f"unexpected request path: {url}")
        return _FakeResponse(payload)


def drive_urlopen(router=None):
    """Patch ``urllib.request.urlopen``; returns the recorder (already active)."""
    recorder = RecordingUrlopen(router)
    patcher = patch("urllib.request.urlopen", recorder)
    patcher.start()
    return recorder, patcher


# ---------------------------------------------------------------------------
# Live-server coverage: both transports over real sockets
# ---------------------------------------------------------------------------


def test_probe_model_modalities_sends_application_user_agent(api_server):
    pytest.importorskip("httpx", reason="the httpx branch requires httpx installed")
    api_server.respond(
        lambda path: (
            200,
            {"data": [{"id": "gpt-vision", "input_modalities": ["text", "image"]}]},
        )
    )
    result = modality_openai_compat.probe_model_modalities(api_server.base_url, "k")
    assert result == {"gpt-vision": ["text", "image"]}, result
    assert api_server.requests[0]["path"] == "/models"
    assert_all_requests_carry_ua(api_server.requests)


def _spy_on_httpx_client():
    """Record ``httpx.Client`` instantiation so a test can prove the httpx
    branch actually ran, rather than silently falling back to urllib."""
    import httpx

    calls = []
    real_client = httpx.Client

    class _Spy(real_client):
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)
            super().__init__(*args, **kwargs)

    return calls, patch.object(httpx, "Client", _Spy)


def _describe(request, server):
    def fake_cfg(key, default=""):
        return {
            "modality_base_url": server.base_url,
            "modality_api_key": "test-key",
        }.get(key, default)

    with patch.object(modality_openai_compat, "_cfg_str", fake_cfg), patch.object(
        modality_openai_compat, "model_for", lambda modality: "gpt-vision"
    ):
        modality_openai_compat.OpenAICompatModalityBackend().describe(request)


def test_describe_sends_application_user_agent_via_httpx(api_server):
    """The httpx branch of ``_post_chat`` (httpx is installed in CI)."""
    pytest.importorskip("httpx", reason="the httpx branch requires httpx installed")
    calls, spy = _spy_on_httpx_client()
    with spy:
        _describe(
            DescribeRequest(
                modality="image", uri="http://example.test/p.png", timeout=5.0
            ),
            api_server,
        )
    assert calls, "httpx.Client was never instantiated: the httpx branch did not run"
    assert_all_requests_carry_ua(api_server.requests)


def test_describe_sends_application_user_agent_via_urllib_fallback(api_server):
    """The urllib branch of ``_post_chat`` when httpx cannot be imported."""
    with _block_httpx():
        _describe(
            DescribeRequest(
                modality="image", uri="http://example.test/p.png", timeout=5.0
            ),
            api_server,
        )
    assert_all_requests_carry_ua(api_server.requests)


def _set_conflict_target(monkeypatch, server):
    monkeypatch.setattr(llm_conflict_detector, "CONFLICT_LLM_BASE_URL", server.base_url)
    monkeypatch.setattr(llm_conflict_detector, "CONFLICT_LLM_API_KEY", "")
    monkeypatch.setattr(llm_conflict_detector, "CONFLICT_LLM_MODEL", "test-model")


def test_conflict_detector_sends_application_user_agent_via_httpx(
    monkeypatch, api_server
):
    pytest.importorskip("httpx", reason="the httpx branch requires httpx installed")
    api_server.respond(
        lambda path: (200, {"choices": [{"message": {"content": '{"conflict": false}'}}]})
    )
    _set_conflict_target(monkeypatch, api_server)
    calls, spy = _spy_on_httpx_client()
    with spy:
        llm_conflict_detector._call_conflict_llm_with_retry("are these two claims in conflict?")
    assert calls, "httpx.Client was never instantiated: the httpx branch did not run"
    assert_all_requests_carry_ua(api_server.requests)


def test_conflict_detector_sends_application_user_agent_via_urllib_fallback(
    monkeypatch, api_server
):
    api_server.respond(
        lambda path: (200, {"choices": [{"message": {"content": '{"conflict": false}'}}]})
    )
    _set_conflict_target(monkeypatch, api_server)
    with _block_httpx():
        llm_conflict_detector._call_conflict_llm_with_retry("are these two claims in conflict?")
    assert_all_requests_carry_ua(api_server.requests)


def test_extraction_client_sends_application_user_agent(api_server):
    client = ExtractionClient(api_key="test-key", base_url=api_server.base_url)
    client._call_api("model", [{"role": "user", "content": "hi"}], 0.0, 16)
    assert_all_requests_carry_ua(api_server.requests)


def test_auto_save_openwebui_api_get_sends_application_user_agent():
    from mnemosyne.integrations import auto_save_openwebui

    recorder, patcher = drive_urlopen(lambda path: {"ok": True})
    try:
        auto_save_openwebui._api_get("http://openwebui.test/api/health", "k")
    finally:
        patcher.stop()
    assert_all_requests_carry_ua(recorder.requests)


# ---------------------------------------------------------------------------
# Importer coverage, including later request nodes
# ---------------------------------------------------------------------------


def _honcho_router(path):
    """Walk all three Honcho nodes: peers → sessions → messages."""
    if path == "/peers":
        return [{"peer_id": "p1"}]
    if path == "/peers/p1/sessions":
        return [{"session_id": "s1"}]
    if path == "/sessions/s1/messages":
        return [{"content": "hello"}]
    return _NoRoute


def test_honcho_extract_walks_every_request_node_with_user_agent():
    """Honcho builds three requests; the last one must carry the UA too."""
    recorder, patcher = drive_urlopen(_honcho_router)
    try:
        items = honcho.HonchoImporter()._extract_via_rest()
    finally:
        patcher.stop()

    assert [r["path"] for r in recorder.requests] == [
        "/peers",
        "/peers/p1/sessions",
        "/sessions/s1/messages",
    ], recorder.requests
    assert items and items[0]["content"] == "hello", items
    assert_all_requests_carry_ua(recorder.requests)


@pytest.mark.parametrize(
    "importer_factory, method",
    [
        (lambda: cognee.CogneeImporter(), "_extract_via_rest"),
        (lambda: letta.LettaImporter(api_key="k"), "_extract_via_rest"),
        (lambda: mem0.Mem0Importer(api_key="k"), "_extract_via_rest"),
        (lambda: supermemory.SuperMemoryImporter(api_key="k"), "_extract_via_rest"),
        (lambda: zep.ZepImporter(api_key="k"), "_extract_via_rest"),
    ],
)
def test_importer_rest_extraction_sends_application_user_agent(
    importer_factory, method
):
    recorder, patcher = drive_urlopen()
    try:
        try:
            getattr(importer_factory(), method)()
        except Exception:
            # A response shape this fixture does not model is irrelevant: the
            # request under test was already constructed and recorded, and it
            # is asserted below regardless of how the parse then went.
            pass
    finally:
        patcher.stop()
    assert_all_requests_carry_ua(recorder.requests)


def test_hindsight_api_extraction_sends_application_user_agent():
    recorder, patcher = drive_urlopen()
    try:
        try:
            hindsight.HindsightImporter(
                base_url="http://hindsight.test"
            )._extract_from_api()
        except Exception:
            pass  # the request was already constructed and recorded
    finally:
        patcher.stop()
    assert_all_requests_carry_ua(recorder.requests)


# ---------------------------------------------------------------------------
# Forward-looking guard + helper contract
# ---------------------------------------------------------------------------


def test_new_outbound_http_site_must_declare_a_user_agent():
    """Supplement to the request-level tests above.

    Those tests pin today's request sites; this guard fails when a *new*
    module performing outbound HTTP is added without an explicit
    User-Agent, so the regression cannot reappear in code nobody has
    written yet.
    """
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not any(marker in source for marker in OUTBOUND_MARKERS):
            continue
        if "application_user_agent" in source or "user-agent" in source.lower():
            continue
        offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        "outbound HTTP without an explicit User-Agent: " + ", ".join(offenders)
    )


def test_helper_resolves_the_installed_version():
    assert application_user_agent() == expected_user_agent()
    assert application_user_agent().startswith("Mnemosyne/")
