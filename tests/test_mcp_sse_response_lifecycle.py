"""Regression tests for the SSE transport response lifecycle (issue #910).

``handle_sse`` runs ``SseServerTransport.connect_sse()``, which owns the ASGI
response: it sends ``http.response.start`` itself and streams until the client
disconnects. Whatever the route handler returns after the stream ends must not
emit a *second* ``http.response.start`` — starlette/uvicorn reject it with
``RuntimeError: Expected ASGI message 'http.response.body', but got
'http.response.start'`` on every disconnect or server shutdown with open
sessions.
"""

import asyncio
import contextlib

import pytest


def _starlette_available() -> bool:
    try:
        import starlette  # noqa: F401
        import mcp  # noqa: F401
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _starlette_available(),
    reason="starlette/mcp not installed -- SSE lifecycle tests skipped",
)


def _sse_get_scope():
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/sse",
        "raw_path": b"/sse",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"localhost"), (b"accept", b"text/event-stream")],
        "client": ("127.0.0.1", 12345),
        "server": ("localhost", 80),
    }


class TestSseResponseLifecycle:
    def test_disconnect_sends_exactly_one_response_start(self, monkeypatch):
        """Controlled disconnect → exactly one http.response.start.

        Drives the real SseServerTransport with a hand-written scope: the
        request is delivered, the stream start is observed, then a controlled
        http.disconnect ends the stream. The route handler must return without
        sending another ASGI response (pre-fix it returned JSONResponse({}),
        producing a second http.response.start).
        """
        monkeypatch.delenv("MNEMOSYNE_MCP_TOKEN", raising=False)
        from mnemosyne.mcp_server import _build_sse_app

        app = _build_sse_app(host="127.0.0.1")
        sent = []

        async def drive():
            request_delivered = False
            stream_started = asyncio.Event()
            disconnect = asyncio.Event()

            async def receive():
                nonlocal request_delivered
                if not request_delivered:
                    # ASGI requires the server to deliver http.request before
                    # the app responds, even for a bodyless GET.
                    request_delivered = True
                    return {"type": "http.request", "body": b"", "more_body": False}
                await disconnect.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                sent.append(message)
                if message["type"] == "http.response.start":
                    stream_started.set()

            task = asyncio.create_task(app(_sse_get_scope(), receive, send))
            try:
                await asyncio.wait_for(stream_started.wait(), timeout=10)
                disconnect.set()
                await asyncio.wait_for(task, timeout=10)
            finally:
                # If anything above failed, the ASGI task may still hold the
                # open stream; cancel so the test reports instead of hanging.
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(BaseException):
                        await task

        asyncio.run(drive())

        starts = [m for m in sent if m["type"] == "http.response.start"]
        assert len(starts) == 1, (
            f"expected exactly one http.response.start, got {len(starts)}: {sent!r}"
        )
        assert starts[0]["status"] == 200
        headers = dict(
            (k.decode(), v.decode()) for k, v in starts[0].get("headers", [])
        )
        assert headers.get("content-type", "").startswith("text/event-stream")
