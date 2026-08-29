"""Tests for the remote consolidation seam on an empty answer.

A thinking model can spend the whole ``max_tokens`` budget on reasoning and
return HTTP 200 with ``finish_reason=length``, ``reasoning_content`` set and
an empty ``content``. These tests pin what ``local_llm`` does with such a
reply: the cause is named, the per-endpoint extra body rides each request,
and the fallback chain behaves as documented.

Every network test runs against a stub HTTP server on localhost. No live
endpoint is contacted.
"""

import json
import logging
import sqlite3
import tempfile
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from mnemosyne.core import local_llm
from mnemosyne.core.beam import BeamMemory


# ---------------------------------------------------------------------------
# Stub endpoint
# ---------------------------------------------------------------------------

class _Stub:
    """A minimal chat-completions endpoint that records what it was sent."""

    def __init__(self, reply):
        self.reply = reply
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append(json.loads(self.rfile.read(length).decode()))
                body = json.dumps(outer.reply).encode()
                self.send_response(200)
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


EMPTY_ANSWER = {
    "choices": [{
        "index": 0,
        "finish_reason": "length",
        "message": {
            "role": "assistant",
            "content": "",
            "reasoning_content": "thinking about memory one",
        },
    }],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "completion_tokens_details": {"reasoning_tokens": 7},
    },
}

TEXT_ANSWER = {
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {"role": "assistant", "content": "memory one, summarized"},
    }],
    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
}


@pytest.fixture
def primary():
    stub = _Stub(EMPTY_ANSWER)
    yield stub
    stub.close()


@pytest.fixture
def fallback():
    stub = _Stub(TEXT_ANSWER)
    yield stub
    stub.close()


@pytest.fixture
def remote_chain(monkeypatch, primary, fallback):
    """Primary endpoint answers empty, fallback endpoint answers text."""
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", primary.base_url)
    monkeypatch.setattr(local_llm, "LLM_API_KEY", "")
    monkeypatch.setattr(local_llm, "LLM_REMOTE_MODEL", "thinking-model")
    monkeypatch.setattr(local_llm, "LLM_EXTRA_BODY", {"thinking": {"type": "enabled"}})
    monkeypatch.setattr(local_llm, "LLM_FALLBACK_MODELS", ["plain-model"])
    monkeypatch.setattr(local_llm, "LLM_FALLBACK_BASE_URL", fallback.base_url)
    monkeypatch.setattr(local_llm, "LLM_FALLBACK_API_KEY", "")
    monkeypatch.setattr(local_llm, "LLM_FALLBACK_EXTRA_BODY", {"thinking": {"type": "disabled"}})
    monkeypatch.setattr(local_llm, "_last_llm_failure", None)


EMPTY_REASON = (
    "finish_reason=length, reasoning_tokens=7, reasoning_content present, content empty"
)


# ---------------------------------------------------------------------------
# Extra request body
# ---------------------------------------------------------------------------

class TestParseExtraBody:
    def test_unset_and_blank_are_empty(self, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_LLM_EXTRA_BODY", raising=False)
        assert local_llm._parse_extra_body("MNEMOSYNE_LLM_EXTRA_BODY") == {}
        monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", "   ")
        assert local_llm._parse_extra_body("MNEMOSYNE_LLM_EXTRA_BODY") == {}

    def test_invalid_json_is_empty_and_noted(self, monkeypatch, capsys):
        monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", "{not json")
        assert local_llm._parse_extra_body("MNEMOSYNE_LLM_EXTRA_BODY") == {}
        err = capsys.readouterr().err
        assert "MNEMOSYNE_LLM_EXTRA_BODY" in err
        assert "not valid JSON" in err

    @pytest.mark.parametrize("raw", ['[1, 2]', '"text"', "42", "null"])
    def test_non_object_is_empty_and_noted(self, monkeypatch, capsys, raw):
        monkeypatch.setenv("MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY", raw)
        assert local_llm._parse_extra_body("MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY") == {}
        err = capsys.readouterr().err
        assert "MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY" in err
        assert "JSON object" in err

    def test_object_is_returned(self, monkeypatch, capsys):
        monkeypatch.setenv("MNEMOSYNE_LLM_EXTRA_BODY", '{"thinking": {"type": "disabled"}, "n": 1}')
        assert local_llm._parse_extra_body("MNEMOSYNE_LLM_EXTRA_BODY") == {
            "thinking": {"type": "disabled"}, "n": 1,
        }
        assert capsys.readouterr().err == ""

    @pytest.mark.parametrize("key", ["messages", "model", "stream"])
    def test_a_reserved_key_is_dropped_and_noted(self, monkeypatch, capsys, key):
        monkeypatch.setenv(
            "MNEMOSYNE_LLM_EXTRA_BODY",
            json.dumps({key: "hijacked", "thinking": {"type": "disabled"}}),
        )
        assert local_llm._parse_extra_body("MNEMOSYNE_LLM_EXTRA_BODY") == {
            "thinking": {"type": "disabled"},
        }
        err = capsys.readouterr().err
        assert "MNEMOSYNE_LLM_EXTRA_BODY" in err
        assert key in err

    def test_an_object_of_only_reserved_keys_is_empty(self, monkeypatch, capsys):
        monkeypatch.setenv(
            "MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY",
            '{"messages": "hijacked", "model": "evil"}',
        )
        assert local_llm._parse_extra_body("MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY") == {}
        err = capsys.readouterr().err
        assert "messages, model" in err


class TestExtraBodyInRequest:
    def test_extra_body_merges_last_and_keeps_the_rest(self, monkeypatch, fallback):
        monkeypatch.setattr(local_llm, "LLM_MAX_TOKENS", 64)
        text, status, exc = local_llm._call_remote_llm_with_model(
            "memory one", "plain-model", base_url=fallback.base_url, api_key="",
            extra_body={"thinking": {"type": "disabled"}, "max_tokens": 16},
        )
        assert (text, status, exc) == ("memory one, summarized", 200, None)
        sent = fallback.requests[0]
        assert sent["thinking"] == {"type": "disabled"}
        assert sent["max_tokens"] == 16
        assert sent["model"] == "plain-model"
        assert sent["stream"] is False
        assert sent["messages"] == [{"role": "user", "content": "memory one"}]

    def test_no_extra_body_leaves_the_payload_alone(self, fallback):
        local_llm._call_remote_llm_with_model(
            "memory one", "plain-model", base_url=fallback.base_url, api_key="",
        )
        assert "thinking" not in fallback.requests[0]


# ---------------------------------------------------------------------------
# Empty answer
# ---------------------------------------------------------------------------

class TestEmptyAnswer:
    def test_empty_answer_is_named(self, primary):
        text, status, exc = local_llm._call_remote_llm_with_model(
            "memory one", "thinking-model", base_url=primary.base_url, api_key="",
        )
        assert text is None
        assert status == 200
        assert isinstance(exc, local_llm.EmptyAnswer)
        assert exc.finish_reason == "length"
        assert exc.reasoning_tokens == 7
        assert exc.has_reasoning is True
        assert str(exc) == EMPTY_REASON

    def test_empty_answer_with_a_bare_body(self, monkeypatch, fallback):
        """A body with no choices, usage or reasoning still names what it can."""
        monkeypatch.setattr(fallback, "reply", {"choices": []})
        text, status, exc = local_llm._call_remote_llm_with_model(
            "memory one", "plain-model", base_url=fallback.base_url, api_key="",
        )
        assert text is None
        assert status == 200
        assert isinstance(exc, local_llm.EmptyAnswer)
        assert str(exc) == "finish_reason=n/a, content empty"

    def test_empty_answer_ends_the_chain_with_the_reason_kept(self, remote_chain, primary, fallback):
        """Status 200 is not retryable, so the fallback model is never asked;
        the reason for the empty answer is kept for the caller's WARNING."""
        assert local_llm._call_remote_llm("memory one") is None
        assert len(primary.requests) == 1
        assert primary.requests[0]["thinking"] == {"type": "enabled"}
        assert fallback.requests == []
        assert local_llm.last_llm_failure() == (
            f"thinking-model: HTTP 200 with no usable choices ({EMPTY_REASON})"
        )

    def test_transport_failure_is_named_and_the_chain_continues(self, remote_chain, primary, fallback, monkeypatch):
        dead = _Stub(EMPTY_ANSWER)
        dead_url = dead.base_url
        dead.close()
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", dead_url)

        assert local_llm._call_remote_llm("memory one") == "memory one, summarized"
        assert primary.requests == []
        assert fallback.requests[0]["thinking"] == {"type": "disabled"}
        reason = local_llm.last_llm_failure()
        assert reason.startswith("thinking-model: ")
        assert "(timeout=" in reason


# ---------------------------------------------------------------------------
# The reason reaches the sleep() WARNING
# ---------------------------------------------------------------------------

class TestLastFailureReset:
    def test_summarize_resets_the_reason(self, monkeypatch):
        monkeypatch.setattr(local_llm, "_last_llm_failure", "stale")
        monkeypatch.setattr(local_llm, "LLM_BASE_URL", "http://remote/v1")
        monkeypatch.setattr(local_llm, "_call_remote_llm", lambda prompt, temperature=0.3: "memory one, summarized")
        assert local_llm.summarize_memories(["memory one"]) == "memory one, summarized"
        assert local_llm.last_llm_failure() is None


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _seed_old_wm(db_path, session_id, source, n):
    conn = sqlite3.connect(str(db_path))
    ts = (datetime.now() - timedelta(hours=200)).isoformat()
    rows = [
        (f"ea-{source}-{i}", f"memory {i}", source, ts, session_id)
        for i in range(n)
    ]
    conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


class TestSleepWarning:
    def test_aaak_fallback_warning_carries_the_reason(self, temp_db, monkeypatch, caplog):
        beam = BeamMemory(session_id="empty-answer", db_path=temp_db)
        beam.agent_context = "cron"
        _seed_old_wm(temp_db, "empty-answer", "conversation", n=2)

        monkeypatch.setattr(local_llm, "llm_available", lambda: True)
        monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda lines, source=None: [lines])

        def fake_summarize(lines, source=None):
            local_llm._last_llm_failure = (
                f"thinking-model: HTTP 200 with no usable choices ({EMPTY_REASON})"
            )
            return None

        monkeypatch.setattr(local_llm, "_summarize_memories", fake_summarize)
        caplog.set_level(logging.WARNING, logger="mnemosyne.core.beam")

        beam.sleep()

        messages = [r.getMessage() for r in caplog.records if "falling back to AAAK" in r.getMessage()]
        assert len(messages) == 1
        assert "source='conversation'" in messages[0]
        assert f"last_error=thinking-model: HTTP 200 with no usable choices ({EMPTY_REASON})" in messages[0]
