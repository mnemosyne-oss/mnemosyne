"""Behavioral conflict safety through real sleep, persistence and transports."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from email.message import Message
from unittest.mock import Mock
import urllib.request
from urllib.response import addinfourl

import httpx
import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core import llm_conflict_detector as lcd
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.local_llm import llm_available as real_llm_available


@pytest.fixture(autouse=True)
def isolated_conflict_config(monkeypatch):
    monkeypatch.setattr(lcd, "LLM_CONFLICT_DETECTION_ENABLED", True)
    monkeypatch.setattr(lcd, "CONFLICT_LLM_BASE_URL", "https://validator.test/v1")
    monkeypatch.setattr(lcd, "CONFLICT_LLM_API_KEY", "fake-local-test-key")
    monkeypatch.setattr(beam_module, "_CONFLICT_PAIR_BUDGET", 20)
    monkeypatch.setattr(beam_module, "_CONFLICT_TIME_BUDGET_S", 300.0)
    monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: False)
    monkeypatch.setattr("mnemosyne.core.model_refresh.infer_model_update_proposals", lambda items: [])


@pytest.fixture
def beam(tmp_path):
    mem = BeamMemory(session_id="contract", db_path=tmp_path / "memory.db")
    yield mem
    mem.conn.close()


def add_rows(mem, source="conversation", count=3, session=None):
    """Insert actual ordered rows without invoking an embedding backend."""
    session = session or mem.session_id
    ids = [f"{session}-{source}-{i}" for i in range(count)]
    for i, mid in enumerate(ids):
        mem.conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            (mid, f"The project meeting date changed to day {i}", source,
             f"2026-01-01T{10+i:02d}:00:00", session),
        )
    mem.conn.commit()
    return ids


def assert_counts(result, resolved, detected):
    assert result["conflicts_resolved"] == resolved
    assert result["conflicts_detected_only"] == detected


def assert_validation(mem, older, newer):
    row = mem.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id=?", (older,)
    ).fetchone()
    assert row["valid_until"] is not None
    assert row["superseded_by"] == newer
    records = mem.conn.execute(
        "SELECT memory_id, validator, action, new_content, note FROM memory_validations"
    ).fetchall()
    assert len(records) == 1
    assert tuple(records[0][:4]) == (older, "llm_conflict", "invalidated", "corrected")
    assert json.loads(records[0]["note"]) == {"confidence": 0.97, "replacement_id": newer}


@pytest.mark.parametrize("failed_first", [False, True])
def test_three_rows_one_successful_replacement(beam, monkeypatch, failed_first):
    older, first, second = add_rows(beam)
    monkeypatch.setattr(beam, "_detect_conflicts", lambda rows: [(older, first), (older, second)])
    validate = Mock(return_value=(True, 0.97, "corrected"))
    monkeypatch.setattr(lcd, "validate_conflict_pair", validate)
    real_invalidate = beam.invalidate
    invalidations = []

    def invalidate(mid, replacement_id, **kwargs):
        invalidations.append((mid, replacement_id))
        if failed_first and len(invalidations) == 1:
            return False
        return real_invalidate(mid, replacement_id=replacement_id, **kwargs)

    monkeypatch.setattr(beam, "invalidate", invalidate)
    assert_counts(beam.sleep(force=True), 1, 1)
    assert_validation(beam, older, second if failed_first else first)
    assert validate.call_count == (2 if failed_first else 1)
    assert len(invalidations) == validate.call_count


def test_success_tracking_survives_source_boundary(beam, monkeypatch):
    older, first, second = add_rows(beam)
    add_rows(beam, "notes", count=2)
    candidates = iter([[(older, first)], [(older, second)]])
    monkeypatch.setattr(beam, "_detect_conflicts", lambda rows: next(candidates))
    validate = Mock(return_value=(True, 0.97, "corrected"))
    monkeypatch.setattr(lcd, "validate_conflict_pair", validate)
    assert_counts(beam.sleep(force=True), 1, 1)
    assert_validation(beam, older, first)
    assert validate.call_count == 1


@pytest.mark.parametrize("verdict", [
    (False, 0.9, None), ("false", 0.9, "wrong"), None,
    ValueError("fake-local-test-key"),
])
def test_every_unsuccessful_call_is_detected_only(beam, monkeypatch, caplog, verdict):
    older, newer = add_rows(beam, count=2)
    monkeypatch.setattr(beam, "_detect_conflicts", lambda rows: [(older, newer)])
    validate = Mock(side_effect=verdict) if isinstance(verdict, Exception) else Mock(return_value=verdict)
    monkeypatch.setattr(lcd, "validate_conflict_pair", validate)
    assert_counts(beam.sleep(force=True), 0, 1)
    assert validate.call_count == 1
    assert beam.conn.execute("SELECT COUNT(*) FROM memory_validations").fetchone()[0] == 0
    assert beam.conn.execute("SELECT valid_until FROM working_memory WHERE id=?", (older,)).fetchone()[0] is None
    assert "fake-local-test-key" not in caplog.text


@pytest.mark.parametrize("budget_kind", ["pairs", "time"])
def test_budgets_span_all_source_groups(beam, monkeypatch, budget_kind):
    add_rows(beam, count=3)
    add_rows(beam, source="notes", count=3)
    monkeypatch.setattr(beam, "_detect_conflicts", lambda rows: [(rows[0]["id"], r["id"]) for r in rows[1:]])
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    monkeypatch.setattr(beam_module, "_CONFLICT_PAIR_BUDGET", 1 if budget_kind == "pairs" else 20)
    monkeypatch.setattr(beam_module, "_CONFLICT_TIME_BUDGET_S", 1.0)

    def validate(*args, **kwargs):
        assert kwargs["deadline"] == 101.0
        if budget_kind == "time":
            now[0] = 101.0
        return False, 0.0, None

    call = Mock(side_effect=validate)
    monkeypatch.setattr(lcd, "validate_conflict_pair", call)
    assert_counts(beam.sleep(force=True), 0, 4)
    assert call.call_count == 1


@pytest.mark.parametrize("api", ["sleep", "sleep_all_sessions"])
@pytest.mark.parametrize("recent", [False, True])
def test_noop_counter_shape(beam, api, recent):
    if recent:
        beam.remember("A recent project update", dedupe=False)
    result = getattr(beam, api)()
    assert result["status"] == "no_op"
    assert_counts(result, 0, 0)
    if api == "sleep_all_sessions":
        assert result["sessions_scanned"] == 0
        assert result["session_results"] == []


def test_actual_all_session_aggregate(beam, monkeypatch):
    add_rows(beam, count=2)
    add_rows(beam, count=2, session="other")
    monkeypatch.setattr(BeamMemory, "_detect_conflicts", lambda self, rows: [(rows[0]["id"], rows[1]["id"])])
    call = Mock(side_effect=[(True, 0.97, "corrected"), (False, 0.8, None)])
    monkeypatch.setattr(lcd, "validate_conflict_pair", call)
    result = beam.sleep_all_sessions(force=True)
    assert result["errors"] == 0
    assert result["sessions_consolidated"] == 2
    assert len(result["session_results"]) == 2
    assert_counts(result, 1, 1)
    for key in ("conflicts_resolved", "conflicts_detected_only"):
        assert result[key] == sum(r[key] for r in result["session_results"])
    assert sorted((r["conflicts_resolved"], r["conflicts_detected_only"]) for r in result["session_results"]) == [(0, 1), (1, 0)]
    assert call.call_count == 2


@pytest.mark.parametrize("api", ["sleep", "sleep_all_sessions"])
def test_dry_run_no_calls_cost_or_durable_writes(beam, monkeypatch, api):
    add_rows(beam, count=2)
    monkeypatch.setattr(BeamMemory, "_detect_conflicts", lambda self, rows: [(rows[0]["id"], rows[1]["id"])])
    monkeypatch.setattr("mnemosyne.core.local_llm.llm_available", lambda: True)
    calls = []
    for target in ("mnemosyne.core.llm_conflict_detector.validate_conflict_pair",
                   "mnemosyne.core.local_llm._summarize_memories",
                   "mnemosyne.core.model_refresh.infer_model_update_proposals",
                   "mnemosyne.core.llm_conflict_detector.log_cost"):
        call = Mock(side_effect=AssertionError("dry run must not call this"))
        monkeypatch.setattr(target, call)
        calls.append(call)
    before = list(beam.conn.iterdump())
    result = getattr(beam, api)(force=True, dry_run=True)
    assert_counts(result, 0, 1)
    for call in calls:
        call.assert_not_called()
    assert list(beam.conn.iterdump()) == before


@pytest.mark.parametrize("endpoint", ["http://validator.test", "ftp://validator.test", "https://", "https://user:secret@validator.test", None])
def test_unsafe_endpoint_rejected_before_validation(beam, monkeypatch, endpoint):
    older, newer = add_rows(beam, count=2)
    monkeypatch.setattr(beam, "_detect_conflicts", lambda rows: [(older, newer)])
    monkeypatch.setattr(lcd, "CONFLICT_LLM_BASE_URL", endpoint)
    call = Mock(return_value=(True, 0.97, "corrected"))
    monkeypatch.setattr(lcd, "validate_conflict_pair", call)
    assert_counts(beam.sleep(force=True), 0, 1)
    call.assert_not_called()


@pytest.mark.parametrize("backend", ["httpx", "urllib"])
def test_credentialed_http_blocked_at_transport(monkeypatch, backend, caplog):
    monkeypatch.setattr(lcd, "CONFLICT_LLM_BASE_URL", "http://validator.test/?secret=fake-local-test-key")
    if backend == "urllib":
        monkeypatch.setitem(sys.modules, "httpx", None)
    client = Mock(side_effect=AssertionError("no client construction"))
    opener = Mock(side_effect=AssertionError("no opener construction"))
    monkeypatch.setattr(httpx, "Client", client)
    monkeypatch.setattr(urllib.request, "build_opener", opener)
    assert lcd._call_conflict_llm_with_retry("private memory") is None
    client.assert_not_called()
    opener.assert_not_called()
    assert "fake-local-test-key" not in caplog.text
    assert "private memory" not in caplog.text


@pytest.mark.parametrize("backend", ["httpx", "urllib"])
@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("target", ["https://other.test/target", "http://validator.test/target", "https://validator.test/target"])
def test_redirects_never_replay_credentials(monkeypatch, backend, code, target, caplog):
    requests = []
    monkeypatch.setattr(time, "sleep", lambda delay: None)
    if backend == "httpx":
        real_client = httpx.Client

        def wire(request):
            requests.append((str(request.url), request.headers.get("Authorization")))
            if str(request.url) == target:
                return httpx.Response(200, json={"choices": [{"message": {"content": "unexpected"}}]})
            return httpx.Response(code, headers={"Location": target})

        def client(**kwargs):
            assert kwargs["follow_redirects"] is False
            return real_client(transport=httpx.MockTransport(wire), **kwargs)

        monkeypatch.setattr(httpx, "Client", client)
    else:
        monkeypatch.setitem(sys.modules, "httpx", None)
        real_opener = urllib.request.build_opener

        class FakeWire(urllib.request.BaseHandler):
            handler_order = 100

            def https_open(self, request):
                requests.append((request.full_url, request.get_header("Authorization")))
                headers = Message()
                headers["Location"] = target
                response = addinfourl(io.BytesIO(b"{}"), headers, request.full_url,
                                      200 if request.full_url == target else code)
                response.msg = "test response"
                return response

            http_open = https_open

        monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: real_opener(FakeWire(), *handlers))
    assert lcd._call_conflict_llm_with_retry("private memory") is None
    assert requests == [("https://validator.test/v1/chat/completions", "Bearer fake-local-test-key")] * 3
    assert "fake-local-test-key" not in caplog.text
    assert "private memory" not in caplog.text


@pytest.mark.parametrize("value", [[], None, "false", {},
    {"is_conflict": "false", "confidence": 0.9},
    {"is_conflict": 1, "confidence": 0.9},
    {"is_conflict": True, "confidence": "0.9"},
    {"is_conflict": True, "confidence": True},
    {"is_conflict": True, "confidence": float("nan")},
    {"is_conflict": True, "confidence": float("inf")},
    {"is_conflict": True, "confidence": -0.1},
    {"is_conflict": True, "confidence": 1.1},
    {"is_conflict": True, "confidence": 0.9, "correct_fact": {}},
])
def test_malformed_json_types_abstain(monkeypatch, value):
    monkeypatch.setattr(lcd, "_call_conflict_llm_with_retry", lambda *args, **kwargs: (json.dumps(value), 10, 10))
    assert lcd.validate_conflict_pair("old", "new", "session") == (False, 0.0, None)


@pytest.mark.parametrize("raw,expected", [(None, (20, 300.0)), ("", (20, 300.0)),
    ("bad", (20, 300.0)), ("0", (20, 300.0)), ("-2", (20, 300.0)),
    ("nan", (20, 300.0)), ("inf", (20, 300.0)), ("-inf", (20, 300.0)),
    (" 7 ", (7, 7.0)), ("1.5", (20, 1.5)),
])
def test_budget_env_fresh_import(raw, expected):
    env = os.environ.copy()
    for key in ("MNEMOSYNE_CONFLICT_PAIR_BUDGET", "MNEMOSYNE_CONFLICT_TIME_BUDGET_S"):
        env.pop(key, None)
        if raw is not None:
            env[key] = raw
    code = "from mnemosyne.core import beam; print(beam._CONFLICT_PAIR_BUDGET, beam._CONFLICT_TIME_BUDGET_S)"
    run = subprocess.run([sys.executable, "-c", code], env=env, cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip().splitlines()[-1] == f"{expected[0]} {expected[1]}"


@pytest.mark.parametrize("backend", ["httpx", "urllib"])
def test_transport_deadline_bounds_attempts_and_backoff(monkeypatch, backend):
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    sleep = Mock()
    monkeypatch.setattr(time, "sleep", sleep)
    timeouts = []

    def fail(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        now[0] += 0.25
        raise OSError("fake-local-test-key")

    if backend == "httpx":
        monkeypatch.setattr(httpx, "Client", fail)
    else:
        monkeypatch.setitem(sys.modules, "httpx", None)
        monkeypatch.setattr(urllib.request, "build_opener", lambda *args: type("Opener", (), {"open": fail})())
    assert lcd._call_conflict_llm_with_retry("private", deadline=100.5) is None
    assert timeouts == [0.5]
    sleep.assert_not_called()
    for deadline in (99.0, float("nan"), float("inf")):
        assert lcd._call_conflict_llm_with_retry("private", deadline=deadline) is None
    assert timeouts == [0.5]


@pytest.mark.parametrize("api", ["sleep", "sleep_all_sessions"])
@pytest.mark.parametrize("backend", ["local", "remote", "host"])
@pytest.mark.parametrize("dry_run", [False, True])
def test_dry_run_avoids_availability_and_download_but_keeps_failure_warning(
    beam, monkeypatch, caplog, api, backend, dry_run,
):
    """Even availability checks can lazily download a model; logs must not call them."""
    from mnemosyne.core import local_llm
    calls = []

    def blocked_download():
        calls.append("download")
        raise RuntimeError("probe blocks actual model download")

    def available():
        calls.append("availability")
        return real_llm_available()

    add_rows(beam, count=2)
    monkeypatch.setattr(beam, "_detect_conflicts", lambda rows: [])
    monkeypatch.setattr(local_llm, "llm_available", available)
    monkeypatch.setattr(local_llm, "LLM_ENABLED", True)
    monkeypatch.setattr(local_llm, "LLM_BASE_URL", "https://model.test" if backend == "remote" else "")
    monkeypatch.setattr(local_llm, "_llm_available", None)
    monkeypatch.setattr(local_llm, "_llm_instance", None)
    monkeypatch.setattr(local_llm, "_host_backend_will_handle_call", lambda: backend == "host")
    monkeypatch.setattr(local_llm, "_model_path", lambda: None)
    monkeypatch.setattr(local_llm, "_download_model", blocked_download)
    monkeypatch.setattr(local_llm, "_summarize_memories", lambda *a, **kw: None)
    before = list(beam.conn.iterdump())
    result = getattr(beam, api)(force=True, dry_run=dry_run)
    if dry_run:
        assert calls == []
        assert "LLM summarization failed" not in caplog.text
        assert list(beam.conn.iterdump()) == before
    else:
        assert "availability" in calls
        assert "LLM summarization failed" in caplog.text
        assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 1
    assert result["conflicts_resolved"] == result["conflicts_detected_only"] == 0


@pytest.mark.parametrize("api", ["sleep", "sleep_all_sessions"])
@pytest.mark.parametrize("budget", [1, 100])
def test_budget_warning_totals_all_source_groups(beam, monkeypatch, caplog, api, budget):
    for source in ("conversation", "notes", "import"):
        add_rows(beam, source=source, count=3)
    pairs = []

    def detect(rows):
        result = [(rows[0]["id"], rows[1]["id"]), (rows[1]["id"], rows[2]["id"])]
        pairs.extend(result)
        return result

    monkeypatch.setattr(beam, "_detect_conflicts", detect)
    monkeypatch.setattr(beam_module, "_CONFLICT_PAIR_BUDGET", budget)
    validate = Mock(return_value=(False, 0.9, None))
    monkeypatch.setattr(lcd, "validate_conflict_pair", validate)
    result = getattr(beam, api)(force=True)
    assert len(pairs) == 6
    assert_counts(result, 0, len(pairs))
    assert validate.call_count == min(budget, len(pairs))
    messages = [r.getMessage() for r in caplog.records if "conflict validation budget reached" in r.getMessage()]
    skipped = len(pairs) - validate.call_count
    assert messages == ([f"conflict validation budget reached: {skipped} pair(s) skipped this sleep"] if skipped else [])


def test_conflict_failure_logs_type_not_sensitive_message(beam, monkeypatch, caplog):
    ids = add_rows(beam, count=2)
    monkeypatch.setattr(beam, "_detect_conflicts", lambda rows: [(ids[0], ids[1])])
    monkeypatch.setattr(lcd, "validate_conflict_pair", Mock(side_effect=TimeoutError("private-prompt-token")))
    assert_counts(beam.sleep(force=True), 0, 1)
    assert "Conflict validation/persistence failed (TimeoutError)" in caplog.text
    assert "private-prompt-token" not in caplog.text
