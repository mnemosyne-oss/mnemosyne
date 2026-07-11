"""
Tests for Mnemosyne Sync (mnemosyne/core/sync.py + sync_server.py)
==================================================================
Covers: event log, encryption roundtrip, conflict resolution,
pull/push protocol, and end-to-end client-server sync.
"""

import base64
import hashlib
import hmac
import json
import os
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

import pytest

from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.sync import (
    SyncEngine,
    SyncEvent,
    SyncEncryption,
    ConflictResolution,
)


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    instance = Mnemosyne(db_path=path)
    yield instance
    try:
        os.unlink(path)
    except OSError:
        pass


# --------------------------------------------------------------------------
# SyncEvent
# --------------------------------------------------------------------------

def test_sync_event_roundtrip():
    ev = SyncEvent(
        event_id=str(uuid.uuid4()),
        memory_id="mem_1",
        operation="CREATE",
        timestamp="2026-06-14T10:00:00Z",
        device_id="dev-a",
        payload='{"content": "hello"}',
        importance=0.8,
    )
    d = ev.to_dict()
    restored = SyncEvent.from_dict(d)
    assert restored.event_id == ev.event_id
    assert restored.memory_id == "mem_1"
    assert restored.operation == "CREATE"


# --------------------------------------------------------------------------
# Encryption
# --------------------------------------------------------------------------

def test_generate_key_is_base64_32_bytes():
    import base64
    key = SyncEncryption.generate_key()
    raw = base64.urlsafe_b64decode(key)
    assert len(raw) == 32


def test_encrypt_decrypt_roundtrip():
    key = SyncEncryption.generate_key()
    enc = SyncEncryption.from_config(key)
    payload = {"content": "User prefers dark mode", "importance": 0.9}
    ciphertext = enc.encrypt(payload)
    # Ciphertext must not contain the plaintext
    assert "dark mode" not in ciphertext
    assert enc.decrypt(ciphertext) == payload


def test_decrypt_with_wrong_key_fails():
    enc1 = SyncEncryption.from_config(SyncEncryption.generate_key())
    enc2 = SyncEncryption.from_config(SyncEncryption.generate_key())
    ct = enc1.encrypt({"secret": "value"})
    with pytest.raises(Exception):
        enc2.decrypt(ct)


# --------------------------------------------------------------------------
# Conflict resolution
# --------------------------------------------------------------------------

def _ev(mid, ts, dev, imp=0.5, op="UPDATE"):
    return SyncEvent(
        event_id=str(uuid.uuid4()),
        memory_id=mid,
        operation=op,
        timestamp=ts,
        device_id=dev,
        payload="{}",
        importance=imp,
    )


def test_resolve_latest_timestamp_wins():
    e1 = _ev("m", "2026-06-14T10:00:00Z", "a", imp=0.9)
    e2 = _ev("m", "2026-06-14T10:01:00Z", "b", imp=0.1)
    assert ConflictResolution.resolve([e1, e2]).device_id == "b"


def test_resolve_importance_tiebreak():
    e1 = _ev("m", "2026-06-14T10:00:00Z", "a", imp=0.3)
    e2 = _ev("m", "2026-06-14T10:00:00Z", "b", imp=0.9)
    assert ConflictResolution.resolve([e1, e2]).device_id == "b"


def test_resolve_single_event():
    e1 = _ev("m", "2026-06-14T10:00:00Z", "a")
    assert ConflictResolution.resolve([e1]) is e1


def test_resolve_empty_raises():
    with pytest.raises(ValueError):
        ConflictResolution.resolve([])


def test_detect_conflicts_in_window():
    local = [_ev("m", "2026-06-14T10:00:00Z", "a")]
    remote = [_ev("m", "2026-06-14T10:00:02Z", "b")]
    groups = ConflictResolution.detect_conflicts(local, remote, window_seconds=10)
    assert len(groups) == 1


def test_detect_no_conflict_different_memory():
    local = [_ev("m1", "2026-06-14T10:00:00Z", "a")]
    remote = [_ev("m2", "2026-06-14T10:00:00Z", "b")]
    groups = ConflictResolution.detect_conflicts(local, remote)
    assert len(groups) == 0


# --------------------------------------------------------------------------
# SyncEngine event log + protocol
# --------------------------------------------------------------------------

def test_log_and_pull_events(mem):
    engine = SyncEngine(mem, device_id="test-device")
    engine.log_event(
        memory_id="m1",
        operation="CREATE",
        payload={"content": "hello"},
        importance=0.8,
    )
    pull = engine.pull_changes(since_cursor=None)
    assert pull["total"] == 1
    assert pull["events"][0]["memory_id"] == "m1"


def test_invalid_operation_raises(mem):
    engine = SyncEngine(mem, device_id="test-device")
    with pytest.raises(ValueError):
        engine.log_event(memory_id="m1", operation="FROBNICATE")


def test_push_routes_through_remember_pipeline(mem):
    engine = SyncEngine(mem, device_id="test-device")
    events = [{
        "event_id": str(uuid.uuid4()),
        "memory_id": "remote_m1",
        "operation": "CREATE",
        "timestamp": "2026-06-14T12:00:00Z",
        "device_id": "remote-device",
        "payload": json.dumps({"content": "incoming sync memory"}),
        "parent_event_ids": "[]",
        "importance": 0.7,
    }]
    result = engine.push_changes(events)
    assert result["accepted"] == 1
    # Memory should be recallable via the full pipeline
    hits = mem.recall("incoming sync memory", top_k=3)
    assert any("incoming sync memory" in h.get("content", "") for h in hits)


def test_push_deduplicates_by_hash(mem):
    engine = SyncEngine(mem, device_id="test-device")
    ev = {
        "event_id": str(uuid.uuid4()),
        "memory_id": "dup_m1",
        "operation": "CREATE",
        "timestamp": "2026-06-14T12:00:00Z",
        "device_id": "remote",
        "payload": json.dumps({"content": "dedup test"}),
        "parent_event_ids": "[]",
        "importance": 0.5,
        "event_hash": "fixedhash123",
    }
    first = engine.push_changes([ev])
    second = engine.push_changes([ev])
    assert first["accepted"] == 1
    assert second["duplicates"] == 1


def test_get_status(mem):
    engine = SyncEngine(mem, device_id="test-device")
    engine.log_event("m1", "CREATE", payload={"content": "a"})
    engine.log_event("m2", "UPDATE", payload={"content": "b"})
    status = engine.get_status()
    assert status["total_events"] == 2
    assert status["device_id"] == "test-device"


# --------------------------------------------------------------------------
# End-to-end client-server sync
# --------------------------------------------------------------------------

def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _jwt(payload, secret="test-secret", header=None, signature=None):
    header = header or {"alg": "HS256", "typ": "JWT"}

    def b64(obj):
        raw = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    signing_input = f"{b64(header)}.{b64(payload)}"
    if signature is None:
        digest = hmac.new(
            secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256
        ).digest()
        signature = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"{signing_input}.{signature}"


def _request_sync_status_with_jwt(token, secret="test-secret"):
    from http.server import HTTPServer
    from mnemosyne.core.sync_server import SyncHTTPHandler

    class DummySync:
        def get_status(self):
            return {"ok": True}

    SyncHTTPHandler.sync_engine = DummySync()
    SyncHTTPHandler.api_key = None
    SyncHTTPHandler.jwt_secret = secret
    server = HTTPServer(("127.0.0.1", 0), SyncHTTPHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{server.server_address[1]}/sync/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, resp.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()


def test_sync_server_rejects_forged_hs256_jwt_signature():
    token = _jwt(
        {"sub": "attacker", "exp": time.time() + 3600},
        signature="forged",
    )

    with pytest.raises(urllib.error.HTTPError) as exc:
        _request_sync_status_with_jwt(token)

    assert exc.value.code == 401
    assert "invalid JWT signature" in exc.value.read().decode("utf-8")


def test_sync_server_rejects_unsupported_jwt_algorithm():
    token = _jwt(
        {"sub": "attacker", "exp": time.time() + 3600},
        header={"alg": "none", "typ": "JWT"},
        signature="forged",
    )

    with pytest.raises(urllib.error.HTTPError) as exc:
        _request_sync_status_with_jwt(token)

    assert exc.value.code == 401
    assert "unsupported JWT algorithm" in exc.value.read().decode("utf-8")


def test_sync_server_rejects_malformed_jwt_signature_cleanly():
    token = _jwt(
        {"sub": "attacker", "exp": time.time() + 3600},
        signature="förged",
    )

    with pytest.raises(urllib.error.HTTPError) as exc:
        _request_sync_status_with_jwt(token)

    assert exc.value.code == 401
    assert "invalid JWT signature" in exc.value.read().decode("utf-8")


def test_sync_server_accepts_valid_hs256_jwt():
    token = _jwt({"sub": "client", "exp": time.time() + 3600})

    status, body = _request_sync_status_with_jwt(token)

    assert status == 200
    assert json.loads(body) == {"ok": True}


def test_sync_server_rejects_expired_jwt():
    token = _jwt({"sub": "client", "exp": time.time() - 1})

    with pytest.raises(urllib.error.HTTPError) as exc:
        _request_sync_status_with_jwt(token)

    assert exc.value.code == 401


def test_e2e_plaintext_sync():
    from mnemosyne.core.sync_server import run_sync_server

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f1:
        db_local = f1.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f2:
        db_remote = f2.name

    local_mem = Mnemosyne(db_path=db_local)
    remote_mem = Mnemosyne(db_path=db_remote)
    port = _free_port()
    api_key = "test-key"

    t = threading.Thread(
        target=run_sync_server,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "beam_instance": remote_mem,
            "device_id": "remote-server",
            "api_key": api_key,
            "require_encrypted_payloads": False,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.5)

    local_engine = SyncEngine(local_mem, device_id="local-laptop")
    local_mem.remember("E2E plaintext memory", source="conversation", importance=0.8)

    result = local_engine.sync_with(
        remote_url=f"http://127.0.0.1:{port}",
        mode="bidirectional",
        api_key=api_key,
    )
    assert not result["errors"]
    assert result["push"]["accepted"] >= 1

    hits = remote_mem.recall("E2E plaintext memory", top_k=3)
    assert any("plaintext memory" in h.get("content", "") for h in hits)

    os.unlink(db_local)
    os.unlink(db_remote)


def test_e2e_encrypted_sync():
    from mnemosyne.core.sync_server import run_sync_server

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f1:
        db_local = f1.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f2:
        db_remote = f2.name

    local_mem = Mnemosyne(db_path=db_local)
    remote_mem = Mnemosyne(db_path=db_remote)
    port = _free_port()
    api_key = "test-key"

    t = threading.Thread(
        target=run_sync_server,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "beam_instance": remote_mem,
            "device_id": "remote-server",
            "api_key": api_key,
            "initialize_surface": True,
        },
        daemon=True,
    )
    t.start()
    time.sleep(0.5)

    # Local has the key; remote does NOT (stores opaque ciphertext)
    enc = SyncEncryption.from_config(SyncEncryption.generate_key())
    local_engine = SyncEngine(
        local_mem,
        device_id="local-laptop",
        encryption=enc,
        surface_only=True,
        initialize_surface=True,
    )
    local_mem.remember(
        "E2E encrypted secret", source="conversation", importance=0.9, scope="global"
    )

    result = local_engine.sync_with(
        remote_url=f"http://127.0.0.1:{port}",
        mode="bidirectional",
        api_key=api_key,
    )
    assert not result["errors"]
    assert result["push"]["accepted"] >= 1

    # Remote stored events but cannot read content (no key)
    remote_engine = SyncEngine(remote_mem, device_id="check")
    status = remote_engine.get_status()
    assert status["total_events"] >= 1
    # Remote working memory should NOT contain the plaintext secret
    remote_hits = remote_mem.recall("encrypted secret", top_k=3)
    assert not any("encrypted secret" in h.get("content", "") for h in remote_hits)

    os.unlink(db_local)
    os.unlink(db_remote)
