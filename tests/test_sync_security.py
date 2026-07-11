"""Security regressions for sync CLI and server configuration."""

from __future__ import annotations

import os

import pytest

from mnemosyne.cli import _read_secret_file
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.sync_server import run_sync_server


def test_surface_mode_refuses_unmarked_db_without_claiming_rows(tmp_path):
    from mnemosyne.core.sync import SyncEngine

    memory = Mnemosyne(db_path=tmp_path / "private-unmarked.db")
    memory.remember("private global", source="test", scope="global")
    before_columns = {
        row[1] for row in memory.beam.conn.execute("PRAGMA table_info(working_memory)")
    }

    with pytest.raises(ValueError, match="not initialized"):
        SyncEngine(memory, surface_only=True)

    after_columns = {
        row[1] for row in memory.beam.conn.execute("PRAGMA table_info(working_memory)")
    }
    assert after_columns == before_columns
    assert "sync_surface_id" not in after_columns


def test_non_loopback_plain_http_sync_is_rejected(tmp_path):
    from mnemosyne.core.sync import SyncEngine

    memory = Mnemosyne(db_path=tmp_path / "https-required.db")
    engine = SyncEngine(memory)

    with pytest.raises(ValueError, match="require HTTPS"):
        engine.sync_with("http://relay.example", mode="push", api_key="secret")
    with pytest.raises(ValueError, match="require HTTPS"):
        engine.get_status(remote_url="http://relay.example", api_key="secret")


def test_sync_engine_rejects_required_encryption_without_key(tmp_path):
    from mnemosyne.core.sync import SyncEngine

    memory = Mnemosyne(db_path=tmp_path / "missing-key.db")
    with pytest.raises(ValueError, match="no encryption key"):
        SyncEngine(memory, require_encryption=True, encryption=None)


def test_read_secret_file_accepts_private_nonempty_file(tmp_path):
    path = tmp_path / "secret.key"
    path.write_text("  secret-value\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)

    assert _read_secret_file(str(path), "test secret") == "secret-value"


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits only")
def test_read_secret_file_rejects_group_readable_file(tmp_path):
    path = tmp_path / "insecure.key"
    path.write_text("secret", encoding="utf-8")
    path.chmod(0o640)

    with pytest.raises(PermissionError, match="group/others"):
        _read_secret_file(str(path), "test secret")


@pytest.mark.skipif(os.name == "nt", reason="POSIX no-follow semantics")
def test_read_secret_file_rejects_symlink(tmp_path):
    target = tmp_path / "target.key"
    target.write_text("secret", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "link.key"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="securely open"):
        _read_secret_file(str(link), "test secret")


def test_read_secret_file_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.key"
    path.write_text("\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o600)

    with pytest.raises(ValueError, match="empty"):
        _read_secret_file(str(path), "test secret")


def test_non_loopback_server_requires_authentication(tmp_path):
    memory = Mnemosyne(db_path=tmp_path / "server.db")

    with pytest.raises(ValueError, match="authentication is required"):
        run_sync_server(
            host="0.0.0.0",
            port=0,
            beam_instance=memory,
            daemon=True,
        )


def test_server_requires_complete_tls_pair(tmp_path):
    memory = Mnemosyne(db_path=tmp_path / "server-tls.db")

    with pytest.raises(ValueError, match="configured together"):
        run_sync_server(
            host="127.0.0.1",
            port=0,
            beam_instance=memory,
            tls_cert="cert.pem",
            daemon=True,
        )


def test_server_rejects_oversized_and_invalid_json_once(tmp_path):
    import json
    import urllib.error
    import urllib.request

    memory = Mnemosyne(db_path=tmp_path / "server-body.db")
    server = run_sync_server(
        host="127.0.0.1",
        port=0,
        beam_instance=memory,
        api_key="body-key",
        max_body_bytes=32,
        daemon=True,
        initialize_surface=True,
    )
    remote = f"http://127.0.0.1:{server.server_address[1]}"
    headers = {
        "Authorization": "Bearer body-key",
        "Content-Type": "application/json",
    }

    try:
        oversized = urllib.request.Request(
            f"{remote}/sync/push", data=b"x" * 64, headers=headers, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as oversized_error:
            urllib.request.urlopen(oversized, timeout=5)
        assert oversized_error.value.code == 413
        assert json.loads(oversized_error.value.read())["error"] == "Request body too large"

        invalid = urllib.request.Request(
            f"{remote}/sync/pull", data=b"{", headers=headers, method="POST"
        )
        with pytest.raises(urllib.error.HTTPError) as invalid_error:
            urllib.request.urlopen(invalid, timeout=5)
        assert invalid_error.value.code == 400
        assert "Invalid JSON" in json.loads(invalid_error.value.read())["error"]

        invalid_limit = urllib.request.Request(
            f"{remote}/sync/pull",
            data=b'{"limit": 0}',
            headers=headers,
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as limit_error:
            urllib.request.urlopen(invalid_limit, timeout=5)
        assert limit_error.value.code == 400
        assert "between 1 and 10000" in json.loads(limit_error.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_relay_rejects_plaintext_events_by_default(tmp_path):
    import json
    import urllib.request

    relay = Mnemosyne(db_path=tmp_path / "relay-plaintext.db")
    server = run_sync_server(
        host="127.0.0.1",
        port=0,
        beam_instance=relay,
        api_key="relay-key",
        daemon=True,
        initialize_surface=True,
    )
    remote = f"http://127.0.0.1:{server.server_address[1]}"
    event = {
        "event_id": "plaintext-event",
        "memory_id": "plaintext-memory",
        "operation": "CREATE",
        "timestamp": "2026-07-11T10:00:00+00:00",
        "device_id": "client",
        "payload": json.dumps({"content": "must be rejected", "source": "test"}),
        "parent_event_ids": "[]",
        "importance": 0.5,
        "event_hash": "plaintext-hash",
    }
    request = urllib.request.Request(
        f"{remote}/sync/push",
        data=json.dumps({"events": [event]}).encode(),
        headers={
            "Authorization": "Bearer relay-key",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        assert result["accepted"] == 0
        assert result["errors"] == 1
        assert relay.beam.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0
        assert relay.beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_remote_status_response_is_bounded(tmp_path, monkeypatch):
    import urllib.request

    from mnemosyne.core.sync import SyncEngine

    class OversizedResponse:
        headers = {"Content-Length": "11"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            raise AssertionError("body must not be read after oversized Content-Length")

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *_args, **_kwargs: OversizedResponse()
    )
    memory = Mnemosyne(db_path=tmp_path / "bounded-response.db")
    engine = SyncEngine(memory, max_response_bytes=10)

    status = engine.get_status(remote_url="https://relay.invalid")

    assert "exceeds configured size limit" in status["remote_error"]


def test_remote_status_is_authenticated_and_read_only(tmp_path):
    from mnemosyne.core.sync import SyncEngine

    relay = Mnemosyne(db_path=tmp_path / "relay-status.db")
    client = Mnemosyne(db_path=tmp_path / "client-status.db")
    server = run_sync_server(
        host="127.0.0.1",
        port=0,
        beam_instance=relay,
        api_key="status-key",
        daemon=True,
        initialize_surface=True,
    )
    remote = f"http://127.0.0.1:{server.server_address[1]}"
    engine = SyncEngine(client, device_id="client")

    try:
        before = engine.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
        status = engine.get_status(remote_url=remote, api_key="status-key")
        after = engine.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0]
        assert status["remote_status"]["device_id"]
        assert before == after == 0

        denied = engine.get_status(remote_url=remote, api_key="wrong-key")
        assert "remote_error" in denied
        assert engine.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0
    finally:
        server.shutdown()
        server.server_close()
