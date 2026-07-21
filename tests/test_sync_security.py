"""Security regressions for sync CLI and server configuration."""

from __future__ import annotations

import json
import os

import pytest

from mnemosyne.cli import _read_secret_file
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.sync_server import run_sync_server


def test_top_level_sync_help_lists_required_db_path(monkeypatch, capsys):
    from mnemosyne import cli

    monkeypatch.setattr(cli.sys, "argv", ["mnemosyne", "--help"])
    cli.run_cli()
    help_text = capsys.readouterr().out

    assert "sync --db-path <path> --remote <url>" in help_text
    assert "sync-init --db-path <path>" in help_text
    assert "sync-serve --db-path <path>" in help_text
    assert "sync-status --db-path <path>" in help_text


@pytest.mark.parametrize(
    ("handler_name", "args"),
    [
        ("cmd_sync", ["--remote", "https://relay.example"]),
        ("cmd_sync_serve", []),
        ("cmd_sync_status", []),
    ],
)
def test_sync_cli_commands_reject_missing_db_path(
    handler_name, args, capsys
):
    from mnemosyne import cli

    with pytest.raises(SystemExit) as exc_info:
        getattr(cli, handler_name)(args)

    assert exc_info.value.code == 2
    assert "--db-path" in capsys.readouterr().err


def test_deployment_readme_initializes_client_surface_before_sync():
    from pathlib import Path

    readme = (
        Path(__file__).resolve().parents[1] / "deploy" / "sync" / "README.md"
    ).read_text(encoding="utf-8")
    client_section = readme.split("From a client machine:", 1)[1].split(
        "## Fly.io", 1
    )[0]

    init_command = 'mnemosyne sync-init --db-path "$MNEMOSYNE_SYNC_DB"'
    sync_command = 'mnemosyne sync --db-path "$MNEMOSYNE_SYNC_DB"'
    assert init_command in client_section
    assert sync_command in client_section
    assert client_section.index(init_command) < client_section.index(sync_command)


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


def test_network_sync_refuses_unscoped_private_db_by_default(tmp_path):
    from mnemosyne.core.sync import SyncEngine

    memory = Mnemosyne(db_path=tmp_path / "private-sync.db", session_id="private")
    memory.remember("must stay private", source="test", scope="session")
    engine = SyncEngine(memory)

    with pytest.raises(ValueError, match="requires surface_only=True"):
        engine.sync_with("https://relay.example", mode="push")
    assert engine.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0


def test_non_loopback_plain_http_sync_is_rejected(tmp_path):
    from mnemosyne.core.sync import SyncEngine

    memory = Mnemosyne(db_path=tmp_path / "https-required.db")
    engine = SyncEngine(memory, allow_unscoped_sync=True)

    with pytest.raises(ValueError, match="require HTTPS"):
        engine.sync_with("http://relay.example", mode="push", api_key="secret")
    with pytest.raises(ValueError, match="require HTTPS"):
        engine.get_status(remote_url="http://relay.example", api_key="secret")


def test_sync_init_requires_explicit_confirmation_for_existing_rows(tmp_path, capsys):
    from mnemosyne.cli import cmd_sync_init

    db_path = tmp_path / "legacy-shared.db"
    memory = Mnemosyne(db_path=db_path, session_id="hermes_shared_surface")
    memory.remember("existing shared", source="test", scope="global")

    cmd_sync_init(["--db-path", str(db_path)])
    preview = json.loads(capsys.readouterr().out)
    assert preview["status"] == "confirmation_required"
    assert preview["existing_rows"] == 1
    sync_meta_exists = memory.beam.conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sync_meta'"
    ).fetchone()
    assert sync_meta_exists is None

    cmd_sync_init(
        ["--db-path", str(db_path), "--claim-existing", "--yes"]
    )
    applied = json.loads(capsys.readouterr().out)
    assert applied["status"] == "initialized"
    assert applied["claimed_rows"] == 1
    assert memory.beam.conn.execute(
        "SELECT sync_surface_id FROM working_memory"
    ).fetchone()[0] == "shared-surface-v1"


def test_sync_cli_claims_new_hermes_shared_surface_rows(tmp_path, monkeypatch, capsys):
    from mnemosyne.cli import cmd_sync, cmd_sync_init
    from mnemosyne.core.sync import SyncEngine

    db_path = tmp_path / "hermes-shared.db"
    cmd_sync_init(["--db-path", str(db_path)])
    capsys.readouterr()

    memory = Mnemosyne(db_path=db_path, session_id="hermes_shared_surface")
    memory_id = memory.beam.remember(
        "new Hermes shared memory",
        source="surface_manual",
        scope="global",
    )

    observed = {}

    def fake_sync_with(engine, remote_url, mode="bidirectional", api_key=None, **_kwargs):
        observed["surface_session_id"] = engine.surface_session_id
        observed["unowned_rows"] = engine.conn.execute(
            """SELECT COUNT(*) FROM working_memory
               WHERE sync_surface_id IS NULL OR sync_surface_id != ?""",
            (engine.surface_id,),
        ).fetchone()[0]
        return {
            "remote": remote_url,
            "mode": mode,
            "push": {"accepted": 0, "duplicates": 0, "conflicts": 0},
            "pull": None,
            "errors": [],
        }

    monkeypatch.setattr(SyncEngine, "sync_with", fake_sync_with)

    cmd_sync(
        [
            "--db-path",
            str(db_path),
            "--remote",
            "http://127.0.0.1:8765",
            "--mode",
            "push",
        ]
    )

    assert observed == {
        "surface_session_id": "hermes_shared_surface",
        "unowned_rows": 0,
    }
    assert memory.beam.conn.execute(
        "SELECT sync_surface_id FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()[0] == "shared-surface-v1"


def test_packaged_deployment_commands_match_hardened_cli():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    compose = (root / "deploy/sync/docker-compose.yml").read_text()
    fly = (root / "deploy/sync/fly.toml").read_text()

    for rendered in (compose, fly):
        assert "--db-path /data/relay.db" in rendered
        assert "--initialize-surface" in rendered
        assert "--behind-tls-proxy" in rendered
        assert "/healthz" in rendered


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


def test_non_loopback_server_requires_tls_or_explicit_proxy(tmp_path):
    memory = Mnemosyne(db_path=tmp_path / "server-no-tls.db")

    with pytest.raises(ValueError, match="HTTPS is required"):
        run_sync_server(
            host="0.0.0.0",
            port=0,
            beam_instance=memory,
            api_key="test-key",
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
    import hashlib
    import hmac
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

        invalid_limit_body = b'{"limit": 0}'
        invalid_limit_headers = dict(headers)
        invalid_limit_headers["X-Mnemosyne-Body-MAC"] = hmac.new(
            b"body-key", invalid_limit_body, hashlib.sha256
        ).hexdigest()
        invalid_limit = urllib.request.Request(
            f"{remote}/sync/pull",
            data=invalid_limit_body,
            headers=invalid_limit_headers,
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
    import hashlib
    import hmac
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
        "surface_id": "shared-surface-v1",
        "payload": json.dumps({"content": "must be rejected", "source": "test"}),
        "parent_event_ids": "[]",
        "importance": 0.5,
        "event_hash": "plaintext-hash",
    }
    body = json.dumps({"events": [event]}).encode()
    request = urllib.request.Request(
        f"{remote}/sync/push",
        data=body,
        headers={
            "Authorization": "Bearer relay-key",
            "Content-Type": "application/json",
            "X-Mnemosyne-Body-MAC": hmac.new(
                b"relay-key", body, hashlib.sha256
            ).hexdigest(),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read())
        assert result["accepted"] == 0
        assert result["errors"] == 1
        assert any(
            word in result["details"][0].lower()
            for word in ("plaintext", "encrypt")
        )
        assert relay.beam.conn.execute("SELECT COUNT(*) FROM memory_events").fetchone()[0] == 0
        assert relay.beam.conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_authenticated_server_rejects_post_without_body_mac(tmp_path):
    import urllib.error
    import urllib.request

    relay = Mnemosyne(db_path=tmp_path / "relay-body-mac.db")
    server = run_sync_server(
        host="127.0.0.1",
        port=0,
        beam_instance=relay,
        api_key="relay-key",
        daemon=True,
        initialize_surface=True,
    )
    remote = f"http://127.0.0.1:{server.server_address[1]}"
    body = json.dumps({"events": []}).encode()
    request = urllib.request.Request(
        f"{remote}/sync/push",
        data=body,
        headers={
            "Authorization": "Bearer relay-key",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        assert error.value.code == 401
        assert "body MAC" in json.loads(error.value.read())["error"]
    finally:
        server.shutdown()
        server.server_close()


def test_api_key_auth_rejects_non_ascii_bearer_without_server_error(tmp_path):
    import http.client

    relay = Mnemosyne(db_path=tmp_path / "relay-non-ascii-auth.db")
    server = run_sync_server(
        host="127.0.0.1",
        port=0,
        beam_instance=relay,
        api_key="relay-key",
        daemon=True,
        initialize_surface=True,
    )

    try:
        connection = http.client.HTTPConnection(
            "127.0.0.1", server.server_address[1], timeout=5
        )
        connection.request(
            "GET",
            "/sync/status",
            headers={"Authorization": "Bearer café"},
        )
        response = connection.getresponse()
        assert response.status == 401
        assert "Invalid or missing API key" in response.read().decode()
        connection.close()
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
    import urllib.request

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
        with urllib.request.urlopen(f"{remote}/healthz", timeout=5) as response:
            assert json.loads(response.read()) == {"status": "ok"}
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
