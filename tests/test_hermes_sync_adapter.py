"""Regression tests for the Hermes shared-surface sync adapter."""

from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTEGRATION_SRC = PROJECT_ROOT / "integrations" / "hermes" / "src"
SHARED_SESSION_ID = "hermes_shared_surface"
SURFACE_ID = "shared-surface-v1"


def _import_sync_module(package: str, import_root: Path):
    for name in list(sys.modules):
        if name == package or name.startswith(f"{package}."):
            del sys.modules[name]
    sys.path.insert(0, str(import_root))
    try:
        return importlib.import_module(f"{package}.sync_adapter")
    finally:
        sys.path.remove(str(import_root))


@pytest.fixture(params=(
    ("hermes_memory_provider", PROJECT_ROOT),
    ("mnemosyne_hermes", INTEGRATION_SRC),
))
def sync_module(request):
    package, import_root = request.param
    return _import_sync_module(package, import_root)


class _Response:
    headers: dict[str, str] = {}

    def __init__(self, payload: dict):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size: int = -1) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def _database_snapshot(conn: sqlite3.Connection) -> tuple[list[tuple], list[tuple]]:
    schema = conn.execute(
        "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    rows = conn.execute("SELECT * FROM working_memory ORDER BY id").fetchall()
    return [tuple(row) for row in schema], [tuple(row) for row in rows]


def test_push_tool_discovers_shared_global_memory_with_surface_id(
    tmp_path, monkeypatch, sync_module
):
    beam = BeamMemory(
        session_id=SHARED_SESSION_ID,
        db_path=tmp_path / "shared.db",
    )
    memory_id = beam.remember(
        "shared sync probe",
        source="surface_manual",
        importance=0.8,
        scope="global",
        metadata={"source_profile_session": "private-session", "safe": "kept"},
    )
    requests: list[dict] = []

    def fake_urlopen(request, **_kwargs):
        body = json.loads(request.data.decode("utf-8"))
        requests.append(body)
        event_ids = [event["event_id"] for event in body["events"]]
        return _Response(
            {
                "accepted": len(event_ids),
                "duplicates": 0,
                "conflicts": 0,
                "errors": 0,
                "acknowledged_event_ids": event_ids,
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = sync_module.SyncAdapter(
        beam,
        {"remote": "https://sync.example", "token": "relay-token"},
    )

    result = json.loads(adapter.handle_tool_call("mnemosyne_sync_push", {}))

    assert result["status"] == "ok"
    assert result["pushed"] == 1
    assert len(requests) == 1
    [event] = requests[0]["events"]
    assert event["memory_id"] == memory_id
    assert event["surface_id"] == SURFACE_ID
    payload = json.loads(event["payload"])
    assert payload["content"] == "shared sync probe"
    assert "session_id" not in payload
    assert json.loads(payload["metadata_json"]) == {"safe": "kept"}


@pytest.mark.parametrize(
    ("row_session", "scope"),
    (
        (SHARED_SESSION_ID, "session"),
        ("foreign-session", "global"),
    ),
)
def test_surface_startup_rejects_mixed_db_without_mutation(
    tmp_path, sync_module, row_session, scope
):
    db_path = tmp_path / f"mixed-{row_session}-{scope}.db"
    writer = BeamMemory(session_id=row_session, db_path=db_path)
    writer.remember("must remain private", source="test", scope=scope)
    shared_beam = BeamMemory(session_id=SHARED_SESSION_ID, db_path=db_path)
    before = _database_snapshot(shared_beam.conn)

    adapter = sync_module.SyncAdapter(
        shared_beam,
        {"remote": "https://sync.example"},
    )

    assert adapter.is_ready is False
    result = json.loads(adapter.handle_tool_call("mnemosyne_sync_push", {}))
    assert result["status"] == "error"
    assert "dedicated global shared surface" in result["error"]
    assert _database_snapshot(shared_beam.conn) == before


def test_surface_startup_rejects_non_shared_session_without_mutation(
    tmp_path, sync_module
):
    beam = BeamMemory(session_id="private-session", db_path=tmp_path / "private.db")
    before = _database_snapshot(beam.conn)

    adapter = sync_module.SyncAdapter(beam, {"remote": "https://sync.example"})

    assert adapter.is_ready is False
    assert "expected session" in adapter._error
    assert _database_snapshot(beam.conn) == before
