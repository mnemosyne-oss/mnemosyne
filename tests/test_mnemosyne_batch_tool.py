import json
from pathlib import Path

from hermes_memory_provider import MnemosyneMemoryProvider
from mnemosyne.core.beam import BeamMemory


def _beam(tmp_path):
    return BeamMemory(session_id="test_provider", db_path=Path(tmp_path) / "test.db")


def _provider(tmp_path):
    provider = MnemosyneMemoryProvider()
    provider._beam = _beam(tmp_path)
    provider._session_id = "test_provider"
    provider._agent_context = "primary"
    return provider


def _count_matching(beam, text):
    row = beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE content = ?",
        (text,),
    ).fetchone()
    return row[0]


def test_batch_schema_registered_and_dispatches(tmp_path):
    provider = _provider(tmp_path)
    names = {schema["name"] for schema in provider.get_tool_schemas()}
    assert "mnemosyne_batch" in names

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [{"action": "remember", "content": "batch dispatch"}],
    }))
    assert result["status"] == "ok"
    assert result["results"][0]["status"] == "stored"


def test_batch_multiple_remember_returns_ids(tmp_path):
    provider = _provider(tmp_path)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "batch alpha"},
            {"action": "remember", "content": "batch beta"},
        ],
    }))

    assert result["status"] == "ok"
    ids = [item["memory_id"] for item in result["results"]]
    assert len(ids) == 2
    assert all(ids)
    assert provider._beam.get(ids[0])["content"] == "batch alpha"
    assert provider._beam.get(ids[1])["content"] == "batch beta"


def test_batch_update_and_invalidate(tmp_path):
    provider = _provider(tmp_path)
    update_id = provider._beam.remember("batch old", importance=0.3)
    invalidate_id = provider._beam.remember("batch expires", importance=0.3)

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "update", "memory_id": update_id, "content": "batch new", "importance": "0.8"},
            {"action": "invalidate", "memory_id": invalidate_id},
        ],
    }))

    assert result["status"] == "ok"
    assert [item["status"] for item in result["results"]] == ["updated", "invalidated"]
    updated = provider._beam.get(update_id)
    assert updated["content"] == "batch new"
    assert updated["importance"] == 0.8
    invalidated = provider._beam.conn.execute(
        "SELECT valid_until FROM working_memory WHERE id = ?",
        (invalidate_id,),
    ).fetchone()
    assert invalidated[0]


def test_batch_extract_remember_uses_provider_default_scope(tmp_path):
    provider = _provider(tmp_path)
    provider._default_scope = "session"

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "scope parity extract", "extract": True},
        ],
    }))

    assert result["status"] == "ok"
    memory_id = result["results"][0]["memory_id"]
    row = provider._beam.conn.execute(
        "SELECT scope FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()
    assert row[0] == "session"


def test_batch_failure_rolls_back_earlier_remember(tmp_path):
    provider = _provider(tmp_path)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "rollback me"},
            {"action": "update", "memory_id": "missing", "content": "x"},
        ],
    }))

    assert result["status"] == "error"
    assert result["failed_index"] == 1
    assert result["action"] == "update"
    assert _count_matching(provider._beam, "rollback me") == 0


def test_batch_failure_rolls_back_earlier_update(tmp_path):
    provider = _provider(tmp_path)
    memory_id = provider._beam.remember("before update", importance=0.3)

    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "update", "memory_id": memory_id, "content": "after update"},
            {"action": "forget", "memory_id": "missing"},
        ],
    }))

    assert result["status"] == "error"
    assert result["failed_index"] == 1
    assert result["action"] == "forget"
    assert provider._beam.get(memory_id)["content"] == "before update"


def test_batch_audit_events_emit_only_after_successful_commit(tmp_path):
    provider = _provider(tmp_path)
    events = []
    provider._audit_event = lambda name, **kwargs: events.append((name, kwargs))

    failed = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "audit rollback"},
            {"action": "update", "memory_id": "missing", "content": "x"},
        ],
    }))
    assert failed["status"] == "error"
    assert events == []

    ok = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "audit commit"},
        ],
    }))
    assert ok["status"] == "ok"
    assert [event[0] for event in events] == ["remember"]


def test_batch_dry_run_writes_nothing(tmp_path):
    provider = _provider(tmp_path)
    existing_id = provider._beam.remember("dry existing", importance=0.3)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "dry_run": True,
        "operations": [
            {"action": "remember", "content": "dry new"},
            {"action": "update", "memory_id": existing_id, "content": "dry changed"},
        ],
    }))

    assert result["status"] == "dry_run"
    assert [item["status"] for item in result["results"]] == ["would_store", "would_update"]
    assert _count_matching(provider._beam, "dry new") == 0
    assert provider._beam.get(existing_id)["content"] == "dry existing"


def test_batch_unknown_action_rejected_before_mutation(tmp_path):
    provider = _provider(tmp_path)
    result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
        "operations": [
            {"action": "remember", "content": "should not write"},
            {"action": "search", "query": "x"},
        ],
    }))

    assert result["status"] == "error"
    assert result["failed_index"] == 1
    assert _count_matching(provider._beam, "should not write") == 0


def test_batch_requires_exact_ids_for_destructive_ops(tmp_path):
    provider = _provider(tmp_path)
    for action in ("update", "forget", "invalidate"):
        op = {"action": action}
        if action == "update":
            op["content"] = "x"
        result = json.loads(provider.handle_tool_call("mnemosyne_batch", {
            "operations": [op],
        }))
        assert result["status"] == "error"
        assert result["failed_index"] == 0
        assert result["action"] == action
