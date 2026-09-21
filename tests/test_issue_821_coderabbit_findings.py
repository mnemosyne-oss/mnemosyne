"""Focused regressions for the CodeRabbit a8fbad10 findings on PR #955."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
HERMES_SRC = ROOT / "integrations" / "hermes" / "src"
if str(HERMES_SRC) not in sys.path:
    sys.path.insert(0, str(HERMES_SRC))

PROVIDERS = ("hermes_memory_provider", "mnemosyne_hermes")


@pytest.mark.parametrize("provider_name", PROVIDERS)
@pytest.mark.parametrize(
    ("field", "action"),
    (("note", "attest"), ("new_content", "update"), ("validator", "attest")),
)
def test_provider_validate_rejects_persisted_text_before_lookup(
    provider_name: str, field: str, action: str
):
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    class NoLookupBeam:
        @property
        def conn(self):
            raise AssertionError("validation rejection must precede target lookup")

    module = importlib.import_module(provider_name)
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    provider._beam = NoLookupBeam()
    provider._agent_identity = "allowed validator"
    args = {"memory_id": "missing", "action": action, field: "ISSUE821 blocked text"}
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")

    with write_policy_operation(strict):
        result = json.loads(provider._handle_validate(args))

    assert result == {
        "status": "filtered",
        "memory_id": "missing",
        "store": "private",
        "bank": "private",
    }


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_provider_validate_allows_note_and_update(provider_name: str, tmp_path: Path):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    module = importlib.import_module(provider_name)
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    provider._beam = BeamMemory(
        session_id=f"validate-{provider_name}", db_path=tmp_path / f"{provider_name}.db"
    )
    provider._agent_identity = "allowed validator"
    provider._audit_event = lambda *_args, **_kwargs: None
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    try:
        memory_id = provider._beam.remember("allowed original")
        with write_policy_operation(strict):
            attest = json.loads(provider._handle_validate({
                "memory_id": memory_id,
                "action": "attest",
                "note": "allowed evidence",
            }))
            update = json.loads(provider._handle_validate({
                "memory_id": memory_id,
                "action": "update",
                "new_content": "allowed replacement",
                "note": "allowed reason",
            }))

        assert attest["status"] == "validation_attest"
        assert update["status"] == "validation_update"
        assert provider._beam.get(memory_id)["content"] == "allowed replacement"
        rows = provider._beam.conn.execute(
            "SELECT action, new_content, note FROM memory_validations ORDER BY rowid"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("attest", None, "allowed evidence"),
            ("update", "allowed replacement", "allowed reason"),
        ]
    finally:
        provider._beam.conn.close()


@pytest.mark.parametrize(
    ("field", "action"),
    (("note", "attest"), ("new_content", "update"), ("validator", "attest")),
)
def test_mcp_validate_rejects_persisted_text_before_lookup(
    monkeypatch, field: str, action: str
):
    from mnemosyne import mcp_tools
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    lookups = 0

    def no_lookup(**_kwargs):
        nonlocal lookups
        lookups += 1
        raise AssertionError("validation rejection must precede target lookup")

    monkeypatch.setattr(mcp_tools, "_create_instance", no_lookup)
    args = {"memory_id": "missing", "action": action, field: "ISSUE821 blocked text"}
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    with write_policy_operation(strict):
        result = mcp_tools._handle_validate(args)

    assert result == {
        "status": "filtered",
        "memory_id": "missing",
        "store": "private",
        "bank": "default",
    }
    assert lookups == 0


def test_mcp_validate_allows_note_and_update(monkeypatch, tmp_path: Path):
    from mnemosyne import mcp_tools
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.memory import Mnemosyne

    memory = Mnemosyne(session_id="mcp-validate", db_path=tmp_path / "mcp.db")
    monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: memory)
    strict = WritePolicySnapshot((r"^ISSUE821",), "strict")
    try:
        memory_id = memory.remember("allowed original")
        with write_policy_operation(strict):
            result = mcp_tools.handle_tool_call("mnemosyne_validate", {
                "memory_id": memory_id,
                "action": "update",
                "validator": "allowed validator",
                "new_content": "allowed replacement",
                "note": "allowed reason",
            })
        assert result["status"] == "validation_update"
        assert memory.get(memory_id)["content"] == "allowed replacement"
        row = memory.conn.execute(
            "SELECT validator, new_content, note FROM memory_validations"
        ).fetchone()
        assert tuple(row) == (
            "allowed validator", "allowed replacement", "allowed reason"
        )
    finally:
        memory.conn.close()


def test_mcp_validate_rejects_effective_fallback_validator(monkeypatch):
    from mnemosyne import mcp_tools
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    monkeypatch.setenv("MNEMOSYNE_AUTHOR_ID", "ISSUE821 blocked fallback validator")
    monkeypatch.setattr(
        mcp_tools,
        "_create_instance",
        lambda **_kwargs: pytest.fail("validator rejection must precede lookup"),
    )
    with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
        result = mcp_tools._handle_validate({
            "memory_id": "missing",
            "action": "attest",
        })

    assert result == {
        "status": "filtered",
        "memory_id": "missing",
        "store": "private",
        "bank": "default",
    }


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_provider_validate_rejects_effective_fallback_validator(
    provider_name: str,
):
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    module = importlib.import_module(provider_name)
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    provider._agent_identity = "ISSUE821 blocked fallback validator"
    provider._beam = types.SimpleNamespace(conn=types.SimpleNamespace(
        execute=lambda *_args, **_kwargs: pytest.fail(
            "validator rejection must precede lookup"
        )
    ))
    with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
        result = json.loads(provider._handle_validate({
            "memory_id": "missing",
            "action": "attest",
        }))

    assert result == {
        "status": "filtered",
        "memory_id": "missing",
        "store": "private",
        "bank": "private",
    }


@pytest.mark.parametrize("provider_name", PROVIDERS)
def test_batch_staging_handles_explicit_null_content(
    provider_name: str, tmp_path: Path, monkeypatch
):
    module = importlib.import_module(provider_name)
    provider = module.MnemosyneMemoryProvider.__new__(module.MnemosyneMemoryProvider)
    monkeypatch.setattr(module, "_write_approval_enabled", lambda: True)
    home = tmp_path / provider_name
    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: home
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)

    result = json.loads(provider._handle_batch({"operations": [{
        "action": "update",
        "memory_id": "memory-1",
        "content": None,
        "importance": 0.8,
    }]}))

    assert result["status"] == "staged"
    assert result["count"] == 1
    pending_id = result["pending_ids"][0]
    record = json.loads(
        (home / "pending" / "memory" / f"{pending_id}.json").read_text()
    )
    assert record["summary"] == ""
    assert record["payload"]["content"] is None


def test_remember_batch_preserves_rejected_positions_and_ids(tmp_path: Path):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    beam = BeamMemory(session_id="positioned-batch", db_path=tmp_path / "batch.db")
    items = [
        {"content": "allowed first"},
        {"content": "ISSUE821 rejected middle"},
        {"content": "allowed last"},
    ]
    try:
        with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
            memory_ids = beam.remember_batch(items)
        assert len(memory_ids) == len(items)
        assert memory_ids[0] is not None
        assert memory_ids[1] is None
        assert memory_ids[2] is not None
        assert memory_ids[0] != memory_ids[2]
        rows = beam.conn.execute(
            "SELECT id, content FROM working_memory ORDER BY rowid"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            (memory_ids[0], "allowed first"),
            (memory_ids[2], "allowed last"),
        ]
    finally:
        beam.conn.close()


@pytest.mark.parametrize("field", ("subject", "predicate", "object"))
def test_triple_store_admits_each_field_before_mutation(field: str, tmp_path: Path):
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.triples import TripleStore

    store = TripleStore(db_path=tmp_path / f"{field}.db")
    values = {"subject": "allowed subject", "predicate": "allowed predicate", "object": "allowed object"}
    values[field] = "ISSUE821 blocked triple field"
    try:
        with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
            assert store.add(**values) is None
            accepted = store.add(
                "allowed subject", "allowed predicate", "allowed object"
            )
        assert accepted is not None
        rows = store.conn.execute(
            "SELECT subject, predicate, object FROM triples"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("allowed subject", "allowed predicate", "allowed object")
        ]
    finally:
        store.conn.close()


@pytest.mark.parametrize("field", ("subject", "predicate", "object"))
def test_mcp_triple_add_rejects_each_field_before_target_lookup(
    field: str, monkeypatch
):
    from mnemosyne import mcp_tools
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation

    lookups = 0

    def no_lookup(**_kwargs):
        nonlocal lookups
        lookups += 1
        raise AssertionError("triple rejection must precede target lookup")

    monkeypatch.setattr(mcp_tools, "_create_instance", no_lookup)
    values = {"subject": "allowed subject", "predicate": "prefers", "object": "allowed object"}
    values[field] = "ISSUE821 blocked triple field"
    with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
        result = mcp_tools.handle_tool_call("mnemosyne_triple_add", values)
    assert result == {"status": "filtered", "store": "triples"}
    assert lookups == 0


def test_mcp_occurred_on_preserves_valid_from_as_annotation_value(
    monkeypatch, tmp_path: Path
):
    from mnemosyne import mcp_tools
    from mnemosyne.core.memory import Mnemosyne

    memory = Mnemosyne(session_id="occurred-on", db_path=tmp_path / "occurred.db")
    monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: memory)
    try:
        result = mcp_tools.handle_tool_call("mnemosyne_triple_add", {
            "subject": "memory-1",
            "predicate": "occurred_on",
            "object": "graduated college",
            "valid_from": "2010-06-15",
        })
        assert result["status"] == "added"
        assert result["annotation_id"] is not None
        rows = memory.beam.annotations.query_by_memory(
            "memory-1", kind="occurred_on"
        )
        assert [row["value"] for row in rows] == ["2010-06-15"]
    finally:
        memory.conn.close()


def test_remember_media_rejects_title_before_asset_upsert(tmp_path: Path):
    from mnemosyne.core.beam import BeamMemory
    from mnemosyne.core.filters import WritePolicySnapshot

    beam = BeamMemory(session_id="media-title", db_path=tmp_path / "media-title.db")
    try:
        result = beam.remember_media(
            "https://example.test/allowed.png",
            title="ISSUE821 blocked media title",
            _write_policy=WritePolicySnapshot((r"^ISSUE821",), "strict"),
        )
        assert result.status == "filtered"
        assert result.asset_id == ""
        assert beam.conn.execute(
            "SELECT COUNT(*) FROM media_assets"
        ).fetchone()[0] == 0
    finally:
        beam.conn.close()


def test_annotation_counts_only_inserted_admitted_rows(tmp_path: Path):
    from mnemosyne.core.annotations import AnnotationStore
    from mnemosyne.core.filters import WritePolicySnapshot, write_policy_operation
    from mnemosyne.core.triples import TripleStore

    db_path = tmp_path / "annotations.db"
    annotations = AnnotationStore(db_path=db_path)
    triples = TripleStore(db_path=db_path)
    duplicate = "allowed duplicate fact text"
    added = "allowed newly inserted fact"
    rejected = "ISSUE821 blocked annotation fact"
    try:
        assert annotations.add("memory-1", "fact", duplicate) is not None
        with write_policy_operation(WritePolicySnapshot((r"^ISSUE821",), "strict")):
            assert annotations.add_many(
                "memory-1", "fact", [duplicate, added, rejected]
            ) == 1
            assert triples.add_facts(
                "memory-2", [duplicate, duplicate, added, rejected]
            ) == 2
        rows = annotations.conn.execute(
            "SELECT memory_id, value FROM annotations ORDER BY memory_id, value"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("memory-1", duplicate),
            ("memory-1", added),
            ("memory-2", duplicate),
            ("memory-2", added),
        ]
    finally:
        annotations.conn.close()
        triples.conn.close()
