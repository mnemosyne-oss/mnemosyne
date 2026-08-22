import json
from typing import Any, cast

import pytest

import mnemosyne
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.memory import Mnemosyne


def _rows(payload: Any) -> list[dict[str, Any]]:
    assert isinstance(payload, list)
    return payload


def _envelope(payload: Any) -> dict[str, Any]:
    assert isinstance(payload, dict)
    return payload


def _insert_episodic(beam, memory_id, content, metadata):
    beam.conn.execute(
        """
        INSERT INTO episodic_memory
            (id, content, source, timestamp, session_id, importance, metadata_json, scope)
        VALUES (?, ?, 'test', '2026-01-01T00:00:00+00:00', ?, 0.8, ?, 'global')
        """,
        (memory_id, content, beam.session_id, json.dumps(metadata)),
    )
    beam.conn.commit()


def test_projection_is_opt_in_allowlisted_and_non_mutating(tmp_path):
    mem = Mnemosyne(db_path=tmp_path / "metadata.db", session_id="test")
    memory_id = mem.remember(
        "unique metadata projection probe",
        metadata={"project_id": "p1", "private_note": "hidden", "count": 3},
        scope="global",
    )
    compact = _rows(mem.recall("unique metadata projection probe", top_k=5))
    assert "metadata" not in next(row for row in compact if row["id"] == memory_id)

    hydrated = _rows(mem.recall(
        "unique metadata projection probe",
        top_k=5,
        metadata_keys=["project_id", "count"],
    ))
    projected = next(row for row in hydrated if row["id"] == memory_id)
    assert projected["metadata"] == {"project_id": "p1", "count": 3}
    assert "private_note" not in projected["metadata"]
    assert "metadata" not in next(row for row in compact if row["id"] == memory_id)


def test_projection_bulk_hydrates_cross_tier_id_collision_in_two_queries(tmp_path):
    beam = BeamMemory(session_id="test", db_path=tmp_path / "bulk.db")
    shared_id = beam.remember(
        "working projection marker",
        metadata={"project_id": "working-project"},
        scope="global",
    )
    _insert_episodic(
        beam,
        shared_id,
        "episodic projection marker",
        {"project_id": "episodic-project"},
    )
    results = [
        {"id": shared_id, "tier": "working", "content": "working"},
        {"id": shared_id, "tier": "episodic", "content": "episodic"},
        {"id": "cf_synthetic", "tier": "fact", "content": "synthetic"},
    ]
    statements = []
    beam.conn.set_trace_callback(statements.append)
    try:
        hydrated = beam._hydrate_recall_metadata(results, ["project_id"])
    finally:
        beam.conn.set_trace_callback(None)

    queries = [s for s in statements if s.startswith("SELECT id, metadata_json FROM")]
    assert len(queries) == 2
    assert hydrated[0]["metadata"] == {"project_id": "working-project"}
    assert hydrated[1]["metadata"] == {"project_id": "episodic-project"}
    assert "metadata" not in hydrated[2]


def test_module_level_recall_supports_projection(tmp_path, monkeypatch):
    mem = Mnemosyne(db_path=tmp_path / "module.db", session_id="test")
    memory_id = mem.remember(
        "module projection marker",
        metadata={"project_id": "module-project"},
        scope="global",
    )
    monkeypatch.setattr("mnemosyne.core.memory._get_default", lambda _bank=None: mem)
    results = _rows(mnemosyne.recall(
        "module projection marker",
        metadata_keys=["project_id"],
    ))
    result = next(row for row in results if row["id"] == memory_id)
    assert result["metadata"] == {"project_id": "module-project"}


@pytest.mark.parametrize(
    "metadata_keys",
    ["project_id", [""], [1], ["x" * 129], ["x"] * 33],
)
def test_projection_rejects_invalid_key_requests(tmp_path, metadata_keys):
    mem = Mnemosyne(db_path=tmp_path / "invalid.db", session_id="test")
    with pytest.raises(ValueError):
        mem.recall("missing", metadata_keys=metadata_keys)


def test_projection_handles_malformed_metadata_and_duplicate_keys(tmp_path):
    mem = Mnemosyne(db_path=tmp_path / "malformed.db", session_id="test")
    memory_id = mem.remember("malformed metadata marker", scope="global")
    mem.beam.conn.execute(
        "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
        ("not-json", memory_id),
    )
    mem.beam.conn.commit()
    results = _rows(mem.recall(
        "malformed metadata marker",
        metadata_keys=["project_id", "project_id"],
    ))
    result = next(row for row in results if row["id"] == memory_id)
    assert result["metadata"] == {}


def test_projection_rejects_oversized_values_and_malformed_pages(tmp_path):
    beam = BeamMemory(session_id="test", db_path=tmp_path / "limits.db")
    memory_id = beam.remember(
        "oversized metadata marker",
        metadata={"large": "x" * (16 * 1024)},
        scope="global",
    )
    with pytest.raises(ValueError, match="exceeds 16384 bytes"):
        beam._hydrate_recall_metadata(
            [{"id": memory_id, "tier": "working"}],
            ["large"],
        )
    with pytest.raises(ValueError, match="dictionary"):
        beam._hydrate_recall_metadata(cast(Any, ["not-a-result"]), ["large"])
    with pytest.raises(ValueError, match="at most 500"):
        beam._hydrate_recall_metadata([{}] * 501, ["large"])


def test_empty_projection_returns_copies_without_queries(tmp_path):
    beam = BeamMemory(session_id="test", db_path=tmp_path / "empty.db")
    results = [{"id": "missing", "tier": "working"}]
    statements = []
    beam.conn.set_trace_callback(statements.append)
    try:
        hydrated = beam._hydrate_recall_metadata(results, ())
    finally:
        beam.conn.set_trace_callback(None)
    assert hydrated == results
    assert hydrated is not results
    assert hydrated[0] is not results[0]
    assert not any("metadata_json" in statement for statement in statements)


def test_projection_cannot_bypass_recall_session_scope(tmp_path):
    db_path = tmp_path / "scope.db"
    writer = Mnemosyne(db_path=db_path, session_id="private-session")
    private_id = writer.remember(
        "session-private metadata probe",
        metadata={"project_id": "private-project"},
        scope="session",
    )
    reader = Mnemosyne(db_path=db_path, session_id="other-session")
    results = _rows(reader.recall(
        "session-private metadata probe",
        top_k=10,
        metadata_keys=["project_id"],
    ))
    assert private_id not in {row["id"] for row in results}


def test_projection_preserves_explain_envelope(tmp_path):
    mem = Mnemosyne(db_path=tmp_path / "explain.db", session_id="test")
    memory_id = mem.remember(
        "explain projection marker",
        metadata={"project_id": "explain-project"},
        scope="global",
    )
    payload = _envelope(mem.recall(
        "explain projection marker",
        top_k=5,
        explain=True,
        metadata_keys=["project_id"],
    ))
    result = next(row for row in payload["results"] if row["id"] == memory_id)
    assert result["metadata"] == {"project_id": "explain-project"}
    assert payload["explain"]


def test_projection_does_not_expand_enhanced_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    mem = Mnemosyne(db_path=tmp_path / "cache.db", session_id="test")
    memory_id = mem.remember(
        "enhanced cache projection marker",
        metadata={"project_id": "cache-project"},
        scope="global",
    )
    hydrated = _rows(mem.recall(
        "enhanced cache projection marker",
        metadata_keys=["project_id"],
    ))
    result = next(row for row in hydrated if row["id"] == memory_id)
    assert result["metadata"] == {"project_id": "cache-project"}

    compact = _rows(mem.recall("enhanced cache projection marker"))
    assert "metadata" not in next(row for row in compact if row["id"] == memory_id)
