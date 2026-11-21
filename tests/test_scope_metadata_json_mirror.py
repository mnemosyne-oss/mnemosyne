"""
Regression tests: scope must mirror into metadata_json on every write path.

JSON-based readers (curator checks, lr_dump, json_extract(metadata_json,
'$.scope')) read scope from metadata_json. Before this fix:

- `BeamMemory.remember(..., scope="global")` wrote `scope` to the dedicated
  `working_memory.scope` COLUMN but left metadata_json as the caller's
  metadata only (`'{}'` when no metadata was passed) — the "two-table trap".
- The dedup-update path updated the scope column but never touched
  metadata_json.
- The legacy `memories` dual-write has no scope column at all, so scope was
  lost from that table entirely.

These tests pin the mirror behavior on all three paths.
"""

import json
import tempfile
from pathlib import Path

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.memory import Mnemosyne


def _temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _json_scope(row_metadata_json):
    """Return $.scope from a metadata_json string, or None if absent."""
    try:
        meta = json.loads(row_metadata_json) if row_metadata_json else {}
    except Exception:
        return None
    return meta.get("scope")


def test_remember_mirrors_scope_into_metadata_json():
    db = next(_temp_db())
    beam = BeamMemory(session_id="scope-mirror", db_path=db)
    mem_id = beam.remember("scope mirror insert", source="test", scope="global")

    row = beam.conn.execute(
        "SELECT scope, metadata_json FROM working_memory WHERE id = ?",
        (mem_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == "global"  # dedicated column
    assert _json_scope(row[1]) == "global"  # metadata_json mirror


def test_remember_preserves_caller_metadata_when_mirroring_scope():
    db = next(_temp_db())
    beam = BeamMemory(session_id="scope-mirror", db_path=db)
    mem_id = beam.remember(
        "scope mirror metadata", source="test", scope="global",
        metadata={"foo": "bar"},
    )

    row = beam.conn.execute(
        "SELECT metadata_json FROM working_memory WHERE id = ?",
        (mem_id,),
    ).fetchone()
    meta = json.loads(row[0])
    assert meta["scope"] == "global"
    assert meta["foo"] == "bar"  # caller metadata preserved


def test_dedup_update_mirrors_scope_into_metadata_json():
    db = next(_temp_db())
    beam = BeamMemory(session_id="scope-mirror", db_path=db)
    # First write with no metadata -> metadata_json '{}' before the fix.
    mem_id = beam.remember("dedup scope mirror", source="test", scope="global")

    row = beam.conn.execute(
        "SELECT scope, metadata_json FROM working_memory WHERE id = ?",
        (mem_id,),
    ).fetchone()
    assert _json_scope(row[1]) == "global"  # INSERT path mirrors

    # Re-remember identical content -> dedup-update path. Change the scope
    # so the update actually rewrites the row.
    beam.remember("dedup scope mirror", source="test", scope="session")

    row = beam.conn.execute(
        "SELECT scope, metadata_json FROM working_memory WHERE id = ?",
        (mem_id,),
    ).fetchone()
    assert row[0] == "session"  # column updated
    assert _json_scope(row[1]) == "session"  # metadata_json updated too


def test_legacy_memories_dual_write_carries_scope():
    db = next(_temp_db())
    mem = Mnemosyne(db_path=db)
    mem_id = mem.remember("legacy scope mirror", source="test", scope="global")

    row = mem.conn.execute(
        "SELECT metadata_json FROM memories WHERE id = ?",
        (mem_id,),
    ).fetchone()
    assert row is not None
    assert _json_scope(row[0]) == "global"  # scope rides in metadata_json
