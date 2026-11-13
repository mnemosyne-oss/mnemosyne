"""All-or-nothing JSON import boundary (#727).

A failure in any section must leave the target unchanged. The import runs
under one deferred-commit scope, and every store import joins it (savepoint
when a transaction is already open).
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from mnemosyne.core.annotations import AnnotationStore
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.canonical import CanonicalStore
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.triples import TripleStore

_TABLES = (
    "working_memory",
    "memories",
    "triples",
    "annotations",
    "canonical_facts",
    "memory_events",
)


def _source(tmp_path):
    source = Mnemosyne(session_id="s", db_path=str(tmp_path / "source.db"))
    memory_id = source.remember("atomically restorable memory", scope="global")
    TripleStore(db_path=source.db_path, conn=source.conn).add(
        "project", "uses", "sqlite"
    )
    AnnotationStore(db_path=source.db_path, conn=source.conn).add(
        memory_id, "mentions", "sqlite"
    )
    CanonicalStore(db_path=source.db_path, conn=source.conn).remember(
        "s", "identity", "name", "Alice"
    )
    counts = _counts(source)
    export_path = tmp_path / "export.json"
    source.export_to_file(str(export_path))
    return export_path, counts


def _target(tmp_path):
    target = Mnemosyne(session_id="s", db_path=str(tmp_path / "target.db"))
    # Materialize every store schema before the import so a rolled-back DDL
    # cannot make the count assertions ambiguous.
    TripleStore(db_path=target.db_path, conn=target.conn)
    AnnotationStore(db_path=target.db_path, conn=target.conn)
    CanonicalStore(db_path=target.db_path, conn=target.conn)
    return target


def _counts(memory: Mnemosyne) -> dict[str, int]:
    return {
        table: memory.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in _TABLES
    }


def _assert_unchanged(memory: Mnemosyne) -> None:
    assert _counts(memory) == {table: 0 for table in _TABLES}
    assert memory.conn.in_transaction is False


def test_successful_import_commits_every_section(tmp_path):
    export_path, source_counts = _source(tmp_path)
    target = _target(tmp_path)

    stats = target.import_from_file(str(export_path))

    assert stats["beam"]
    assert _counts(target) == source_counts
    assert target.conn.in_transaction is False


def test_failure_in_the_beam_section_rolls_back_everything(tmp_path, monkeypatch):
    export_path, _ = _source(tmp_path)
    target = _target(tmp_path)
    original = BeamMemory.import_from_dict

    def fail(self, data, force=False):
        original(self, data, force=force)
        raise RuntimeError("injected beam failure")

    monkeypatch.setattr(BeamMemory, "import_from_dict", fail)

    with pytest.raises(RuntimeError, match="injected beam failure"):
        target.import_from_file(str(export_path))

    _assert_unchanged(target)


def test_failure_in_the_legacy_section_rolls_back_everything(tmp_path):
    export_path, _ = _source(tmp_path)
    data = json.loads(export_path.read_text())
    data["legacy_memories"] = [
        {
            "id": "legacy-one",
            "content": "legacy row",
            "source": "test",
            "timestamp": "2026-01-01T00:00:00",
            "session_id": "s",
            "importance": 0.5,
            "metadata_json": "{}",
        }
    ]
    data["legacy_embeddings"] = [{"memory_id": "legacy-one", "embedding_json": None}]
    export_path.write_text(json.dumps(data))
    target = _target(tmp_path)

    with pytest.raises(sqlite3.IntegrityError):
        target.import_from_file(str(export_path))

    _assert_unchanged(target)


def test_failure_in_the_triples_section_rolls_back_everything(tmp_path, monkeypatch):
    export_path, _ = _source(tmp_path)
    target = _target(tmp_path)

    def fail(self, *_args, **_kwargs):
        raise RuntimeError("injected triples failure")

    monkeypatch.setattr(TripleStore, "import_all", fail)

    with pytest.raises(RuntimeError, match="injected triples failure"):
        target.import_from_file(str(export_path))

    _assert_unchanged(target)


def test_failure_in_the_annotations_section_rolls_back_everything(tmp_path, monkeypatch):
    export_path, _ = _source(tmp_path)
    target = _target(tmp_path)

    def fail(self, *_args, **_kwargs):
        raise RuntimeError("injected annotations failure")

    monkeypatch.setattr(AnnotationStore, "import_all", fail)

    with pytest.raises(RuntimeError, match="injected annotations failure"):
        target.import_from_file(str(export_path))

    _assert_unchanged(target)


def test_failure_in_the_canonical_section_rolls_back_everything(tmp_path, monkeypatch):
    export_path, _ = _source(tmp_path)
    target = _target(tmp_path)

    def fail(self, *_args, **_kwargs):
        raise RuntimeError("injected canonical failure")

    monkeypatch.setattr(CanonicalStore, "import_all", fail)

    with pytest.raises(RuntimeError, match="injected canonical failure"):
        target.import_from_file(str(export_path))

    _assert_unchanged(target)


def test_failure_in_the_sync_section_rolls_back_everything(tmp_path):
    export_path, _ = _source(tmp_path)
    data = json.loads(export_path.read_text())
    data["sync_events"] = [
        {
            "event_id": "event-one",
            "memory_id": "memory-one",
            "operation": "CREATE",
            "timestamp": "2026-01-01T00:00:00",
            "device_id": "device-one",
            "payload": {"not": "a text column"},
            "event_hash": "hash-one",
        }
    ]
    export_path.write_text(json.dumps(data))
    target = _target(tmp_path)

    with pytest.raises(sqlite3.Error):
        target.import_from_file(str(export_path))

    _assert_unchanged(target)
