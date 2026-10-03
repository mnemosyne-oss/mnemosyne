import json
import sqlite3

import mnemosyne.core.importers.hindsight as hindsight_module
from mnemosyne.core.importers import HindsightImporter, import_from_provider
from mnemosyne.core.memory import Mnemosyne


def _sample_items():
    return [
        {
            "id": "hs-world-1",
            "text": "Phin prefers full subject names instead of subject codes.",
            "fact_type": "world",
            "mentioned_at": "2026-04-29T01:36:00+00:00",
            "date": "2026-04-29",
            "proof_count": 2,
            "tags": ["session:school-preferences"],
            "entities": ["Phin"],
            "context": "User preference",
        },
        {
            "id": "hs-exp-1",
            "text": "Hindsight to Mnemosyne migration must preserve timestamps.",
            "fact_type": "experience",
            "mentioned_at": "2026-05-07T00:57:24.052845+00:00",
            "chunk_id": "chunk-abc",
            "proof_count": 1,
        },
    ]


def test_hindsight_importer_preserves_timestamps_and_uses_episodic_memory(tmp_path):
    export = tmp_path / "hindsight-export.json"
    export.write_text(json.dumps({"items": _sample_items()}), encoding="utf-8")

    db_path = tmp_path / "mnemosyne.db"
    mem = Mnemosyne(session_id="default", db_path=db_path)
    result = HindsightImporter(file_path=str(export), bank="hermes").run(mem)

    assert result.failed == 0
    assert result.imported == 2
    assert result.skipped == 0

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, content, source, timestamp, session_id, metadata_json, veracity, scope, channel_id "
        "FROM episodic_memory ORDER BY timestamp"
    ).fetchall()
    assert len(rows) == 2
    assert conn.execute("SELECT COUNT(*) FROM working_memory").fetchone()[0] == 0

    first = rows[0]
    assert first["content"] == "Phin prefers full subject names instead of subject codes."
    assert first["source"] == "hindsight:world"
    assert first["timestamp"] == "2026-04-29T01:36:00+00:00"
    assert first["session_id"] == "session_school-preferences"
    assert first["veracity"] == "imported"
    assert first["scope"] == "global"
    assert first["channel_id"] == "hindsight"
    metadata = json.loads(first["metadata_json"])
    assert metadata["migration_source"] == "hindsight"
    assert metadata["hindsight_bank"] == "hermes"
    assert metadata["hindsight_id"] == "hs-world-1"
    assert metadata["hindsight_fact_type"] == "world"

    fts_hits = conn.execute(
        "SELECT COUNT(*) FROM fts_episodes WHERE fts_episodes MATCH ?",
        ("timestamps",),
    ).fetchone()[0]
    assert fts_hits == 1


def test_hindsight_importer_skips_duplicates_with_stable_ids(tmp_path):
    export = tmp_path / "hindsight-export.json"
    export.write_text(json.dumps(_sample_items()), encoding="utf-8")

    db_path = tmp_path / "mnemosyne.db"
    mem = Mnemosyne(session_id="default", db_path=db_path)
    importer = HindsightImporter(file_path=str(export), bank="hermes")

    first = importer.run(mem)
    second = importer.run(mem)

    assert first.imported == 2
    assert second.imported == 0
    assert second.skipped == 2

    conn = sqlite3.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 2


def test_hindsight_provider_registry_import(tmp_path):
    export = tmp_path / "hindsight-export.json"
    export.write_text(json.dumps({"items": _sample_items()[:1]}), encoding="utf-8")

    db_path = tmp_path / "mnemosyne.db"
    mem = Mnemosyne(session_id="default", db_path=db_path)
    result = import_from_provider("hindsight", mem, file_path=str(export), bank="hermes")

    assert result.provider == "hindsight"
    assert result.imported == 1



def test_hindsight_importer_adds_quality_metadata_and_can_skip_low_value(tmp_path):
    export = tmp_path / "hindsight-export.json"
    export.write_text(json.dumps({"items": [
        _sample_items()[0],
        {
            "id": "hs-meta-prompt",
            "text": "Review the conversation above and consider saving to memory if appropriate. Focus on user preferences.",
            "fact_type": "experience",
            "mentioned_at": "2026-05-15T23:00:00+00:00",
        },
    ]}), encoding="utf-8")

    db_path = tmp_path / "mnemosyne.db"
    mem = Mnemosyne(session_id="default", db_path=db_path)
    result = HindsightImporter(file_path=str(export), bank="hermes", skip_low_value=True).run(mem)

    assert result.failed == 0
    assert result.imported == 1
    assert result.skipped == 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT metadata_json FROM episodic_memory").fetchone()
    metadata = json.loads(row["metadata_json"])
    assert metadata["migration_source"] == "hindsight"
    assert metadata["import_quality_score"] == 1.0
    assert "import_quality_flags" not in metadata


def test_hindsight_importer_canonicalizes_offset_bearing_valid_until(tmp_path):
    """#525: Hindsight _insert_episodic writes raw valid_until to
    episodic_memory, so offset-bearing and space-separated naive values
    must be canonicalized to aware UTC at the import boundary. Otherwise
    the SQL chronological predicates misjudge them."""
    from datetime import datetime, timedelta, timezone

    future_offset = (datetime.now(timezone.utc) + timedelta(hours=4)).astimezone(
        timezone(timedelta(hours=-2))
    )
    past_offset = (datetime.now(timezone.utc) - timedelta(hours=4)).astimezone(
        timezone(timedelta(hours=14))
    )
    space_naive = (datetime.now(timezone.utc) + timedelta(hours=2)).replace(
        tzinfo=None
    ).strftime("%Y-%m-%d %H:%M:%S")

    export = tmp_path / "hindsight-valid-until.json"
    export.write_text(json.dumps({"items": [
        {**_sample_items()[0], "id": "hs-future", "valid_until": future_offset.isoformat()},
        {**_sample_items()[1], "id": "hs-past", "valid_until": past_offset.isoformat()},
        {
            "id": "hs-space",
            "text": "Space-separated naive expiry imported from Hindsight.",
            "fact_type": "world",
            "mentioned_at": "2026-05-07T00:57:24.052845+00:00",
            "valid_until": space_naive,
        },
    ]}), encoding="utf-8")

    db_path = tmp_path / "mnemosyne.db"
    mem = Mnemosyne(session_id="default", db_path=db_path)
    result = HindsightImporter(file_path=str(export), bank="hermes").run(mem)

    assert result.failed == 0
    assert result.imported == 3

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    future_row = conn.execute(
        "SELECT valid_until FROM episodic_memory WHERE content = ?",
        ("Phin prefers full subject names instead of subject codes.",),
    ).fetchone()
    future_dt = datetime.fromisoformat(future_row["valid_until"])
    assert future_dt.utcoffset() == timedelta(0)
    assert future_dt > datetime.now(timezone.utc)

    past_row = conn.execute(
        "SELECT valid_until FROM episodic_memory WHERE content = ?",
        ("Hindsight to Mnemosyne migration must preserve timestamps.",),
    ).fetchone()
    past_dt = datetime.fromisoformat(past_row["valid_until"])
    assert past_dt.utcoffset() == timedelta(0)
    assert past_dt < datetime.now(timezone.utc)

    space_row = conn.execute(
        "SELECT valid_until FROM episodic_memory WHERE content = ?",
        ("Space-separated naive expiry imported from Hindsight.",),
    ).fetchone()
    assert "T" in space_row["valid_until"]
    assert datetime.fromisoformat(space_row["valid_until"]).utcoffset() == timedelta(0)

    # A still-valid imported summary must survive episodic recall; an
    # expired one must not.
    recall_contents = {r["content"] for r in mem.beam.recall("subject codes", top_k=20)}
    assert "Phin prefers full subject names instead of subject codes." in recall_contents
    recall_past = {r["content"] for r in mem.beam.recall("preserve timestamps", top_k=20)}
    assert "Hindsight to Mnemosyne migration must preserve timestamps." not in recall_past


def test_hindsight_importer_date_only_valid_until_passes_through(tmp_path):
    """#525: date-only valid_until values keep pass-through semantics
    across the Hindsight import boundary."""
    export = tmp_path / "hindsight-date-only.json"
    export.write_text(json.dumps({"items": [
        {
            **_sample_items()[0],
            "id": "hs-date",
            "valid_until": "2099-12-31",
        },
    ]}), encoding="utf-8")

    db_path = tmp_path / "mnemosyne.db"
    mem = Mnemosyne(session_id="default", db_path=db_path)
    result = HindsightImporter(file_path=str(export), bank="hermes").run(mem)
    assert result.imported == 1

    conn = sqlite3.connect(db_path)
    stored = conn.execute(
        "SELECT valid_until FROM episodic_memory WHERE content = ?",
        ("Phin prefers full subject names instead of subject codes.",),
    ).fetchone()[0]
    assert stored == "2099-12-31"


def test_hindsight_importer_generates_binary_vectors_when_embedding_backend_available(tmp_path, monkeypatch):
    class FakeEmbeddingBackend:
        @staticmethod
        def available():
            return True

        @staticmethod
        def embed(texts):
            assert texts == ["Phin prefers full subject names instead of subject codes."]
            return [object()]

    monkeypatch.setattr(hindsight_module, "_embeddings", FakeEmbeddingBackend)
    monkeypatch.setattr(hindsight_module, "_vec_available", lambda conn: False)
    monkeypatch.setattr(hindsight_module, "_vec_insert", None)
    monkeypatch.setattr(hindsight_module, "_mib", lambda vec: b"binary-vector")

    export = tmp_path / "hindsight-export.json"
    export.write_text(json.dumps({"items": _sample_items()[:1]}), encoding="utf-8")

    db_path = tmp_path / "mnemosyne.db"
    mem = Mnemosyne(session_id="default", db_path=db_path)
    result = HindsightImporter(file_path=str(export), bank="hermes").run(mem)

    assert result.failed == 0
    assert result.imported == 1

    conn = sqlite3.connect(db_path)
    binary_vector = conn.execute("SELECT binary_vector FROM episodic_memory").fetchone()[0]
    assert binary_vector == b"binary-vector"
