"""Deletion cascade coverage for derived rows and the legacy mirror.

A tier-scoped forget must remove everything derived from the memory: shared
child rows, ``memoria_*`` rows keyed by ``source_memory_id``, and ``facts``
keyed by ``source_msg_id``. The legacy ``memories`` mirror is not a parent
tier and must not keep the cascade from running (#1131, #1035, #1126).
"""
from __future__ import annotations

import mnemosyne.core.beam as beam
from mnemosyne.core.beam import BeamMemory


DERIVED_TABLES = (
    ("memoria_facts", "source_memory_id"),
    ("memoria_timelines", "source_memory_id"),
    ("memoria_instructions", "source_memory_id"),
    ("memoria_preferences", "source_memory_id"),
    ("memoria_kg", "source_memory_id"),
    ("memoria_persona", "source_memory_id"),
    ("facts", "source_msg_id"),
)


def _make_beam(tmp_path) -> BeamMemory:
    return BeamMemory(session_id="s", db_path=str(tmp_path / "m.db"))


def _ensure_legacy_table(memory: BeamMemory) -> None:
    # BeamMemory creates the BEAM schema; the legacy mirror belongs to the
    # facade schema (memory.init_db). Create it here to exercise the guard.
    memory.conn.execute(
        "CREATE TABLE IF NOT EXISTS memories ("
        "id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT, timestamp TEXT, "
        "session_id TEXT DEFAULT 'default', importance REAL DEFAULT 0.5, "
        "metadata_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    memory.conn.commit()


def _seed_derived(memory: BeamMemory, memory_id: str) -> None:
    conn = memory.conn
    conn.execute(
        "INSERT INTO memoria_facts (session_id, fact_type, key, value, source_memory_id) "
        "VALUES ('s', 'entity', 'zeolite', 'zeolite', ?)",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO memoria_timelines (session_id, date, description, source_memory_id) "
        "VALUES ('s', '2026-01-01', 'event', ?)",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO memoria_instructions (session_id, instruction, source_memory_id) "
        "VALUES ('s', 'instruction', ?)",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO memoria_preferences (session_id, preference, source_memory_id) "
        "VALUES ('s', 'preference', ?)",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO memoria_kg (session_id, subject, predicate, object, source_memory_id) "
        "VALUES ('s', 'a', 'b', 'c', ?)",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO memoria_persona (session_id, tier, topic, content, source_memory_id) "
        "VALUES ('s', 'long_term', 'topic', 'content', ?)",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO facts (fact_id, session_id, subject, predicate, object, source_msg_id) "
        "VALUES (?, 's', 'a', 'stated', 'b', ?)",
        (f"fact-{memory_id}", memory_id),
    )
    conn.commit()


def _count_derived(memory: BeamMemory, memory_id: str) -> dict[str, int]:
    return {
        table: memory.conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (memory_id,)
        ).fetchone()[0]
        for table, column in DERIVED_TABLES
    }


def _count(memory: BeamMemory, sql: str, memory_id: str) -> int:
    return memory.conn.execute(sql, (memory_id,)).fetchone()[0]


def test_forget_working_deletes_derived_rows(tmp_path):
    memory = _make_beam(tmp_path)
    memory_id = memory.remember("zeolite contract", scope="global")
    _seed_derived(memory, memory_id)
    assert set(_count_derived(memory, memory_id).values()) == {1}

    assert memory.forget_working(memory_id) is True

    assert _count_derived(memory, memory_id) == {
        table: 0 for table, _ in DERIVED_TABLES
    }


def test_forget_working_ignores_the_legacy_mirror(tmp_path):
    memory = _make_beam(tmp_path)
    memory_id = memory.remember("mirrored memory", scope="global")
    _ensure_legacy_table(memory)
    conn = memory.conn
    conn.execute(
        "INSERT INTO memories (id, content, session_id) VALUES (?, ?, 's')",
        (memory_id, "mirrored memory"),
    )
    conn.execute(
        "INSERT INTO annotations (memory_id, kind, value, source, confidence) "
        "VALUES (?, 'mentions', 'thing', 'test', 0.8)",
        (memory_id,),
    )
    conn.execute(
        "INSERT INTO memory_embeddings (memory_id, embedding_json, model) "
        "VALUES (?, '[]', 'test')",
        (memory_id,),
    )
    conn.commit()

    assert memory.forget_working(memory_id) is True

    assert _count(memory, "SELECT COUNT(*) FROM memories WHERE id = ?", memory_id) == 0
    assert _count(memory, "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", memory_id) == 0
    assert (
        _count(memory, "SELECT COUNT(*) FROM memory_embeddings WHERE memory_id = ?", memory_id)
        == 0
    )


def test_forget_working_keeps_children_for_a_same_id_episodic_parent(tmp_path):
    memory = _make_beam(tmp_path)
    memory_id = memory.remember("shared id", scope="global")
    _ensure_legacy_table(memory)
    conn = memory.conn
    conn.execute(
        "INSERT INTO episodic_memory (id, content, session_id, scope) VALUES (?, ?, 's', 'global')",
        (memory_id, "shared id"),
    )
    conn.execute(
        "INSERT INTO memories (id, content, session_id) VALUES (?, ?, 's')",
        (memory_id, "shared id"),
    )
    conn.execute(
        "INSERT INTO annotations (memory_id, kind, value, source, confidence) "
        "VALUES (?, 'mentions', 'thing', 'test', 0.8)",
        (memory_id,),
    )
    conn.commit()
    _seed_derived(memory, memory_id)
    annotations_before = _count(
        memory, "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", memory_id
    )
    derived_before = _count_derived(memory, memory_id)

    assert memory.forget_working(memory_id) is True

    assert _count(memory, "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", memory_id) == 1
    assert (
        _count(memory, "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", memory_id)
        == annotations_before
    )
    assert _count_derived(memory, memory_id) == derived_before
    assert _count(memory, "SELECT COUNT(*) FROM memories WHERE id = ?", memory_id) == 1


def test_forget_episodic_deletes_derived_rows(tmp_path):
    memory = _make_beam(tmp_path)
    memory_id = "episodic-only"
    memory.conn.execute(
        "INSERT INTO episodic_memory (id, content, session_id, scope) VALUES (?, ?, 's', 'global')",
        (memory_id, "episodic content"),
    )
    memory.conn.commit()
    _seed_derived(memory, memory_id)

    assert memory.forget_episodic(memory_id) is True

    assert set(_count_derived(memory, memory_id).values()) == {0}


def test_trim_deletes_derived_rows_for_trimmed_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(beam, "WORKING_MEMORY_MAX_ITEMS", 1)
    memory = _make_beam(tmp_path)
    memory_id = memory.remember("first memory", scope="global")
    _seed_derived(memory, memory_id)
    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 1
    # Backdate the row so the keep-newest-N boundary is deterministic.
    memory.conn.execute(
        "UPDATE working_memory SET timestamp = ? WHERE id = ?",
        ("2020-01-01T00:00:00+00:00", memory_id),
    )
    memory.conn.commit()

    memory.remember("second memory", scope="global")

    assert memory.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()[0] == 0
    assert set(_count_derived(memory, memory_id).values()) == {0}


def test_delayed_entity_extraction_after_forget_writes_nothing(tmp_path):
    memory = _make_beam(tmp_path)
    live = memory.remember("Abdias likes Mnemosyne", scope="global")
    before = _count(memory, "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", live)
    beam._extract_and_store_entities(memory, live, "Abdias likes Mnemosyne")
    assert _count(memory, "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", live) > before

    memory_id = memory.remember("Abdias likes Mnemosyne again", scope="global")
    assert memory.forget_working(memory_id) is True
    beam._extract_and_store_entities(memory, memory_id, "Abdias likes Mnemosyne")
    assert _count(
        memory, "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", memory_id
    ) == 0


def test_delayed_graph_enrichment_after_forget_writes_nothing(tmp_path):
    memory = _make_beam(tmp_path)
    if memory.episodic_graph is None:
        import pytest

        pytest.skip("episodic graph unavailable")
    live = memory.remember("Abdias met Maya in Berlin", scope="global")
    memory._ingest_graph_and_veracity(live, "Abdias met Maya in Berlin", "test")
    assert _count(memory, "SELECT COUNT(*) FROM gists WHERE memory_id = ?", live) > 0

    memory_id = memory.remember("Abdias met Maya in Rome", scope="global")
    assert memory.forget_working(memory_id) is True
    memory._ingest_graph_and_veracity(memory_id, "Abdias met Maya in Rome", "test")
    assert _count(memory, "SELECT COUNT(*) FROM gists WHERE memory_id = ?", memory_id) == 0
