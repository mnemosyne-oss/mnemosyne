"""Derived voices must not return rows whose source memory is gone (#1035).

The deletion cascade removes derived rows at delete time. Rows created before
that contract, or preserved by an import, can still point at a missing parent.
The read guard drops those from ``memoria_retrieve`` and ``fact_recall`` without
deleting anything. Legacy ``source_msg_id`` message indices are left alone.
"""
from __future__ import annotations

from mnemosyne.core.beam import BeamMemory


def _memory(tmp_path) -> BeamMemory:
    return BeamMemory(session_id="s", db_path=str(tmp_path / "m.db"))


def _insert_derived(beam: BeamMemory, source: str, suffix: str) -> None:
    beam.conn.execute(
        "INSERT INTO memoria_facts "
        "(session_id, fact_type, key, value, context_snippet, source_memory_id) "
        "VALUES ('s', 'entity', ?, ?, ?, ?)",
        (f"Zeolite{suffix}", f"Zeolite{suffix}", f"Zeolite{suffix} contract", source),
    )
    beam.conn.execute(
        "INSERT INTO facts "
        "(fact_id, session_id, subject, predicate, object, source_msg_id) "
        "VALUES (?, 's', 'a', 'stated', ?, ?)",
        (f"fact-{suffix}", f"Zeolite{suffix} contract", source),
    )
    beam.conn.commit()


def test_orphaned_derived_rows_are_not_returned(tmp_path):
    beam = _memory(tmp_path)
    live = beam.remember("Zeolitealpha source memory", scope="global")
    orphan = "0123456789abcdef"
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE id = ?", (orphan,)
    ).fetchone()[0] == 0
    _insert_derived(beam, live, "alpha")
    _insert_derived(beam, orphan, "omega")

    assert beam.memoria_retrieve("what is Zeolitealpha", ability="IE")["facts"]
    assert beam.memoria_retrieve("what is Zeoliteomega", ability="IE")["facts"] == []

    assert beam.fact_recall("Zeolitealpha")
    assert beam.fact_recall("Zeoliteomega") == []


def test_legacy_message_index_sources_are_not_filtered(tmp_path):
    beam = _memory(tmp_path)
    _insert_derived(beam, "42", "beta")

    assert beam.memoria_retrieve("what is Zeolitebeta", ability="IE")["facts"]
    assert beam.fact_recall("Zeolitebeta")


def test_source_liveness_predicate(tmp_path):
    beam = _memory(tmp_path)
    live = beam.remember("live source", scope="global")

    assert beam._derived_source_is_alive(live) is True
    assert beam._derived_source_is_alive("0123456789abcdef") is False
    assert beam._derived_source_is_alive(None) is True
    assert beam._derived_source_is_alive("42") is True
