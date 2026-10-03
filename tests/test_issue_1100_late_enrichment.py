"""Regression coverage for issue #1100 late enrichment writes.

The no-new-orphan contract is intentionally bounded to the current storage
identity: working-memory id + session.  It prevents delayed enrichment from
writing after that parent has been deleted, including a same-id parent that is
recreated in another session.  Strict same-session ABA/incarnation safety needs
a durable parent generation and is deliberately outside this migration-free
fix.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

import mnemosyne.core.beam as beam_module
from mnemosyne.core.beam import BeamMemory


@pytest.fixture(autouse=True)
def _disable_embeddings(monkeypatch):
    """Keep the concurrency probes focused on annotation/gist admission."""
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)


def _counts(db_path: Path, memory_id: str) -> tuple[int, int, int]:
    conn = sqlite3.connect(str(db_path))
    try:
        parent = conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()[0]
        annotations = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        gists = conn.execute(
            "SELECT COUNT(*) FROM gists WHERE memory_id = ?", (memory_id,)
        ).fetchone()[0]
        return parent, annotations, gists
    finally:
        conn.close()


def _annotation_count(db_path: Path, memory_id: str, kind: str, value: str) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM annotations "
            "WHERE memory_id = ? AND kind = ? AND value = ?",
            (memory_id, kind, value),
        ).fetchone()[0]
    finally:
        conn.close()


def _gist_text(db_path: Path, memory_id: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT text FROM gists WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def test_delayed_remember_enrichment_cannot_recreate_children_after_forget(
    tmp_path: Path, monkeypatch
):
    """Delete-before-write on separate connections leaves no child rows."""
    db_path = tmp_path / "issue-1100-remember.db"
    deleter = BeamMemory(session_id="owner", db_path=db_path)
    entered = threading.Event()
    resume = threading.Event()
    original = beam_module._extract_and_store_entities
    result: dict[str, object] = {}
    errors: list[BaseException] = []

    def paused_extract(beam, memory_id, content):
        entered.set()
        if not resume.wait(10):
            raise TimeoutError("timed out waiting to resume enrichment")
        return original(beam, memory_id, content)

    monkeypatch.setattr(beam_module, "_extract_and_store_entities", paused_extract)

    def writer():
        try:
            beam = BeamMemory(session_id="owner", db_path=db_path)
            result["memory_id"] = beam.remember(
                "Alice and Bob worked on the issue 1100 race",
                source="document",
                memory_id="issue-1100-remember",
                extract_entities=True,
            )
        except BaseException as exc:  # surface worker failures in the test
            errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(10), "remember() never reached the enrichment barrier"

    assert deleter.forget_working("issue-1100-remember") is True
    assert _counts(db_path, "issue-1100-remember") == (0, 0, 0)

    resume.set()
    thread.join(10)
    assert not thread.is_alive()
    assert errors == []
    assert result["memory_id"] == "issue-1100-remember"
    assert _counts(db_path, "issue-1100-remember") == (0, 0, 0)


def test_delayed_dedup_enrichment_cannot_recreate_children_after_forget(
    tmp_path: Path, monkeypatch
):
    """The dedup-update tail uses the same guarded child-write boundary."""
    db_path = tmp_path / "issue-1100-dedup.db"
    owner = BeamMemory(session_id="owner", db_path=db_path)
    content = "Alice and Bob worked on the issue 1100 dedup race"
    memory_id = owner.remember(content, memory_id="issue-1100-dedup")
    entered = threading.Event()
    resume = threading.Event()
    original = beam_module._extract_and_store_entities
    result: dict[str, object] = {}
    errors: list[BaseException] = []

    def paused_extract(beam, delayed_id, delayed_content):
        entered.set()
        if not resume.wait(10):
            raise TimeoutError("timed out waiting to resume dedup enrichment")
        return original(beam, delayed_id, delayed_content)

    monkeypatch.setattr(beam_module, "_extract_and_store_entities", paused_extract)

    def writer():
        try:
            beam = BeamMemory(session_id="owner", db_path=db_path)
            result["memory_id"] = beam.remember(content, extract_entities=True)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(10), "dedup remember() never reached the barrier"

    assert owner.forget_working(memory_id) is True
    resume.set()
    thread.join(10)

    assert not thread.is_alive()
    assert errors == []
    assert result["memory_id"] == memory_id
    assert _counts(db_path, memory_id) == (0, 0, 0)


def test_delayed_batch_enrichment_cannot_recreate_children_after_forget(
    tmp_path: Path, monkeypatch
):
    """Batch gist + later entity writes are rejected after parent deletion."""
    db_path = tmp_path / "issue-1100-batch.db"
    deleter = BeamMemory(session_id="owner", db_path=db_path)
    entered = threading.Event()
    resume = threading.Event()
    original = BeamMemory._ingest_graph_and_veracity
    captured: dict[str, str] = {}
    result: dict[str, object] = {}
    errors: list[BaseException] = []

    def paused_graph(self, memory_id, content, source, veracity="unknown"):
        captured["memory_id"] = memory_id
        entered.set()
        if not resume.wait(10):
            raise TimeoutError("timed out waiting to resume batch enrichment")
        return original(self, memory_id, content, source, veracity)

    monkeypatch.setattr(BeamMemory, "_ingest_graph_and_veracity", paused_graph)

    def writer():
        try:
            beam = BeamMemory(session_id="owner", db_path=db_path)
            result["ids"] = beam.remember_batch(
                [{"content": "Alice and Bob worked on the issue 1100 batch race",
                  "source": "document"}],
                extract_entities=True,
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=writer)
    thread.start()
    assert entered.wait(10), "remember_batch() never reached the graph barrier"

    memory_id = captured["memory_id"]
    assert deleter.forget_working(memory_id) is True
    assert _counts(db_path, memory_id) == (0, 0, 0)

    resume.set()
    thread.join(10)
    assert not thread.is_alive()
    assert errors == []
    assert result["ids"] == [memory_id]
    assert _counts(db_path, memory_id) == (0, 0, 0)


def test_child_write_before_forget_is_removed_on_separate_connection(tmp_path: Path):
    """The opposite ordering remains safe: write first, delete second."""
    db_path = tmp_path / "issue-1100-write-first.db"
    writer = BeamMemory(session_id="owner", db_path=db_path)
    memory_id = writer.remember(
        "Alice and Bob worked on the issue 1100 write-first case",
        source="document",
        memory_id="issue-1100-write-first",
        extract_entities=True,
    )

    parent, annotations, gists = _counts(db_path, memory_id)
    assert parent == 1
    assert annotations > 0
    assert gists > 0

    result: dict[str, bool] = {}
    errors: list[BaseException] = []

    def delete_on_other_connection():
        try:
            beam = BeamMemory(session_id="owner", db_path=db_path)
            result["forgotten"] = beam.forget_working(memory_id)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=delete_on_other_connection)
    thread.start()
    thread.join(10)

    assert not thread.is_alive()
    assert errors == []
    assert result["forgotten"] is True
    assert _counts(db_path, memory_id) == (0, 0, 0)


def test_live_parent_still_receives_annotations_and_gist(tmp_path: Path):
    """The guard is admission, not a blanket suppression of enrichment."""
    db_path = tmp_path / "issue-1100-live.db"
    beam = BeamMemory(session_id="owner", db_path=db_path)
    memory_id = beam.remember(
        "Alice and Bob worked on a valid episodic enrichment",
        source="document",
        memory_id="issue-1100-live",
        extract_entities=True,
    )

    parent, annotations, gists = _counts(db_path, memory_id)
    assert parent == 1
    assert annotations > 0
    assert gists > 0


def test_late_old_session_tail_does_not_attach_to_same_id_new_session(
    tmp_path: Path,
):
    """The bounded fix rejects same-ID recreation in another session."""
    db_path = tmp_path / "issue-1100-cross-session.db"
    memory_id = "issue-1100-recreated"
    old = BeamMemory(session_id="old-session", db_path=db_path)
    old.remember(
        "Alice owns the old incarnation",
        source="old-source",
        memory_id=memory_id,
    )
    assert old.forget_working(memory_id) is True

    new = BeamMemory(session_id="new-session", db_path=db_path)
    new.remember(
        "Bob owns the new incarnation",
        source="new-source",
        memory_id=memory_id,
        dedupe=False,
    )
    new_gist = _gist_text(db_path, memory_id)
    assert new_gist is not None

    old._add_temporal_triple(
        memory_id,
        "2001-01-01T00:00:00",
        "late-old-source",
        "Alice owns the old incarnation",
    )
    old._ingest_graph_and_veracity(
        memory_id,
        "Alice owns the old incarnation",
        "late-old-source",
    )

    assert _annotation_count(
        db_path, memory_id, "has_source", "late-old-source"
    ) == 0
    assert _gist_text(db_path, memory_id) == new_gist


def test_episodic_same_id_does_not_admit_late_working_children(tmp_path: Path):
    """Admission is tier-specific: an episodic parent is not a working parent."""
    db_path = tmp_path / "issue-1100-tier.db"
    memory_id = "issue-1100-tier"
    beam = BeamMemory(session_id="owner", db_path=db_path)
    beam.remember(
        "Alice owns the working incarnation",
        source="working-source",
        memory_id=memory_id,
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory "
        "(id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, 'episodic incarnation', 'test', datetime('now'), ?, 0.6, 'session')",
        (memory_id, beam.session_id),
    )
    beam.conn.execute(
        "UPDATE gists SET text = 'episodic-owned gist' WHERE memory_id = ?",
        (memory_id,),
    )
    beam.conn.commit()

    # A competing episodic parent makes the shared gist ambiguous, so the
    # existing forget cascade intentionally preserves it (#1002).
    assert beam.forget_working(memory_id) is True
    assert _gist_text(db_path, memory_id) == "episodic-owned gist"

    beam._add_temporal_triple(
        memory_id,
        "2001-01-01T00:00:00",
        "late-working-source",
        "Alice owns the working incarnation",
    )
    beam._ingest_graph_and_veracity(
        memory_id,
        "Alice owns the working incarnation",
        "late-working-source",
    )

    assert _annotation_count(
        db_path, memory_id, "has_source", "late-working-source"
    ) == 0
    assert _gist_text(db_path, memory_id) == "episodic-owned gist"


def test_guarded_annotation_write_does_not_commit_caller_transaction(tmp_path: Path):
    """A guarded child write stays rollbackable inside a caller-owned txn."""
    db_path = tmp_path / "issue-1100-caller-txn.db"
    beam = BeamMemory(session_id="owner", db_path=db_path)
    memory_id = beam.remember(
        "Caller transaction marker",
        memory_id="issue-1100-caller-txn",
    )

    beam.conn.execute("BEGIN")
    beam._add_temporal_triple(
        memory_id,
        "2001-01-01T00:00:00",
        "caller-owned-source",
        "Caller transaction marker",
    )
    beam._ingest_graph_and_veracity(
        memory_id,
        "Caller transaction graph marker",
        "caller-owned-source",
    )
    assert beam.conn.in_transaction is True
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM annotations "
        "WHERE memory_id = ? AND kind = 'has_source' AND value = ?",
        (memory_id, "caller-owned-source"),
    ).fetchone()[0] == 1

    beam.conn.rollback()

    assert beam.conn.in_transaction is False
    assert _annotation_count(
        db_path, memory_id, "has_source", "caller-owned-source"
    ) == 0
