"""Regression coverage for the stable Mnemosyne construction contract."""

from mnemosyne.core.memory import Mnemosyne


def test_mnemosyne_initializes_beam_and_can_remember(tmp_path):
    memory = Mnemosyne(session_id="lifecycle", db_path=tmp_path / "memory.db")

    assert memory.beam is not None
    memory_id = memory.remember("lifecycle smoke test", source="test")
    assert isinstance(memory_id, str)


def test_invalidate_emits_wrapper_event_only_after_success(tmp_path, monkeypatch):
    memory = Mnemosyne(session_id="lifecycle", db_path=tmp_path / "memory.db")
    events = []
    monkeypatch.setattr(memory, "_emit_wrapper", lambda *args, **kwargs: events.append((args, kwargs)))

    monkeypatch.setattr(memory.beam, "invalidate", lambda *args, **kwargs: True)
    assert memory.invalidate("present", replacement_id="replacement") is True

    monkeypatch.setattr(memory.beam, "invalidate", lambda *args, **kwargs: False)
    assert memory.invalidate("missing") is False

    assert events == [
        (("MEMORY_INVALIDATED", "present"), {"replacement_id": "replacement"}),
    ]


def test_invalidate_accepts_authorized_episodic_replacement(tmp_path):
    memory = Mnemosyne(session_id="lifecycle", db_path=tmp_path / "memory.db")
    target_id = memory.beam.remember("working target")
    replacement_id = memory.beam.consolidate_to_episodic(
        "episodic replacement", source_wm_ids=[], source="test", importance=1.0
    )

    assert memory.invalidate(target_id, replacement_id=replacement_id) is True
    row = memory.beam.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (target_id,)
    ).fetchone()
    assert row[0] is not None
    assert row[1] == replacement_id


def test_beam_invalidate_rejects_self_replacement_without_mutation(tmp_path):
    beam = Mnemosyne(session_id="lifecycle", db_path=tmp_path / "memory.db").beam
    memory_id = beam.remember("self replacement target")

    assert beam.invalidate(memory_id, replacement_id=memory_id) is False
    row = beam.conn.execute(
        "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (memory_id,)
    ).fetchone()
    assert tuple(row) == (None, None)


def test_invalidate_replacement_starts_write_transaction_before_lookup(tmp_path):
    """Direct replacement invalidation obtains its write lock before lookup."""
    db_path = tmp_path / "memory.db"
    beam = Mnemosyne(session_id="lifecycle", db_path=db_path).beam
    target_id = beam.remember("atomic invalidation target")
    replacement_id = beam.remember("atomic replacement")

    statements = []
    beam.conn.set_trace_callback(statements.append)
    try:
        assert beam.invalidate(target_id, replacement_id=replacement_id) is True
    finally:
        beam.conn.set_trace_callback(None)

    begin_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.lstrip().upper().startswith("BEGIN IMMEDIATE")
    )
    replacement_lookup_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.lstrip().upper().startswith("SELECT 1 FROM WORKING_MEMORY")
    )
    assert begin_index < replacement_lookup_index
    target_row = beam.conn.execute(
        "SELECT superseded_by FROM working_memory WHERE id = ?", (target_id,)
    ).fetchone()
    assert target_row[0] == replacement_id
