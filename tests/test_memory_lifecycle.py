"""Regression tests for Mnemosyne object lifecycle initialization."""

from mnemosyne.core.memory import Mnemosyne


def test_mnemosyne_initializes_beam_and_can_remember(tmp_path):
    memory = Mnemosyne(session_id="lifecycle", db_path=tmp_path / "memory.db")

    assert memory.beam is not None
    memory_id = memory.remember("lifecycle smoke test", source="test")
    assert isinstance(memory_id, str)

    memory.close()


def test_mnemosyne_destructor_does_not_reinitialize_runtime(tmp_path):
    memory = Mnemosyne(session_id="lifecycle", db_path=tmp_path / "memory.db")
    beam = memory.beam

    memory.__del__()

    assert memory.beam is beam


def test_mnemosyne_reconnects_after_previous_owner_closes(tmp_path):
    db_path = tmp_path / "memory.db"
    first = Mnemosyne(session_id="first", db_path=db_path)
    first.close()

    second = Mnemosyne(session_id="second", db_path=db_path)
    assert second.beam is not None
    assert second.remember("reconnect smoke test", source="test")
    second.close()


def test_closing_old_owner_does_not_close_newer_database(tmp_path):
    old = Mnemosyne(session_id="old", db_path=tmp_path / "old.db")
    new = Mnemosyne(session_id="new", db_path=tmp_path / "new.db")

    old.close()

    assert new.remember("new owner remains usable", source="test")
    new.close()
