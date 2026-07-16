"""Regression coverage for the stable Mnemosyne construction contract."""

from mnemosyne.core.memory import Mnemosyne


def test_mnemosyne_initializes_beam_and_can_remember(tmp_path):
    memory = Mnemosyne(session_id="lifecycle", db_path=tmp_path / "memory.db")

    assert memory.beam is not None
    memory_id = memory.remember("lifecycle smoke test", source="test")
    assert isinstance(memory_id, str)
