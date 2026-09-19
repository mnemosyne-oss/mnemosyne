"""
Core per-write author identity tests (issue #914 companion).

Contract: remember() accepts optional author_id/author_type that override the
instance identity for THAT write only. Read identity (self.author_id, used by
recall author-scoping) is never mutated. None/omitted -> falls back to
self.author_id (exact legacy behavior).
"""

import os
from pathlib import Path
import tempfile

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def beam():
    tmpdir = Path(tempfile.mkdtemp())
    old = os.environ.get("MNEMOSYNE_DATA_DIR")
    os.environ["MNEMOSYNE_DATA_DIR"] = str(tmpdir)
    bm = BeamMemory(session_id="t")
    yield bm
    if old:
        os.environ["MNEMOSYNE_DATA_DIR"] = old
    else:
        os.environ.pop("MNEMOSYNE_DATA_DIR", None)


class TestPerWriteAuthorIdentity:
    def test_per_write_author_id_stored(self, beam):
        """remember(author_id=...) writes that author, instance stays None."""
        beam.remember("fact one", author_id="agent-x")
        row = beam.conn.execute(
            "SELECT author_id, author_type FROM working_memory WHERE content='fact one'"
        ).fetchone()
        assert row[0] == "agent-x"
        assert beam.author_id is None  # read identity untouched

    def test_per_write_falls_back_to_instance(self, beam):
        """No per-write author -> instance identity (legacy behavior)."""
        bm = BeamMemory(session_id="t", author_id="inst-agent")
        bm.remember("fact two")
        row = bm.conn.execute(
            "SELECT author_id FROM working_memory WHERE content='fact two'"
        ).fetchone()
        assert row[0] == "inst-agent"

    def test_per_write_overrides_instance(self, beam):
        bm = BeamMemory(session_id="t", author_id="inst-agent")
        bm.remember("fact three", author_id="other-agent")
        row = bm.conn.execute(
            "SELECT author_id FROM working_memory WHERE content='fact three'"
        ).fetchone()
        assert row[0] == "other-agent"
        assert bm.author_id == "inst-agent"  # instance unchanged

    def test_per_write_author_type(self, beam):
        beam.remember("fact four", author_id="agent-y", author_type="agent")
        row = beam.conn.execute(
            "SELECT author_id, author_type FROM working_memory WHERE content='fact four'"
        ).fetchone()
        assert tuple(row) == ("agent-y", "agent")

    def test_dedup_update_uses_per_write_author(self, beam):
        """Re-remembering same content with a per-write author upgrades the row."""
        beam.remember("dup fact", author_id="first")
        beam.remember("dup fact", author_id="second")
        row = beam.conn.execute(
            "SELECT author_id FROM working_memory WHERE content='dup fact'"
        ).fetchone()
        assert row[0] == "second"

    def test_batch_per_write_author(self, beam):
        """remember_batch items honor per-item author_id."""
        bm = BeamMemory(session_id="t")
        bm.remember_batch([
            {"content": "b1", "author_id": "a1"},
            {"content": "b2"},
        ])
        rows = dict(bm.conn.execute(
            "SELECT content, author_id FROM working_memory WHERE content IN ('b1','b2')"
        ).fetchall())
        assert rows["b1"] == "a1"
        assert rows["b2"] is None
