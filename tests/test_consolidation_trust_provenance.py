"""New episodic writes retain source-derived trust without rewriting history."""

from contextlib import closing
import json
import sqlite3

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core import local_llm
from mnemosyne.core.filters import make_write_policy, write_policy_operation


@pytest.fixture
def memory(tmp_path, monkeypatch):
    """Use a disposable bank with no model or embedding calls."""
    monkeypatch.setattr(beam_module._embeddings, "available", lambda: False)
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)
    monkeypatch.setattr(
        "mnemosyne.core.model_refresh.infer_model_update_proposals", lambda items: []
    )
    bank = beam_module.BeamMemory(
        db_path=tmp_path / "memory.db",
        session_id="trust-review",
        channel_id="writer-channel",
    )
    yield bank
    bank.conn.close()


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, "DERIVED", id="default-consolidation"),
        pytest.param({"source": "sleep_consolidation"}, "DERIVED", id="sleep"),
        pytest.param({"source": "mcp"}, "EXTERNAL_WRITE", id="external"),
        pytest.param(
            {"source": "partner_mcp"}, "EXTERNAL_WRITE", id="external-heuristic"
        ),
        pytest.param({"source": "import"}, "IMPORTED", id="import"),
        pytest.param({"source": "legacy_import"}, "IMPORTED", id="import-heuristic"),
        pytest.param({"source": "user"}, "STATED", id="user"),
        pytest.param({"source": "cli"}, "STATED", id="cli"),
        pytest.param({"source": "unrecognized"}, "STATED", id="unknown"),
        pytest.param({"source": None}, "STATED", id="null-source"),
    ],
)
def test_direct_consolidation_persists_source_trust(memory, kwargs, expected):
    """Persist the established mapping, rather than one tier for every caller."""
    mid = memory.consolidate_to_episodic(
        "A synthetic summary with source provenance.", [], **kwargs
    )

    with closing(sqlite3.connect(memory.db_path)) as reader:
        assert reader.execute(
            "SELECT trust_tier, channel_id FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone() == (expected, "writer-channel")


@pytest.mark.parametrize(
    "llm", [False, True], ids=["heuristic-fallback", "llm-summary"]
)
def test_sleep_persists_derived_summary_trust(memory, monkeypatch, llm):
    """Both actual sleep summary paths must write DERIVED, preserving originals."""
    calls = []

    def summarize(lines, **kwargs):
        calls.append(lines)
        return "A synthetic summary of the user's stored project preference."

    monkeypatch.setattr(local_llm, "llm_available", lambda: llm)
    monkeypatch.setattr(
        local_llm, "chunk_memories_by_budget", lambda lines, **kw: [lines]
    )
    monkeypatch.setattr(local_llm, "_summarize_memories", summarize)
    original = memory.remember(
        "User prefers project notes in plain text.", source="conversation"
    )
    memory.conn.execute(
        "UPDATE working_memory SET timestamp = '2020-01-01T00:00:00' WHERE id = ?",
        (original,),
    )
    memory.conn.commit()

    result = memory.sleep(force=True)

    assert result["status"] == "consolidated"
    assert len(calls) == int(llm)
    rows = memory.conn.execute(
        "SELECT source, trust_tier, channel_id, summary_of, metadata_json FROM episodic_memory"
    ).fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["source"] == "sleep_consolidation"
    assert row["trust_tier"] == "DERIVED"
    assert row["channel_id"] == "writer-channel"
    assert row["summary_of"] == original
    assert json.loads(row["metadata_json"])["llm_used"] is llm
    assert (
        memory.conn.execute(
            "SELECT trust_tier FROM working_memory WHERE id = ?", (original,)
        ).fetchone()[0]
        == "STATED"
    )


@pytest.mark.parametrize("historic_trust", [None, "STATED"])
def test_new_consolidation_leaves_historical_trust_unchanged(memory, historic_trust):
    """Future writes must not guess how old rows acquired their stored tier."""
    old = memory.consolidate_to_episodic("An older summary retained in the bank.", [])
    memory.conn.execute(
        "UPDATE episodic_memory SET trust_tier = ? WHERE id = ?", (historic_trust, old)
    )
    memory.conn.commit()

    new = memory.consolidate_to_episodic(
        "A new summary using source-derived provenance.", []
    )

    assert (
        memory.conn.execute(
            "SELECT trust_tier FROM episodic_memory WHERE id = ?", (old,)
        ).fetchone()[0]
        == historic_trust
    )
    assert (
        memory.conn.execute(
            "SELECT trust_tier FROM episodic_memory WHERE id = ?", (new,)
        ).fetchone()[0]
        == "DERIVED"
    )


@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_consolidation_trust_keeps_caller_transaction_ownership(memory, commit):
    """The provenance and row become visible only when their caller commits."""
    memory.conn.execute("BEGIN")
    mid = memory.consolidate_to_episodic(
        "A caller-owned transactional summary.", [], emit_event=False
    )
    assert memory.conn.in_transaction
    assert (
        memory.conn.execute(
            "SELECT trust_tier FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone()[0]
        == "DERIVED"
    )
    with closing(sqlite3.connect(memory.db_path)) as reader:
        assert (
            reader.execute(
                "SELECT COUNT(*) FROM episodic_memory WHERE id = ?", (mid,)
            ).fetchone()[0]
            == 0
        )
        if commit:
            memory.conn.commit()
        else:
            memory.conn.rollback()
        row = reader.execute(
            "SELECT trust_tier FROM episodic_memory WHERE id = ?", (mid,)
        ).fetchone()
        assert row == (("DERIVED",) if commit else None)


def test_late_consolidation_failure_rolls_back_provenance_and_row(memory):
    """An error after the INSERT must not persist a partially written summary."""
    memory.conn.execute("""
        CREATE TRIGGER reject_event_date BEFORE UPDATE OF event_date ON episodic_memory
        BEGIN SELECT RAISE(ABORT, 'controlled late consolidation failure'); END
    """)
    memory.conn.commit()

    with pytest.raises(
        sqlite3.IntegrityError, match="controlled late consolidation failure"
    ):
        memory.consolidate_to_episodic(
            "A summary whose late write fails.", [], event_date="2026-11-12"
        )

    assert not memory.conn.in_transaction
    with closing(sqlite3.connect(memory.db_path)) as reader:
        assert reader.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 0


def test_derived_source_name_cannot_bypass_public_write_admission(memory):
    """A caller-controlled source label must not confer the internal write exemption."""
    policy = make_write_policy("LOCAL_TRUST_BLOCK", "strict")
    with write_policy_operation(policy):
        mid = memory.consolidate_to_episodic(
            "LOCAL_TRUST_BLOCK rejected content", [], source="sleep_consolidation"
        )
    assert mid is None
    assert (
        memory.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 0
    )
