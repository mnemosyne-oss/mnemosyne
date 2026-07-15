"""Fail-closed and atomic sleep/consolidation regressions."""
from __future__ import annotations

import inspect
import json
import sys
from datetime import datetime, timedelta

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core import aaak
from mnemosyne.core import local_llm
from mnemosyne.core import model_refresh
from mnemosyne.core.beam import BeamMemory


def _seed_old(beam: BeamMemory, memory_id: str = "wm-old", source: str = "conversation") -> None:
    old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES (?, ?, ?, ?, ?)",
        (memory_id, "durable source memory", source, old_ts, beam.session_id),
    )
    beam.conn.commit()


def _source_state(beam: BeamMemory, memory_id: str = "wm-old"):
    return beam.conn.execute(
        "SELECT consolidated_at, consolidation_claimed_at FROM working_memory WHERE id = ?",
        (memory_id,),
    ).fetchone()


def _assert_no_claims_or_episodes(beam: BeamMemory) -> None:
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE consolidation_claimed_at IS NOT NULL"
    ).fetchone()[0] == 0
    assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 0
    assert beam.conn.in_transaction is False


def test_sleep_all_sessions_propagates_owner_and_context(tmp_path, monkeypatch):
    parent = BeamMemory(session_id="primary", db_path=tmp_path / "memory.db")
    parent.canonical_owner_id = "research_profile"
    parent.agent_context = "cron"
    old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
    parent.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES (?, ?, ?, ?, ?)",
        ("alien-old", "alien durable memory", "conversation", old_ts, "alien"),
    )
    parent.conn.commit()
    observed = []

    def fake_sleep(self, dry_run=False, force=False):
        observed.append((self.session_id, self.canonical_owner_id, self.agent_context))
        return {
            "status": "no_op",
            "items_consolidated": 0,
            "summaries_created": 0,
            "llm_used": 0,
            "model_refresh": {"proposals": 0, "applied": 0},
        }

    monkeypatch.setattr(BeamMemory, "sleep", fake_sleep)
    parent.sleep_all_sessions(force=True)

    assert observed == [("alien", "research_profile", "cron")]


def test_claim_tokens_are_unique_even_at_the_same_timestamp():
    now = datetime(2026, 7, 14, 12, 0, 0)

    first = beam_module._new_consolidation_claim_token(now)
    second = beam_module._new_consolidation_claim_token(now)

    assert first != second
    assert first.startswith(f"{now.isoformat()}:")
    assert second.startswith(f"{now.isoformat()}:")


def test_selected_llm_failure_releases_claim_and_creates_no_episode(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    beam.agent_context = "cron"
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda lines, source=None: [lines])
    monkeypatch.setattr(local_llm, "summarize_memories", lambda lines, source=None: None)

    result = beam.sleep()

    assert result["status"] == "failed"
    assert result["items_consolidated"] == 0
    assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 0
    state = _source_state(beam)
    assert state["consolidated_at"] is None
    assert state["consolidation_claimed_at"] is None
    assert beam.conn.in_transaction is False


def test_selected_llm_exception_releases_claim_before_propagating(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    beam.agent_context = "cron"
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    monkeypatch.setattr(
        local_llm,
        "chunk_memories_by_budget",
        lambda lines, source=None: [lines],
    )

    def raise_backend_error(lines, source=None):
        raise RuntimeError("selected backend failed")

    monkeypatch.setattr(local_llm, "summarize_memories", raise_backend_error)

    with pytest.raises(RuntimeError, match="selected backend failed"):
        beam.sleep()

    assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 0
    state = _source_state(beam)
    assert state["consolidated_at"] is None
    assert state["consolidation_claimed_at"] is None
    assert beam.conn.in_transaction is False


def test_compression_plugin_exception_releases_all_claims(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    beam.agent_context = "cron"
    _seed_old(beam, memory_id="wm-first", source="conversation")
    _seed_old(beam, memory_id="wm-second", source="tool")
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)

    class FailingCompressionPlugin:
        enabled = True

        @staticmethod
        def compress_lines(lines):
            raise RuntimeError("compression plugin failed")

    class PluginManager:
        @staticmethod
        def get_plugin(name):
            assert name == "compression"
            return FailingCompressionPlugin()

    monkeypatch.setattr(beam_module._plugins, "get_manager", PluginManager)

    with pytest.raises(RuntimeError, match="compression plugin failed"):
        beam.sleep()

    assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE consolidation_claimed_at IS NOT NULL"
    ).fetchone()[0] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE consolidated_at IS NOT NULL"
    ).fetchone()[0] == 0
    assert beam.conn.in_transaction is False


def test_llm_availability_base_exception_releases_all_claims(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam)

    def interrupt():
        raise KeyboardInterrupt("availability interrupted")

    monkeypatch.setattr(local_llm, "llm_available", interrupt)

    with pytest.raises(KeyboardInterrupt, match="availability interrupted"):
        beam.sleep()

    _assert_no_claims_or_episodes(beam)


def test_conflict_detection_base_exception_releases_all_claims(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam, memory_id="wm-first")
    _seed_old(beam, memory_id="wm-second")

    def interrupt(_items):
        raise SystemExit("conflict interrupted")

    monkeypatch.setattr(beam, "_detect_conflicts", interrupt)

    with pytest.raises(SystemExit, match="conflict interrupted"):
        beam.sleep()

    _assert_no_claims_or_episodes(beam)


def test_aaak_base_exception_releases_all_claims(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)

    def interrupt(_content):
        raise KeyboardInterrupt("AAAK interrupted")

    monkeypatch.setattr(aaak, "encode", interrupt)

    with pytest.raises(KeyboardInterrupt, match="AAAK interrupted"):
        beam.sleep()

    _assert_no_claims_or_episodes(beam)


def test_model_refresh_base_exception_releases_all_claims(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)

    def interrupt(_items):
        raise KeyboardInterrupt("model refresh interrupted")

    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", interrupt)

    with pytest.raises(KeyboardInterrupt, match="model refresh interrupted"):
        beam.sleep()

    _assert_no_claims_or_episodes(beam)


def test_finalize_base_exception_rolls_back_and_releases_all_claims(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)
    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", lambda _items: [])

    def interrupt(**_kwargs):
        raise KeyboardInterrupt("finalize interrupted")

    monkeypatch.setattr(beam, "consolidate_to_episodic", interrupt)

    with pytest.raises(KeyboardInterrupt, match="finalize interrupted"):
        beam.sleep()

    _assert_no_claims_or_episodes(beam)


def test_post_commit_base_exception_releases_unprocessed_group_claims(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam, memory_id="wm-first", source="conversation")
    _seed_old(beam, memory_id="wm-second", source="tool")
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)
    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", lambda _items: [])

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("post-commit interrupted")

    monkeypatch.setattr(beam, "_ingest_graph_and_veracity", interrupt)

    with pytest.raises(KeyboardInterrupt, match="post-commit interrupted"):
        beam.sleep()

    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory WHERE consolidation_claimed_at IS NOT NULL"
    ).fetchone()[0] == 0
    assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 1
    assert beam.conn.in_transaction is False


@pytest.mark.parametrize(
    "source_marker",
    [
        "grouped.setdefault",
        'lines = [item["content"]',
        'aggregated_scope = "session"',
        "degrade_result = (",
    ],
    ids=["grouping", "group-data", "aggregation", "housekeeping"],
)
def test_unhandled_post_claim_base_exception_releases_owned_claims(
    tmp_path, monkeypatch, source_marker
):
    beam = BeamMemory(session_id="outer-claim-guard", db_path=tmp_path / "outer.db")
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)
    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", lambda _items: [])

    implementation = getattr(BeamMemory, "_sleep_impl", BeamMemory.sleep)
    source_lines, first_line = inspect.getsourcelines(implementation)
    matching = [
        first_line + index
        for index, line in enumerate(source_lines)
        if source_marker in line
    ]
    assert len(matching) == 1, (source_marker, matching)
    target_line = matching[0]

    def interrupt_at_target(frame, event, _arg):
        if (
            event == "line"
            and frame.f_code is implementation.__code__
            and frame.f_lineno == target_line
        ):
            raise KeyboardInterrupt(f"interrupted at {source_marker}")
        return interrupt_at_target

    sys.settrace(interrupt_at_target)
    try:
        with pytest.raises(KeyboardInterrupt, match="interrupted at"):
            beam.sleep(force=True)
    finally:
        sys.settrace(None)

    assert beam.conn.in_transaction is False
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory "
        "WHERE consolidation_claimed_at IS NOT NULL"
    ).fetchone()[0] == 0


def test_finalize_marker_failure_rolls_back_episode_and_releases_claim(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    beam.agent_context = "cron"
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda lines, source=None: [lines])
    monkeypatch.setattr(local_llm, "summarize_memories", lambda lines, source=None: "safe summary")
    beam.conn.execute("""
        CREATE TRIGGER reject_sleep_finalize
        BEFORE UPDATE OF consolidated_at ON working_memory
        WHEN NEW.consolidated_at IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'marker failure');
        END
    """)
    beam.conn.commit()

    result = beam.sleep()

    assert result["status"] == "failed"
    assert beam.conn.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0] == 0
    state = _source_state(beam)
    assert state["consolidated_at"] is None
    assert state["consolidation_claimed_at"] is None
    assert beam.conn.in_transaction is False


def test_reclaim_orphans_handles_new_claim_shape(tmp_path):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam)
    stale = (datetime.now() - timedelta(hours=2)).isoformat()
    beam.conn.execute(
        "UPDATE working_memory SET consolidated_at = NULL, consolidation_claimed_at = ? WHERE id = ?",
        (stale, "wm-old"),
    )
    beam.conn.commit()

    result = beam.reclaim_orphans(stale_after_seconds=1)

    assert result["status"] == "reclaimed"
    assert result["reclaimed"] == 1
    state = _source_state(beam)
    assert state["consolidated_at"] is None
    assert state["consolidation_claimed_at"] is None


def test_success_commits_episode_and_source_markers_together(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    beam.agent_context = "cron"
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: True)
    monkeypatch.setattr(local_llm, "chunk_memories_by_budget", lambda lines, source=None: [lines])
    monkeypatch.setattr(local_llm, "summarize_memories", lambda lines, source=None: "safe summary")

    result = beam.sleep()

    assert result["status"] == "consolidated"
    episode = beam.conn.execute(
        "SELECT content, summary_of FROM episodic_memory WHERE source = 'sleep_consolidation'"
    ).fetchone()
    assert episode["content"] == "safe summary"
    assert episode["summary_of"] == "wm-old"
    state = _source_state(beam)
    assert state["consolidated_at"] is not None
    assert state["consolidation_claimed_at"] is None
    assert beam.conn.in_transaction is False


def test_proposal_marker_failure_rolls_back_proposal_insert(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam)
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)
    monkeypatch.setattr(
        model_refresh,
        "infer_model_update_proposals",
        lambda _items: [{
            "category": "model:user",
            "name": "review_only",
            "body": "This proposal must be atomically finalized.",
            "confidence": 0.9,
            "evidence_ids": ["wm-old"],
            "action": "update",
            "reason": "Regression fixture.",
        }],
    )
    beam.conn.execute("""
        CREATE TRIGGER reject_proposal_finalize
        BEFORE UPDATE OF consolidated_at ON working_memory
        WHEN OLD.source = 'sleep_model_refresh_proposal'
             AND NEW.consolidated_at IS NOT NULL
        BEGIN
            SELECT RAISE(ABORT, 'proposal marker failure');
        END
    """)
    beam.conn.commit()

    result = beam.sleep()

    assert result["status"] == "consolidated"
    assert result["model_refresh"]["proposals"] == 0
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory "
        "WHERE source = 'sleep_model_refresh_proposal'"
    ).fetchone()[0] == 0


def test_model_refresh_proposal_does_not_repurpose_matching_user_memory(
    tmp_path, monkeypatch
):
    beam = BeamMemory(session_id="sleep-txn", db_path=tmp_path / "memory.db")
    _seed_old(beam)
    proposal = {
        "category": "model:user",
        "name": "review_only",
        "body": "This proposal must remain a separate review artifact.",
        "confidence": 0.9,
        "evidence_ids": ["wm-old"],
        "action": "update",
        "reason": "Regression fixture.",
    }
    proposal_content = model_refresh.proposal_to_memory_content(proposal)
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES ('wm-user', ?, 'manual', ?, ?)",
        (proposal_content, datetime.now().isoformat(), beam.session_id),
    )
    beam.conn.commit()
    monkeypatch.setattr(local_llm, "llm_available", lambda: False)
    monkeypatch.setattr(
        model_refresh, "infer_model_update_proposals", lambda _items: [proposal]
    )

    result = beam.sleep()

    assert result["model_refresh"]["proposals"] == 1
    assert beam.conn.execute(
        "SELECT source FROM working_memory WHERE id = 'wm-user'"
    ).fetchone()[0] == "manual"
    assert beam.conn.execute(
        "SELECT COUNT(*) FROM working_memory "
        "WHERE source = 'sleep_model_refresh_proposal'"
    ).fetchone()[0] == 1


def test_sleep_all_sessions_aggregates_partial_child_results(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="maintenance", db_path=tmp_path / "memory.db")
    old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES ('child-old', 'child memory', 'conversation', ?, 'child')",
        (old_ts,),
    )
    beam.conn.commit()

    def partial_sleep(_beam, dry_run=False, force=False):
        return {
            "status": "partial",
            "items_consolidated": 1,
            "summaries_created": 1,
            "llm_used": 1,
            "model_refresh": {"proposals": 1, "applied": 0},
        }

    monkeypatch.setattr(BeamMemory, "sleep", partial_sleep)

    result = beam.sleep_all_sessions()

    assert result["status"] == "partial"
    assert result["sessions_consolidated"] == 1
    assert result["items_consolidated"] == 1
    assert result["summaries_created"] == 1
    assert result["llm_used"] == 1
    assert result["model_refresh"] == {"proposals": 1, "applied": 0}


def test_sleep_all_sessions_reports_mixed_failed_child_as_partial(tmp_path, monkeypatch):
    beam = BeamMemory(session_id="maintenance", db_path=tmp_path / "memory.db")
    old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
    beam.conn.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
        "VALUES (?, ?, 'conversation', ?, ?)",
        [
            ("good-old", "good child memory", old_ts, "child-good"),
            ("bad-old", "bad child memory", old_ts, "child-bad"),
        ],
    )
    beam.conn.commit()

    def mixed_sleep(child, dry_run=False, force=False):
        if child.session_id == "child-good":
            return {
                "status": "consolidated",
                "items_consolidated": 1,
                "summaries_created": 1,
                "llm_used": 0,
                "model_refresh": {"proposals": 0, "applied": 0},
            }
        return {
            "status": "failed",
            "items_consolidated": 0,
            "summaries_created": 0,
            "llm_used": 0,
            "failed_groups": [{"source": "conversation"}],
            "model_refresh": {"proposals": 0, "applied": 0},
        }

    monkeypatch.setattr(BeamMemory, "sleep", mixed_sleep)

    result = beam.sleep_all_sessions()

    assert result["status"] == "partial"
    assert result["sessions_scanned"] == 2
    assert result["sessions_consolidated"] == 1
    assert result["items_consolidated"] == 1
    assert result["errors"] == 1
    assert result["error_details"][0]["session_id"] == "child-bad"
    assert {item["status"] for item in result["session_results"]} == {
        "consolidated",
        "failed",
    }
