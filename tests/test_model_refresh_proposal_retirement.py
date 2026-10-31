"""Resolved model-refresh proposals must stop being recall candidates.

Sleep writes every model-refresh proposal as a ``working_memory`` row with
``source='sleep_model_refresh_proposal'`` and sets ``consolidated_at``
immediately, so consolidation never retires it. The resolution paths only
flipped ``metadata["status"]``; ``superseded_by`` stayed NULL, and recall
selects on ``superseded_by IS NULL``. The result: every applied or rejected
proposal was recalled forever -- on a real bank 998 of 6,493 working rows
(15%) were resolved proposals still competing for recall slots.

These tests pin the whole contract: a resolved proposal is retired from
recall while the row survives for audit, the FTS candidate budget no longer
counts retired rows, a pending proposal in the same bank stays visible (the
negative control that keeps the assertions from passing vacuously), and a
resolved proposal keeps its outcome when a second resolution is attempted.
"""

from __future__ import annotations

import json

from mnemosyne.core.beam import (
    BeamMemory,
    _cjk_like_search,
    _cyrillic_like_search,
    _fts_search_working,
)
from mnemosyne.core import model_refresh


def _proposal(token: str, **overrides):
    base = {
        "category": "model:user",
        "name": f"retirement_{token}",
        "body": f"Durable preference about {token} reporting.",
        "confidence": 0.95,
        "evidence_ids": [f"wm-{token}-a", f"wm-{token}-b"],
        "action": "update",
        "reason": "Repeated durable preference.",
    }
    base.update(overrides)
    return base


def _store_proposal(beam, token: str, **overrides):
    proposal = _proposal(token, **overrides)
    metadata = model_refresh.prepare_proposal_metadata(
        proposal, source_wm_ids=proposal["evidence_ids"]
    )
    proposal_id = beam.remember(
        model_refresh.proposal_to_memory_content(proposal),
        source=model_refresh.PROPOSAL_SOURCE,
        metadata=metadata,
    )
    assert proposal_id, "beam.remember must return the stored proposal id"
    # Sleep sets consolidated_at on the proposal at creation time; without it
    # the fixture would be consolidated away and the test would prove nothing.
    beam.conn.execute(
        "UPDATE working_memory SET consolidated_at = datetime('now') WHERE id = ?",
        (proposal_id,),
    )
    beam.conn.commit()
    return proposal_id


def _recalled_ids(beam, token: str):
    return {
        row.get("id")
        for row in beam.recall(f"retirement_{token} reporting preference", top_k=10)
    }


def _row(beam, proposal_id: str) -> dict:
    row = beam.conn.execute(
        "SELECT content, metadata_json, source, consolidated_at, superseded_by "
        "FROM working_memory WHERE id = ?",
        (proposal_id,),
    ).fetchone()
    assert row is not None, "proposal row must survive for audit"
    return dict(row)


class TestResolvedProposalRetirement:

    def test_applied_proposal_is_retired_from_recall(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "applied")

        # Negative control: while pending, the proposal is a recall candidate.
        assert proposal_id in _recalled_ids(beam, "applied")
        assert _row(beam, proposal_id)["superseded_by"] is None

        model_refresh.apply_model_refresh_proposal(beam, proposal_id)

        assert _row(beam, proposal_id)["superseded_by"] == "model_refresh:applied"
        assert proposal_id not in _recalled_ids(beam, "applied")

    def test_rejected_proposal_is_retired_from_recall(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "rejected")

        assert proposal_id in _recalled_ids(beam, "rejected")

        model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="stale")

        assert _row(beam, proposal_id)["superseded_by"] == "model_refresh:rejected"
        assert proposal_id not in _recalled_ids(beam, "rejected")

    def test_row_and_metadata_are_kept_verbatim_for_audit(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        rejected_id = _store_proposal(beam, "audit_reject")
        applied_id = _store_proposal(beam, "audit_apply")

        before_rejected = _row(beam, rejected_id)
        before_applied = _row(beam, applied_id)

        model_refresh.reject_model_refresh_proposal(beam, rejected_id, reason="noise")
        model_refresh.apply_model_refresh_proposal(beam, applied_id)

        for proposal_id, before, status in (
            (rejected_id, before_rejected, "rejected"),
            (applied_id, before_applied, "applied"),
        ):
            after = _row(beam, proposal_id)
            assert after["content"] == before["content"]
            assert after["source"] == model_refresh.PROPOSAL_SOURCE
            assert after["consolidated_at"] == before["consolidated_at"]
            metadata = json.loads(after["metadata_json"])
            assert metadata["status"] == status
            # Resolution adds fields; it must not drop what preparation stored.
            previous = json.loads(before["metadata_json"])
            for key in ("category", "name", "body", "confidence", "source_wm_ids"):
                assert metadata[key] == previous[key]

    def test_other_pending_proposal_stays_eligible(self, tmp_path):
        """Resolving one proposal must not hide a similar pending one."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        first = _store_proposal(beam, "shared")
        second = _store_proposal(beam, "shared", name="retirement_second")

        model_refresh.reject_model_refresh_proposal(beam, first, reason="stale")

        assert _row(beam, first)["superseded_by"] == "model_refresh:rejected"
        assert _row(beam, second)["superseded_by"] is None
        assert second in _recalled_ids(beam, "shared")

    def test_retired_rows_do_not_consume_the_fts_budget(self, tmp_path):
        """Retired rows stay in fts_working; they must leave the pool early.

        Without the eligibility subquery the LIMIT fills up with retired rows
        and a pending proposal never reaches the caller, which then cannot
        even fall back to the working-memory scan.
        """
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        retired = [_store_proposal(beam, f"budget{i}") for i in range(30)]
        for proposal_id in retired:
            model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="bulk")
        survivor = _store_proposal(beam, "budget_survivor")

        ids = {row["id"] for row in _fts_search_working(beam.conn, "retirement budget", k=5)}

        assert survivor in ids
        assert not (ids & set(retired))

    def test_retired_rows_do_not_consume_the_cjk_scan_budget(self, tmp_path):
        """The LIKE fallback scans k*5 rows; retired ones must not use that budget."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        retired = [
            _store_proposal(beam, f"cjk{i}", body=f"測試報告 内容 {i}")
            for i in range(30)
        ]
        for proposal_id in retired:
            model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="bulk")
        survivor = _store_proposal(beam, "cjk_survivor", body="測試報告 内容 bewaard")

        ids = {row["id"] for row in _cjk_like_search(beam.conn, "測試報告", k=5, working=True)}

        assert survivor in ids
        assert not (ids & set(retired))

    def test_retired_rows_do_not_consume_the_cyrillic_scan_budget(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        retired = [
            _store_proposal(beam, f"cyr{i}", body=f"проверка памяти номер {i}")
            for i in range(30)
        ]
        for proposal_id in retired:
            model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="bulk")
        survivor = _store_proposal(beam, "cyr_survivor", body="проверка памяти сохранена")

        ids = {row["id"] for row in _cyrillic_like_search(beam.conn, "проверка памяти", k=5, working=True)}

        assert survivor in ids
        assert not (ids & set(retired))

    def test_resolved_proposal_keeps_its_outcome(self, tmp_path):
        """Re-resolving must not rewrite metadata or flip the marker."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "twice")

        model_refresh.apply_model_refresh_proposal(beam, proposal_id)
        result = model_refresh.reject_model_refresh_proposal(
            beam, proposal_id, reason="late objection"
        )

        assert result["status"] == "applied"
        row = _row(beam, proposal_id)
        assert row["superseded_by"] == "model_refresh:applied"
        assert json.loads(row["metadata_json"])["status"] == "applied"

    def test_retry_repairs_a_resolution_left_eligible(self, tmp_path):
        """A row resolved before the fix (or by an interrupted write) heals."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "legacy")

        metadata = json.loads(_row(beam, proposal_id)["metadata_json"])
        metadata["status"] = "rejected"
        beam.conn.execute(
            "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
            (json.dumps(metadata), proposal_id),
        )
        beam.conn.commit()
        assert _row(beam, proposal_id)["superseded_by"] is None

        model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="again")

        assert _row(beam, proposal_id)["superseded_by"] == "model_refresh:rejected"

    def test_pending_proposal_stays_recall_eligible(self, tmp_path):
        """A proposal nobody resolved must not be hidden by this fix."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "pending")

        assert _row(beam, proposal_id)["superseded_by"] is None
        assert proposal_id in _recalled_ids(beam, "pending")
