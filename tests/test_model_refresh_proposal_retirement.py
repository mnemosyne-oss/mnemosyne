"""Resolved model-refresh proposals must stop being recall candidates.

Sleep stores proposals in working memory for audit. Resolved proposals
retain their metadata status and any real supersession edges. Pending rows
must remain eligible even when resolved rows would fill the candidate pool.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from mnemosyne.core.beam import (
    BeamMemory,
    _cjk_like_search,
    _cyrillic_like_search,
    _fts_precise_terms,
    _fts_query_terms,
    _fts_search_working,
    _fts_staged_rows,
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

        assert _row(beam, proposal_id)["superseded_by"] is None
        assert proposal_id not in _recalled_ids(beam, "applied")

    def test_rejected_proposal_is_retired_from_recall(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "rejected")

        assert proposal_id in _recalled_ids(beam, "rejected")

        model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="stale")

        assert _row(beam, proposal_id)["superseded_by"] is None
        assert proposal_id not in _recalled_ids(beam, "rejected")

    def test_resolving_proposals_preserves_a5_typed_edges(self, tmp_path):
        db_path = tmp_path / "mnemo.db"
        beam = BeamMemory(session_id="retire", db_path=db_path)
        historical = beam.remember("Historical control")
        replacement = beam.remember("Replacement control")
        beam.conn.execute(
            "UPDATE working_memory SET superseded_by = ? WHERE id = ?",
            (replacement, historical),
        )
        beam.conn.commit()

        applied = _store_proposal(beam, "a5_applied")
        rejected = _store_proposal(beam, "a5_rejected")
        beam.conn.execute(
            "UPDATE working_memory SET superseded_by = ? WHERE id = ?",
            (replacement, rejected),
        )
        beam.conn.commit()
        model_refresh.apply_model_refresh_proposal(beam, applied)
        model_refresh.reject_model_refresh_proposal(beam, rejected)
        assert _row(beam, rejected)["superseded_by"] == replacement

        gate = Path(__file__).with_name("canon_sign_head_test.py")
        report = subprocess.run(
            [sys.executable, str(gate), str(db_path)], capture_output=True, text=True,
        )
        a5 = next(line for line in report.stdout.splitlines() if line.startswith("ARM A5 "))
        assert a5.startswith("ARM A5 GREEN :: 2 typed edges checked"), a5

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

        assert _row(beam, first)["superseded_by"] is None
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

    def test_retired_rows_do_not_consume_the_hangul_fts_budget(self, tmp_path):
        """Hangul: eligibility must hold in the exact, precise and expanded stages.

        A Korean query yields three different term lists (literal phrase,
        widened phrase, shortened prefix) and each stage has its own LIMIT.
        A filter that only covers one stage leaves the other two open, which
        the Latin budget test cannot see because it never reaches those
        stages.
        """
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        retired = [
            _store_proposal(beam, f"hangul{i}", body=f"시스템의 포트 설정 {i}번 항목")
            for i in range(30)
        ]
        for proposal_id in retired:
            model_refresh.reject_model_refresh_proposal(beam, proposal_id, reason="bulk")
        survivor = _store_proposal(beam, "hangul_survivor", body="시스템의 포트 설정 보관")

        query = "시스템의"
        exact = _fts_precise_terms(query, widen=False)
        precise = _fts_precise_terms(query)
        expanded = _fts_query_terms(query)
        assert len({tuple(exact), tuple(precise), tuple(expanded)}) == 3

        ids = {row["id"] for row in _fts_search_working(beam.conn, query, k=5)}

        assert survivor in ids
        assert not (ids & set(retired))

        # Negative control: without the predicate the same query fills the
        # pool with retired rows, so the assertions above are not vacuous.
        unfiltered = {
            row["id"]
            for row in _fts_staged_rows(
                beam.conn, "fts_working", "id", query, 10, expanded
            )
        }
        assert unfiltered & set(retired)

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
        assert row["superseded_by"] is None
        assert json.loads(row["metadata_json"])["status"] == "applied"

    def test_preexisting_resolution_is_retired_without_retry(self, tmp_path):
        """Banks resolved before this fix already carry the retirement status."""
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

        assert proposal_id not in _recalled_ids(beam, "legacy")
        assert _row(beam, proposal_id)["superseded_by"] is None

    def test_temporal_voice_excludes_retired_rows_before_limit(self, tmp_path):
        pytest.importorskip("numpy")
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        pending = _store_proposal(beam, "temporal_pending")
        retired = _store_proposal(beam, "temporal_retired")
        beam.conn.execute(
            "UPDATE working_memory SET timestamp = datetime('now', '-1 day') WHERE id = ?",
            (pending,),
        )
        beam.conn.commit()
        engine = PolyphonicRecallEngine(db_path=beam.db_path, conn=beam.conn)

        def first_result():
            return [result.memory_id for result in engine._temporal_voice("what happened recently", limit=1)]

        assert first_result() == [retired]
        model_refresh.reject_model_refresh_proposal(beam, retired)
        assert first_result() == [pending]

    def test_pending_proposal_stays_recall_eligible(self, tmp_path):
        """A proposal nobody resolved must not be hidden by this fix."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "pending")

        assert _row(beam, proposal_id)["superseded_by"] is None
        assert proposal_id in _recalled_ids(beam, "pending")


class TestResolutionClaim:
    """A resolution is a guarded transition, not a read-then-write.

    The canonical write commits on its own connection transaction, so the
    proposal is claimed first (a conditional UPDATE) and the terminal write
    carries the claim token. Two resolvers can therefore never leave the
    proposal status and the canonical store disagreeing.
    """

    def test_second_claim_is_refused(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "claim")
        metadata = json.loads(_row(beam, proposal_id)["metadata_json"])

        first = model_refresh._claim_pending_proposal(beam, proposal_id, metadata)
        assert first, "the first claim must win"
        second = model_refresh._claim_pending_proposal(beam, proposal_id, metadata)
        assert second is None, "a claimed proposal must not be claimable twice"

    def test_stale_claim_can_be_taken_over_after_the_lease(self, tmp_path):
        """A crashed resolver must not strand the proposal forever."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "lease")
        metadata = json.loads(_row(beam, proposal_id)["metadata_json"])
        assert model_refresh._claim_pending_proposal(beam, proposal_id, metadata)
        assert model_refresh._claim_pending_proposal(beam, proposal_id, metadata) is None

        record = json.loads(_row(beam, proposal_id)["metadata_json"])
        record[model_refresh.RESOLUTION_CLAIMED_AT_KEY] = "2020-01-01T00:00:00+00:00"
        beam.conn.execute(
            "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
            (json.dumps(record), proposal_id),
        )
        beam.conn.commit()

        assert model_refresh._claim_pending_proposal(
            beam, proposal_id, metadata
        ), "a claim older than the lease must be reclaimable"

    def test_terminal_write_requires_the_claim(self, tmp_path):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "token")
        metadata = json.loads(_row(beam, proposal_id)["metadata_json"])
        model_refresh._claim_pending_proposal(beam, proposal_id, metadata)

        applied = dict(metadata, status="applied")
        assert model_refresh._save_metadata(
            beam, proposal_id, applied, token="not-the-claim"
        ) is False
        assert json.loads(_row(beam, proposal_id)["metadata_json"])["status"] == "pending"

    def test_apply_reports_a_concurrent_rejection_instead_of_overwriting_it(
        self, tmp_path, monkeypatch
    ):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "race")
        real_claim = model_refresh._claim_pending_proposal

        def racing_claim(beam_, proposal_id_, metadata_):
            # Simulate another connection that commits its outcome between our
            # read and our claim.
            other = dict(metadata_, status="rejected", rejected_by="other")
            model_refresh._save_metadata(beam_, proposal_id_, other)
            return real_claim(beam_, proposal_id_, metadata_)

        monkeypatch.setattr(model_refresh, "_claim_pending_proposal", racing_claim)
        assert model_refresh.auto_apply_enabled()  # normal product path, not config-off
        result = model_refresh.apply_model_refresh_proposal(beam, proposal_id)

        assert result["status"] == "rejected"
        record = json.loads(_row(beam, proposal_id)["metadata_json"])
        assert record["status"] == "rejected"
        assert record["rejected_by"] == "other"
        canonical = beam.conn.execute("SELECT COUNT(*) FROM canonical_facts").fetchone()[0]
        assert canonical == 0, "the losing resolver must not write canonical truth"

    def test_failed_canonical_write_releases_the_claim(self, tmp_path, monkeypatch):
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, "release")
        metadata = json.loads(_row(beam, proposal_id)["metadata_json"])

        from mnemosyne.core.canonical import CanonicalStore

        def boom(*args, **kwargs):
            raise RuntimeError("canonical store unavailable")

        monkeypatch.setattr(CanonicalStore, "remember", boom)
        with pytest.raises(RuntimeError):
            model_refresh.apply_model_refresh_proposal(beam, proposal_id)

        record = json.loads(_row(beam, proposal_id)["metadata_json"])
        assert record["status"] == "pending"
        assert model_refresh.RESOLUTION_TOKEN_KEY not in record
        assert model_refresh._claim_pending_proposal(beam, proposal_id, metadata)


class TestEligibilityMatrixAcrossPaths:
    """The retirement predicate must hold on every retrieval path, not just FTS."""

    @pytest.mark.parametrize("state", ["pending", "applied", "rejected"])
    def test_context_retrieval_matches_the_matrix(self, tmp_path, monkeypatch, state):
        monkeypatch.setenv("MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED", "1")
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, f"ctx_{state}")
        if state != "pending":
            resolver = (
                model_refresh.apply_model_refresh_proposal
                if state == "applied"
                else model_refresh.reject_model_refresh_proposal
            )
            resolver(beam, proposal_id)

        ids = {row["id"] for row in beam.get_context(limit=50)}
        assert (proposal_id in ids) is (state == "pending")

    @pytest.mark.parametrize("state", ["pending", "applied", "rejected"])
    def test_standalone_connection_uses_the_same_predicate(self, tmp_path, state):
        import sqlite3

        from mnemosyne.core.beam import _working_recall_eligibility_sql

        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, f"raw_{state}")
        if state != "pending":
            resolver = (
                model_refresh.apply_model_refresh_proposal
                if state == "applied"
                else model_refresh.reject_model_refresh_proposal
            )
            resolver(beam, proposal_id)

        # A connection outside BeamMemory must see the same eligibility rule.
        raw = sqlite3.connect(beam.db_path)
        try:
            sql = _working_recall_eligibility_sql(raw)
            rows = {
                row[0]
                for row in raw.execute(
                    f"SELECT id FROM working_memory WHERE {sql}"
                ).fetchall()
            }
        finally:
            raw.close()
        assert (proposal_id in rows) is (state == "pending")

    @pytest.mark.parametrize("state", ["pending", "applied", "rejected"])
    def test_dense_working_search_matches_the_matrix(self, tmp_path, state):
        np = pytest.importorskip("numpy")
        from mnemosyne.core.beam import _wm_vec_search

        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        proposal_id = _store_proposal(beam, f"vec_{state}")
        if state != "pending":
            resolver = (
                model_refresh.apply_model_refresh_proposal
                if state == "applied"
                else model_refresh.reject_model_refresh_proposal
            )
            resolver(beam, proposal_id)
        beam.conn.execute(
            "INSERT OR REPLACE INTO memory_embeddings (memory_id, embedding_json) "
            "VALUES (?, ?)",
            (proposal_id, json.dumps([1.0, 0.0, 0.0])),
        )
        beam.conn.commit()

        hits = _wm_vec_search(
            beam.conn, np.array([1.0, 0.0, 0.0], dtype=np.float32), k=5
        )
        assert (proposal_id in {hit["id"] for hit in hits}) is (state == "pending")

    def test_ordinary_and_malformed_metadata_stay_eligible(self, tmp_path):
        """Controls: the CASE on source may not swallow non-proposal rows."""
        beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
        ordinary = beam.remember(
            "Ordinary memory about retirement_controls reporting preference.",
            metadata={"status": "applied"},
        )
        malformed = beam.remember(
            "Malformed proposal metadata control, retirement_controls reporting preference."
        )
        beam.conn.execute(
            "UPDATE working_memory SET source = ?, metadata_json = ? WHERE id = ?",
            (model_refresh.PROPOSAL_SOURCE, "{not json", malformed),
        )
        beam.conn.commit()

        recalled = _recalled_ids(beam, "retirement_controls")
        assert ordinary in recalled
        assert malformed in recalled
