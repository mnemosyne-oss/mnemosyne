"""Malformed model-refresh confidence degrades instead of raising or outranking.

Confidence reaches this subsystem from two unhardened directions: LLM
JSON (json.loads round-trips NaN and Infinity, so a model can emit them
as literals) and persisted metadata_json on legacy banks (any JSON type,
including strings). Pre-fix:

- parse_model_update_proposals clamped NaN to 1.0 (min and max keep
  their first argument when a NaN comparison is False), so a
  NaN-confidence proposal became a top-importance memory. The auto-apply
  gate's ``confidence < minimum`` check is also False for NaN, so the
  same proposal cleared every threshold and reached the canonical store.
- apply_model_refresh_proposal converted stored confidence with a bare
  float(); a legacy text value raised ValueError. sleep() converts the
  same way at its proposal-remember call site, which runs AFTER the
  claim commit: a raise there strands the group's
  consolidation_claimed_at and orphans every later group's claimed rows.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.canonical import CanonicalStore
from mnemosyne.core import model_refresh


def _proposal(**overrides):
    base = {
        "category": "model:user",
        "name": "communication_style",
        "body": "Prefers concise diagnostic answers.",
        "confidence": 0.95,
        "evidence_ids": ["wm-a", "wm-b"],
        "action": "update",
        "reason": "Repeated durable preference.",
    }
    base.update(overrides)
    return base


def _store_proposal(beam, proposal):
    metadata = model_refresh.prepare_proposal_metadata(
        proposal, source_wm_ids=proposal["evidence_ids"]
    )
    return beam.remember(
        model_refresh.proposal_to_memory_content(proposal),
        source=model_refresh.PROPOSAL_SOURCE,
        metadata=metadata,
    )


class TestParseNonFiniteConfidence:

    def test_nan_confidence_is_rejected_not_clamped_to_one(self):
        raw = (
            '[{"category": "model:user", "name": "style", "body": "b", '
            '"confidence": NaN, "evidence_ids": ["e1", "e2"]}]'
        )
        assert model_refresh.parse_model_update_proposals(
            raw, allowed_categories={"model:user"}
        ) == []

    def test_infinity_confidence_is_rejected(self):
        for literal in ("Infinity", "-Infinity"):
            raw = (
                '[{"category": "model:user", "name": "style", "body": "b", '
                f'"confidence": {literal}, "evidence_ids": ["e1", "e2"]}}]'
            )
            assert model_refresh.parse_model_update_proposals(
                raw, allowed_categories={"model:user"}
            ) == []

    def test_boolean_confidence_is_rejected(self):
        """bool subclasses int, so float(true) would read as full
        confidence; it must degrade like any other non-numeric type."""
        for literal in ("true", "false"):
            raw = (
                '[{"category": "model:user", "name": "style", "body": "b", '
                f'"confidence": {literal}, "evidence_ids": ["e1", "e2"]}}]'
            )
            assert model_refresh.parse_model_update_proposals(
                raw, allowed_categories={"model:user"}
            ) == []

    def test_overrange_integer_confidence_is_rejected(self):
        """json.loads keeps integer literals as arbitrary-precision ints,
        so float() raises OverflowError past float range; it must degrade
        instead of raising."""
        literal = "1" + "0" * 400
        raw = (
            '[{"category": "model:user", "name": "style", "body": "b", '
            f'"confidence": {literal}, "evidence_ids": ["e1", "e2"]}}]'
        )
        assert model_refresh.parse_model_update_proposals(
            raw, allowed_categories={"model:user"}
        ) == []

    def test_valid_confidence_still_parses(self):
        """Control: the non-finite filter must not eat well-formed proposals."""
        raw = (
            '[{"category": "model:user", "name": "style", "body": "b", '
            '"confidence": 0.9, "evidence_ids": ["e1", "e2"]}]'
        )
        parsed = model_refresh.parse_model_update_proposals(
            raw, allowed_categories={"model:user"}
        )
        assert len(parsed) == 1
        assert parsed[0]["confidence"] == pytest.approx(0.9)


class TestStoredConfidenceHardening:

    def test_auto_apply_rejects_nan_stored_confidence(self, tmp_path):
        """NaN in persisted metadata must not bypass the confidence gate."""
        beam = BeamMemory(session_id="hard", db_path=tmp_path / "mnemo.db")
        pid = _store_proposal(beam, _proposal(confidence=float("nan")))

        assert model_refresh.maybe_auto_apply_model_refresh_proposal(
            beam, pid, owner_id="default"
        ) is False

        store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
        assert store.recall("default", "model:user", "communication_style") is None

    def test_auto_apply_rejects_boolean_stored_confidence(self, tmp_path):
        """A persisted ``true`` confidence must not gate through as 1.0."""
        beam = BeamMemory(session_id="hard", db_path=tmp_path / "mnemo.db")
        pid = _store_proposal(beam, _proposal(confidence=True))

        assert model_refresh.maybe_auto_apply_model_refresh_proposal(
            beam, pid, owner_id="default"
        ) is False

        store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
        assert store.recall("default", "model:user", "communication_style") is None

    def test_auto_apply_normalizes_overrange_stored_confidence(self, tmp_path):
        """A persisted finite 2.0 cleared the gate and reached the
        canonical store unbounded; it must apply at the clamped 1.0."""
        beam = BeamMemory(session_id="hard", db_path=tmp_path / "mnemo.db")
        pid = _store_proposal(beam, _proposal(confidence=2.0))

        assert model_refresh.maybe_auto_apply_model_refresh_proposal(
            beam, pid, owner_id="default"
        ) is True

        canonical = CanonicalStore(db_path=beam.db_path, conn=beam.conn).recall(
            "default", "model:user", "communication_style"
        )
        assert canonical is not None
        assert canonical["confidence"] == pytest.approx(1.0)

    def test_auto_apply_rejects_negative_stored_confidence(self, tmp_path):
        """A persisted negative confidence clamps to 0.0 and must fail
        the gate rather than persist below the domain floor."""
        beam = BeamMemory(session_id="hard", db_path=tmp_path / "mnemo.db")
        pid = _store_proposal(beam, _proposal(confidence=-3.0))

        assert model_refresh.maybe_auto_apply_model_refresh_proposal(
            beam, pid, owner_id="default"
        ) is False

        store = CanonicalStore(db_path=beam.db_path, conn=beam.conn)
        assert store.recall("default", "model:user", "communication_style") is None

    def test_apply_survives_legacy_text_confidence(self, tmp_path):
        """A legacy bank's text confidence applies at the 0.5 default
        instead of raising ValueError."""
        beam = BeamMemory(session_id="hard", db_path=tmp_path / "mnemo.db")
        pid = _store_proposal(beam, _proposal(confidence="high"))

        result = model_refresh.apply_model_refresh_proposal(
            beam, pid, owner_id="default"
        )

        assert result["status"] == "applied"
        canonical = CanonicalStore(db_path=beam.db_path, conn=beam.conn).recall(
            "default", "model:user", "communication_style"
        )
        assert canonical is not None
        assert canonical["confidence"] == pytest.approx(0.5)


class TestSleepPostClaimRobustness:

    def test_sleep_survives_malformed_proposal_confidence(self, tmp_path, monkeypatch):
        """A malformed confidence surfacing at the proposal-remember call
        site must not abort sleep() mid-batch: the site runs after the
        claim commit, so a raise strands consolidation_claimed_at on the
        current group and leaves every later group claimed with no
        summary."""
        monkeypatch.setattr(
            "mnemosyne.core.local_llm.llm_available", lambda: False
        )
        db_path = tmp_path / "mnemo.db"
        beam = BeamMemory(session_id="hard", db_path=db_path)

        # Two source groups so the blast radius of a first-group raise
        # (second group orphaned) is part of what the test pins.
        old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
        conn = sqlite3.connect(str(db_path))
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                ("wm-a", "conversation row one", "conversation", old_ts, "hard"),
                ("wm-b", "conversation row two", "conversation", old_ts, "hard"),
                ("wm-c", "manual note row", "manual", old_ts, "hard"),
            ],
        )
        conn.commit()
        conn.close()

        def fake_infer(items):
            return [
                _proposal(evidence_ids=[item["id"] for item in items], confidence="high")
            ]

        monkeypatch.setattr(model_refresh, "infer_model_update_proposals", fake_infer)

        result = beam.sleep(dry_run=False)
        assert result["status"] == "consolidated"

        conn = sqlite3.connect(str(db_path))
        try:
            summaries = conn.execute(
                "SELECT COUNT(*) FROM episodic_memory WHERE source = 'sleep_consolidation'"
            ).fetchone()[0]
            assert summaries == 2, "both source groups must be summarized"

            stuck_claims = conn.execute(
                "SELECT COUNT(*) FROM working_memory "
                "WHERE consolidation_claimed_at IS NOT NULL"
            ).fetchone()[0]
            assert stuck_claims == 0, "no group may be left mid-claim"

            proposal_importances = [
                row[0] for row in conn.execute(
                    "SELECT importance FROM working_memory WHERE source = ?",
                    (model_refresh.PROPOSAL_SOURCE,),
                ).fetchall()
            ]
            assert proposal_importances, "proposal rows should still be written"
            assert all(
                imp == pytest.approx(0.5) for imp in proposal_importances
            ), "malformed confidence falls back to the 0.5 default importance"
        finally:
            conn.close()
