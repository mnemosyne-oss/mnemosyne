"""Stress/regression coverage for sleep-time model refresh.

These tests use synthetic ground-truth memories and stubbed model-refresh output.
They prove the architecture around the LLM is useful and guarded without spending
live model quota: strong durable candidates become canonical facts, weak or
unsafe candidates are rejected, and accepted model slots are injected only when
relevant.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.canonical import CanonicalStore
from mnemosyne.core import model_refresh


def _old_rows(db_path: Path, rows: list[tuple[str, str]]) -> None:
    old_ts = (datetime.now() - timedelta(hours=200)).isoformat()
    conn = sqlite3.connect(db_path)
    for memory_id, content in rows:
        conn.execute(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            (memory_id, content, "conversation", old_ts, "stress"),
        )
    conn.commit()
    conn.close()


def test_sleep_model_refresh_auto_applies_good_and_rejects_bad(tmp_path, monkeypatch):
    db_path = tmp_path / "mnemo.db"
    beam = BeamMemory(session_id="stress", db_path=db_path)
    _old_rows(
        db_path,
        [
            ("wm-pr-1", "Users repeatedly ask for upstreamable PRs with clear review rationale."),
            ("wm-pr-2", "Users want PRs split into reviewable architectural layers."),
            ("wm-style-1", "The operator prefers concise diagnostic answers."),
            ("wm-style-2", "The operator dislikes flattery and wants evidence."),
        ],
    )

    def fake_infer(items):
        return [
            {
                "category": "model:workflow",
                "name": "pr_strategy",
                "body": "Prefer upstreamable PRs split into reviewable architectural layers.",
                "confidence": 0.95,
                "evidence_ids": ["wm-pr-1", "wm-pr-2"],
                "action": "update",
                "reason": "Repeated durable workflow preference.",
            },
            {
                "category": "model:user",
                "name": "thin_evidence",
                "body": "A one-memory claim should not become canonical truth.",
                "confidence": 0.97,
                "evidence_ids": ["wm-style-1"],
                "action": "update",
                "reason": "Insufficient evidence.",
            },
            {
                "category": "model:project",
                "name": "latest_pr",
                "body": "PR #123 is done and should be remembered.",
                "confidence": 0.99,
                "evidence_ids": ["wm-pr-1", "wm-pr-2"],
                "action": "update",
                "reason": "Ephemeral task progress.",
            },
            {
                "category": "model:user",
                "name": "outside_batch",
                "body": "Evidence outside the sleep batch must not validate.",
                "confidence": 0.99,
                "evidence_ids": ["wm-style-1", "missing-id"],
                "action": "update",
                "reason": "Bad citation.",
            },
        ]

    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", fake_infer)
    result = beam.sleep(dry_run=False)

    assert result["status"] == "consolidated"
    assert result["model_refresh"]["proposals"] == 4
    assert result["model_refresh"]["applied"] == 1

    proposals = model_refresh.list_model_refresh_proposals(beam, status="all", limit=20)
    statuses = {p["metadata"]["name"]: p["metadata"]["status"] for p in proposals}
    assert statuses == {
        "pr_strategy": "applied",
        "thin_evidence": "rejected",
        "latest_pr": "rejected",
        "outside_batch": "rejected",
    }

    store = CanonicalStore(db_path=db_path, conn=beam.conn)
    assert store.recall("default", "model:workflow", "pr_strategy")["body"].startswith("Prefer upstreamable")
    assert store.recall("default", "model:user", "thin_evidence") is None
    assert store.recall("default", "model:project", "latest_pr") is None


def test_sleep_model_refresh_auto_apply_uses_beam_owner_namespace(tmp_path, monkeypatch):
    db_path = tmp_path / "mnemo.db"
    beam = BeamMemory(session_id="stress", db_path=db_path)
    beam.canonical_owner_id = "research_profile"
    _old_rows(
        db_path,
        [
            ("wm-owner-1", "The research profile prefers evidence-first notes."),
            ("wm-owner-2", "The research profile asks for citations before claims."),
        ],
    )

    def fake_infer(items):
        return [
            {
                "category": "model:workflow",
                "name": "research_style",
                "body": "Prefer evidence-first notes with citations before claims.",
                "confidence": 0.96,
                "evidence_ids": ["wm-owner-1", "wm-owner-2"],
                "action": "update",
                "reason": "Repeated durable workflow preference.",
            }
        ]

    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", fake_infer)
    result = beam.sleep(dry_run=False)

    assert result["model_refresh"] == {"proposals": 1, "applied": 1}
    store = CanonicalStore(db_path=db_path, conn=beam.conn)
    assert store.recall("default", "model:workflow", "research_style") is None
    research_style = store.recall("research_profile", "model:workflow", "research_style")
    assert research_style is not None
    assert research_style["body"].startswith("Prefer evidence-first")


def test_sleep_model_refresh_skips_cron_context(tmp_path, monkeypatch):
    db_path = tmp_path / "mnemo.db"
    beam = BeamMemory(session_id="stress", db_path=db_path)
    beam.agent_context = "cron"
    beam.canonical_owner_id = "cron_profile"
    _old_rows(
        db_path,
        [
            ("wm-cron-1", "Cron summaries should not mutate canonical model slots."),
            ("wm-cron-2", "Cron runs may execute sleep as maintenance."),
        ],
    )

    def fake_infer(items):
        raise AssertionError("cron sleep must not run model-refresh inference")

    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", fake_infer)
    result = beam.sleep(dry_run=False)

    assert result["status"] == "consolidated"
    assert result["model_refresh"] == {"proposals": 0, "applied": 0}
    store = CanonicalStore(db_path=db_path, conn=beam.conn)
    assert store.list("cron_profile", category="model:workflow") == []


def test_sleep_model_refresh_conflict_requires_stronger_evidence(tmp_path):
    db_path = tmp_path / "mnemo.db"
    beam = BeamMemory(session_id="stress", db_path=db_path)
    store = CanonicalStore(db_path=db_path, conn=beam.conn)
    store.remember("default", "model:user", "communication_style", "Prefers verbose narrative answers.")

    source_ids = ["wm-a", "wm-b", "wm-c"]
    weak = {
        "category": "model:user",
        "name": "communication_style",
        "body": "Prefers concise diagnostic answers.",
        "confidence": 0.94,
        "evidence_ids": source_ids[:2],
        "action": "update",
        "reason": "Weak conflict.",
    }
    strong = dict(weak, confidence=0.99, evidence_ids=source_ids, reason="Strong conflict supersession.")

    weak_meta = model_refresh.prepare_proposal_metadata(weak, source_wm_ids=source_ids)
    weak_id = beam.remember(model_refresh.proposal_to_memory_content(weak), source=model_refresh.PROPOSAL_SOURCE, metadata=weak_meta)
    assert model_refresh.maybe_auto_apply_model_refresh_proposal(beam, weak_id, owner_id="default") is False
    assert store.recall("default", "model:user", "communication_style")["body"] == "Prefers verbose narrative answers."

    strong_meta = model_refresh.prepare_proposal_metadata(strong, source_wm_ids=source_ids)
    strong_id = beam.remember(model_refresh.proposal_to_memory_content(strong), source=model_refresh.PROPOSAL_SOURCE, metadata=strong_meta)
    assert model_refresh.maybe_auto_apply_model_refresh_proposal(beam, strong_id, owner_id="default") is True
    assert store.recall("default", "model:user", "communication_style")["body"] == "Prefers concise diagnostic answers."


def test_prefetch_injects_only_relevant_model_slots(tmp_path, monkeypatch):
    from hermes_memory_provider import MnemosyneMemoryProvider

    monkeypatch.delenv("MNEMOSYNE_PREFETCH_MODEL_SLOT_MIN_OVERLAP", raising=False)
    provider = MnemosyneMemoryProvider()
    provider._beam = BeamMemory(session_id="stress", db_path=tmp_path / "mnemo.db")
    provider._agent_identity = "default"
    store = CanonicalStore(db_path=provider._beam.db_path, conn=provider._beam.conn)
    store.remember("default", "model:workflow", "pr_strategy", "Prefer upstreamable PRs split into reviewable architectural layers.")
    store.remember("default", "model:user", "communication_style", "Prefers concise diagnostic answers without flattery.")
    store.remember("default", "model:project", "music_notes", "Ambient playlist tags include drone and texture.")

    pr_block = provider.prefetch("how should we structure this PR strategy")
    assert "## Mnemosyne Model Context" in pr_block
    assert "pr strategy" in pr_block
    assert "upstreamable PRs" in pr_block
    assert "Ambient playlist" not in pr_block

    shell_block = provider.prefetch("what command checks disk usage")
    assert "## Mnemosyne Model Context" not in shell_block

    style_block = provider.prefetch("what communication style should you use")
    assert "communication style" in style_block
    assert "concise diagnostic" in style_block


def test_model_refresh_tool_is_diagnostic_only(tmp_path):
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    provider._beam = BeamMemory(session_id="stress", db_path=tmp_path / "mnemo.db")
    schemas = {schema["name"]: schema for schema in provider.get_tool_schemas()}
    assert schemas["mnemosyne_model_refresh"]["parameters"]["properties"]["action"]["enum"] == ["list"]

    response = json.loads(provider.handle_tool_call("mnemosyne_model_refresh", {"action": "apply", "proposal_id": "x"}))
    assert "diagnostic-only" in response["error"]


@pytest.mark.parametrize("auto_apply", [True, False])
def test_sleep_releases_writer_before_remembering_next_proposal(tmp_path, monkeypatch, auto_apply):
    db_path = tmp_path / "proposals.db"
    beam = BeamMemory(session_id="stress", db_path=db_path)
    _old_rows(db_path, [("evidence-a", "Durable workflow."), ("evidence-b", "Durable workflow twice.")])
    proposals = [
        {
            "category": "model:workflow", "name": name, "body": f"Durable workflow for {name}.",
            "confidence": 0.99, "evidence_ids": ["evidence-a", "evidence-b"],
            "action": "update", "reason": "Repeated preference.",
        }
        for name in ("planning", "reporting")
    ]
    monkeypatch.setattr(model_refresh, "infer_model_update_proposals", lambda items: proposals)
    monkeypatch.setattr(model_refresh, "auto_apply_enabled", lambda: auto_apply)

    real_remember = beam.remember
    writer_available = []

    def remember(content, **kwargs):
        if kwargs.get("source") == model_refresh.PROPOSAL_SOURCE:
            with sqlite3.connect(db_path, timeout=0) as observer:
                try:
                    observer.execute("BEGIN IMMEDIATE")
                except sqlite3.OperationalError as exc:
                    assert "locked" in str(exc)
                    writer_available.append(False)
                else:
                    observer.rollback()
                    writer_available.append(True)
        return real_remember(content, **kwargs)

    monkeypatch.setattr(beam, "remember", remember)
    result = beam.sleep()
    assert result["model_refresh"]["proposals"] == 2
    assert result["model_refresh"]["applied"] == (2 if auto_apply else 0)
    assert writer_available == [True, True]
