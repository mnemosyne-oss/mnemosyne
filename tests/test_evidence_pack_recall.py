"""Read-only candidate-recall contract for future evidence packs."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.config import MnemosyneConfig
from mnemosyne.core.recall_diagnostics import get_recall_diagnostics, reset_recall_diagnostics


@pytest.fixture(autouse=True)
def _reset_config_singleton_after_test():
    MnemosyneConfig.reset_instance()
    yield
    MnemosyneConfig.reset_instance()


def _state(db_path: Path, memory_id: str) -> tuple[int, str | None]:
    with sqlite3.connect(str(db_path)) as conn:
        count, recalled_at = conn.execute(
            "SELECT recall_count, last_recalled FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()
    return (count or 0, recalled_at)




@pytest.mark.parametrize("internal_control", ("_track_recall", "_cross_session"))
def test_evidence_pack_rejects_invalid_internal_controls(tmp_path: Path, monkeypatch, internal_control: str):
    beam = BeamMemory(session_id="guards", db_path=tmp_path / "guards.db")
    calls = []

    def recording_recall(*args, **kwargs):
        calls.append((args, kwargs))
        return []

    monkeypatch.setattr(beam, "recall", recording_recall)
    with pytest.raises(ValueError, match="candidate_k"):
        beam.recall_with_evidence_pack("q", top_k=2, candidate_k=1)
    with pytest.raises(ValueError, match="candidate_k must exceed"):
        beam.recall_with_evidence_pack("q", top_k=2, candidate_k=2, pack_k=1)
    with pytest.raises(ValueError, match="pack_k"):
        beam.recall_with_evidence_pack("q", pack_k=-1)
    with pytest.raises(ValueError, match="explain"):
        beam.recall_with_evidence_pack("q", explain=True)
    with pytest.raises(ValueError, match="internal recall controls"):
        beam.recall_with_evidence_pack("q", **{internal_control: False})
    assert calls == []


def _enable_cross_session_scope(monkeypatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text("cross_session: true\n")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(config_dir))
    MnemosyneConfig.reset_instance()


def test_candidate_only_recall_does_not_mutate_usage_state(tmp_path: Path):
    db_path = tmp_path / "evidence-pack.db"
    beam = BeamMemory(session_id="evidence-pack", db_path=db_path)
    memory_id = beam.remember("Orion evidence-pack candidate fixture", source="test", importance=0.8)

    before = _state(db_path, memory_id)
    results = beam.recall("Orion evidence-pack", top_k=5, _track_recall=False)
    after_candidate = _state(db_path, memory_id)
    assert memory_id in [row["id"] for row in results]
    assert after_candidate == before

    beam.recall("Orion evidence-pack", top_k=5)
    after_normal = _state(db_path, memory_id)
    assert after_normal[0] == before[0] + 1
    assert after_normal[1] is not None




def test_candidate_only_recall_does_not_mutate_episodic_usage_state(tmp_path: Path):
    db_path = tmp_path / "episodic-telemetry.db"
    beam = BeamMemory(session_id="episodic", db_path=db_path)
    memory_id = "episodic-telemetry-id"
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (memory_id, "episodic telemetry unique phrase", "test", "2026-01-01T00:00:00", "episodic", 0.8, "session"),
    )
    beam.conn.commit()
    before = beam.conn.execute("SELECT recall_count, last_recalled FROM episodic_memory WHERE id = ?", (memory_id,)).fetchone()
    results = beam.recall("episodic telemetry unique phrase", top_k=5, _track_recall=False)
    after = beam.conn.execute("SELECT recall_count, last_recalled FROM episodic_memory WHERE id = ?", (memory_id,)).fetchone()
    assert memory_id in [row["id"] for row in results]
    assert after == before




def test_real_recall_uses_working_and_episodic_tier_labels(tmp_path: Path):
    db_path = tmp_path / "tier-labels.db"
    beam = BeamMemory(session_id="tiers", db_path=db_path)
    working_id = beam.remember("working tier unique phrase", source="test", importance=0.8)
    episodic_id = "episodic-tier-id"
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (episodic_id, "episodic tier unique phrase", "test", "2026-01-01T00:00:00", "tiers", 0.8, "session"),
    )
    beam.conn.commit()

    working = beam.recall("working tier unique phrase", top_k=5, _track_recall=False)
    episodic = beam.recall("episodic tier unique phrase", top_k=5, _track_recall=False)
    assert next(row for row in working if row["id"] == working_id)["tier"] == "working"
    assert next(row for row in episodic if row["id"] == episodic_id)["tier"] == "episodic"


def test_evidence_pack_keeps_primary_ranking_and_bounds_supplement(tmp_path: Path):
    db_path = tmp_path / "pack.db"
    beam = BeamMemory(session_id="evidence-pack", db_path=db_path)
    for index in range(4):
        beam.remember(f"Orion evidence fixture {index}", source="test", importance=0.8)

    primary = beam.recall("Orion evidence", top_k=2)
    packed = beam.recall_with_evidence_pack("Orion evidence", top_k=2, candidate_k=4, pack_k=1)

    assert [row["id"] for row in packed["primary"]] == [row["id"] for row in primary]
    # Primary already claims this only session, so no supplemental group remains.
    assert packed["evidence_pack"] == []
    assert {row["id"] for row in packed["primary"]}.isdisjoint(
        {row["id"] for row in packed["evidence_pack"]}
    )


def test_evidence_pack_does_not_leak_other_session_rows(tmp_path: Path):
    db_path = tmp_path / "scope.db"
    local = BeamMemory(session_id="local", db_path=db_path)
    other = BeamMemory(session_id="other", db_path=db_path)
    local_id = local.remember("Orion local scope fixture", source="test", importance=0.8)
    other_id = other.remember("Orion other scope fixture", source="test", importance=0.8)

    packed = local.recall_with_evidence_pack("Orion scope", top_k=1, candidate_k=5, pack_k=5)
    returned_ids = {row["id"] for row in packed["primary"] + packed["evidence_pack"]}

    assert local_id in returned_ids
    assert other_id not in returned_ids


def test_evidence_pack_zero_skips_candidate_recall(tmp_path: Path, monkeypatch):
    beam = BeamMemory(session_id="zero-pack", db_path=tmp_path / "zero-pack.db")
    memory_id = beam.remember("Orion zero pack fixture", source="test", importance=0.8)
    real_recall = beam.recall
    calls = []

    def recording_recall(*args, **kwargs):
        calls.append(kwargs)
        return real_recall(*args, **kwargs)

    monkeypatch.setattr(beam, "recall", recording_recall)
    packed = beam.recall_with_evidence_pack(
        "Orion zero pack", top_k=1, candidate_k=5, pack_k=0
    )

    assert len(calls) == 1
    assert calls[0]["top_k"] == 1
    assert "_track_recall" not in calls[0]
    assert [row["id"] for row in packed["primary"]] == [memory_id]
    assert packed["evidence_pack"] == []


def test_zero_pack_allows_equal_candidate_and_top_k(tmp_path: Path):
    beam = BeamMemory(session_id="equal-k", db_path=tmp_path / "equal-k.db")
    memory_id = beam.remember("Orion equal k fixture", source="test", importance=0.8)

    packed = beam.recall_with_evidence_pack(
        "Orion equal k", top_k=2, candidate_k=2, pack_k=0
    )

    assert [row["id"] for row in packed["primary"]] == [memory_id]
    assert packed["evidence_pack"] == []


def test_cross_session_scope_can_produce_a_bounded_supplemental_pack(tmp_path: Path, monkeypatch):
    _enable_cross_session_scope(monkeypatch, tmp_path)
    db_path = tmp_path / "cross-session-pack.db"
    reader = BeamMemory(session_id="reader", db_path=db_path)
    first = BeamMemory(session_id="first", db_path=db_path)
    second = BeamMemory(session_id="second", db_path=db_path)
    third = BeamMemory(session_id="third", db_path=db_path)
    first.remember("Orion cross session alpha", source="test", importance=0.8, scope="global")
    second_id = second.remember("Orion cross session beta", source="test", importance=0.8, scope="global")
    third_id = third.remember("Orion cross session gamma", source="test", importance=0.8, scope="global")

    plain_primary = reader.recall("Orion cross session", top_k=1)
    packed = reader.recall_with_evidence_pack("Orion cross session", top_k=1, candidate_k=5, pack_k=10)

    assert [row["id"] for row in packed["primary"]] == [row["id"] for row in plain_primary]
    # Only the two non-primary global rows are eligible, even when pack_k is larger.
    assert len(packed["evidence_pack"]) == 2
    assert {row["evidence_rank"] for row in packed["evidence_pack"]} == {2, 3}
    returned_ids = {row["id"] for row in packed["primary"] + packed["evidence_pack"]}
    assert {second_id, third_id}.issubset(returned_ids)


def test_episodic_candidate_enters_pack_with_backfilled_session(tmp_path: Path, monkeypatch):
    _enable_cross_session_scope(monkeypatch, tmp_path)
    db_path = tmp_path / "episodic-pack.db"
    reader = BeamMemory(session_id="reader", db_path=db_path)
    primary = BeamMemory(session_id="primary", db_path=db_path)
    primary.remember("Orion episodic pack primary", source="test", importance=0.9, scope="global")
    episodic_id = "orion-episodic-pack-candidate"
    reader.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            episodic_id, "Orion episodic pack candidate", "test",
            "2026-01-01T00:00:00", "episodic-writer", 0.8, "global",
        ),
    )
    reader.conn.commit()

    packed = reader.recall_with_evidence_pack("Orion episodic pack", top_k=1, candidate_k=5, pack_k=5)
    pack_row = next(row for row in packed["evidence_pack"] if row["id"] == episodic_id)

    assert pack_row["tier"] == "episodic"
    assert pack_row["session_id"] == "episodic-writer"


def test_evidence_pack_honors_temporal_filters(tmp_path: Path, monkeypatch):
    _enable_cross_session_scope(monkeypatch, tmp_path)
    db_path = tmp_path / "temporal-pack.db"
    reader = BeamMemory(session_id="reader", db_path=db_path)
    old_writer = BeamMemory(session_id="old", db_path=db_path)
    current_writer = BeamMemory(session_id="current", db_path=db_path)
    old_id = old_writer.remember("Orion temporal old fixture", source="test", importance=0.8, scope="global")
    current_id = current_writer.remember(
        "Orion temporal current fixture", source="test", importance=0.8, scope="global"
    )
    episodic_current_id = "orion-temporal-episodic-current"
    episodic_future_id = "orion-temporal-episodic-future"
    reader.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            episodic_current_id, "Orion temporal episodic current fixture", "test",
            "2026-02-01T00:00:00", "episodic-current", 0.8, "global",
        ),
    )
    reader.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            episodic_future_id, "Orion temporal episodic future fixture", "test",
            "2027-01-01T00:00:00", "episodic-future", 0.8, "global",
        ),
    )
    reader.conn.execute(
        "UPDATE working_memory SET timestamp = ? WHERE id = ?", ("2020-01-01T00:00:00", old_id)
    )
    reader.conn.execute(
        "UPDATE working_memory SET timestamp = ? WHERE id = ?", ("2026-01-01T00:00:00", current_id)
    )
    reader.conn.commit()

    packed = reader.recall_with_evidence_pack(
        "Orion temporal", top_k=1, candidate_k=10, pack_k=5,
        from_date="2025-01-01", to_date="2026-06-01",
    )
    returned_ids = {row["id"] for row in packed["primary"] + packed["evidence_pack"]}

    assert current_id in returned_ids
    assert episodic_current_id in returned_ids
    assert old_id not in returned_ids
    assert episodic_future_id not in returned_ids


def test_evidence_pack_excludes_consolidated_working_candidates(tmp_path: Path, monkeypatch):
    _enable_cross_session_scope(monkeypatch, tmp_path)
    db_path = tmp_path / "consolidated-evidence.db"
    reader = BeamMemory(session_id="reader", db_path=db_path)
    primary = BeamMemory(session_id="primary", db_path=db_path)
    consolidated = BeamMemory(session_id="consolidated", db_path=db_path)
    hot = BeamMemory(session_id="hot", db_path=db_path)
    primary.remember("Orion consolidated evidence primary", source="test", importance=0.9, scope="global")
    consolidated_id = consolidated.remember(
        "Orion consolidated evidence historical", source="test", importance=0.8, scope="global"
    )
    hot_id = hot.remember("Orion consolidated evidence hot", source="test", importance=0.7, scope="global")
    reader.conn.execute(
        "UPDATE working_memory SET consolidated_at = ? WHERE id = ?",
        ("2026-01-01T00:00:00", consolidated_id),
    )
    reader.conn.commit()

    packed = reader.recall_with_evidence_pack(
        "Orion consolidated evidence", top_k=1, candidate_k=5, pack_k=5
    )
    evidence_ids = {row["id"] for row in packed["evidence_pack"]}
    assert consolidated_id not in evidence_ids
    assert hot_id in evidence_ids


def test_candidate_only_linear_recall_does_not_mutate_diagnostics(tmp_path: Path):
    db_path = tmp_path / "diagnostics.db"
    beam = BeamMemory(session_id="diagnostics", db_path=db_path)
    beam.remember("Orion diagnostics candidate fixture", source="test", importance=0.8)
    reset_recall_diagnostics()

    before = get_recall_diagnostics()
    beam.recall("Orion diagnostics", top_k=5, _track_recall=False)
    after = get_recall_diagnostics()
    assert after["totals"] == before["totals"]
    assert after["by_tier"] == before["by_tier"]


def test_evidence_pack_linear_pass_records_one_diagnostic_call(tmp_path: Path):
    db_path = tmp_path / "linear-diagnostics.db"
    beam = BeamMemory(session_id="linear", db_path=db_path)
    beam.remember("Orion linear diagnostics fixture", source="test", importance=0.8)
    reset_recall_diagnostics()

    beam.recall_with_evidence_pack(
        "Orion linear diagnostics", top_k=1, candidate_k=5, pack_k=2
    )

    diagnostics = get_recall_diagnostics()
    assert diagnostics["totals"]["calls"] == 1


def test_evidence_pack_polyphonic_candidate_only_pass_is_telemetry_neutral(tmp_path: Path, monkeypatch):
    from mnemosyne.core.polyphonic_recall import PolyphonicResult

    class FakeEngine:
        def __init__(self, results):
            self.results = results

        def recall(self, *, query, query_embedding, top_k, **kwargs):
            return self.results

    db_path = tmp_path / "polyphonic-telemetry.db"
    beam = BeamMemory(session_id="polyphonic", db_path=db_path)
    memory_id = beam.remember("Orion polyphonic candidate fixture", source="test", importance=0.8)
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")
    monkeypatch.setattr(
        beam,
        "_get_polyphonic_engine",
        lambda: FakeEngine([PolyphonicResult(memory_id=memory_id, combined_score=0.9, voice_scores={}, metadata={})]),
    )

    reset_recall_diagnostics()
    before = _state(db_path, memory_id)
    beam.recall_with_evidence_pack("Orion polyphonic", top_k=1, candidate_k=2, pack_k=1)
    after = _state(db_path, memory_id)
    diagnostics = get_recall_diagnostics()
    # Only the public primary call mutates either usage state or diagnostics;
    # the wider candidate pass is telemetry-neutral under the polyphonic path.
    assert after[0] == before[0] + 1
    assert after[1] is not None
    assert diagnostics["totals"]["calls"] == 1
    assert all(tier["total_hits"] == 0 for tier in diagnostics["by_tier"].values())


def test_provenance_backfill_keeps_same_ids_separate_by_tier(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "collision.db"
    beam = BeamMemory(session_id="reader", db_path=db_path)
    shared_id = "same-id"
    now = "2026-01-01T00:00:00"
    beam.conn.execute(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (shared_id, "working", "test", now, "primary-session", 0.5, "global"),
    )
    beam.conn.execute(
        "INSERT INTO episodic_memory (id, content, source, timestamp, session_id, importance, scope) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (shared_id, "episodic", "test", now, "candidate-session", 0.5, "global"),
    )
    beam.conn.commit()
    calls = iter([
        [{"id": shared_id, "tier": "working", "timestamp": now}],
        [{"id": shared_id, "tier": "episodic", "timestamp": now}],
    ])
    monkeypatch.setattr(beam, "recall", lambda *args, **kwargs: next(calls))

    packed = beam.recall_with_evidence_pack("q", top_k=1, candidate_k=2, pack_k=1)
    assert packed["primary"][0]["session_id"] == "primary-session"
    assert packed["evidence_pack"][0]["session_id"] == "candidate-session"


def test_evidence_pack_excludes_synthetic_candidate_tiers(tmp_path: Path, monkeypatch):
    beam = BeamMemory(session_id="reader", db_path=tmp_path / "synthetic-tier.db")
    calls = iter([
        [],
        [{"id": "memoria_source_test", "tier": "memoria", "session_id": "reader"}],
    ])
    monkeypatch.setattr(beam, "recall", lambda *args, **kwargs: next(calls))

    packed = beam.recall_with_evidence_pack("q", top_k=1, candidate_k=2, pack_k=1)

    assert packed == {"primary": [], "evidence_pack": []}
