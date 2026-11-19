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


def test_competing_connection_cannot_resolve_during_apply(tmp_path, monkeypatch):
    import sqlite3

    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "race")
    real_read = model_refresh._proposal_row
    lock_observed = []

    def racing_read(beam_, proposal_id_):
        row = real_read(beam_, proposal_id_)
        with sqlite3.connect(beam.db_path, timeout=0) as competitor:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                competitor.execute(
                    "UPDATE working_memory SET metadata_json = ? WHERE id = ?",
                    ('{"status":"rejected"}', proposal_id_),
                )
        lock_observed.append(True)
        return row

    with monkeypatch.context() as race:
        race.setattr(model_refresh, "_proposal_row", racing_read)
        result = model_refresh.apply_model_refresh_proposal(beam, proposal_id)

    assert lock_observed == [True]
    assert result["status"] == "applied"
    assert model_refresh.reject_model_refresh_proposal(beam, proposal_id)["status"] == "applied"
    assert beam.conn.execute("SELECT count(*) FROM canonical_facts").fetchone()[0] == 1


def test_failed_canonical_write_leaves_proposal_retryable(tmp_path, monkeypatch):
    from mnemosyne.core.canonical import CanonicalStore

    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "retry")

    def fail(*args, **kwargs):
        raise RuntimeError("canonical store unavailable")

    with monkeypatch.context() as unavailable:
        unavailable.setattr(CanonicalStore, "remember", fail)
        with pytest.raises(RuntimeError, match="canonical store unavailable"):
            model_refresh.apply_model_refresh_proposal(beam, proposal_id)

    assert json.loads(_row(beam, proposal_id)["metadata_json"])["status"] == "pending"
    assert model_refresh.apply_model_refresh_proposal(beam, proposal_id)["status"] == "applied"


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


def test_failed_terminal_write_rolls_back_canonical_value(tmp_path):
    import sqlite3

    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "atomic_failure")
    beam.conn.execute("""
        CREATE TRIGGER abort_proposal_resolution BEFORE UPDATE OF metadata_json
        ON working_memory WHEN NEW.metadata_json LIKE '%"status": "applied"%'
        BEGIN SELECT RAISE(ABORT, 'injected terminal write failure'); END
    """)
    with pytest.raises(sqlite3.IntegrityError, match="injected terminal write failure"):
        model_refresh.apply_model_refresh_proposal(beam, proposal_id)

    assert beam.conn.execute("SELECT count(*) FROM canonical_facts").fetchone()[0] == 0
    assert json.loads(_row(beam, proposal_id)["metadata_json"])["status"] == "pending"
    beam.conn.execute("DROP TRIGGER abort_proposal_resolution")
    assert model_refresh.reject_model_refresh_proposal(beam, proposal_id)["status"] == "rejected"


@pytest.mark.parametrize("outcome", ["applied", "rejected"])
@pytest.mark.parametrize("commit", [True, False], ids=["commit", "rollback"])
def test_resolution_leaves_the_callers_transaction_open(tmp_path, monkeypatch, outcome, commit):
    import sqlite3

    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "caller_transaction")
    invalidations = []
    monkeypatch.setattr(beam, "_invalidate_query_cache_after_commit", invalidations.append)
    beam.conn.execute("BEGIN IMMEDIATE")
    beam.conn.execute("UPDATE working_memory SET importance = 0 WHERE id = ?", (proposal_id,))
    resolve = (
        model_refresh.apply_model_refresh_proposal
        if outcome == "applied"
        else model_refresh.reject_model_refresh_proposal
    )
    assert resolve(beam, proposal_id)["status"] == outcome
    assert beam.conn.in_transaction
    assert invalidations == []
    with sqlite3.connect(beam.db_path) as observer:
        assert observer.execute("SELECT count(*) FROM canonical_facts").fetchone()[0] == 0

    finish = beam.conn.commit if commit else beam.conn.rollback
    finish()
    assert invalidations == (["model_refresh"] if commit else [])
    assert json.loads(_row(beam, proposal_id)["metadata_json"])["status"] == (
        outcome if commit else "pending"
    )
    assert beam.conn.execute("SELECT count(*) FROM canonical_facts").fetchone()[0] == (
        int(commit and outcome == "applied")
    )
    # Rolling back discards the hook instead of handing it to a later commit.
    beam.conn.execute("BEGIN IMMEDIATE")
    beam.conn.commit()
    assert invalidations == (["model_refresh"] if commit else [])


def test_rejected_proposal_does_not_fill_memoria_preference_limit(tmp_path):
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    retired = beam.remember(
        "I prefer compact reporting for weekly updates.",
        source=model_refresh.PROPOSAL_SOURCE,
        metadata={"status": "pending"},
    )
    assert "compact reporting" in beam.memoria_retrieve(
        "what do i prefer reporting", ability="PF", top_k=1,
    )["context"]
    model_refresh.reject_model_refresh_proposal(beam, retired)
    assert "compact reporting" not in beam.memoria_retrieve(
        "what do i prefer reporting", ability="PF", top_k=1,
    )["context"]

    ordinary = beam.remember("I prefer clear reporting for project updates.")
    assert ordinary
    result = beam.memoria_retrieve("what do i prefer reporting", ability="PF", top_k=1)
    assert "clear reporting" in result["context"]
    assert "compact reporting" not in result["context"]
    assert "compact reporting" not in result["facts"][0]["preference"]


def test_rejected_proposal_is_hidden_from_existing_preference_history(tmp_path):
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    retired = beam.remember(
        "I prefer weekly reporting via compact charts.",
        source=model_refresh.PROPOSAL_SOURCE,
        metadata={"status": "pending"},
    )
    ordinary = beam.remember("I prefer weekly reporting.")
    assert beam.conn.execute(
        "SELECT evolution FROM memoria_preferences WHERE source_memory_id = ?", (ordinary,),
    ).fetchone()[0].endswith("via compact charts")
    model_refresh.reject_model_refresh_proposal(beam, retired)

    result = beam.memoria_retrieve("what do i prefer reporting", ability="PF", top_k=1)
    assert "weekly reporting" in result["context"]
    assert "compact charts" not in result["context"]
    assert result["facts"][0]["evolution"] is None


@pytest.mark.parametrize(
    ("ability", "query", "table", "columns", "retired_values", "eligible_values"),
    [
        ("IE", "what is the latency", "memoria_facts", "fact_type,key,value",
         ("metric", "ms", "250"), ("metric", "ms", "80")),
        ("TR", "what happened", "memoria_timelines", "date,description",
         ("2026-06-03", "retired timeline"), ("2026-06-02", "eligible timeline")),
        ("CR", "Reporting", "memoria_kg", "subject,predicate,object",
         ("user", "negation", "Reporting retired"), ("user", "negation", "Reporting eligible")),
        ("MR", "Reporting", "memoria_kg", "subject,predicate,object",
         ("Reporting", "requires", "retired tool"), ("Reporting", "requires", "eligible tool")),
        ("EO", "list sequence", "memoria_facts", "fact_type,value",
         ("sequence", "retired step"), ("sequence", "eligible step")),
        ("IF", "which reporting instruction", "memoria_instructions", "instruction,topic",
         ("retired rule", "reporting"), ("eligible rule", "reporting")),
        ("PF", "what do i prefer reporting", "memoria_preferences", "preference,topic",
         ("retired preference", "reporting"), ("eligible preference", "reporting")),
    ],
)
def test_memoria_specialists_filter_retired_sources_before_limit(
    tmp_path, ability, query, table, columns, retired_values, eligible_values,
):
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "specialist")
    statement = (
        f"INSERT INTO {table} (session_id,message_idx,source_memory_id,{columns}) "
        f"VALUES (?,?,?,{','.join('?' for _ in retired_values)})"
    )
    beam.conn.execute(statement, ("retire", 1, proposal_id, *retired_values))
    beam.conn.execute(statement, ("retire", 2, None, *eligible_values))
    beam.conn.commit()
    value_index = 0 if ability in ("IF", "PF") else -1
    retired_text = retired_values[value_index]
    eligible_text = eligible_values[value_index]
    pending = beam.memoria_retrieve(query, ability=ability, top_k=1)
    assert retired_text in pending["context"]
    model_refresh.reject_model_refresh_proposal(beam, proposal_id)
    resolved = beam.memoria_retrieve(query, ability=ability, top_k=1)
    assert eligible_text in resolved["context"]
    assert retired_text not in resolved["context"]
    assert retired_text not in str(resolved["facts"])


def test_cross_session_preference_history_excludes_retired_source(tmp_path):
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "cross_session")
    beam.conn.executemany(
        "INSERT INTO memoria_preferences (session_id,message_idx,preference,topic,source_memory_id) "
        "VALUES (?,?,?,?,?)",
        [
            ("other", 1, "use compact charts", "reporting", proposal_id),
            ("retire", 2, "use weekly summaries", "reporting", None),
        ],
    )
    beam.conn.commit()
    beam._deduplicate_memoria_cross_session()
    query = "what do i prefer reporting"
    assert "use compact charts" in beam.memoria_retrieve(query, ability="PF")["context"]
    model_refresh.reject_model_refresh_proposal(beam, proposal_id)
    assert "use compact charts" not in beam.memoria_retrieve(query, ability="PF")["context"]

    beam.conn.execute("UPDATE memoria_preferences SET evolution = NULL")
    beam.conn.commit()
    beam._deduplicate_memoria_cross_session()
    assert beam.conn.execute(
        "SELECT evolution FROM memoria_preferences WHERE session_id = 'retire'"
    ).fetchone()[0] is None



def test_fact_history_omits_value_copied_from_retired_proposal(tmp_path):
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    retired = _store_proposal(beam, "fact_history")
    beam._insert_fact("retire", 1, "metric", "ms", "250", "", 0.5, source_memory_id=retired)
    beam._insert_fact("retire", 2, "metric", "ms", "80", "", 0.5)
    beam.conn.commit()
    model_refresh.reject_model_refresh_proposal(beam, retired)
    result = beam.memoria_retrieve("what is the latency", ability="IE", top_k=1)
    assert result["facts"][0]["value"] == "80"
    assert result["facts"][0]["previous_value"] is None
    assert "250" not in result["context"]


@pytest.mark.parametrize("previously_deactivated", [False, True], ids=["fresh", "legacy"])
def test_instruction_dedup_keeps_latest_ordinary_instruction(tmp_path, previously_deactivated):
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "instruction_dedup")
    beam.conn.executemany(
        "INSERT INTO memoria_instructions (session_id,message_idx,instruction,topic,source_memory_id,active) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("retire", 1, "Older ordinary rule", "reporting", None, int(not previously_deactivated)),
            ("retire", 2, "Latest ordinary rule", "reporting", None, int(not previously_deactivated)),
            ("other", 3, "Proposal-only rule", "reporting", proposal_id, 1),
        ],
    )
    beam.conn.commit()
    if not previously_deactivated:
        beam._deduplicate_memoria_cross_session()
        ordinary = beam.memoria_retrieve("reporting", ability="IF")
        assert [fact["instruction"] for fact in ordinary["facts"]] == ["Latest ordinary rule"]
    model_refresh.reject_model_refresh_proposal(beam, proposal_id)
    if previously_deactivated:
        beam._deduplicate_memoria_cross_session()
    result = beam.memoria_retrieve("reporting", ability="IF")
    assert [fact["instruction"] for fact in result["facts"]] == ["Latest ordinary rule"]


def test_kg_dedup_preserves_ordinary_triple_when_proposal_is_retired(tmp_path):
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "kg_dedup")
    beam.conn.executemany(
        "INSERT INTO memoria_kg (session_id,message_idx,subject,predicate,object,source_memory_id) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("retire", 1, "Reporting", "requires", "a checklist", proposal_id),
            ("retire", 2, "Reporting", "requires", "a checklist", None),
            ("other", 3, "Reporting", "requires", "a checklist", None),
        ],
    )
    beam.conn.commit()
    # A pending proposal must not consume the only durable copy.
    beam._deduplicate_memoria_cross_session()
    model_refresh.reject_model_refresh_proposal(beam, proposal_id)
    result = beam.memoria_retrieve("Reporting", ability="MR")
    assert [fact["object"] for fact in result["facts"]] == ["a checklist"]
    assert [
        row["session_id"]
        for row in beam.conn.execute(
            "SELECT session_id FROM memoria_kg WHERE source_memory_id IS NULL"
        )
    ] == ["retire"]


@pytest.mark.parametrize("query", ["atlas", "atlas OR"], ids=["fts", "like-fallback"])
def test_fact_recall_filters_retired_proposals_before_candidate_limits(tmp_path, monkeypatch, query):
    monkeypatch.setenv("MNEMOSYNE_FACT_RECALL_ENABLED", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    proposal_id = _store_proposal(beam, "raw_fact")
    beam.conn.executemany(
        "INSERT INTO facts (fact_id,session_id,subject,predicate,object,source_msg_id,confidence) "
        "VALUES (?,?,?,?,?,?,?)",
        [
            (f"retired-{idx}", "retire", "Atlas", "tracks", f"draft-{idx}", proposal_id, 0.99)
            for idx in range(4)
        ] + [("legacy-fact", "retire", "Atlas", "tracks", "approved note", "42", 0.9)],
    )
    beam.conn.commit()

    assert beam.fact_recall(query, top_k=1)[0]["source_msg_id"] == proposal_id
    assert any(row["id"].startswith("cf_retired-") for row in beam.recall(query, top_k=10))
    model_refresh.reject_model_refresh_proposal(beam, proposal_id)
    assert [row["fact_id"] for row in beam.fact_recall(query, top_k=1)] == ["legacy-fact"]
    results = beam.recall(query, top_k=10)
    assert any(row["id"] == "cf_legacy-fact" for row in results)
    assert not any(row["id"].startswith("cf_retired-") for row in results)


def test_consolidated_fact_recall_keeps_mixed_and_legacy_sources(tmp_path):
    from mnemosyne.core.veracity_consolidation import VeracityConsolidator

    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    rejected = _store_proposal(beam, "consolidated_rejected")
    applied = _store_proposal(beam, "consolidated_applied")
    pending = _store_proposal(beam, "consolidated_pending")
    ordinary = beam.remember("Durable reporting source.")
    assert ordinary
    consolidator = VeracityConsolidator(db_path=beam.db_path, conn=beam.conn)
    cases = {
        "obsolete": [rejected, applied],
        "confirmed": [rejected, ordinary],
        "pending": [applied, pending],
        "legacy": [rejected, "42"],
        "unattributed": [None],
    }
    for value, sources in cases.items():
        for source_id in sources:
            consolidator.consolidate_fact(
                "Atlas", f"tracks_{value}", value, veracity="stated", source=source_id,
            )

    expected = {f"Atlas tracks_{value} {value}" for value in cases}
    assert {item["content"] for item in beam.fact_recall("atlas")} == expected
    model_refresh.reject_model_refresh_proposal(beam, rejected)
    model_refresh.apply_model_refresh_proposal(beam, applied)

    expected.remove("Atlas tracks_obsolete obsolete")
    assert {item["content"] for item in beam.fact_recall("atlas")} == expected
    assert {item.object for item in consolidator.get_consolidated_facts("Atlas")} == (
        cases.keys() - {"obsolete"}
    )
    assert {
        item.metadata["object"] for item in beam._get_polyphonic_engine()._fact_voice("atlas")
    } == cases.keys() - {"obsolete"}


def test_instruction_repair_invalidates_warmed_enhanced_recall_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMOSYNE_ENHANCED_RECALL", "1")
    monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "0")
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    # Exercise the IF reader through the actual cache and recall supplement
    # without expanding the production query classifier's routes.
    monkeypatch.setattr(beam, "_classify_ability", lambda query: "IF")
    proposal_id = _store_proposal(beam, "cached_instruction")
    beam.conn.executemany(
        "INSERT INTO memoria_instructions (session_id,message_idx,instruction,topic,source_memory_id,active) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("retire", 1, "Use reporting checklists", "reporting", None, 0),
            ("other", 2, "Skip reporting checklists", "reporting", proposal_id, 1),
        ],
    )
    beam.conn.commit()
    model_refresh.reject_model_refresh_proposal(beam, proposal_id)

    def recall():
        return beam.recall_enhanced(
            "reporting checklists", top_k=5, use_mmr=False,
            use_synonyms=False, use_weibull=False, use_intent=False, _track_recall=False,
        )

    assert recall() == []
    hits_before = beam._query_cache.stats()["hits"]
    assert recall() == []
    assert beam._query_cache.stats()["hits"] == hits_before + 1
    beam._deduplicate_memoria_cross_session()
    assert beam.memoria_retrieve("reporting checklists", ability="IF")["facts"]
    assert any("Use reporting checklists" in row["content"] for row in recall())


@pytest.mark.parametrize("retired_would_win", [True, False], ids=["retired-primary", "retired-opponent"])
def test_auto_conflicts_ignore_retired_only_proposal_facts(tmp_path, retired_would_win):
    from mnemosyne.core.veracity_consolidation import VeracityConsolidator

    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    consolidator = VeracityConsolidator(db_path=beam.db_path, conn=beam.conn, federation_probe=False)
    proposals = [_store_proposal(beam, f"conflict_{index}") for index in range(3)]
    for proposal_id in proposals:
        retired = consolidator.consolidate_fact("Atlas", "uses", "draft", "inferred", proposal_id)
    for _ in range(1 if retired_would_win else 4):
        ordinary = consolidator.consolidate_fact("Atlas", "uses", "approved", "stated", "legacy")
    assert (retired.confidence > ordinary.confidence) is retired_would_win
    for proposal_id in proposals:
        model_refresh.reject_model_refresh_proposal(beam, proposal_id)

    assert consolidator.run_consolidation_pass() == 0
    assert [fact.object for fact in consolidator.get_consolidated_facts("Atlas")] == ["approved"]
    assert all(
        row["superseded_by"] is None
        for row in beam.conn.execute("SELECT superseded_by FROM consolidated_facts")
    )


def test_consolidation_repairs_only_invalid_retired_winner_links(tmp_path):
    from mnemosyne.core.veracity_consolidation import VeracityConsolidator, compute_fact_id

    beam = BeamMemory(session_id="retire", db_path=tmp_path / "mnemo.db")
    consolidator = VeracityConsolidator(db_path=beam.db_path, conn=beam.conn, federation_probe=False)
    proposal_id = _store_proposal(beam, "repair_winner")

    def fact(subject, value, source=None):
        consolidator.consolidate_fact(subject, "uses", value, "stated", source)
        return compute_fact_id(subject, "uses", value)

    retired = fact("Atlas", "draft", proposal_id)
    ordinary = fact("Atlas", "approved")
    separate = fact("Other", "plan")
    valid_winner = fact("Elm", "current")
    valid_loser = fact("Elm", "past")
    mixed_winner = fact("Oak", "current", proposal_id)
    assert fact("Oak", "current", "42") == mixed_winner
    mixed_loser = fact("Oak", "past")
    intermediate = fact("Elm", "proposal", proposal_id)
    chained = fact("Elm", "older")
    retired_loser = fact("Atlas", "early draft", proposal_id)
    chained_to_retired = fact("Atlas", "previous approved")

    edges = {
        ordinary: retired,
        separate: retired,
        valid_loser: valid_winner,
        mixed_loser: mixed_winner,
        intermediate: valid_winner,
        chained: intermediate,
        retired_loser: retired,
        chained_to_retired: retired_loser,
    }
    for loser, winner in edges.items():
        beam.conn.execute("UPDATE consolidated_facts SET superseded_by = ? WHERE id = ?", (winner, loser))
        beam.conn.execute(
            "INSERT OR IGNORE INTO conflicts (fact_a_id,fact_b_id) VALUES (?,?)",
            (loser, winner),
        )
        beam.conn.execute(
            "UPDATE conflicts SET resolution = ?, resolved_at = '2026-01-01' "
            "WHERE min(fact_a_id,fact_b_id) = min(?,?) AND max(fact_a_id,fact_b_id) = max(?,?)",
            (f"superseded_by_{winner}", loser, winner, loser, winner),
        )
    beam.conn.commit()
    model_refresh.reject_model_refresh_proposal(beam, proposal_id)

    def assert_repaired():
        assert {item.object for item in consolidator.get_consolidated_facts("Atlas")} == {
            "approved", "previous approved",
        }
        assert {
            row["id"]: row["superseded_by"] for row in beam.conn.execute(
                "SELECT id,superseded_by FROM consolidated_facts WHERE superseded_by IS NOT NULL"
            )
        } == {
            loser: winner for loser, winner in edges.items()
            if loser not in {ordinary, chained_to_retired}
        }
        ledger = beam.conn.execute(
            "SELECT resolution,resolved_at FROM conflicts "
            "WHERE min(fact_a_id,fact_b_id) = min(?,?) AND max(fact_a_id,fact_b_id) = max(?,?)",
            (ordinary, retired, ordinary, retired),
        ).fetchone()
        assert tuple(ledger) == (None, None)

    consolidator.run_consolidation_pass(pair_budget=0)
    assert_repaired()
    consolidator.run_consolidation_pass(pair_budget=0)
    assert_repaired()
