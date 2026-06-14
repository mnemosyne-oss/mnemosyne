"""Tests for fact_recall() ranking + content (raw `facts` source).

Covers the fix where fact hits were scored by a flat stored confidence (the FTS
rank was discarded) and surfaced with only the object as content. After the fix:

  * content is the full subject-predicate-object triple, and
  * the score reflects FTS relevance (rank position) combined with confidence,
    so two equal-confidence facts no longer collapse to the same score.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from mnemosyne.core.beam import BeamMemory


def _insert_fact(beam, fact_id, subject, predicate, obj, confidence):
    beam.conn.execute(
        "INSERT INTO facts (fact_id, session_id, subject, predicate, object, "
        "timestamp, source_msg_id, confidence) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (fact_id, "test", subject, predicate, obj,
         "2026-01-01T00:00:00", "msg1", confidence),
    )
    beam.conn.commit()


def test_fact_recall_full_triple_content_and_relevance_scoring():
    with tempfile.TemporaryDirectory() as tmp:
        beam = BeamMemory(session_id="test", db_path=str(Path(tmp) / "facts.db"))

        # Two facts with the SAME confidence, both matching the query term.
        _insert_fact(beam, "f1", "python", "is", "a language", 0.6)
        _insert_fact(beam, "f2", "java", "is", "a language too", 0.6)

        results = beam.fact_recall("language", top_k=10)

        assert len(results) == 2, results

        # content is the full subject-predicate-object triple, not the bare object.
        contents = {r["content"] for r in results}
        assert "python is a language" in contents, contents
        assert "java is a language too" in contents, contents
        assert "a language" not in contents  # never the bare object alone
        # subject is preserved in every hit's content.
        assert all(r["content"].startswith(r["subject"]) for r in results)

        # Equal confidence, but the score is no longer a flat constant: FTS rank
        # position now differentiates the two hits (pre-fix both were 0.6).
        scores = [round(r["score"], 6) for r in results]
        assert len(set(scores)) > 1, f"expected relevance-differentiated scores, got {scores}"
        # scores stay within (0, confidence]; relevance in (0, 1].
        assert all(0.0 < s <= 0.6 + 1e-9 for s in scores), scores


if __name__ == "__main__":  # allow direct execution without pytest
    test_fact_recall_full_triple_content_and_relevance_scoring()
    print("ok")
