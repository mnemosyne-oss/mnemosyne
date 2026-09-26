"""Regression: consolidation inherits source authorship only on tuple unanimity.

dplush review comment 5808202311 on PR #926 (mnemosyne-oss/mnemosyne): the
consolidation path aggregated ``author_id`` and ``author_type`` INDEPENDENTLY,
so a group of ``alice/human`` + ``bob/human`` produced ``maintenance/human``
once ``author_id`` fell back to the beam identity while the unanimous source
``author_type`` was kept. That pair's halves come from different identities,
which misattributes the episodic record.

Contract under test:
  1. source attribution is inherited ONLY when the full
     ``(author_id, author_type)`` tuple is identical across every source row;
  2. on any mismatch the fallback is BOTH beam identity fields together --
     never a source ``author_id`` beside an uncorrelated ``author_type`` (nor
     the mirror-image mix);
  3. no consolidation path can emit such a mixed identity.

These tests drive real ``BeamMemory`` rows in a temp DB rather than a mocked
beam, so they exercise the actual aggregation + INSERT path end to end.
"""

from __future__ import annotations

import logging
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory

BEAM_PAIR = ("maintenance", "bot")
ALICE = ("alice", "human")
BOB = ("bob", "human")


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _beam(db_path, session_id="s1", pair=BEAM_PAIR):
    return BeamMemory(
        session_id=session_id,
        db_path=db_path,
        author_id=pair[0],
        author_type=pair[1],
    )


def _seed_old_wm(beam, rows, ts_offset_hours=200, session_id=None):
    """Insert old working_memory rows with explicit authors (bypasses
    remember() so consolidation is the only writer under test)."""
    conn = sqlite3.connect(str(beam.db_path))
    ts = (datetime.now() - timedelta(hours=ts_offset_hours)).isoformat()
    conn.executemany(
        "INSERT INTO working_memory "
        "(id, content, source, timestamp, session_id, author_id, author_type) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (rid, content, source, ts, session_id or beam.session_id, aid, atype)
            for rid, content, source, aid, atype in rows
        ],
    )
    conn.commit()
    conn.close()


def _ep_rows(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        return [
            tuple(row)
            for row in conn.execute(
                "SELECT author_id, author_type FROM episodic_memory ORDER BY rowid ASC"
            ).fetchall()
        ]
    finally:
        conn.close()


def _ep_author(db_path):
    rows = _ep_rows(db_path)
    assert rows, "consolidation produced no episodic row"
    return rows[-1]


def _assert_attributable(emitted, source_pairs, beam_pair):
    """Assertion-level invariant: an emitted identity must BE one of the
    candidate identities -- a source row's complete pair, or the beam pair.

    A mixed pair (a source ``author_id`` beside an uncorrelated
    ``author_type``, in either direction) matches no single identity and
    fails here.
    """
    candidates = {tuple(pair) for pair in source_pairs} | {tuple(beam_pair)}
    assert emitted in candidates, (
        f"consolidation emitted a mixed identity {emitted!r}; it matches no "
        f"single identity (expected one of {sorted(candidates, key=repr)!r})"
    )


def _assert_no_mixed_episodic_rows(db_path, source_pairs, beam_pair):
    for emitted in _ep_rows(db_path):
        _assert_attributable(emitted, source_pairs, beam_pair)


GROUP_CASES = [
    pytest.param(
        [("a1", "alpha fact", "conversation", "alice", "human"),
         ("a2", "beta fact", "conversation", "alice", "human")],
        ALICE,
        id="unanimous-pair-inherited",
    ),
    pytest.param(
        [("a1", "alpha fact", "conversation", "alice", "human"),
         ("a2", "beta fact", "conversation", "bob", "human")],
        BEAM_PAIR,
        id="distinct-author-ids-same-author-type",
    ),
    pytest.param(
        [("a1", "alpha fact", "conversation", "alice", "human"),
         ("a2", "beta fact", "conversation", "alice", "agent")],
        BEAM_PAIR,
        id="same-author-id-mixed-author-types",
    ),
    pytest.param(
        [("a1", "alpha fact", "conversation", "alice", "human"),
         ("a2", "beta fact", "conversation", "alice", None)],
        BEAM_PAIR,
        id="sibling-missing-author-type",
    ),
    pytest.param(
        [("a1", "alpha fact", "conversation", "alice", "human"),
         ("a2", "beta fact", "conversation", None, "human")],
        BEAM_PAIR,
        id="sibling-missing-author-id",
    ),
    pytest.param(
        [("a1", "alpha fact", "conversation", None, None),
         ("a2", "beta fact", "conversation", None, None)],
        BEAM_PAIR,
        id="all-authors-absent",
    ),
    pytest.param(
        [("a1", "alpha fact", "conversation", b"alice", "human"),
         ("a2", "beta fact", "conversation", b"alice", "human")],
        BEAM_PAIR,
        id="non-text-author-values-degrade-to-absent",
    ),
]

# The alien-session maintenance path (sleep_all_sessions) is a second
# consolidation path; it must obey the same unanimity contract. Its cases
# assert the EXACT expected pair, not merely attributability: a regression
# that always stamped the caller's pair would stay "attributable" for the
# unanimous group and still be caught here.
SLEEP_ALL_SESSION_CASES = [
    pytest.param(
        [("a1", "alpha fact", "conversation", "alice", "human"),
         ("a2", "beta fact", "conversation", "bob", "human")],
        BEAM_PAIR,
        id="mixed-author-ids-same-author-type",
    ),
    pytest.param(
        [("a1", "alpha fact", "conversation", "alice", "human"),
         ("a2", "beta fact", "conversation", "alice", "human")],
        ALICE,
        id="unanimous-pair-inherited",
    ),
]


class TestTupleUnanimityRegression:
    def test_distinct_author_ids_same_type_do_not_produce_a_mixed_record(
        self, temp_db
    ):
        """dplush's exact repro: two distinct author_ids sharing one
        author_type must not be consolidated into a record that mixes them.

        Pre-fix this group emitted ``maintenance/human`` -- the beam
        ``author_id`` beside the unanimous source ``author_type``.
        """
        beam = _beam(temp_db)
        _seed_old_wm(beam, [
            ("a1", "alpha fact", "conversation", "alice", "human"),
            ("a2", "beta fact", "conversation", "bob", "human"),
        ])

        result = beam.sleep()

        assert result["status"] == "consolidated"
        emitted = _ep_author(temp_db)
        # The reported failure mode, named explicitly.
        assert emitted != ("maintenance", "human")
        assert emitted != ("alice", "bot")
        assert emitted != ("bob", "bot")
        _assert_attributable(emitted, [ALICE, BOB], BEAM_PAIR)
        assert emitted == BEAM_PAIR

    @pytest.mark.parametrize("rows,expected", GROUP_CASES)
    def test_sleep_never_emits_a_mixed_identity(self, temp_db, rows, expected):
        beam = _beam(temp_db)
        _seed_old_wm(beam, rows)

        result = beam.sleep()

        assert result["status"] == "consolidated"
        emitted = _ep_author(temp_db)
        _assert_attributable(emitted, [(r[3], r[4]) for r in rows], BEAM_PAIR)
        _assert_no_mixed_episodic_rows(temp_db, [(r[3], r[4]) for r in rows], BEAM_PAIR)
        assert emitted == expected

    @pytest.mark.parametrize("rows,expected", SLEEP_ALL_SESSION_CASES)
    def test_sleep_all_sessions_never_emits_a_mixed_identity(
        self, temp_db, rows, expected
    ):
        """The alien-session maintenance path (sleep_all_sessions) is a
        second consolidation path; it must obey the same invariant, and it
        must inherit the unanimous source pair (not merely avoid a mix)."""
        caller = _beam(temp_db, session_id="caller")
        _seed_old_wm(caller, rows, session_id="s1")

        caller.sleep_all_sessions()

        emitted = _ep_author(temp_db)
        source_pairs = [(r[3], r[4]) for r in rows]
        _assert_attributable(emitted, source_pairs, BEAM_PAIR)
        _assert_no_mixed_episodic_rows(temp_db, source_pairs, BEAM_PAIR)
        assert emitted == expected


class TestConsolidateToEpisodicPairAtomicity:
    def test_explicit_pair_is_stamped_verbatim(self, temp_db):
        beam = _beam(temp_db)
        eid = beam.consolidate_to_episodic(
            "summary", ["wm1"], author_id="alice", author_type="human"
        )
        row = tuple(beam.conn.execute(
            "SELECT author_id, author_type FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone())
        _assert_attributable(row, [ALICE], BEAM_PAIR)
        assert row == ALICE

    def test_absent_pair_falls_back_to_the_beam_pair(self, temp_db):
        beam = _beam(temp_db)
        eid = beam.consolidate_to_episodic("summary", ["wm1"])
        row = tuple(beam.conn.execute(
            "SELECT author_id, author_type FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone())
        _assert_attributable(row, [ALICE], BEAM_PAIR)
        assert row == BEAM_PAIR

    @pytest.mark.parametrize("kwargs", [
        pytest.param({"author_id": "alice"}, id="author-id-only"),
        pytest.param({"author_type": "human"}, id="author-type-only"),
    ])
    def test_half_supplied_pair_does_not_emit_a_mixed_identity(
        self, temp_db, caplog, kwargs
    ):
        """A caller supplying one field only has supplied an identity it
        cannot correlate with the other field's owner; the incomplete pair
        must degrade to the beam pair rather than emit a mix."""
        beam = _beam(temp_db)
        with caplog.at_level(logging.WARNING, logger="mnemosyne.core.beam"):
            eid = beam.consolidate_to_episodic("summary", ["wm1"], **kwargs)

        row = tuple(beam.conn.execute(
            "SELECT author_id, author_type FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone())
        _assert_attributable(row, [ALICE], BEAM_PAIR)
        assert row == BEAM_PAIR
        assert any(
            "incomplete author pair" in record.getMessage()
            for record in caplog.records
        ), "the incomplete-pair degradation must be observable in the log"

    @pytest.mark.parametrize("kwargs,expected", [
        pytest.param({}, BEAM_PAIR, id="no-author-kwargs"),
        pytest.param({"author_id": None, "author_type": None}, BEAM_PAIR,
                     id="explicit-nones"),
        pytest.param({"author_id": "alice"}, BEAM_PAIR, id="author-id-only"),
        pytest.param({"author_type": "human"}, BEAM_PAIR, id="author-type-only"),
        pytest.param({"author_id": "alice", "author_type": "human"}, ALICE,
                     id="full-pair"),
        pytest.param({"author_id": "bob", "author_type": "human"}, BOB,
                     id="other-full-pair"),
    ])
    def test_no_kwarg_combination_emits_a_mixed_identity(
        self, temp_db, kwargs, expected
    ):
        """Every kwarg combination must emit the EXACT expected pair.

        Attributability alone is too weak: a regression that always stamped
        the beam pair would keep passing the "not a mix" check for every
        case, and one that always used the caller's pair would pass the
        full-pair cases. The expected-pair column pins each case, including
        the two complete-pair cases that must be inherited verbatim.
        """
        beam = _beam(temp_db)
        eid = beam.consolidate_to_episodic("summary", ["wm1"], **kwargs)
        row = tuple(beam.conn.execute(
            "SELECT author_id, author_type FROM episodic_memory WHERE id = ?", (eid,)
        ).fetchone())
        _assert_attributable(row, [ALICE, BOB], BEAM_PAIR)
        assert row == expected
