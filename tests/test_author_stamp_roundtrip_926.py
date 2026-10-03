"""Round-trip regression: author stamps survive export_to_dict/import_from_dict.

Regression for dplush's 2026-09-16 review on PR #926 ("changes requested"):

  "The new per-write author_id / author_type metadata is not included in
   export_to_dict() or restored by import_from_dict(). This makes the
   backup/restore path silently erase every author stamp introduced by this
   PR for both working and episodic rows. I reproduced this on the PR head
   (da9a630): a working and episodic row written with ("alice", "human")
   both restore as (None, None)."

Both working_memory and episodic_memory must carry the stamp through a full
export -> import cycle. A pre-PR export (no author keys) must still import,
restoring NULL rather than raising.
"""
from __future__ import annotations

import os

from mnemosyne.core.beam import BeamMemory, init_beam

AUTHOR_ID = "alice"
AUTHOR_TYPE = "human"


def _beam(tmp: str, session_id: str = "sess-a") -> BeamMemory:
    os.environ["MNEMOSYNE_DATA_DIR"] = tmp
    db = os.path.join(tmp, "rt.db")
    init_beam(db_path=db)
    return BeamMemory(session_id=session_id, db_path=db)


def test_working_memory_author_stamp_survives_round_trip(tmp_path):
    """A working row written with an author restores with the same author."""
    src = _beam(str(tmp_path / "a"))
    src.remember(content="Secret A: the sky is green", scope="session",
                 author_id=AUTHOR_ID, author_type=AUTHOR_TYPE)

    exported = src.export_to_dict()
    row = next(r for r in exported["working_memory"]
               if "sky is green" in (r.get("content") or ""))
    assert row.get("author_id") == AUTHOR_ID, "export must carry the working author_id"
    assert row.get("author_type") == AUTHOR_TYPE, "export must carry the working author_type"

    dst = _beam(str(tmp_path / "b"))
    dst.import_from_dict(exported)
    conn = dst.conn
    got = conn.execute(
        "SELECT author_id, author_type FROM working_memory WHERE content LIKE '%sky is green%'"
    ).fetchone()
    assert got is not None, "row must be present after import"
    assert tuple(got) == (AUTHOR_ID, AUTHOR_TYPE), (
        f"working author stamp lost on restore: {tuple(got)!r}"
    )


def test_episodic_memory_author_stamp_survives_round_trip(tmp_path):
    """An episodic row restores with its author (sleep-time consolidation path)."""
    src = _beam(str(tmp_path / "a"))
    ep_id = src.consolidate_to_episodic(
        summary="AlphaEvolve changed the search.",
        source_wm_ids=["wm-1"],
        author_id=AUTHOR_ID,
        author_type=AUTHOR_TYPE,
    )
    exported = src.export_to_dict()
    row = next((r for r in exported["episodic_memory"] if r.get("id") == ep_id), None)
    assert row is not None, "episodic row must be exported"
    assert row.get("author_id") == AUTHOR_ID, "export must carry the episodic author_id"
    assert row.get("author_type") == AUTHOR_TYPE, "export must carry the episodic author_type"

    dst = _beam(str(tmp_path / "b"))
    dst.import_from_dict(exported)
    got = dst.conn.execute(
        "SELECT author_id, author_type FROM episodic_memory WHERE id = ?", (ep_id,)
    ).fetchone()
    assert got is not None, "episodic row must be present after import"
    assert tuple(got) == (AUTHOR_ID, AUTHOR_TYPE), (
        f"episodic author stamp lost on restore: {tuple(got)!r}"
    )


def test_pre_pr_export_without_author_keys_still_imports(tmp_path):
    """Backward compatibility: an export predating this PR has no author keys."""
    src = _beam(str(tmp_path / "a"))
    src.remember(content="Legacy row with no author", scope="session")
    exported = src.export_to_dict()

    # Simulate a pre-PR export: strip the author keys entirely.
    for tbl in ("working_memory", "episodic_memory"):
        for row in exported.get(tbl, []):
            row.pop("author_id", None)
            row.pop("author_type", None)

    dst = _beam(str(tmp_path / "b"))
    dst.import_from_dict(exported)
    got = dst.conn.execute(
        "SELECT author_id, author_type FROM working_memory WHERE content LIKE '%Legacy row%'"
    ).fetchone()
    assert got is not None, "legacy export must still import"
    assert tuple(got) == (None, None), "absent author keys restore as NULL"
