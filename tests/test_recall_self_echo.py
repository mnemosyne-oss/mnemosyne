"""Regression tests: recall(exclude_session_id=...) drops fresh self-echo rows.

A memory provider that captures the live conversation into working_memory
should not get those rows echoed back by its own prefetch: they are already
verbatim in the requester's context window. recall() accepts up to two
session-key variants and excludes RECENT rows under those keys (default
window 6h, MNEMOSYNE_SELF_ECHO_HOURS). Older rows under the same key stay
recallable -- the key persists across history replays/migrations where the
content is no longer on screen.

Boundary coverage:
- the exclusion cutoff is built on the SAME naive-local clock as the
  production writers (remember()/remember_batch() stamp datetime.now()),
  so the window does not silently shift by the host TZ offset;
- MNEMOSYNE_SELF_ECHO_HOURS is validated (garbage falls back, non-positive
  clamps to a small floor) instead of crashing recall or excluding history
  of every age;
- the polyphonic engine path honors the exclusion too (post-voice, on the
  combined candidate set).
"""

from datetime import datetime, timedelta

import pytest

from mnemosyne.core.beam import BeamMemory


@pytest.fixture
def temp_db(tmp_path):
    return tmp_path / "test.db"


def _make_beam(db_path, session_id):
    return BeamMemory(session_id=session_id, db_path=db_path)


def _store_in_session(beam, session_key, content, age_hours=0.0,
                      timestamp=None, row_id=None):
    """Insert a working row with an arbitrary session_id and age.

    Returns the inserted TEXT id (working_memory.id), not the sqlite rowid —
    recall results and PolyphonicResult carry the id column."""
    rid = row_id or f"id-{session_key}-{abs(hash(content)) % 10**8}"
    ts = timestamp or (datetime.now() - timedelta(hours=age_hours)).isoformat()
    cur = beam.conn.cursor()
    cur.execute(
        "INSERT INTO working_memory (id, session_id, content, source, timestamp, importance)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (
            rid,
            session_key,
            content,
            "conversation",
            ts,
            0.5,
        ),
    )
    beam.conn.commit()
    return rid


def test_fresh_rows_in_excluded_session_are_not_recalled(temp_db):
    beam = _make_beam(temp_db, "other-session")
    _store_in_session(beam, "live-session", "User is planning a trip to Lisbon next spring")

    results = beam.recall(
        "Lisbon trip planning",
        exclude_session_id="live-session",
    )
    assert all("Lisbon" not in r["content"] for r in results), (
        "fresh row from the excluded session must not be echoed back"
    )


def test_exclusion_accepts_two_key_variants(temp_db):
    beam = _make_beam(temp_db, "other-session")
    _store_in_session(beam, "hermes:group:123", "User prefers window seats on trains")
    _store_in_session(beam, "group:123", "User prefers aisle seats on buses")

    results = beam.recall(
        "seat preferences",
        exclude_session_id="hermes:group:123",
        exclude_session_id_alt="group:123",
    )
    contents = " ".join(r["content"] for r in results)
    assert "window seats" not in contents
    assert "aisle seats" not in contents


def test_old_rows_stay_and_fresh_rows_go_under_same_key(temp_db):
    """Both directions of the clause in one test: a fresh row under the
    excluded key is dropped while an old row under the SAME key is kept."""
    beam = _make_beam(temp_db, "other-session")
    _store_in_session(
        beam, "live-session", "User finished reading a biography of Ada Lovelace",
        age_hours=12.0,
    )
    _store_in_session(
        beam, "live-session", "User finished skimming an Ada Lovelace zine today",
        age_hours=0.0,
    )

    results = beam.recall(
        "Ada Lovelace biography reading",
        exclude_session_id="live-session",
    )
    contents = " ".join(r["content"] for r in results)
    assert "Ada Lovelace" in contents, (
        "rows older than the echo window must stay recallable"
    )
    assert "zine" not in contents, (
        "the fresh counterpart row under the same key must be excluded"
    )


def test_no_exclusion_kwarg_returns_everything(temp_db):
    beam = _make_beam(temp_db, "live-session")
    _store_in_session(beam, "live-session", "User is evaluating espresso grinders")

    results = beam.recall("espresso grinders")
    assert any("espresso grinders" in r["content"] for r in results), (
        "without the kwarg, session scoping behaves exactly as before"
    )


class TestCutoffClockAndValidation:
    """The cutoff must ride the same clock as the writers, and the env knob
    must be validated (auditor findings: TZ-shift leak, garbage crash,
    non-positive blanket exclusion)."""

    def test_cutoff_uses_naive_local_clock(self, monkeypatch):
        """Deterministic: pin the clock, assert exact cutoff math — catches
        the aware/utcnow regression on EVERY host TZ."""
        from mnemosyne.core import beam as beam_module

        monkeypatch.delenv("MNEMOSYNE_SELF_ECHO_HOURS", raising=False)
        fixed_now = datetime(2026, 9, 5, 12, 0, 0)

        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed_now

        monkeypatch.setattr(beam_module, "datetime", _FixedDatetime)
        cutoff = datetime.fromisoformat(beam_module._self_echo_cutoff_iso())
        assert cutoff == fixed_now - timedelta(hours=6), (
            "cutoff must be exactly now() - 6h on the writers' clock"
        )

    def test_non_finite_env_falls_back_not_crashes(self, monkeypatch):
        """nan/inf bypass < comparisons (NaN comparisons are always False)
        and used to crash timedelta construction (ValueError/OverflowError)."""
        from mnemosyne.core import beam as beam_module

        for bad in ("nan", "inf", "-inf", "1e309", "-1e309", "NaN", "Infinity"):
            monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_HOURS", bad)
            cutoff = datetime.fromisoformat(beam_module._self_echo_cutoff_iso())
            delta = datetime.now() - cutoff
            assert 0 < delta.total_seconds() < 6 * 3600 + 5, (
                f"{bad!r} must fall back to the 6h default, not crash"
            )

    def test_recall_survives_non_finite_env(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_HOURS", "nan")
        beam = _make_beam(temp_db, "other-session")
        _store_in_session(beam, "live-session", "User is planning a trip to Lisbon next spring")
        results = beam.recall("Lisbon trip planning", exclude_session_id="live-session")
        assert all("Lisbon" not in r["content"] for r in results), (
            "non-finite env must fall back to the default window and still exclude"
        )

    def test_garbage_env_falls_back_not_crashes(self, monkeypatch):
        from mnemosyne.core import beam as beam_module

        monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_HOURS", "abc")
        cutoff = datetime.fromisoformat(beam_module._self_echo_cutoff_iso())
        delta = datetime.now() - cutoff
        assert 0 < delta.total_seconds() < 6 * 3600 + 5, (
            "garbage value must fall back to the 6h default"
        )

    def test_non_positive_env_clamps_to_floor(self, monkeypatch):
        from mnemosyne.core import beam as beam_module

        for bad in ("-1", "0"):
            monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_HOURS", bad)
            cutoff = datetime.fromisoformat(beam_module._self_echo_cutoff_iso())
            delta = datetime.now() - cutoff
            assert delta.total_seconds() <= 60, (
                f"{bad!r} must clamp to the small positive floor, not "
                "exclude history of every age"
            )

    def test_recall_survives_garbage_env(self, temp_db, monkeypatch):
        monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_HOURS", "abc")
        beam = _make_beam(temp_db, "other-session")
        _store_in_session(beam, "live-session", "User is planning a trip to Lisbon next spring")
        results = beam.recall("Lisbon trip planning", exclude_session_id="live-session")
        assert all("Lisbon" not in r["content"] for r in results)


class TestPolyphonicExclusion:
    """The engine path must honor the exclusion too (post-voice, combined
    candidate set) -- with MNEMOSYNE_POLYPHONIC_RECALL=1 the kwargs used to
    be dropped silently."""

    def test_polyphonic_path_honors_exclusion(self, temp_db, monkeypatch):
        beam = _make_beam(temp_db, "other-session")
        mid = _store_in_session(
            beam, "live-session", "User is planning a trip to Lisbon"
        )
        assert isinstance(mid, str), (
            "guard: the comparison target must be the TEXT id column"
        )
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")

        engine = beam._get_polyphonic_engine()
        results = engine.recall(
            "what was said recently about Lisbon",
            top_k=10,
            exclude_session_id="live-session",
        )
        # Positive control: prove the row REACHES the candidate set without
        # the kwarg, so the exclusion is the only thing keeping it out.
        engine2 = beam._get_polyphonic_engine()
        unfiltered = engine2.recall(
            "what was said recently about Lisbon", top_k=10
        )
        assert any(r.memory_id == mid for r in unfiltered), (
            "positive control failed: row never reached the candidate set"
        )
        leaked = [r for r in results if r.memory_id == mid]
        assert not leaked, (
            "polyphonic engine must drop fresh rows under the excluded key"
        )

    def test_polyphonic_recall_end_to_end(self, temp_db, monkeypatch):
        beam = _make_beam(temp_db, "other-session")
        mid = _store_in_session(
            beam, "live-session", "User is planning a trip to Lisbon"
        )
        monkeypatch.setenv("MNEMOSYNE_POLYPHONIC_RECALL", "1")

        res = beam.recall(
            "what was said recently about Lisbon",
            exclude_session_id="live-session",
        )
        leaked = [r for r in res if r.get("id") == mid]
        assert not leaked, (
            "recall() under the polyphonic flag must honor the exclusion"
        )

    def test_aware_rows_use_local_wall_clock(self, monkeypatch):
        """Pure-function regression against the UTC-wall mutant, on EVERY
        host: re-run the window math under TZ=Asia/Tokyo via time.tzset().
        Construction (Tokyo, UTC+9): cutoff naive-local 04:00; an aware row
        stamped 2026-09-04T19:00:00Z is 04:00 next-day local — exactly the
        cutoff — so use 19:00Z+1min = local 04:01, INSIDE. The UTC-wall
        mutant compares the raw 19:00Z wall vs 04:00 and excludes it."""
        import time as _time
        from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine

        # TZ=Asia/Tokyo: cutoff naive-local 2026-09-05 04:00 (locals see
        # this as 2026-09-04 19:00Z). Row: 2026-09-04 19:30Z == local
        # 04:30 — 30 min fresh, inside the 6h window.
        monkeypatch.setenv("TZ", "Asia/Tokyo")
        _time.tzset()
        try:
            row_ts = "2026-09-04T19:30:00+00:00"
            cutoff_local = datetime(2026, 9, 5, 4, 0, 0)
            inside = PolyphonicRecallEngine._row_inside_echo_window(
                row_ts, cutoff_local
            )
        finally:
            _time.tzset()
        assert inside, (
            "an aware row 30-min fresh on the writers' local wall clock "
            "must be inside the window; the UTC-wall mutant excludes it "
            "(compares 19:30 vs 04:00)"
        )
