"""Tests for E8b — fleet census of conflict rows at sleep time.

The census exists because the disposition policy states its accepted scope
as a bound: while NO two home banks hold the same order-normalized conflict
pair, the federation handshake only needs to probe the shared surface. That
bound has to be recomputed at every sleep pass, or it silently decays into
an assertion. These tests pin the four things that make it a bound rather
than a slogan:

  1. a planted home↔home twin is FLAGGED, and the same fixture without it
     reports zero (both directions, deterministic);
  2. the census never writes to the banks it reads;
  3. the emitted stamp carries both Mnemosyne clocks, each named with its
     zone (``created_at`` is naive-UTC, ``valid_from``/``valid_until`` are
     naive-local);
  4. it is exercised from the sleep-time path, not only by direct import.
"""

import hashlib
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from mnemosyne.core import fleet_census
from mnemosyne.core.beam import BeamMemory

# Canonical conflicts DDL, copied from the source of truth in
# mnemosyne/core/veracity_consolidation.py (_init_conflicts_table). The
# census only READS this table; the fixture owns creating it.
_CONFLICTS_DDL = """
CREATE TABLE IF NOT EXISTS conflicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_a_id TEXT NOT NULL,
    fact_b_id TEXT NOT NULL,
    conflict_type TEXT,
    resolution TEXT,
    resolved_at TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

# The planted twin. Deliberately NOT pre-sorted: the detector does not
# canonicalize orientation, so a census that compared raw (a, b) would miss
# the swapped occurrence — exactly the failure the order-normalized key
# exists to prevent.
_TWIN = ("cf_zzz_twin", "cf_aaa_twin")


def _make_bank(path: Path, rows) -> Path:
    """Create a bank holding a conflicts table with the given rows.

    rows: iterable of (fact_a_id, fact_b_id, resolution) — resolution None
    means an OPEN conflict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_CONFLICTS_DDL)
        conn.executemany(
            "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type, resolution) "
            "VALUES (?, ?, ?, ?)",
            [(a, b, "contradiction", res) for a, b, res in rows],
        )
        conn.commit()
    finally:
        conn.close()
    return path


def _build_fleet(root: Path, banks) -> Path:
    """Build a fleet of home banks under root/<name>/mnemosyne.db."""
    for name, rows in banks.items():
        _make_bank(root / name / "mnemosyne.db", rows)
    return root


@pytest.fixture
def fleet_env(tmp_path, monkeypatch):
    """Isolate the census from the live host: pin the root, clear the rest."""
    root = tmp_path / "fleet"
    root.mkdir()
    monkeypatch.setenv(fleet_census.FLEET_ROOT_ENV, str(root))
    # Ambient overrides would redirect the shared-surface classification
    # away from the fixture and make these tests host-dependent.
    monkeypatch.delenv("MNEMOSYNE_SHARED_DB_PATH", raising=False)
    monkeypatch.delenv("MNEMOSYNE_HOME", raising=False)
    monkeypatch.delenv("MNEMOSYNE_FLEET_CENSUS", raising=False)
    return root


@pytest.fixture
def beam_db(tmp_path):
    """A beam DB kept OUTSIDE the fleet root, so it never joins the census."""
    return tmp_path / "beam.db"


def _seed_old_wm(db_path, session_id, n=3, ts_offset_hours=200):
    ts = (datetime.now() - timedelta(hours=ts_offset_hours)).isoformat()
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executemany(
            "INSERT INTO working_memory (id, content, source, timestamp, session_id) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (f"e8b-{session_id}-{i}", f"e8b-content-{i}", "conversation", ts, session_id)
                for i in range(n)
            ],
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 0. Root resolution — the scope of the fleet, resolved at read time
# --------------------------------------------------------------------------

class TestRootResolution:

    def test_profile_scoped_home_resolves_to_the_shared_hermes_root(
            self, tmp_path, monkeypatch):
        """HERMES_HOME inside a profile home must not narrow the fleet.

        A census scoped to one seat's home would still print a fleet bound —
        and under-report twins, which is the failure direction that matters.
        """
        root = tmp_path / "hermes"
        profile_home = root / "profiles" / "keeper"
        profile_home.mkdir(parents=True)
        monkeypatch.delenv(fleet_census.FLEET_ROOT_ENV, raising=False)
        monkeypatch.setenv("HERMES_HOME", str(profile_home))
        monkeypatch.setenv("HERMES_PROFILE", "keeper")

        assert fleet_census.default_fleet_root() == root

    def test_plain_hermes_home_is_the_fleet_root(self, tmp_path, monkeypatch):
        deploy_root = tmp_path / "deploy" / "hermes"
        deploy_root.mkdir(parents=True)
        monkeypatch.delenv(fleet_census.FLEET_ROOT_ENV, raising=False)
        monkeypatch.setenv("HERMES_HOME", str(deploy_root))
        monkeypatch.delenv("HERMES_PROFILE", raising=False)

        assert fleet_census.default_fleet_root() == deploy_root

    def test_home_matching_profile_name_but_not_under_profiles_is_not_widened(
            self, tmp_path, monkeypatch):
        """Only a recognisable ``<root>/profiles/<name>`` layout is widened."""
        home = tmp_path / "keeper"
        home.mkdir()
        monkeypatch.delenv(fleet_census.FLEET_ROOT_ENV, raising=False)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_PROFILE", "keeper")

        assert fleet_census.default_fleet_root() == home

    def test_fleet_root_env_overrides_everything(self, tmp_path, monkeypatch):
        override = tmp_path / "pinned"
        monkeypatch.setenv(fleet_census.FLEET_ROOT_ENV, str(override))
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "keeper"))
        monkeypatch.setenv("HERMES_PROFILE", "keeper")

        assert fleet_census.default_fleet_root() == override

    def test_absent_hermes_home_falls_back_to_dot_hermes(self, monkeypatch):
        monkeypatch.delenv(fleet_census.FLEET_ROOT_ENV, raising=False)
        monkeypatch.delenv("HERMES_HOME", raising=False)

        assert fleet_census.default_fleet_root() == Path.home() / ".hermes"

    def test_shared_db_path_follows_the_root_and_its_override(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MNEMOSYNE_SHARED_DB_PATH", raising=False)
        monkeypatch.delenv("MNEMOSYNE_HOME", raising=False)
        assert fleet_census.shared_db_path(tmp_path) == (
            tmp_path / "mnemosyne" / "data" / "shared" / "mnemosyne.db"
        )

        explicit = tmp_path / "elsewhere" / "surface.db"
        monkeypatch.setenv("MNEMOSYNE_SHARED_DB_PATH", str(explicit))
        assert fleet_census.shared_db_path(tmp_path) == explicit


# --------------------------------------------------------------------------
# 0b. Path identity — absolute, deduplicated, symlink-safe
#
# The bound is only as good as the identity behind "two banks". Two failure
# shapes make that identity lie: a RELATIVE root or caller-supplied banks
# reach ``Path.as_uri()`` inside the reader and raise ValueError, which is
# not a ``sqlite3.Error`` — the whole census dies and the sleep result keeps
# only ``{"error": ...}``, no bound. And a SYMLINKED ``.db`` under the root
# is the same file under two names: counted as two holders it manufactures a
# home↔home twin out of one bank. Census paths are resolved and deduped at
# the entry point, so neither shape can reach the reader.
# --------------------------------------------------------------------------

class TestPathIdentity:

    def test_relative_fleet_root_completes_the_census(self, tmp_path, monkeypatch):
        """``MNEMOSYNE_FLEET_ROOT=./fleet`` must census, not explode."""
        root = tmp_path / "fleet"
        _build_fleet(root, {
            "home-a": [_TWIN + (None,)],
            "home-b": [(_TWIN[1], _TWIN[0], None)],
        })
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(fleet_census.FLEET_ROOT_ENV, "fleet")
        monkeypatch.delenv("MNEMOSYNE_SHARED_DB_PATH", raising=False)
        monkeypatch.delenv("MNEMOSYNE_HOME", raising=False)

        report = fleet_census.census()

        assert Path(report["root"]).is_absolute()
        assert report["unreadable"] == []
        assert report["banks_scanned"] == 2
        # The planted twin is still found through a relative root — the fix
        # must not merely survive, it must still measure the bound.
        assert report["home_home_twin_count"] == 1, report
        assert report["bound_holds"] is False

    def test_relative_bank_paths_from_caller_are_resolved(self, tmp_path, monkeypatch):
        """A caller may hand the census relative ``banks=`` paths."""
        _build_fleet(tmp_path, {
            "home-a": [_TWIN + (None,)],
            "home-b": [(_TWIN[1], _TWIN[0], None)],
        })
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(fleet_census.FLEET_ROOT_ENV, str(tmp_path))
        monkeypatch.delenv("MNEMOSYNE_SHARED_DB_PATH", raising=False)
        monkeypatch.delenv("MNEMOSYNE_HOME", raising=False)

        report = fleet_census.census(banks=[
            Path("home-a/mnemosyne.db"),
            Path("./home-b/../home-b/mnemosyne.db"),
        ])

        assert report["unreadable"] == []
        assert report["banks_scanned"] == 2
        assert report["home_home_twin_count"] == 1, report

    def test_symlinked_bank_is_counted_once(self, fleet_env):
        """One file behind two names is one holder, not a self-manufactured twin.

        Without resolve-and-dedup, ``mirror.db`` (a symlink to the real
        bank) is discovered as a second home bank holding the same pair, and
        the bound trips on a fleet of one.
        """
        real = _make_bank(fleet_env / "home-a" / "mnemosyne.db", [_TWIN + (None,)])
        link = real.parent / "mirror.db"
        link.symlink_to(real)

        report = fleet_census.census()

        assert report["banks_scanned"] == 1, report
        assert report["home_home_twin_count"] == 0, report
        assert report["bound_holds"] is True


# --------------------------------------------------------------------------
# 1. The bound, both directions
# --------------------------------------------------------------------------

class TestHomeHomeBound:

    def test_planted_home_home_twin_is_flagged(self, fleet_env):
        _build_fleet(fleet_env, {
            "home-a": [_TWIN + (None,)],
            # Same pair, OPPOSITE orientation: a raw (a, b) comparison sees
            # two different pairs and reports a clean fleet.
            "home-b": [(_TWIN[1], _TWIN[0], None)],
            "home-c": [("cf_solo_a", "cf_solo_b", None)],
        })

        report = fleet_census.census()

        assert report["banks_scanned"] == 3
        assert report["home_home_twin_count"] == 1, report
        assert report["bound_holds"] is False
        twin = report["home_home_twins"][0]
        assert twin["pair"] == ["cf_aaa_twin", "cf_zzz_twin"]
        assert sorted(twin["banks"]) == [
            "home-a/mnemosyne.db", "home-b/mnemosyne.db",
        ], twin

    def test_clean_fleet_reports_zero_and_holds(self, fleet_env):
        _build_fleet(fleet_env, {
            "home-a": [_TWIN + (None,)],
            "home-b": [("cf_other_a", "cf_other_b", None)],
        })

        report = fleet_census.census()

        assert report["home_home_twin_count"] == 0, report
        assert report["home_home_twins"] == []
        assert report["bound_holds"] is True

    def test_report_is_deterministic_across_runs(self, fleet_env):
        _build_fleet(fleet_env, {
            "home-a": [_TWIN + (None,), ("cf_x", "cf_y", None)],
            "home-b": [(_TWIN[1], _TWIN[0], None)],
        })

        first = fleet_census.census()
        second = fleet_census.census()

        # Every semantic field must match. The stamps are wall clock and are
        # the ONLY fields allowed to differ.
        volatile = {"read_at_utc", "read_at_local", "local_zone", "utc_offset_seconds"}
        assert {k: v for k, v in first.items() if k not in volatile} == {
            k: v for k, v in second.items() if k not in volatile
        }

    def test_shared_surface_twin_is_not_a_home_home_twin(self, fleet_env):
        """A pair on the shared surface AND one home bank is accepted scope.

        The shared surface is not a home bank: the handshake already probes
        it. So this must NOT trip the home↔home bound — but it must still be
        visible, under its own key, or the census would hide real twins.
        """
        _make_bank(
            fleet_env / "mnemosyne" / "data" / "shared" / "mnemosyne.db",
            [_TWIN + (None,)],
        )
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})

        report = fleet_census.census()

        assert report["home_home_twin_count"] == 0, report
        assert report["bound_holds"] is True
        assert len(report["surface_home_twins"]) == 1, report
        assert report["surface_home_twins"][0]["pair"] == ["cf_aaa_twin", "cf_zzz_twin"]
        # The surface bank is labelled, never path-confused with a home bank.
        assert [b["label"] for b in report["banks"] if b["is_shared"]] == ["shared"]

    def test_within_bank_duplicate_pair_is_reported(self, fleet_env):
        """The per-bank invariant is reported here, not enforced here."""
        _build_fleet(fleet_env, {
            "home-a": [_TWIN + (None,), (_TWIN[1], _TWIN[0], "resolved")],
        })

        report = fleet_census.census()

        assert report["home_home_twin_count"] == 0
        collisions = report["within_bank_collisions"]
        assert len(collisions) == 1, report
        assert collisions[0]["pair"] == ["cf_aaa_twin", "cf_zzz_twin"]
        assert collisions[0]["occurrences"] == 2

    def test_identical_display_names_cannot_collapse_two_banks(self, fleet_env):
        """Regression guard for the bug this census was written against.

        An earlier probe of the same fleet keyed its per-bank map by a
        display label built from the path, and every profile bank plus the
        root bank collapsed onto one label ("home:default"). A pair present
        in five banks then looked like one bank holding it and the bound read
        0. Twins here are keyed by absolute path, so two banks whose
        basenames are identical must still count as two holders.
        """
        _make_bank(fleet_env / "mnemosyne" / "data" / "mnemosyne.db", [_TWIN + (None,)])
        _make_bank(
            fleet_env / "profiles" / "keeper" / "mnemosyne" / "data" / "mnemosyne.db",
            [(_TWIN[1], _TWIN[0], None)],
        )

        report = fleet_census.census()

        assert report["banks_scanned"] == 2
        labels = sorted(b["label"] for b in report["banks"])
        assert labels == [
            "mnemosyne/data/mnemosyne.db",
            "profiles/keeper/mnemosyne/data/mnemosyne.db",
        ], labels
        assert report["home_home_twin_count"] == 1, report
        assert report["home_home_twins"][0]["banks"] == labels

    def test_open_count_tracks_resolution(self, fleet_env):
        _build_fleet(fleet_env, {
            "home-a": [
                ("cf_1", "cf_2", None),
                ("cf_3", "cf_4", None),
                ("cf_5", "cf_6", "newer_wins"),
            ],
        })

        report = fleet_census.census()

        bank = report["banks"][0]
        assert bank["total_conflicts"] == 3
        assert bank["open_conflicts"] == 2
        assert report["total_conflict_rows"] == 3


# --------------------------------------------------------------------------
# 2. Read-only, enforced by the engine
# --------------------------------------------------------------------------

def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_and_counts(path: Path):
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        schema = conn.execute(
            "SELECT type, name, sql FROM sqlite_master ORDER BY type, name"
        ).fetchall()
        counts = conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
    finally:
        conn.close()
    return schema, counts


class TestReadOnly:

    def test_census_writes_nothing_to_the_banks_it_reads(self, fleet_env):
        _build_fleet(fleet_env, {
            "home-a": [_TWIN + (None,)],
            "home-b": [(_TWIN[1], _TWIN[0], None)],
        })
        banks = sorted(fleet_env.rglob("*.db"))
        assert len(banks) == 2

        before = {
            bank: (
                _schema_and_counts(bank),
                _file_digest(bank),
                bank.stat().st_mtime_ns,
            )
            for bank in banks
        }

        report = fleet_census.census()
        assert report["home_home_twin_count"] == 1

        after = {
            bank: (
                _schema_and_counts(bank),
                _file_digest(bank),
                bank.stat().st_mtime_ns,
            )
            for bank in banks
        }
        assert after == before, "census mutated a bank it read"
        # No WAL/SHM sidecars were conjured next to the banks either.
        assert not list(fleet_env.rglob("*.db-wal"))
        assert not list(fleet_env.rglob("*.db-shm"))

    def test_census_succeeds_against_a_write_protected_bank(self, fleet_env):
        """mode=ro, proven by making the file unwritable for the engine.

        If the census ever opened a bank read-write, SQLite would need to
        take a lock and (for a WAL bank) create sidecars; against a
        read-only file that path fails. Skipped for a privileged runner,
        where the mode bit cannot bind the engine.
        """
        import os

        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("running privileged: the read-only mode bit cannot bind")

        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        bank = fleet_env / "home-a" / "mnemosyne.db"
        bank.chmod(0o444)
        try:
            report = fleet_census.census()
            assert report["banks_scanned"] == 1
            assert report["unreadable"] == []
            assert report["banks"][0]["total_conflicts"] == 1
        finally:
            bank.chmod(0o644)

    def test_unreadable_bank_is_reported_not_fatal(self, fleet_env):
        """A corrupt .db is a finding, not a crash: the census is a discovery
        instrument and must still report the banks it CAN read."""
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        garbage = fleet_env / "home-b" / "mnemosyne.db"
        garbage.parent.mkdir(parents=True, exist_ok=True)
        garbage.write_bytes(b"this is not a sqlite file" * 40)

        report = fleet_census.census()

        assert report["banks_scanned"] == 1
        assert len(report["unreadable"]) == 1, report
        assert report["unreadable"][0]["path"] == str(garbage)
        # Fail closed on partial data: the bound is UNMEASURED, not held.
        # Reading bound_holds=True here would report a maintained bound on a
        # census that could not see one of its banks.
        assert report["bound_holds"] is False, report

    def test_bound_holds_true_on_a_fully_readable_fleet_with_no_twins(self, fleet_env):
        """The carve-out the rule above must not swallow: a complete read
        with no home↔home twin reports the bound as holding."""
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})

        report = fleet_census.census()

        assert report["unreadable"] == []
        assert report["home_home_twin_count"] == 0
        assert report["bound_holds"] is True, report

    def test_non_bank_db_is_not_counted(self, fleet_env):
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        other = fleet_env / "home-b" / "mnemosyne.db"
        other.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(other))
        conn.execute("CREATE TABLE unrelated (id INTEGER)")
        conn.commit()
        conn.close()

        report = fleet_census.census()
        assert report["banks_scanned"] == 1

    def test_empty_db_file_is_skipped(self, fleet_env):
        empty = fleet_env / "home-b" / "mnemosyne.db"
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.touch()

        report = fleet_census.census()
        assert report["banks_scanned"] == 0
        assert report["unreadable"] == []

    def test_cache_and_backup_dirs_are_pruned(self, fleet_env):
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        for dirname in ("backups", "cache", ".cache"):
            _make_bank(fleet_env / dirname / "old" / "mnemosyne.db", [_TWIN + (None,)])

        report = fleet_census.census()

        # Had the copies been counted, this would be a 4-bank home↔home twin.
        assert report["banks_scanned"] == 1
        assert report["home_home_twin_count"] == 0


# --------------------------------------------------------------------------
# 3. The clock seam
# --------------------------------------------------------------------------

def test_stamp_carries_both_clocks_each_named_with_its_zone(fleet_env):
    report = fleet_census.census()

    assert "read_at_utc" in report
    assert "read_at_local" in report
    assert report["local_zone"], "local zone abbreviation missing"

    # Both are naive (no offset suffix): the seam is that the STORED clocks
    # are naive, and a stamp that carried an offset would invite the very
    # comparison mix-up the seam warns about.
    utc_stamp = report["read_at_utc"]
    local_stamp = report["read_at_local"]
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?", utc_stamp), utc_stamp
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?", local_stamp), local_stamp

    parsed_utc = datetime.fromisoformat(utc_stamp)
    parsed_local = datetime.fromisoformat(local_stamp)
    # The two stamps must differ by exactly the reported offset: proof they
    # are the same instant read through two zones, not one clock duplicated.
    delta = (parsed_local - parsed_utc).total_seconds()
    assert delta == report["utc_offset_seconds"], report


# --------------------------------------------------------------------------
# 4. Exercised from the sleep-time path
# --------------------------------------------------------------------------

class TestSleepPath:

    def test_single_session_sleep_emits_the_census(self, fleet_env, beam_db):
        _build_fleet(fleet_env, {
            "home-a": [_TWIN + (None,)],
            "home-b": [(_TWIN[1], _TWIN[0], None)],
        })
        beam = BeamMemory(session_id="s1", db_path=beam_db)
        _seed_old_wm(beam_db, "s1", n=2)

        result = beam.sleep(dry_run=True)

        assert result["status"] == "dry_run"
        census = result["fleet_conflict_census"]
        assert census["kind"] == "fleet_conflict_census"
        assert census["home_home_twin_count"] == 1, census
        assert census["bound_holds"] is False

    def test_no_op_sleep_still_emits_the_census(self, fleet_env, beam_db):
        """A pass with nothing to consolidate still ran a sleep, so it still
        owes the bound. Without this the bound would only refresh on busy
        days."""
        _build_fleet(fleet_env, {
            "home-a": [_TWIN + (None,)],
            "home-b": [(_TWIN[1], _TWIN[0], None)],
        })
        beam = BeamMemory(session_id="s1", db_path=beam_db)

        result = beam.sleep(dry_run=True)

        assert result["status"] == "no_op"
        assert result["fleet_conflict_census"]["home_home_twin_count"] == 1

    def test_sleep_all_sessions_takes_one_census_for_the_pass(self, fleet_env, beam_db):
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        beam = BeamMemory(session_id="s1", db_path=beam_db)
        _seed_old_wm(beam_db, "s1", n=2)
        _seed_old_wm(beam_db, "s2", n=2)

        result = beam.sleep_all_sessions(dry_run=True)

        assert result["status"] == "dry_run"
        assert result["fleet_conflict_census"]["home_home_twin_count"] == 0
        assert result["session_results"], "fixture produced no sessions to consolidate"
        for session_result in result["session_results"]:
            assert "fleet_conflict_census" not in session_result, (
                "sleep_all_sessions took the census per session; the pass owes "
                "one census, not one per session"
            )

    def test_memory_facade_sleep_exposes_the_census(self, fleet_env, tmp_path, monkeypatch):
        """The public wrapper (what the MCP/CLI sleep handlers call) carries
        the census through unchanged."""
        from mnemosyne.core.memory import Mnemosyne

        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        mem = Mnemosyne(session_id="s1", db_path=tmp_path / "facade.db")

        result = mem.sleep(dry_run=True)

        assert result["fleet_conflict_census"]["bound_holds"] is True
        assert result["fleet_conflict_census"]["banks_scanned"] == 1

    def test_census_failure_does_not_fail_the_sleep(self, fleet_env, beam_db, monkeypatch):
        """A broken fleet must not turn a completed consolidation into an
        error: the census is a diagnostic, not a consolidation step."""
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        beam = BeamMemory(session_id="s1", db_path=beam_db)
        _seed_old_wm(beam_db, "s1", n=2)

        def _boom(root=None, banks=None):
            raise RuntimeError("census exploded")

        monkeypatch.setattr(fleet_census, "census", _boom)
        # The method resolves the module attribute at call time.
        result = beam.sleep(dry_run=True)

        assert result["status"] == "dry_run"
        assert result["fleet_conflict_census"] == {"error": "RuntimeError"}

    def test_env_opt_out_skips_the_walk(self, fleet_env, beam_db, monkeypatch):
        """``MNEMOSYNE_FLEET_CENSUS=0`` must stop the full-fleet walk.

        ``sleep()`` runs on every pass and the walk's cost tracks the size of
        the tree, so an operator needs a switch that does not require a code
        change. The switch is read at CALL time: a sleep pass that cached it
        would keep walking after the operator turned it off.
        """
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        beam = BeamMemory(session_id="s1", db_path=beam_db)
        _seed_old_wm(beam_db, "s1", n=2)

        def _boom(root=None, banks=None):
            raise AssertionError("census walk ran despite the env opt-out")

        monkeypatch.setattr(fleet_census, "census", _boom)

        # On by default: the same beam, the same fixture, walks the fleet.
        assert fleet_census.census_enabled() is True
        monkeypatch.setenv(fleet_census.CENSUS_ENV, "0")
        result = beam.sleep(dry_run=True)

        assert result["status"] == "dry_run"
        assert "fleet_conflict_census" not in result, result

        # And back on again without a restart — resolved per call.
        monkeypatch.setenv(fleet_census.CENSUS_ENV, "1")
        assert fleet_census.census_enabled() is True

    def test_env_opt_out_also_applies_to_sleep_all_sessions(
            self, fleet_env, beam_db, monkeypatch):
        _build_fleet(fleet_env, {"home-a": [_TWIN + (None,)]})
        beam = BeamMemory(session_id="s1", db_path=beam_db)
        _seed_old_wm(beam_db, "s1", n=2)
        monkeypatch.setenv(fleet_census.CENSUS_ENV, "off")

        result = beam.sleep_all_sessions(dry_run=True)

        assert "fleet_conflict_census" not in result, result


# --------------------------------------------------------------------------
# 5. The env opt-out's own semantics
# --------------------------------------------------------------------------

class TestCensusEnabled:

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", " off "])
    def test_falsy_values_disable(self, value, monkeypatch):
        monkeypatch.setenv(fleet_census.CENSUS_ENV, value)
        assert fleet_census.census_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "maybe", ""])
    def test_anything_else_stays_on(self, value, monkeypatch):
        """The default must be ON: the census is what keeps R6's bound
        measured, so only an explicit off-switch may stop it."""
        monkeypatch.setenv(fleet_census.CENSUS_ENV, value)
        assert fleet_census.census_enabled() is True

    def test_unset_means_on(self, monkeypatch):
        monkeypatch.delenv(fleet_census.CENSUS_ENV, raising=False)
        assert fleet_census.census_enabled() is True