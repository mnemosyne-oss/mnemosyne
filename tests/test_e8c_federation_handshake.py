"""Tests for E8c — the federation handshake in the upsert path.

Policy ``policy:conflict-disposition-2026-09-26`` v3 (why seat), R3 §(e)–(i),
R7, R8. The handshake exists because the v2 rule — "probe the shared surface
only; home↔home twins are accepted scope" — was falsified by E8b's first live
run: ONE home↔home twin held by FIVE non-shared banks, with an earlier
label-keyed probe reporting 0 while it was live. So these tests pin:

  1. probe SCOPE is the fleet census's own enumeration, in R7's order
     (shared, root, then every other non-shared bank), never the upserting
     bank — and the probe set equals the census's bank set, so scope cannot
     drift from the bound the census maintains;
  2. probe RESULTS accumulate into ONE chain and ONE audit entry per upsert
     (not one entry per probe);
  3. canonical preference is deterministic (shared > root > oldest profile);
  4. a probe failure FAILS CLOSED: no home insert, and a
     ``federation_probe_failed`` row naming the target;
  5. the per-probe budget is 250 ms with 1 retry / 500 ms backoff, and the
     total wall-clock cap is unset unless asked for;
  6. probed banks are NOT written to (digest + mtime + schema + counts);
  7. the gate is exercised from the real upsert path
     (``VeracityConsolidator.consolidate_fact`` -> ``_record_conflict``), not
     only by direct import.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import pytest

from mnemosyne.core import federation_handshake as fh
from mnemosyne.core import fleet_census
from mnemosyne.core.veracity_consolidation import VeracityConsolidator, compute_fact_id

# Canonical conflicts DDL, copied from the source of truth in
# mnemosyne/core/veracity_consolidation.py. The handshake only READS this
# table; the fixture owns creating it.
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

# Deliberately NOT pre-sorted: the detector does not canonicalize orientation
# (15 of 32 live rows violated fact_a_id < fact_b_id on 2026-09-26), so a
# probe that compared raw (a, b) would miss the swapped occurrence — exactly
# the failure the order-normalized key exists to prevent.
PAIR = ("cf_zzz_twin", "cf_aaa_twin")

SUBJECT, PREDICATE = "Astraea", "is"
OBJECT_OLD, OBJECT_NEW = "researching", "continuous"

#: The conflict pair the consolidation path actually produces: the detector
#: records ``(new_id, existing_id)`` when the second object lands. Integration
#: tests must plant THIS pair, not an arbitrary one, or they prove nothing.
UPSERT_PAIR = (
    compute_fact_id(SUBJECT, PREDICATE, OBJECT_NEW),
    compute_fact_id(SUBJECT, PREDICATE, OBJECT_OLD),
)


def _noop_sleep(_seconds: float) -> None:
    return None


class _FakeClock:
    """A monotonic clock that only advances when the test says so.

    The budget tests run a ~0.2 s total against a REAL clock: a scheduler
    stall before the event under test can consume the budget early and change
    WHICH guard fires (the pre-probe skip instead of the post-loop recheck),
    so the assertion can fail without the code under test misbehaving. Patch
    it in place of ``fh.time.monotonic`` and advance it from inside the
    injected ``sleep`` callback to make the budget deterministic.
    """

    def __init__(self):
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _make_bank(path: Path, rows=()) -> Path:
    """Create a bank holding a conflicts table.

    rows: iterable of ``(fact_a_id, fact_b_id, resolution)`` or
    ``(fact_a_id, fact_b_id, resolution, created_at)``. ``resolution`` None
    means an OPEN conflict.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(_CONFLICTS_DDL)
        payload = []
        for row in rows:
            if len(row) == 4:
                payload.append((row[0], row[1], "contradiction", row[2], row[3]))
            else:
                payload.append((row[0], row[1], "contradiction", row[2], None))
        for a, b, kind, resolution, created in payload:
            if created is None:
                conn.execute(
                    "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type, resolution) "
                    "VALUES (?, ?, ?, ?)", (a, b, kind, resolution))
            else:
                conn.execute(
                    "INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type, resolution, created_at) "
                    "VALUES (?, ?, ?, ?, ?)", (a, b, kind, resolution, created))
        conn.commit()
    finally:
        conn.close()
    return path


def _make_bank_without_pair_columns(path: Path) -> Path:
    """A bank whose ``conflicts`` table exists but has no pair columns.

    Passes the bank predicate (``_check_bank`` only asks whether the table
    exists) and fails the probe itself — the shape needed to exercise a probe
    that starts and then takes real time, rather than being skipped at scope.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE conflicts (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    return path


def _shared(fleet: Path) -> Path:
    return fleet / "mnemosyne" / "data" / "shared" / "mnemosyne.db"


def _root_bank(fleet: Path) -> Path:
    return fleet / "mnemosyne" / "data" / "mnemosyne.db"


def _profile(fleet: Path, name: str) -> Path:
    return fleet / "profiles" / name / "mnemosyne" / "data" / "mnemosyne.db"


def _build_fleet(fleet: Path, *, shared_rows=(), root_rows=(), profiles=()) -> Path:
    """Build a fleet in the layout the live bound was measured against."""
    _make_bank(_shared(fleet), shared_rows)
    _make_bank(_root_bank(fleet), root_rows)
    for name, rows in profiles:
        _make_bank(_profile(fleet, name), rows)
    return fleet


def _conflict_rows(bank: Path):
    conn = sqlite3.connect(f"{bank.as_uri()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM conflicts ORDER BY id")]
    finally:
        conn.close()


def _audit_rows(bank: Path):
    conn = sqlite3.connect(f"{bank.as_uri()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM memory_audit_events ORDER BY event_id")]
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    for row in rows:
        row["metadata"] = json.loads(row["metadata_json"]) if row.get("metadata_json") else {}
    return rows


def _fact_count(bank: Path) -> int:
    conn = sqlite3.connect(f"{bank.as_uri()}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT COUNT(*) FROM consolidated_facts").fetchone()[0]
    finally:
        conn.close()


def _fingerprint(bank: Path) -> dict:
    """Schema + counts + bytes + mtime, so a write anywhere shows up."""
    conn = sqlite3.connect(f"{bank.as_uri()}?mode=ro", uri=True)
    try:
        tables = sorted(
            row[0] for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL")
        )
        conflicts = conn.execute("SELECT COUNT(*) FROM conflicts").fetchone()[0]
    finally:
        conn.close()
    stat = bank.stat()
    digest = hashlib.sha256(bank.read_bytes()).hexdigest()
    return {
        "tables": tables,
        "conflicts": conflicts,
        "sha256": digest,
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """No ambient Mnemosyne config may reach these tests.

    Unset means "derive the fleet root from the upserting bank", which is what
    the integration cases exercise; an ambient override from the host would
    silently point them at a different fleet.
    """
    for name in (
        fleet_census.FLEET_ROOT_ENV,
        "MNEMOSYNE_SHARED_DB_PATH",
        "MNEMOSYNE_HOME",
        fh.HANDSHAKE_ENV,
        fh.PROBE_TIMEOUT_ENV,
        fh.TOTAL_BUDGET_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    return None


# ---------------------------------------------------------------------------
# 1. Probe scope — the census's enumeration, in R7's order (R3 §(e))
# ---------------------------------------------------------------------------
class TestProbeScope:

    def test_probes_shared_rank0_root_rank1_then_every_other_bank(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            profiles=[("alpha", ()), ("beta", ()), ("gamma", ())],
        )
        home = _profile(fleet, "alpha")

        decision = fh.probe_fleet(PAIR, home, fleet_root=fleet, sleep=_noop_sleep)

        assert decision.probed_banks == [
            str(_shared(fleet).resolve()),
            str(_root_bank(fleet).resolve()),
            str(_profile(fleet, "beta").resolve()),
            str(_profile(fleet, "gamma").resolve()),
        ]
        assert str(home.resolve()) not in decision.probed_banks
        assert decision.proceed is True

    def test_probe_set_equals_the_census_bank_set(self, tmp_path):
        """R3 §(e): v3 reuses the census enumeration so scope cannot drift."""
        fleet = _build_fleet(
            tmp_path / "fleet",
            shared_rows=[PAIR + (None,)],
            root_rows=[PAIR[::-1] + (None,)],
            profiles=[("alpha", [PAIR + (None,)]), ("beta", ()), ("gamma", ())],
        )
        home = _profile(fleet, "alpha")
        census = fleet_census.census(root=fleet)
        census_banks = {entry["path"] for entry in census["banks"]}

        decision = fh.probe_fleet(PAIR, home, fleet_root=fleet, sleep=_noop_sleep)

        assert set(decision.probed_banks) | {str(home.resolve())} == census_banks
        # The chain's length equals the census's own holder count for the
        # pair, over the surface+home twin set it reports.
        twin = next(t for t in census["surface_home_twins"]
                    if t["pair"] == sorted(PAIR))
        assert len(decision.chain) == len(twin["banks"]) - 1  # minus the home bank

    def test_standalone_bank_has_no_peers_and_inserts_anyway(self, tmp_path):
        """The gate is on by default, but a bank outside a fleet has nothing
        to probe — which is what makes default-on safe."""
        home = _make_bank(tmp_path / "standalone" / "mnemosyne.db", [])

        decision = fh.probe_fleet(PAIR, home, sleep=_noop_sleep)

        assert decision.probed_banks == []
        assert decision.chain == []
        assert decision.proceed is True

    def test_absent_shared_surface_and_root_bank_are_out_of_scope_not_failures(
            self, tmp_path):
        """A fleet with no shared surface and no root bank must not refuse.

        Their paths are derived from the fleet root, so a fleet that simply
        never created them would otherwise fail every insert closed. R3 §(e)
        probes every LIVE bank; a path that is not a file is not live.
        """
        fleet = tmp_path / "fleet"
        _make_bank(_profile(fleet, "alpha"), [])
        _make_bank(_profile(fleet, "beta"), [])

        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        assert decision.refused is False
        assert decision.proceed is True
        assert decision.probed_banks == [str(_profile(fleet, "beta").resolve())]


# ---------------------------------------------------------------------------
# 2. Chain accumulation and one audit entry per upsert (R3 §(f))
# ---------------------------------------------------------------------------
class TestChain:

    def test_four_holders_produce_a_four_long_chain_and_one_audit_entry(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            shared_rows=[],
            root_rows=[PAIR + (None,)],
            profiles=[
                ("alpha", ()),
                ("beta", [PAIR[::-1] + (None,)]),
                ("gamma", [PAIR + (None,)]),
                # fourth holder, non-open (R7 counts closed holders too)
                ("delta", [PAIR + ("superseded_by_x",)]),
            ],
        )
        home = _profile(fleet, "alpha")

        decision = fh.probe_fleet(PAIR, home, fleet_root=fleet, sleep=_noop_sleep)

        assert len(decision.chain) == 4
        assert decision.proceed is False
        assert decision.federated is True
        assert {entry["bank_path"] for entry in decision.chain} == {
            str(_root_bank(fleet).resolve()),
            str(_profile(fleet, "beta").resolve()),
            str(_profile(fleet, "gamma").resolve()),
            str(_profile(fleet, "delta").resolve()),
        }
        # Chain entries carry the policy's field names for both spellings
        # (R3 §(f) says `resolution`, R7 §(d) says `resolution_status`).
        for entry in decision.chain:
            assert entry["resolution"] == entry["resolution_status"]
            assert set(entry) >= {"id", "bank_path", "resolution_status", "created_at"}

        events = decision.audit_events()
        assert [e["action"] for e in events] == ["federation_probe"]
        metadata = events[0]["metadata"]
        assert len(metadata["federation_chain"]) == 4
        assert metadata["chain_length"] == 4
        assert metadata["canonical_preference"] == "shared > root > oldest profile (R7)"
        assert metadata["proceeded"] is False
        assert metadata["refused"] is False

    def test_one_audit_entry_per_upsert_not_one_per_probe(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            profiles=[("alpha", ()), ("beta", ()), ("gamma", ()), ("delta", ())],
        )
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        assert decision.probes_run == 5  # shared, root, beta, gamma, delta
        assert len(decision.audit_events()) == 1

    def test_orientation_of_the_caller_pair_cannot_change_the_outcome(self, tmp_path):
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])
        _make_bank(_shared(fleet), [PAIR + (None,)])  # ensure surface holds it

        forward = fh.probe_fleet(PAIR, _profile(fleet, "alpha"),
                                 fleet_root=fleet, sleep=_noop_sleep)
        reverse = fh.probe_fleet(PAIR[::-1], _profile(fleet, "alpha"),
                                 fleet_root=fleet, sleep=_noop_sleep)

        assert forward.pair == reverse.pair
        assert forward.federated_to == reverse.federated_to


# ---------------------------------------------------------------------------
# 3. R7 canonical preference
# ---------------------------------------------------------------------------
class TestCanonicalPreference:

    def test_shared_wins_over_root_and_profiles(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            shared_rows=[PAIR + (None,)],
            root_rows=[PAIR + (None,)],
            profiles=[("alpha", ()), ("beta", [PAIR + (None,)])],
        )
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        shared_id = _conflict_rows(_shared(fleet))[0]["id"]
        assert decision.federated_to == f"{shared_id}@{_shared(fleet).resolve()}"

    def test_root_wins_when_shared_does_not_hold_it(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            shared_rows=[],
            root_rows=[PAIR + (None,)],
            profiles=[("alpha", ()), ("beta", [PAIR + (None,)])],
        )
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        root_id = _conflict_rows(_root_bank(fleet))[0]["id"]
        assert decision.federated_to == f"{root_id}@{_root_bank(fleet).resolve()}"

    def test_oldest_profile_wins_by_created_at(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            profiles=[
                ("alpha", ()),
                ("beta", [PAIR + (None, "2026-09-29 09:08:44")]),
                ("gamma", [PAIR[::-1] + (None, "2026-09-29 09:52:45")]),
            ],
        )
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        beta_id = _conflict_rows(_profile(fleet, "beta"))[0]["id"]
        assert decision.federated_to == f"{beta_id}@{_profile(fleet, 'beta').resolve()}"

    def test_null_created_at_is_not_treated_as_oldest(self, tmp_path):
        """Unknown age must not win canonicality over a known-older row."""
        fleet = _build_fleet(
            tmp_path / "fleet",
            profiles=[
                ("alpha", ()),
                ("beta", [PAIR + (None, "2026-09-29 09:08:44")]),
                ("gamma", [PAIR + (None,)]),
            ],
        )
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        beta_id = _conflict_rows(_profile(fleet, "beta"))[0]["id"]
        assert decision.federated_to == f"{beta_id}@{_profile(fleet, 'beta').resolve()}"


# ---------------------------------------------------------------------------
# 4. Fail-closed on probe failure (R3 §(g)) and the time budget (R8)
# ---------------------------------------------------------------------------
class TestFailClosed:

    def _broken_peer(self, fleet: Path) -> Path:
        """A non-empty ``*.db`` file that cannot be read as a database.

        A *file* (not a directory) with non-sqlite bytes: the census walk
        enumerates it, ``sqlite3`` opens it lazily, and the schema read then
        raises ``DatabaseError('file is not a database')`` — a real probe
        failure, which R3 §(g) makes a refusal.
        """
        broken = fleet / "profiles" / "broken" / "mnemosyne" / "data" / "mnemosyne.db"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"this is not a sqlite database\n")
        return broken

    def test_unreadable_peer_refuses_and_names_the_target(self, tmp_path):
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ())])
        broken = self._broken_peer(fleet)

        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        assert decision.refused is True
        assert decision.proceed is False
        assert [f["probe_target"] for f in decision.failed] == [str(broken.resolve())]
        assert decision.failed[0]["error_class"]

        failed_events = [e for e in decision.audit_events()
                         if e["action"] == "federation_probe_failed"]
        assert len(failed_events) == 1
        metadata = failed_events[0]["metadata"]
        assert metadata["probe_target"] == str(broken.resolve())
        assert metadata["pair"] == sorted(PAIR)
        assert metadata["upserting_bank"] == str(_profile(fleet, "alpha").resolve())

    def test_retry_is_bounded_to_one_backoff_sleep(self, tmp_path):
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ())])
        self._broken_peer(fleet)
        slept: list = []

        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet,
            backoff_s=0.5, sleep=slept.append)

        assert decision.refused is True
        assert slept == [0.5]

    def test_refusal_wins_even_when_a_holder_was_already_found(self, tmp_path):
        """A found holder does not excuse a failed probe (R3 §(g))."""
        fleet = _build_fleet(
            tmp_path / "fleet",
            root_rows=[PAIR + (None,)],
            profiles=[("alpha", ())],
        )
        self._broken_peer(fleet)

        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        assert decision.refused is True
        assert decision.chain  # observed, reported, but not acted on
        assert len(decision.chain) == 1

    def test_probe_timeout_is_classified_as_timeout(self):
        assert fh._classify_error(sqlite3.OperationalError("interrupted")) == "timeout"
        assert fh._classify_error(TimeoutError("x")) == "timeout"

    def test_total_budget_defaults_to_a_bounded_lock_hold(self, monkeypatch):
        """R8's total is CAPPED by default.

        The probe runs while the home bank's write lock is held, so an
        unbounded sequence would hold that lock for N x 250 ms (4.75 s today).
        The cap bounds the LOCK HOLD; exceeding it is fail-closed and R1's next
        detection cycle retries.
        """
        assert fh.total_budget_seconds() == fh.DEFAULT_TOTAL_BUDGET_S
        monkeypatch.setenv(fh.TOTAL_BUDGET_ENV, "0")
        assert fh.total_budget_seconds() is None
        monkeypatch.setenv(fh.TOTAL_BUDGET_ENV, "none")
        assert fh.total_budget_seconds() is None
        monkeypatch.setenv(fh.TOTAL_BUDGET_ENV, "750")
        assert fh.total_budget_seconds() == 0.75

    def test_exceeding_the_total_budget_refuses_every_remaining_probe(self, tmp_path):
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet,
            total_budget_s=1e-9, sleep=_noop_sleep)

        assert decision.refused is True
        assert decision.probes_run == 0
        assert {f["error_class"] for f in decision.failed} == {"total_budget_exceeded"}
        assert len(decision.failed) == 3  # shared, root, beta — all refused unfunded
        assert [e["action"] for e in decision.audit_events()] == (
            ["federation_probe_failed"] * 3 + ["federation_probe"])

    def test_an_explicit_zero_budget_means_unbounded(self, tmp_path):
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet,
            total_budget_s=0, sleep=_noop_sleep)
        assert decision.refused is False
        assert decision.probes_run == 3

    def test_scope_resolution_runs_inside_the_total_budget(self, tmp_path,
                                                           monkeypatch):
        """The clock starts BEFORE scope, so scope backoff is budgeted.

        Scope decides bank liveness with the same retry/backoff budget a
        probe uses, and it does so under the home bank's write lock. If the
        deadline starts after scope returns, a candidate set full of
        unreadable banks spends ``retries x backoff`` per bank outside the
        cap the operator configured.
        """
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ())])
        broken = self._broken_peer(fleet)
        good = _profile(fleet, "beta")
        _make_bank(good, [])

        clock = _FakeClock()
        monkeypatch.setattr(fh.time, "monotonic", clock)

        def _slow(_seconds: float) -> None:
            clock.advance(0.3)  # one backoff outruns the 0.2 s budget

        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet,
            banks=[broken, good], total_budget_s=0.2, backoff_s=0.1,
            sleep=_slow)

        assert decision.refused is True
        # The good bank sorts after the broken one; if scope were unbudgeted
        # it would have been probed, spending more than the cap.
        assert decision.probes_run == 0, decision.probed_banks
        assert "total_budget_exceeded" in {f["error_class"] for f in decision.failed}

    def test_a_last_probe_overrunning_the_budget_refuses_the_upsert(self, tmp_path, monkeypatch):
        """The deadline is rechecked AFTER the loop, on a probe that SUCCEEDED.

        The pre-probe guard only asks whether a probe may START. A last
        probe that started inside the budget and finished past it would
        otherwise leave ``proceed`` true, so the caller inserts a conflict
        row whose probe sequence overran R8's cap. The probe here SUCCEEDS
        by construction — it advances the fake clock past the deadline and
        reports a holder — so only the post-loop recheck can catch it: this
        test isolates that
        check instead of reaching it through the retry path (retry sleeps
        now respect the absolute deadline, so the old slow-backoff shape
        cannot manufacture an overrun anymore).
        """
        fleet = _build_fleet(
            tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])

        clock = _FakeClock()
        monkeypatch.setattr(fh.time, "monotonic", clock)

        def _overrun_probe(bank, pair, **_kwargs):
            clock.advance(0.35)  # started before the deadline; ends past it
            return [{"id": 1, "bank_path": str(bank), "resolution": None,
                     "resolution_status": None, "created_at": "2026-09-29T00:00:00"}], None

        monkeypatch.setattr(fh, "_probe_one", _overrun_probe)
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet,
            banks=[_profile(fleet, "beta")],
            # Absent explicit surface/root paths: a missing file is not a
            # bank and is skipped without a failure, so beta is the ONLY
            # probeable peer. Without this, the shared probe is the one that
            # overruns and the in-loop guard then refuses root and beta
            # unfunded — assertions that would still pass with the post-loop
            # recheck deleted.
            shared_db=tmp_path / "absent-shared.db",
            root_db=tmp_path / "absent-root.db",
            total_budget_s=0.2, sleep=_noop_sleep)

        assert decision.refused is True
        assert decision.proceed is False
        assert decision.probes_run == 1
        # The overrun is reported once per unfunded candidate, and only as
        # total_budget_exceeded — a successful probe never authorizes the
        # insert past the cap.
        assert {f["error_class"] for f in decision.failed} == {"total_budget_exceeded"}
        assert len(decision.failed) == 1  # only the post-loop recheck fires
        assert decision.chain

    def test_probe_one_deadline_skips_a_backoff_that_cannot_fit(self, tmp_path):
        """R8 §(b) bounds the whole retry sequence, not each attempt alone.

        A 5 s per-probe budget inside a deadline with 100 ms left may get
        ONE attempt: the 500 ms backoff cannot land before the deadline, so
        ``_probe_one`` must stop and report the probe's own failure instead
        of sleeping past the cap while the caller holds the home write lock.
        """
        broken = _make_bank_without_pair_columns(
            tmp_path / "overrun-peer" / "mnemosyne.db")
        slept: list = []

        hits, failure = fh._probe_one(
            broken, PAIR, timeout_s=5.0, retries=2, backoff_s=0.5,
            sleep=slept.append,
            deadline=time.monotonic() + 0.1, total_budget_s=0.1)

        assert hits == []
        assert failure is not None
        assert slept == []  # every backoff that would overrun: skipped

    def test_check_bank_retrying_reports_budget_stop_never_a_silent_skip(
            self, tmp_path):
        """``(False, None)`` means "not a bank" — a budget stop must not wear it.

        A candidate whose retries ran out of the absolute deadline was never
        DECIDED; skipping it silently is the fail-open R3 §(g) forbids, so
        the helper records ``total_budget_exceeded`` instead.
        """
        broken = tmp_path / "unreadable.db"
        broken.write_bytes(b"this is not a sqlite database\n")

        should_probe, failure = fh._check_bank_retrying(
            broken, retries=1, backoff_s=0.5, sleep=_noop_sleep,
            deadline=time.monotonic() - 1.0,  # already out of budget
            total_budget_s=0.2)

        assert should_probe is False
        assert failure is not None
        assert failure["error_class"] == "total_budget_exceeded"
        assert failure["probe_target"] == str(broken)

    def test_a_locked_peer_refuses_within_the_probe_budget(self, tmp_path):
        """A held write lock must not outlast the probe budget.

        The progress handler bounds SQLite's virtual-machine work only; a
        busy-wait for a lock is not VM work and the handler is never called
        during one, so Python's 5 s connect default used to be the real
        bound on a locked peer regardless of the configured budget.
        """
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])
        locked = _profile(fleet, "beta")
        blocker = sqlite3.connect(str(locked))
        try:
            blocker.execute("BEGIN EXCLUSIVE")
            started = time.monotonic()
            decision = fh.probe_fleet(
                PAIR, _profile(fleet, "alpha"), fleet_root=fleet,
                per_probe_timeout_s=0.05, retries=0, sleep=_noop_sleep)
            elapsed = time.monotonic() - started
        finally:
            blocker.rollback()
            blocker.close()

        assert decision.refused is True
        assert elapsed < 1.0, f"lock wait escaped the budget: {elapsed:.2f}s"
        assert decision.proceed is False

    def test_probe_one_bounds_its_own_connect_lock_wait(self, tmp_path):
        """The same bound, asserted at the connection that owns it."""
        locked = _make_bank(tmp_path / "locked" / "mnemosyne.db", [])
        blocker = sqlite3.connect(str(locked))
        try:
            blocker.execute("BEGIN EXCLUSIVE")
            started = time.monotonic()
            hits, failure = fh._probe_one(
                locked, PAIR, timeout_s=0.05, retries=0, backoff_s=0.0,
                sleep=_noop_sleep)
            elapsed = time.monotonic() - started
        finally:
            blocker.rollback()
            blocker.close()

        assert hits == []
        assert failure is not None
        assert elapsed < 1.0, f"lock wait escaped the per-probe budget: {elapsed:.2f}s"

    def test_per_probe_budget_defaults_to_250ms(self):
        assert fh.per_probe_timeout_seconds() == 0.25

    def test_millisecond_env_vars_are_converted_to_seconds(self, monkeypatch):
        """The ``*_MS`` suffix is the fleet's convention (``MNEMOSYNE_BUSY_TIMEOUT_MS``,
        the E8b census), so the unit is converted rather than read as seconds —
        a 1000x budget error otherwise."""
        monkeypatch.setenv(fh.PROBE_TIMEOUT_ENV, "250")
        assert fh.per_probe_timeout_seconds() == 0.25
        monkeypatch.setenv(fh.PROBE_TIMEOUT_ENV, "50")
        assert fh.per_probe_timeout_seconds() == 0.05
        monkeypatch.setenv(fh.TOTAL_BUDGET_ENV, "2500")
        assert fh.total_budget_seconds() == 2.5

    def test_infinite_and_unusable_durations_are_refused(self, monkeypatch):
        """Configured durations are validated, not coerced.

        This replaces an earlier pin of the old semantics, where a
        non-positive or non-numeric ``MNEMOSYNE_FEDERATION_PROBE_TIMEOUT_MS``
        fell back to the default with a warning. That fallback hid typos,
        and ``value > 0`` let ``inf`` through — an infinite per-probe
        timeout removes the deadline entirely, and this knob has no
        documented opt-out that says so.
        """
        for bad in ("inf", "-inf", "Infinity", "nan", "0", "-1", "", "250ms", "1e400"):
            monkeypatch.setenv(fh.PROBE_TIMEOUT_ENV, bad)
            with pytest.raises(ValueError) as excinfo:
                fh.per_probe_timeout_seconds()
            assert fh.PROBE_TIMEOUT_ENV in str(excinfo.value)
            assert repr(bad) in str(excinfo.value)

        # The total budget has unbounded SENTINELS ("0"/"none"/"off"/""), so
        # its unusable set is the rest: garbage, non-finite, and negatives.
        for bad in ("inf", "-inf", "Infinity", "nan", "-1", "250ms", "1e400"):
            monkeypatch.setenv(fh.TOTAL_BUDGET_ENV, bad)
            with pytest.raises(ValueError) as excinfo:
                fh.total_budget_seconds()
            assert fh.TOTAL_BUDGET_ENV in str(excinfo.value)

    def test_unbounded_sentinels_still_remove_the_total_cap(self, monkeypatch):
        """The documented opt-out survives the validation tightening."""
        for sentinel in ("0", "none", "off", "false", "unlimited", "unbounded", ""):
            monkeypatch.setenv(fh.TOTAL_BUDGET_ENV, sentinel)
            assert fh.total_budget_seconds() is None, sentinel

    def test_per_probe_has_no_unbounded_sentinel(self, monkeypatch):
        """Only the TOTAL budget has a documented unbounded mode.

        The per-probe timeout is the one that keeps a single locked peer from
        stalling the sequence, so there is no spelling that removes it.
        """
        for sentinel in ("none", "off", "false", "unlimited", "unbounded"):
            monkeypatch.setenv(fh.PROBE_TIMEOUT_ENV, sentinel)
            with pytest.raises(ValueError):
                fh.per_probe_timeout_seconds()

    def test_probe_fleet_validates_its_duration_arguments(self, tmp_path):
        home = _make_bank(tmp_path / "standalone" / "mnemosyne.db", [])
        for bad in (float("inf"), float("nan"), 0, -1.0):
            with pytest.raises(ValueError):
                fh.probe_fleet(PAIR, home, per_probe_timeout_s=bad, sleep=_noop_sleep)
        # The total cap has an unbounded spelling (0/negative); non-finite
        # still raises, because it is neither a cap nor a documented opt-out.
        for bad in (float("inf"), float("nan")):
            with pytest.raises(ValueError):
                fh.probe_fleet(PAIR, home, total_budget_s=bad, sleep=_noop_sleep)
        for unbounded in (0, -1.0):
            assert fh.probe_fleet(
                PAIR, home, total_budget_s=unbounded, sleep=_noop_sleep
            ).refused is False


# ---------------------------------------------------------------------------
# 5. Probed banks are never written to (same standard as E8b)
# ---------------------------------------------------------------------------
class TestNoWrites:

    def test_probed_banks_are_byte_identical_after_a_federated_upsert(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            shared_rows=[UPSERT_PAIR + (None,)],
            root_rows=[UPSERT_PAIR + (None,)],
            profiles=[("alpha", ()), ("beta", [UPSERT_PAIR + (None,)])],
        )
        home = _profile(fleet, "alpha")
        probed = [
            _shared(fleet),
            _root_bank(fleet),
            _profile(fleet, "beta"),
        ]
        before = {str(p): _fingerprint(p) for p in probed}

        # A real upsert through the consolidation path, not just a probe.
        consolidator = VeracityConsolidator(db_path=home)
        try:
            consolidator.consolidate_fact(SUBJECT, PREDICATE, OBJECT_OLD, "stated", "s1")
            consolidator.consolidate_fact(SUBJECT, PREDICATE, OBJECT_NEW, "stated", "s2")
        finally:
            consolidator.close()

        after = {str(p): _fingerprint(p) for p in probed}
        assert after == before
        # And the home bank genuinely refused the row.
        assert _conflict_rows(home) == []
        assert len(_audit_rows(home)) == 1


# ---------------------------------------------------------------------------
# 6. The upsert path itself (integration through consolidate_fact)
# ---------------------------------------------------------------------------
class TestUpsertPath:

    def _upsert_once(self, bank: Path, pair_holder: bool = True) -> None:
        consolidator = VeracityConsolidator(db_path=bank)
        try:
            consolidator.consolidate_fact(SUBJECT, PREDICATE, OBJECT_OLD, "stated", "s1")
            consolidator.consolidate_fact(SUBJECT, PREDICATE, OBJECT_NEW, "stated", "s2")
        finally:
            consolidator.close()

    def test_first_insert_grows_then_the_other_four_are_skipped(self, tmp_path):
        """Criterion: 5 non-shared banks, one row total, four skips."""
        fleet = _build_fleet(
            tmp_path / "fleet",
            profiles=[("alpha", ()), ("beta", ()), ("gamma", ()), ("delta", ())],
        )
        banks = [
            _root_bank(fleet),
            _profile(fleet, "alpha"),
            _profile(fleet, "beta"),
            _profile(fleet, "gamma"),
            _profile(fleet, "delta"),
        ]
        for bank in banks:
            self._upsert_once(bank)

        grown = [bank for bank in banks if _conflict_rows(bank)]
        assert grown == [_root_bank(fleet)]  # first writer wins; R7(b) at root

        pair = (compute_fact_id(SUBJECT, PREDICATE, OBJECT_NEW),
                compute_fact_id(SUBJECT, PREDICATE, OBJECT_OLD))
        key = fleet_census._norm_pair(*pair)

        federated = []
        for bank in banks[1:]:
            audits = _audit_rows(bank)
            probe_entries = [a for a in audits if a["action"] == "federation_probe"]
            assert len(probe_entries) == 1
            metadata = probe_entries[0]["metadata"]
            assert metadata["pair"] == list(key)
            assert metadata["federated_to"] is not None
            assert metadata["proceeded"] is False
            federated.append(metadata["federated_to"])

        root_id = _conflict_rows(_root_bank(fleet))[0]["id"]
        assert set(federated) == {f"{root_id}@{_root_bank(fleet).resolve()}"}
        # One grown row in the whole fleet: the twin cannot form.
        assert sum(len(_conflict_rows(b)) for b in banks) == 1

    def test_probe_failure_writes_a_probe_failed_row_and_no_insert(self, tmp_path):
        """Criterion (c): a failing peer refuses the home upsert."""
        fleet = _build_fleet(
            tmp_path / "fleet",
            profiles=[("alpha", ()), ("beta", ())],
        )
        broken = fleet / "profiles" / "broken" / "mnemosyne" / "data" / "mnemosyne.db"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"this is not a sqlite database\n")
        home = _profile(fleet, "alpha")

        self._upsert_once(home)

        assert _conflict_rows(home) == []
        audits = _audit_rows(home)
        failed = [a for a in audits if a["action"] == "federation_probe_failed"]
        assert [a["metadata"]["probe_target"] for a in failed] == [str(broken.resolve())]
        probe = [a for a in audits if a["action"] == "federation_probe"]
        assert len(probe) == 1
        assert probe[0]["metadata"]["refused"] is True

    def test_transient_peer_failure_recovers_and_the_home_row_is_recorded(
            self, tmp_path, monkeypatch):
        """R8 §(c)'s single retry exists so a transient failure is not a LOST
        conflict: the peer is re-checked, the sequence completes, and the home
        row lands with a non-refused audit entry."""
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])
        home = _profile(fleet, "alpha")

        real = fh._check_bank
        state = {"calls": 0}

        def flaky(path, **kwargs):
            state["calls"] += 1
            if state["calls"] == 1:
                raise sqlite3.OperationalError("database is locked")
            return real(path, **kwargs)

        monkeypatch.setattr(fh, "_check_bank", flaky)
        self._upsert_once(home)

        assert state["calls"] > 1  # the retry actually ran
        assert len(_conflict_rows(home)) == 1
        metadata = _audit_rows(home)[0]["metadata"]
        assert metadata["refused"] is False
        assert metadata["proceeded"] is True

    def test_probe_refusal_refuses_the_conflict_row_not_the_detected_fact(
            self, tmp_path):
        """R3 §(g) refuses the upsert R2 defines — the CONFLICTS insert.

        Rolling back the consolidated fact too would drop THIS bank's own data
        on another bank's account (a peer was unreadable), and the fact is
        detected state, not the duplicate R3 exists to stop. R1's metric stream
        carries the signal into the next detection cycle.
        """
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ())])
        broken = fleet / "profiles" / "broken" / "mnemosyne" / "data" / "mnemosyne.db"
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"this is not a sqlite database\n")
        home = _profile(fleet, "alpha")

        self._upsert_once(home)

        assert _conflict_rows(home) == []
        assert _fact_count(home) == 2  # both detected facts committed
        audits = _audit_rows(home)
        assert [a["action"] for a in audits] == [
            "federation_probe_failed", "federation_probe"]
        assert audits[1]["metadata"]["refused"] is True

    def test_root_insert_grows_while_profiles_skip_and_name_root(self, tmp_path):
        """Criterion (b): twin at root + 2 profiles -> root grows."""
        fleet = _build_fleet(
            tmp_path / "fleet",
            profiles=[("alpha", ()), ("beta", ())],
        )
        self._upsert_once(_root_bank(fleet))
        self._upsert_once(_profile(fleet, "alpha"))
        self._upsert_once(_profile(fleet, "beta"))

        assert len(_conflict_rows(_root_bank(fleet))) == 1
        assert _conflict_rows(_profile(fleet, "alpha")) == []
        assert _conflict_rows(_profile(fleet, "beta")) == []
        root_id = _conflict_rows(_root_bank(fleet))[0]["id"]
        expected = f"{root_id}@{_root_bank(fleet).resolve()}"
        for name in ("alpha", "beta"):
            metadata = _audit_rows(_profile(fleet, name))[0]["metadata"]
            assert metadata["federated_to"] == expected
            assert len(metadata["federation_chain"]) == 1

    def test_shared_holder_is_preferred_from_the_upsert_path(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            shared_rows=[UPSERT_PAIR + (None,)],
            root_rows=[UPSERT_PAIR + (None,)],
            profiles=[("alpha", ())],
        )
        home = _profile(fleet, "alpha")
        self._upsert_once(home)

        assert _conflict_rows(home) == []
        metadata = _audit_rows(home)[0]["metadata"]
        shared_id = _conflict_rows(_shared(fleet))[0]["id"]
        assert metadata["federated_to"] == f"{shared_id}@{_shared(fleet).resolve()}"
        assert len(metadata["federation_chain"]) == 2  # shared + root, in rank order
        assert metadata["federation_chain"][0]["is_shared"] is True

    def test_env_opt_out_restores_legacy_behaviour(self, tmp_path, monkeypatch):
        fleet = _build_fleet(
            tmp_path / "fleet",
            root_rows=[UPSERT_PAIR + (None,)],
            profiles=[("alpha", ())],
        )
        monkeypatch.setenv(fh.HANDSHAKE_ENV, "0")
        home = _profile(fleet, "alpha")

        self._upsert_once(home)

        assert len(_conflict_rows(home)) == 1
        assert _audit_rows(home) == []

    def test_constructor_opt_out_restores_legacy_behaviour(self, tmp_path):
        fleet = _build_fleet(
            tmp_path / "fleet",
            root_rows=[UPSERT_PAIR + (None,)],
            profiles=[("alpha", ())],
        )
        home = _profile(fleet, "alpha")
        consolidator = VeracityConsolidator(db_path=home, federation_probe=False)
        try:
            consolidator.consolidate_fact(SUBJECT, PREDICATE, OBJECT_OLD, "stated", "s1")
            consolidator.consolidate_fact(SUBJECT, PREDICATE, OBJECT_NEW, "stated", "s2")
        finally:
            consolidator.close()

        assert len(_conflict_rows(home)) == 1

    def test_gate_runs_from_a_bank_whose_fleet_root_is_derived(self, tmp_path):
        """No env, no explicit root: the bank's own location names the fleet."""
        fleet = _build_fleet(
            tmp_path / "fleet",
            root_rows=[UPSERT_PAIR + (None,)],
            profiles=[("alpha", ())],
        )
        home = _profile(fleet, "alpha")
        assert fh.fleet_root_for_bank(home) == fleet.resolve()

        self._upsert_once(home)

        assert _conflict_rows(home) == []
        assert _audit_rows(home)[0]["metadata"]["federated_to"]

    def test_audit_stamp_carries_both_clocks(self, tmp_path):
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ())])
        decision = fh.probe_fleet(
            PAIR, _profile(fleet, "alpha"), fleet_root=fleet, sleep=_noop_sleep)

        local = datetime.fromisoformat(decision.read_at_local)
        utc = datetime.fromisoformat(decision.read_at_utc)
        assert decision.local_zone
        assert (local - utc).total_seconds() == decision.utc_offset_seconds

        metadata = decision.audit_events()[0]["metadata"]
        for key in ("read_at_utc", "read_at_local", "local_zone", "utc_offset_seconds"):
            assert key in metadata


# ---------------------------------------------------------------------------
# 8. Probe placement and the pending queue (PR #1082 review threads
#    PRRT_kwDOR6YkqM6nPOsB / POsH): the handshake must run OUTSIDE the
#    home bank's write transaction, and a pair refused by a transient
#    probe failure must be QUEUED, not dropped.
# ---------------------------------------------------------------------------
def _pending_pairs(bank: Path):
    conn = sqlite3.connect(f"{bank.as_uri()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT * FROM conflict_probe_pending ORDER BY created_at")]
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    return rows


class TestPostCommitProbe:

    def _double_upsert(self, consolidator) -> None:
        consolidator.consolidate_fact(SUBJECT, PREDICATE, OBJECT_OLD, "stated", "s1")
        consolidator.consolidate_fact(SUBJECT, PREDICATE, OBJECT_NEW, "stated", "s2")

    def test_probe_never_runs_inside_the_home_write_transaction(
            self, tmp_path, monkeypatch):
        """Review thread POsB: the fleet probe must not hold the home
        bank's ``BEGIN IMMEDIATE`` -- no filesystem walk, no per-peer
        connection, no retry sleep inside the write lock."""
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])
        home = _profile(fleet, "alpha")
        real = fh.probe_fleet
        tx_at_probe: list = []

        consolidator = VeracityConsolidator(db_path=home)
        try:
            def spy(pair, bank, **kwargs):
                # ``in_transaction`` on the consolidator's own connection:
                # False now means the upsert's tx had already committed
                # before the handshake ran.
                tx_at_probe.append(consolidator.conn.in_transaction)
                return real(pair, bank, **kwargs)

            monkeypatch.setattr(fh, "probe_fleet", spy)
            self._double_upsert(consolidator)
        finally:
            consolidator.close()

        assert tx_at_probe, "the gate must have probed at least once"
        assert not any(tx_at_probe), (
            "probe_fleet ran while the home bank was mid-transaction -- "
            "the write lock still covers the fleet probe")
        # End state unchanged: empty peers, home row grows.
        assert len(_conflict_rows(home)) == 1

    def test_home_bank_accepts_a_second_writer_while_the_probe_runs(
            self, tmp_path, monkeypatch):
        """Red at the pre-fix head: with the probe inside the upsert's
        ``BEGIN IMMEDIATE``, an intruder connection with ``busy_timeout``
        of zero raises ``database is locked`` at exactly the moment the
        handshake runs. Post-fix the probe is after the commit, so the
        write lock is free and the intruder's write lands."""
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])
        home = _profile(fleet, "alpha")
        real = fh.probe_fleet
        outcomes: list = []

        def probe_with_intruder(pair, bank, **kwargs):
            other = sqlite3.connect(str(home), timeout=0)
            try:
                other.execute("BEGIN IMMEDIATE")
                other.execute(
                    "INSERT INTO consolidated_facts "
                    "(id, subject, predicate, object) "
                    "VALUES ('intruder', 'Zed', 'is', 'here')")
                other.commit()
                outcomes.append("free")
            except sqlite3.OperationalError as exc:
                outcomes.append(f"locked: {exc}")
            finally:
                other.close()
            return real(pair, bank, **kwargs)

        monkeypatch.setattr(fh, "probe_fleet", probe_with_intruder)
        consolidator = VeracityConsolidator(db_path=home)
        try:
            self._double_upsert(consolidator)
        finally:
            consolidator.close()

        assert outcomes == ["free"], outcomes
        assert len(_conflict_rows(home)) == 1

    def test_refused_pair_is_queued_not_dropped_and_the_next_pass_records_it(
            self, tmp_path):
        """Review thread POsH: one transient peer error used to delete the
        contradiction forever (no row, no re-check on the UPDATE branch).
        Now the refused pair is durable in ``conflict_probe_pending`` --
        across connections, i.e. across process death -- and the sleep
        path's ``run_consolidation_pass`` re-probes and records it once
        the peer is gone."""
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ())])
        broken = _profile(fleet, "broken")
        broken.parent.mkdir(parents=True, exist_ok=True)
        broken.write_bytes(b"this is not a sqlite database\n")
        home = _profile(fleet, "alpha")

        consolidator = VeracityConsolidator(db_path=home)
        try:
            self._double_upsert(consolidator)
            # Fail-closed today: no conflict row...
            assert _conflict_rows(home) == []
            # ...the detected facts still commit...
            assert _fact_count(home) == 2
            # ...the refusal is audited...
            audits = _audit_rows(home)
            assert [a["action"] for a in audits] == [
                "federation_probe_failed", "federation_probe"]
            assert audits[1]["metadata"]["refused"] is True
            # ...and the pair is queued, NOT lost.
            queued = _pending_pairs(home)
            assert [(q["fact_a_id"], q["fact_b_id"]) for q in queued] == [UPSERT_PAIR]
        finally:
            consolidator.close()

        # The peer disappears (healed / removed). A FRESH consolidator --
        # standing in for the next sleep after any restart -- settles it.
        shutil.rmtree(fleet / "profiles" / "broken")
        consolidator = VeracityConsolidator(db_path=home)
        try:
            consolidator.run_consolidation_pass()
        finally:
            consolidator.close()

        assert len(_conflict_rows(home)) == 1
        assert _pending_pairs(home) == []

    def test_a_crashed_probe_keeps_the_pair_queued_and_the_upsert_clean(
            self, tmp_path, monkeypatch):
        """The queue is written inside the fact's transaction, so even a
        handshake module that raises out of contract leaves the pair
        durable and the caller's write successful."""
        fleet = _build_fleet(tmp_path / "fleet", profiles=[("alpha", ()), ("beta", ())])
        home = _profile(fleet, "alpha")

        def detonate(pair, bank, **kwargs):
            raise RuntimeError("handshake exploded")

        monkeypatch.setattr(fh, "probe_fleet", detonate)
        consolidator = VeracityConsolidator(db_path=home)
        try:
            self._double_upsert(consolidator)  # must NOT raise
            assert _fact_count(home) == 2
            assert _conflict_rows(home) == []
            queued = _pending_pairs(home)
            assert [(q["fact_a_id"], q["fact_b_id"]) for q in queued] == [UPSERT_PAIR]
        finally:
            consolidator.close()

    def test_opted_out_upsert_still_lands_one_conflict_row(self, tmp_path, monkeypatch):
        """With the handshake disabled the queue is a pass-through: same
        end state as the old inline path, and no audit noise."""
        fleet = _build_fleet(
            tmp_path / "fleet",
            root_rows=[UPSERT_PAIR + (None,)],
            profiles=[("alpha", ())],
        )
        monkeypatch.setenv(fh.HANDSHAKE_ENV, "0")
        home = _profile(fleet, "alpha")

        consolidator = VeracityConsolidator(db_path=home)
        try:
            self._double_upsert(consolidator)
        finally:
            consolidator.close()

        assert len(_conflict_rows(home)) == 1
        assert _pending_pairs(home) == []
        assert _audit_rows(home) == []
