"""
Fleet Conflict Census
=====================

Read-only, cross-bank discovery of conflict-row twins, run on the sleep-time
path so the accepted-scope bound is MEASURED rather than asserted.

Why this exists (policy, not invention): the federation handshake between
banks probes the shared surface only, so a conflict pair that lands in two
DIFFERENT home banks is accepted scope. That acceptance is bounded by the
fleet census — per bank holding a ``conflicts`` table, the order-normalized
``(min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id))`` pairs,
cross-joined across banks — and the bound has to be RECOMPUTED at every
sleep pass, because a non-zero home↔home twin count means the bound no
longer holds and the disposition policy must be revised to widen the
handshake. An asserted bound decays as the fleet grows; a recomputed one
does not.

Scope, delimited on purpose:

  - DISCOVERY ONLY. This module never resolves, dedups, invalidates or
    writes anything, in any bank, ever. Every bank is opened through
    SQLite's ``mode=ro`` URI, so "no writes" is enforced by the engine
    rather than promised by this code, and the call site treats a census
    failure as a logged diagnostic rather than a consolidation failure.
  - The per-bank invariant (one order-normalized unique pair key) lives in
    the schema: the canonical DDL in ``veracity_consolidation`` for fresh
    banks, plus the E8 package migration for existing ones. This census is
    the cross-bank DISCOVERY instrument ONLY — it must not attempt
    cross-bank dedup, which is the surface-vs-home law's job, not a
    schema's and not a reporter's.
  - No LLM, and no conflict *detection*. sleep()'s inline heuristic +
    validated invalidation path stays exactly where it is; this module
    reads the rows that path already wrote.

Discovery rule. A bank is "any SQLite file under the fleet root that holds
a ``conflicts`` table". Walking the root (rather than enumerating known
bank directories) is deliberate: bank files live in several layouts — a
data dir's ``banks/<name>/``, a profile's own storage, a dedicated
surface directory — and the census must see the same set the measurement
it maintains a bound for was taken over. The walk is pruned of
cache/backup/build directories, so its cost tracks the number of bank
files, not the size of the tree.

Identity, not labels. Twins are keyed by ABSOLUTE PATH; the ``label``
field exists only so a human can read the report. This is not cosmetic:
an earlier probe of this same fleet keyed its per-bank map by a display
label, and the label function collapsed five distinct home banks onto one
name — so a pair present in all five looked like one bank holding it, and
the home↔home bound read 0 while the twin was live. A census that a bound
is stated against must not be able to lose a bank to a name collision.

Two clocks (Mnemosyne's record quirks). ``conflicts.created_at`` is naive
UTC; ``memory.valid_from`` / ``valid_until`` are naive LOCAL. The emitted
stamp therefore carries both, each key named with its zone, plus the local
zone abbreviation — so a reader comparing stamp to rows never has to guess
which clock they hold, and the two are never mixed.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Environment override for the fleet root. Resolved at call time, never
#: captured at import time, so a test (or a narrowed maintenance run) can
#: point the census at a fixture.
FLEET_ROOT_ENV = "MNEMOSYNE_FLEET_ROOT"

#: Opt-out for the every-sleep census walk: ``MNEMOSYNE_FLEET_CENSUS=0``
#: (or any value in :data:`_FALSY`). Checked at CALL time, never cached — a
#: sleep pass that reads the flag once and keeps it would defeat the point of
#: an operator being able to stop the walk without restarting the gateway.
#: The walk itself stays cheap (pruned, ``stat``-only discovery), so this is
#: the escape hatch for a pathologically large tree, not a correctness gate.
CENSUS_ENV = "MNEMOSYNE_FLEET_CENSUS"

#: Values that mean "off". Same set the federation handshake uses
#: (``federation_handshake._FALSY``); an unset or unrecognised value is ON.
_FALSY = frozenset({"0", "false", "no", "off"})


def census_enabled() -> bool:
    """Whether the sleep-time fleet census runs. ``MNEMOSYNE_FLEET_CENSUS``.

    Unset (or any value outside :data:`_FALSY`) means enabled — the census is
    what keeps R6's acceptance bound measured rather than asserted, so its
    default must stay on. Resolved on every call so an operator can stop and
    restart the walk at runtime.
    """
    raw = os.environ.get(CENSUS_ENV)
    if raw is None:
        return True
    return raw.strip().lower() not in _FALSY

#: Directories that never hold a live bank. Pruned during the walk so the
#: census stays cheap enough to run on every sleep pass. Mirrors the skip
#: list the acceptance measurement used (``backups``, ``cache``,
#: ``.cache``) plus package/build trees that cannot contain a bank.
_SKIP_DIR_NAMES = frozenset({
    "backups",
    "cache",
    ".cache",
    "__pycache__",
    "node_modules",
    ".git",
    "site-packages",
})


def default_fleet_root() -> Path:
    """Return the fleet root, resolved fresh on every call.

    ``MNEMOSYNE_FLEET_ROOT`` wins; then the Hermes root; then ``~/.hermes``
    — the layout the live bound was measured against. Never cached: a
    frozen root would silently census yesterday's fleet.

    The Hermes root is NOT simply ``HERMES_HOME``. Inside a profile-scoped
    session ``HERMES_HOME`` is that profile's home (``<root>/profiles/
    <name>``, with ``HERMES_PROFILE`` naming it), and defaulting to it would
    silently census ONE seat's banks while reporting a fleet bound — a scope
    leak that under-reports twins, which is the failure direction that
    matters here. When the home is recognisably a profile home, the fleet
    root is the shared Hermes root above it.
    """
    override = os.environ.get(FLEET_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    hermes_home = os.environ.get("HERMES_HOME")
    if not hermes_home:
        return Path.home() / ".hermes"
    home = Path(hermes_home).expanduser()
    profile = os.environ.get("HERMES_PROFILE")
    if profile and home.name == profile and home.parent.name == "profiles":
        return home.parent.parent
    return home


def shared_db_path(root: Path) -> Path:
    """Return the shared-surface bank, resolved the way the MCP surface does.

    Same precedence as ``mcp_tools._shared_db_path``: an explicit
    ``MNEMOSYNE_SHARED_DB_PATH``, else ``MNEMOSYNE_HOME``/data/shared, else
    ``<root>/mnemosyne/data/shared/mnemosyne.db``.
    """
    override = os.environ.get("MNEMOSYNE_SHARED_DB_PATH")
    if override:
        return Path(override).expanduser()
    mnemosyne_home = os.environ.get("MNEMOSYNE_HOME") or str(Path(root) / "mnemosyne")
    return Path(mnemosyne_home) / "data" / "shared" / "mnemosyne.db"


def discover_banks(root: Path) -> List[Path]:
    """Return candidate bank files under ``root`` (non-empty ``*.db``).

    Candidates still have to prove they hold a ``conflicts`` table when
    read; this only spares the reader a connect on obviously-empty files.
    """
    root = Path(root)
    candidates: List[Path] = []
    if not root.is_dir():
        return candidates
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIR_NAMES)
        for filename in sorted(filenames):
            if not filename.endswith(".db"):
                continue
            path = Path(dirpath) / filename
            try:
                if path.stat().st_size == 0:
                    continue
            except OSError:
                continue
            candidates.append(path)
    return candidates


def _norm_pair(a: str, b: str) -> tuple:
    """Order-normalize a pair exactly as the unique pair key does."""
    return (a, b) if a <= b else (b, a)


def _read_bank(bank: Path) -> Optional[Dict[str, Any]]:
    """Read one bank's conflict rows.

    Returns ``None`` when the file holds no ``conflicts`` table (not a
    bank). A bank that looks like one but cannot be read raises
    ``sqlite3.Error``; the caller records it under ``unreadable`` instead
    of aborting the census. The connection is opened ``mode=ro`` — a
    read-only URI that SQLite refuses to write through.
    """
    connection = sqlite3.connect(f"{bank.as_uri()}?mode=ro", uri=True)
    try:
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflicts'"
        ).fetchone():
            return None
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT fact_a_id, fact_b_id, resolution FROM conflicts"
        ).fetchall()
    finally:
        connection.close()

    open_conflicts = 0
    skipped_null_pair_rows = 0
    pairs: Dict[tuple, int] = {}
    for row in rows:
        fact_a_id = row["fact_a_id"]
        fact_b_id = row["fact_b_id"]
        if fact_a_id is None or fact_b_id is None:
            # Canonical DDL declares both NOT NULL; a foreign schema may not.
            # A NULL cannot name a pair, so it is counted and skipped rather
            # than folded into a fake key.
            skipped_null_pair_rows += 1
            continue
        if row["resolution"] is None:
            open_conflicts += 1
        key = _norm_pair(str(fact_a_id), str(fact_b_id))
        pairs[key] = pairs.get(key, 0) + 1
    return {
        "total_conflicts": len(rows),
        "open_conflicts": open_conflicts,
        "distinct_normalized_pairs": len(pairs),
        "skipped_null_pair_rows": skipped_null_pair_rows,
        "pairs": pairs,
    }


def census(root: Optional[Path] = None, banks: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Census conflict rows across every bank under the fleet root.

    Args:
        root: Fleet root to walk. Defaults to :func:`default_fleet_root`,
            resolved at call time.
        banks: Explicit bank files, for callers (and tests) that already
            know the set and do not want the walk.

    The root and every bank path are expanded and resolved to absolute,
    deduplicated values before any bank is read: twins are keyed by file
    identity, so a symlinked bank counts once and a relative path names
    the same fleet as its absolute spelling.

    Returns:
        A JSON-serializable dict. The load-bearing key is
        ``home_home_twin_count`` / ``bound_holds``: the bound is
        ``home_home_twin_count == 0``, and a pair counts as a home↔home
        twin when two or more NON-shared banks hold it. Pairs that involve
        the shared surface are reported separately under
        ``surface_home_twins`` (a pair may legitimately appear in both
        lists — the shared surface is not a home bank, so its presence
        never excuses two home banks holding the same pair).
    """
    root_path = (Path(root).expanduser() if root is not None else default_fleet_root()).resolve()
    raw_banks = list(banks) if banks is not None else discover_banks(root_path)
    # Paths are the identity twins are keyed on, so normalize before
    # reading: resolve() makes a relative root or caller-supplied relative
    # banks absolute (``Path.as_uri()`` raises ValueError on relative
    # paths — not a sqlite3.Error, and it would abort the whole census),
    # and a symlinked bank collapses to the file it points at instead of
    # counting as a second holder. Dedup + sort keep the walk deterministic.
    bank_paths = sorted({Path(b).expanduser().resolve() for b in raw_banks})
    surface_path = shared_db_path(root_path)
    try:
        surface_resolved = surface_path.resolve()
    except OSError:  # pragma: no cover - defensive; resolve() rarely raises
        surface_resolved = surface_path

    # One instant, read through two zones. Taking two separate now() calls
    # would leave the stamps a few microseconds apart and quietly break the
    # offset identity a reader uses to check they are the same moment.
    _utc_now = datetime.now(timezone.utc)
    _local_now = _utc_now.astimezone()
    offset = _local_now.utcoffset()

    report: Dict[str, Any] = {
        "kind": "fleet_conflict_census",
        "root": str(root_path),
        "shared_db": str(surface_path),
        # Both clocks, each named with its zone. ``conflicts.created_at``
        # is naive UTC; ``valid_from``/``valid_until`` are naive local.
        "read_at_utc": _utc_now.replace(tzinfo=None).isoformat(),
        "read_at_local": _local_now.replace(tzinfo=None).isoformat(),
        "local_zone": _local_now.tzname() or "local",
        "utc_offset_seconds": int(offset.total_seconds()) if offset else 0,
        "banks": [],
        "unreadable": [],
    }

    labels: List[str] = []
    owners: Dict[tuple, List[str]] = {}
    shared_labels = set()
    within_bank_collisions: List[Dict[str, Any]] = []
    total_rows = 0

    for bank in sorted(bank_paths):
        try:
            resolved = bank.resolve()
        except OSError:  # pragma: no cover - defensive
            resolved = bank
        is_shared = resolved == surface_resolved
        try:
            rel = bank.relative_to(root_path).as_posix()
        except ValueError:
            rel = bank.as_posix()

        try:
            facts = _read_bank(bank)
        except sqlite3.Error as exc:
            report["unreadable"].append({"path": str(bank), "error": repr(exc)})
            logger.warning("fleet census: unreadable bank %s (%s)", bank, type(exc).__name__)
            continue
        if facts is None:
            continue

        label = "shared" if is_shared else rel
        labels.append(label)
        if is_shared:
            shared_labels.add(label)
        total_rows += facts["total_conflicts"]

        report["banks"].append({
            "label": label,
            "path": str(bank),
            "is_shared": is_shared,
            "total_conflicts": facts["total_conflicts"],
            "open_conflicts": facts["open_conflicts"],
            "distinct_normalized_pairs": facts["distinct_normalized_pairs"],
            "skipped_null_pair_rows": facts["skipped_null_pair_rows"],
        })

        for key, occurrences in facts["pairs"].items():
            owners.setdefault(key, []).append(label)
            if occurrences > 1:
                within_bank_collisions.append({
                    "pair": [key[0], key[1]],
                    "bank": label,
                    "occurrences": occurrences,
                })

    cross_bank_twins = []
    home_home_twins = []
    surface_home_twins = []
    for key in sorted(owners):
        holders = owners[key]
        if len(holders) < 2:
            continue
        holders_sorted = sorted(holders)
        involves_shared = any(h in shared_labels for h in holders)
        entry = {
            "pair": [key[0], key[1]],
            "banks": holders_sorted,
            "involves_shared": involves_shared,
        }
        cross_bank_twins.append(entry)
        home_holders = [h for h in holders_sorted if h not in shared_labels]
        if len(home_holders) >= 2:
            home_home_twins.append({"pair": [key[0], key[1]], "banks": home_holders})
        if involves_shared and home_holders:
            surface_home_twins.append({
                "pair": [key[0], key[1]],
                "banks": holders_sorted,
            })

    report.update({
        "banks_scanned": len(labels),
        "total_conflict_rows": total_rows,
        "within_bank_collisions": sorted(
            within_bank_collisions, key=lambda e: (e["bank"], e["pair"])
        ),
        "cross_bank_twins": cross_bank_twins,
        "home_home_twins": home_home_twins,
        "home_home_twin_count": len(home_home_twins),
        "surface_home_twins": surface_home_twins,
        # R6's accepted-scope bound, stated as a boolean so a caller does not
        # have to remember which direction of the count is the safe one.
        # Fail CLOSED on partial data: a bank that could not be read is an
        # unmeasured holder, so a census taken over a fleet with one
        # unreadable bank must not report the bound as holding — "no twins
        # found" and "no twins COULD be found" are different facts, and only
        # the second one is safe to read as a maintained bound.
        "bound": "home-to-home normalized conflict twins held by 2+ non-shared banks",
        "bound_holds": len(home_home_twins) == 0 and not report["unreadable"],
    })
    return report