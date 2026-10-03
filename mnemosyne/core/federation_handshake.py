"""
Federation Handshake — cross-bank dedup of newborn conflict rows
================================================================

Policy: ``policy:conflict-disposition-2026-09-26`` v3 (why seat), rules
R3 §(e)–(i), R7 and R8. This module is that policy's engineering handoff
(E8c): the upsert-time gate that stops the same order-normalized conflict
pair from landing in N banks of one fleet.

The defect it closes is measured, not hypothesized. On 2026-09-29 one
home↔home twin (``cf_beb4bf87d48cbded7ec53b4f × cf_c9eb90673026b30246852f55``)
was live in FIVE non-shared banks (root + keeper + scales + senses + why).
The earlier handshake probed the shared surface only, so every home bank
persisted its own copy. An even earlier probe of the same fleet keyed its
per-bank map by a display *label*, collapsed those five banks onto one
name, and reported **0 twins with the twin live**. Scope here is therefore
keyed on absolute paths and read from the fleet census's own enumeration.

What this module is
-------------------

``probe_fleet(pair, home_bank)`` answers one question: *does any other live
bank in this fleet already hold this pair?* It returns a
:class:`HandshakeDecision` — the chain of holders found, the failures seen,
and whether the caller may proceed with its own insert.

  - **R3 §(e) scope.** Peers are the banks the fleet census enumerates
    (``fleet_census.discover_banks`` / ``shared_db_path`` — imported, never
    re-implemented, so probe scope cannot drift from the bound the census
    maintains). Probe order: shared surface first, root bank second, every
    other non-shared bank third in fleet-enumeration order. The upserting
    bank itself is never probed.
  - **R3 §(f) per-probe behaviour + one entry per upsert.** Every peer is
    probed; hits accumulate into ``chain``; the caller writes ONE audit
    entry for the upsert (after all probes), not one per probe. Empty chain
    → proceed. Non-empty chain → skip the home insert and federate.
  - **R3 §(g) fail-CLOSED, per probe.** Any probe that raises (timeout,
    permission denied, schema mismatch, sqlite I/O, …) refuses the home
    upsert outright and names the failure. Silence would be the failure
    class the whole policy exists to close.
  - **R7 canonical preference.** shared > root > oldest profile, chosen
    deterministically and recorded in the audit entry with the rule text.
  - **R8 time budget.** 250 ms per probe, 1 retry × 500 ms backoff, and a
    total wall-clock cap that defaults to 2 s. v3 leaves the total to
    engineering; it is CAPPED rather than left unfixed because the probe runs
    while the home bank's write lock is held, so an uncapped sequence would
    hold that lock for N × 250 ms (4.75 s today, ~9.5 s with retries).
    Measured latency for a local read-only sqlite probe is ~0.23 ms, so the
    default carries ~400× headroom. Exceeding the cap is fail-closed, like any
    other probe failure; ``MNEMOSYNE_FEDERATION_TOTAL_BUDGET_MS`` moves or
    removes it (``0``/``none``/``off`` are the only ways to remove it).
    Three corollaries the review round made explicit: the clock starts
    BEFORE scope resolution and is rechecked after the last probe, because
    the scope's own liveness reads retry and back off under the write lock;
    the cap bounds SQLite lock waits too, not only VM work (a busy-wait is
    not VM work and the progress handler never fires during one); and a
    configured duration that is unparseable, non-finite or non-positive
    raises instead of silently becoming the default.

What this module deliberately is not
------------------------------------

It never writes to a probed bank: every peer is opened through SQLite's
``mode=ro`` URI, so "read-only" is enforced by the engine, not promised by
this code. It runs no LLM. It does not adjudicate — the pre-existing twin
rows are another lane's business; this module only stops *new* duplicates
from forming.

Schema note (honest reconciliation of the policy text)
------------------------------------------------------

R3 §(f) writes the probe as ``WHERE min_id=? AND max_id=?``. The shipped
schema has no ``min_id``/``max_id`` columns: E8's order-normalized key is an
EXPRESSION index on ``(min(fact_a_id, fact_b_id), max(fact_a_id, fact_b_id))``.
The predicate below is that same expression, so the probe reads the pair by
exactly the key the schema enforces (and can use the index). This is an
equivalent spelling, not a scope change.

Two clocks (Mnemosyne's record quirks): ``conflicts.created_at`` is naive
UTC; ``valid_from``/``valid_until`` are naive LOCAL. The decision stamp
carries both, each named with its zone, plus the local zone abbreviation and
offset — derived from ONE instant so a reader can check they agree.
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from mnemosyne.core import fleet_census

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_BACKOFF_S",
    "DEFAULT_PER_PROBE_TIMEOUT_S",
    "DEFAULT_RETRIES",
    "DEFAULT_TOTAL_BUDGET_S",
    "HandshakeDecision",
    "PeerSpec",
    "fleet_root_for_bank",
    "probe_enabled",
    "probe_fleet",
    "resolve_fleet_root",
    "root_db_path",
    "total_budget_seconds",
]

#: Disable the handshake entirely with ``MNEMOSYNE_FEDERATION_HANDSHAKE=0``.
#: On by default, mirroring the E8b census (which rides the sleep path by
#: default with a private opt-out). Policy v3 makes the expanded probe
#: permanent — "the policy cannot be violated by a probe bug if it always
#: probes the full set" (R3 §(i)) — so there is no ``bound_holds`` gate here.
HANDSHAKE_ENV = "MNEMOSYNE_FEDERATION_HANDSHAKE"

#: Per-probe budget (R3 §(h) / R8 §(a)), in milliseconds.
PROBE_TIMEOUT_ENV = "MNEMOSYNE_FEDERATION_PROBE_TIMEOUT_MS"

#: Optional TOTAL wall-clock cap for one full probe sequence, in
#: milliseconds. Defaults to :data:`DEFAULT_TOTAL_BUDGET_S` (2 s) because the
#: probe runs while the home bank's write lock is held (see
#: ``VeracityConsolidator._record_conflict``): without a cap a full sequence
#: would hold that lock for N x 250 ms — 4.75 s today, ~9.5 s with retries.
#: The cap bounds the LOCK HOLD, not correctness: exceeding it is fail-closed
#: (the home row is not written and a ``federation_probe_failed`` row with
#: ``error_class='total_budget_exceeded'`` is recorded), and R1's next
#: detection cycle retries automatically. Measured typical is ~4.8 ms for 19
#: probes, so the default carries ~400x headroom. Set it to ``0``, ``none``
#: or ``off`` for an unbounded sequence (v3's literal ratified state); set a
#: positive number of milliseconds to move the cap. R8 §(d) may amend v3's
#: text if a cap is ratified.
TOTAL_BUDGET_ENV = "MNEMOSYNE_FEDERATION_TOTAL_BUDGET_MS"

DEFAULT_PER_PROBE_TIMEOUT_S = 0.25
DEFAULT_RETRIES = 1
DEFAULT_BACKOFF_S = 0.5

#: Default total cap in seconds. See :data:`TOTAL_BUDGET_ENV`.
DEFAULT_TOTAL_BUDGET_S = 2.0

_FALSY = frozenset({"0", "false", "no", "off"})
_UNBOUNDED = frozenset({"", "0", "none", "off", "false", "unlimited", "unbounded"})


class NotABank(Exception):
    """Raised by :func:`_check_bank` for a file with no ``conflicts`` table.

    Not a failure: the census's bank predicate is "holds a ``conflicts``
    table", so a file without one is simply not part of the fleet. Raised
    rather than returned so the caller can distinguish it from a real read
    error, which R3 §(g) makes a refusal.
    """


# ---------------------------------------------------------------------------
# Configuration resolution (every knob resolved at call time, never frozen)
# ---------------------------------------------------------------------------
def _env_is_falsy(name: str) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in _FALSY


def probe_enabled() -> bool:
    """Whether the federation handshake runs. ``MNEMOSYNE_FEDERATION_HANDSHAKE``.

    Unset (or any value outside the falsy set) means enabled.
    """
    return not _env_is_falsy(HANDSHAKE_ENV)


def _env_seconds(name: str, default_seconds: float) -> float:
    """Read a millisecond-valued env var and return SECONDS.

    The ``*_MS`` suffix is the fleet's convention (``MNEMOSYNE_BUSY_TIMEOUT_MS``
    and the E8b census both follow it), so the unit is converted here rather
    than left for a caller to misread.

    Unset → the default. PRESENT but unparseable, non-finite or non-positive →
    :class:`ValueError`: a configured value that cannot be honoured is a
    configuration error, and substituting the default would hide a typo
    behind a timeout that looks deliberate. Infinity especially is not "a
    very large timeout" — it removes the deadline, and this knob has no
    documented way to do that.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default_seconds
    return _finite_positive_ms(name, raw) / 1000.0


def _finite_positive_ms(name: str, raw: str, *, hint: Optional[str] = None) -> float:
    """Parse a configured millisecond duration, refusing unusable values.

    Raises rather than defaulting (see :func:`_env_seconds`), and names the
    offending ``var=value`` so the operator does not have to guess which
    knob is wrong. ``hint`` lets the total budget's message mention its
    unbounded sentinels, which the per-probe knob does not have.
    """
    expectation = hint or "a finite, positive number of milliseconds"
    try:
        value = float(raw.strip())
    except ValueError:
        raise ValueError(f"{name}={raw!r} is invalid: expected {expectation}") from None
    if not math.isfinite(value):
        raise ValueError(
            f"{name}={raw!r} is invalid: expected {expectation} — infinity "
            f"removes the timeout instead of enlarging it"
        )
    if value <= 0:
        raise ValueError(f"{name}={raw!r} is invalid: expected {expectation}")
    return value


def per_probe_timeout_seconds() -> float:
    """Per-probe budget in seconds (R8 §(a)).

    ``MNEMOSYNE_FEDERATION_PROBE_TIMEOUT_MS`` (milliseconds) overrides the
    250 ms default.
    """
    return _env_seconds(PROBE_TIMEOUT_ENV, DEFAULT_PER_PROBE_TIMEOUT_S)


def total_budget_seconds() -> Optional[float]:
    """Total wall-clock cap for one probe sequence, or ``None`` (unbounded).

    Defaults to :data:`DEFAULT_TOTAL_BUDGET_S`, which bounds how long the
    probe can hold the home bank's write lock. ``MNEMOSYNE_FEDERATION_TOTAL_BUDGET_MS``
    overrides it: a positive number of milliseconds moves the cap, and ``0`` /
    ``none`` / ``off`` (or any other :data:`_UNBOUNDED` spelling) removes it —
    that sentinel set is the ONLY sanctioned way to run unbounded, because it
    is explicit and self-documenting. Any other present-but-unusable value
    (garbage, ``inf``, ``nan``, a negative number) raises: it is a
    configuration error, not a request for the default. Exceeding the cap is
    fail-closed, like any other probe failure.
    """
    raw = os.environ.get(TOTAL_BUDGET_ENV)
    if raw is None:
        return DEFAULT_TOTAL_BUDGET_S
    stripped = raw.strip().lower()
    if stripped in _UNBOUNDED:
        return None
    return _finite_positive_ms(
        TOTAL_BUDGET_ENV, raw,
        hint=("a finite, positive number of milliseconds, or one of "
              + "/".join(sorted(_UNBOUNDED - {""})) + " for an unbounded budget"),
    ) / 1000.0


def root_db_path(fleet_root: Path) -> Path:
    """Return the ROOT bank path for a fleet root.

    Mirrors :func:`fleet_census.shared_db_path` precedence so the root is
    resolved the same way the census resolves the surface:
    ``MNEMOSYNE_HOME`` wins, else ``<fleet_root>/mnemosyne``; then
    ``data/mnemosyne.db``. R7 §(b) names this bank the second-ranked
    canonical holder.
    """
    mnemosyne_home = os.environ.get("MNEMOSYNE_HOME") or str(Path(fleet_root) / "mnemosyne")
    return Path(mnemosyne_home) / "data" / "mnemosyne.db"


def fleet_root_for_bank(bank: Path) -> Optional[Path]:
    """Derive the fleet root that CONTAINS ``bank``, or ``None``.

    Engineering's scoping decision, and the reason this gate is safe to leave
    on: a bank that is not part of a recognisable fleet has no peers to probe,
    so its insert proceeds exactly as it did before this module existed.
    Without this, a standalone install would walk (and probe) an unrelated
    tree, and a test writing a bank under a temp dir would probe the
    developer's live fleet.

    Derivation is by PATH SHAPE, not by walking up to the first ancestor that
    looks fleet-ish. That distinction is load-bearing: pytest's ``tmp_path``
    is frequently *inside* a real fleet (``~/.hermes/profiles/<seat>/cache/
    scratch/pytest-of-…``), so an ancestor walk would hand a temp bank the
    live fleet as its scope. Two shapes are recognised, matching the layout
    the fleet census walks:

      * ``<D>/mnemosyne/data/mnemosyne.db``            (the root bank)
      * ``<D>/profiles/<name>/mnemosyne/data/mnemosyne.db`` (a profile bank)
      * ``<D>/mnemosyne/data/shared/mnemosyne.db``     (the surface)

    ``D`` is returned only when it actually holds a ``profiles`` directory —
    i.e. it is a fleet with more than one bank, which is the only case where
    probing other banks can matter. A root-only install has no peers to
    federate to; ``MNEMOSYNE_FLEET_ROOT`` covers that deployment (and tests).
    """
    try:
        resolved = Path(bank).expanduser().resolve()
    except OSError:  # pragma: no cover - defensive
        return None

    parts = list(resolved.parts)
    if not parts or parts[-1] != "mnemosyne.db":
        return None
    if len(parts) >= 2 and parts[-2] == "shared":
        parts = parts[:-2]
    if len(parts) < 4 or parts[-2] != "data" or parts[-3] != "mnemosyne":
        return None

    index = len(parts) - 3
    if index >= 2 and parts[index - 2] == "profiles":
        root = Path(*parts[:index - 2])
    else:
        root = Path(*parts[:index])
    if len(root.parts) <= 1:
        return None
    try:
        if not (root / "profiles").is_dir():
            return None
    except OSError:  # pragma: no cover - defensive
        return None
    return root


def resolve_fleet_root(home_bank: Path,
                       fleet_root: Optional[Path] = None) -> Optional[Path]:
    """Resolve the fleet root for one handshake, in precedence order.

    Explicit argument → ``MNEMOSYNE_FLEET_ROOT`` → :func:`fleet_root_for_bank`
    (derived from the upserting bank). ``None`` means "this bank is not in a
    fleet", and the handshake then has zero peers.
    """
    if fleet_root is not None:
        return Path(fleet_root).expanduser()
    override = os.environ.get(fleet_census.FLEET_ROOT_ENV)
    if override:
        return Path(override).expanduser()
    return fleet_root_for_bank(home_bank)


# ---------------------------------------------------------------------------
# Decision types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PeerSpec:
    """One bank to probe, with its rank in R7's canonical preference."""

    path: str
    is_shared: bool
    is_root: bool

    @property
    def rank(self) -> int:
        # R7: shared first, root second, profiles third.
        if self.is_shared:
            return 0
        if self.is_root:
            return 1
        return 2


@dataclass
class HandshakeDecision:
    """The outcome of one full probe sequence for one upsert.

    The caller inserts its own conflict row only when :attr:`proceed` is
    true. :meth:`audit_events` yields the rows to write: one
    ``federation_probe_failed`` per failed probe (R3 §(g)), then ONE
    ``federation_probe`` entry for the upsert carrying the whole chain
    (R3 §(f)) and R7's canonical selection.
    """

    pair: Tuple[str, str]
    home_bank: str
    chain: List[Dict[str, Any]] = field(default_factory=list)
    failed: List[Dict[str, Any]] = field(default_factory=list)
    federated_to: Optional[str] = None
    refused: bool = False
    elapsed_ms: float = 0.0
    probes_run: int = 0
    probed_banks: List[str] = field(default_factory=list)
    read_at_utc: Optional[str] = None
    read_at_local: Optional[str] = None
    local_zone: Optional[str] = None
    utc_offset_seconds: int = 0

    #: R7's preference order, stated in the audit entry so a reader does not
    #: have to reconstruct which holder was canonical and why.
    canonical_preference: str = "shared > root > oldest profile (R7)"

    @property
    def proceed(self) -> bool:
        """True when the caller may write its own conflict row."""
        return not self.refused and not self.chain

    @property
    def federated(self) -> bool:
        """True when a holder was found elsewhere and the home insert is skipped."""
        return not self.refused and bool(self.chain)

    def _stamp(self) -> Dict[str, Any]:
        return {
            "read_at_utc": self.read_at_utc,
            "read_at_local": self.read_at_local,
            "local_zone": self.local_zone,
            "utc_offset_seconds": self.utc_offset_seconds,
        }

    @staticmethod
    def _public(entry: Dict[str, Any]) -> Dict[str, Any]:
        """Strip internal bookkeeping (``_rank``) from a chain entry."""
        return {k: v for k, v in entry.items() if not k.startswith("_")}

    def audit_events(self) -> List[Dict[str, Any]]:
        """Ordered ``(action, metadata)`` pairs for this upsert.

        Failures first (one row per failed probe — R3 §(g)), then the single
        per-upsert entry (R3 §(f)). Action names: ``federation_probe_failed``
        is the policy's own name, verbatim; ``federation_probe`` is
        engineering's name for the per-upsert row, which v3 requires but does
        not name.
        """
        events: List[Dict[str, Any]] = []
        for failure in self.failed:
            events.append({
                "action": "federation_probe_failed",
                "metadata": {
                    "pair": list(self.pair),
                    "probe_target": failure.get("probe_target"),
                    "error_class": failure.get("error_class"),
                    "error_message": failure.get("error_message"),
                    "upserting_bank": self.home_bank,
                    **self._stamp(),
                },
            })
        events.append({
            "action": "federation_probe",
            "metadata": {
                "pair": list(self.pair),
                "upserting_bank": self.home_bank,
                "probed_banks": list(self.probed_banks),
                "probes_run": self.probes_run,
                "federation_chain": [self._public(e) for e in self.chain],
                "chain_length": len(self.chain),
                "federated_to": self.federated_to,
                "canonical_preference": self.canonical_preference,
                "proceeded": self.proceed,
                "refused": self.refused,
                "elapsed_ms": round(self.elapsed_ms, 3),
                **self._stamp(),
            },
        })
        return events


# ---------------------------------------------------------------------------
# Error classification (R3 §(g) names the classes it expects)
# ---------------------------------------------------------------------------
def _classify_error(exc: BaseException, path: Optional[Path] = None) -> str:
    """Map a probe exception onto R3 §(g)'s failure vocabulary.

    Classes: ``timeout``, ``permission_denied``, ``schema_mismatch``,
    ``sqlite_io``, ``path_error``, ``io_error``, or the exception's own type
    name when nothing narrower fits. The names are engineering's spellings of
    R3 §(g)'s list — the policy names the classes by example, not as literals.
    """
    message = str(exc).lower()
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, PermissionError):
        return "permission_denied"
    if isinstance(exc, sqlite3.OperationalError):
        if "interrupted" in message:
            return "timeout"
        if "no such table" in message or "no such column" in message:
            return "schema_mismatch"
        if "unable to open" in message:
            # SQLite reports a permission problem as "unable to open database
            # file"; check the filesystem rather than guessing.
            if path is not None and not os.access(path, os.R_OK):
                return "permission_denied"
            return "sqlite_io"
        return "sqlite_io"
    if isinstance(exc, sqlite3.DatabaseError):
        return "sqlite_io"
    if isinstance(exc, ValueError):
        # Path.as_uri() on a relative path; nothing to probe.
        return "path_error"
    if isinstance(exc, OSError):
        return "io_error"
    return type(exc).__name__


def _failure(path: Path, exc: BaseException) -> Dict[str, Any]:
    return {
        "probe_target": str(path),
        "error_class": _classify_error(exc, path),
        "error_message": str(exc)[:500],
    }


def _budget_failure(path: Optional[Path], budget_s: Optional[float]) -> Dict[str, Any]:
    """The recorded failure for a probe that was not run inside its budget.

    Surfaced with ``error_class='total_budget_exceeded'`` — the same class
    the in-loop guard already used, so a reader sees one failure vocabulary
    whether the budget ran out before scope, between probes, or after the
    last probe finished.
    """
    return {
        "probe_target": str(path) if path is not None else "",
        "error_class": "total_budget_exceeded",
        "error_message": (
            f"total probe budget of {budget_s}s exceeded ({TOTAL_BUDGET_ENV})"
        ),
    }


def _connect_timeout_seconds(limit_s: Optional[float]) -> float:
    """SQLite busy-wait bound for one probe connection, in seconds.

    ``sqlite3.connect``'s ``timeout`` is the OTHER half of the probe budget.
    The progress handler aborts SQLite's virtual-machine work, but it is
    never called while the engine waits for a lock, so without this knob the
    connect silently keeps Python's 5 s default and a locked peer outlasts
    any configured budget — per-probe or total — before retry handling runs.
    Defaults to the per-probe budget when a caller does not clamp it.
    """
    if limit_s is None:
        return DEFAULT_PER_PROBE_TIMEOUT_S
    return max(0.0, limit_s)


def _check_bank(path: Path, *, lock_timeout_s: Optional[float] = None) -> None:
    """Raise if ``path`` is not a readable bank.

    Same predicate the fleet census uses — holds a ``conflicts`` table —
    read through the same ``mode=ro`` URI, so this neither writes nor counts
    a non-bank. :class:`NotABank` is raised for "not a bank" (skip);
    anything else propagates and R3 §(g) turns it into a refusal. The
    connection's lock wait is bounded (:func:`_connect_timeout_seconds`)
    rather than left at the engine default.
    """
    connection = sqlite3.connect(
        f"{path.as_uri()}?mode=ro", uri=True,
        timeout=_connect_timeout_seconds(lock_timeout_s),
    )
    try:
        if not connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='conflicts'"
        ).fetchone():
            raise NotABank(str(path))
    finally:
        connection.close()


def _check_bank_retrying(
    path: Path,
    *,
    retries: int,
    backoff_s: float,
    sleep: Callable[[float], None],
    lock_timeout_s: Optional[float] = None,
    deadline: Optional[float] = None,
    total_budget_s: Optional[float] = None,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Decide whether ``path`` is a probeable bank, with R8's retry budget.

    Returns ``(should_probe, failure_or_None)``:

      * ``(True, None)``  — a readable bank: probe it;
      * ``(False, None)`` — not a bank at all (no ``conflicts`` table), or a
        missing file: skipped, exactly as the fleet census skips it;
      * ``(False, failure)`` — a read error OR a budget stop, recorded for
        R3 §(g) refusal.

    The retry/backoff here is the same budget ``_probe_one`` uses, so a
    transient read error at enumeration time cannot escape the fail-closed
    path just because it happened one step earlier.

    When the caller passes an absolute ``deadline``, it bounds the whole
    sequence, not each attempt alone: the lock wait is clamped to the time
    left, a backoff that cannot land before the deadline is skipped, and an
    abandoned decision is reported as a FAILURE (``total_budget_exceeded``
    or the probe's own error) — never as ``(False, None)``, which means
    "not a bank" and would silently drop the candidate fail-OPEN.
    """
    # A path that is not a FILE is not a live bank: it is out of scope, not a
    # failure. This matters for the shared surface and the root bank, which
    # are named from the fleet root rather than discovered — a fleet can
    # simply not have one, and refusing every insert because a surface was
    # never created would be fail-closed in the wrong direction.
    try:
        if not path.is_file():
            return False, None
    except OSError:  # pragma: no cover - defensive
        return False, None
    last_failure: Optional[Dict[str, Any]] = None
    for attempt in range(retries + 1):
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            # Out of budget at the top of an attempt: report, never skip.
            return False, last_failure or _budget_failure(path, total_budget_s)
        lock_limit = lock_timeout_s
        if remaining is not None:
            lock_limit = remaining if lock_limit is None else min(lock_limit, remaining)
        try:
            _check_bank(path, lock_timeout_s=lock_limit)
            return True, None
        except NotABank:
            return False, None
        except Exception as exc:  # noqa: BLE001 - classify, never swallow
            last_failure = _failure(path, exc)
            if attempt < retries:
                if deadline is not None and backoff_s >= deadline - time.monotonic():
                    # The retry's backoff cannot land inside the caller's
                    # budget: abandon it with the real failure on record.
                    return False, last_failure
                sleep(backoff_s)
                continue
            return False, last_failure
    return False, last_failure or _budget_failure(path, total_budget_s)  # pragma: no cover


def _probe_one(
    bank: Path,
    pair: Tuple[str, str],
    *,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    sleep: Callable[[float], None],
    lock_timeout_s: Optional[float] = None,
    deadline: Optional[float] = None,
    total_budget_s: Optional[float] = None,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Probe ONE bank for the pair. Returns ``(hits, failure_or_None)``.

    Read-only by construction: the connection is opened through SQLite's
    ``mode=ro`` URI, so the engine refuses writes even if this code tried.
    ``timeout_s`` is enforced with a progress handler — a query that outruns
    its budget is aborted by SQLite itself (``interrupted``), which
    :func:`_classify_error` reports as ``timeout``. That handler bounds VM
    steps ONLY: the lock wait inside ``connect`` is bounded separately by
    ``lock_timeout_s`` (clamped to ``timeout_s``), because a busy-wait is not
    VM work and the progress handler is never called during one. Bounded
    retries: one retry after ``backoff_s``; beyond that the probe is a
    failure and the upsert refuses (R3 §(g) / R8 §(c)).

    The caller's absolute ``deadline`` (R8 §(b)'s clock, started before
    scope) bounds the WHOLE sequence, not each attempt alone: every attempt
    recomputes the time left and clamps its query budget and lock wait to
    it, and a backoff that cannot land before the deadline is skipped —
    stopping with the probe's own failure on record (or
    ``total_budget_exceeded`` if no attempt could start at all). Without
    that, a retrying probe (2 × 250 ms + 500 ms backoff) could start with
    10 ms left and overrun the cap while still holding the home write lock.
    """
    lock_cap = timeout_s if lock_timeout_s is None else min(timeout_s, max(0.0, lock_timeout_s))
    last_failure: Optional[Dict[str, Any]] = None
    for attempt in range(retries + 1):
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            return [], last_failure or _budget_failure(bank, total_budget_s)
        attempt_timeout = timeout_s if remaining is None else min(timeout_s, remaining)
        lock_timeout = min(lock_cap, attempt_timeout)
        connection: Optional[sqlite3.Connection] = None
        try:
            # The per-attempt deadline covers the lock wait too: start it
            # before connect so a 5-second default can never hide inside a
            # 250 ms budget.
            attempt_deadline = time.monotonic() + attempt_timeout
            connection = sqlite3.connect(
                f"{bank.as_uri()}?mode=ro", uri=True,
                timeout=_connect_timeout_seconds(lock_timeout),
            )

            def _budget_exceeded() -> int:
                return 1 if time.monotonic() > attempt_deadline else 0

            connection.set_progress_handler(_budget_exceeded, 1000)
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT id, resolution, created_at FROM conflicts "
                "WHERE min(fact_a_id, fact_b_id) = ? "
                "AND max(fact_a_id, fact_b_id) = ? "
                "ORDER BY created_at ASC, id ASC",
                (pair[0], pair[1]),
            ).fetchall()
            return [
                {
                    "id": row["id"],
                    "bank_path": str(bank),
                    "resolution": row["resolution"],
                    "resolution_status": row["resolution"],
                    "created_at": row["created_at"],
                }
                for row in rows
            ], None
        except Exception as exc:  # noqa: BLE001 - every failure must fail closed
            last_failure = _failure(bank, exc)
            if attempt < retries:
                if deadline is not None and backoff_s >= deadline - time.monotonic():
                    # The retry's backoff cannot land inside the caller's
                    # budget: stop here with the real failure on record —
                    # the caller refuses, and the lock is released early.
                    return [], last_failure
                sleep(backoff_s)
        finally:
            if connection is not None:
                try:
                    connection.set_progress_handler(None, 0)
                except sqlite3.Error:  # pragma: no cover - defensive
                    pass
                connection.close()
    return [], last_failure


# ---------------------------------------------------------------------------
# Probe scope (R3 §(e)) — the census's enumeration, in R7's order
# ---------------------------------------------------------------------------
def _resolve(path: Path) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except OSError:  # pragma: no cover - defensive
        return Path(path).expanduser()


def _probe_scope(
    home_bank: Path,
    *,
    fleet_root: Optional[Path],
    banks: Optional[Sequence[Path]],
    shared_db: Optional[Path],
    root_db: Optional[Path],
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    sleep: Callable[[float], None] = time.sleep,
    probe_timeout_s: float = DEFAULT_PER_PROBE_TIMEOUT_S,
    deadline: Optional[float] = None,
    total_budget_s: Optional[float] = None,
) -> Tuple[List[PeerSpec], List[Dict[str, Any]]]:
    """Return ``(ordered_peers, failures)`` for one upsert.

    Candidate set: the caller's explicit ``banks`` when given, else the fleet
    census's OWN walk (:func:`fleet_census.discover_banks`) — imported, never
    re-implemented, so the probe set cannot drift from the set the acceptance
    bound is measured over. The shared surface and root bank are added
    explicitly so they hold their R7 slots even if a caller passes a banks
    list that omits them.

    Per candidate:

      * not a bank (no ``conflicts`` table) → skipped, exactly as the census
        skips it — this is not a failure, it is the bank predicate;
      * a live bank → probed;
      * anything that raises while deciding → collected as a FAILURE, because
        a probe set that silently dropped an unreadable bank is the fail-open
        R3 §(g) forbids.

    Order: shared (rank 0), root (rank 1), then every other non-shared bank
    in fleet-enumeration order. The upserting bank is never a peer.

    The scope walk is INSIDE the caller's total budget (``deadline``): the
    per-candidate liveness read retries and backs off exactly like a probe
    does, so an unbounded scope pass could hold the home bank's write lock
    for ``retries x backoff_s`` per candidate bank while the caller's clock
    had not started yet. Past the deadline every remaining candidate is
    recorded as ``total_budget_exceeded`` and skipped, so an overrun surfaces
    as the same fail-closed refusal a probe failure produces.
    """
    home = _resolve(home_bank)
    surface = shared_db if shared_db is not None else (
        fleet_census.shared_db_path(fleet_root) if fleet_root is not None else None
    )
    surface = _resolve(surface) if surface is not None else None
    root_bank = root_db if root_db is not None else (
        root_db_path(fleet_root) if fleet_root is not None else None
    )
    root_bank = _resolve(root_bank) if root_bank is not None else None

    if banks is not None:
        raw_candidates = [_resolve(Path(b)) for b in banks]
    elif fleet_root is not None:
        raw_candidates = sorted({_resolve(Path(b)) for b in fleet_census.discover_banks(fleet_root)})
    else:
        raw_candidates = []

    failures: List[Dict[str, Any]] = []
    ordered: List[PeerSpec] = []
    seen: set = set()

    def _precheck(path: Path) -> bool:
        """True when ``path`` should be probed; records any read failure.

        Budget-aware: past ``deadline`` the candidate is not probed at all
        and a ``total_budget_exceeded`` failure is recorded instead, so the
        scope walk cannot outrun the cap the caller started before calling
        here. The lock wait is clamped to what is left of the budget, never
        left at the engine's 5 s default.
        """
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            failures.append(_budget_failure(path, total_budget_s))
            return False
        lock_timeout = probe_timeout_s if remaining is None else min(probe_timeout_s, remaining)
        should_probe, failure = _check_bank_retrying(
            path, retries=retries, backoff_s=backoff_s, sleep=sleep,
            lock_timeout_s=lock_timeout,
            deadline=deadline, total_budget_s=total_budget_s)
        if failure is not None:
            failures.append(failure)
        return should_probe

    def _add(path: Path) -> None:
        key = str(path)
        if key in seen or path == home:
            return
        # Mark the path decided BEFORE the precheck: a candidate that is
        # skipped (not a bank, or out of budget) must not be re-decided when
        # the same file turns up again in the candidate list, or a budget
        # overrun would record one failure per duplicate spelling of it.
        seen.add(key)
        if not _precheck(path):
            return
        ordered.append(PeerSpec(
            path=key,
            is_shared=surface is not None and key == str(surface),
            is_root=root_bank is not None and key == str(root_bank)
            and not (surface is not None and key == str(surface)),
        ))

    if surface is not None:
        _add(surface)
    if root_bank is not None:
        _add(root_bank)
    for candidate in raw_candidates:
        _add(candidate)

    ordered.sort(key=lambda peer: (peer.rank, peer.path))
    return ordered, failures


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def _select_canonical(chain: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """R7: shared > root > oldest profile, deterministically.

    Chain entries carry the peer's rank from its :class:`PeerSpec`; profile
    entries are ordered by ``conflicts.created_at`` (naive UTC, so lexical
    order is chronological), then path, then id — a total order, so the
    selection cannot depend on dict or filesystem order. A NULL
    ``created_at`` sorts last: unknown age is not treated as "oldest",
    which would hand canonicality to a row whose age nobody knows.
    """
    if not chain:
        return None

    def _key(entry: Dict[str, Any]) -> Tuple[int, int, str, str, int]:
        rank = int(entry.get("_rank", 2))
        created = entry.get("created_at")
        return (
            rank,
            1 if created is None else 0,
            created or "",
            str(entry.get("bank_path") or ""),
            int(entry.get("id") or 0),
        )

    return min(chain, key=_key)


def probe_fleet(
    pair: Tuple[str, str],
    home_bank: Path,
    *,
    fleet_root: Optional[Path] = None,
    banks: Optional[Sequence[Path]] = None,
    shared_db: Optional[Path] = None,
    root_db: Optional[Path] = None,
    per_probe_timeout_s: Optional[float] = None,
    retries: int = DEFAULT_RETRIES,
    backoff_s: float = DEFAULT_BACKOFF_S,
    total_budget_s: Optional[float] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> HandshakeDecision:
    """Probe the fleet for ``pair`` and decide whether the home insert may run.

    Args:
        pair: ``(fact_a_id, fact_b_id)`` — normalized here to
            ``(min, max)``, R2's key. Orientation of the caller's arguments
            cannot change the outcome.
        home_bank: The upserting bank. Never probed. Its location also
            derives the fleet root when neither ``fleet_root`` nor
            ``MNEMOSYNE_FLEET_ROOT`` is given (:func:`resolve_fleet_root`).
        fleet_root: Explicit fleet root; overrides env and derivation.
        banks: Explicit peer candidates (tests, and callers that already
            know the set). When given, the census walk is skipped.
        shared_db, root_db: Explicit surface/root bank paths. Defaults
            mirror the census (:func:`fleet_census.shared_db_path`) and
            :func:`root_db_path`.
        per_probe_timeout_s: R8 §(a) budget, in seconds. Defaults to the env
            value (:data:`PROBE_TIMEOUT_ENV`) or 250 ms. Must be finite and
            positive; a configured value that is not raises ``ValueError``
            rather than being silently replaced by a default.
        retries, backoff_s: R8 §(c) — 1 retry × 500 ms by default.
        total_budget_s: R8 §(b) cap. ``None`` uses the resolved default
            (:func:`total_budget_seconds`, 2 s unless overridden); a positive
            number sets the cap; ``0`` or negative means unbounded, matching
            the env convention (where ``0``/``none``/``off`` are the
            documented sentinels). ``inf``/``nan`` raise.
        sleep: Injectable backoff sleep (tests pass a no-op).

    Returns:
        :class:`HandshakeDecision`. ``decision.proceed`` gates the insert;
        ``decision.refused`` is the fail-closed case; ``decision.chain`` is
        the audit chain; ``decision.audit_events()`` is what to persist.
    """
    timeout_s = per_probe_timeout_s if per_probe_timeout_s is not None else per_probe_timeout_seconds()
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError(
            f"per_probe_timeout_s={per_probe_timeout_s!r} is invalid: expected "
            f"a finite, positive number of seconds"
        )
    if total_budget_s is None:
        budget_s = total_budget_seconds()
    elif not math.isfinite(total_budget_s):
        raise ValueError(
            f"total_budget_s={total_budget_s!r} is invalid: expected a finite "
            f"number of seconds, or 0/negative for an unbounded sequence"
        )
    elif total_budget_s > 0:
        budget_s = total_budget_s
    else:
        # 0/negative == unbounded, matching the env convention.
        budget_s = None

    # R2's key. The census owns normalization; reusing its function (rather
    # than re-spelling it) is what keeps this probe's key from drifting from
    # the schema invariant and from the census's own pair accounting.
    normalized = fleet_census._norm_pair(str(pair[0]), str(pair[1]))
    home = _resolve(Path(home_bank))
    root_resolved = resolve_fleet_root(home, fleet_root)

    _utc_now = datetime.now(timezone.utc)
    _local_now = _utc_now.astimezone()
    offset = _local_now.utcoffset()

    decision = HandshakeDecision(
        pair=normalized,
        home_bank=str(home),
        read_at_utc=_utc_now.replace(tzinfo=None).isoformat(),
        read_at_local=_local_now.replace(tzinfo=None).isoformat(),
        local_zone=_local_now.tzname() or "local",
        utc_offset_seconds=int(offset.total_seconds()) if offset else 0,
    )

    # R8's clock starts HERE, before scope resolution: the scope's own
    # liveness reads retry and back off, and they run while the home bank's
    # write lock is held. Starting the clock after scope (as this did before)
    # left that stretch unbudgeted, so a candidate set full of unreadable
    # banks could hold the lock well past the cap the operator configured.
    started = time.monotonic()
    deadline = None if budget_s is None else started + budget_s

    peers, scope_failures = _probe_scope(
        home,
        fleet_root=root_resolved,
        banks=banks,
        shared_db=shared_db,
        root_db=root_db,
        retries=retries,
        backoff_s=backoff_s,
        sleep=sleep,
        probe_timeout_s=timeout_s,
        deadline=deadline,
        total_budget_s=budget_s,
    )
    decision.failed.extend(scope_failures)
    decision.refused = bool(scope_failures)

    budget_exceeded_recorded = any(
        failure.get("error_class") == "total_budget_exceeded"
        for failure in decision.failed
    )
    chain: List[Dict[str, Any]] = []

    for peer in peers:
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            decision.failed.append(_budget_failure(Path(peer.path), budget_s))
            budget_exceeded_recorded = True
            decision.refused = True
            continue
        hits, failure = _probe_one(
            Path(peer.path),
            normalized,
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
            sleep=sleep,
            lock_timeout_s=remaining,
            deadline=deadline,
            total_budget_s=budget_s,
        )
        decision.probes_run += 1
        decision.probed_banks.append(peer.path)
        if failure is not None:
            if failure.get("error_class") == "total_budget_exceeded":
                budget_exceeded_recorded = True
            decision.failed.append(failure)
            decision.refused = True
            continue
        for hit in hits:
            entry = dict(hit)
            entry["_rank"] = peer.rank
            entry["is_shared"] = peer.is_shared
            entry["is_root"] = peer.is_root
            chain.append(entry)

    decision.elapsed_ms = (time.monotonic() - started) * 1000.0
    decision.chain = chain

    # Recheck AFTER the loop. The pre-probe guard only asks whether a probe
    # may START; a last probe that started inside the budget and finished
    # past it would otherwise still report `proceed`, and the caller would
    # insert a conflict row whose probe sequence overran R8's cap.
    if (deadline is not None and not budget_exceeded_recorded
            and time.monotonic() > deadline):
        decision.failed.append(_budget_failure(
            Path(decision.probed_banks[-1]) if decision.probed_banks else None,
            budget_s,
        ))
        decision.refused = True

    if decision.refused:
        # Fail closed: no home insert, whatever the chain holds. The chain is
        # still reported so a reader sees what was observed before the
        # failure rather than an empty picture.
        return decision

    canonical = _select_canonical(chain)
    if canonical is not None:
        decision.federated_to = f"{canonical['id']}@{canonical['bank_path']}"
    return decision