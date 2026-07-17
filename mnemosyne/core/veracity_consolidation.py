"""
Mnemosyne Veracity-Weighted Consolidation
=========================================
Our novel contribution: Bayesian confidence scoring + conflict resolution.

Veracity tiers:
- stated:     1.0  (user explicitly stated)
- inferred:   0.7  (inferred from context)
- tool:       0.5  (tool output, may be stale)
- imported:   0.6  (imported from external source)
- unknown:    0.8  (default, unverified)

Bayesian updating:
- confidence = 1 - (0.7^n) where n = mention count
- More mentions = higher confidence
- Contradictions detected and flagged

Conflict resolution:
- Same subject + predicate = potential conflict
- Higher confidence wins
- Lower confidence flagged for review
- Consolidation: periodic synthesis of high-confidence facts
"""

import contextlib
import hashlib
import logging
import sqlite3
import json
import threading
import unicodedata
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path


def compute_fact_id(subject: str, predicate: str, object: str) -> str:
    """Deterministic ID for a (subject, predicate, object) tuple.

    Pre-fix `consolidated_facts.id` used
    ``f"cf_{subject}_{predicate}_{object}".replace(" ", "_")[:100]``.
    The 100-char truncation silently collided on long content: two
    distinct facts with the same first ~95 chars after replace
    produced identical PKs. The IntegrityError that resulted from
    the INSERT was swallowed by the calling code at
    `beam.py:_ingest_graph_and_veracity` (broad `except: pass` in
    Phase 4), producing silent data loss.

    Post-fix: SHA-256 hash of NFC-normalized SPO with length-prefix
    framing. Always-uniform 27 chars (`cf_` + 24 hex). Properties:

      - **Collision-safe across content lengths.** Hash never
        truncates the input -- long SPOs are encoded fully.
      - **Smuggle-safe.** Length-prefix framing (``b"3:foo4:isax"``)
        makes the encoding injective: two distinct SPO tuples can
        never produce the same byte string regardless of whether a
        component contains a separator-like character.
        ``compute_fact_id("a\\x1f", "b", "c")`` no longer collides
        with ``compute_fact_id("a", "\\x1fb", "c")``.
      - **Unicode-stable.** NFC normalization applied per field so
        ``"café"`` (NFC) and ``"café"`` (NFD) hash identically.
      - **Codebase-consistent.** SHA-256 matches the digest choice
        used elsewhere (`beam.py:770`, `importers/base.py:181`,
        `importers/hindsight.py:144`).

    Two facts with the same SPO produce the same ID (idempotency
    preserved -- `consolidate_fact`'s dedup still relies on SPO
    equality, not ID equality). Distinct facts produce distinct
    IDs regardless of content length.

    Backward compat: existing rows keep their stored pre-fix IDs.
    `consolidate_fact` continues to dedup by SPO match (the
    SELECT WHERE subject=? AND predicate=? AND object=? at line
    matching pre-fix), so old rows are found correctly on UPDATE.
    Only newly inserted rows get the new format. Cross-format DBs
    work indefinitely. `_fact_voice` reads the stored row id
    (via ConsolidatedFact.id) rather than recomputing -- preserves
    RRF-key alignment for legacy rows.

    /review history:
      - Codex adv + Perf + Claude (3-source MED on E2 PR #82)
        caught the original collision risk.
      - Codex structured + Codex adv + Maintainability (3-source
        GATE FAIL on this PR's commit 1) caught the \\x1f smuggling
        + SHA-1/codebase-inconsistency + missing Unicode norm.

    Raises:
        TypeError: if any of subject/predicate/object is not a str.
        ValueError: if any of subject/predicate/object is empty.
    """
    for name, value in (
        ("subject", subject),
        ("predicate", predicate),
        ("object", object),
    ):
        if not isinstance(value, str):
            raise TypeError(
                f"compute_fact_id: {name} must be a str, got "
                f"{type(value).__name__}"
            )
        if value == "":
            raise ValueError(
                f"compute_fact_id: {name} must be non-empty"
            )
    # NFC normalize each component so different normalization forms
    # of the same logical text produce the same ID. Length-prefix
    # framing makes the encoding injective -- different SPO tuples
    # cannot share a byte representation regardless of in-field
    # separator characters.
    parts: List[bytes] = []
    for value in (subject, predicate, object):
        b = unicodedata.normalize("NFC", value).encode("utf-8")
        parts.append(f"{len(b)}:".encode("ascii") + b)
    return "cf_" + hashlib.sha256(b"".join(parts)).hexdigest()[:24]


logger = logging.getLogger(__name__)


# Veracity weights
VERACITY_WEIGHTS = {
    "stated": 1.0,
    "inferred": 0.7,
    "tool": 0.5,
    "imported": 0.6,
    "unknown": 0.8,
}

# Canonical allowlist for trust-boundary clamping. Anything outside this
# set bypasses the recall weighting (VERACITY_WEIGHTS.get(..., 0.8) falls
# back to the 'unknown' weight) AND pollutes the contamination filter
# downstream (which compares `veracity != 'stated'`). Callers at the
# trust boundary (LLM output, importers, MCP tool args, batch ingest)
# should clamp via clamp_veracity() so non-canonical labels don't
# persist as garbage in the row.
VERACITY_ALLOWED = frozenset(VERACITY_WEIGHTS.keys())


# Cap on the raw value included in the WARNING log. Without this, an
# importer pushing 100k items with embedded long strings as veracity
# values can flood log aggregators (cost) AND leak user content into
# operator logs (privacy). 80 chars is enough to debug typos / case
# issues without being a privacy or storage hazard.
_VERACITY_WARN_VALUE_CAP = 80


def clamp_veracity(raw, *, context: str = "veracity") -> str:
    """Normalize and clamp a veracity label to the canonical allowlist.

    Behavior:
        - None / empty / whitespace → 'unknown' silently
        - Case-and-whitespace normalize then match against VERACITY_ALLOWED
        - Anything else → 'unknown' with a WARNING log (raw value
          truncated to %d chars to bound log volume)

    `context` appears in the warning so the operator can see where
    the bad label came from (e.g. 'remember_batch.default',
    'remember_batch.per_item', 'mnemosyne_remember').
    """ % _VERACITY_WARN_VALUE_CAP
    if raw is None:
        return "unknown"
    norm = str(raw).strip().lower()
    if not norm:
        return "unknown"
    if norm in VERACITY_ALLOWED:
        return norm
    # Truncate the raw value for the log line. %r quoting prevents
    # control-character injection into log aggregators; the cap
    # prevents log-flood and content leakage from upstream typos.
    raw_str = str(raw)
    if len(raw_str) > _VERACITY_WARN_VALUE_CAP:
        raw_for_log = raw_str[:_VERACITY_WARN_VALUE_CAP] + "...[truncated]"
    else:
        raw_for_log = raw_str
    logger.warning(
        "%s received unknown veracity %r; clamping to 'unknown'",
        context, raw_for_log,
    )
    return "unknown"


def aggregate_veracity(source_veracities) -> str:
    """E4.a.1: aggregate per-source veracity labels into one summary label.

    Used at consolidation time when a `sleep()` summarizes N working_memory
    rows into one episodic_memory row. Pre-fix the episodic INSERT didn't
    set the veracity column, so it defaulted to 'unknown' (0.8 multiplier)
    even when every source row was 'stated' (1.0). That destroys the
    trust signal `remember_batch` had populated per-row.

    Aggregation rule:
        - Empty / no-valid-labels input → 'unknown' (matches schema default)
        - 'unknown' is treated as low-priority: only counted when NO
          non-'unknown' canonical labels are present in the source set.
          Rationale: the column default is 'unknown', so legacy rows that
          never had veracity explicitly set store the literal 'unknown'
          string. The aggregator can't distinguish "operator marked unknown"
          from "nobody set it" -- letting 'unknown' dilute confident
          signals systematically deflates summaries from mixed-vintage
          sessions (review H1, 2026-05-11).
        - Within the candidates (non-'unknown' if any, else all-'unknown'):
            * Single distinct label → that label
            * Clear majority → the majority label
            * Multi-way tie → the label with the LOWEST weight (most
              conservative; e.g. {stated, tool} ties resolve to 'tool')

    Rationale: a summary inherits confidence of its sources collectively.
    Homogeneous-stated sources should produce stated summaries (don't
    deflate). Mixed sources reflect the most-common signal. Tied multi-way
    sources resolve toward caution rather than overclaiming.

    Non-canonical labels in input (e.g., raw LLM output bleeding through
    a caller's clamp) are silently dropped from the count -- operators
    rely on `clamp_veracity()` at the trust boundary; the aggregator is
    not a clamp, it's a reducer.

    Args:
        source_veracities: iterable of veracity strings (typically from
            a SELECT on working_memory.veracity for source rows).

    Returns:
        One canonical veracity label (member of VERACITY_ALLOWED).
    """
    if not source_veracities:
        return "unknown"
    # Filter to canonical labels; non-canonical values don't vote.
    valid = [v for v in source_veracities if v in VERACITY_ALLOWED]
    if not valid:
        return "unknown"
    # H1 review fix: 'unknown' is the schema default -- treat as
    # low-priority so legacy rows don't dilute confident signals.
    non_unknown = [v for v in valid if v != "unknown"]
    candidates = non_unknown if non_unknown else valid
    # Mode (most-frequent label) among candidates.
    counts: Dict[str, int] = {}
    for v in candidates:
        counts[v] = counts.get(v, 0) + 1
    max_count = max(counts.values())
    most_common = [v for v, c in counts.items() if c == max_count]
    if len(most_common) == 1:
        return most_common[0]
    # Tie: pick the most conservative (lowest weight).
    return min(most_common, key=lambda v: VERACITY_WEIGHTS[v])


@dataclass
class ConsolidatedFact:
    """A fact that has been through consolidation."""
    subject: str
    predicate: str
    object: str
    confidence: float
    mention_count: int
    first_seen: str
    last_seen: str
    sources: List[str]
    veracity: str
    superseded: bool = False
    # Stored `consolidated_facts.id` -- present for rows fetched from
    # the DB, None for transient ConsolidatedFact returns from
    # consolidate_fact (which carries the value but doesn't re-roundtrip
    # through this dataclass for the inserted row's id). Consumers like
    # `polyphonic_recall._fact_voice` should use this when available so
    # legacy-format rows in mixed-format DBs keep their stored IDs as
    # RRF fusion keys instead of being recomputed to the new hash form.
    id: Optional[str] = None


class VeracityConsolidator:
    """
    Bayesian confidence consolidation with conflict detection.
    
    Builds on:
    - Memanto's conflict resolution (arXiv:2604.22085)
    - REMem's fact preservation (arXiv:2602.13530)
    - Our novel veracity-weighted Bayesian updating
    """
    
    def __init__(self, db_path: Path = None, conn=None):
        if conn is not None:
            self.conn = conn
            self.db_path = db_path or Path(":memory:")
        else:
            self.db_path = db_path or Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            # Apply the same PRAGMA settings BeamMemory's _get_connection
            # uses (journal_mode=WAL, busy_timeout=5000ms). Required for
            # `_serialized_write`'s `BEGIN IMMEDIATE` to behave correctly
            # under contention: without WAL the lock blocks readers; without
            # busy_timeout contention raises `database is locked` instantly
            # instead of waiting up to 5s. Duplicated from the E2.a.5 fix
            # so this branch is self-contained whether it lands before or
            # after #84. /review (Claude CRITICAL) caught the
            # branch-rebase dependency.
            try:
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA busy_timeout=5000")
            except sqlite3.Error:
                # Best-effort: in-memory or otherwise-constrained
                # environments may not support WAL. Continue.
                pass
        self.conn.row_factory = sqlite3.Row
        self._owns_connection = conn is None

        # Same-connection writer serialization. `BEGIN IMMEDIATE` provides
        # database-level serialization across CONNECTIONS, but two threads
        # sharing the same `VeracityConsolidator` instance (and therefore
        # the same `self.conn`) would both see `conn.in_transaction = True`
        # after the first thread's BEGIN -- defeating the nested-tx skip
        # logic and recreating the race within a single SQL transaction.
        # `RLock` (not `Lock`) so the contextmanager can recursively enter
        # for nested calls within the same thread (e.g.,
        # `run_consolidation_pass` calling `resolve_conflict_by_facts`).
        # /review (Codex structured P2 GATE FAIL) caught the same-conn race.
        self._write_lock = threading.RLock()

        self._init_tables()
    
    def _init_tables(self):
        """Initialize consolidation schema."""
        cursor = self.conn.cursor()
        
        # Consolidated facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS consolidated_facts (
                id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                mention_count INTEGER DEFAULT 1,
                first_seen TEXT,
                last_seen TEXT,
                sources_json TEXT,
                veracity TEXT DEFAULT 'unknown',
                superseded_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_subject ON consolidated_facts(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_predicate ON consolidated_facts(predicate)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cf_object ON consolidated_facts(object)")
        
        # Conflicts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_a_id TEXT NOT NULL,
                fact_b_id TEXT NOT NULL,
                conflict_type TEXT,
                resolution TEXT,
                resolved_at TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    @contextlib.contextmanager
    def _serialized_write(self):
        """Serialize the body's SELECT-then-write under the SQLite writer lock.

        Wraps the block in ``BEGIN IMMEDIATE`` when the connection is
        not already in a transaction. Concurrent calls queue on the
        writer lock rather than racing on SELECT-then-INSERT/UPDATE
        patterns. Nested invocations (caller already in a tx)
        participate in that tx without starting their own BEGIN.

        Shared by ``resolve_conflict``, ``resolve_conflict_by_facts``,
        and ``run_consolidation_pass`` so the four write methods
        (including ``consolidate_fact``) all use one canonical
        serialization pattern.

        Caller-contract caveat (per E2.a.7 ledger row): a DEFERRED
        outer transaction (Python sqlite3's default implicit tx) does
        NOT acquire the writer lock until its own first INSERT/UPDATE.
        Two threads each in their own DEFERRED outer tx can both pass
        the SELECT-no-match check before either writes -- the race
        window reopens inside the outer scope. Race safety inside a
        caller-owned outer tx requires either (a) the outer tx is
        ``BEGIN IMMEDIATE`` or ``BEGIN EXCLUSIVE``, OR (b) the caller
        is the only writer (e.g., E2's single-threaded batch
        enrichment loop).

        Raises whatever the body raises after attempting rollback (if
        we own the tx) so callers can implement retry / circuit-break
        policies.
        """
        # Capture conn at entry -- defense against the body swapping
        # self.conn (closing the original, reassigning, etc.). All of
        # commit/rollback/cursor must target the SAME connection we
        # opened BEGIN IMMEDIATE on. /review (Codex adversarial MED).
        conn = self.conn
        # Acquire instance lock BEFORE BEGIN IMMEDIATE so two threads
        # sharing this VeracityConsolidator instance (and therefore
        # this conn) serialize at the Python level -- SQLite's writer
        # lock alone doesn't protect them because they share the same
        # transaction once the first thread starts one.
        with self._write_lock:
            cursor = conn.cursor()
            started_tx = False
            if not conn.in_transaction:
                # Let OperationalError propagate. If `database is locked`
                # fires after busy_timeout, the caller's error handler is
                # the right place to decide what to do; silent fallthrough
                # would reintroduce the race we're closing.
                cursor.execute("BEGIN IMMEDIATE")
                started_tx = True
            try:
                yield
                if started_tx:
                    conn.commit()
            except Exception:
                if started_tx:
                    try:
                        conn.rollback()
                    except sqlite3.Error as rb_exc:
                        logger.error(
                            "_serialized_write: rollback failed after "
                            "error (connection may be in undefined state): %s",
                            rb_exc,
                        )
                raise

    def bayesian_update(self, current_confidence: float, veracity: str) -> float:
        """
        Update confidence using Bayesian formula.
        
        Formula: new_confidence = 1 - (0.7^n) where n = mention count
        But we approximate with: new = old + (1 - old) * veracity_weight * 0.3
        
        Args:
            current_confidence: Current confidence level
            veracity: Veracity tier
            
        Returns:
            float: Updated confidence
        """
        weight = VERACITY_WEIGHTS.get(veracity, 0.8)
        increment = (1.0 - current_confidence) * weight * 0.3
        return min(current_confidence + increment, 1.0)
    
    def consolidate_fact(self, subject: str, predicate: str, object: str,
                        veracity: str = "unknown", source: str = None) -> ConsolidatedFact:
        """
        Add or update a fact in consolidation.

        Args:
            subject: Fact subject
            predicate: Fact predicate
            object: Fact object
            veracity: Veracity tier
            source: Source memory ID

        Returns:
            ConsolidatedFact: The consolidated result

        Concurrency: SELECT-then-INSERT/UPDATE wrapped in
        ``_serialized_write``. Pre-fix two threads with same SPO
        could both pass the no-match SELECT and both attempt
        INSERT, raising IntegrityError. Bayesian confidence math
        is path-dependent so concurrent UPDATEs also raced.

        Post-fix: the helper's RLock serializes same-instance
        callers + `BEGIN IMMEDIATE` serializes cross-connection
        writers via the SQLite writer lock. Caller-contract caveat
        for DEFERRED outer txs: see `_serialized_write` docstring.

        Pre-DRY (PR #84) this method had its own inline
        BEGIN IMMEDIATE pattern. The helper-based version here
        unifies it with the other three write methods so all four
        share one canonical pattern AND the instance RLock
        protects all four (including consolidate_fact, which
        previously lacked RLock acquisition).
        """
        with self._serialized_write():
            cursor = self.conn.cursor()

            # Check if fact already exists
            cursor.execute("""
                SELECT * FROM consolidated_facts
                WHERE subject = ? AND predicate = ? AND object = ?
            """, (subject, predicate, object))

            row = cursor.fetchone()
            now = datetime.now().isoformat()

            if row:
                # Update existing fact
                new_confidence = self.bayesian_update(row["confidence"], veracity)
                new_count = row["mention_count"] + 1

                sources = json.loads(row["sources_json"] or "[]")
                if source and source not in sources:
                    sources.append(source)

                cursor.execute("""
                    UPDATE consolidated_facts
                    SET confidence = ?, mention_count = ?, last_seen = ?,
                        sources_json = ?, veracity = ?, updated_at = ?
                    WHERE id = ?
                """, (new_confidence, new_count, now, json.dumps(sources),
                      veracity, now, row["id"]))

                return ConsolidatedFact(
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    confidence=new_confidence,
                    mention_count=new_count,
                    first_seen=row["first_seen"],
                    last_seen=now,
                    sources=sources,
                    veracity=veracity
                )

            else:
                # Check for conflicts (same subject+predicate, different object)
                cursor.execute("""
                    SELECT * FROM consolidated_facts
                    WHERE subject = ? AND predicate = ? AND object != ?
                """, (subject, predicate, object))

                conflicts = cursor.fetchall()

                # Insert new fact. E2.a.4 (PR #83): use `compute_fact_id`
                # (SHA-256 of NFC-normalized SPO + length-prefix framing)
                # for collision-free deterministic IDs. The merge of
                # PR #83 accidentally left the OLD truncating f-string
                # pattern at this call site even though `compute_fact_id`
                # exists at line 38; same hunk that dropped the new logic
                # also dropped the call.
                fact_id = compute_fact_id(subject, predicate, object)
                base_confidence = VERACITY_WEIGHTS.get(veracity, 0.8) * 0.5

                sources = [source] if source else []

                cursor.execute("""
                    INSERT INTO consolidated_facts
                    (id, subject, predicate, object, confidence, mention_count,
                     first_seen, last_seen, sources_json, veracity)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (fact_id, subject, predicate, object, base_confidence, 1,
                      now, now, json.dumps(sources), veracity))

                # Record conflicts. Pass `commit=False` so the helper's
                # internal commit doesn't end our `_serialized_write`
                # transaction mid-loop -- see _record_conflict docstring
                # for the atomicity rationale.
                for conflict in conflicts:
                    self._record_conflict(
                        fact_id, conflict["id"], "contradiction",
                        commit=False,
                    )

                return ConsolidatedFact(
                    subject=subject,
                    predicate=predicate,
                    object=object,
                    confidence=base_confidence,
                    mention_count=1,
                    first_seen=now,
                    last_seen=now,
                    sources=sources,
                    veracity=veracity
                )

    def _record_conflict(self, fact_a_id: str, fact_b_id: str,
                         conflict_type: str, commit: bool = True):
        """Record a conflict between two facts.

        commit (default True): whether to call self.conn.commit() after
            the INSERT. `consolidate_fact` passes `commit=False` when
            invoking this helper from within its own `_serialized_write`
            scope so the fact INSERT and its conflict rows commit
            atomically as part of the outer transaction. Without this
            opt-out, _record_conflict's commit would end the outer
            transaction mid-loop, leaving the fact INSERT durable but
            allowing later conflict-record failures to leak partial
            state. /review (E2.a.5 4-source HIGH) caught this pattern
            in the inline version; preserved in the DRY refactor.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO conflicts (fact_a_id, fact_b_id, conflict_type)
            VALUES (?, ?, ?)
        """, (fact_a_id, fact_b_id, conflict_type))
        if commit:
            self.conn.commit()
    
    def resolve_conflict(self, conflict_id: int, winning_fact_id: str):
        """
        Resolve a conflict by marking the losing fact as superseded.

        Args:
            conflict_id: Conflict to resolve
            winning_fact_id: The fact that wins

        Concurrency: SELECT-then-write pattern wrapped in
        ``_serialized_write``. Pre-fix two concurrent
        ``resolve_conflict`` calls on the same conflict_id with
        different winning_fact_ids could both pass the SELECT and
        last-writer-wins on the UPDATE, leaving BOTH facts
        superseded (each marking the other). Post-fix the second
        call sees the first's commit and either confirms the same
        winner or finds the conflict already resolved.
        """
        with self._serialized_write():
            cursor = self.conn.cursor()

            # Get conflict details
            cursor.execute(
                "SELECT * FROM conflicts WHERE id = ?", (conflict_id,)
            )
            conflict = cursor.fetchone()

            if not conflict:
                return

            # Already-resolved guard: first-writer-wins semantics. Pre-fix
            # serialization alone didn't fix the case where two callers
            # passed different winning_fact_id values -- even with BEGIN
            # IMMEDIATE the second call would still mark the OTHER fact
            # superseded (the conflict's read returned the same fact ids
            # both times). With the guard, the second call sees the
            # conflict is resolved and returns without overwriting. Log
            # a WARNING so operators can spot conflicting writes.
            # /review (Codex adv + Maintainability + Claude on E2.a.5)
            # flagged the same-conflict-id race; the regression test
            # for E2.a.6 surfaced this additional gap.
            if conflict["resolution"] is not None:
                logger.warning(
                    "resolve_conflict: conflict %d already resolved "
                    "(resolution=%r); ignoring re-resolution attempt "
                    "with winning_fact_id=%r",
                    conflict_id, conflict["resolution"], winning_fact_id,
                )
                return

            # Reject ambiguous calls: the winning id must match one of
            # the conflict's stored fact ids exactly. Pre-fix the
            # comparison silently defaulted to fact_b_id as the loser
            # whenever winning_fact_id != fact_a_id, which produced the
            # wrong supersession when callers passed a derived-but-stale
            # id (legacy/new format divergence).
            fact_a_id = conflict["fact_a_id"]
            fact_b_id = conflict["fact_b_id"]
            if winning_fact_id == fact_a_id:
                losing_id = fact_b_id
            elif winning_fact_id == fact_b_id:
                losing_id = fact_a_id
            else:
                logger.warning(
                    "resolve_conflict: winning_fact_id %r matches neither "
                    "fact_a_id %r nor fact_b_id %r; declining to resolve",
                    winning_fact_id, fact_a_id, fact_b_id,
                )
                return

            # Mark as superseded
            now = datetime.now().isoformat()
            cursor.execute(
                """
                UPDATE consolidated_facts
                SET superseded_by = ?, updated_at = ?
                WHERE id = ?
                """,
                (winning_fact_id, now, losing_id),
            )

            # Mark conflict as resolved
            cursor.execute(
                """
                UPDATE conflicts
                SET resolution = ?, resolved_at = ?
                WHERE id = ?
                """,
                (f"superseded_by_{winning_fact_id}", now, conflict_id),
            )

            # `_serialized_write` commits on context exit when we own
            # the tx; participates in caller-owned tx otherwise.
    
    def get_conflicts(self) -> List[Dict]:
        """Get all unresolved conflicts."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM conflicts WHERE resolution IS NULL
            ORDER BY created_at DESC
        """)
        
        conflicts = []
        for row in cursor.fetchall():
            conflicts.append({
                "id": row["id"],
                "fact_a_id": row["fact_a_id"],
                "fact_b_id": row["fact_b_id"],
                "type": row["conflict_type"],
                "created_at": row["created_at"]
            })
        
        return conflicts
    
    def get_consolidated_facts(self, subject: str = None, min_confidence: float = 0.5) -> List[ConsolidatedFact]:
        """
        Get consolidated facts, optionally filtered by subject and confidence.
        
        Args:
            subject: Filter by subject
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of ConsolidatedFact
        """
        cursor = self.conn.cursor()
        
        if subject:
            cursor.execute("""
                SELECT * FROM consolidated_facts
                WHERE subject = ? AND confidence >= ? AND superseded_by IS NULL
                ORDER BY confidence DESC, mention_count DESC
            """, (subject, min_confidence))
        else:
            cursor.execute("""
                SELECT * FROM consolidated_facts
                WHERE confidence >= ? AND superseded_by IS NULL
                ORDER BY confidence DESC, mention_count DESC
            """, (min_confidence,))
        
        facts = []
        for row in cursor.fetchall():
            facts.append(ConsolidatedFact(
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                confidence=row["confidence"],
                mention_count=row["mention_count"],
                first_seen=row["first_seen"],
                last_seen=row["last_seen"],
                sources=json.loads(row["sources_json"] or "[]"),
                veracity=row["veracity"],
                superseded=row["superseded_by"] is not None,
                # Preserve the stored id (pre-fix legacy or post-fix
                # hash form) so callers like polyphonic_recall._fact_voice
                # use it for RRF keys instead of recomputing -- keeps
                # mixed-format DBs internally consistent.
                id=row["id"],
            ))
        
        return facts
    
    def get_high_confidence_summary(self, subject: str, threshold: float = 0.8) -> str:
        """
        Generate a summary of high-confidence facts about a subject.
        
        Args:
            subject: Subject to summarize
            threshold: Confidence threshold
            
        Returns:
            str: Human-readable summary
        """
        facts = self.get_consolidated_facts(subject, min_confidence=threshold)
        
        if not facts:
            return f"No high-confidence facts about {subject}."
        
        lines = [f"High-confidence facts about {subject}:"]
        for fact in facts:
            lines.append(f"  - {fact.subject} {fact.predicate} {fact.object} "
                        f"(conf: {fact.confidence:.2f}, mentions: {fact.mention_count})")
        
        return "\n".join(lines)
    
    def run_consolidation_pass(self):
        """
        Background consolidation pass.

        1. Find facts with multiple mentions
        2. Boost confidence
        3. Detect conflicts
        4. Auto-resolve obvious conflicts (higher confidence wins)

        Concurrency: wrapped in ``_serialized_write`` so a concurrent
        write (e.g., `consolidate_fact` or `resolve_conflict`) doesn't
        interleave with the pass's read-decide-resolve loop. Inner
        ``resolve_conflict_by_facts`` calls participate in this scope's
        transaction (their own ``_serialized_write`` will detect the
        outer tx and skip BEGIN).
        """
        with self._serialized_write():
            cursor = self.conn.cursor()

            # Find facts ready for consolidation (mention_count > 2)
            cursor.execute(
                """
                SELECT * FROM consolidated_facts
                WHERE mention_count > 2 AND superseded_by IS NULL
                ORDER BY mention_count DESC
                """
            )

            # Materialize the row list before iterating + writing --
            # mixing fetch with writes on the same cursor can confuse
            # the iteration state under some sqlite3 builds.
            primary_rows = cursor.fetchall()

            for row in primary_rows:
                subject = row["subject"]
                predicate = row["predicate"]

                # Find conflicts
                cursor.execute(
                    """
                    SELECT * FROM consolidated_facts
                    WHERE subject = ? AND predicate = ? AND object != ?
                    AND superseded_by IS NULL
                    """,
                    (subject, predicate, row["object"]),
                )

                conflicts = cursor.fetchall()
                for conflict in conflicts:
                    # Auto-resolve: higher confidence wins
                    if row["confidence"] > conflict["confidence"]:
                        self.resolve_conflict_by_facts(
                            row["id"], conflict["id"]
                        )

    def resolve_conflict_by_facts(self, winning_id: str, losing_id: str):
        """Resolve conflict by marking losing fact as superseded.

        Concurrency: this is a single-statement UPDATE so it doesn't
        have the SELECT-then-write race shape of ``resolve_conflict``.
        Still wrapped in ``_serialized_write`` for consistency: a
        concurrent ``resolve_conflict`` on the same losing_id would
        interleave on the `superseded_by` column, and the wrap makes
        the override deterministic (later writer wins after the
        earlier one commits, instead of both racing inside their
        respective reads).
        """
        with self._serialized_write():
            now = datetime.now().isoformat()
            cursor = self.conn.cursor()

            cursor.execute(
                """
                UPDATE consolidated_facts
                SET superseded_by = ?, updated_at = ?
                WHERE id = ?
                """,
                (winning_id, now, losing_id),
            )
    
    def get_stats(self) -> Dict:
        """Get consolidation statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NULL")
        active_facts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM consolidated_facts WHERE superseded_by IS NOT NULL")
        superseded_facts = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM conflicts WHERE resolution IS NULL")
        unresolved_conflicts = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(confidence) FROM consolidated_facts WHERE superseded_by IS NULL")
        avg_confidence = cursor.fetchone()[0] or 0.0
        
        cursor.execute("SELECT AVG(mention_count) FROM consolidated_facts WHERE superseded_by IS NULL")
        avg_mentions = cursor.fetchone()[0] or 0.0
        
        return {
            "active_facts": active_facts,
            "superseded_facts": superseded_facts,
            "unresolved_conflicts": unresolved_conflicts,
            "avg_confidence": round(avg_confidence, 3),
            "avg_mentions": round(avg_mentions, 2),
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# --- Testing ---
if __name__ == "__main__":
    import tempfile
    import os
    
    print("Veracity Consolidation Tests")
    print("=" * 60)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    cons = VeracityConsolidator(db_path=Path(db_path))
    
    # Test 1: Basic consolidation
    print("\nTest 1: Basic consolidation")
    fact1 = cons.consolidate_fact("Alice", "is", "developer", "stated", "mem_001")
    print(f"  Initial: {fact1.subject} {fact1.predicate} {fact1.object} (conf: {fact1.confidence:.2f})")
    
    # Test 2: Bayesian update
    print("\nTest 2: Bayesian update")
    fact2 = cons.consolidate_fact("Alice", "is", "developer", "stated", "mem_002")
    print(f"  Updated: {fact2.subject} {fact2.predicate} {fact2.object} (conf: {fact2.confidence:.2f}, mentions: {fact2.mention_count})")
    
    # Test 3: Conflict detection
    print("\nTest 3: Conflict detection")
    fact3 = cons.consolidate_fact("Alice", "is", "manager", "inferred", "mem_003")
    print(f"  Conflict: {fact3.subject} {fact3.predicate} {fact3.object} (conf: {fact3.confidence:.2f})")
    
    conflicts = cons.get_conflicts()
    print(f"  Unresolved conflicts: {len(conflicts)}")
    
    # Test 4: Conflict resolution
    # ID format changed from f-string to hash (collision safety fix).
    # Use compute_fact_id rather than hard-coding "cf_Alice_is_developer".
    print("\nTest 4: Conflict resolution")
    if conflicts:
        cons.resolve_conflict(
            conflicts[0]["id"],
            compute_fact_id("Alice", "is", "developer"),
        )
        print(f"  Resolved conflict #{conflicts[0]['id']}")
    
    # Test 5: High-confidence summary
    print("\nTest 5: High-confidence summary")
    summary = cons.get_high_confidence_summary("Alice", threshold=0.5)
    print(summary)
    
    # Test 6: Stats
    print("\nTest 6: Stats")
    stats = cons.get_stats()
    print(f"  {stats}")
    
    # Cleanup
    cons.close()
    os.unlink(db_path)
    
    print("\n" + "=" * 60)
    print("Veracity consolidation tests passed!")
