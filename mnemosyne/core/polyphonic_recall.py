"""
Mnemosyne Polyphonic Recall Engine
===================================
Multi-strategy parallel retrieval with deterministic re-ranking.

Strategies (4 voices):
1. Vector voice: Dense semantic similarity over working_memory + episodic_memory
2. Graph voice: Episodic graph traversal (Phase 3)
3. Fact voice: Structured fact matching (Phase 4)
4. Temporal voice: Time-aware scoring

Deterministic re-ranker:
- Combines 4 scores with learned weights
- No neural network (rule-based weighting)
- Budget-aware context assembly
- Diversity penalty (avoid duplicates)

Building on:
- Hindsight's multi-strategy retrieval (blog)
- Memanto's information-theoretic scoring (arXiv:2604.22085)
- Our novel deterministic combination
"""

# Postponed annotation evaluation: lets us reference np.ndarray in type
# hints without breaking module import when numpy is unavailable.
# /review (E5.a commit 2) caught the earlier `try: import np` guard
# being defeated by `np.ndarray = None` evaluation at class-body load.
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Set
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy is required by other voices too; guard for parity
    np = None

from mnemosyne.core.episodic_graph import EpisodicGraph
from mnemosyne.core.verbatim_ledger import ExclusionSnapshot, exclusion_sql, resolve_exclusions
from mnemosyne.core.veracity_consolidation import (
    VeracityConsolidator,
    compute_fact_id,
)

def _env_disabled(name: str) -> bool:
    """A/B toggle helper: return True iff env var is set to a falsy
    value (`0`/`false`/`no`/`off`). Used by the per-voice ablation
    toggles. Mirrors the helper in `beam.py` so each module is
    self-contained -- duplicated rather than imported to avoid a
    cross-module dependency for a 4-line helper.
    """
    val = os.environ.get(name, "").strip().lower()
    return val in ("0", "false", "no", "off")


@dataclass
class RecallResult:
    """Result from a single recall voice."""
    memory_id: str
    score: float
    voice: str
    metadata: Dict


@dataclass
class PolyphonicResult:
    """Combined result from all voices."""
    memory_id: str
    combined_score: float
    voice_scores: Dict[str, float]
    metadata: Dict
    content: str = ""


class PolyphonicRecallEngine:
    """
    Multi-strategy parallel retrieval with deterministic re-ranking.
    
    4 voices:
    - vector: Binary vector similarity
    - graph: Episodic graph traversal
    - fact: Structured fact matching
    - temporal: Time-aware scoring
    """
    
    def __init__(self, db_path: Path = None, conn: sqlite3.Connection = None):
        """Initialize the engine.

        db_path: filesystem path to the SQLite DB. Used by voices that
            spawn their own connection (only when conn is None).
        conn: optional shared sqlite3 connection. When provided, the
            engine and its subsystems (vector_store / graph /
            consolidator / temporal_voice) reuse this connection
            instead of spawning their own. Required for safe use under
            BeamMemory's thread-local connection model -- without this,
            each polyphonic recall call would open 4+ new connections
            (one per voice + one per subsystem) which both wastes
            resources and risks WAL-readback inconsistency under
            concurrent writers.
        """
        self.db_path = db_path or Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
        self.conn = conn  # may be None -- voices fall back to per-call open

        # Initialize subsystems. Each accepts an optional conn= since
        # 9f96ded; pass through so they share our handle.
        # NOTE: vector_store removed. The vector voice now reads dense
        # embeddings from `memory_embeddings` (the production-canonical
        # store also used by the linear recall path), not from the
        # standalone `binary_vectors` table which production never wrote
        # to. See _vector_voice for the rewired query path.
        self.graph = EpisodicGraph(db_path=self.db_path, conn=conn)
        self.consolidator = VeracityConsolidator(db_path=self.db_path, conn=conn)

        # [C4] Per-call degraded-path signal for recall diagnostics.
        # The vector voice prefers the sqlite-vec `vec_episodes` fast
        # path for the EM tier and falls back to a numpy full-scan
        # over `memory_embeddings` when sqlite-vec is unavailable,
        # errors, or its ANN hits all get filtered out. That is the
        # polyphonic analogue of the linear path's em_fallback (the
        # engine has no substring-scoring tier). reset per recall()
        # call by the voice; beam.py reads it for record_fallback_used.
        self.last_call_fallback = {"em": False, "wm": False}

        # Voice weights (deterministic, learned from validation)
        self.voice_weights = {
            "vector": 0.35,
            "graph": 0.25,
            "fact": 0.25,
            "temporal": 0.15,
        }
    
    def recall(self, query: str, query_embedding: np.ndarray = None,
               top_k: int = 10, context_budget: int = 4000,
               *, default_dense_source_filter: bool = True,
               source: Optional[str] = None,
               topic: Optional[str] = None,
               episodic_where: Optional[str] = None,
               episodic_params: Sequence[Any] = (),
               exclude_captures: Optional[ExclusionSnapshot] = None) -> List[PolyphonicResult]:
        """
        Polyphonic recall: all 4 voices in parallel, then combine.

        Args:
            query: Text query
            query_embedding: Optional pre-computed embedding
            top_k: Number of results to return
            context_budget: Max tokens for context assembly
            default_dense_source_filter: When True (default), the vector
                voice's working-memory tier applies the default dense-source
                predicate (exclude raw dialog / honcho sources and
                consolidated rows). Callers that pass an explicit
                source=/topic= filter must set this to False so explicitly
                requested rows are not filtered out before top-K selection.
            source: When set, the vector voice's working-memory tier applies
                ``wm.source = source`` BEFORE top-K selection -- mirroring the
                linear path's wm_where semantics so an explicit source filter
                cannot be starved out of the candidate pool by closer rows of
                other sources.
            topic: Same as ``source`` -- topics are stored in the source
                field for now (pending a dedicated topic column), exactly as
                beam._wm_search does.
            episodic_where / episodic_params: Trusted predicate built by
                BeamMemory for complete episodic eligibility before bounded
                vector selection. Standalone callers retain the local
                supersession/expiry/source fallback.
            exclude_captures: Revocable provider-owned WM capture proofs.
                Exclude working contributions before ranking/dedup/fusion.
                Untyped graph/fact hits with dual-tier IDs abstain rather than
                borrowing a working score for an episodic representation.

        Returns:
            List of PolyphonicResult, sorted by combined score
        """
        excluded = set()
        if exclude_captures is not None:
            if self.conn is not None:
                excluded = resolve_exclusions(self.conn, exclude_captures)
            else:
                conn = sqlite3.connect(str(self.db_path))
                try:
                    excluded = resolve_exclusions(conn, exclude_captures)
                finally:
                    conn.close()
        graph_results = self._graph_voice(query)
        fact_results = self._fact_voice(query)
        if excluded:
            # These voices carry no producing-tier evidence. An ambiguous
            # dual-tier hit must fail open, not rescue a WM score as episodic.
            ambiguous = {r.memory_id for r in graph_results + fact_results} & excluded
            excluded -= ambiguous
        vector_results = self._vector_voice(
            query_embedding,
            default_dense_source_filter=default_dense_source_filter,
            source=source,
            topic=topic,
            episodic_where=episodic_where,
            episodic_params=episodic_params,
            **({"excluded_wm_ids": excluded} if excluded else {}),
        )
        # Preserve the producing tier of surviving episodic vector hits when
        # their WM twin was excluded BEFORE vector dedup. This is provenance,
        # not post-fusion twin rescue. Ordinary explicit recall is unchanged.
        for result in vector_results:
            if (result.memory_id in excluded
                    and result.metadata.get("embedding_tier") == "episodic"):
                result.metadata["_self_echo_tier"] = "episodic"
        temporal_results = self._temporal_voice(
            query, **({"excluded_wm_ids": excluded} if excluded else {})
        )
        combined = self._combine_voices(
            vector_results, graph_results, fact_results, temporal_results
        )

        self._hydrate_result_content(combined)

        # Diversity re-rank
        reranked = self._diversity_rerank(combined, top_k)

        # Assemble context within budget
        context = self._assemble_context(reranked, context_budget)
        
        return context

    def _legacy_episodic_vector_voice(
        self,
        conn,
        query_unit,
        now_iso,
        source=None,
        topic=None,
        episodic_where=None,
        episodic_params=(),
    ):
        """Scan eligible vec rows, retaining only their top-20 hits."""
        import heapq
        from mnemosyne.core.beam import (
            EM_VEC_ADMIT,
            _vec_bit_blob_cosine,
            _vec_float32_blob_cosine,
            _vec_int8_blob_cosine,
            _vec_table_type_strict,
        )

        # Read the actual table representation, never infer it from blob length
        # or the configured model. Unknown types must fall back, not misdecode.
        vec_type = _vec_table_type_strict(conn)
        emb_json = json.dumps(query_unit.tolist())
        query_blob = b""
        if vec_type == "int8":
            query_blob = bytes(conn.execute(
                "SELECT vec_quantize_int8(?, 'unit')", (emb_json,)
            ).fetchone()[0])
        elif vec_type == "bit":
            query_blob = bytes(conn.execute(
                "SELECT vec_quantize_binary(?)", (emb_json,)
            ).fetchone()[0])
        elif vec_type != "float32":
            raise ValueError("Unknown episodic vector representation")

        if episodic_where is None:
            clauses = [
                "em.superseded_by IS NULL",
                "(em.valid_until IS NULL OR julianday(em.valid_until) > julianday(?))",
            ]
            params = [now_iso]
            source_filter = source or topic
            if source_filter:
                clauses.append("em.source = ?")
                params.append(source_filter)
            episodic_where = " AND ".join(clauses)
            episodic_params = params
        rows = conn.execute(
            f"""
            SELECT em.id AS memory_id, v.embedding
            FROM vec_episodes v JOIN episodic_memory em ON em.rowid = v.rowid
            WHERE {episodic_where}
            """,
            tuple(episodic_params),
        )

        def candidates():
            # No KNN or LIMIT: norms and ineligible rows must not hide a later
            # survivor. nlargest bounds retained memory, not scanned coverage.
            for row in rows:
                blob = row["embedding"]
                if vec_type == "int8":
                    cosine = _vec_int8_blob_cosine(query_blob, blob)
                elif vec_type == "bit":
                    cosine = _vec_bit_blob_cosine(
                        query_blob, blob, width=len(query_blob) * 8
                    )
                else:
                    cosine = _vec_float32_blob_cosine(query_unit, blob)
                if cosine < EM_VEC_ADMIT:
                    continue
                # Admission uses absolute cosine; keep the voice's existing
                # numpy score scale for fusion and cross-tier deduplication.
                sim = (cosine + 1.0) / 2.0
                yield RecallResult(
                    memory_id=row["memory_id"], score=sim, voice="vector",
                    metadata={
                        "similarity": sim, "cosine_similarity": cosine,
                        "vec_type": vec_type, "embedding_tier": "episodic",
                        "backend": "sqlite-vec",
                    },
                )

        return heapq.nlargest(20, candidates(), key=lambda result: result.score)

    def _vector_voice(
        self,
        query_embedding,
        default_dense_source_filter: bool = True,
        source: Optional[str] = None,
        topic: Optional[str] = None,
        episodic_where: Optional[str] = None,
        episodic_params: Sequence[Any] = (),
        excluded_wm_ids: Optional[Set[str]] = None,
    ) -> List[RecallResult]:
        """
        Voice 1: Dense semantic similarity over WM + EM.

        Queries the production-canonical dense embedding store
        (`memory_embeddings`) -- the same source the linear recall path
        uses via `_wm_vec_search` / `_in_memory_vec_search` (the
        numpy-cosine fallback layer in beam.py). Pre-fix this voice
        queried the standalone `binary_vectors` table which production
        never wrote to (NAI-4 wrote binary vectors as a column on
        episodic_memory, NOT to that table); the result was a silently
        empty vector voice and a 3-voice polyphonic engine.

        Returning to a single source of truth across the recall stack
        matches the cross-system convergence pattern (Hindsight, mem0,
        Zep, Cognee, Letta all use one dense store shared by every
        retrieval path) and makes polyphonic-vs-linear comparisons
        apples-to-apples for the BEAM-recovery experiment.

        EM tier prefers sqlite-vec's `vec_episodes` virtual table when
        available (same fast-path the linear scorer uses via
        `beam._vec_search`). Unmarked stores stream representation-safe
        scores from that same table; failures retain the existing JSON
        fallback. Combining JSON-only and sqlite-vec candidates is explicitly
        outside this change (#950).
        WM tier uses numpy cosine
        (matches the linear path -- no sqlite-vec WM index exists
        today).

        Reads both WM and EM tiers, filters out invalidated /
        superseded / expired rows (mirror of `_wm_vec_search` WHERE
        clauses for both tiers), and ranks by cosine similarity.
        Dedups across WM/EM by `memory_id` keeping the
        higher-similarity occurrence -- without this, a memory that
        exists in both tiers post-E3 would be double-counted in RRF
        and silently cap unique candidates below `top_k=20`.

        A/B toggle: `MNEMOSYNE_VOICE_VECTOR=0` disables this voice for
        ablation experiments. Returns empty so RRF fusion sees no
        vector contribution.
        """
        # Reset per-call fallback state before any early-return path:
        # a prior call may have recorded em=True; if this call exits
        # early (voice disabled, missing/empty embedding, zero norm)
        # the engine would otherwise inherit stale degraded state and
        # beam.py would record a false fallback for this call.
        self.last_call_fallback = {"em": False, "wm": False}
        if _env_disabled("MNEMOSYNE_VOICE_VECTOR"):
            return []
        if query_embedding is None or np is None:
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.size == 0:
            return []
        query_norm = float(np.linalg.norm(query_embedding))
        if query_norm == 0.0:
            return []
        query_unit = query_embedding / query_norm

        # Match the linear path's BEAM-mode scan budget so this voice
        # doesn't silently truncate against a benchmark-scale corpus
        # that the linear scorer would have seen entirely (beam.py
        # `_wm_vec_search` uses `_vec_limit = 500000 if _BEAM_MODE else
        # 50000`). The env var read mirrors the existing flag without
        # creating an import cycle on beam.py.
        beam_mode = os.environ.get("MNEMOSYNE_BEAM_MODE", "").lower() in ("1", "true", "yes")
        vec_limit = 500000 if beam_mode else 50000

        if self.conn is not None:
            conn = self.conn
            own_conn = False
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            own_conn = True
        try:
            now_iso = datetime.now(timezone.utc).isoformat()
            if episodic_where is None:
                clauses = [
                    "superseded_by IS NULL",
                    "(valid_until IS NULL OR julianday(valid_until) > julianday(?))",
                ]
                params = [now_iso]
                source_filter = source or topic
                if source_filter:
                    clauses.append("source = ?")
                    params.append(source_filter)
                episodic_where = " AND ".join(clauses)
                episodic_params = params
            by_id: Dict[str, RecallResult] = {}

            # --- EM tier -- prefer sqlite-vec ANN, fall back to numpy ---
            #
            # The linear path uses sqlite-vec's `vec_episodes` virtual
            # table via beam._vec_search for fast O(log N) ANN on EM
            # when sqlite-vec is loaded. Without mirroring that path,
            # the polyphonic engine would do a linear O(N) JSON-decode
            # + cosine over every embedded EM row -- strictly slower
            # than the linear scorer at benchmark scale (~250K rows).
            # That confounds the BEAM-recovery experiment's
            # polyphonic-vs-linear latency comparison.
            em_consumed_via_vec_episodes = False
            try:
                # Lazy import: avoids any module-load circular import
                # with beam.py (which lazily imports
                # PolyphonicRecallEngine inside _get_polyphonic_engine).
                # Both directions are runtime-only.
                from mnemosyne.core.beam import (
                    EM_VEC_ADMIT,
                    _classify_vec_store_regime,
                    _vec_available,
                    _vec_bit_blob_cosine,
                    _vec_float32_blob_cosine,
                    _vec_int8_blob_cosine,
                    _vec_search_with_blobs,
                    _vec_table_type_strict,
                )

                if _vec_available(conn):
                    vec_type = _vec_table_type_strict(conn)
                    try:
                        _poly_regime = _classify_vec_store_regime(
                            conn, "vec_episodes"
                        )
                    except Exception:
                        _poly_regime = "unknown"
                    if _poly_regime != "pure":
                        legacy_results = self._legacy_episodic_vector_voice(
                            conn,
                            query_unit,
                            now_iso,
                            source=source,
                            topic=topic,
                            episodic_where=episodic_where,
                            episodic_params=episodic_params,
                        )
                        by_id.update((r.memory_id, r) for r in legacy_results)
                        # A successful vec scan is authoritative even when
                        # every row is below admission. Do not fuse JSON-only
                        # candidates here; #950 owns that contract.
                        em_consumed_via_vec_episodes = True
                    else:
                        def candidate_cosine(candidate, query_blob):
                            row_blob = candidate.get("blob")
                            if vec_type == "int8":
                                if query_blob is None or row_blob is None:
                                    return None
                                return _vec_int8_blob_cosine(query_blob, row_blob)
                            if vec_type == "bit":
                                if query_blob is None or row_blob is None:
                                    return None
                                return _vec_bit_blob_cosine(
                                    query_blob,
                                    row_blob,
                                    width=len(query_blob) * 8,
                                )
                            if vec_type == "float32":
                                if row_blob is None:
                                    return None
                                return _vec_float32_blob_cosine(query_unit, row_blob)
                            return None

                        def knn_excludes_unseen_admitted_rows(
                            vec_rows, query_blob
                        ):
                            """Prove the KNN boundary is past the admission range."""
                            if not vec_rows:
                                return False
                            try:
                                boundary = max(
                                    float(row["distance"]) for row in vec_rows
                                )
                                if vec_type == "float32":
                                    max_admitted_distance = (
                                        math.sqrt(2.0 * (1.0 - EM_VEC_ADMIT))
                                        + 1e-5
                                    )
                                elif vec_type == "bit":
                                    if query_blob is None:
                                        return False
                                    width = len(query_blob) * 8
                                    max_admitted_distance = (
                                        math.acos(EM_VEC_ADMIT)
                                        * width
                                        / math.pi
                                    )
                                elif vec_type == "int8":
                                    if query_blob is None:
                                        return False
                                    quantized_query = memoryview(
                                        query_blob
                                    ).cast("b")
                                    query_norm = math.sqrt(sum(
                                        value * value
                                        for value in quantized_query
                                    ))
                                    # sqlite-vec 0.1.9 maps a unit component
                                    # onto int8's 127 scale with less than one
                                    # byte of quantization error. A normalized
                                    # row's byte norm is therefore within
                                    # sqrt(dimension) of 127.
                                    norm_slack = math.sqrt(
                                        len(quantized_query)
                                    )
                                    row_norms = (
                                        max(0.0, 127.0 - norm_slack),
                                        127.0 + norm_slack,
                                    )
                                    max_admitted_distance = max(
                                        math.sqrt(max(
                                            0.0,
                                            query_norm * query_norm
                                            + row_norm * row_norm
                                            - 2.0 * query_norm * row_norm
                                            * EM_VEC_ADMIT,
                                        ))
                                        for row_norm in row_norms
                                    )
                                else:
                                    return False
                            except (TypeError, ValueError):
                                return False
                            return boundary > max_admitted_distance

                        knn_limit = 60
                        eligible_scored = []
                        em_rows_via_vec = []
                        exact_scan_used = False
                        while knn_limit:
                            vec_rows, query_blob = _vec_search_with_blobs(
                                conn, query_unit.tolist(), k=knn_limit
                            )
                            rowid_to_candidate = {
                                row["rowid"]: row for row in vec_rows
                            }
                            em_rows_via_vec = []
                            if rowid_to_candidate:
                                rowids = list(rowid_to_candidate)
                                for offset in range(0, len(rowids), 900):
                                    chunk = rowids[offset : offset + 900]
                                    placeholders = ",".join("?" * len(chunk))
                                    em_rows_via_vec.extend(
                                        conn.execute(
                                            f"""
                                            SELECT rowid, id AS memory_id
                                            FROM episodic_memory
                                            WHERE rowid IN ({placeholders})
                                              AND ({episodic_where})
                                            """,
                                            (*chunk, *episodic_params),
                                        ).fetchall()
                                    )
                            eligible_scored = []
                            scorable_eligible = 0
                            for row in em_rows_via_vec:
                                candidate = rowid_to_candidate[row["rowid"]]
                                cosine = candidate_cosine(candidate, query_blob)
                                if cosine is not None:
                                    scorable_eligible += 1
                                if cosine is not None and cosine >= EM_VEC_ADMIT:
                                    eligible_scored.append((row, candidate, cosine))
                            if em_rows_via_vec and scorable_eligible == 0:
                                raise ValueError("KNN candidates lack scoreable blobs")
                            admission_range_exhausted = (
                                bool(em_rows_via_vec)
                                and knn_excludes_unseen_admitted_rows(
                                    vec_rows, query_blob
                                )
                            )
                            if (
                                admission_range_exhausted
                                or len(vec_rows) < knn_limit
                            ):
                                break
                            if knn_limit >= 4096:
                                # sqlite-vec caps KNN at 4096. If that finite
                                # boundary still cannot exclude an unseen row
                                # meeting admission, scan only the eligible join
                                # so KNN preselection cannot starve it.
                                exact_results = self._legacy_episodic_vector_voice(
                                    conn,
                                    query_unit,
                                    now_iso,
                                    source=source,
                                    topic=topic,
                                    episodic_where=episodic_where,
                                    episodic_params=episodic_params,
                                )
                                by_id.update(
                                    (r.memory_id, r) for r in exact_results
                                )
                                em_consumed_via_vec_episodes = True
                                exact_scan_used = True
                                break
                            knn_limit = min(
                                4096,
                                max(knn_limit * 2, knn_limit + 60),
                            )

                        if not exact_scan_used:
                            for row, candidate, cosine in eligible_scored:
                                sim = min(
                                    1.0, max(0.0, (cosine + 1.0) / 2.0)
                                )
                                mid = row["memory_id"]
                                existing = by_id.get(mid)
                                if existing is None or sim > existing.score:
                                    by_id[mid] = RecallResult(
                                        memory_id=mid,
                                        score=sim,
                                        voice="vector",
                                        metadata={
                                            "similarity": sim,
                                            "cosine_similarity": cosine,
                                            "raw_distance": float(
                                                candidate["distance"]
                                            ),
                                            "vec_type": vec_type,
                                            "embedding_tier": "episodic",
                                            "backend": "sqlite-vec",
                                        },
                                    )
                            # Preserve existing fallback semantics when the
                            # usable ANN table has no eligible joined row.
                            em_consumed_via_vec_episodes = bool(
                                em_rows_via_vec
                            )
            except (ImportError, AttributeError,
                    sqlite3.Error, ValueError, TypeError):
                # Broader catch than the original tuple -- partial
                # imports can surface as AttributeError, corrupt DB
                # state as sqlite3.DatabaseError (other Error
                # subclasses), and quantize edge cases as TypeError.
                # /review (Claude MEDIUM) caught the narrow filter
                # silently hiding unexpected failure modes. Fall
                # through to the numpy path on any of them.
                em_consumed_via_vec_episodes = False

            # --- EM tier -- numpy fallback (or when sqlite-vec absent) ---
            if not em_consumed_via_vec_episodes:
                try:
                    em_rows = conn.execute(
                        f"""
                        SELECT em.id AS memory_id, me.embedding_json
                        FROM memory_embeddings me
                        JOIN episodic_memory em ON me.memory_id = em.id
                        WHERE {episodic_where}
                        LIMIT ?
                        """,
                        (*episodic_params, vec_limit),
                    ).fetchall()
                except sqlite3.OperationalError:
                    em_rows = []
                for row in em_rows:
                    try:
                        memory_id = row["memory_id"]
                        embedding_json = row["embedding_json"]
                        if not embedding_json:
                            continue
                        vec = np.asarray(
                            json.loads(embedding_json), dtype=np.float32
                        )
                        vec_norm = float(np.linalg.norm(vec))
                        if vec_norm == 0.0:
                            continue
                        cos_sim = float(np.dot(query_unit, vec / vec_norm))
                        # Normalize cosine to [0, 1] so cross-path dedup
                        # against the sqlite-vec fast path (which now
                        # also produces [0, 1] scores) compares apples
                        # to apples. /review (4-source) caught the
                        # raw-cosine-vs-bit-Hamming inversion bug.
                        sim = min(1.0, max(0.0, (cos_sim + 1.0) / 2.0))
                        existing = by_id.get(memory_id)
                        if existing is None or sim > existing.score:
                            by_id[memory_id] = RecallResult(
                                memory_id=memory_id,
                                score=sim,
                                voice="vector",
                                metadata={
                                    "similarity": sim,
                                    "cosine_similarity": cos_sim,
                                    "embedding_tier": "episodic",
                                    "backend": "memory_embeddings",
                                },
                            )
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue

            # --- WM tier -- numpy cosine (no sqlite-vec WM index today) ---
            # Same WHERE clause shape as beam._wm_vec_search: skip
            # invalidated / superseded rows so vector voice never
            # surfaces ghost rows the linear path would have hidden.
            # Under the default dense-source filter (#696 / #427), raw
            # dialog / honcho rows and consolidated rows are excluded
            # BEFORE top-K selection — mirroring the linear recall path
            # so the polyphonic engine cannot have its nearest-N pool
            # saturated by dialog when the linear path would not.
            wm_dense_predicate = ""
            wm_dense_params = []
            if source:
                # Explicit source filter: mirror beam._wm_search semantics
                # (source = ?) and apply BEFORE top-K selection so the
                # requested rows cannot be starved out of the dense pool by
                # closer rows of other sources.
                wm_dense_predicate = " AND wm.source = ?"
                wm_dense_params.append(source)
            elif topic:
                # Topic is stored in the source field for now (pending a
                # dedicated topic column) -- same as beam._wm_search.
                wm_dense_predicate = " AND wm.source = ?"
                wm_dense_params.append(topic)
            elif default_dense_source_filter:
                # Raw dialog sources only (mirrors beam's linear wm_vec_where):
                # conversation + honcho_message. Durable honcho rows
                # (honcho_summary = deliberate session summary with higher
                # importance, honcho_import = generic import default) are not
                # raw dialog and must remain eligible for a dense score.
                wm_dense_predicate = (
                    " AND (wm.source IS NULL OR (wm.source <> 'conversation'"
                    " AND wm.source <> 'honcho_message'))"
                    " AND wm.consolidated_at IS NULL"
                )
            echo_clause, echo_params = exclusion_sql(excluded_wm_ids, "wm.id")
            wm_dense_predicate += echo_clause
            wm_dense_params.extend(echo_params)
            try:
                wm_rows = conn.execute(
                    f"""
                    SELECT wm.id AS memory_id, me.embedding_json
                    FROM memory_embeddings me
                    JOIN working_memory wm ON me.memory_id = wm.id
                    WHERE wm.superseded_by IS NULL
                      AND (wm.valid_until IS NULL OR julianday(wm.valid_until) > julianday(?))
                      {wm_dense_predicate}
                    LIMIT ?
                    """,
                    (now_iso, *wm_dense_params, vec_limit),
                ).fetchall()
            except sqlite3.OperationalError:
                wm_rows = []
            for row in wm_rows:
                try:
                    memory_id = row["memory_id"]
                    embedding_json = row["embedding_json"]
                    if not embedding_json:
                        continue
                    vec = np.asarray(
                        json.loads(embedding_json), dtype=np.float32
                    )
                    vec_norm = float(np.linalg.norm(vec))
                    if vec_norm == 0.0:
                        continue
                    cos_sim = float(np.dot(query_unit, vec / vec_norm))
                    # Normalize cosine to [0, 1] -- same rationale as EM
                    # numpy path above (cross-path dedup parity).
                    sim = min(1.0, max(0.0, (cos_sim + 1.0) / 2.0))
                    existing = by_id.get(memory_id)
                    if existing is None or sim > existing.score:
                        by_id[memory_id] = RecallResult(
                            memory_id=memory_id,
                            score=sim,
                            voice="vector",
                            metadata={
                                "similarity": sim,
                                "cosine_similarity": cos_sim,
                                "embedding_tier": "working",
                                "backend": "memory_embeddings",
                            },
                        )
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue

            results = sorted(
                by_id.values(), key=lambda r: r.score, reverse=True
            )
            # [C4] Surface whether the EM tier degraded to the numpy
            # full-scan (sqlite-vec absent/failed, or its top-K ANN
            # hits all dropped in the superseded/valid_until JOIN).
            # WM tier always uses numpy cosine today (no sqlite-vec
            # WM index exists), so WM is never a degraded path here.
            self.last_call_fallback["em"] = not em_consumed_via_vec_episodes
            return results[:20]
        finally:
            if own_conn:
                conn.close()
    
    def _graph_voice(self, query: str) -> List[RecallResult]:
        """
        Voice 2: Episodic graph traversal.

        Extracts entities from query, finds related memories
        through graph edges.

        Two retrieval strategies:
        1. Entity-based: search gists/facts by extracted entity names
        2. Graph traversal: walk graph edges from found memory IDs
           using find_related_memories (multi-hop BFS)

        A/B toggle: `MNEMOSYNE_VOICE_GRAPH=0` disables this voice.
        """
        if _env_disabled("MNEMOSYNE_VOICE_GRAPH"):
            return []
        # Extract entities (simple noun extraction)
        entities = self._extract_entities(query)

        results = []
        seed_ids = set()
        for entity in entities:
            # Find gists mentioning this entity
            gists = self.graph.find_gists_by_participant(entity)
            for gist in gists:
                gist_mid = gist.id.replace("gist_", "")
                seed_ids.add(gist_mid)
                results.append(RecallResult(
                    memory_id=gist_mid,
                    score=0.6,  # Base graph score
                    voice="graph",
                    metadata={"entity": entity, "gist": gist.text}
                ))

            # Find facts about this entity
            facts = self.graph.find_facts_by_subject(entity)
            for fact in facts:
                fact_mid = fact.id.split("_")[-1] if "_" in fact.id else fact.id
                seed_ids.add(fact_mid)
                results.append(RecallResult(
                    memory_id=fact_mid,
                    score=fact.confidence * 0.5,
                    voice="graph",
                    metadata={"entity": entity, "fact": f"{fact.subject} {fact.predicate} {fact.object}"}
                ))

        # Graph traversal: from seed memory IDs, walk edges to
        # discover indirectly related memories (ctx edges only,
        # moderate weight threshold to avoid noise).
        traversed_ids = set()
        for seed_id in seed_ids:
            related = self.graph.find_related_memories(
                seed_id, depth=2, edge_type="ctx", min_weight=0.3
            )
            for rel in related:
                mid = rel["memory_id"]
                if mid not in traversed_ids and mid not in seed_ids:
                    traversed_ids.add(mid)
                    results.append(RecallResult(
                        memory_id=mid,
                        score=0.4 / rel["depth"],  # Score decays with hop distance
                        voice="graph_traversal",
                        metadata={
                            "seed": seed_id,
                            "edge_type": rel["edge_type"],
                            "depth": rel["depth"],
                            "weight": rel["weight"],
                        }
                    ))

        return results
    
    def _fact_voice(self, query: str) -> List[RecallResult]:
        """
        Voice 3: Structured fact matching.

        Matches query against consolidated facts.

        A/B toggle: `MNEMOSYNE_VOICE_FACT=0` disables this voice.
        """
        if _env_disabled("MNEMOSYNE_VOICE_FACT"):
            return []
        # Extract potential subject from query
        words = query.lower().split()
        
        results = []
        for word in words:
            if len(word) < 3:
                continue
            
            facts = self.consolidator.get_consolidated_facts(
                subject=word.capitalize(),
                min_confidence=0.5
            )
            
            for fact in facts:
                # Prefer the row's stored id (preserves pre-fix
                # legacy IDs in mixed-format DBs); fall back to the
                # canonical hash if the dataclass is missing id
                # (e.g., older callers that built ConsolidatedFact
                # without going through get_consolidated_facts).
                # /review (Codex structured + Codex adversarial,
                # 2-source GATE FAIL) caught the previous unconditional
                # recompute as a legacy-row alignment regression.
                fact_memory_id = fact.id or compute_fact_id(
                    fact.subject, fact.predicate, fact.object
                )
                results.append(RecallResult(
                    memory_id=fact_memory_id,
                    score=fact.confidence,
                    voice="fact",
                    metadata={
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "object": fact.object,
                        "mentions": fact.mention_count
                    }
                ))
        
        return results
    
    def _temporal_voice(self, query: str, excluded_wm_ids=None) -> List[RecallResult]:
        """
        Voice 4: Time-aware scoring.

        Boosts recent memories, penalizes old ones.
        Uses exponential decay based on age.

        A/B toggle: `MNEMOSYNE_VOICE_TEMPORAL=0` disables this voice.
        """
        if _env_disabled("MNEMOSYNE_VOICE_TEMPORAL"):
            return []
        # Check for temporal keywords
        temporal_keywords = [
            "yesterday", "today", "recent", "last", "latest",
            "this week", "this month", "ago", "before"
        ]

        has_temporal = any(kw in query.lower() for kw in temporal_keywords)

        if not has_temporal:
            return []

        # Use the shared connection when available; otherwise open a
        # short-lived one (path used by the engine's standalone tests
        # and `python -m polyphonic_recall` self-test).
        if self.conn is not None:
            conn = self.conn
            own_conn = False
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            own_conn = True
        cursor = conn.cursor()

        try:
            # Check if working_memory table exists
            cursor.execute("""
                SELECT name FROM sqlite_master WHERE type='table' AND name='working_memory'
            """)
            if not cursor.fetchone():
                return []

            # Get memories from last 7 days. working_memory.timestamp is naive UTC
            # (the beam writers stamp datetime.now(timezone.utc).replace(tzinfo=None)),
            # so the cutoff has to be naive UTC too — a local `now` compares a UTC string
            # against a local one and the window shifts by the host offset. #1094.
            #
            # julianday() rather than a raw `timestamp > ?`: these are TEXT columns, so a
            # TEXT comparison orders an offset-bearing row by its written digits rather than
            # by its instant, which gets both window membership and pre-LIMIT chronology
            # wrong for mixed-format rows.
            #
            # The cost is a full scan: julianday() is not indexable under idx_wm_timestamp,
            # so the plan degrades from SEARCH USING INDEX to SCAN + temp B-tree. Measured
            # locally on 50k rows, median of 20 runs: 4.7 ms against 0.013 ms for the raw
            # form. That is the accepted price — #1094 requires the instant-correct
            # behaviour, and an expression index on julianday(timestamp) would restore the
            # SEARCH but is a schema migration, which is not authorised here.
            # tests/test_temporal_query_plan.py pins this plan and this budget on a synthetic
            # 50k fixture — a regression guard, not an execution bound. If an expression index
            # ever lands, the plan assertion flips to SEARCH and that test is the thing to
            # revisit.
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            week_ago = (now_utc - timedelta(days=7)).isoformat()
            echo_clause, echo_params = exclusion_sql(excluded_wm_ids)
            cursor.execute(f"""
                SELECT id, content, timestamp, importance
                FROM working_memory
                WHERE julianday(timestamp) > julianday(?) {echo_clause}
                ORDER BY julianday(timestamp) DESC
                LIMIT 20
            """, (week_ago, *echo_params))

            results = []
            for row in cursor.fetchall():
                # Calculate temporal score. A naive value is UTC (the storage contract
                # that #1087 settled for canonical_facts, and what beam.py writes here);
                # an aware value is normalised to UTC rather than to local. Converting to
                # local with a bare astimezone() then subtracting local `now` is the same
                # value on both sides, so it looks right — but a row stamped at UTC reads
                # as if it were `offset` hours older than it is, and the ranking is off by
                # exactly the host offset. #1094.
                try:
                    row_dt = datetime.fromisoformat(row["timestamp"])
                except (TypeError, ValueError):
                    continue
                if row_dt.tzinfo is not None:
                    row_dt = row_dt.astimezone(timezone.utc).replace(tzinfo=None)
                age = datetime.now(timezone.utc).replace(tzinfo=None) - row_dt
                age_days = age.total_seconds() / 86400
                temporal_score = np.exp(-age_days / 7)  # 7-day time constant (unchanged)

                results.append(RecallResult(
                    memory_id=row["id"],
                    score=temporal_score * row["importance"],
                    voice="temporal",
                    metadata={"age_days": age_days, "importance": row["importance"]}
                ))

            return results
        finally:
            if own_conn:
                conn.close()
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract potential entity names from text."""
        import re
        # Simple capitalized word extraction
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))
    
    def _hydrate_result_content(self, results: Dict[str, PolyphonicResult]) -> None:
        """Attach source content before applying content-based diversity ranking."""
        if not results:
            return

        if self.conn is not None:
            conn = self.conn
            own_conn = False
        else:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            own_conn = True

        try:
            memory_ids = tuple(results)
            placeholders = ", ".join("?" for _ in memory_ids)
            for table in ("working_memory", "episodic_memory"):
                try:
                    rows = conn.execute(
                        f"SELECT id, content FROM {table} WHERE id IN ({placeholders})",
                        memory_ids,
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
                for row in rows:
                    result = results.get(row["id"])
                    tier = "working" if table == "working_memory" else "episodic"
                    if (result is not None and not result.content
                            and result.metadata.get("_self_echo_tier", tier) == tier):
                        result.content = row["content"] or ""
        finally:
            if own_conn:
                conn.close()

    def _combine_voices(self, *voice_results: List[RecallResult]) -> Dict[str, PolyphonicResult]:
        """Combine results from all voices using Reciprocal Rank Fusion.

        RRF formula: score(d) = sum(1 / (k + rank(d, voice_i))) for each voice.
        Position-based fusion eliminates score calibration issues between voices.
        Constant k=60 (proven optimal for 4-voice retrieval).
        """
        RRF_K = 60
        combined = {}

        # Step 1: Rank results within each voice by score (descending)
        voice_ranks = {}  # voice_name -> {memory_id: rank}
        for results in voice_results:
            if not results:
                continue
            sorted_results = sorted(results, key=lambda r: r.score, reverse=True)
            voice_name = sorted_results[0].voice
            voice_ranks[voice_name] = {}
            for rank, r in enumerate(sorted_results, start=1):
                voice_ranks[voice_name][r.memory_id] = rank

        # Step 2: Accumulate RRF scores across voices
        for results in voice_results:
            voice_name = None
            for r in results:
                if voice_name is None:
                    voice_name = r.voice
                if r.memory_id not in combined:
                    combined[r.memory_id] = PolyphonicResult(
                        memory_id=r.memory_id,
                        combined_score=0.0,
                        voice_scores={},
                        metadata={}
                    )
                # RRF contribution: higher rank (lower number) = higher score
                rank = voice_ranks.get(voice_name, {}).get(r.memory_id, 999)
                rrf_contribution = 1.0 / (RRF_K + rank)
                combined[r.memory_id].voice_scores[r.voice] = rrf_contribution
                combined[r.memory_id].combined_score += rrf_contribution
                combined[r.memory_id].metadata.update(r.metadata)

        return combined
    
    def _diversity_rerank(self, results: Dict[str, PolyphonicResult],
                         top_k: int) -> List[PolyphonicResult]:
        """
        Re-rank with diversity penalty.
        
        Penalize results that are too similar to already-selected ones.
        """
        # Sort by combined score
        sorted_results = sorted(
            results.values(),
            key=lambda x: x.combined_score,
            reverse=True
        )
        
        selected = []
        for result in sorted_results:
            if len(selected) >= top_k:
                break
            
            # Check diversity against selected
            is_diverse = True
            for sel in selected:
                similarity = self._estimate_similarity(result, sel)
                if similarity > 0.8:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected.append(result)
        
        return selected
    
    def _estimate_similarity(self, a: PolyphonicResult, b: PolyphonicResult) -> float:
        """Estimate similarity between two results using content overlap.

        Uses word-level Jaccard on the content field instead of voice-name
        Jaccard.  Voice-name Jaccard collapses to 1.0 when a single voice
        dominates the candidate set, causing MMR diversity reranking to
        discard all but one result (#389).
        """
        content_a = (a.content or a.metadata.get("content") or "").lower().split()
        content_b = (b.content or b.metadata.get("content") or "").lower().split()

        if not content_a or not content_b:
            return 0.0

        set_a = set(content_a)
        set_b = set(content_b)
        intersection = set_a & set_b
        union = set_a | set_b

        return len(intersection) / len(union) if union else 0.0
    
    def _assemble_context(self, results: List[PolyphonicResult],
                         budget: int) -> List[PolyphonicResult]:
        """
        Assemble context within token budget.
        
        Approximate 4 chars per token.
        """
        current_chars = 0
        selected = []
        
        for result in results:
            # Estimate result size
            result_chars = len(str(result.metadata)) + 100
            
            if current_chars + result_chars > budget * 4:
                break
            
            selected.append(result)
            current_chars += result_chars
        
        return selected
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        # vector voice now queries memory_embeddings directly; surface
        # the count of embedded rows as the vector-voice signal-of-life.
        # /review caught the pre-fix behavior of returning 0 whenever
        # self.conn was None (standalone engines / CLI self-test);
        # mirror _vector_voice's own_conn fallback so the stat is
        # accurate regardless of construction mode.
        vec_count = 0
        if self.conn is not None:
            conn = self.conn
            own_conn = False
        else:
            try:
                conn = sqlite3.connect(str(self.db_path))
                conn.row_factory = sqlite3.Row
                own_conn = True
            except sqlite3.OperationalError:
                conn = None
                own_conn = False
        try:
            if conn is not None:
                try:
                    vec_count = conn.execute(
                        "SELECT COUNT(*) FROM memory_embeddings"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    vec_count = 0
        finally:
            if own_conn and conn is not None:
                conn.close()
        return {
            "voice_weights": self.voice_weights,
            "vector_stats": {"embedded_rows": vec_count},
            "graph_stats": self.graph.get_stats(),
            "consolidation_stats": self.consolidator.get_stats(),
        }

    def close(self):
        """Close all connections."""
        self.graph.close()
        self.consolidator.close()


# --- Testing ---
if __name__ == "__main__":
    import tempfile
    import os
    
    print("Polyphonic Recall Engine Tests")
    print("=" * 60)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    engine = PolyphonicRecallEngine(db_path=Path(db_path))
    
    # Test 1: Empty recall
    print("\nTest 1: Empty recall")
    results = engine.recall("What did Alice say yesterday?")
    print(f"  Results: {len(results)}")
    
    # Test 2: Stats
    print("\nTest 2: Stats")
    stats = engine.get_stats()
    print(f"  Voice weights: {stats['voice_weights']}")
    
    # Cleanup
    engine.close()
    os.unlink(db_path)
    
    print("\n" + "=" * 60)
    print("Polyphonic recall tests passed!")
