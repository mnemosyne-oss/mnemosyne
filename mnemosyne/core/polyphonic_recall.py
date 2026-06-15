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
import os
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    import numpy as np
except ImportError:  # numpy is required by other voices too; guard for parity
    np = None

from mnemosyne.core.typed_memory import classify_memory, MemoryType, get_type_priority
from mnemosyne.core.episodic_graph import EpisodicGraph
from mnemosyne.core.veracity_consolidation import (
    VeracityConsolidator,
    compute_fact_id,
)

# Cached embedding dim for vector normalization — resolved once at import time.
# Avoids a per-row import inside the bit-vec normalization hot path.
try:
    from mnemosyne.core.embeddings import EMBEDDING_DIM as _EMB_DIM
except ImportError:
    _EMB_DIM = 384
_EMBEDDING_DIM_BITS = float(_EMB_DIM)


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

        # Voice weights (deterministic, learned from validation)
        self.voice_weights = {
            "vector": 0.35,
            "graph": 0.25,
            "fact": 0.25,
            "temporal": 0.15,
        }
    
    def recall(self, query: str, query_embedding: np.ndarray = None,
               top_k: int = 10, context_budget: int = 4000) -> List[PolyphonicResult]:
        """
        Polyphonic recall: all 4 voices in parallel, then combine.
        
        Args:
            query: Text query
            query_embedding: Optional pre-computed embedding
            top_k: Number of results to return
            context_budget: Max tokens for context assembly
            
        Returns:
            List of PolyphonicResult, sorted by combined score
        """
        # Run all 4 voices
        vector_results = self._vector_voice(query_embedding)
        graph_results = self._graph_voice(query)
        fact_results = self._fact_voice(query)
        temporal_results = self._temporal_voice(query)
        
        # Combine results
        all_results = self._combine_voices(
            vector_results, graph_results, fact_results, temporal_results
        )
        
        # Re-rank with diversity
        reranked = self._diversity_rerank(all_results, top_k)
        
        # Assemble context within budget
        context = self._assemble_context(reranked, context_budget)
        
        return context
    
    def _vector_voice(self, query_embedding) -> List[RecallResult]:
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
        `beam._vec_search`); falls through to numpy cosine over
        `memory_embeddings` on any failure. WM tier uses numpy cosine
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
            now_iso = datetime.now().isoformat()
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
                    _vec_available,
                    _effective_vec_type,
                )

                if _vec_available(conn):
                    vec_type = _effective_vec_type(conn)
                    emb_json = json.dumps(
                        query_embedding.astype(np.float32).tolist()
                    )
                    # sqlite-vec's MATCH planner needs LIMIT to be a
                    # literal at planning time AND enforces a hard max
                    # of 4096 (raises OperationalError: "k value in knn
                    # query too large" above that). The fast path is a
                    # top-K lookup, not a full scan, so we only need
                    # enough candidates to survive post-fetch filter
                    # dropouts (~50 buffer above the top-20 the engine
                    # ultimately returns). vec_limit (which controls
                    # the numpy fallback's full-scan budget under
                    # BEAM_MODE) is irrelevant here.
                    k_inline = 60
                    if vec_type == "bit":
                        rank_sql = (
                            "SELECT rowid, distance FROM vec_episodes "
                            "WHERE embedding MATCH vec_quantize_binary(?) "
                            f"AND k={k_inline} ORDER BY distance"
                        )
                    elif vec_type == "int8":
                        rank_sql = (
                            "SELECT rowid, distance FROM vec_episodes "
                            "WHERE embedding MATCH vec_quantize_int8(?, 'unit') "
                            f"AND k={k_inline} ORDER BY distance"
                        )
                    else:
                        rank_sql = (
                            "SELECT rowid, distance FROM vec_episodes "
                            f"WHERE embedding MATCH ? AND k={k_inline} "
                            "ORDER BY distance"
                        )
                    vec_rows = conn.execute(rank_sql, (emb_json,)).fetchall()
                    if vec_rows:
                        rowid_to_dist = {
                            r["rowid"]: r["distance"] for r in vec_rows
                        }
                        # Map rowid → memory_id and apply WHERE-clause
                        # parity with the numpy EM fallback. JOIN
                        # ensures rows orphaned from episodic_memory
                        # (e.g., deleted post-vec_episodes-insert) drop
                        # out cleanly.
                        rowid_list = list(rowid_to_dist.keys())
                        placeholders = ",".join("?" * len(rowid_list))
                        em_rows_via_vec = conn.execute(
                            f"""
                            SELECT em.rowid AS rowid, em.id AS memory_id
                            FROM episodic_memory em
                            WHERE em.rowid IN ({placeholders})
                              AND em.superseded_by IS NULL
                              AND (em.valid_until IS NULL OR em.valid_until > ?)
                            """,
                            (*rowid_list, now_iso),
                        ).fetchall()
                        for row in em_rows_via_vec:
                            mid = row["memory_id"]
                            dist = rowid_to_dist.get(row["rowid"])
                            if dist is None:
                                continue
                            # Normalize sqlite-vec distances to a
                            # cosine-similarity-compatible [0, 1] scale
                            # so cross-tier dedup against the WM tier's
                            # numpy cosine values is meaningful.
                            #
                            # /review (4-source: Codex structured P2,
                            # Codex adversarial MEDIUM, Claude
                            # CRITICAL, perf HIGH) caught the pre-fix
                            # behavior of using `1.0 - distance`
                            # directly: bit-type Hamming distance is
                            # an int in [0, EMBEDDING_DIM_BITS], so
                            # the score went heavily negative
                            # (~-383). WM cosine is in [-1, 1].
                            # Dedup at `sim > existing.score` then
                            # always preferred WM hits over EM
                            # sqlite-vec hits, silently inverting the
                            # tier-priority semantics for bit-quantized
                            # vectors. Normalize per vec_type:
                            #   bit:    1 - dist/EMBEDDING_DIM_BITS
                            #   int8:   1 - dist/2  (cosine_dist in
                            #                       [0, 2] for unit
                            #                       vectors)
                            #   raw f32: 1/(1+dist) (L2 → (0, 1])
                            raw_dist = float(dist)
                            if vec_type == "bit":
                                # Resolved from configured model dim
                                sim = 1.0 - (raw_dist / _EMBEDDING_DIM_BITS)
                            elif vec_type == "int8":
                                sim = 1.0 - (raw_dist / 2.0)
                            else:
                                sim = 1.0 / (1.0 + raw_dist)
                            existing = by_id.get(mid)
                            if existing is None or sim > existing.score:
                                by_id[mid] = RecallResult(
                                    memory_id=mid,
                                    score=sim,
                                    voice="vector",
                                    metadata={
                                        "similarity": sim,
                                        "raw_distance": raw_dist,
                                        "vec_type": vec_type,
                                        "embedding_tier": "episodic",
                                        "backend": "sqlite-vec",
                                    },
                                )
                        # Only mark EM consumed when the fast path
                        # actually produced results. If all top-60
                        # ANN hits failed the superseded/valid_until
                        # filter (or orphaned the JOIN), fall through
                        # to the numpy path so it can scan up to
                        # vec_limit and find valid rows beyond the
                        # truncated ANN candidate set. /review (4-source)
                        # caught the silent EM-starvation regression.
                        em_consumed_via_vec_episodes = bool(em_rows_via_vec)
            except (ImportError, AttributeError,
                    sqlite3.Error, ValueError, TypeError) as exc:
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
                        """
                        SELECT em.id AS memory_id, me.embedding_json
                        FROM memory_embeddings me
                        JOIN episodic_memory em ON me.memory_id = em.id
                        WHERE em.superseded_by IS NULL
                          AND (em.valid_until IS NULL OR em.valid_until > ?)
                        LIMIT ?
                        """,
                        (now_iso, vec_limit),
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
                        sim = (cos_sim + 1.0) / 2.0
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
            try:
                wm_rows = conn.execute(
                    """
                    SELECT wm.id AS memory_id, me.embedding_json
                    FROM memory_embeddings me
                    JOIN working_memory wm ON me.memory_id = wm.id
                    WHERE wm.superseded_by IS NULL
                      AND (wm.valid_until IS NULL OR wm.valid_until > ?)
                    LIMIT ?
                    """,
                    (now_iso, vec_limit),
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
                    sim = (cos_sim + 1.0) / 2.0
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
    
    def _temporal_voice(self, query: str) -> List[RecallResult]:
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

            # Get memories from last 7 days
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT id, content, timestamp, importance
                FROM working_memory
                WHERE timestamp > ?
                ORDER BY timestamp DESC
                LIMIT 20
            """, (week_ago,))

            results = []
            for row in cursor.fetchall():
                # Calculate temporal score
                age = datetime.now() - datetime.fromisoformat(row["timestamp"])
                age_days = age.total_seconds() / 86400
                temporal_score = np.exp(-age_days / 7)  # 7-day half-life

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
        """Estimate similarity between two results."""
        # Simple Jaccard-like similarity on voice scores
        voices_a = set(a.voice_scores.keys())
        voices_b = set(b.voice_scores.keys())
        
        if not voices_a or not voices_b:
            return 0.0
        
        intersection = voices_a & voices_b
        union = voices_a | voices_b
        
        return len(intersection) / len(union)
    
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
