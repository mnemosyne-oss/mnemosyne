"""
Self-Harmonizing Memory Reasoning (SHMR)
========================================
Built on ECHO-OR research (AxDSan/ECHO-OR) but fully rearchitected for
continuous local memory orchestration inside Mnemosyne's BEAM architecture.

Core idea: related memories "echo" each other in the background, negotiating
contradictions, surfacing hidden patterns, and converging into stable beliefs.

This is Mnemosyne's signature reasoning layer -- no Honcho dreams, no Hindsight
reflections, no Mem0 static graphs. Memories actively resonate and self-correct.
"""

import os
import time
import logging
import json
from typing import List, Dict, Optional

import numpy as np

from mnemosyne.core import embeddings as _embeddings

logger = logging.getLogger("mnemosyne.shmr")

# --- Config ---
SHMR_BATCH_SIZE = int(os.environ.get("MNEMOSYNE_SHMR_BATCH_SIZE", "50"))
SHMR_MAX_ITERATIONS = int(os.environ.get("MNEMOSYNE_SHMR_MAX_ITERATIONS", "3"))
SHMR_SIMILARITY_THRESHOLD = float(os.environ.get("MNEMOSYNE_SHMR_SIMILARITY_THRESHOLD", "0.70"))
SHMR_HARMONY_THRESHOLD = float(os.environ.get("MNEMOSYNE_SHMR_HARMONY_THRESHOLD", "0.60"))
SHMR_MODEL = os.environ.get("MNEMOSYNE_SHMR_MODEL", "")
SHMR_MIN_CLUSTER_SIZE = int(os.environ.get("MNEMOSYNE_SHMR_MIN_CLUSTER_SIZE", "2"))
SHMR_TEMPERATURE = float(os.environ.get("MNEMOSYNE_SHMR_TEMPERATURE", "0.2"))

EMBEDDING_DIM = _embeddings.EMBEDDING_DIM  # derived from configured model

# --- SQL Schema ---
FACTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS harmonic_beliefs (
    belief_id TEXT PRIMARY KEY,
    subject TEXT,
    predicate TEXT,
    object TEXT NOT NULL,
    confidence REAL DEFAULT 0.5,
    provenance TEXT,   -- JSON array of source fact_ids or memory_ids
    cluster_id TEXT,
    iteration INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS memory_resonance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    cluster_count INTEGER,
    beliefs_generated INTEGER,
    contradictions_resolved INTEGER,
    harmony_score_avg REAL,
    duration_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_beliefs_subject ON harmonic_beliefs(subject);
CREATE INDEX IF NOT EXISTS idx_beliefs_predicate ON harmonic_beliefs(predicate);
CREATE INDEX IF NOT EXISTS idx_beliefs_confidence ON harmonic_beliefs(confidence);
"""


def _init_schema(conn):
    """Ensure SHMR tables exist."""
    conn.executescript(FACTS_SCHEMA_SQL)
    conn.commit()


def _embed(text: str) -> np.ndarray:
    """Embed text using Mnemosyne's embedding pipeline (BAAI/bge-small)."""
    emb = _embeddings.embed(text)
    if emb.ndim > 1:
        emb = emb.flatten()
    return emb.astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def _cluster_by_similarity(
    items: List[Dict],
    threshold: float,
) -> List[List[Dict]]:
    """Greedy connected-components clustering by cosine similarity.

    Each item must have an 'embedding' key with a numpy array.
    Returns list of clusters (each cluster is a list of items).
    """
    if not items:
        return []

    n = len(items)
    # Build adjacency: items are connected if sim >= threshold
    adj = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            sim = _cosine_similarity(items[i]["embedding"], items[j]["embedding"])
            if sim >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components (BFS)
    visited = set()
    clusters = []
    for i in range(n):
        if i in visited:
            continue
        cluster = []
        stack = [i]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster.append(items[node])
            stack.extend(adj[node] - visited)
        clusters.append(cluster)

    return clusters


def _format_cluster_for_llm(cluster: List[Dict]) -> str:
    """Format a memory cluster as a prompt for the LLM harmonizer."""
    lines = ["=== MEMORY CLUSTER ==="]
    for i, item in enumerate(cluster):
        subject = item.get("subject", "unknown")
        predicate = item.get("predicate", "stated")
        obj = item.get("object", item.get("content", ""))
        confidence = item.get("confidence", 0.5)
        timestamp = item.get("timestamp", "unknown")
        source = item.get("source", "fact")
        lines.append(
            f"[{i}] ({source}, conf={confidence:.2f}) {subject} | {predicate} | {obj}"
        )
    return "\n".join(lines)


HARMONY_PROMPT = """You are the Self-Harmonizing Memory Reasoner for Mnemosyne.
These memories belong to the same semantic cluster -- they all relate to the
same entities, topics, or events. Your job is to harmonize them:

1. **Resolve contradictions**: If two memories conflict, determine which is more
   likely true based on recency, specificity, and internal consistency. Flag the
   weaker one as dampened, not deleted.
2. **Extract higher-order beliefs**: Find patterns that span multiple memories.
   What does this cluster as a whole tell us? What's the stable truth?
3. **Dampen noise, amplify signal**: Low-confidence or stale memories get lower
   weight. Corroborated facts get reinforced.
4. **Output only stable beliefs**: Return NEW or UPDATED facts with confidence
   scores. Don't regurgitate every input fact -- synthesize.

Output as JSON array of belief objects:
[{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.0-1.0,
  "action": "create"|"update"|"dampen", "target_fact_id": null|"fact_id",
  "rationale": "one sentence explaining why"}]

RULES:
- Confidence 0.9+ = highly corroborated (multiple sources agree)
- Confidence 0.5-0.8 = reasonable inference from the cluster
- Confidence <0.4 = speculative, mark as such
- Use "dampen" to reduce confidence of contradicted facts (never delete)
- Use "update" to modify an existing fact with new information
- Output 1-5 beliefs per cluster (don't over-generate)"""


def _call_llm(prompt: str, system: str = "") -> str:
    """Call the configured LLM for harmonization.

    Uses the same LLM chain as mnemosyne_sleep's summarization:
    local_llm first, fallback to cloud extraction client.
    """
    # Try local LLM first
    try:
        from mnemosyne.core.local_llm import _call_local_llm
        result = _call_local_llm(prompt, system=system, temperature=SHMR_TEMPERATURE)
        if result and len(result.strip()) > 10:
            return result
    except Exception:
        pass

    # Fallback to cloud extraction client
    try:
        from mnemosyne.core.extraction import ExtractionConfig, ExtractionClient
        config = ExtractionConfig()
        client = ExtractionClient(config)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        result = client.chat(messages, temperature=SHMR_TEMPERATURE)
        if result:
            return result
    except Exception:
        pass

    return ""


def _compute_harmony_score(
    beliefs: List[Dict],
    cluster: List[Dict],
) -> float:
    """Score how well the harmonized beliefs represent the cluster.

    Uses cosine similarity between belief embeddings and cluster centroid,
    plus a consistency bonus for beliefs that don't contradict each other.
    """
    if not beliefs or not cluster:
        return 0.0

    # Compute cluster centroid
    cluster_embs = np.array([item.get("embedding", np.zeros(EMBEDDING_DIM))
                              for item in cluster])
    centroid = cluster_embs.mean(axis=0)

    # Score each belief against centroid
    belief_scores = []
    for belief in beliefs:
        belief_text = f"{belief.get('predicate', '')} {belief.get('object', '')}"
        try:
            belief_emb = _embed(belief_text)
            sim = _cosine_similarity(belief_emb, centroid)
            belief_scores.append(sim * belief.get("confidence", 0.5))
        except Exception:
            belief_scores.append(0.3)

    # Consistency bonus: penalize if beliefs contradict each other
    consistency_bonus = 1.0
    if len(beliefs) > 1:
        belief_embs = []
        for b in beliefs:
            try:
                belief_embs.append(_embed(f"{b.get('predicate','')} {b.get('object','')}"))
            except Exception:
                belief_embs.append(np.zeros(EMBEDDING_DIM))
        belief_embs = np.array(belief_embs)

        # Check pairwise similarity of beliefs (if they're too different,
        # that suggests the LLM produced contradictory beliefs)
        pairwise_sims = []
        for i in range(len(belief_embs)):
            for j in range(i + 1, len(belief_embs)):
                pairwise_sims.append(_cosine_similarity(belief_embs[i], belief_embs[j]))
        if pairwise_sims:
            avg_pairwise = np.mean(pairwise_sims)
            # Lower pairwise similarity = potential contradiction = penalty
            consistency_bonus = min(1.0, avg_pairwise + 0.3)

    avg_belief_score = np.mean(belief_scores) if belief_scores else 0.0
    return float(avg_belief_score * consistency_bonus)


def _extract_json_from_llm_output(text: str) -> List[Dict]:
    """Robust JSON extraction from LLM output (handles markdown wrappers)."""
    import re

    # Try direct parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and "beliefs" in parsed:
            return parsed["beliefs"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from ```json ... ``` block
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try extracting bare array
    array_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback: parse line by line for { ... } objects
    objects = re.findall(r'\{[^{}]*\}', text)
    results = []
    for obj_str in objects:
        try:
            results.append(json.loads(obj_str))
        except (json.JSONDecodeError, TypeError):
            continue
    return results


def _apply_beliefs(conn, beliefs: List[Dict], cluster: List[Dict], cluster_id: str):
    """Write harmonized beliefs to the database and update source facts."""
    import hashlib
    cursor = conn.cursor()
    now = __import__("datetime").datetime.now().isoformat()

    for belief in beliefs:
        action = belief.get("action", "create")
        subject = belief.get("subject", "entity")
        predicate = belief.get("predicate", "related_to")
        obj = belief.get("object", "")
        confidence = max(0.1, min(1.0, belief.get("confidence", 0.5)))

        belief_id = hashlib.sha256(
            f"{cluster_id}:{subject}:{predicate}:{obj[:50]}".encode()
        ).hexdigest()[:24]

        if action == "dampen":
            target_id = belief.get("target_fact_id")
            if target_id:
                cursor.execute(
                    "UPDATE facts SET confidence = MAX(0.1, confidence - 0.15) WHERE fact_id = ?",
                    (target_id,)
                )

        elif action == "update":
            target_id = belief.get("target_fact_id")
            if target_id:
                cursor.execute(
                    "UPDATE facts SET object = ?, confidence = ? WHERE fact_id = ?",
                    (obj, confidence, target_id)
                )

        # Always store the belief (create/update both produce a belief)
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO harmonic_beliefs
                (belief_id, subject, predicate, object, confidence,
                 provenance, cluster_id, iteration, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                belief_id, subject, predicate, obj, confidence,
                json.dumps([c.get("fact_id", "") for c in cluster if c.get("fact_id")]),
                cluster_id, 0, now
            ))
        except Exception:
            continue

    conn.commit()


def harmonize(beam, batch_size: int = None, max_iterations: int = None,
              similarity_threshold: float = None) -> Dict:
    """Run one harmonic cycle over recent memories.

    Called automatically by mnemosyne_sleep() after consolidation.
    Can also be called directly via MCP tool for on-demand harmonization.

    Args:
        beam: BeamMemory instance
        batch_size: Max memories to process (default: MNEMOSYNE_SHMR_BATCH_SIZE)
        max_iterations: Refinement iterations per cluster (default: 3)
        similarity_threshold: Cosine threshold for clustering (default: 0.70)

    Returns:
        Dict with stats: clusters_found, beliefs_generated, contradictions_resolved,
        harmony_score_avg, duration_ms
    """
    if batch_size is None:
        batch_size = SHMR_BATCH_SIZE
    if max_iterations is None:
        max_iterations = SHMR_MAX_ITERATIONS
    if similarity_threshold is None:
        similarity_threshold = SHMR_SIMILARITY_THRESHOLD

    t0 = time.perf_counter()
    _init_schema(beam.conn)
    cursor = beam.conn.cursor()

    # --- Step 1: Pull echo candidates ---
    # Prioritize: recent facts + high-confidence episodic memories
    candidates = []

    # Facts (status = active or NULL)
    fact_rows = cursor.execute("""
        SELECT fact_id, subject, predicate, object, confidence, timestamp
        FROM facts
        WHERE status = 'active' OR status IS NULL
        ORDER BY created_at DESC
        LIMIT ?
    """, (batch_size,)).fetchall()

    for row in fact_rows:
        candidates.append({
            "fact_id": row["fact_id"],
            "subject": row["subject"],
            "predicate": row["predicate"],
            "object": row["object"],
            "confidence": row["confidence"],
            "timestamp": row["timestamp"],
            "source": "fact",
            "embedding": _embed(row["object"]),
        })

    # Also pull recent episodic memories
    ep_rows = cursor.execute("""
        SELECT id, content, importance, created_at
        FROM episodic_memory
        ORDER BY created_at DESC
        LIMIT ?
    """, (batch_size // 2,)).fetchall()

    for row in ep_rows:
        content = row["content"]
        if content and len(content) > 10:
            candidates.append({
                "fact_id": f"ep_{row['id']}",
                "subject": "memory",
                "predicate": "contains",
                "object": content[:300],
                "confidence": row["importance"],
                "timestamp": row["created_at"],
                "source": "episodic",
                "embedding": _embed(content[:300]),
            })

    if len(candidates) < SHMR_MIN_CLUSTER_SIZE:
        return {
            "clusters_found": 0,
            "beliefs_generated": 0,
            "contradictions_resolved": 0,
            "harmony_score_avg": 0.0,
            "duration_ms": int((time.perf_counter() - t0) * 1000),
            "status": "insufficient_candidates",
        }

    # --- Step 2: Cluster by semantic similarity ---
    clusters = _cluster_by_similarity(candidates, similarity_threshold)
    # Filter small clusters
    clusters = [c for c in clusters if len(c) >= SHMR_MIN_CLUSTER_SIZE]

    total_beliefs = 0
    total_contradictions = 0
    harmony_scores = []

    # --- Step 3: Harmonize each cluster ---
    for cluster_idx, cluster in enumerate(clusters):
        cluster_id = f"shmr_{int(time.time())}_{cluster_idx}"

        for iteration in range(max_iterations):
            context = _format_cluster_for_llm(cluster)
            full_prompt = context + "\n\n" + HARMONY_PROMPT

            try:
                llm_output = _call_llm(full_prompt)
                if not llm_output:
                    continue

                beliefs = _extract_json_from_llm_output(llm_output)
                if not beliefs:
                    continue

                # Score the result
                score = _compute_harmony_score(beliefs, cluster)
                harmony_scores.append(score)

                if score >= SHMR_HARMONY_THRESHOLD:
                    _apply_beliefs(beam.conn, beliefs, cluster, cluster_id)
                    total_beliefs += len([b for b in beliefs
                                          if b.get("action") in ("create", "update")])
                    total_contradictions += len([b for b in beliefs
                                                 if b.get("action") == "dampen"])
                    break  # Converged for this cluster
                else:
                    # Repopulate cluster with beliefs for next iteration
                    for b in beliefs:
                        cluster.append({
                            "subject": b.get("subject", ""),
                            "predicate": b.get("predicate", ""),
                            "object": b.get("object", ""),
                            "confidence": b.get("confidence", 0.5),
                            "source": "belief_candidate",
                            "embedding": _embed(b.get("object", "")),
                        })

            except Exception as e:
                logger.warning(f"SHMR cluster {cluster_id} iteration {iteration}: {e}")
                continue

    duration_ms = int((time.perf_counter() - t0) * 1000)
    avg_score = float(np.mean(harmony_scores)) if harmony_scores else 0.0

    # --- Step 4: Log the resonance ---
    try:
        cursor.execute("""
            INSERT INTO memory_resonance_log
            (session_id, cluster_count, beliefs_generated,
             contradictions_resolved, harmony_score_avg, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            beam.session_id,
            len(clusters),
            total_beliefs,
            total_contradictions,
            round(avg_score, 4),
            duration_ms,
        ))
        beam.conn.commit()
    except Exception:
        pass

    return {
        "clusters_found": len(clusters),
        "beliefs_generated": total_beliefs,
        "contradictions_resolved": total_contradictions,
        "harmony_score_avg": round(avg_score, 4),
        "duration_ms": duration_ms,
        "status": "harmonized" if total_beliefs > 0 else "no_convergence",
    }


def recall_beliefs(beam, query: str, top_k: int = 10) -> List[Dict]:
    """Search harmonic beliefs for a given query.

    Used by recall() when harmonic=True flag is set.
    """
    cursor = beam.conn.cursor()
    _init_schema(beam.conn)

    try:
        query_emb = _embed(query)
        query_blob = query_emb.tobytes()

        # Search by embedding on object text
        results = []
        rows = cursor.execute("""
            SELECT belief_id, subject, predicate, object, confidence,
                   provenance, created_at
            FROM harmonic_beliefs
            ORDER BY confidence DESC
            LIMIT ?
        """, (top_k * 2,)).fetchall()

        # Score by embedding similarity
        scored = []
        for row in rows:
            try:
                belief_emb = _embed(row["object"])
                sim = _cosine_similarity(query_emb, belief_emb)
                scored.append((sim * row["confidence"], row))
            except Exception:
                scored.append((row["confidence"] * 0.3, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        for score, row in scored[:top_k]:
            results.append({
                "content": row["object"],
                "score": round(score, 4),
                "belief_id": row["belief_id"],
                "subject": row["subject"],
                "predicate": row["predicate"],
                "provenance": row["provenance"],
                "source": "harmonic_belief",
            })

        return results
    except Exception:
        return []


# ============================================================
#  Phase 3A: Reflective Recall (single-pass fact synthesis)
# ============================================================

REFLECTION_PROMPT = """You are a memory reasoning assistant. You have retrieved facts
from a conversation database and need to synthesize a coherent answer.

QUESTION: {question}

RETRIEVED FACTS:
{fact_context}

Based on these facts, provide a concise synthesis (2-4 sentences) that:
1. Answers the question directly if the facts are sufficient
2. Identifies any contradictions or gaps in the facts
3. Notes temporal context (dates, order of events) if present
4. If facts are insufficient, states what's missing clearly

SYNTHESIS:"""


def reflect(beam, question: str, facts: List[Dict] = None,
            top_k: int = 10) -> Optional[str]:
    """Single-pass reflective synthesis over retrieved facts.

    Takes a question and a list of fact dicts (from fact_recall()), sends them
    to an LLM, and returns a coherent synthesis paragraph. This synthesis is
    then injected as additional context for the final answering LLM.

    This is Phase 3A: lightweight, works with any LLM, no iteration needed.
    Phase 3B (SHMR harmonize()) replaces this with multi-iteration harmony loop.

    Args:
        beam: BeamMemory instance (for fact_recall if facts not provided)
        question: The question to synthesize for
        facts: Pre-retrieved facts (if None, calls fact_recall automatically)
        top_k: Max facts to include in the reflection

    Returns:
        Synthesis string, or None if no facts available.
    """
    # Get facts if not provided
    if facts is None and beam is not None:
        try:
            facts = beam.fact_recall(question, top_k=top_k)
        except Exception:
            return None

    if not facts:
        return None

    # Build fact context (limit to top_k, sort by score)
    sorted_facts = sorted(facts, key=lambda f: f.get("score", 0), reverse=True)[:top_k]
    fact_lines = []
    for i, f in enumerate(sorted_facts):
        content = f.get("content", "")
        score = f.get("score", 0.5)
        source = f.get("source", "fact")
        fact_lines.append(f"[{i}] ({source}, conf={score:.2f}) {content}")

    fact_context = "\n".join(fact_lines)
    prompt = REFLECTION_PROMPT.format(question=question, fact_context=fact_context)

    synthesis = _call_llm(prompt)
    if synthesis and len(synthesis.strip()) > 10:
        return synthesis.strip()
    return None


def get_resonance_log(beam, limit: int = 10) -> List[Dict]:
    """Get recent harmonization run logs."""
    cursor = beam.conn.cursor()
    _init_schema(beam.conn)
    try:
        rows = cursor.execute("""
            SELECT * FROM memory_resonance_log
            ORDER BY created_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []
