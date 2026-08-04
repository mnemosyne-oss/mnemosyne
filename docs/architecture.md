# Architecture

Mnemosyne is a local-first memory system built entirely on SQLite. There is no external database and no required network service: storage, indexing, and retrieval all run in-process against a single file.

Network access is opt-in and never on the retrieval path. Four subsystems can reach out when you configure them: sync (`mnemosyne sync`), remote LLM summarization (`MNEMOSYNE_LLM_BASE_URL`), API-served embeddings (`MNEMOSYNE_EMBEDDING_API_URL`), and LLM conflict detection. With none of them configured, Mnemosyne makes no network calls and needs no API keys.

## BEAM: Bilevel Episodic-Associative Memory

The core storage model is **BEAM**. It has two searchable memory tiers, working and episodic, plus a non-searchable scratchpad:

```
┌─────────────────────────────────────────────────┐
│                  BEAM Tiers                      │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  Working Memory                         │    │
│  │  Hot context, auto-injected into prompts│    │
│  │  TTL-based eviction (default: 168h)     │    │
│  │  Max items: 10,000                      │    │
│  └───────────────────┬─────────────────────┘    │
│                      │ sleep() consolidation     │
│  ┌───────────────────▼─────────────────────┐    │
│  │  Episodic Memory                        │    │
│  │  Long-term storage                      │    │
│  │  Hybrid search: vector + FTS5           │    │
│  │  Summaries from consolidation           │    │
│  └─────────────────────────────────────────┘    │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  Scratchpad                             │    │
│  │  Temporary agent reasoning workspace    │    │
│  │  Max items: 1,000                       │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
```

### Working Memory

- Stores recent, high-priority context
- Auto-injected into LLM prompts via the `pre_llm_call` hook
- Evicted by TTL (configurable, default 168 hours / 7 days) or item count limit
- Supports session-scoped and global-scope memories
- Uses FTS5 for fast keyword search within the tier

### Episodic Memory

- Long-term storage for consolidated memories
- Populated by the `sleep()` consolidation process
- Hybrid search combining three signals:
  - **50% vector similarity** for semantic relevance via sqlite-vec
  - **30% FTS5 rank** for keyword and lexical relevance
  - **20% importance score**, the user-assigned weight
- Vector compression: `int8` (default), `float32`, or `bit` (32x smaller)

### Scratchpad

- Ephemeral workspace for agent reasoning chains
- Not searchable, not consolidated, cleared explicitly or by item limit
- Useful for intermediate steps, TODO tracking, and multi-turn reasoning

Because the scratchpad is neither searched nor consolidated, it is not a memory tier in the retrieval sense. "Bilevel" in the BEAM name refers to working and episodic.

> **Two different things are called "tiers".** The tiers above are *storage* tiers. Separately, episodic memories carry a *freshness* tier (1, 2, or 3) that decays with age and multiplies their recall score. See [Tier degradation](#tier-degradation) below. The two concepts are unrelated and both predate this document.

## Sleep Cycle (Consolidation)

The `sleep()` function summarizes stale working memories into episodic memory:

1. Fetches working memories past TTL or below importance threshold
2. Groups them by source
3. Attempts LLM summarization (host backend, remote OpenAI-compatible, local GGUF, or AAAK keyword fallback)
4. Stores the summary in episodic memory with embeddings
5. Stamps `consolidated_at` on the source rows in `working_memory`
6. Logs the consolidation event

**Consolidation is additive, not destructive.** Step 5 marks the originals; it does not delete them. Rows with `consolidated_at` set are exempt from TTL and item-count trimming (`_trim_working_memory` in `mnemosyne/core/beam.py`) and stay recallable until an explicit `forget()`. Only *unconsolidated* rows are subject to the TTL window.

The one visible consequence: consolidated rows are excluded from hot prompt-injection context (`get_context()`) so they do not crowd out fresh memories. Set `MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED=1` to restore the older behaviour. This does not affect `recall()`, where consolidated rows are always eligible.

`sleep()` is **synchronous** and blocks until the cycle completes. Background consolidation is a host concern: the Hermes provider runs it on its own daemon thread behind its own lock.

### Tier degradation

Independently of the storage tiers, episodic memories age through three freshness tiers that scale their recall score:

| Tier | Age | Score multiplier |
|---|---|---|
| 1 | newer than `MNEMOSYNE_TIER2_DAYS` (30) | `MNEMOSYNE_TIER1_WEIGHT` (1.0) |
| 2 | older than 30 days | `MNEMOSYNE_TIER2_WEIGHT` (0.5) |
| 3 | older than `MNEMOSYNE_TIER3_DAYS` (180) | `MNEMOSYNE_TIER3_WEIGHT` (0.25) |

Entering tier 3 also compresses content to `MNEMOSYNE_TIER3_MAX_CHARS` (300), via LLM summarization when `MNEMOSYNE_SMART_COMPRESS` is on and truncation otherwise.

```python
from mnemosyne import sleep
result = sleep()
print(f"Consolidated {result['consolidated']} memories")
```

## SQLite Backend

By default, the main database lives at `~/.hermes/mnemosyne/data/mnemosyne.db`. Named memory banks use separate SQLite files under `~/.hermes/mnemosyne/data/banks/<name>/`, and standalone `TripleStore()` may use `triples.db` in the data directory.

### Tables

A current database holds **36 tables**. They are created by idempotent `init_*`
functions that run on every open, so an existing database acquires new tables and
columns without a migration step. Most DDL lives in `init_beam` in
`mnemosyne/core/beam.py`; newer subsystems own their own module and init function.

**Core storage**

| Table | Purpose |
|---|---|
| `working_memory` | Hot tier, recent context |
| `episodic_memory` | Long-term consolidated memories |
| `scratchpad` | Temporary reasoning entries |
| `memories` | Legacy table (backward compatibility) |
| `consolidation_log` | History of sleep cycle operations |
| `memory_validations` | Collaborative attest / invalidate records |

**Search indexes**

| Table | Purpose |
|---|---|
| `vec_episodes` | sqlite-vec virtual table for episodic embeddings |
| `vec_working` | sqlite-vec virtual table for working-memory embeddings |
| `vec_facts` | Declared for fact embeddings. No writer yet; recreated empty on reindex |
| `fts_episodes` | FTS5 external-content index over `episodic_memory` |
| `fts_working` | FTS5 index over `working_memory` |
| `fts_facts` | FTS5 index over `facts` |
| `memory_embeddings` | JSON embedding fallback when sqlite-vec is unavailable |

**Knowledge graph and facts**

| Table | Purpose |
|---|---|
| `triples` | Temporal subject-predicate-object knowledge graph |
| `annotations` | Per-memory annotations: mentions, facts, sources, dates |
| `facts` | Extracted fact rows backing `fts_facts` |
| `canonical_facts` | One current value per `(owner, category, name)` slot |
| `consolidated_facts` | Veracity-weighted merged facts |
| `conflicts` | Detected contradictions between facts |
| `graph_edges` | Weighted edges for multi-hop traversal |
| `gists` | Episode gists used by the graph voice |

**MEMORIA layer**

| Table | Purpose |
|---|---|
| `memoria_facts` | Extracted durable facts |
| `memoria_timelines` | Time-ordered event records |
| `memoria_instructions` | Standing instructions extracted from conversation |
| `memoria_preferences` | Extracted user preferences |
| `memoria_kg` | MEMORIA knowledge-graph rows |
| `memoria_persona` | L3 persona store; the prompt path reads the opt-in `persona.md` file, not this table |

**Self-harmonizing reasoning (SHMR)**

| Table | Purpose |
|---|---|
| `harmonic_beliefs` | Negotiated beliefs produced from clustered memories |
| `memory_resonance_log` | SHMR clustering and harmonization audit trail |

**Sync**

| Table | Purpose |
|---|---|
| `memory_events` | Append-only mutation log that sync replicates |
| `sync_meta` | Device identity and sync cursors |
| `sync_memory_state` | Last-known synced state, used to discover local mutations |
| `sync_outbox_ack` | Acknowledgement tracking for pushed events |

**Operations**

| Table | Purpose |
|---|---|
| `query_cache` | Cached recall results, semantic and exact-key tiers |
| `hygiene_audit_log` | Record of noise audits and cleanup actions |
| `cost_entries` | Token and cost accounting for LLM calls |

There are no schema-level foreign keys. `PRAGMA foreign_keys=ON` was evaluated and
rejected (issue #503): it broke tests that intentionally create orphan rows, and an
FK on `memory_embeddings` had previously caused every embedding insert to fail
silently. Referential integrity is checked at the application layer and reported by
`mnemosyne doctor`.

### Extensions

- **sqlite-vec** provides native vector similarity search (HNSW-style) in SQLite
- **FTS5** provides full-text search, built into SQLite 3.35+

FTS5 is available on any modern SQLite build. When sqlite-vec is unavailable but embeddings are still available, Mnemosyne falls back to JSON vectors in `memory_embeddings` plus NumPy cosine scoring. If no embedding provider is available, recall falls back to lexical/keyword retrieval.

## Retrieval

`recall()` selects one of three engines at call time. The linear hybrid scorer is the
default; the other two are opt-in.

### 1. Linear hybrid (default)

```
Query string
    │
    ├─── Vector search (sqlite-vec, or memory_embeddings + NumPy fallback)
    │         Semantic similarity via cosine distance
    │
    ├─── FTS5 search (top_k × 3)
    │         Keyword/lexical matching
    │
    └─── Merge + re-rank
              base  = 0.5 × vec_similarity
                    + 0.3 × fts_rank
                    + 0.2 × importance
              score = base × recency_decay × tier_weight × veracity_multiplier
              score += graph_bonus (≤0.08) + fact_bonus (≤0.10) + binary_bonus
              score *= (1 + temporal_weight × temporal_boost)
              Return top_k results
```

The three base weights are set by `MNEMOSYNE_VEC_WEIGHT`, `MNEMOSYNE_FTS_WEIGHT`, and
`MNEMOSYNE_IMPORTANCE_WEIGHT`, normalized to sum to 1.0, and can be overridden per
query. They are read from the environment directly rather than through `config.yaml`.

### 2. Polyphonic (`MNEMOSYNE_POLYPHONIC_RECALL=1`)

Four independent retrieval voices run and their ranked lists are fused with
Reciprocal Rank Fusion at k=60, followed by a diversity re-rank and context assembly
under a token budget:

| Voice | Signal | Default fusion weight | Kill switch |
|---|---|---|---|
| vector | dense embeddings from `memory_embeddings` | 0.35 | `MNEMOSYNE_VOICE_VECTOR=0` |
| graph | multi-hop traversal over `graph_edges` | 0.25 | `MNEMOSYNE_VOICE_GRAPH=0` |
| fact | structured matches against `facts` | 0.25 | `MNEMOSYNE_VOICE_FACT=0` |
| temporal | `event_date` proximity | 0.15 | `MNEMOSYNE_VOICE_TEMPORAL=0` |

Each result carries per-voice scores, so you can see which signal surfaced it.

### 3. Enhanced (`MNEMOSYNE_ENHANCED_RECALL=1`)

Adds query expansion, synonym handling, intent classification, MMR diversity, Weibull
survival re-scoring, and associative expansion on top of the linear path. Results are
cached in `query_cache`, keyed on a digest that includes the resolved weights and every
pipeline toggle. The cache is invalidated on `invalidate()` and on dedup-update.

### Degradation ladder

sqlite-vec → `memory_embeddings` with NumPy cosine → lexical FTS5 only. Each step down
is automatic. With `MNEMOSYNE_NO_EMBEDDINGS=1`, or when no embedding provider is
reachable, recall still works lexically.

## Temporal Knowledge Graph

The `TripleStore` provides time-aware subject-predicate-object triples:

```python
from mnemosyne.core.triples import TripleStore

kg = TripleStore()
kg.add("Maya", "assigned_to", "auth-migration", valid_from="2026-01-15")

# Query current state
kg.query("Maya")  # → Maya is assigned to auth-migration

# Query as-of a past date
kg.query("Maya", as_of="2026-01-10")  # → empty (not yet assigned)

# Adding a new assignment auto-invalidates the old one
kg.add("Maya", "assigned_to", "api-gateway", valid_from="2026-03-01")
```

When a triple is added for an existing `(subject, predicate)` pair, the previous triple's `valid_until` is automatically set, enabling point-in-time queries.

## Data Flow

```
remember(content, importance, scope)
    │
    ├── Write to working_memory (BEAM)
    ├── Write to memories (legacy, backward compat)
    └── Generate embedding (if fastembed available)

recall(query, top_k)
    │
    ├── Search working_memory (FTS5 fast path)
    ├── Search episodic_memory (hybrid: vec + FTS5 + importance)
    ├── Merge, de-duplicate, re-rank
    └── Return top_k results

sleep()
    │
    ├── Fetch stale working memories (past TTL)
    ├── Chunk by token budget
    ├── Summarize via LLM
    │     ├── Host backend (if MNEMOSYNE_HOST_LLM_ENABLED=true and registered)
    │     ├── Remote OpenAI-compatible API (if BASE_URL set)
    │     ├── Local GGUF (ctransformers / llama-cpp-python)
    │     └── AAAK encoding (keyword-based, no LLM)
    ├── Store summary in episodic_memory with embedding
    └── Remove originals from working_memory
```
