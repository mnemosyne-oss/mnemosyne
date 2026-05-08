# Architecture

Mnemosyne is a local-first memory system built entirely on SQLite. No external databases, no network calls, no API keys. Everything runs in-process.

## BEAM — Bilevel Episodic-Associative Memory

The core storage model is **BEAM**, a three-tier architecture:

```
┌─────────────────────────────────────────────────┐
│                  BEAM Tiers                      │
│                                                  │
│  ┌─────────────────────────────────────────┐    │
│  │  Working Memory                         │    │
│  │  Hot context, auto-injected into prompts│    │
│  │  TTL-based eviction (default: 24h)      │    │
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
- Evicted by TTL (configurable, default 24 hours) or item count limit
- Supports session-scoped and global-scope memories
- Uses FTS5 for fast keyword search within the tier

### Episodic Memory

- Long-term storage for consolidated memories
- Populated by the `sleep()` consolidation process
- Hybrid search combining three signals:
  - **50% vector similarity** — semantic relevance via sqlite-vec
  - **30% FTS5 rank** — keyword/lexical relevance
  - **20% importance score** — user-assigned weight
- Vector compression: `int8` (default), `float32`, or `bit` (32x smaller)

### Scratchpad

- Ephemeral workspace for agent reasoning chains
- Not searchable, not consolidated — cleared explicitly or by item limit
- Useful for intermediate steps, TODO tracking, and multi-turn reasoning

## Sleep Cycle (Consolidation)

The `sleep()` function moves stale working memories into episodic memory:

1. Fetches working memories past TTL or below importance threshold
2. Groups them by source
3. Attempts LLM summarization (local TinyLlama, remote OpenAI-compatible, or AAAK fallback)
4. Stores the summary in episodic memory with embeddings
5. Removes the originals from working memory
6. Logs the consolidation event

```python
from mnemosyne import sleep
result = sleep()
print(f"Consolidated {result['consolidated']} memories")
```

## SQLite Backend

By default, the main database lives at `~/.hermes/mnemosyne/data/mnemosyne.db`. Named memory banks use separate SQLite files under `~/.hermes/mnemosyne/data/banks/<name>/`, and standalone `TripleStore()` may use `triples.db` in the data directory.

### Tables

| Table | Purpose |
|---|---|
| `working_memory` | Hot tier — recent context |
| `episodic_memory` | Long-term consolidated memories |
| `vec_episodes` | sqlite-vec virtual table for episodic embeddings |
| `scratchpad` | Temporary reasoning entries |
| `consolidation_log` | History of sleep cycle operations |
| `triples` | Temporal knowledge graph |
| `memories` | Legacy table (backward compatibility) |
| `memory_embeddings` | Legacy embeddings (backward compatibility) |

### Extensions

- **sqlite-vec** — native vector similarity search (HNSW-style) in SQLite
- **FTS5** — full-text search, built into SQLite 3.35+

Both extensions are optional. Without sqlite-vec, Mnemosyne falls back to keyword-only retrieval. FTS5 is available on any modern SQLite build.

## Hybrid Search Pipeline

```
Query string
    │
    ├─── Vector search (sqlite-vec, top_k × 3)
    │         Semantic similarity via cosine distance
    │
    ├─── FTS5 search (top_k × 3)
    │         Keyword/lexical matching
    │
    └─── Merge + re-rank
              Score = 0.5 × vec_similarity
                    + 0.3 × fts_rank
                    + 0.2 × importance
              Return top_k results
```

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
