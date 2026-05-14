# Mnemosyne vs Hindsight Self-Hosted

**Last updated:** 2026-05-14 · Mnemosyne v2.8.0

This is an honest, technical comparison between Mnemosyne and Hindsight self-hosted (local Docker, not the managed Cloud product). Every claim below is grounded in source code — no fabricated benchmarks or aspirational APIs.

> **TL;DR:** They are not direct competitors. Hindsight is a **memory engine** with sophisticated NLP and multi-signal retrieval. Mnemosyne is a **memory layer** optimized for simplicity, speed, and single-machine deployments. Choose based on what you actually need.

---

## Architecture

| Dimension | Mnemosyne | Hindsight Self-Hosted |
|---|---|---|
| **Process model** | In-process Python library | Separate Docker containers (FastAPI + PostgreSQL) |
| **IPC overhead** | Zero (direct function calls) | HTTP + JSON serialization to localhost:8888 |
| **Database** | SQLite (single file, WAL mode) | PostgreSQL + pgvector extension |
| **Embedding model** | fastembed ONNX — BAAI/bge-small-en-v1.5 (~67MB) | sentence-transformers PyTorch (~500MB) |
| **Vector search** | sqlite-vec (int8/bit/float32) or numpy fallback | pgvector HNSW (mature, optimized) |
| **Cold start** | Instant (if models cached locally) | ~5–10s (Docker container boot + model loading) |
| **Vector storage** | int8 (default): 384 bytes per 384-dim vector | float32: ~1536 bytes per 384-dim vector |
| | bit (optional): 48 bytes per 384-dim vector |  |
| **Runtime memory** | ~10–20MB per session (SQLite + ONNX) | ~100–300MB (PostgreSQL pool + PyTorch) |

---

## Setup

| Step | Mnemosyne | Hindsight Self-Hosted |
|---|---|---|
| Install | `pip install mnemosyne-memory` | `docker compose up` (requires Docker) |
| Dependencies (core) | Python stdlib + SQLite — nothing else | Docker, PostgreSQL, pgvector extension |
| Dependencies (semantic) | `pip install mnemosyne-memory[embeddings]` → fastembed (ONNX) | sentence-transformers (PyTorch) or TEI container |
| Dependencies (LLM) | `pip install mnemosyne-memory[llm]` → ctransformers + GGUF, or remote OpenAI-compatible API | Local LLM or remote API |
| Configuration | Zero config. Sensible defaults. Env vars optional. | YAML config, PostgreSQL connection, port bindings |
| Number of containers | 0 | 2–3 (API server, PostgreSQL, optional embedding worker) |

**Verdict:** Mnemosyne has a materially simpler install path. `pip install` vs Docker Compose is a real difference, especially on resource-constrained environments or ephemeral VMs.

---

## Memory Model

### Mnemosyne: BEAM (Bilevel Episodic-Associative Memory)

Three SQLite tables:

| Tier | Purpose | Behavior |
|---|---|---|
| **Working memory** | Hot, recent context | Auto-injected into prompts via `pre_llm_call` hook. TTL-based eviction (default 24h). Max 10,000 items. FTS5 indexed. |
| **Episodic memory** | Long-term consolidated storage | Populated by `sleep()` consolidation. Hybrid vector + FTS5 search. |
| **Scratchpad** | Temporary agent workspace | Not searchable, not consolidated. Cleared explicitly. Max 1,000 items. |

Additional: **TripleStore** — temporal knowledge graph with `valid_from`/`valid_until` for point-in-time queries.

**Core operations:** `remember()`, `recall()`, `sleep()` — intentionally simple.

### Hindsight: Retain / Recall / Reflect

| Operation | Mnemosyne equivalent | Gap? |
|---|---|---|
| **Retain** — LLM-driven fact extraction, entity normalization, 2–5 structured facts per chunk | `remember()` stores raw text + optional embedding. `extract=True` enables LLM fact extraction. | **Partial gap.** Mnemosyne can extract facts via LLM but has no automatic entity normalization. |
| **Recall** — 4-way parallel (semantic + BM25 + graph + temporal), RRF fusion, cross-encoder rerank | `recall()` — hybrid (vector + FTS5 + importance) × recency decay. Single-pass. | **Design difference.** Mnemosyne is simpler by choice. |
| **Reflect** — Agentic loop with tool calling, mental models, disposition traits | `sleep()` — LLM summarization of working → episodic. No agentic loop, no mental models. | **Gap.** Mnemosyne does consolidation, not reasoning about knowledge. |

**Honest assessment:** Hindsight's model is more sophisticated. Mnemosyne's model is intentionally simpler — fewer moving parts, fewer failure modes, but also fewer capabilities.

---

## Retrieval

| Feature | Mnemosyne | Hindsight Self-Hosted |
|---|---|---|
| **Vector search** | sqlite-vec (cosine distance) | pgvector HNSW |
| **Keyword search** | SQLite FTS5 | PostgreSQL full-text + BM25 |
| **Graph search** | TripleStore (subject-predicate-object, temporal) | Native knowledge graph with co-occurrence tracking |
| **Temporal search** | `from_date`/`to_date` filters + configurable exponential decay boost (`temporal_weight`, `temporal_halflife`) | Native date parsing, `occurred_start/end`, temporal recall strategy |
| **Scoring formula** | `score = vec_weight × vec_sim + fts_weight × fts_rank + importance_weight × importance`, then × recency decay | 4-way parallel retrieval → RRF fusion → cross-encoder rerank |
| **Default weights** | 50% vector, 30% FTS, 20% importance | Learned fusion weights |
| **Configurable?** | Yes — `vec_weight`, `fts_weight`, `importance_weight` params per query, or via env vars | Yes — configurable strategies |
| **Reranking** | None (single-pass) | Cross-encoder rerank |

**The trade-off:** Mnemosyne's single-pass scoring is faster but less precise. Hindsight's 4-way + rerank pipeline finds more relevant results at the cost of latency and compute.

---

## Entity Extraction

| Feature | Mnemosyne | Hindsight Self-Hosted |
|---|---|---|
| **Method** | Regex patterns + pure Python Levenshtein distance | spaCy NLP pipeline + LLM extraction |
| **Patterns** | `@mentions`, `#hashtags`, `"quoted phrases"`, capitalized sequences (2–5 words) | Full NLP: named entities, noun phrases, coreference |
| **Fuzzy matching** | Levenshtein distance with prefix/substring bonuses. `"Abdias"` ≈ `"Abdias J"` (similarity: 0.925) | Trigram/full resolution strategies. Entity co-occurrence tracking. |
| **Storage** | TripleStore triples: `(memory_id, "mentions", "entity_name")` | Structured entity table with normalization |
| **Speed** | ~0.01ms per extraction | Heavier (spaCy model loading + inference) |
| **Accuracy** | Good for proper nouns, handles, hashtags. Misses pronouns, complex references. | Higher recall — resolves "she", "the project", complex entity mentions. |
| **Opt-in?** | `extract_entities=True` on `remember()` | Always on |

**Verdict:** Mnemosyne's regex approach is fast and dependency-free but misses many entity types that spaCy catches. This is a deliberate trade-off: speed and simplicity over NLP accuracy.

---

## Fact Extraction

| Feature | Mnemosyne | Hindsight Self-Hosted |
|---|---|---|
| **Method** | LLM-driven (`extraction.py`): sends text to LLM, parses 2–5 factual statements | LLM-driven Retain pipeline with provenance tracking |
| **Fallback chain** | Remote OpenAI-compatible API → local ctransformers GGUF → skip (graceful) | N/A (runs inside container) |
| **Storage** | TripleStore: `(memory_id, "fact", fact_text)` | Structured fact table with evidence tracking |
| **Opt-in?** | `extract=True` on `remember()` | Always on via Retain |
| **Automatic?** | No — caller must opt in per memory | Yes — automatic on all ingested text |

---

## Integrations

### MCP (Model Context Protocol)

Mnemosyne provides an MCP server with **6 tools** and **2 transports**:

| Tool | Description |
|---|---|
| `mnemosyne_remember` | Store a memory (supports entity extraction, fact extraction, bank selection) |
| `mnemosyne_recall` | Search memories with hybrid scoring and configurable weights |
| `mnemosyne_sleep` | Run consolidation cycle |
| `mnemosyne_scratchpad_read` | Read agent scratchpad |
| `mnemosyne_scratchpad_write` | Write to scratchpad |
| `mnemosyne_get_stats` | Get memory statistics |

```
mnemosyne mcp                          # stdio transport (Claude Desktop, etc.)
mnemosyne mcp --transport sse --port 8080  # SSE transport (web clients)
mnemosyne mcp --bank project_a            # scoped to a specific bank
```

### Hermes Agent Integration

`plugin.yaml` registers **15 tools** and **3 hooks**:

**Tools:** `mnemosyne_remember`, `mnemosyne_recall`, `mnemosyne_stats`, `mnemosyne_triple_add`, `mnemosyne_triple_query`, `mnemosyne_sleep`, `mnemosyne_scratchpad_write`, `mnemosyne_scratchpad_read`, `mnemosyne_scratchpad_clear`, `mnemosyne_invalidate`, `mnemosyne_export`, `mnemosyne_update`, `mnemosyne_forget`, `mnemosyne_import`, `mnemosyne_diagnose`

**Hooks:** `pre_llm_call` (context injection), `on_session_start` (session init), `post_tool_call` (memory capture)

### Hindsight Integration

Custom HTTP API on port 8888. Native `openclaw-hindsight` plugin exists for OpenClaw. Hermes integration via HTTP client.

| | Mnemosyne | Hindsight |
|---|---|---|
| **Hermes** | Native (in-process, no serialization) | HTTP client |
| **OpenClaw** | Planned (adapter not yet built) | Native plugin exists |
| **MCP** | 6 tools, stdio + SSE | Custom HTTP API |
| **Cross-machine** | Export/import JSON only | Any agent with HTTP access to port 8888 |

---

## Memory Banks

| Feature | Mnemosyne | Hindsight |
|---|---|---|
| **Named banks** | `BankManager` — create, list, delete, rename banks | `banks` table with strict isolation |
| **Isolation** | Per-bank SQLite file under `~/.hermes/mnemosyne/data/banks/<name>/` | PostgreSQL schema-level isolation |
| **Usage** | `Mnemosyne(bank="work")` or `mnemosyne mcp --bank work` | API-level bank selection |
| **Multi-tenancy** | No access control | HindClaw extension (JWT/API key multi-tenancy) |
| **Stats** | `BankManager.get_bank_stats(name)` — exists, size, path | Per-bank metrics |

---

## Additional Features

### Mnemosyne-specific

| Feature | Module | Description |
|---|---|---|
| **Streaming** | `core/streaming.py` | `MemoryStream` with push (callbacks) and pull (iterator) patterns. Thread-safe event buffer. |
| **Delta sync** | `core/streaming.py` | `DeltaSync` — incremental synchronization between Mnemosyne instances with checkpointed resume. |
| **Pattern detection** | `core/patterns.py` | `PatternDetector` — temporal (hour/weekday), content (keyword frequency, co-occurrence), sequence patterns. |
| **Memory compression** | `core/patterns.py` | `MemoryCompressor` — dictionary-based, RLE, and semantic compression strategies. |
| **Plugin system** | `core/plugins.py` | `MnemosynePlugin` base class with 4 lifecycle hooks. Built-in: `LoggingPlugin`, `MetricsPlugin`, `FilterPlugin`. Discovers plugins from `~/.hermes/mnemosyne/plugins/`. |
| **Diagnostics** | `diagnose.py` | PII-safe health check — dependencies, database state, vector readiness. No memory content or API keys. |
| **Cost logging** | `core/cost_log.py` | Tracks LLM API usage and costs. |

### Hindsight-specific (not in Mnemosyne)

| Feature | Description |
|---|---|
| **Automatic entity normalization** | "Abdias" and "Abdias J" resolved to same entity automatically |
| **Cross-encoder reranking** | Second-pass neural reranking of retrieval results |
| **Mental models** | Agent reasoning about user preferences and traits |
| **Agentic reflection** | Tool-calling loop during Reflect phase |
| **Conflict detection** | Automatic contradiction detection and merging |
| **Multi-machine sharing** | Network API for distributed agents |
| **Multi-tenancy** | Per-user isolation with access control via HindClaw |

---

## Performance Characteristics

| Metric | Mnemosyne | Hindsight Self-Hosted |
|---|---|---|
| **Recall latency (10K corpus)** | ~2–10ms — in-process SQLite + sqlite-vec, no HTTP overhead | ~50–200ms — HTTP round-trip + PostgreSQL + 4-way retrieval + rerank |
| **IPC model** | Direct Python function call | HTTP POST to localhost:8888 → JSON serialization → response parsing |
| **Storage footprint** | ~50–100MB SQLite file per 10K memories | ~200–500MB PostgreSQL + WAL per 10K memories |
| **Model download** | One-time ~67MB (fastembed ONNX) | One-time ~500MB (sentence-transformers PyTorch) |
| **Runtime memory** | ~10–20MB per session | ~100–300MB (PostgreSQL pool + PyTorch runtime) |
| **Consolidation** | `sleep()` — LLM summarization or AAAK fallback, runs on-demand | Background consolidation engine with evidence tracking |

> **Important caveat on latency numbers:** Mnemosyne's latency advantage comes from being an in-process library calling SQLite directly, compared to HTTP round-trips to a local Docker container. This is an architectural advantage, not a retrieval-quality advantage. If Hindsight were called as a library (not over HTTP), the gap would narrow significantly.

---

## Dependency Profile

| Mode | Mnemosyne Dependencies | Network Calls |
|---|---|---|
| **Minimal (keyword only)** | Python stdlib + SQLite | **None** |
| **Semantic search** | + fastembed ONNX (~67MB one-time download) | **One-time** model download, then none |
| **Vector search in SQLite** | + sqlite-vec (pip-installable C extension) | None |
| **LLM consolidation** | + ctransformers + GGUF (~600MB), or remote API | **One-time** model download, or remote LLM API |
| **MCP server** | + mcp, starlette, uvicorn (for SSE) | None at runtime |

| Mode | Hindsight Dependencies | Network Calls |
|---|---|---|
| **Full stack** | Docker, PostgreSQL, pgvector, sentence-transformers | **One-time** model download, then none |
| **TEI variant** | + HuggingFace TEI container | None at runtime |

Both systems are fully offline after initial setup. The difference is weight: Mnemosyne's core is Python stdlib + SQLite. Hindsight requires Docker + PostgreSQL even for basic operation.

---

## When to Choose What

### Choose Mnemosyne if:

- You want `pip install` with zero containers
- You need the fastest possible recall latency for interactive agent loops
- You're running on a resource-constrained environment (VPS, ephemeral VM, CI)
- You're building a single-user, single-machine agent (Hermes, Claude Desktop, etc.)
- You want an MCP-compatible memory layer (stdio + SSE)
- You want full control over the memory model and don't need automatic "magic"
- You value fewer moving parts over sophisticated NLP
- You want memory banks with per-bank SQLite isolation without standing up PostgreSQL

### Choose Hindsight Self-Hosted if:

- You need entity resolution ("Abdias" and "Abdias J" are the same person)
- You need automatic structured fact extraction from raw text
- You need cross-machine memory sharing via network API
- You need multi-tenant memory banks with access control
- You need temporal reasoning ("what did I say last Tuesday?") with automatic date extraction
- You need the highest recall quality (4-way retrieval + cross-encoder rerank)
- You need an OpenClaw integration today (not planned)
- You're okay with Docker + PostgreSQL complexity as a trade-off for richer capabilities

### Neither is "better." They serve different points on the simplicity-sophistication spectrum.

---

## Known Gaps in Mnemosyne (honest list)

| Gap | Severity | Workaround |
|---|---|---|
| No automatic entity normalization | Medium | `extract_entities=True` captures entities; fuzzy matching helps but doesn't resolve coreference |
| No cross-machine network API | Medium for multi-agent setups | Export/import JSON; same-machine sharing via shared SQLite file. **Can now import FROM Hindsight directly** — migrate without data loss |
| No cross-encoder reranking | Low for most queries | Hybrid scoring with configurable weights covers common cases |
| No automatic conflict detection | Medium | Manual `invalidate(memory_id, replacement_id=new_id)` |
| No multi-tenancy / access control | High for SaaS use cases | Use per-bank SQLite isolation for domain separation |
| No mental models / agentic reflection | Low | `sleep()` does consolidation; reasoning about knowledge is the caller's job |
| OpenClaw adapter not yet built | Medium for OpenClaw users | Hermes integration is native; OpenClaw requires MCP adapter work |
| Temporal queries require explicit dates | Low | `valid_until` and `superseded_by` are manual; `TripleStore` supports `as_of` queries |

---

*This page was rewritten for v2.1 after community feedback about inaccurate comparisons. Every feature listed for Mnemosyne has been verified against the source code. Hindsight features are based on public documentation and the author's direct codebase analysis. If anything here is wrong, please open an issue — we'll fix it.*
