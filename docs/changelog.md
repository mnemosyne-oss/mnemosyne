# Changelog

See [CHANGELOG.md](../CHANGELOG.md) in the repository root for the full version history.

## Recent Releases

### 3.6 - Mnemosyne Sync (June 2026)

- **Bidirectional delta sync:** New `mnemosyne sync` CLI for synchronizing memory between Mnemosyne instances (desktop to VPS, local to remote)
- **Append-only event log:** New `memory_events` table with all 11 fields (event_id, memory_id, operation, timestamp, device_id, payload, parent_event_ids, importance, expiry, event_hash, synced_at)
- **SyncEngine:** Event-log-based sync orchestrator with pull_changes/push_changes protocol, cursor-based pagination, and idempotent deduplication
- **HTTP sync server:** stdlib-only server (no FastAPI deps) with POST /sync/pull, POST /sync/push, GET /sync/status
- **CLI commands:** `mnemosyne sync`, `mnemosyne sync-serve`, `mnemosyne sync-status`, `mnemosyne sync-generate-key`
- **Authentication:** API key (Bearer) and JWT support on sync endpoints
- **Conflict resolution:** v1 strategy (timestamp->importance->device_id), 5-second detection window
- **Client-side encryption (optional):** XChaCha20-Poly1305 via Fernet/PyNaCl fallback, key derivation via Argon2id->PBKDF2
- **Full memory pipeline integration:** Incoming sync events route through `remember()` for FTS5, embeddings, entity extraction
- **Security docs:** [docs/security.md](docs/security.md) with threat model, liability disclaimers, comparison table
- **Sync docs:** [docs/sync.md](docs/sync.md) with protocol reference, deployment examples (Docker Compose, Caddy, Nginx, Fly.io, SSH tunnel)
- **Zero hard deps added:** `[sync]` extra pulls in cryptography; server uses stdlib only

### 2.4 — Hindsight Importer + Host LLM Adapter (May 2026)

- **Import FROM Hindsight:** New `HindsightImporter` for migrating Hindsight memories into Mnemosyne
- **Dual input:** JSON export files OR live Hindsight HTTP API (`/v1/default/banks/{bank}/memories/list`)
- **Episodic-direct:** Writes to `episodic_memory` (not working memory), preserving original timestamps, fact types, session IDs, metadata, scope, and veracity
- **Stable IDs:** SHA256-based deduplication with `hs_` prefix — re-importing is idempotent
- **Smart importance:** Derived from Hindsight `fact_type` (world=0.75, experience=0.65, observation=0.55) plus `proof_count` bonus
- **Full metadata preservation:** hindsight_id, fact_type, context, dates, entities, chunk_id, tags, consolidation timestamps
- **CLI:** `mnemosyne import-hindsight <file|url> [bank]`
- **Registry:** Registered as the 7th supported provider alongside Mem0, Letta, Zep, Cognee, Honcho, SuperMemory
- **Tests:** 102 lines of regression tests covering timestamp preservation, episodic-only import, duplicate skipping, FTS indexing, provider-registry usage
- **Why:** Before this, migrating from Hindsight went through `remember()` which assigned current timestamps and wrote to working memory. Historical context was lost. Now migrations preserve the full temporal record with zero data loss.

- **Host LLM Adapter:** Route consolidation and fact extraction through Hermes' authenticated auxiliary client
- **OAuth unblock:** Hermes users on ChatGPT/Codex subscriptions can now use LLM-backed memory ops without `MNEMOSYNE_LLM_BASE_URL`
- **LLMBackend Protocol:** Tiny one-method interface (`complete()`) in `mnemosyne/core/llm_backends.py` — any host can register a backend
- **HermesAuxLLMBackend:** Routes through `agent.auxiliary_client.call_llm(task="compression", ...)` with lazy imports
- **Host-skips-remote rule (A3):** When host attempt fails, remote URL is skipped — falls to local GGUF. Prevents stale URL leaks.
- **Daemon thread safety:** `on_session_end()` runs sleep with 15s join; `shutdown()` drains 2s before unregistering backend
- **Temperature contracts:** Extraction uses `0.0` (deterministic), consolidation uses `0.3` (paraphrasing acceptable)
- **Tests:** 7 new tests covering registry round-trip, precedence, A3 rule, gate semantics, shutdown drain, bullet-list parsing
- **Live verified:** End-to-end with `openai-codex` OAuth through ChatGPT backend

### 2.1 — BEAM Benchmark (May 2026)

- **Benchmark:** End-to-end BEAM evaluation against ICLR 2026 dataset (Tavakoli et al.)
- **End-to-end:** 35.4% at 100K (competitive with RAG/LIGHT), 19.3% at 500K, 19.2% at 1M
- **Pipeline:** LLM answering + LLM-as-judge rubric scoring, matching paper protocol
- **Known issues:** Episodic consolidation not producing entries. Performance degrades at scale.
- See: [docs/beam-benchmark.md](docs/beam-benchmark.md)

### 2.0

- **Add:** Entity extraction with Levenshtein fuzzy matching (Phase 1)
- **Add:** LLM-driven structured fact extraction with fallback chain (Phase 2)
- **Add:** Temporal recall with exponential decay scoring (Phase 3)
- **Add:** Configurable hybrid scoring weights (Phase 4)
- **Add:** Memory banks for per-domain SQLite isolation (Phase 5)
- **Add:** MCP server with 6 tools, stdio + SSE transports (Phase 6)
- **Add:** Streaming event bus, pattern detection, plugin system (Phase 8)
- **Add:** SQLite WAL mode + busy timeout for concurrency
- **Fix:** Test mocking for extraction fallback (env vars don't affect module-level constants)
- **Tests:** 292 tests passing. Zero failures.

### 1.13.0

- **Add:** Temporal query capabilities — `recall(from_date="...", to_date="...", source="...", topic="...")`
- **Add:** Auto temporal triples on `remember()` — `(memory_id, occurred_on, YYYY-MM-DD)` and `(memory_id, has_source, source)`
- **Fix:** `UnboundLocalError` in fallback scoring when FTS5 returns empty but temporal filters active
- **Tests:** 6 new temporal query tests. All 25 passing.

### 1.12.0

- **Fix:** Embeddings generated but discarded when sqlite-vec absent — now falls back to `memory_embeddings` table with numpy cosine similarity
- **Add:** `_in_memory_vec_search()` — cosine similarity via numpy when sqlite-vec unavailable
- **Add:** `mnemosyne_diagnose` — PII-safe diagnostic tool for troubleshooting
- **Docs:** Replaced junk files with proper developer documentation

### 1.11.0

- **Fix:** Context overflow on consolidation — `sleep()` now chunks memories to fit the LLM context window
- **Fix:** No remote/API model support — added OpenAI-compatible remote LLM client (`MNEMOSYNE_LLM_BASE_URL`)
- **Add:** `chunk_memories_by_budget()` for token-aware batch splitting
- **Add:** `_call_remote_llm()` with httpx primary and urllib fallback
- **Tests:** 7 new tests (2 for chunking, 5 for remote API). All 24 passing.

### 1.10.2

- **Add:** `mnemosyne_update` and `mnemosyne_forget` tools for full CRUD in Hermes plugin
- **Fix:** Auto-sleep dict key, module-level `remember()` signature, BEAM sync on update
- **Fix:** sqlite-vec KNN query LIMIT parameter for vec virtual table planner
- **Fix:** Triple tools in MemoryProvider (missing module-level functions)
- **Remove:** 307 lines of dead code (unused quantization functions, ghost imports)

### 1.10.0

- **Add:** `hermes mnemosyne stats --global` — cross-session working memory stats

### 1.9.0

- **Release:** PyPI package `mnemosyne-memory` now live
- **CI:** GitHub Actions for tests (Python 3.9–3.12) and automated releases
- **Packaging:** Modern `pyproject.toml` with PEP 517

### 1.7

- **Fix:** Subagent context writes polluting persistent memory
- **Fix:** Cross-session recall consistency with global scope preservation
- **Fix:** Fallback keyword scoring for Chinese and spaceless languages

### 1.5

- **Fix:** 6 critical bugs (stats, recall tracking, vector similarity, hardcoded session_id)

### 1.0

First major release. Production-ready.

- BEAM architecture (working + episodic + scratchpad)
- Native vector search via sqlite-vec
- FTS5 full-text hybrid search
- Temporal triples (knowledge graph)
- Hermes plugin integration
- Sub-millisecond latency on CPU

---

For the complete history, see [CHANGELOG.md](../CHANGELOG.md).
For releases, see [GitHub Releases](https://github.com/AxDSan/mnemosyne/releases).
