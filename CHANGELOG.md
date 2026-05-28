# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [SemVer](https://semver.org/) starting from v3.1.2.

## [3.1.2] — 2026-05-28

### Fixed

- **Irrelevant context injection in recall.** Three root-cause fixes for
  [#198](https://github.com/AxDSan/mnemosyne/issues/198):
  - Strict fact matching is now the default. Set `MNEMOSYNE_LENIENT_FACT_MATCH=1`
    to opt back into permissive matching (which matched any query word against any
    stored fact, dragging in unrelated memories with a false +20% score boost).
  - Entity prefix similarity (`similarity()` in `entities.py`) now requires a
    minimum 30% length ratio. Short prefixes like "her" no longer match "Hermes" at
    0.828.
  - Single-token strict fact queries (5+ chars, stopword-filtered) now match.
    Queries like "hermes", "python", "react" were silently rejected.
- `.codegraph/` no longer accidentally tracked in git.

### Changed

- `MNEMOSYNE_STRICT_FACT_MATCH` env var removed. Use `MNEMOSYNE_LENIENT_FACT_MATCH=1`
  to opt back into permissive fact matching.
- `RELEASING.md` added with official SemVer release policy.
- `.githooks/pre-push` hook validates tags match `__version__` and SemVer format.
- Git hooks path set to `.githooks` (run `git config core.hooksPath .githooks` on clones).

## [3.1.1] — 2026-05-28

### Added

- **Preferred embedding env vars.** `MNEMOSYNE_EMBEDDING_API_URL` and `MNEMOSYNE_EMBEDDING_API_KEY` are now the preferred names for custom embedding endpoints. The old `OPENROUTER_BASE_URL` and `OPENROUTER_API_KEY` names still work as fallbacks for backward compatibility. Restores the v2.8.x naming convention. ([#193](https://github.com/AxDSan/mnemosyne/issues/193))

## [3.1.0] — 2026-05-26

### Added

- **Shared surface memory CRUD.** Cross-agent shared memory database with dedicated read/write/search/delete/stats API. Each agent's shared surfaces are fully isolated from private memories. (`5a0b16a`)
- **Multilingual MEMORIA.** Language detection pipelines for German, Russian, and Chinese. MEMORIA now auto-detects the input language and applies language-specific extraction patterns. (`afd53c3`, `669a7cf`, `0f486cc`)
- **Custom embedding endpoints.** Configure any OpenAI-compatible embedding provider via `OPENROUTER_BASE_URL` (set to your own server URL), with Jina model dimension auto-detection and custom SSL cert support. Add `MNEMOSYNE_EMBEDDINGS_VIA_API=true` if using OpenRouter-hosted models. (`d0a8421`)
- **Deterministic `get(id)` primitive.** Direct memory retrieval by memory ID — no vector search, no ranking, just the exact memory. Useful for tool calls, confirmation UI, and graph traversal seed points. (`022929b`)
- **`hermes mnemosyne stats` command.** Exposes memoria-specific statistics (fact count, instruction count, preference count, language distribution) via the CLI. (`8b146dd`)
- **Chinese and multilingual embedding models.** Auto-dimension detection for models that don't expose fixed output sizes, enabling seamless use of multilingual embedding providers. (`f37f4bb`)
- **Community health files.** `CODE_OF_CONDUCT.md`, `SECURITY.md`, and a GitHub PR template for smoother community contributions. (`c2bf1d3`)
- **Community badges.** 100% Python badge added to README via shields.io. (`22e212f`)

### Fixed

- **sqlite-vec int8 search syntax.** The `AND k=N` clause (required by sqlite-vec's int8 vector type for proper search) replaces the standard `LIMIT` clause in vec_search. Without this fix, `int8` vector search silently returned wrong results. (`0a41e3b`)
- **Hermes plugin tool schemas.** All 6 hermes_plugin tool schemas now include the `bank` parameter, enabling multi-bank operation from the Hermes plugin layer. (`8cd718d`)
- **sqlite-vec extension loading.** `_get_connection` now correctly loads the `sqlite-vec` extension before any vector operations, preventing `no such function: vec_distance_cosine` crashes. (`a0de5f3`)
- **Working memory vector generation.** `remember()` now generates and persists the vector embedding on every call, not just during recall-time lazy generation. (`892f136`)
- **Active DB path in diagnose.** `mnemosyne diagnose` now reports the actual provider-level database path instead of the base config path. (`00ca612`)
- **Timezone normalization in temporal recall.** Temporal queries now properly normalize timezone-aware timestamps, fixing off-by-hour windowing errors. (`f4b18f7`)
- **MEMORIA regex cross-session dedup.** Tightened regex patterns to prevent fact duplication across sessions and improved metric extraction. (`81cc6fc`)
- **MULTILINGUAL_PATTERNS deduplication.** Removed duplicate `instruction` keys and false positive German patterns across multiple iterations. (`3f0e250`, `a16aa6e`, `cd3b1b2`)
- **E1 ingest type safety.** Fixed `tool count assertion` and `_lang string/int TypeError` during conversation ingestion. (`ed85e51`)
- **Fact accumulation metadata skip.** Fixed metadata keys being incorrectly counted in fact accumulation during `ingest_conversation`. (`86d8c1e`)
- **MEMORIA JSON parsing.** `_parse_facts` now handles both structured JSON and raw text output from the MEMORIA extraction prompt. (`d863220`)
- **String boolean config handling.** YAML config `true`/`false` strings are now properly coerced to Python booleans in `_apply_provider_config`. (`21a157d`)
- **Vector type probing.** Schema preservation during vector type probing prevents table corruption on re-probe. (`67fca7a`)
- **Sys.path ordering.** Fixed import resolution for `Hermes MemoryProvider` by moving sys.path setup before mnemosyne imports. (`62b0218`)
- **Test stability.** Patched lambda mocks and disabled embeddings in recall diagnostics tests to prevent CI flakiness. (`4ba74eb`, `066a3c6`, `e3bdc63`)
- **Config import in eval tool.** Moved logging import to module level in evaluation tool to prevent CI import errors.

### Changed

- **UPDATING.md rewritten.** Complete restructuring covering v2.7→v3.1 path, PEP 668 troubleshooting, and schema verification steps. (`dc170ce`)
- **README overhaul.** Centered hero section, table of contents, imperative tone throughout. (`887c8c0`)
- **BEAM benchmarks accuracy.** Corrected Hindsight benchmark from false 64.1% to 73.4% and removed unsupported SOTA claims. (`341c82e`)

### Removed

- **DEVOPS.md from git tracking.** Private operational doc removed from version control. (`34483af`)
- **Local scratch and benchmark artifacts.** Cleaned up development artifacts from the repo. (`7826de9`)
- **Personal emails from source files.** PII filter-repo scrub with .mailmap and PII pre-commit hook added. (`58507ea`)

## [3.0.0] — 2026-05-18

### Added

- **MEMORIA Architecture.** Structured fact extraction and retrieval system.
  New SQLite tables (`memoria_facts`, `memoria_timelines`, `memoria_kg`,
  `memoria_instructions`, `memoria_preferences`) with fact versioning,
  previous-value tracking, and valid-from/to windows.
- **Structured retrieval router.** `memoria_retrieve()` dispatches queries
  by ability (IE, MR, KU, TR, CR, EO, ABS, IF, PF, SUM) to specialized
  retrieval paths with different SQL strategies per question type.
- **Gap analysis loop.** Recursive re-querying for multi-hop and temporal
  questions. Extracts ISO dates from context, performs hard keyword
  searches for GAP-identified missing information.
- **Strict fact matching** (wysie, #143). Token-based conservative matching
  behind `MNEMOSYNE_STRICT_FACT_MATCH=1`. Filters stopwords, requires
  multi-token overlap or distinctive structural markers.
- **Proactive memory linking** (coe0718, #146). Zero-LLM graph edge creation
  at ingestion via content similarity (FTS5) and entity overlap strategies.
  Gated behind `MNEMOSYNE_PROACTIVE_LINKING=1`.
- **Benchmark LLM consolidation.** The evaluation harness now routes
  `beam.sleep()` summarization through OpenRouter with a cheap flash model
  instead of AAAK compression. The pipeline itself is unchanged — this is
  a benchmark config change only.

### Changed

- **Namespace migration.** All `nous_` tables/functions renamed to
  `memoria_` to avoid implying affiliation with any external entity.
- **Fact versioning.** Metrics with the same key now create version chains
  instead of overwriting. Previous values preserved for temporal recall.
- **Retrieval engine upgrade.** BEAM benchmark retrieval moved from
  FTS5-only to structured MEMORIA routing with 4-layer fallback.

### Fixed

- **KU key collision.** Context-aware metric keys prevent different metrics
  (e.g., `response_time_ms` vs `connection_timeout_ms`) from colliding on
  generic key names.
- **CR UNION search.** Contradiction resolution now searches both episodic
  memory and structured facts via UNION query.
- **EO strict JSON mode.** Event ordering prompts now force JSON-only output
  with negative examples to prevent rambling.
- **IE latest-value guidance.** Information extraction prompts now
  prioritize most recent values for evolving facts.
- **TR token bump.** Temporal reasoning max_tokens increased from 1024 to
  2048 to accommodate date extraction preamble.

### Performance

- BEAM 100K OVERALL: 65.2% (Llama 3.3 70B) — passes Honcho (63.0%)
- IE: 91.5%, MR: 87.5%, KU: 50%, TR: 75%, ABS: 100%
- Ingestion: 36s for 188 messages with full MEMORIA extraction

## [2.9.0] — 2026-05-17

### Fixed

- **MCP SDK 1.x compatibility** (`mcp_server.py`). The `stdio_server()`
  transport no longer accepts a `Server` object as argument since v0.9.1;
  the stream pair is obtained via `async with stdio_server()` and then
  passed to `server.run()`. Tool definitions are now returned as `Tool`
  Pydantic objects instead of raw dicts, matching the SDK 1.x `list_tools`
  handler signature. Both stdio and SSE transports are patched.

## [2.8.0] — 2026-05-14

### Added

- **CompressionPlugin** (`mnemosyne/core/plugins.py`) — new built-in plugin providing optional pre-compression of memory content before LLM summarization. Disabled by default; enabled via `MnemosyneConfig.compression.enabled = True` or the deprecated `MNEMOSYNE_USE_CAVEMAN=1` env var. Supports the `rust_cave_001` provider for stopword-based compression. Unknown providers fall back gracefully (no-op). Includes `compress_lines(text, provider)` method and `_plugins.get_manager().get_plugin("compression")` access point.
- **Deprecated env var** — `MNEMOSYNE_USE_CAVEMAN=1` still activates compression but emits a `DeprecationWarning` pointing to the config-based path (`MnemosyneConfig.compression.enabled = True`). `MNEMOSYNE_USE_CAVEMAN=0` explicitly disables it.
- **Test coverage** — 7 new tests in `tests/test_plugins.py` covering: disabled by default, enabled via config, `compress_lines` noop when disabled, `compress_lines` works with caveman provider, deprecated env var fallback, registered as builtin plugin, unknown provider fallback.
- **Provider tool parity (15 → 17 tools).** Added missing `export`, `import`, `diagnose`, `graph_query`, and `graph_link` tools to the Hermes memory provider.
- **Graph traversal & link memory.** BFS multi-hop traversal with `edge_type` and `min_weight` filtering, integrated into polyphonic recall's `_graph_voice`.
- **Entity extraction quality fix.** Case-insensitive meta-word stopword filtering blocks noise words (ASSISTANT, USER, SKILL) from mention annotations.
- **Bad domain database (669K entries).** Crowdsourced blocklists from BlocklistProject, Phishing Army, and URL shorteners. Sub-microsecond lookups for Discord link filtering.
- **IP:port detection in link filter.** Raw IP addresses like `182.3.4.5:8877` are now caught alongside domain-based URLs.
- **Automated version bump script.** Deterministic version bumper that updates all 8 version-carrying files and runs verification grep.

### Changed

- **Beam.py migration** — `beam.py` no longer directly imports and calls `rust_cave_001`. Instead it checks `_plugins.get_manager().get_plugin("compression")` and delegates to `CompressionPlugin.compress_lines()`. The `rust_cave_001` dependency is now fully encapsulated behind the plugin interface.
- **MNEMOSYNE_USE_CAVEMAN** — still activates compression but emits a `DeprecationWarning` pointing to the config-based path. Use `MnemosyneConfig.compression.enabled = True` instead.
- **Test assertion counts** — 3 existing assertion counts in `test_plugins.py` bumped from 3→4 to account for the 4th built-in plugin.

### Fixed

- **CI embedding timeout.** `fastembed` model downloads blocked subprocess tests. Added `MNEMOSYNE_NO_EMBEDDINGS` env guard and lazy-loading in `available()`.
- **Provider export/import routing.** Fixed handlers to route through the `Mnemosyne` wrapper instead of `BeamMemory` directly.
- **Stale version references.** Six files across the repo still displayed v2.7 after the initial v2.8.0 build (plugin yamls, docs pages, README badge, codebase surface). All corrected.
## [2.7.0] — 2026-05-12

### Fixed

- **LLM_MAX_TOKENS default too low for reasoning models (#81).** Default raised from 256 → 2048 tokens. Reasoning models (DeepSeek V4, Claude thinking, Kimi K2) need ~2K tokens to complete chain-of-thought and produce usable consolidation output. Previously `finish_reason=length` on reasoning models. Configurable via `MNEMOSYNE_LLM_MAX_TOKENS` env var.

### Added

- **Disaster recovery CLI commands (#69, D2+D3).** New `mnemosyne backup`, `mnemosyne restore`, `mnemosyne verify`, and `mnemosyne backups` commands. Backup and restore now use the sqlite3 online backup API (lock-aware, WAL-safe, atomic) instead of raw `shutil.copyfileobj`. Exposes the existing DR module (`mnemosyne/dr/recovery.py`) to users via first-class CLI.

- **Content sanitization on ingest (#69, D1).** `BeamMemory.remember()`, `remember_batch()`, and `Mnemosyne.remember()` now detect binary-shaped content and extract it to content-addressed blob storage (`~/.hermes/mnemosyne/blobs/`). Three-stage detection: (1) `data:` URI prefix decodes base64 payload, (2) >1MB content always extracted, (3) >100KB content with Shannon entropy >5.0 bits/char extracted. Prevents SQLite corruption and DB bloat from inline images, base64 payloads, and encoded blobs.

**E6.a — follow-up gaps surfaced by the E6 review**
- `Mnemosyne.forget()` and `BeamMemory.forget_working()` now cascade-delete annotations for the forgotten memory_id. Pre-fix, `mentions` / `fact` / `occurred_on` / `has_source` rows stayed in the annotations table after forget — they leaked through `export_to_file`, kept surfacing in `_find_memories_by_entity` and `_find_memories_by_fact`, and remained queryable through MCP tools. Privacy regression introduced by E6 (annotations table didn't exist pre-E6, so the cascade gap is new).
- `mnemosyne_triple_add` MCP tool now routes annotation-flavored predicates (`mentions`, `fact`, `occurred_on`, `has_source`) to `AnnotationStore.add()` instead of `TripleStore.add()`. Pre-fix, an agent calling the tool with `predicate="mentions"` would silently invalidate prior `(subject, "mentions")` annotation rows via the same auto-invalidation bug E6 was designed to fix — the bug remained reachable from the MCP layer. Current-truth predicates (anything outside `ANNOTATION_KINDS`) still route to `TripleStore` for backward compatibility.

**E6 — TripleStore silent-destruction bug**
- `TripleStore.add()` auto-invalidates rows with matching `(subject, predicate)` regardless of `object`. Every production write used annotation semantics (`(memory_id, "mentions", entity)`, `(memory_id, "fact", text)`, etc.), so each new annotation for a memory silently set `valid_until` on prior annotation rows with the same key. Effect: entity / fact graphs on each Mnemosyne database have lost data any time a memory had more than one entity or fact extracted.
- Fix splits storage into two purpose-specific tables:
  - `triples` table retains current-truth temporal semantics with auto-invalidation, suitable for facts like `(user, prefers, X)` later superseded by `(user, prefers, Y)`. No production caller writes here today; the table is preserved for future use.
  - New `annotations` table (`mnemosyne/core/annotations.py`, `AnnotationStore`) is append-only and now hosts `mentions`, `fact`, `occurred_on`, `has_source` — all multi-valued by design.
- Production call sites migrated to `AnnotationStore`:
  - `BeamMemory._extract_and_store_entities`, `_extract_and_store_facts`, `_add_temporal_triple`
  - `BeamMemory._find_memories_by_entity`, `_find_memories_by_fact`
  - `Mnemosyne.remember(extract_entities=True)` and `Mnemosyne.remember(extract=True)`
- **Auto-migration on first BeamMemory init.** Existing databases auto-migrate annotation-flavored rows from `triples` to `annotations` with a backup written to `{db}.pre_e6_backup`. Set `MNEMOSYNE_AUTO_MIGRATE=0` to disable auto-migration and run `python scripts/migrate_triplestore_split.py` manually instead.
- **`TripleStore.add_facts()` is deprecated.** Emits `DeprecationWarning`; legacy write behavior preserved for backward compatibility. New code should call `AnnotationStore.add_many(memory_id, "fact", facts)` directly.

### Added

- `mnemosyne/core/annotations.py` — `AnnotationStore` class + `ANNOTATION_KINDS` constant (`mentions`, `fact`, `occurred_on`, `has_source`)
- `scripts/migrate_triplestore_split.py` — idempotent, transactional, file-level-backup migration script with `--dry-run`, `--no-backup`, `--db PATH` flags
- `MNEMOSYNE_AUTO_MIGRATE` env var (default `1`; set to `0` for explicit operator control)
- `scripts/mnemosyne-stats.py` — new `annotations` section in JSON output alongside the existing `triples` section
- 30+ new tests covering the new store, the migration script, the auto-migrate hook, and end-to-end production-path regression guards

## [2.5] — 2026-05-10

### Added

**NAI-0 Algorithmic Sprint**
- `BeamMemory.format_context(results, format="bullet"|"json")` — structured context formatting
- `BeamMemory._sandwich_order()` — U-shaped attention ordering (high-first, medium-middle, high-last)
- `BeamMemory._fact_line()` — clean one-line fact format with date, source, confidence
- `BeamMemory._format_context_json()` / `_format_context_bullet()` — JSON and markdown output
- RRF (Reciprocal Rank Fusion) in `PolyphonicRecallEngine._combine_voices()` with k=60 constant
- Covering indexes: `idx_em_scope_imp`, `idx_wm_session_recall`, `idx_mem_emb_type`
- `tools/bench_nai0.py` — minimal 20-question benchmark for quick before/after measurement

**Self-Healing Quality Pipeline** (`scripts/heal_quality.py`, PR #67 by ether-btc)
- Detects degraded episodic memory entries (bullet-format, <300 chars) and repairs them via a 4-stage LLM-as-Judge closed loop: Extract → Generate → Judge → Repair
- Fault taxonomy: `truncated`, `generic`, `missing_facts`, `wrong_format`
- Judge scores 4 dimensions (factual density, format compliance, length sufficiency, grounding) each 0-100
- Repair strategies are fault-specific: context doubling, specificity enforcement, fact injection, format rewrite
- Loop with `MAX_RETRIES` (default 3) and automatic escalation to stronger model after 2 failures
- Quality provenance in `metadata_json`: `quality_score`, `judge_model`, `consolidated_at`, `fault_before_repair`, `retry_loop_count`
- Configurable via env: `MNEMOSYNE_HEAL_JUDGE_THRESHOLD`, `MNEMOSYNE_HEAL_MAX_RETRIES`, `MNEMOSYNE_HEAL_MIN_LEN`, `MNEMOSYNE_HEAL_BUDGET`, `MNEMOSYNE_HEAL_ESCALATE_AFTER`
- Works with any LLM backend (MiniMax M2.7 via mmx-cli, local GGUF, or remote OpenAI-compatible API)
- CLI: `python scripts/heal_quality.py [--detect-only] [--entry-id ID] [--dry-run]`

**Chunked LLM Summarization** (`mnemosyne/core/local_llm.py`)
- Splits large memory lists into context-window-sized chunks before summarization
- Two-pass: summarize each chunk individually, then consolidate chunk summaries
- Fixes truncation issues with smaller models (Qwen2.5-1.5B) on large sessions

### Changed
- `BeamMemory.recall()` default `top_k`: 5 → 40
- Polyphonic recall voice combination: weighted average → position-based RRF
- `mnemosyne/__init__.py`: version bump to 2.5.0

## [2.4] — 2026-05-07

### Added

**Hindsight Importer — migrate FROM Hindsight INTO Mnemosyne**
- New `HindsightImporter` class in `mnemosyne/core/importers/hindsight.py`
- Import from Hindsight JSON exports OR live Hindsight HTTP API (`/v1/default/banks/{bank}/memories/list`)
- Writes directly to `episodic_memory` (not working memory) — preserves original timestamps, fact types, session grouping, metadata, scope, and veracity
- Stable duplicate skipping via SHA256-based IDs (`hs_` prefix)
- Importance scoring derived from Hindsight `fact_type` (world=0.75, experience=0.65, observation=0.55) + proof_count bonus
- Full metadata preservation: hindsight_id, fact_type, context, dates, entities, chunk_id, tags, consolidation timestamps
- CLI: `mnemosyne import-hindsight <file.json|url> [bank]`
- Registered in provider registry alongside Mem0, Letta, Zep, Cognee, Honcho, SuperMemory
- 102 lines of regression tests: timestamp preservation, episodic-only import, stable duplicate skipping, FTS indexing, provider-registry usage

**Host LLM Adapter — route consolidation through Hermes' authenticated provider**
- New `mnemosyne/core/llm_backends.py` — tiny `LLMBackend` Protocol (one method: `complete()`), process-global registry, `CallableLLMBackend` dataclass for tests
- New `hermes_memory_provider/hermes_llm_adapter.py` — `HermesAuxLLMBackend` routes through `agent.auxiliary_client.call_llm(task="compression", ...)`
- `MnemosyneMemoryProvider.initialize()` registers the backend; `shutdown()` unregisters it with a brief drain for in-flight threads
- `summarize_memories()` and `extract_facts()` consult host first when `MNEMOSYNE_HOST_LLM_ENABLED=true`
- **Host-skips-remote rule (A3):** When host attempt produces no usable text, remote URL is skipped — falls straight to local GGUF. Prevents stale URL leaks.
- `llm_available()` returns `True` when host backend is registered, so Hermes-only users don't get short-circuited by `beam.sleep()`
- `on_session_end()` runs sleep in daemon thread with 15s join timeout; `shutdown()` drains 2s before unregistering
- Fact extraction uses `temperature=0.0` for determinism; consolidation stays at `0.3`
- 7 new tests covering registry round-trip, host-route precedence, A3 skip-remote rule, gate semantics, shutdown drain race, daemon exception logging, bullet-list output preservation
- Live end-to-end verified with `openai-codex` OAuth subscription through ChatGPT backend

### Why this matters

**Hindsight importer:** Before this, migrating FROM Hindsight required going through `remember()`, which assigned current timestamps and wrote to working memory. Historical memories lost their original context. Now Hindsight migrations preserve the full temporal record with zero data loss.

**Host LLM adapter:** Hermes users on OAuth-backed providers (ChatGPT/Codex subscriptions) could not use Mnemosyne's LLM-backed operations because `MNEMOSYNE_LLM_BASE_URL` expects an OpenAI-compatible API key endpoint, not OAuth. Now they can route through Hermes' already-authenticated auxiliary client with zero extra credentials.

---

## [2.3.1] — 2026-05-06

### Fixed

- **Auto-sleep consolidation blocks TUI agent**: `_maybe_auto_sleep()` now runs in a background thread with a 5-second timeout instead of synchronously. Local LLM summarization (ctransformers) can no longer hang the agent worker thread. (#23)
- `MNEMOSYNE_AUTO_SLEEP_ENABLED` env var now controls auto-sleep behavior. Default is `false` (disabled) for interactive safety. Set to `true` to re-enable.
- Config schema updated to reflect new default.

## [2.3] — 2026-05-05

### Added

**Tiered Episodic Degradation — long-term recall without unbounded growth**
- Three degradation tiers: Tier 1 (0-30d, full detail), Tier 2 (30-180d, LLM-compressed), Tier 3 (180d+, entity-extracted signal)
- Automatic tier promotion during `sleep()` — no manual maintenance
- Tier multipliers in recall scoring: cold memories need 4x stronger semantic match
- Configurable via `MNEMOSYNE_TIER2_DAYS`, `MNEMOSYNE_TIER3_DAYS`, `MNEMOSYNE_TIER*_WEIGHT`
- Mnemonics can now truthfully claim "remembers what you told it a year ago"

**Smart Compression — entity-aware tier 2→3 extraction**
- `_extract_key_signal()` scores sentences by entity density (proper nouns, acronyms, security terms, tech stack, urgency)
- Preserves facts buried anywhere in a long memory, not just the first sentence
- Configurable: `MNEMOSYNE_SMART_COMPRESS=1` (default on), `MNEMOSYNE_TIER3_MAX_CHARS=300`

**Memory Confidence — veracity signal for every memory**
- New `veracity` field: `stated`, `inferred`, `tool`, `imported`, `unknown`
- `remember(veracity="stated")` — set confidence at write time
- `recall(veracity="stated")` — filter by confidence level
- Recall applies veracity multiplier to scores (stated=1.0x, inferred=0.7x, tool=0.5x)
- `get_contaminated()` — surface non-stated memories for review
- Configurable weights via `MNEMOSYNE_*_WEIGHT` env vars

### Fixed
- `local_llm.summarize()` → `summarize_memories()` — would crash on LLM degradation path
- SQLite connection conflicts in batch degradation tests
- Removed hallucinated Phase 2 from roadmap

## [2.2] — 2026-05-02

### Added

**Cross-Provider Importers — migrate from any memory platform**
- New `mnemosyne/core/importers/` module with 6 provider importers
- **Mem0:** SDK pagination → REST → structured export fallback chain; preserves user/agent/app scoping
- **Letta (MemGPT):** AgentFile `.af` format parsing (JSON/YAML/TOML); memory blocks → working_memory, messages → episodic
- **Zep:** users → sessions → `memory.get()` per-session iteration; messages + summaries + facts extraction
- **Cognee:** `get_graph_data()` nodes/edges extraction; nodes → episodic memories, edges → triples
- **Honcho:** peers → sessions → `context()` + messages; peer identity preserved as author_id
- **SuperMemory:** `documents.list()` + `search.execute()`; container tags mapped to channel_id
- **Agentic importer:** generates ready-to-run Python migration scripts and AI agent instructions for all 6 providers

**CLI: `hermes mnemosyne import` extended**
- `--from <provider>` — import directly from Mem0, Letta, Zep, etc.
- `--list-providers` — show all supported providers with docs links
- `--generate-script` — generate a migration script for any provider
- `--agentic` — output instructions to give your AI agent for extraction
- `--dry-run` — validate and transform without writing

**Plugin tool updated**
- `mnemosyne_import` schema extended with `provider`, `api_key`, `user_id`, `agent_id`, `dry_run`, `channel_id` params

### Changed

- README: added "Migrate from other memory providers" section with examples

## [2.1] — 2026-05-02

### Added

**Multi-Agent Identity Layer**
- New columns `author_id`, `author_type`, `channel_id` on `working_memory` and `episodic_memory` with indexes
- `Mnemosyne(author_id=..., author_type=..., channel_id=...)` constructor params
- `remember()` auto-populates identity columns from session context
- `recall(author_id=..., author_type=..., channel_id=...)` filter params
- `get_stats(author_id=..., author_type=..., channel_id=...)` filter params
- Cross-session channel recall: when `channel_id` is provided, scope expands to include all memories in that channel regardless of session
- MCP server: per-connection instances replace module-level cache; identity via tool args or env vars (`MNEMOSYNE_AUTHOR_ID`, `MNEMOSYNE_AUTHOR_TYPE`, `MNEMOSYNE_CHANNEL_ID`)
- Hermes plugin `_get_memory()` reads identity from environment variables

### Changed
- MCP `_get_instance()` renamed to `_create_instance()` — creates fresh instances per connection
- Episodic memory SELECTs and recall-tracking UPDATEs use dynamic session/channel scope

## [2.0] — 2026-04-29

### Added

**Phase 1: Entity Sketching**
- Regex-based entity extraction (`@mentions`, `#hashtags`, quoted phrases, capitalized sequences)
- Pure-Python Levenshtein distance with O(min) space optimization
- Fuzzy entity matching with prefix/substring bonuses and configurable threshold
- `extract_entities=True` parameter on `remember()` — backward compatible, default False

**Phase 2: Structured Fact Extraction**
- LLM-driven fact extraction via `extract_facts()` and `extract_facts_safe()`
- Graceful fallback chain: remote OpenAI-compatible API → local ctransformers GGUF → skip
- Fact parsing with numbering/bullet cleanup, length filter, cap at 5 facts

**Phase 3: Temporal Recall**
- Exponential decay temporal scoring: `exp(-hours_delta / halflife)`
- `temporal_weight`, `query_time`, `temporal_halflife` parameters on `recall()`
- Environment variable `MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS` for global default
- Temporal boost applied across all recall tiers (working, episodic, entity, fact)

**Phase 4: Configurable Hybrid Scoring**
- User-tunable scoring weights: `vec_weight`, `fts_weight`, `importance_weight`
- `_normalize_weights()` with env var fallback and sensible defaults (50/30/20)
- Per-query weight overrides without global state mutation

**Phase 5: Memory Banks**
- `BankManager` class for named namespace isolation
- Per-bank SQLite files under `banks/<name>/mnemosyne.db`
- Bank operations: create, delete, list, rename, exists check, stats
- `Mnemosyne(bank="work")` constructor parameter
- Bank name validation (alphanumeric + hyphens/underscores, max 64 chars)

**Phase 6: MCP Server**
- Model Context Protocol server with 6 tools
- stdio transport (Claude Desktop, etc.) and SSE transport (web clients)
- Per-bank instance caching
- CLI entry: `mnemosyne mcp`

**Phase 7: Hermes Agent Integration**
- 15 Hermes tools: remember, recall, stats, triple_add, triple_query, sleep, scratchpad_write/read/clear, invalidate, export, update, forget, import, diagnose
- 3 lifecycle hooks: `pre_llm_call` (context injection), `on_session_start`, `post_tool_call`
- AAAK compression for context injection
- Session-aware memory instances

**Phase 8: v2 Differentiation**
- `MemoryStream` — push (callbacks) and pull (iterator) event stream, thread-safe
- `DeltaSync` — checkpoint-based incremental synchronization between instances
- `MemoryCompressor` — dictionary-based, RLE, and semantic compression
- `PatternDetector` — temporal (hour/weekday), content (keyword, co-occurrence), sequence patterns
- `MnemosynePlugin` ABC with 4 lifecycle hooks
- `PluginManager` with auto-discovery from `~/.hermes/mnemosyne/plugins/`
- 3 built-in plugins: `LoggingPlugin`, `MetricsPlugin`, `FilterPlugin`

### Changed

- **CLI rewritten** — all commands now use v2 `Mnemosyne`/`BeamMemory` instead of stale v1 `MnemosyneCore`
- **SQLite WAL mode** — both `memory.py` and `beam.py` now use WAL journal mode with 5s busy timeout for better concurrency
- **FastEmbed cache** — model cache persists at `~/.hermes/cache/fastembed` instead of ephemeral `/tmp`
- **Legacy dual-write** — uses `INSERT OR REPLACE` for dedup safety

### Fixed

- `cli.py` DATA_DIR hardcoded to stale v1 path — now uses `MNEMOSYNE_DATA_DIR` env var
- Duplicate `_recency_decay()` definitions in `beam.py` merged into single function
- SQLite concurrency test failures — WAL mode + proper tearDown cleanup
- `plugin.yaml` declared only 9 of 15 tools — now declares all 15

### Tests

- 292 tests passing (up from unknown baseline)
- New test files: `test_entities.py`, `test_entity_integration.py`, `test_banks.py`, `test_mcp_tools.py`, `test_streaming.py`, `test_temporal_recall.py`
- All test tearDown methods handle WAL `-wal`/`-shm` files

---

## [1.13] — 2026-04-28

### Added

- **Temporal queries** — query the knowledge graph with time awareness (`temporal_halflife`, `temporal_weight`)
- **Memory bank isolation** — separate namespaces for different projects or contexts
- **Configurable hybrid scoring** — tune vector vs. FTS vs. importance weights per query
- **PII-safe diagnostic tool** (`mnemosyne_diagnose`) — inspect your memory without exposing sensitive data

### Fixed

- `sqlite-vec` LIMIT parameter handling
- Triples module-level helpers
- Embeddings fallback when `sqlite-vec` is absent
- Memory embeddings table auto-creation for sqlite-vec fallback

---

## [1.12] — 2026-04-26

### Added

- **Feature comparison matrix** vs. cloud providers (Honcho, Zep, Mem0, Hindsight)
- **DevOps policy** — comprehensive procedures for releases, security, and operations

### Changed

- Documentation cleanup — replaced placeholder files with proper repo docs

---

## [1.11] — 2026-04-25

### Added

- **Token-aware batch sizing** in consolidation — no more OOM on large memory sets
- **Remote API support** for LLM summarization in `sleep()`

### Fixed

- Consolidation edge cases with mixed local/remote LLM configs

---

## [1.10] — 2026-04-24

### Added

- **`mnemosyne_update` tool** — modify existing memories without full replacement
- **`mnemosyne_forget` tool** — targeted memory deletion
- **Global stats flag** — `hermes mnemosyne stats --global` for workspace-wide metrics

### Fixed

- Working memory scope handling across sessions (PR #11)
- Default scope set to 'global' for migrated memories
- Working memory stats and recall tracking consistency

---

## [1.9] — 2026-04-23

### Added

- **PyPI release** — `pip install mnemosyne-memory` works out of the box
- **CI/CD pipeline** — GitHub Actions for testing and release automation
- **`pyproject.toml`** — modern Python packaging
- **UPDATING.md** — migration guide for existing users

### Fixed

- Plugin `register()` export for Hermes plugin loader discovery
- Cross-session recall inconsistency (Issue #7, Bug 2)
- Subagent context write blocking (PR #8)

---

## [1.8] — 2026-04-22

### Added

- **Plugin auto-discovery** — `register()` method for Hermes plugin CLI
- **Bug report template** — official GitHub issue template

### Fixed

- 6 bugs from Issue #6 — edge cases in recall, scope handling, and tool registration

---

## [1.7] — 2026-04-22

### Added

- **PEP 668 PSA** — documentation for Ubuntu 24.04 / Debian 12 users hitting `externally-managed-environment`

### Fixed

- Provider `register_cli` using nested parser instead of subparser
- `sys.path` injection with graceful `ImportError` fallback

---

## [1.6] — 2026-04-21

### Added

- **Feature request template** — GitHub issue template for enhancements
- **Simple versioning** adopted — MAJOR.MINOR instead of semver

### Fixed

- `fastembed` dependency correction (was incorrectly listing `sentence-transformers`)
- Benchmarks restored to README with LongMemEval scores

---

## [1.5] — 2026-04-20

### Added

- **Export/import** — cross-machine memory migration (`mnemosyne_export` / `mnemosyne_import`)
- **One-command installer** — `curl | bash` setup for new users
- **MemoryProvider mode** — deploy Mnemosyne as a standalone memory provider via plugin system
- **Anchored table of contents** in README

### Changed

- README fully rewritten — professional, community-focused, removed bloat
- FluxSpeak branding removed from LICENSE and metadata (Mnemosyne is its own thing)

---

## [1.4] — 2026-04-19

### Added

- **Temporal validity** — memories can have expiration dates
- **Global scope** — memories visible across all sessions
- **Local LLM-based sleep()** — summarization without cloud APIs
- **Recall tracking** — knows what you already remembered
- **Recency decay** — older memories naturally fade in relevance

### Fixed

- Path type bug in memory override skill
- `plugin.yaml` moved to repo root for Hermes compatibility

---

## [1.3] — 2026-04-17

### Added

- **Memory override skill** — bake memory into pre_llm_call and session_start hooks
- **Critical deprecation notice** for legacy memory tool

---

## [1.2] — 2026-04-13

### Added

- **Scale limits** — tested and documented for 1M+ token capacity
- **Legacy DB migration script** — upgrade path from early schemas

### Changed

- Auto-logging of `tool_execution` disabled by default (privacy)

---

## [1.1] — 2026-04-10

### Added

- **BEAM architecture** — sqlite-vec + FTS5 + sleep consolidation
- **BEAM benchmarks** — dedicated benchmark suite with published results
- **Dense retrieval** via fastembed
- **AAAK compression** — compressed memory format for context injection
- **Temporal triples** — structured fact storage with subject/predicate/object

### Fixed

- Thread-local connection bug

---

## [1.0] — 2026-04-05

### Added

- **Initial release** — zero-dependency AI memory system
- **`remember()` / `recall()` / `sleep()`** — core memory cycle
- **SQLite + fastembed embeddings** — local vector search
- **Hermes plugin registration** — basic tool integration
- **AAAK compression** — early context compression for token limits

[2.4]: https://github.com/AxDSan/mnemosyne/releases/tag/v2.4
[2.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v2.0
[1.13]: https://github.com/AxDSan/mnemosyne/releases/tag/v1.13
[1.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v1