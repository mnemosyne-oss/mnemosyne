# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [SemVer](https://semver.org/) starting from v3.1.2.

## [Unreleased]

## [3.9.0] — 2026-06-18

### Added

- **Synchronous memory reindex** (issue #308, PR by @Milgauss). New `mnemosyne
  reindex` command that rebuilds all vectors (working, episodic, facts) after an
  embedding model or dimension change. Reuses existing write helpers for
  consistent encodings across all five representations. Auto-backup first,
  `--dry-run`, `--model`, `--no-backup`, `--yes`. Synchronous/blocking with a
  duration warning.
- **vec_working migration diagnostics** (contributed by Denis H). `mnemosyne
  diagnose --repair-vec-working` reports migration coverage and idempotently
  backfills missing vec_working rows from the memory_embeddings fallback.
- **Bidirectional memory sync** with optional client-side encryption
  (issue #287). Event-log-based delta sync between Mnemosyne instances using
  the SyncEngine protocol:
  - `memory_events` table: append-only event log with conflict detection
  - stdlib-only HTTP sync server (no FastAPI deps)
  - `mnemosyne sync`, `sync-serve`, `sync-status`, `sync-generate-key` CLI
  - Encrypted payload detection and causal version chains for conflict
    resolution
  - Sync tutorial, troubleshooting guide, and deploy configs (Docker, Caddy,
    Fly.io)
- **Hermes plugin improvements:**
  - `mnemosyne-hermes upgrade` — smart install-method detection (pipx / uv-tool
    / pip), version comparison, auto re-register after upgrade (PR #319)
  - `mnemosyne-hermes cleanup` — removes plugin, old hermes-mnemosyne dir,
    resets config; `--dry-run` safe (PR #317)
  - `mnemosyne-hermes status` now shows Hermes' Python version + mismatch
    warning (PR #316)
  - `install --dry-run` for safe pre-flight checks
  - Sync tool schemas (SYNC_PUSH, SYNC_PULL, SYNC_STATUS) added to both
    provider copies. Total tool count: 25 -> 28
- **Sleep orphan-claim recovery** (issue #293). Added `reclaim_orphans()`
  to clear stale consolidation claims when `sleep()` was interrupted after
  claiming working-memory rows but before writing an episodic summary.

### Changed

- **vec_working dedicated table for working vector search** (contributed by
  Denis H). Working-memory vectors now live in a dedicated sqlite-vec table,
  with memory_embeddings as the compatibility fallback. New rows written to
  both, recall prefers vec_working when available. Import/backfill paths
  mirror to both stores.
- **CLI version no longer depends on `__author__`** (removed in v3.7.0).
  Imports `__version__` only for resilience across releases.
- **Lower prefetch noise from raw conversation turns.** sync_turn() now writes
  user messages at 0.5 importance (was 0.3) and assistant messages at 0.15
  (was 0.2).

### Fixed

- **fact_recall ranking by query relevance** (issue #309, PR by @Milgauss).
  fact_recall() now preserves FTS rank order (was re-ordering by stored
  confidence, collapsing all facts from the same path to one score), uses
  `relevance * confidence` scoring, and returns full subject-predicate-object
  triples as content. Opt-in via `MNEMOSYNE_FACT_RECALL_ENABLED`.
- **Audit log table renamed to `audit_log`** to avoid collision with the sync
  engine's `memory_events` table. Both were creating tables named
  `memory_events` with incompatible schemas — the audit silently failed on
  INSERT after beam.py created its version first.
- **UTC Z timestamp parsing on Python 3.10** in sync conflict detection.
  Normalizes trailing `Z` before `datetime.fromisoformat()`.
- **Security docs corrected** — documentation claimed XChaCha20 and keyring
  integration; actual code uses Fernet/XSalsa20 and key-manager-only key
  sources. `from_config()` scope fixed.
- **Provider diagnostic messages** — `register_memory_provider()` now catches
  construction failures and prints the actual exception, Python version, and
  Hermes' Python info to stderr instead of a vague warning.

### Performance

- **Dedicated vec_working table** — working vector search uses a focused
  sqlite-vec table instead of the shared memory_embeddings table, reducing
  candidate set size.
- **Query embedding cached once per recall() call** (PR #298). Previously the
  embedding model was invoked multiple times from different filter paths within
  the same recall.
- **Get_context hot path split** (contributed by Denis H). Separate global and
  session queries with targeted indexes instead of a broad OR or
  temporary-sort query shape.

## [3.7.0] — 2026-06-13

### Added

- **Usage-driven working memory decay** (issue #289). Memory now lives longer
  (default TTL 168h, was 24h), and frequently recalled items get their TTL
  bumped (capped at `MNEMOSYNE_WM_BUMP_CAP_HOURS`, default 24h per bump).
  - `MNEMOSYNE_WM_BUMP_CAP_HOURS` env var — configurable refresh ceiling
  - `MNEMOSYNE_WM_PINNED_IDS` env var — comma-separated memory IDs to pin
  - `pinned` column on `working_memory` — sleep consolidation skips pinned items

### Fixed

- **Temporal-triple lifecycle re-applied** (issue #246 regression). Triple
  `supersede`/`valid_until`/`end` lifecycle was absent from v3.5.0 and v3.6.0
  despite appearing merged. Re-applied cleanly. New `mnemosyne_triple_end` tool
  and `end_triple()` module function added.
- **Optional local LLM fallback log level.** `diagnose` no longer logs a
  warning when the optional fallback model is absent.
- **`sleep(force=False)` assertion corrected.** The `force` flag path now works
  without throwing.
- **`HERMES_HOME` resolution priority.** Check `HERMES_HOME` env var before
  falling back to `Path.home()` across beam, banks, memory, and integration
  files.
- **Packaging cleanup:** `openclaw` dependency removed from `[all]` extra.
  Python 3.9 classifier dropped (3.10+ only).

## [3.6.0] — 2026-06-10

### Added

- **Owner-scoped canonical (single-source-of-truth) facts** (issue #256). A new
  `CanonicalStore` (`mnemosyne/core/canonical.py`) gives long-running personas an
  identity layer where each `(owner_id, category, name)` slot holds exactly one
  current value. Restating a stable self-fact is a no-op (no duplicate
  accumulation); a new value supersedes the old one, which is preserved as
  history — the TripleStore `valid_until` pattern, extended with an owner
  dimension. Implemented as **one SQLite table plus a partial unique index**
  (`… WHERE valid_until IS NULL`); no new dependency, no FTS table.
  - Two new tools, `mnemosyne_remember_canonical` and `mnemosyne_recall_canonical`
    (the latter covers exact-slot read, category/whole-bank listing, version
    history, and owner-scoped substring search). Exposed on both the Hermes
    provider and the MCP surface — total tool count 23 → 25.
  - `BeamMemory` now exposes `self.canonical`, sharing its thread-local
    connection (no extra file descriptor), mirroring `self.annotations`.
  - Owner isolation is enforced by construction: the provider derives `owner_id`
    from the active profile identity and never reads it from tool args, so one
    profile cannot read or write another's canonical bank. The shared surface is
    untouched and keeps its cross-profile role.
  - Fully additive and opt-in: the `canonical_facts` table is created lazily on
    first init; existing tables, tools, and recall output are unchanged.

- **Hermes Holographic Memory importer** (`mnemosyne/core/importers/holographic.py`).
  Reads directly from Hermes' SQLite-based holographic memory plugin
  (`~/.hermes/memory_store.db`) — preserves content, category, tags, trust scores,
  timestamps, and entity links. Trust scores map to Mnemosyne importance (both 0-1).
  Entity extraction flag passes through to `mnemosyne.remember()` for annotation-store
  entity recall. Category/tag/min_trust filtering for targeted imports.
  Fully dry-run compatible. (`--from holographic`)

- **API embedding fallback chain.** `embed()` and `embed_query()` now fall through
  to local fastembed when the API embedding call fails (network outage, rate limit,
  timeout). The fallback model is configurable via `MNEMOSYNE_EMBEDDING_FALLBACK_MODEL`
  (default: `BAAI/bge-small-en-v1.5`). `available()` now accounts for fallback
  capability, so recall doesn't skip vector search just because the API is down.
  (#269)

### Fixed

- **Fact recall no longer treats one plain shared word as relevance for broad
  queries.** Single-token fact matches are now limited to lookup-style queries or
  distinctive structured identifiers, preventing unrelated high-importance facts
  from surfacing on conversational glue words while preserving direct lookups.

- **Holographic import CLI no longer demands an API key.** Holographic is a local
  SQLite importer (no API key needed) but the generic provider path checked for
  `--api-key` on every non-`hindsight` provider. Added `--db-path` and `--min-trust`
  CLI flags and a holographic special case (same pattern as hindsight) that skips
  the key gate. Import parity with docs at `api-reference.md` is now operational.

- **Provider registration + db_path on non-isolated init** (fixes #254, #255).
  `register()` now calls `register_memory_provider()` — the provider was silently
  failing to load. `BeamMemory()` now derives `db_path` from `hermes_home` when
  available instead of falling back to `Path.home()`, preventing silent data loss
  across processes. Installer auto-cleans old `hermes-mnemosyne` plugin directory
  and migrates config.

- **Embeddings deps are now unconditional.** Vector search (fastembed + sqlite-vec)
  is not optional — it's what makes recall work. The `[embeddings]` extra is now
  a hard dependency, so fresh installs don't silently ship with FTS5-only keyword
  search.

- **Hermes host LLM registration in CLI path.** Both copies of `cli.py` now call
  `register_hermes_host_llm()` before creating `BeamMemory`. Previously the
  registration only happened inside `MnemosyneMemoryProvider.initialize()` which the
  CLI handler never hits, so `MNEMOSYNE_HOST_LLM_ENABLED=true` was silently ignored
  when running `hermes mnemosyne sleep` from the terminal.

- **Per-entity identity injection in prefetch.** The provider now includes per-contact
  identity memories in every prefetch regardless of recall query, ensuring the agent
  always has the user's stable self-descriptors without requiring an explicit identity
  search.

- **Entity performance: skip Levenshtein when length ratio rules out a match.**
  The prefix-guard branch now bails out early when the token length ratio exceeds a
  threshold, avoiding expensive string edits on obviously non-matching candidates.

- **Docs generator overhaul.** Rewritten to be merge-conflict-free, single-source
  ground truth (24 MCP tools, 9 config keys), canonical copies always written to
  `docs/api/`. Website sibling writes guarded with `isdir` + `isfile` checks.
  Removed ghost `mnemosyne_end` tool (23 real tools). Plugin path corrected from
  `~/.hermes/plugins/memory/mnemosyne/` to `~/.hermes/plugins/mnemosyne/`. Switched
  from hardcoded `python3.11` path to dynamic resolution.

### Tests

- **Recall relevance before importance** (contributed by [WXBR](https://github.com/WXBR)).
  Proves high-importance unrelated memories cannot surface for an unrelated query.
  Locks in the invariant that importance may boost ordering only after a candidate
  has passed relevance, instead of rescuing unrelated rows.

## [3.4.0] — 2026-06-01

### Added

- **Known dimensions for local SentenceTransformers multilingual models.**
  `paraphrase-multilingual-MiniLM-L12-v2`, `all-MiniLM-L6-v2`, and
  `paraphrase-multilingual-mpnet-base-v2` are now listed for low-resource
  local multilingual embedding setups.

### Fixed

- **Unicode recall tokenization for Latin-script languages.** Recall lexical
  gates now keep diacritics inside tokens, so words like `Stoßlüften`,
  `Bürgeramt`, and `Primärquellen` are no longer split into ASCII fragments.

## [3.3.0] — 2026-06-01

### Added

- **`sync_roles` config for role-based autosave filtering.** `sync_turn()` now
  checks `memory.mnemosyne.sync_roles` before persisting conversation turns.
  Default `["user", "assistant"]` preserves existing behavior. Set to `["user"]`
  to save only user turns, or `[]` to disable conversation autosave while keeping
  explicit `mnemosyne_remember` calls working. Unknown roles are warned and ignored.
  (Contributed by **bitr8**, closes #209.)
- **`MNEMOSYNE_SYNC_TURN_USER_LIMIT` / `MNEMOSYNE_SYNC_TURN_ASSISTANT_LIMIT` env vars.**
  `sync_turn()` now respects configurable truncation limits instead of hardcoded
  500/800 slices. Defaults to `500` (user) and `800` (assistant) for backward
  compatibility. Set to `0` to disable truncation.
- **Fact recall merged into standard `beam.recall()` path.** Set
  `MNEMOSYNE_FACT_RECALL_ENABLED=1` to merge LLM-extracted facts (from `extract=true`)
  into recall results. Facts are deduplicated against regular memories by content
  hash and scored at 0.9x their confidence.
- **Auto-default `scope=global` when `extract=true`.** If a caller doesn't
  explicitly pass `scope`, setting `extract=true` now infers `scope=global`
  instead of the default `session`. Explicit scope overrides are respected.
- **`fact_recall()` now searches `consolidated_facts`** (sleep-consolidated fact
  triples) in addition to the raw `facts` table. Previously only accessible
  through polyphonic recall (`MNEMOSYNE_POLYPHONIC_RECALL=1`). Fact data stored
  with `extract=true` is now visible through the default recall path.
- **`MNEMOSYNE_EMBEDDING_API_URL` independent of `OPENROUTER_BASE_URL`.**
  Embedding models can now use local llama.cpp, OpenAI, Anthropic, or any
  other provider without requiring OpenRouter configuration. Also fixes a bug
  where `_OPENAI_BASE_URL` was stale after env read. (Contributed by
  **mia-fourier**, PR #206.)

### Fixed

- **`remember()` silently never stored embeddings.** Only `remember_batch()`
  called `_vec_insert()`. The Hermes provider uses `remember()`, so thousands
  of working memories had no vectors, making conflict detection always a no-op
  and degrading vector recall quality. Added `_vec_insert()` call to `remember()`.
  Threshold for conflict detection relaxed from 0.92 to 0.88 (32 conflicts found
  vs 23 in real data).
- **Hardcoded embedding dimension in `binary_vectors.py`.** `EMBEDDING_DIM` was
  hardcoded to 384 (bge-small-en-v1.5), causing `maximally_informative_binarization`
  to silently truncate larger embeddings (e.g. 1024-dim multilingual-e5-large) to
  the first 384 components, losing up to 62.5% of vector information. The dimension
  is now derived from `mnemosyne.core.embeddings.EMBEDDING_DIM` at import time with
  a 384 fallback when the embeddings module is unavailable. `BYTES_PER_VECTOR`,
  `compression_ratio`, and `theoretical_size_mb` in `get_stats()` are likewise
  computed from the resolved dimension instead of hardcoded constants.
  (Contributed by **Whishp**, PR #200.)
- **Same hardcoded 384 in `shmr.py` and `polyphonic_recall.py`.** `shmr.py` used
  the identical hardcoded constant. `polyphonic_recall.py` hardcoded `384` for
  bit-type vector normalization, silently breaking for non-384-dim models.
  Both now derive from `embeddings.EMBEDDING_DIM`. (Contributed by **Whishp**.)
- **Last hardcoded 384 in `test_integration.py`.** `np.random.randn(384)` on
  line 238 missed in the earlier pass. Now uses EMBEDDING_DIM like the rest.
  (Contributed by **Whishp**.)
- **Plugin directory named `mnemosyne` shadows pip package.** Hermes adds
  `~/.hermes/plugins/` to `sys.path`, so a symlink named `mnemosyne` resolves
  before the actual `mnemosyne-memory` pip package, causing `ModuleNotFoundError`
  on `from mnemosyne.core.memory import Mnemosyne`. The try/except swallowed
  this silently — tools never registered. Renamed to `hermes-mnemosyne`.
  (Fixes #212.)
- **Cross-session deletion of scope=global memories blocked.** `forget_working()`
  used `WHERE id = ? AND session_id = ?`, preventing deletion of global memories
  returned by recall() from a different session. Now uses the same pattern as
  `invalidate()`: `WHERE id = ? AND (session_id = ? OR scope = 'global')`.
  (Fixes #204.)
- **`_vec_insert()` ran inside deferred transaction.** sqlite-vec virtual table
  writes were silently lost when the transaction never committed. Now commits
  after each `_vec_insert` call. (Contributed by **chinesewebman**.)
- **`shutil.rmtree()` crashes on symlink targets.** Users who installed via
  `deploy_hermes_provider.sh` have a symlink at `~/.hermes/plugins/mnemosyne/`.
  `shutil.rmtree()` raises `Cannot call rmtree on a symbolic link`. Fixed with
  `is_symlink()` detection and `unlink()` fallback.
- **Directory junctions used on Windows.** Instead of symlinks (which require
  admin), the installer now creates directory junctions. No admin required.
- **Dead `hermes_plugin` tests breaking CI collection.** 4 test files still
  imported from the removed `hermes_plugin/` directory, causing
  `ModuleNotFoundError` and killing the entire test suite. Deleted:
  `test_hermes_plugin_session.py`, `test_hermes_plugin_tools.py`,
  `test_c13_memory_context_single_injection.py`,
  `test_c27_provider_init_error_visible.py`. Pruned 2 MCP-routing classes
  from `test_e6a_followup_gaps.py`.

### Changed

- **refactor: modular Hermes provider.** Split the 2007-line `__init__.py`
  monolith into 5 clean modules: `tools.py` (460L — 23 tool schemas),
  `__init__.py` (1515L — MemoryProvider), `audit.py` (138L),
  `cli.py` (332L), `hermes_llm_adapter.py` (164L). Moved to
  `integrations/hermes/src/mnemosyne_hermes/` following the MemoriLabs
  pattern. Ships as standalone `mnemosyne-hermes` pip package. Removed
  legacy `hermes_plugin/` directory, root `plugin.yaml`, and
  `deploy_hermes_provider.sh` hack.
- **refactor: consolidate `extensions/` and `hermes/` into `integrations/`.**
  Single directory for all external adapters: `integrations/hermes/`,
  `integrations/obsidian-mnemosyne/`, `integrations/vscode-mnemosyne/`.
  Python-package integrations stay in `mnemosyne/integrations/`.
- **Drop Python 3.9 CI support.** EOL since Nov 2025. `requires-python`
  bumped to `>=3.10` in `pyproject.toml` and `setup.py`. MCP and OpenClaw
  extras already gated on `>=3.10`, so this formalizes existing behavior.
- **`MNEMOSYNE_EMBEDDING_API_URL` env var no longer falls back to
  `OPENROUTER_BASE_URL`.** Embedding providers are independent of the
  general routing endpoint.

### Documentation

- **LongMemEval 98.9% recall benchmark restored** to README alongside BEAM
  numbers. Comparison table now shows both: `65.2% BEAM / 98.9% LongMem`.
- **Hermes Plugin section** revamped: 23 tools in 5 categories, pip install
  `mnemosyne-hermes` flow, `hermes tools disable memory` step, updated TOC.
- **Standalone README** for `mnemosyne-hermes`: Memori-inspired, no em-dashes,
  professional formatting, header image.
- **Hermes-first positioning** in root README.
- **Advise disabling built-in Hermes memory** when using Mnemosyne (prevents
  double-injection and token waste).
- **Multilingual embedding setup** documented in README with `MNEMOSYNE_EMBEDDING_MODEL`
  env var and Language Support section.
- **New env vars documented** in `integrations/hermes/README.md` config table:
  `SYNC_TURN_USER_LIMIT`, `SYNC_TURN_ASSISTANT_LIMIT`, `FACT_RECALL_ENABLED`,
  `PREFETCH_CONTENT_CHARS`.
- **Install script link fixed** in `hermes-mcp.md`. (Contributed by
  **Joao Fernandes**, PR #201.)
- **UPDATING.md** updated for v3.1.2 release notes.

### Tests

- 26 tests for `sync_roles` config (bitr8)
- 8 tests for sync_turn content limit env vars
- 4 tests for fact recall integration
- 5 tests for auto-scope-global
- Pre-existing fact concurrency, polyphonic, and prefetch tests preserved and passing

**Contributors:** Abdias J, Whishp, mia-fourier, bitr8, chinesewebman, Joao Fernandes

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

[3.7.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.7.0
[3.6.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.6.0
[3.5.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.5.0
[3.4.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.4.0