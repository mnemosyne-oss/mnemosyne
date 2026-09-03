# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [SemVer](https://semver.org/) starting from v3.1.2.

## [Unreleased]

### Added

- **BEAM initialization status is now available through the additive public Python `BeamInitResult`.** It reports the configured embedding dimension, any dimension mismatch, and immutable stored dimensions for each vector table.
- **Remote consolidation: a per-endpoint extra request body, and an empty answer is named (#878).** `MNEMOSYNE_LLM_EXTRA_BODY` and `MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY` each take a JSON object that is merged into the chat-completions payload last, one for the primary endpoint and one for the fallback endpoint, so a provider-specific key such as a thinking-mode toggle rides the same request; unset or invalid means nothing is merged, and the reserved keys `messages`, `model` and `stream` are dropped so the escape hatch cannot silently change the request the logs describe. A 2xx reply with no answer text (a thinking model that spends the whole `max_tokens` budget on reasoning returns `finish_reason=length`, `reasoning_content` set, `content` empty) now comes back from `_call_remote_llm_with_model` as an `EmptyAnswer` carrying `finish_reason` and `usage.completion_tokens_details.reasoning_tokens`. `local_llm.last_llm_failure()` keeps the most recent remote failure as `model: reason`, and the `sleep()` WARNING that announces the AAAK fallback carries it as `last_error=`, where before it named no cause.
- **Multimodal memory: images, video and audio can become recallable memories (RFCs 0002, 0003, 0004).** `BeamMemory.remember_media(ref)` takes a reference to a piece of media, registers it, describes it through a configured modality provider, and writes the description back as an ordinary memory that hybrid recall already understands. Nothing about text recall changes.

  The stack is additive throughout. Two sidecar tables, `media_assets` and `media_moments`, are created `IF NOT EXISTS` by their own store when a bank is opened, so existing databases acquire them with no migration step and no change to any existing table. No new package dependency is introduced.

  It is off unless configured. `modality_enabled` defaults to `false` and every endpoint and model key defaults to empty, so an installation that does not opt in behaves exactly as before. The provider seam is named after the protocol rather than a vendor: `MNEMOSYNE_MODALITY_BASE_URL`, `MNEMOSYNE_MODALITY_API_KEY`, `MNEMOSYNE_MODALITY_VISION_MODEL`, `MNEMOSYNE_MODALITY_VIDEO_MODEL`, `MNEMOSYNE_MODALITY_AUDIO_MODEL` and `MNEMOSYNE_MODALITY_TIMEOUT` point it at any OpenAI-compatible endpoint, and a second backend can be added without inheriting the first one's name.

  `remember_media()` returns a `MediaIngestResult` rather than a bare id, because the ingest path degrades in stages and the caller needs to see which one it landed on: `ok`, `partial`, `unavailable` or `refused`. `unavailable` is a success, not an error. It means the asset was registered and can be described later once a provider is configured.

  Supporting pieces: `ContentResolver` with a `BlobResolver` implementation gives the blob store a reader, so a stored reference can be turned back into bytes; `remember()` accepts an explicit `memory_type` that overrides the content classifier and `dedupe=False` for callers that must write a row per call, both defaulting to current behaviour, with an unrecognized `memory_type` logging a warning and falling back to classification rather than writing a bad value; and `mnemosyne doctor` grows a media orphan check that counts both orphan kinds while treating only one of them as a warning, reporting reference columns only and never user content.
- **Native MCP Streamable HTTP transport for `mnemosyne mcp`.** `--transport streamable-http` (alias `http`) serves the modern MCP `http` transport on a single configurable endpoint (`--path`, default `/mcp`) that handles GET, POST, and DELETE, so clients POST JSON-RPC directly to it with no `/messages` route to proxy. Responses stream via SSE upgrade by default or are JSON-only with `--json-response`. Auth policy matches SSE: loopback binds need no token; non-loopback binds require `MNEMOSYNE_MCP_TOKEN` bearer auth. A non-loopback `streamable-http` bind exposes the selected local SQLite-backed memory bank to network clients and additionally requires `MNEMOSYNE_MCP_ALLOWED_HOSTS`, with `MNEMOSYNE_MCP_ALLOWED_ORIGINS` optionally restricting browser origins. Tracks #598 (this PR: #749); thanks @ekinnee for filing the issue and for the implementation (PR #599) shipped in the same window.

- **`MNEMOSYNE_JOURNAL_MODE` overrides the SQLite journal mode for store connections.** WAL readback on Linux containers over macOS virtiofs intermittently surfaces as `database disk image is malformed` at every open; deployments on such filesystems can now set `MNEMOSYNE_JOURNAL_MODE=delete` (or any sqlite journal mode) and the sync client (it rides the beam connection) and every connection that sets a journal mode (memory, beam, query cache, veracity consolidator) honors it. Only `wal` persists in the database file; every other mode is per-connection and reverts to SQLite's default (`delete`) on reopen, so each connection re-applies the mode rather than relying on persistence. WAL remains the default; the value is trimmed and lower-cased, unset or blank falls back to `wal`, and non-blank invalid values warn and fall back to `wal`. `memory` and `off` remove disk-backed rollback protection and can corrupt the database after a crash.
- **MCP clients can now retire canonical facts with `mnemosyne_forget_canonical` (#723).** The tool is discoverable and callable by default over MCP; retirement removes the current slot from active recall while preserving it as history.
- **`MNEMOSYNE_MODEL_CACHE_DIR` relocates the local GGUF cache (#708).** The ~656 MB consolidation model was pinned to `~/.hermes/mnemosyne/models`, so the only way off a small home partition was a symlink. The variable is environment-only and read at import, matching `MNEMOSYNE_LLM_REPO` / `MNEMOSYNE_LLM_FILE`; unset or blank keeps the historical path, `~` is expanded, and the value is used for the cached-file lookup, the directory creation and `hf_hub_download` alike. Existing models are never moved, copied or deleted. An explicitly set path is authoritative: when it cannot be created or written to, the local GGUF attempt fails with an error naming both the variable and the selected path rather than silently falling back to the default, which would reinstate the location the user moved away from. The error is logged as well as raised, because the download path degrades to AAAK on any exception and a raised message alone would never reach the user.
- **CLI version reporting (#642).** `mnemosyne --version` / `mnemosyne version` and `mnemosyne-hermes --version` / `mnemosyne-hermes version` report installed distribution versions without initializing Mnemosyne data. `hermes mnemosyne version` now reports both core and Hermes-provider versions.
- **Embedding dimension in doctor diagnostics.** `collect_runtime_diagnostics` (surfaced by `mnemosyne doctor`) now reports the resolved `embeddings_dim` alongside `embeddings_model`, so operators can confirm their `MNEMOSYNE_EMBEDDING_DIM` / model-table resolution without inspecting a traceback. Complements the fail-loud unknown-model resolver (#521); the version bump is deferred to that PR to avoid a duplicate bump.

### Changed

- **Entity extraction no longer stores whole quoted spans as entities (#891).** The `"..."` and `'...'` patterns in `_ENTITY_PATTERNS` captured any quoted span of 2-50 characters, so conversational and roleplay text wrote dialogue into the `mentions` vocabulary: `'Okay,'`, `'Talia pauses.'`, `'the light is fading.'`. Punctuation-bearing values also slip past the stop-word filter, which compares exact strings (`'okay,' != 'okay'`). Measured on one production store: 589 of 1,270 distinct `mentions` values (46%) carried punctuation or spaces, and of 9,831 `references` edges written by proactive linking, 127 connected pairs sharing such a fragment, 103 of them on nothing else, so junk vocabulary became graph topology that recall reads back. Both patterns are removed. A real name inside quotes is unaffected because quotes do not block `\b`, so it still extracts from the capitalized single-word and multi-word patterns; the values that disappear are exactly the spans no other pattern can produce, which are the lowercase and punctuation-bearing ones. A quoted lowercase single word was already dropped by the existing lowercase filter. Existing annotation rows are not cleaned retroactively.

- **Unknown embedding models now fail loud at startup instead of silently assuming 384 dimensions (#518, #521).** `_get_embedding_dim` resolves an explicit `MNEMOSYNE_EMBEDDING_DIM` first (must be a positive integer), then the built-in model table, and raises `ValueError` for an unknown model with no explicit dimension rather than falling back to 384 (bge-small's dimension). A vec0 table is dimensioned at creation, so a silent 384 guess baked the wrong dimension into a fresh database and corrupted vector search for anyone using a model absent from the table (e.g. `mxbai-embed-large` via a custom endpoint). Dimension resolution is centralized in `embeddings._get_embedding_dim`; Beam delegates to it, removing a duplicate resolver that could drift. Embeddings-disabled invocations keep the 384 fallback (the dimension is unused there).

  **Breaking:** pointing `MNEMOSYNE_EMBEDDING_API_URL` at a custom endpoint with a model not in the built-in table now requires `MNEMOSYNE_EMBEDDING_DIM=<N>`, otherwise direct core/MCP-provider startup exits at import with an actionable error (the `mnemosyne-hermes` wrapper catches this and reports the provider unavailable instead of exiting). Blank/empty `MNEMOSYNE_EMBEDDING_DIM` and `MNEMOSYNE_EMBEDDING_MODEL` (common in Docker Compose and `.env` files) are normalized to unset/default rather than treated as explicit invalid values.

  **Upgrade note for stores created under the old silent-384 fallback:** setting the model's true dimension can trigger the existing dimension-mismatch guard. Use the documented reindex/recovery path rather than treating the override as a one-step fix. See [docs/migration-4.0.md](docs/migration-4.0.md).

### Fixed

- **CI no longer hangs silently on an `mcp` release (#871).** `mcp` 2.1.0 deadlocks the streamable-http test teardown, so every matrix job burned its full time budget with zero `FAILED` lines and no commit to blame. The dependency now excludes 2.1.0, and the CI pytest invocations run under `pytest-timeout` (900 s per test, thread method) so a future hang surfaces as a named failure instead of a bare red job.
- **Fact extraction no longer persists truncated or value-free objects (#837).** The rule-based `EpisodicGraph.extract_facts` regexes matched their optional article inside the next word, so `"Alice is already ready"` stored `(Alice, is, lready)`; and nothing guarded the object side, so `"Bob is different"` stored `(Bob, is, different)` and `"Carol uses an extremely reliable editor"` stored `(Carol, uses, extremely)`. Those rows reached `facts`, `graph_edges` and `consolidated_facts` through `remember`, its dedup-update branch, `remember_batch` and `consolidate_to_episodic`, and `fact_recall` surfaced them. Because every such triple shares `(subject, predicate)` with the real facts about that subject, the veracity consolidator also read each one as a contradiction. The article group is now anchored as a whole word, and a new `_is_low_quality_object` rejects a lone lowercase object that is a function word, a transient-state adjective, a filler, or a stance/degree adverb. The guard is a closed word list, not a suffix or shape rule, so names and nouns such as `Sally`, `Italy`, `family` and `developer` cannot be rejected, and a capitalised token (`Rust`, `ComfyUI`) always passes. The patterns capture one object token and still do, so an adjective phrase reaches the guard as its leading modifier and the rule is about that word alone; widening the capture would change every object row and is deliberately not part of this fix. Article-led subjects are rejected when the article opens a common-noun phrase (`"The silence is different"`), and kept when it opens a name (`"The Matrix is a film"`, `"A New Hope has a sequel"`), which the word after the article decides. Existing junk rows are not cleaned retroactively. Restores, in a narrower shape, the fix from #248, whose commits are no longer reachable from `main` (#862); thanks @ekinnee for the independent report.
- **Optional `embeddings` and `all` installs cap `onnxruntime` below 1.29.** This avoids `blkid` stderr on minimal Linux/aarch64 systems.
- **CI stopped being able to verify anything, because an unpinned dependency changed ASGI behaviour (#860).** `tests/test_mcp_streamable_http.py` drives the authenticated SSE GET by hand through the TestClient portal, and its `receive()` never delivered the initial `http.request` message. That violates the ASGI contract, but mcp 2.0.0 answered without waiting for it, so the driver passed. mcp 2.1.0 reads the request body to enforce `max_request_body_size` (SDK #3336), so the handler now blocks before `http.response.start`, the test's wait fails, and `TestClient.__exit__` then blocks forever draining a task group that still holds the wedged ASGI task. The job ran to its six-hour ceiling and reported nothing.

  The dependency is declared `mcp>=2.0.0` with no upper bound, so every run resolves the newest release at install time. mcp 2.1.0 was published on 2026-08-24 at 19:04 UTC; every green run predates it and every hung run follows it. This was not intermittent and not a race: 0 hangs in 22 consecutive runs on 2.0.0, then 4 hangs in 4 runs on 2.1.x, and the same boundary reproduces locally on the unmodified test.

  The server itself is unaffected. Driven over a real socket, mcp 2.0.0 and 2.1.1 both answer the session GET with 200 and `text/event-stream` immediately, so no released version of Mnemosyne is affected and the requirement stays unbounded. Only the hand-written scope could omit a message a real server always sends.

  Three changes: `receive()` now delivers `http.request` before blocking; the ASGI call runs as a portal task whose future is cancelled in a `finally`, so a stuck stream or any failing assertion reports instead of wedging teardown; and `pytest-timeout` caps any single test at 300 seconds, roughly a hundred times the slowest test in the suite, so the next surprise of this shape costs five minutes and names itself instead of costing six silent hours.
- **Native Windows no longer defaults to an install mode that cannot succeed (#857).** `mnemosyne-hermes install` defaulted to `symlink` on every platform, but Windows only permits creating a symbolic link with Developer Mode enabled or an elevated shell. Without one, the install failed with `WinError 1314`, so it worked for some users and not others depending on a privilege nobody thinks to check. On native Windows the default is now persistent wrapper mode, which writes a real plugin directory and needs no privilege; `--mode symlink` still works for anyone who holds it, and nothing changes on Linux, macOS or WSL. An omitted `--mode` resolves to wrapper *before* installation begins rather than switching after a failure, and an explicit `--mode symlink` that hits `WinError 1314` is never switched automatically: it fails with recovery guidance, and the message says so plainly.
- **CLI failure boundaries now emit stable sanitized error codes.**
- **The Core wheel no longer ships `examples/` as an installed top-level package.** #729 excluded the repository-only `integrations` tree from root package discovery, but the same greedy finder still swept `examples`, so installing `mnemosyne-memory` placed a top-level `examples` package into `site-packages`, where it can collide with or shadow any other distribution's `examples` module and a user's own `import examples`. `examples*` is now excluded. The wheel regression suite asserts the entire top-level surface rather than individual leaked directories, so the next repository-root directory cannot reach `site-packages` unnoticed.
- **Portable JSON exports now disclose partial data (#602).** The additive completeness manifest lists populated persisted surfaces omitted entirely and exported sections that omit populated fields; import reports the source artifact's evidence instead of implying a lossless restore. Older export files remain importable with unknown completeness.

- **Hermes plugin tools no longer talk to a second, never-initialized provider.** `register()` constructed one `MnemosyneMemoryProvider` for MemoryManager and a second for PluginManager tool handlers. Desktop/`tool_call` hit the empty instance and returned `Mnemosyne not initialized` while the CLI and `hermes memory status` used the live DB. Both paths now share one instance, and a primary-context tool call lazy-initializes if Hermes never called `initialize()`.
- **Native Windows Hermes venv discovery now finds `Scripts/python.exe` (#809).** Implicit `mnemosyne-hermes install` discovery now supports validated native Windows virtual-environment layouts through launcher siblings, known Hermes roots, the active prefix, and `VIRTUAL_ENV`; explicit `--python` remains authoritative.
- **Windows Hermes symlink installs now explain WinError 1314 recovery (#807).** When Windows denies symbolic-link creation because Developer Mode or the symbolic-link privilege is unavailable, the installer fails closed and prints a command-safe persistent wrapper retry using the resolved Hermes Python; it does not switch modes automatically.
- **CJK-labelled secrets are now detected, flagged and redacted (#806).** A secret introduced by a Chinese/Japanese/Korean label with a fullwidth separator (`数据库密码：s3cr3t_...`) previously bypassed the write classifier, hygiene secret flagging and doctor preview redaction. `detect_secrets` now recognizes a curated set of CJK labels (`密码`/`密钥`/`令牌`/`口令`/`私钥`, `パスワード`/`秘密鍵`/`トークン`, `비밀번호`/`키`) followed by an ASCII or fullwidth separator, with a credential-value predicate that requires a non-CJK, token-like value (8+ chars, at least one ASCII letter or digit) so ordinary Chinese policy prose such as `密码：建议每90天更换一次` is never classified as a secret. The write classifier and hygiene consume this through `detect_secrets`; doctor preview compiles the same canonical patterns for redaction.
- **Hermes wrapper validation timeout is configurable (#804).** `mnemosyne-hermes install --mode wrapper` now accepts `--import-timeout SECONDS` (default: 60) for both selected-Python validation probes, rejects non-positive/non-finite values, and gives a retry command when validation times out.
- **Committed memory invalidations no longer report failure when enhanced-recall cache eviction fails (#594).** The mutation remains successful and the cache error is logged for reconciliation.
- **Hermes providers no longer clear the shared host LLM backend while another primary provider remains active (#551).**
- **The OpenAI-compatible modality retry test is deterministic under load (#798).** Its localhost stub handles one request at a time and records response statuses, so the 401/no-retry contract is checked against the response actually served.
- **Raw dialog no longer starves distilled facts out of the dense recall voice (#696).** Conversational capture (`source='conversation'`, and legacy `honcho_*` imports) is topically identical to the queries that retrieve it, so those rows saturated the nearest-N working-memory vector pool and pushed distilled facts beyond it. An affected fact surfaced with `dense_score=0.0` or did not surface at all. Dialog sources are now excluded from the working-memory dense candidate pool while remaining fully reachable through FTS. #608 widened the candidate neighbourhood, which helps a shallow flood; this is what makes that capacity effective against the flood itself.
- **`hermes mnemosyne export` honors the resolved bank instead of leaking the default (#690).** Explicit and profile-resolved bank selections are now passed to the export-side `Mnemosyne` instance. A selected bank is validated through a read-only SQLite preflight before any Beam, Mnemosyne or output initialization, and a missing, directory-incomplete, table-incomplete or column-incomplete bank is rejected without creating an output artifact or mutating the bank. Validation failures do not expose filesystem paths. Export with no selected bank is unchanged.
- **An uncached local model download now warns before it starts (#703).** The first use of the local GGUF path could spend a long time fetching roughly 656 MB with nothing said. A single warning now names the model file, the HuggingFace repository and the destination cache path, states the size for the built-in default artifact, and explains both the pre-cache option and the AAAK-only opt-out via `MNEMOSYNE_LLM_ENABLED=false`. Default, cache, download, retry and fallback behavior are unchanged, and nothing is written to CLI or MCP stdout.
- **Hermes wrapper installs are no longer clobbered by a forced symlink install.** Wrapper mode is the Docker-safe integration path, and a generic forced symlink install could remove its import bootstrap and leave profile links resolving to the package directory. A wrapper-to-symlink downgrade now requires an explicit request, and wrapper refreshes are validated and staged before they replace a working install. Legacy fresh symlink installs, opted-in profile links and the `upgrade` path are unchanged.
- **Forgetting a working memory now removes its associated gists (#782).** Direct and batch forget paths previously left derived gist rows behind, allowing stale context to survive deletion. Cleanup is atomic and preserves the existing session authorization boundary.
- **The `mnemosyne-stats.py` test suite now runs against a hermetic pytest-owned database instead of the developer's real one (#783).** `tests/test_mnemosyne_stats.py` shelled out to the stats CLI without an environment override, so on a developer machine it resolved the ambient `MNEMOSYNE_DATA_DIR` / `HERMES_HOME` / `HOME`, read and reported on the real Mnemosyne database, and wrote snapshots into real home directories; `test_rapid_fire` flaked when a live database had concurrent writers, and the tests exposed the developer's stored memories. An autouse module fixture now points the subprocess at a seeded `tmp_path` bank plus tmp home/wiki dirs and re-points the assertion helpers at the same locations, making the tests hermetic and ordering-independent.
- **Automatic working-memory consolidation no longer calls `sleep_all_sessions()`, and `auto_sleep_enabled: false` is honored (#771).** The Hermes provider's `_maybe_auto_sleep()` previously selected `sleep_all_sessions()` by capability probing, which could sweep unrelated sessions. Its worker now calls `sleep()` on the `BeamMemory` instance bound to the triggering session. The provider also reads the core `auto_sleep_enabled` config key (via the Mnemosyne config bridge, matching the root provider) in addition to the Hermes `auto_sleep` key, so `mnemosyne config set auto_sleep_enabled false` disables automatic consolidation.
- **The Core wheel no longer ships the repository-only Hermes provider source/test tree (#729).** The root setuptools package finder did not exclude the nested `integrations/` tree, so `mnemosyne-memory` wheels bundled the standalone Hermes provider and its tests even though it is published separately. `integrations*` is now excluded from Core package discovery, while the standalone `mnemosyne-hermes` package remains separate; regression tests build both wheels and assert their contents.
- **`valid_until` timestamps are now aware UTC everywhere (#525).** `invalidate()` wrote a naive local wall-clock ISO value while SQLite-side surfaces (doctor, repair, MCP validate) compare against UTC `julianday('now')` / `CURRENT_TIMESTAMP`, so expiry checks disagreed by the host's UTC offset and shifted with DST. The write path and every Python-side `valid_until > ?` comparison now use `datetime.now(timezone.utc)`. All read filters compare stored values chronologically (`julianday`) rather than by ISO string ordering, so offset-bearing and space-separated legacy rows are judged by their actual instant; offset-bearing values are canonicalized to UTC at every supported persistence boundary (`remember`, `consolidate_to_episodic`, `import_from_dict`, Hindsight import, sync-apply). Legacy rows written without an offset are interpreted as UTC (the same interpretation SQLite already applies), and only an exact `YYYY-MM-DD` `valid_until` input keeps pass-through semantics (any other parseable form, including lowercase `t` separators, is normalized; unparseable values pass through unchanged).
- **SHMR clustering no longer crashes with a dimension mismatch (#762).** `harmonize()`'s `_embed()` passed a `str` to `embeddings.embed()`, which expects `List[str]`; the string was iterated per character, so each embedding's dimension scaled with the text length and `_cluster_by_similarity()` failed whenever two candidates had different lengths. `_embed()` now wraps the text in a list, returns a fixed-dimension vector, and degrades to zeros when embeddings are unavailable. The `harmonize()` facts query also drops a filter on a `status` column that the `facts` table does not have, so the candidate step no longer raises `OperationalError`.
- **MCP `tools/list` no longer advertises tools that cannot be called (#728).** Eight schemas (`mnemosyne_triple_end`, `mnemosyne_sync_push`/`pull`/`status`, `mnemosyne_persona_promote`/`demote`/`list`/`reinforce`) were published over MCP without a dispatch handler, so every `tools/call` for them failed with `Unknown tool`. The advertised surface is now filtered to the handler registry, and a parity test asserts the advertised set matches it exactly.
- **The Hermes provider's failure diagnostic missed two virtualenvs over one base interpreter (#709).** `register_memory_provider()` compared `_hp.resolve()` against `Path(sys.executable).resolve()`. A venv's `bin/python` is a symlink to the interpreter it was created from, so resolving collapsed two distinct environments onto that one binary and skipped the diagnostic in exactly the case it exists to report; on macOS it also rewrote `/tmp` to `/private/tmp`. It now uses the `_hermes_python_mismatch()` helper added for #736, which compares environment roots, so the provider diagnostic and `mnemosyne-hermes status` answer the question the same way. That helper now normalises both sides with `os.path.normpath` before deriving the root: without it a path spelled `<venv>/bin/../bin/python` yielded `<venv>/bin/..`, which names `<venv>` but does not compare equal to it, so one environment was reported as two. Normalising is lexical and does not follow symlinks, so venv identity is preserved.
- **Recall no longer silently misses leading-hyphen and symbolic query fragments (#744).** Queries containing leading-hyphen fragments such as ``rm -rf`` or ``--force`` could produce invalid FTS5 queries or no usable FTS terms (a token must start with a word character, and FTS5 treats a leading ``-`` as the NOT / column-exclusion operator), and symbolic code names such as ``C++`` or ``C#`` were dropped by the three-character meaningful-token gate, so recall returned an empty list without an error. Leading-hyphen fragments are now split into their components and matched through the FTS5 and lexical paths (``-v``-style single-character flags are included while stopwords and digits stay excluded); literal flag queries reject bare-component-only candidates regardless of configurable scoring weights. Symbolic code names are admitted as exact lexical tokens on both sides, so ``C++`` recalls memories containing ``C++`` without admitting ``c``-token distractors.
- **`mnemosyne-hermes status` now reports the real interpreter mismatch (#736).** The warning compared interpreter paths but claimed a Python version mismatch and printed a bare version number instead of a runnable fix; it now compares the Hermes and installer environments and emits a shell-quoted `→ Run: <python> -m pip install -U 'mnemosyne-hermes[all]'` command.
- **A query embedding whose dimension disagreed with the store's `vec0` tables crashed `recall()` (#753, fixed in #754).** `_vec_search` (the episodic KNN over `vec_episodes`) executed its MATCH without exception handling, so `sqlite3.OperationalError: Dimension mismatch for query vector` propagated straight out of `recall()` and took down the calling process — while the write path (`_wm_vec_upsert`) logged and dropped the mismatched vector, and the working-memory KNN (`_wm_vec_search_sqlite`) already returned `[]`. Most often hit when a process resolves a different `MNEMOSYNE_EMBEDDING_DIM` than the one that dimensioned the store. `_vec_search` now degrades the same way: vector recall is disabled for that call, `recall()` falls back to its other voices, and the log carries actionable guidance: the existing `_dim_mismatch_message()` self-heal steps when the configured dimension disagrees with the store, or a pointer at the embedding endpoint (explicitly not a reindex) when the endpoint serves a differently-dimensioned query vector while store and configuration agree.
- **MCP SSE authentication rejects malformed non-ASCII bearer tokens with `401` instead of returning a server error (#739).**
- **Thread-local SQLite connection churn no longer accumulates file descriptors.** Connection creation now periodically runs process-wide cyclic-garbage collection, reclaiming unreachable SQLite handles without closing connections still referenced by live objects. Because collection scans all unreachable cycles, its occasional tail latency depends on process heap size.
- **API embedding failures now leave a redacted diagnostic trace (#735).** Final HTTP, network, and invalid-response failures still degrade to keyword-only retrieval, but now log the endpoint and safe error class or status without request content, API keys, URL userinfo, query strings, or fragments.
- **Hermes tool discovery now honors `memory.mnemosyne.tools` (#725).** Tools outside the configured allowlist are no longer advertised through Hermes provider schemas before provider initialization.
- **Truncated LLM reasoning traces no longer reach memory persistence (#734).** Malformed or unbalanced `<think>` output is rejected before fact extraction, model-refresh parsing, or sleep consolidation; sleep falls back to AAAK rather than persisting a partial LLM summary.
- **Episodic degradation preserves atomic vector refreshes (#691).** Refreshing sqlite-vec embeddings no longer commits inside a degradation savepoint, so a failed refresh rolls back its content and vector update together.
- **Hermes interpreter discovery accepted an unvalidated candidate (follow-up to #618/#620).** #620 taught `_find_hermes_python()` to follow a shell-wrapper launcher through its `exec` target, which fixed the reported case. Two paths still returned the wrong interpreter: a launcher that is neither a symlink nor an `exec` wrapper resolves to itself, so a sibling `python` in a shim directory such as `~/.local/bin` (commonly a Homebrew or system symlink) was still returned as "Hermes' Python"; and the known-install-root branches returned `candidate.resolve()`, which follows a venv's `bin/python` symlink to its base interpreter and discards the venv. An implicitly discovered candidate is now returned only when its directory is a real virtualenv (`pyvenv.cfg`) and its interpreter is executable, from the launcher, the install roots, `sys.prefix` and `VIRTUAL_ENV` alike, and no branch resolves the interpreter symlink. An explicit non-empty `--python` stays authoritative and deliberately bypasses that validation; an empty one is rejected rather than falling through to discovery. `--python` is now authoritative and reaches symlink-mode discovery and `--dry-run`, where it previously affected only wrapper installs. **Behavior change:** a symlink install fails closed when no validated interpreter is found, naming `--python`, where it previously proceeded; `--no-bootstrap` continues without dependency validation, since it already installs nothing into Hermes' environment.
- **Windows Git Bash/MSYS backup destinations no longer silently land on a drive-relative path (#659).** `mnemosyne backup /c/...` now writes to the intended `C:/...` destination. Ambiguous POSIX-rooted destinations are rejected before backup creation instead of reporting success for a different location; native Windows, UNC, and relative paths remain supported.
- **The built wheel now ships `hermes_memory_provider/plugin.yaml` (#656).** `pyproject.toml` declared no `package-data` for `hermes_memory_provider`, so a normal `pip install mnemosyne-memory` (unlike an editable install) omitted the manifest Hermes' plugin loader requires, leaving the documented `hermes_memory_provider` symlink install pointed at a directory with no `plugin.yaml`.
- **Invalidation replacement links now require an accessible memory (#676).** `mnemosyne_invalidate` rejects an unknown or out-of-scope non-empty `replacement_id` before changing the target, so rejected replacements do not create links at invalidation time.
- **`bge-m3` embedding alias resolves its 1024-dimensional vectors (#666).** The unqualified model name now resolves identically to `BAAI/bge-m3`, avoiding an unknown-model startup error when no explicit dimension override is set.
- **MCP invalidate now reports scope-safe failure (#660).** `mnemosyne_invalidate` returns `memory_not_found` instead of claiming success when its target is outside the current scope or cannot be mutated, preserving scope isolation.
- **Recall diagnostics were dead under `MNEMOSYNE_POLYPHONIC_RECALL=1`.** The polyphonic branch of `BeamMemory.recall()` returned before the C4 recording block, so every recall that ran through the polyphonic engine (vector/graph/fact/temporal voices) never incremented `mnemosyne_recall_diagnostics` counters — the tool reported `calls: 0` under the flag that production deployments use. The polyphonic branch now records tier hits and call counts itself, mapping engine voices to the existing diagnostic tiers (`vector`→`wm_vec`, `graph`→`em_vec`, `fact`→`em_fts`). Recording is read-only signal and never alters recall behavior. Documented in `docs/benchmarking.md`.
- **`fallback_rate` was dead under `MNEMOSYNE_POLYPHONIC_RECALL=1`.** The polyphonic diagnostics block (added in #668) recorded tier hits and call counts but never `record_fallback_used()`, so `mnemosyne_recall_diagnostics` reported `wm_fallback_rate`/`em_fallback_rate` as `0` on every polyphonic recall — including when the vector voice degraded from the sqlite-vec fast path to a numpy full-scan (sqlite-vec absent, failing, or its top-K ANN hits all dropped in the superseded/valid_until JOIN). The engine now exposes a per-call degraded-path flag and the polyphonic block records it as `em_fallback_used`. `wm_fallback_rate` stays `0` by design: the polyphonic engine has no substring-scoring tier for working memory. Recording is read-only signal and never alters recall behavior. Documented in `docs/benchmarking.md`.

- **Persisted Enhanced Recall cache was stale after fresh `remember()` writes (#556).** `BeamMemory.remember()` now uses the established persisted-cache invalidation helper after successful new-memory and dedup-update writes, so a fresh writer evicts results warmed by another instance before the next fresh enhanced-recall request. Live peer in-memory coherence remains tracked separately in #552.
- **Hermes wrapper runtime compatibility guard (#625).** The legacy provider and newly registered persistent wrappers reject selected Mnemosyne site-packages whose virtualenv targets a different Python major/minor, or has an unreadable version, before activation/import; the error directs operators to recreate the Mnemosyne environment using Hermes' Python. Existing persistent wrapper artifacts must be force-refreshed or re-registered from a compatible Hermes-Python venv to receive this guard.
- **Vector rebuild failures are now reported explicitly (#603).** Reindexing fails on incomplete embedding batches or derived-vector write failures. `vec_working` repair also fails when its final coverage check remains incomplete. `mnemosyne diagnose --repair-vec-working` returns a non-zero exit code when a requested repair fails.
- **Persona token-cap truncation (#621).** `render_persona_markdown` now skips oversized topic sections and continues evaluating later sections, so smaller persona sections that still fit within the approximate token cap are retained.
- **Silent hermes_plugin import failure in legacy provider (#649).** `hermes_memory_provider/__init__.py` `register()` replaced bare `except Exception: pass` with `logger.warning(...)` so that failures to import the legacy `hermes_plugin/` directory are visible in logs. Previously, a missing `__init__.py` (or stale `.pyc` files) silently prevented hook registration (pre_llm_call memory injection, tools) with no diagnostic output.
- **`degrade_batch` now honors `config.yaml` at the BEAM consumer (#482).** Episodic degradation resolves `degrade_batch` as `config.yaml > MNEMOSYNE_DEGRADE_BATCH > 100` once per complete degradation pass. Reloaded YAML applies to the next pass without changing the candidate limits of a running pass.
- **BEAM recall weights now honor `config.yaml` at runtime (#482).** `vec_weight`, `fts_weight`, and `importance_weight` now resolve as `config.yaml > MNEMOSYNE_*_WEIGHT > defaults` in direct and Hermes-provider recall paths. Reloaded weights apply to the next request, and enhanced recall cache entries are isolated by the effective weight snapshot.
- **Packaged Hermes plugin manifests match the released package version (#588).**
- **Hermes provider discovery and registration work through the provider `register()` bridge (#565).**
- **Invalidating a nonexistent memory explicitly reports `memory_not_found` (#542).**
- **sqlite-vec candidate retrieval is widened before working-memory filters (#608).** Matching results are no longer excluded prematurely.
- **File-import dry run.** File-import `--dry-run` now passes through the core, MCP, both Hermes providers, and CLI surfaces to clone-based validation without changing the active database or audit data. Dry-run responses report `"status": "dry_run"` so clients cannot mistake simulated import statistics for a completed import.
- **Repaired `hygiene audit --json` → `hygiene clean` workflow (#606).** `hygiene clean` now unwraps the audit envelope produced by `hygiene audit --json` and validates each candidate before cleanup. Raw candidate arrays remain supported, and candidates with persisted `importance` values outside `[0, 1]` are accepted so the audit-to-clean pipeline completes without manual editing.
- **Model-refresh confidence: NaN cleared every gate and legacy text crashed sleep mid-batch.** JSON round-trips NaN and Infinity, and `parse_model_update_proposals` clamped NaN to 1.0 (`min` and `max` keep their first argument when a NaN comparison is False), so a NaN-confidence proposal became a top-importance memory; the auto-apply gate's `confidence < minimum` check is also False for NaN, so the same proposal reached the canonical store regardless of threshold. Separately, `apply_model_refresh_proposal` and sleep()'s proposal-remember call site converted stored confidence with a bare `float()`, so a legacy bank's text value (for example `"high"`) raised ValueError. The sleep call site runs after the claim commit, so that raise stranded the group's `consolidation_claimed_at` and orphaned every later group's claimed rows. Non-numeric and non-finite confidence now degrades per site: skipped at parse, 0.0 at the auto-apply gate, 0.5 at apply and at proposal importance. Finite values outside [0.0, 1.0] clamp to the domain bound on every path before auto-apply threshold checks and canonical storage, so a persisted 2.0 cannot remain unbounded. Hardening split out of #546 per review.
- **Russian and Spanish MEMORIA patterns contained literal backslash escapes (#560).** The `ru` instruction pattern was written with `\\\\s+` and `[^.,;!?\\\\n]` inside a raw string, so it required a literal backslash in the text and Russian instruction extraction matched nothing at all. Six `es` patterns (`negation`, `decision`, `entity`, `sequence`, `instruction`, `preference`) carried the same doubled `\\\\n`, which turned the newline exclusion into an exclusion of the letter `n` and truncated every capture at the first `n`. Both are now single-escaped, and the locale guard test in `tests/test_memoria_instruction_boundaries.py` rejects any future doubled escape.
- **Hermes session switches left Mnemosyne memory bound to the previous session (#601).**
  The standalone `mnemosyne-hermes` provider now rebinds its `BeamMemory` session when Hermes
  rotates the agent session through `/new`, `/resume`, `/branch`, undo, or context
  compression, so subsequent writes, reads, and tools use the active session.

## [3.15.1] - 2026-07-30

### Fixed

- **`test_stored_offset_bearing_valid_until_chronologically_filtered` failed for two hours every day (#525 follow-up).** The test stores a space-separated naive `valid_until` two hours ahead and asserts a lexical filter drops it, since a space separator sorts before a `T` separator. That only holds while the date components match. Between 22:00 and 00:00 UTC the value rolled into tomorrow, sorted after aware-UTC now, and the sanity assertion failed on an unchanged tree. The naive value is now clamped to the current UTC date, behind a 30-second validity margin so a clamp near midnight cannot leave the row expiring mid-test, so the trap it sets actually holds, verified across all 1440 minutes of the day. Behavior under test is unchanged; this was a defect in the fixture, not in `valid_until` handling.
- **MEMORIA instruction extraction inverted "whenever X" into "never X" (#507).** The instruction pattern was not word-boundary anchored, so `never` matched inside `whenever` and the extractor stored the opposite of what the user said — on a production bank, "Good - whenever needed we can use it." was recorded as the instruction "never needed we can use it". All five locale patterns (en/de/ru/it/es) are now anchored with a leading `\b`. Genuine instructions are unaffected, including those preceded by another word or punctuation ("Note: never push to main", "wherever you go, always run the tests"). Reported by @Axmr1 from a 61-row production audit; original diagnosis and fix approach from @Sanjays2402 (#508) and @Souptik96 (#549).
- **`mnemosyne_recall` crashed on its own schema default (#555).** The tool schema declared `query_time` with `"default": ""`, but `_parse_query_time` mapped only `None` to "now" — a blank string fell through to the ISO parser and raised `Invalid query_time format: ''`. Any MCP harness that sends declared defaults could not call `mnemosyne_recall` at all. Blank and whitespace-only values are now treated as unset, the MCP handler normalizes `""` to `None` (matching the `or None` idiom already used for `valid_until` and `as_of`), and the schema no longer advertises a default that means "omitted". Thanks @dalkommatt for the report and the diagnosis.
- **Enhanced Recall served invalidated rows until TTL expiry (#550, #554).** `BeamMemory.invalidate()` now clears the query cache after a successful update, including the persisted `query_cache.db` when the instance has no in-memory cache of its own. Missing or unauthorized IDs leave the cache untouched. Remaining gaps are tracked in #552 (live peer coherence) and #553 (`forget_working`).
- **Catastrophic regex backtracking in version-string extraction (#544).** The pattern used by `extract_and_store_facts` could be driven into exponential backtracking by Title-Case input, hanging every `remember()` and import on attacker- or user-supplied content. The separator is now `\s+`, which makes each whitespace-delimited word consumable exactly one way. Behavioral equivalence was verified across a 200,000-string fuzz with zero differences.

## [3.15.0] - 2026-07-20

> Never published to PyPI. No `v3.15.0` tag was cut, so this section
> documents work that first reached users in 3.15.1. Kept for history
> rather than folded, so the individual changes stay attributable.

### Fixed

- **Memory browser startup and bank resolution (#532).** The browser now renders its CSS template safely, resolves the default and named-bank databases using the canonical paths, and opens databases read-only so a missing path cannot create an empty database.
- **Trim-before-embedding race (#491).** Working-memory embedding storage now atomically checks that its parent row still exists. If trimming or concurrent deletion removes the parent before the fallback insert executes, both fallback and `vec_working` writes become a clean no-op instead of logging an embedding-storage failure.
- **Jina v2 base embedding models silently fell back to 384 dimensions.** The
  `jinaai/jina-embeddings-v2-base-{es,en,de,zh,code}` models output 768-dim
  vectors, but were absent from `_get_embedding_dim`'s table and so resolved to
  the 384 unknown-model fallback — a silent dimension mismatch that corrupts
  vector similarity search for anyone using these popular models (notably the
  `-es` Spanish/English bilingual model). Added explicit 768-dim entries plus a
  regression test in `tests/test_embeddings_multilingual.py`.

## [3.14.0] - 2026-07-17

### Added

- **Write-approval gate for Mnemosyne memory writes (#456).** When `memory.write_approval: true` is set in Hermes config.yaml, `mnemosyne_remember` and `mnemosyne_batch` stage writes to `pending/memory/<id>.json` instead of committing directly. The new `mnemosyne_apply_pending` tool replays approved records through the BEAM write path. Both standalone and bundled Hermes providers are supported.
- **`mnemosyne_forget_canonical` tool (#435).** Completes the CRUD surface for canonical facts: remember, recall, and now forget (retire) a slot. Stamps `valid_until` on the current row, preserving history. Nothing is deleted.
- **Safe doctor and selected repair workflow.** The `mnemosyne doctor` command now supports a `--safe` mode that performs read-only diagnostics, plus a `--repair` mode that can fix selected issues (orphaned references, WAL cleanup, vec_working migration gaps).
- **Query/document embedding prompt prefix env vars (#401).** `MNEMOSYNE_EMBEDDING_QUERY_PREFIX` and `MNEMOSYNE_EMBEDDING_DOCUMENT_PREFIX` allow customizing embedding prompt prefixes for BGE-style models.
- **Shared-surface sync hardening (#442).** Blind-relay security: sync event payloads are sanitized before public broadcast, cross-model security findings are closed, and the sync server binds to a dedicated shared surface.
- **CLA requirement for contributors.** All PRs now require a signed Contributor License Agreement. Branch protection enforces the `license/cla` check on main.

### Fixed

- **Config set crash (#481).** `mnemosyne config set` no longer crashes with AttributeError after writing the value. The `REQUIRES_RESTART` check now imports from the module-level set.
- **Expired Discord links (#479).** All Discord invite links updated to `discord.gg/nousresearch`.
- **onnxruntime thread affinity spam in LXC containers (#453).** `TextEmbedding` now receives an explicit `threads=` parameter, preventing `pthread_setaffinity_np` EINVAL errors in unprivileged containers. Override with `MNEMOSYNE_EMBEDDING_THREADS` env var.
- **Polyphonic recall collapse (#389).** `_estimate_similarity` now uses word-level content Jaccard instead of voice-name Jaccard, preventing MMR diversity reranking from collapsing to a single result when one voice dominates.
- **Content mutation (#387).** Temporal annotations (`[DATES:]`, `[DURATIONS:]`) are now stored in metadata instead of appended to the content field, preserving byte-identical content for verbatim-reproduction workflows.
- **Polyphonic content hydration (#471).** `PolyphonicResult` now carries a `content` attribute, hydrated from the database before diversity ranking, so the content-Jaccard scorer works correctly.
- **Main CI contracts restored (#480).** Post-merge CI test alignment fixed, restoring green CI on main.
- **Unsafe connection lifecycle change reverted (#477, #382).** The `__del__`-based WAL cleanup was reverted. Safe multi-DB lifecycle support needs a defined lease contract, not a destructor.
- **Embedding API retries (#475, #478).** HTTP 429/5xx and transient network failures now receive bounded exponential-backoff retries with jitter. Permanent 4xx errors still fail fast.
- **Hermes provider diagnostics.** `mnemosyne diagnose` now resolves logs under `HERMES_HOME` when set, and the doctor tool respects the resolved bank.
- **Local LLM SSE errors (#447).** Streaming set to false to prevent SSE errors on certain local LLM backends.
- **Legacy memory_embeddings FK migration (#452).** Databases created by old DDL now migrate their foreign keys correctly on init.
- **Hyphenated recall query expansion.** Compound query tokens are now expanded for better matching.
- **Hermes provider defaults after config bridge.** New auto-seeded configs preserve user-only autosave and skip noisy contexts.
- **Code audit cleanup (#460).** Callable import, duplicate import, and F-rules linting fixed.

### Security

- Sync event payloads are sanitized before public broadcast.
- Cross-model security findings in sync layer closed.
- Branch protection enforces `license/cla` check on main (no admin bypass).
  `FOREIGN KEY (memory_id) REFERENCES memories(id)` constraint on
  `memory_embeddings`. The `memories` table is unused — working_memory
  ids are stored instead. When `PRAGMA foreign_keys=ON` was enabled
  (#408), every embedding insert silently failed with
  `IntegrityError: FOREIGN KEY constraint failed`. This release adds
  an idempotent migration that rebuilds the table without the FK
  and removes the FK from the `memory.py` DDL so fresh
  databases are clean.
- **`mcp_tools.py` validate(delete) path now cascades to child rows.**
  The bare `DELETE FROM working_memory` previously left orphaned
  `memory_embeddings`, `annotations`, and `vec_working` rows behind.
  The path now deletes all dependent rows before removing the parent,
  with guarded vec_working handling for sqlite-vec-unavailable environments.

## [3.12.2] - 2026-07-11

### Fixed

- **Config reload now bridges to the Hermes provider.** `mnemosyne config
  set` and `mnemosyne config reload` previously wrote to the Mnemosyne
  config.yaml but the Hermes provider only read from the Hermes config.yaml
  (`memory.mnemosyne.<key>`). The two files never connected, so config
  changes appeared to do nothing. Now the provider falls back to the
  Mnemosyne config singleton when the Hermes config has no value, and
  `MnemosyneConfig.get()` auto-reloads on file mtime changes so `config set`
  takes effect immediately without an explicit reload.

- **Config.yaml auto-seed on all entry points.** The auto-seed now fires on
  `Mnemosyne()` and `BeamMemory()` init, not just explicit config imports.
  Idempotent — checks file existence first.

- **Test isolation for config auto-seed.** Config profile tests now create
  an empty config.yaml before init so the auto-seed doesn't override test
  env vars with defaults.

## [3.12.1] - 2026-07-11

### Added

- **Config.yaml auto-seed on first access.** Mnemosyne now creates a
  `config.yaml` at the standard location with all 106 known keys and their
  default values. The file is created automatically on first access — no
  manual setup needed. For each key, if the corresponding env var is set,
  its value is used instead of the default, ensuring existing env var
  configurations are never silently overridden. Hot-reload with
  `mnemosyne config reload`. Precedence unchanged: config.yaml > env vars
  > hardcoded defaults.

### Fixed

- **Config.yaml auto-seed respects existing env vars.** The initial
  implementation wrote all defaults blindly, which would silently override
  any `MNEMOSYNE_*` env vars the user had set (since config.yaml takes
  precedence over env vars). Now each key checks for an active env var
  before writing. Type coercion is applied: env var strings are parsed as
  bool/int/float to match the default type.

## [3.12.0] - 2026-07-11

### Added

- **Config.yaml system with profiles, hot-reload, and write filters.**
  Mnemosyne now supports profile-based configuration, hot-reloading config
  changes without restart, and write filters for fine-grained control over
  what gets stored. (#431, #433)

- **`MNEMOSYNE_CROSS_SESSION` env var for cross-session recall.**
  When set, `recall()` searches across all sessions instead of only the
  current one. (#371)

- **Atomic `mnemosyne_batch` tool.** Batch multiple memory operations
  (remember, update, forget, invalidate) in a single atomic transaction
  via the Hermes provider. (#400)

- **Sync turn diagnostics.** `sync_turn` now exposes diagnostic information
  for debugging sync pipeline issues. (#115162b)

- **Read-only doctor hygiene signals.** The doctor diagnostic tool now
  reports hygiene signals (foreign key gaps, orphaned rows, stale
  connections) without requiring write access. (#71e013d)

- **Orphan diagnostics to doctor.** Doctor now detects orphaned memory
  rows with no corresponding FTS5 or embedding entries. (#417)

- **CLI bank selection, bank list, and schema migration.**
  `mnemosyne store` and other CLI commands now honor `MNEMOSYNE_BANK`.
  New `mnemosyne bank list` command for multi-tenant visibility.
  New `mnemosyne migrate` command for 3.11.0-era banks. (#404)

- **Hermes memory providers skill v2.0.0.** Bundled skill for the Hermes
  ecosystem documenting all memory providers. (#4ee3a58)

- **Zero and Pi agent integrations.** Mnemosyne now integrates with the Zero
  agent framework and Pi agent. (#418, #c0a7176)

- **Layered agent memory roadmap.** Architecture document defining the
  L0-L4 memory layer model for AI agents. (#96e6978)

### Fixed

- **`MNEMOSYNE_ENHANCED_RECALL=1` now routes through the full enhanced
  recall pipeline.** `Mnemosyne.recall()` always called `beam.recall()`
  directly, bypassing `beam.recall_enhanced()` entirely. The flag had zero
  effect on production call paths. Now routes to `recall_enhanced()` when
  the flag is set. (#436, reported by @ValentinSergief with full RCA)

- **SSE transport Route handlers no longer crash Starlette.** Route
  handlers returning `None` caused Starlette crashes in SSE transport
  mode. (#383)

- **Diagnostics fallback DB path now respects `HERMES_HOME`.**
  The diagnostics tool used a hardcoded fallback path instead of
  resolving from `HERMES_HOME`. (#384)

- **Veracity forwarding through `Mnemosyne.remember()`.** The module-level
  `remember()` function now forwards the veracity argument to the
  underlying beam, fixing the MCP remember handler silently dropping
  veracity. (#399, #386)

- **Namespace collision: `tools/` renamed to `_benchmarks/`.** The
  `tools/` directory collided with `hermes-agent` tool discovery.
  Renamed to avoid the conflict. (#9ca278a)

- **Profile bank resolution in standalone CLI.** CLI commands loaded
  standalone now correctly resolve the profile bank. (#6725b80)

- **ASGI middleware replaced with pure-ASGI bearer auth.** Replaced
  `BaseHTTPMiddleware` with a pure-ASGI approach for bearer auth in
  MCP SSE transport, fixing Mount compatibility. (#be8c865)

- **Current-state recall ranking.** Fixed a bug where recall ranking
  used stale scores instead of current-state values. (#416)

- **Bank name validation before path operations.** Bank names are now
  validated before any filesystem path operations, preventing directory
  traversal and invalid characters. (#415)

- **SQLite write lock released across consolidation LLM calls.**
  `BeamMemory` no longer holds the SQLite write lock while waiting for
  LLM consolidation responses, preventing WAL checkpoint blocking.
  (#432, reported by @kirocop in #382)

- **Recall-touch transaction rolled back on failure.** The recall-touch
  UPDATE now properly rolls back the transaction on failure instead of
  leaving a stale write lock. (#f418044)

- **`PRAGMA foreign_keys=ON` in both connection factories.**
  Foreign key enforcement is now enabled in both the main and the
  thread-local connection factories. (#408, reported by @Iman-Sharif)

- **Hygiene audit CLI and table handling hardened.** The doctor CLI
  now handles edge cases in table detection and reporting. (#f072e1a)

- **Host LLM timeout now configurable.** Added `MNEMOSYNE_LLM_TIMEOUT`
  env var (default 60s) for remote LLM consolidation and extraction
  calls. (#d290193)

- **Hermes provider fixes (6 commits):**
  - Auto-sleep default enabled across both provider surfaces (#429)
  - Bundled memory override skill installer (#424)
  - Cross-session recall and CLI default scope (#422)
  - Pip sync adapter parity with core (#419)
  - L3 persona prompt parity restored (#ed05503)
  - `HERMES_HOME` leak in CLI bank test (#6664e81)

### Changed

- **Default prompt context excludes consolidated working-memory rows.**
  `BeamMemory.get_context()` no longer includes rows where
  `consolidated_at IS NOT NULL`. Set `MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED=1`
  to restore legacy behavior. (#427)

### Documentation

- Installation steps revised for Hermes users (#414, @bruvv)
- Pi agent integration docs added (#c0a7176)
- Hermes Tweet compatibility table (#e032008, @Burak Bayır)
- `.coderabbit.yaml` with grouped reviews and architectural rigor (#da4832a)

### Thanks

@dplush (Denis H) — 11 commits: sync diagnostics, recall ranking, bank validation,
orphan detection, auto-sleep, cross-session recall, batch tool, L3 persona,
hygiene audit, veracity forwarding, pip sync parity

@codxt — 3 commits: CLI bank selection + migration, ASGI middleware fix,
layered memory roadmap

@Milgauss — 2 commits: SQLite write lock fix, recall-touch rollback

@TurgutKural — 2 commits: profile bank resolution, host LLM timeout

@ValentinSergief — thorough ENHANCED_RECALL RCA with file+line references

@PlainWu, @ClaytonChew, @bruvv, @justanotherAIcontributor, @BurakBayır,
@Iman-Sharif, @webtecnica — bug reports, fixes, and docs improvements

## [3.11.0] - 2026-06-30

### Added

- **Automated sleep model refresh.** During `sleep()`, Mnemosyne now asks
  the LLM for structured candidate updates to canonical model slots (user
  model, workflow model, project model). Validates the LLM response against
  the expected schema, generates proposals with confidence scores, and
  auto-applies or auto-rejects them by policy. New `mnemosyne_model_refresh`
  diagnostic tool for inspecting proposal outcomes.

- **Recall diagnostics and task progress tools.** `mnemosyne_recall_diagnostics`
  exposes per-row recall scoring breakdowns (weights, scores, signal
  contributions) for debugging hybrid ranking. `mnemosyne_task_progress`
  tracks multi-step task state across sessions with create/update/get/list
  operations.

- **`MNEMOSYNE_LLM_TIMEOUT` env var.** Configurable HTTP timeout for remote
  LLM consolidation and extraction calls (default 60s). Useful for deployments
  routing through local proxies or models with long generation times. (#375)

- **Tool whitelist allowlist.** Hermes Mnemosyne providers can now restrict
  exposed tools with the optional `memory.mnemosyne.tools` config key while
  preserving memory context and prefetch behavior. Unknown names raise a
  clear startup error so typos don't silently lose tools.

- **Hermes wrapper install mode for read-only / Docker deployments.**
  `mnemosyne-hermes install --mode wrapper --python <path>` creates a stable
  `$HERMES_HOME/plugins/mnemosyne/` shim that imports from the selected Python
  environment instead of symlinking into a rebuildable Hermes venv.
  `mnemosyne-hermes status` reports wrapper mode, target interpreter, import
  health, and stale/broken targets.

### Changed

- **Tool schemas consolidated to single source of truth.** All 37+ tool schema
  definitions moved from duplicate copies in `hermes_memory_provider/__init__.py`
  and `integrations/hermes/src/mnemosyne_hermes/tools.py` to a shared
  `mnemosyne/tool_schemas.py` module. Both provider copies import from the
  canonical source, ensuring tool definitions stay in sync.

- **Hermes sync role default now saves user turns only.** The `sync_roles`
  default changed from `["user", "assistant"]` to `["user"]` so automatic
  turn autosave avoids assistant transcript noise. Set
  `memory.mnemosyne.sync_roles: ["user", "assistant"]` in `config.yaml` to
  restore the prior behavior.

### Fixed

- **`mnemosyne backup` now works with sqlite-vec databases.** `create_backup()`
  loads the sqlite-vec extension on backup connections so `iterdump()` and
  `Connection.backup()` can serialize vec0 virtual tables. Previously raised
  `OperationalError: no such module: vec0` on all 3.10.x installs.

- **Named Hermes profiles now get the plugin link** (issue #365). Both
  `mnemosyne-install` and `mnemosyne-hermes install` now scan
  `~/.hermes/profiles/*/config.yaml` for `memory.provider: mnemosyne`, creating
  or removing the plugin symlink in each matching profile's `plugins/` directory.
  Previously the link was only created under the default `~/.hermes/`.

- **Host LLM backend registration in skip-context sessions.**
  `register_hermes_host_llm()` was at the end of `initialize()`, after the
  skip-context early return. Cron, subagent, and background sessions never
  reached it, so `mnemosyne_sleep` silently fell back to AAAK. Registration
  now fires before the skip-context check; `shutdown()` only unregisters when
  the session is not in a skip context (#368, supersedes #361).

- **`HERMES_HOME` respected for fastembed cache default.** The default ONNX
  model cache path resolves to `<HERMES_HOME>/cache/fastembed` (falling back
  to `~/.hermes/cache/fastembed`). `MNEMOSYNE_FASTEMBED_CACHE_DIR` still
  overrides.

- **`mnemosyne` CLI bank-aware under `profile_isolation`.** CLI commands
  (`stats`, `inspect`, `sleep`, `export`) now resolve the active profile bank
  instead of always reading the default bank, which reported empty state when
  the profile bank held the data. (#362, #363)

- **Scope model refresh auto-apply edge cases.** The auto-apply logic in
  sleep's model-refresh pass now handles edge cases around session boundaries
  and empty proposal sets.

## [3.10.1] - 2026-06-22

### Security

- **Fix critical JWT signature verification bypass in sync server
  ([GHSA-xcw4-53cc-hv32](https://github.com/AxDSan/mnemosyne/security/advisories/GHSA-xcw4-53cc-hv32),
  CVSS 9.1).** The sync server's authentication check decoded JWT bearer
  tokens but never verified their HMAC-SHA256 signatures, allowing any
  well-formed token (including `alg: none`) to be accepted. An
  unauthenticated attacker with network access to the sync endpoint could
  impersonate any user, read their sync state, and push malicious sync
  state to corrupt the local database.
  - Replaces broken decode with a from-scratch HS256 verifier
  - Constant-time signature comparison via `hmac.compare_digest`
  - Strict `alg: HS256` allowlist (rejects `none`, RS256, etc.)
  - UTC-aware `exp` validation with leeway
  - Loud, specific error messages
  - Reported by Denis Hache (@dplush) via private channel on 2026-06-13
  - Patched on 2026-06-19 (commit `a0b6b871`)

### Upgrade

```bash
pip install --upgrade mnemosyne-memory==3.10.1
```

If you operate a sync server with network exposure, upgrade immediately.
If you cannot upgrade right away, restrict network access to the sync
endpoint (firewall, reverse proxy with mTLS, or localhost bind with SSH
tunnel). The vulnerability is not exploitable against an unreachable
endpoint.

- **hermes integration:** `hermes mnemosyne <stats|sleep|inspect|export>` are now
  bank-aware under `profile_isolation` — they resolve the active profile bank (or an
  explicit `--bank`) instead of always reading the default bank, which reported empty
  state when the profile bank held the data. (#362, #363)

## [3.10.0] - 2026-06-18

### Added

- **L3 persona layer** — always-on behavioral rules tier that survives past
  the 24-hour working-memory TTL. New `memoria_persona` SQLite table with
  tiered retention (`permanent` / `long_term` / `working`). New tools:
  `mnemosyne_persona_promote`, `mnemosyne_persona_demote`,
  `mnemosyne_persona_list`, `mnemosyne_persona_reinforce`.
- **Rule-based persona extractor** (no LLM by default). Reads working_memory
  and episodic_memory, filters by source/importance, deduplicates by topic,
  renders Markdown grouped by topic. Deterministic and zero-cost.
- **Auto-injection into system prompt** via `persona.md`. Reads
  `~/.hermes/memory/persona.md` and includes it in the
  `system_prompt_block()` of the hermes provider. Feature-gated by
  `MNEMOSYNE_PERSONA_ENABLED=true` (default OFF). Mtime-cached for hot-path
  efficiency. Token cap enforced (`MNEMOSYNE_PERSONA_TOKEN_CAP`, default 1500).
- **5 trigger conditions** for persona regeneration (matches Hy-Memory
  PersonaTrigger pattern): explicit request, cold start, recovery,
  threshold (default 50 new memories), daily sync window.

### Design notes

- Schema migration is additive; existing tables untouched.
- Tool count: 28 -> 32.
- No breaking changes to existing `mnemosyne_remember` / `mnemosyne_recall`
  behavior.
- Default OFF to preserve opt-in upgrade story; turn on with
  `MNEMOSYNE_PERSONA_ENABLED=true` after upgrading.

## [3.9.0] - 2026-06-18

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

- **auto-sleep uses `sleep_all_sessions()` causing timeout** (issue #342, PR by
  @ruangraung). `_maybe_auto_sleep()` called `sleep_all_sessions()` which loops
  ALL sessions instead of just the current one, always exceeding the timeout on
  databases with many sessions. Now uses session-scoped `beam.sleep()`.
- **daemon thread SQLite connection race** (issue #342, PR by @ruangraung). Both
  `_maybe_auto_sleep()` and `on_session_end()` ran `beam.sleep()` in daemon
  threads but reused `self._beam.conn` (the same SQLite connection as the main
  thread). Concurrent writes caused silent episodic INSERT failures. Now creates
  isolated `BeamMemory` instances in daemon threads so each gets its own
  connection via `_thread_local`.
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

## [3.7.0] - 2026-06-13

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

## [3.6.0] - 2026-06-10

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

## [3.4.0] - 2026-06-01

### Added

- **Known dimensions for local SentenceTransformers multilingual models.**
  `paraphrase-multilingual-MiniLM-L12-v2`, `all-MiniLM-L6-v2`, and
  `paraphrase-multilingual-mpnet-base-v2` are now listed for low-resource
  local multilingual embedding setups.

### Fixed

- **Unicode recall tokenization for Latin-script languages.** Recall lexical
  gates now keep diacritics inside tokens, so words like `Stoßlüften`,
  `Bürgeramt`, and `Primärquellen` are no longer split into ASCII fragments.

## [3.3.0] - 2026-06-01

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

## [3.1.1] - 2026-05-28

### Added

- **Preferred embedding env vars.** `MNEMOSYNE_EMBEDDING_API_URL` and `MNEMOSYNE_EMBEDDING_API_KEY` are now the preferred names for custom embedding endpoints. The old `OPENROUTER_BASE_URL` and `OPENROUTER_API_KEY` names still work as fallbacks for backward compatibility. Restores the v2.8.x naming convention. ([#193](https://github.com/AxDSan/mnemosyne/issues/193))

## [3.1.0] - 2026-05-26

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

## [3.0.0] - 2026-05-18

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

## [2.9.0] - 2026-05-17

### Fixed

- **MCP SDK 1.x compatibility** (`mcp_server.py`). The `stdio_server()`
  transport no longer accepts a `Server` object as argument since v0.9.1;
  the stream pair is obtained via `async with stdio_server()` and then
  passed to `server.run()`. Tool definitions are now returned as `Tool`
  Pydantic objects instead of raw dicts, matching the SDK 1.x `list_tools`
  handler signature. Both stdio and SSE transports are patched.

## [2.8.0] - 2026-05-14

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

## [2.7.0] - 2026-05-12

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

## [2.5] - 2026-05-10

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

## [2.4] - 2026-05-07

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

## [2.3.1] - 2026-05-06

### Fixed

- **Auto-sleep consolidation blocks TUI agent**: `_maybe_auto_sleep()` now runs in a background thread with a 5-second timeout instead of synchronously. Local LLM summarization (ctransformers) can no longer hang the agent worker thread. (#23)
- `MNEMOSYNE_AUTO_SLEEP_ENABLED` env var now controls auto-sleep behavior. Default is `false` (disabled) for interactive safety. Set to `true` to re-enable.
- Config schema updated to reflect new default.

## [2.3] - 2026-05-05

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

## [2.2] - 2026-05-02

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

## [2.1] - 2026-05-02

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

## [2.0] - 2026-04-29

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

## [1.13] - 2026-04-28

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

## [1.12] - 2026-04-26

### Added

- **Feature comparison matrix** vs. cloud providers (Honcho, Zep, Mem0, Hindsight)
- **DevOps policy** — comprehensive procedures for releases, security, and operations

### Changed

- Documentation cleanup — replaced placeholder files with proper repo docs

---

## [1.11] - 2026-04-25

### Added

- **Token-aware batch sizing** in consolidation — no more OOM on large memory sets
- **Remote API support** for LLM summarization in `sleep()`

### Fixed

- Consolidation edge cases with mixed local/remote LLM configs

---

## [1.10] - 2026-04-24

### Added

- **`mnemosyne_update` tool** — modify existing memories without full replacement
- **`mnemosyne_forget` tool** — targeted memory deletion
- **Global stats flag** — `hermes mnemosyne stats --global` for workspace-wide metrics

### Fixed

- Working memory scope handling across sessions (PR #11)
- Default scope set to 'global' for migrated memories
- Working memory stats and recall tracking consistency

---

## [1.9] - 2026-04-23

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

## [1.8] - 2026-04-22

### Added

- **Plugin auto-discovery** — `register()` method for Hermes plugin CLI
- **Bug report template** — official GitHub issue template

### Fixed

- 6 bugs from Issue #6 — edge cases in recall, scope handling, and tool registration

---

## [1.7] - 2026-04-22

### Added

- **PEP 668 PSA** — documentation for Ubuntu 24.04 / Debian 12 users hitting `externally-managed-environment`

### Fixed

- Provider `register_cli` using nested parser instead of subparser
- `sys.path` injection with graceful `ImportError` fallback

---

## [1.6] - 2026-04-21

### Added

- **Feature request template** — GitHub issue template for enhancements
- **Simple versioning** adopted — MAJOR.MINOR instead of semver

### Fixed

- `fastembed` dependency correction (was incorrectly listing `sentence-transformers`)
- Benchmarks restored to README with LongMemEval scores

---

## [1.5] - 2026-04-20

### Added

- **Export/import** — cross-machine memory migration (`mnemosyne_export` / `mnemosyne_import`)
- **One-command installer** — `curl | bash` setup for new users
- **MemoryProvider mode** — deploy Mnemosyne as a standalone memory provider via plugin system
- **Anchored table of contents** in README

### Changed

- README fully rewritten — professional, community-focused, removed bloat
- FluxSpeak branding removed from LICENSE and metadata (Mnemosyne is its own thing)

---

## [1.4] - 2026-04-19

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

## [1.3] - 2026-04-17

### Added

- **Memory override skill** — bake memory into pre_llm_call and session_start hooks
- **Critical deprecation notice** for legacy memory tool

---

## [1.2] - 2026-04-13

### Added

- **Scale limits** — tested and documented for 1M+ token capacity
- **Legacy DB migration script** — upgrade path from early schemas

### Changed

- Auto-logging of `tool_execution` disabled by default (privacy)

---

## [1.1] - 2026-04-10

### Added

- **BEAM architecture** — sqlite-vec + FTS5 + sleep consolidation
- **BEAM benchmarks** — dedicated benchmark suite with published results
- **Dense retrieval** via fastembed
- **AAAK compression** — compressed memory format for context injection
- **Temporal triples** — structured fact storage with subject/predicate/object

### Fixed

- Thread-local connection bug

---

## [1.0] - 2026-04-05

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
[3.8.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.8.0
[3.9.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.9.0
[3.10.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.10.0
[3.10.1]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.10.1
[3.11.1]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.11.1
[3.11.0]: https://github.com/AxDSan/mnemosyne/releases/tag/v3.11.0
