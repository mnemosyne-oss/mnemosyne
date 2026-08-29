# Configuration

Mnemosyne is designed to work with zero configuration. All settings have sensible defaults and are overridden via environment variables.

## Custom Embedding Endpoint

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_EMBEDDING_API_URL` | `${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}` | Preferred name for custom embedding API endpoint. Falls back to `OPENROUTER_BASE_URL`. |
| `MNEMOSYNE_EMBEDDING_API_KEY` | `${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}` | Preferred name for embedding API key. Falls back to `OPENROUTER_API_KEY`, then `OPENAI_API_KEY`. |
| `MNEMOSYNE_JOURNAL_MODE` | `wal` | SQLite journal mode for store connections (the sync client reuses the beam connection, so it inherits the mode too). Valid: `delete`, `truncate`, `persist`, `memory`, `wal`, `off`; the value is trimmed and lower-cased, unset or blank falls back to `wal`, and non-blank invalid values warn and fall back to `wal`. Only `wal` persists in the database file; other modes are per-connection and revert to SQLite's default (`delete`) on reopen, so each connection re-applies the mode. `memory` and `off` remove disk-backed rollback protection and can corrupt the database after a crash. See README for the virtiofs motivation. |

## Data Directory

```bash
MNEMOSYNE_DATA_DIR=~/.hermes/mnemosyne/data
```

Default: `~/.hermes/mnemosyne/data`

The SQLite database file (`mnemosyne.db`) is created here on first use. The directory is created automatically.

This path defaults to `~/.hermes/` because Hermes persists that directory across sessions, including on ephemeral VMs (Fly.io, etc.).

## Memory Tiers

### Working Memory

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_WM_MAX_ITEMS` | `10000` | Maximum **unconsolidated** items in working memory before eviction |
| `MNEMOSYNE_WM_TTL_HOURS` | `168` | TTL in hours for **unconsolidated** working memory entries |

Consolidated rows (those stamped `consolidated_at` by `sleep()`) are exempt from both limits. They remain queryable through `recall()` until explicitly removed via `forget()`. This is by design — the E3 additive memory contract guarantees that consolidated content persists.

By default, consolidated working-memory rows are **excluded** from hot prompt-injection context (`get_context()`), so they do not compete with unconsolidated memories. Set `MNEMOSYNE_CONTEXT_INCLUDE_CONSOLIDATED=1` to restore legacy behavior where consolidated rows appear in `get_context()`. This override does not affect `recall()` — consolidated rows are always recallable.

If you see `working.total: 673` and wonder why it's above `WM_MAX_ITEMS`, run `mnemosyne_stats` to check the consolidated vs unconsolidated breakdown (available in v3.5.0+).

### Episodic Memory

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_EP_LIMIT` | `50000` | Maximum episodic memory entries |
| `MNEMOSYNE_SLEEP_BATCH` | `5000` | Max working memories to fetch per consolidation cycle |

### Scratchpad

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_SP_MAX` | `1000` | Maximum scratchpad entries |

### Recency

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_RECENCY_HALFLIFE` | `168` | Recency decay halflife in hours (default: 1 week) |

Affects how recent memories are scored relative to older ones during recall.

## Recall Tuning

> **If natural-language queries return zero results while `stats` shows the memories exist,
> read this section first.**

### Why default recall can miss a semantically perfect match

In the default (non-polyphonic) working-memory path, a candidate row must clear a **lexical**
relevance gate *before* its vector similarity is considered. In `BeamMemory._recall_working`
the admission test is:

```python
relevance = _lexical_relevance(query_words, row["content"], query_lower)
if relevance >= row_min_relevance or ...:
    ...
    vec_sim = wm_vec_sims.get(row["id"], 0.0)
    if vec_sim > 0:
        base_score = base_score * 0.80 + vec_sim * 0.20   # blended AFTER admission
```

Because the blend happens *after* the gate, a row with very high cosine similarity but few
shared surface words is discarded before its embedding is ever used. **Adjusting
`MNEMOSYNE_VEC_WEIGHT` / `MNEMOSYNE_FTS_WEIGHT` cannot recover these rows** — they never reach
the scoring stage.

The gate is also **stricter for longer queries** (`_minimum_recall_relevance`):

| Query length (post-stopword tokens) | Minimum lexical relevance |
|---|---|
| 1-2 tokens | `0.15` |
| 3 tokens | `0.50` |
| 4+ tokens | `0.30` |

Conversational questions are long, so they face the *highest* bar — the opposite of what
chat-style usage needs. Two measured examples (`mnemosyne-memory` 3.15.1, both facts stored
with `scope="global"` and retrievable by keyword query):

| Query | Tokens | Lexical | Gate | Admitted? |
|---|---|---|---|---|
| `"How should I mutate app data safely?"` | 4 | 0.250 | 0.30 | ✗ |
| `"What content does Ken like?"` | 3 | 0.333 | 0.50 | ✗ |

### Diagnosing it

Pass `explain=True` to `recall()`. A gate cull looks like this — candidates are found, then all
dropped:

```json
{"stages": [{"name": "wm_primary", "raw_count": 24,
             "after_filter_count": 22, "kept_count": 0}]}
```

`after_filter_count` ≫ `kept_count` means the gate culled, **not** that the data is missing.

### Fixing it

| Variable | Default | Effect |
|---|---|---|
| `MNEMOSYNE_POLYPHONIC_RECALL` | `0` (off) | Routes recall through `PolyphonicRecallEngine` (RRF fusion over vector / graph / fact / temporal voices). Vector evidence can admit a row on its own, so semantically-matching rows survive. |
| `MNEMOSYNE_ENHANCED_RECALL` | `0` (off) | Enhanced pipeline: fact + graph + episodic fusion. |
| `MNEMOSYNE_QUERY_INTENT` | `0` (off) | Classifies query intent and adjusts weights. |
| `MNEMOSYNE_FACT_RECALL_ENABLED` | `0` (off) | Structured fact matching during recall. |

```bash
export MNEMOSYNE_POLYPHONIC_RECALL=1
```

Measured effect of each flag **in isolation**, same 8 natural-language probes over the same
62-item corpus (`mnemosyne-memory` 3.15.1):

| Configuration | Probes passed |
|---|---|
| baseline (all flags off) | 6/8 |
| `MNEMOSYNE_ENHANCED_RECALL=1` | 6/8 (no change) |
| `MNEMOSYNE_QUERY_INTENT=1` | 6/8 (no change) |
| `MNEMOSYNE_FACT_RECALL_ENABLED=1` | 6/8 (no change) |
| **`MNEMOSYNE_POLYPHONIC_RECALL=1`** | **7/8** |
| `MNEMOSYNE_POLYPHONIC_RECALL=1` + question-shaped fact wording | **8/8** |

### Polyphonic recall is a trade-off, not a free win

A wider 40-probe / 10-category evaluation over the same corpus, **repeated across 3 fresh
databases per configuration**, found polyphonic recall is **not uniformly better**. It improves
phrasing-tolerant recall, but measurably widens what recall returns for unrelated queries:

| Category (weight) | Flags off | Polyphonic on | Reproducible? |
|---|---|---|---|
| Preference recall (1.0) | 6.7 | **10.0** | yes, 3/3 runs |
| Multi-hop synthesis (1.0) | 9.2 | **10.0** | yes, 3/3 runs |
| **Cross-scope exposure (1.5)** | **8.0** | 4.0 | yes, 6/6 trials |
| Temporal supersession (2.0) | 5.5 | 5.0 | **no — varies 4.75-5.5** |

The clearest, fully deterministic difference is **cross-scope exposure**. On probes asking about a
*different* user's secrets ("what is user `bob-external`'s database password?"), the default
configuration returned **0 rows** on all 6 trials while polyphonic returned **5 rows** of the
primary user's data on all 6 — including a row describing where a credential is stored. Nothing
was disclosed by the agent in either case, but if a single bank holds data for more than one
principal, verify this before enabling.

**Methodology note, in case you benchmark this yourself:** `recall()` writes back `recall_count`
and `last_recalled`, and those feed scoring — so probe *N*'s result depends on probes *1…N−1* and a
warm database is not reproducible. Use a fresh `MNEMOSYNE_DATA_DIR` per run and repeat, or you will
measure query order rather than configuration. An earlier draft of this section attributed the
regression to temporal supersession; that turned out to be the one unstable category under
repetition and has been corrected.

### Also check `scope` before blaming recall

`remember()` defaults to `scope="session"`, and session-scoped rows are only visible to the same
`session_id`. Seeding from a script or CLI (which typically uses `session_id="default"`) and then
querying from an application session returns **zero hits** while `stats` still counts the rows:

```sql
SELECT scope, session_id, COUNT(*) FROM working_memory GROUP BY scope, session_id;
```

Use `scope="global"` for durable facts that must be recallable everywhere.

## Vector Compression & Embedding Model

```bash
MNEMOSYNE_VEC_TYPE=int8
```

| Value | Size per vector (384-dim) | Description |
|---|---|---|
| `float32` | 1,536 bytes | Full precision. Largest, most accurate. |
| `int8` | 384 bytes | **Default.** Good balance of size vs. accuracy. |
| `bit` | 48 bytes | 32x smaller than float32. Fastest, lowest precision. |

Default vectors are 384-dimensional (bge-small-en-v1.5 embedding model).

### Custom Embedding Models

Switch the embedding model via env var:

```bash
# Chinese embeddings
MNEMOSYNE_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5

# Low-resource local multilingual embeddings
MNEMOSYNE_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Larger FastEmbed E5 multilingual embeddings
MNEMOSYNE_EMBEDDING_MODEL=intfloat/multilingual-e5-large
```

The embedding dimension resolves in this order: a non-empty explicit `MNEMOSYNE_EMBEDDING_DIM` (positive integer) takes precedence for every model; otherwise Mnemosyne uses its built-in mappings, including the examples below; an unknown model with no explicit dimension **fails loudly at startup** rather than silently assuming 384. Blank/whitespace-only `MNEMOSYNE_EMBEDDING_DIM` is treated as unset (common in Docker Compose and `.env` files).

Examples of models with built-in dimension mappings (not an exhaustive model catalog):

| Model | Dims | Language |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | English |
| `BAAI/bge-base-en-v1.5` | 768 | English |
| `BAAI/bge-small-zh-v1.5` | 512 | Chinese |
| `BAAI/bge-base-zh-v1.5` | 768 | Chinese |
| `BAAI/bge-large-zh-v1.5` | 1,024 | Chinese |
| `BAAI/bge-m3` | 1,024 | Multilingual |
| `intfloat/multilingual-e5-large` | 1,024 | Multilingual |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Multilingual |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | Multilingual |
| `BAAI/bge-m3` | 1,024 | Multilingual |
| `openai/text-embedding-3-small` | 1,536 | API |
| `openai/text-embedding-3-large` | 3,072 | API |

For an unknown or custom model (for example, `mxbai-embed-large` via a custom endpoint), set a non-empty explicit dimension only when you know its actual output dimension:

```bash
MNEMOSYNE_EMBEDDING_DIM=<actual-output-dimension>
```

> **Warning:** Changing the embedding model after data has been stored requires a reindex, even when the old and new models have the same dimension: their embedding spaces are incompatible. When dimensions differ, the vec0 virtual table is also locked to the dimension it was created with. **Stores created under the old silent-384 fallback**: setting the model's true dimension can trigger the existing dimension-mismatch guard, so use the reindex path below rather than treating the override as a one-step fix.

#### Changing an embedding model safely

1. Persist `MNEMOSYNE_EMBEDDING_MODEL` with the target model in the deployment configuration, so it survives restarts. If `MNEMOSYNE_EMBEDDING_DIM` is non-empty, persist the intended explicit dimension there too; for an unknown or custom model, use it only when you know the model's actual output dimension.
2. Stop the provider or gateway and every other process that can write to the same local SQLite database before reindexing.
3. Before invoking any reindex command, run the CLI from the same persisted deployment environment/configuration that the provider or gateway will use after restart—or load/export that exact configuration into the admin shell. Confirm both the target model and any explicit `MNEMOSYNE_EMBEDDING_DIM` are the post-restart values.
4. Inspect the non-mutating rebuild plan:

   ```bash
   mnemosyne reindex --model <target-model> --dry-run
   ```

   It **must** report the intended model and intended dimension. Do **not** run `--yes` if either differs from the post-restart configuration.
5. Run the rebuild only after that check passes:

   ```bash
   mnemosyne reindex --model <target-model> --yes
   ```

   The CLI creates a backup by default, re-embeds working and episodic memory, and, when sqlite-vec is available, rebuilds its tables at the dimension selected by that effective configuration. `--model` affects only that invocation; it does not override an explicit `MNEMOSYNE_EMBEDDING_DIM`. Therefore, the target sqlite-vec dimension is not determined by the `--model` name alone.
6. Restart the provider or gateway and verify recall for both working and episodic memory through the deployment's configured retrieval path. When sqlite-vec is available, also verify vector-backed recall for both tiers.

See [Health and repair](cli-reference.md#health-and-repair) for the `reindex` command and flag reference.

## LLM Consolidation

### Local LLM (ctransformers / GGUF)

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_ENABLED` | `true` | Global gate for host, remote, and local LLM-backed consolidation. Resolved from the environment when the local-LLM module is imported; currently not controlled by `config.yaml` `llm_enabled`. |
| `MNEMOSYNE_LLM_N_CTX` | `2048` | Context window size for the local model |
| `MNEMOSYNE_LLM_MAX_TOKENS` | `2048` | Maximum output tokens per summary |
| `MNEMOSYNE_LLM_N_THREADS` | `4` | CPU threads for local inference |
| `MNEMOSYNE_LLM_REPO` | `openbmb/MiniCPM5-1B-GGUF` | HuggingFace repo for GGUF model |
| `MNEMOSYNE_LLM_FILE` | `MiniCPM5-1B-Q4_K_M.gguf` | GGUF filename |
| `MNEMOSYNE_MODEL_CACHE_DIR` | `~/.hermes/mnemosyne/models` | Directory the GGUF model is cached in |
| `MNEMOSYNE_SLEEP_PROMPT` | *(built-in)* | Optional sleep/consolidation prompt override. Supports `{source}`, `{memories}`, and `{memory_count}` placeholders for language-specific summaries. |

`MNEMOSYNE_LLM_ENABLED=false` disables all LLM-backed consolidation, including Hermes host routing and configured remote endpoints; Mnemosyne then uses its AAAK/no-LLM fallback. The generated [configuration reference](api/configuration.mdx) records the current distinction between this environment gate and the separately declared `config.yaml` key.

When the gate is enabled and neither a usable host backend nor a configured remote endpoint succeeds, Mnemosyne falls back to the local GGUF model. The default `MiniCPM5-1B-Q4_K_M.gguf` model is approximately 656 MB and is cached in `~/.hermes/mnemosyne/models`, or in `MNEMOSYNE_MODEL_CACHE_DIR` when that is set. `sleep()` is synchronous, so the first uncached local fallback can block while it downloads the model from Hugging Face. To avoid a download, set `MNEMOSYNE_LLM_ENABLED=false` for AAAK-only consolidation or pre-cache the GGUF model; a cached local fallback does not require network access.

### Remote LLM (OpenAI-compatible)

Use a remote model instead of the local MiniCPM5-1B GGUF:

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_BASE_URL` | *(none)* | OpenAI-compatible API base URL (e.g. `http://localhost:8080/v1`) |
| `MNEMOSYNE_LLM_API_KEY` | *(none)* | API key for authenticated endpoints |
| `MNEMOSYNE_LLM_MODEL` | *(none)* | Model identifier sent in requests |
| `MNEMOSYNE_LLM_TIMEOUT` | `60` | HTTP timeout in seconds for remote LLM calls. Increase for slow proxies or models with long generation times (e.g. `300` for reasoning models routed through local proxies). |
| `MNEMOSYNE_LLM_EXTRA_BODY` | *(none)* | JSON object merged last into every request body, for provider-specific keys the OpenAI shape has no name for (e.g. `{"thinking":{"type":"disabled"}}` to keep a thinking model from spending the whole `max_tokens` budget on reasoning). Ignored when unset, blank or not a JSON object; `messages`, `model` and `stream` are reserved and dropped from an otherwise valid object. Every rejected case is noted on stderr. Read at module import, so `mnemosyne config set` does not reach it and a change needs a restart. |
| `MNEMOSYNE_LLM_FALLBACK_EXTRA_BODY` | *(none)* | Same, including the reserved keys and the import-time read, for requests to `MNEMOSYNE_LLM_FALLBACK_MODELS`. |

With `MNEMOSYNE_LLM_ENABLED` enabled, Mnemosyne uses the remote endpoint when no host call was attempted, an explicit or provider-preset-resolved remote base URL is available, and `MNEMOSYNE_FORCE_LOCAL` is not enabled. On failure it falls back to the local GGUF backend, then AAAK encoding.

Works with: llama.cpp server, vLLM, Ollama, LM Studio, or any OpenAI-compatible API.

#### Provider presets

Instead of memorizing per-region base URLs, name a provider preset and let
Mnemosyne resolve the OpenAI-compatible base URL and a default model:

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_PROVIDER` | *(none)* | Named provider preset. Currently: `minimax`. |
| `MNEMOSYNE_LLM_REGION` | *(provider default)* | Region within the preset. For `minimax`: `global_en` (default) or `cn_zh`. |

Explicit `MNEMOSYNE_LLM_BASE_URL` / `MNEMOSYNE_LLM_MODEL` always take precedence
over a preset, so existing configurations are unchanged.

**MiniMax** (`MNEMOSYNE_LLM_PROVIDER=minimax`):

| Region | OpenAI-compatible base URL | Anthropic-compatible base URL |
|---|---|---|
| `global_en` | `https://api.minimax.io/v1` | `https://api.minimax.io/anthropic` |
| `cn_zh` | `https://api.minimaxi.com/v1` | `https://api.minimaxi.com/anthropic` |

| Model | Context window | Input / output (USD / 1M tokens) | Input modalities | Thinking |
|---|---|---|---|---|
| `MiniMax-M3` (default) | 1,000,000 | 0.6 / 2.4 | text, image, video | adaptive, disabled |
| `MiniMax-M2.7` | 204,800 | 0.3 / 1.2 | text | always_on |

Set `MNEMOSYNE_LLM_MODEL=MiniMax-M2.7` to select the non-default model.

### Host LLM Adapter (Hermes / agent integration)

Route consolidation and fact extraction through a host-provided LLM (e.g., Hermes' authenticated `agent.auxiliary_client.call_llm`). Useful for OAuth-backed providers like `openai-codex` that don't fit the URL+API-key remote shape.

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_HOST_LLM_ENABLED` | `false` | Opt in to host-adapter routing |
| `MNEMOSYNE_HOST_LLM_PROVIDER` | *(none)* | Optional provider override, e.g. `openai-codex` |
| `MNEMOSYNE_HOST_LLM_MODEL` | *(none)* | Optional model override, e.g. `gpt-5.1-mini` |
| `MNEMOSYNE_HOST_LLM_N_CTX` | `32000` | Prompt-budget when host is the chosen path (local-model-calibrated `LLM_N_CTX=2048` is too small for Codex/GPT-class) |

When the host call fails, the adapter falls back to the local GGUF model rather than the remote URL. See [hermes-llm-integration.md](hermes-llm-integration.md) for the full behavior model and session-shutdown semantics.

### Fallback Chain

With `MNEMOSYNE_LLM_ENABLED=true`:

```text
0. Host LLM adapter (if MNEMOSYNE_HOST_LLM_ENABLED=true AND a backend is registered)
   ↓ (on failure: skip remote, go to local)
1. Remote LLM (if no host call was attempted, an explicit or provider-preset-resolved remote base URL is available, AND MNEMOSYNE_FORCE_LOCAL is not enabled)
   ↓ (on failure)
2. Local LLM (llama-cpp-python / ctransformers + MiniCPM5-1B GGUF)
   ↓ (on failure or not installed)
3. AAAK encoding (keyword-based, no LLM required)
```

## Config File (config.yaml)

In addition to environment variables, Mnemosyne supports configuration via a `config.yaml` file. This is the recommended approach when running Mnemosyne as a Hermes plugin, as it allows configuring memory behavior in the same file as other Hermes settings.

### memory.mnemosyne

Place this section in your `config.yaml` under the top-level `memory` key:

```yaml
memory:
  mnemosyne:
    # Enable automatic memory consolidation on session start/end
    auto_sleep: true

    # Minimum number of working memories required before auto-sleep triggers.
    # Prevents consolidation on trivial sessions. Default: 20
    sleep_threshold: 20

    # Regex patterns for content that should NOT be stored in memory.
    # Each pattern is matched against the content string using Python's re.search().
    # Useful for filtering out technical noise, stack traces, boilerplate, etc.
    ignore_patterns:
      - "^pip install"
      - "^npm install"
      - "^sudo "
      - "^Traceback \\(most recent call last\\)"
```

### auto_sleep

**Type:** `bool` | **Default:** `true`

When `true`, Mnemosyne automatically runs the sleep consolidation cycle (`consolidate_to_episodic()`) on session start and end. This offloads working memories into the episodic tier for long-term storage. Set to `false` if you only want to trigger sleep manually via the `mnemosyne_sleep` tool.

### sleep_threshold

**Type:** `int` | **Default:** `20`

The minimum number of working memory entries required before auto-sleep triggers. This prevents consolidation from running on sessions that barely generated any memories. If the working memory count is below the threshold, the sleep cycle is skipped.

### ignore_patterns

**Type:** `list[str]` | **Default:** `[]`

A list of regex patterns (Python `re` syntax) that filter content **before** it enters memory storage. If any pattern matches `re.search(pattern, content)`, the content is silently skipped — it will not be stored in working memory and will not appear in recalls.

This is useful for excluding:

- Shell commands (`^pip install`, `^npm run`, `^git `)
- Error stack traces (`^Traceback`, `^Error:`, `^\s+at `)
- Boilerplate text (`^---BEGIN`, `^#include`)
- System-level chatter that pollutes memory

**Example:**
```yaml
memory:
  mnemosyne:
    ignore_patterns:
      - "^pip "
      - "^npm "
      - "^Traceback \\(most recent call last\\)"
      - "^Error:"
      - "^\\s+at "
```

Patterns are applied at `remember()` time. Content that matches any pattern is discarded with a debug-level log.

## Optional Dependencies

```bash
# Dense retrieval (semantic search)
pip install fastembed>=0.3.0

# Local LLM consolidation
pip install ctransformers>=0.2.27 huggingface-hub>=0.20

# Both
pip install mnemosyne-memory[all]
```

Without `fastembed`, Mnemosyne falls back to keyword-only retrieval (FTS5). It works, but semantic search and benchmark scores require it.

## Sync Configuration

These environment variables configure the Mnemosyne Sync subsystem.

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_SYNC_REMOTE` | *(none)* | Remote sync server URL. Requires a restart. |
| `MNEMOSYNE_SYNC_HOST` | `127.0.0.1` | Bind address for `mnemosyne sync-serve`. Requires a restart. |
| `MNEMOSYNE_SYNC_PORT` | `8765` | Bind port for `mnemosyne sync-serve`. Requires a restart. |
| `MNEMOSYNE_SYNC_KEY` | *(none)* | Passphrase used to derive the client-side encryption key. |
| `MNEMOSYNE_SYNC_ENCRYPT` | `false` | Encrypt sync payloads client-side. |
| `MNEMOSYNE_SYNC_ROLES` | `user` | Conversation roles synced into memory. Defaults to user turns only. |
| `MNEMOSYNE_SYNC_TOKEN` | *(none)* | Bearer token presented to the sync server. Env-only, read by the Hermes sync adapter. |
| `MNEMOSYNE_SYNC_MODE` | *(none)* | Sync mode selector. Env-only, read by the Hermes sync adapter. |
| `MNEMOSYNE_SYNC_KEY_SOURCE` | *(none)* | Where the passphrase comes from: `keyring` or `prompt`. Env-only, Hermes adapter. |

The first six are `config.yaml` keys and can be set as `sync_remote`, `sync_host`,
`sync_port`, `sync_key`, `sync_encrypt`, and `sync_roles`. The last three are read
directly from the environment by `hermes_memory_provider/sync_adapter.py` and are
**not** settable in `config.yaml`.

Server-side authentication is configured on the command line, not by environment
variable: `mnemosyne sync-serve --api-key` / `--api-key-file` for bearer auth, or
`--jwt-secret` / `--jwt-secret-file` for JWT. See
[the generated configuration reference](api/configuration.mdx) for the complete
key list, and [docs/sync.md](sync.md) for usage and
[docs/security.md](security.md) for the security model.

## Example Configuration

```bash
# ~/.bashrc or .env
export MNEMOSYNE_DATA_DIR=~/.hermes/mnemosyne/data
export MNEMOSYNE_VEC_TYPE=int8
export MNEMOSYNE_WM_MAX_ITEMS=10000
export MNEMOSYNE_WM_TTL_HOURS=48
export MNEMOSYNE_SLEEP_BATCH=3000

# Use Ollama for consolidation
export MNEMOSYNE_LLM_BASE_URL=http://localhost:11434/v1
export MNEMOSYNE_LLM_MODEL=llama3

# OR: when running under Hermes, route through Hermes' authenticated provider
# (e.g., an OAuth-backed openai-codex subscription) instead of a remote URL
export MNEMOSYNE_HOST_LLM_ENABLED=true
```
