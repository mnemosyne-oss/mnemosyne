# Configuration

Mnemosyne is designed to work with zero configuration. All settings have sensible defaults and are overridden via environment variables.

## Custom Embedding Endpoint

|| Variable | Default | Description |
||---|---|---|
|| `MNEMOSYNE_EMBEDDING_API_URL` | `${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}` | Preferred name for custom embedding API endpoint. Falls back to `OPENROUTER_BASE_URL`. |
|| `MNEMOSYNE_EMBEDDING_API_KEY` | `${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}` | Preferred name for embedding API key. Falls back to `OPENROUTER_API_KEY`, then `OPENAI_API_KEY`. |

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
| `MNEMOSYNE_WM_TTL_HOURS` | `24` | TTL in hours for **unconsolidated** working memory entries |

Consolidated rows (those stamped `consolidated_at` by `sleep()`) are exempt from both limits. They remain queryable until explicitly removed via `forget()`. This is by design — the E3 additive memory contract guarantees that consolidated content persists.

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

# Or any fastembed-supported model
MNEMOSYNE_EMBEDDING_MODEL=intfloat/multilingual-e5-base
```

The embedding dimension is **auto-detected** from the model name. Supported models with known dimensions:

| Model | Dims | Language |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | English |
| `BAAI/bge-base-en-v1.5` | 768 | English |
| `BAAI/bge-small-zh-v1.5` | 512 | Chinese |
| `BAAI/bge-base-zh-v1.5` | 768 | Chinese |
| `BAAI/bge-large-zh-v1.5` | 1,024 | Chinese |
| `BAAI/bge-m3` | 1,024 | Multilingual |
| `intfloat/multilingual-e5-small` | 384 | Multilingual |
| `intfloat/multilingual-e5-base` | 768 | Multilingual |
| `intfloat/multilingual-e5-large` | 1,024 | Multilingual |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilingual |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | Multilingual |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | Multilingual |
| `BAAI/bge-m3` | 1,024 | Multilingual |
| `openai/text-embedding-3-small` | 1,536 | API |
| `openai/text-embedding-3-large` | 3,072 | API |

For unsupported models, set the dimension explicitly:

```bash
MNEMOSYNE_EMBEDDING_DIM=768
```

> **Warning:** Changing the embedding model after data has been stored will cause a dimension mismatch. The vec0 virtual table is locked to the dimension it was created with. To switch models, delete and re-create the database, or run the migration tool.

## LLM Consolidation

### Local LLM (ctransformers / GGUF)

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_ENABLED` | `true` | Enable LLM summarization during sleep cycle |
| `MNEMOSYNE_LLM_N_CTX` | `2048` | Context window size for the local model |
| `MNEMOSYNE_LLM_MAX_TOKENS` | `2048` | Maximum output tokens per summary |
| `MNEMOSYNE_LLM_N_THREADS` | `4` | CPU threads for local inference |
| `MNEMOSYNE_LLM_REPO` | `openbmb/MiniCPM5-1B-GGUF` | HuggingFace repo for GGUF model |
| `MNEMOSYNE_LLM_FILE` | `MiniCPM5-1B-Q4_K_M.gguf` | GGUF filename |
| `MNEMOSYNE_SLEEP_PROMPT` | *(built-in)* | Optional sleep/consolidation prompt override. Supports `{source}`, `{memories}`, and `{memory_count}` placeholders for language-specific summaries. |

### Remote LLM (OpenAI-compatible)

Use a remote model instead of the local MiniCPM5-1B GGUF:

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_BASE_URL` | *(none)* | OpenAI-compatible API base URL (e.g. `http://localhost:8080/v1`) |
| `MNEMOSYNE_LLM_API_KEY` | *(none)* | API key for authenticated endpoints |
| `MNEMOSYNE_LLM_MODEL` | *(none)* | Model identifier sent in requests |
| `MNEMOSYNE_LLM_TIMEOUT` | `60` | HTTP timeout in seconds for remote LLM calls. Increase for slow proxies or models with long generation times (e.g. `300` for reasoning models routed through local proxies). |

When `MNEMOSYNE_LLM_BASE_URL` is set, Mnemosyne uses the remote endpoint for consolidation. Falls back to local ctransformers if the remote is unreachable, then to AAAK encoding.

Works with: llama.cpp server, vLLM, Ollama, LM Studio, or any OpenAI-compatible API.

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

```
0. Host LLM adapter (if MNEMOSYNE_HOST_LLM_ENABLED=true AND a backend is registered)
   ↓ (on failure: skip remote, go to local)
1. Remote LLM (if MNEMOSYNE_LLM_BASE_URL is set AND host is not enabled)
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
| `MNEMOSYNE_SYNC_API_KEY` | *(none)* | API key for authenticating sync requests |
| `MNEMOSYNE_SYNC_JWT` | *(none)* | JWT token for JWT-based sync auth |
| `MNEMOSYNE_SYNC_KEY` | *(none)* | Base64-encoded 32-byte encryption key for client-side payload encryption |
| `MNEMOSYNE_SYNC_KEY_SOURCE` | `env` | Where to read the encryption key: `env`, `keyring`, `prompt`, `file` |
| `MNEMOSYNE_SYNC_KEY_FILE` | *(none)* | Path to a file containing the base64-encoded encryption key |
| `MNEMOSYNE_SYNC_PASSPHRASE` | *(none)* | Passphrase for key derivation (PBKDF2/Argon2id) |
| `MNEMOSYNE_SYNC_SERVER_PORT` | `8765` | Default port for `mnemosyne sync serve` |
| `MNEMOSYNE_SYNC_INSECURE` | `false` | Allow plain HTTP connections (dev only) |

See [docs/sync.md](sync.md) for full usage and [docs/security.md](security.md) for the security model.

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
