# Configuration

Mnemosyne is designed to work with zero configuration. All settings have sensible defaults and are overridden via environment variables.

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
| `MNEMOSYNE_WM_MAX_ITEMS` | `10000` | Maximum items in working memory |
| `MNEMOSYNE_WM_TTL_HOURS` | `24` | Time-to-live for working memory entries (hours) |

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

## Vector Compression

```bash
MNEMOSYNE_VEC_TYPE=int8
```

| Value | Size per vector | Description |
|---|---|---|
| `float32` | 1,536 bytes | Full precision. Largest, most accurate. |
| `int8` | 384 bytes | **Default.** Good balance of size vs. accuracy. |
| `bit` | 48 bytes | 32× smaller than float32. Fastest, lowest precision. |

All values use 384-dimensional vectors (bge-small-en-v1.5 embedding model).

## LLM Consolidation

### Local LLM (ctransformers / GGUF)

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_ENABLED` | `true` | Enable LLM summarization during sleep cycle |
| `MNEMOSYNE_LLM_N_CTX` | `2048` | Context window size for the local model |
| `MNEMOSYNE_LLM_MAX_TOKENS` | `256` | Maximum output tokens per summary |
| `MNEMOSYNE_LLM_N_THREADS` | `4` | CPU threads for local inference |
| `MNEMOSYNE_LLM_REPO` | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` | HuggingFace repo for GGUF model |
| `MNEMOSYNE_LLM_FILE` | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | GGUF filename |

### Remote LLM (OpenAI-compatible)

Use a remote model instead of local TinyLlama:

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_BASE_URL` | *(none)* | OpenAI-compatible API base URL (e.g. `http://localhost:8080/v1`) |
| `MNEMOSYNE_LLM_API_KEY` | *(none)* | API key for authenticated endpoints |
| `MNEMOSYNE_LLM_MODEL` | *(none)* | Model identifier sent in requests |

When `MNEMOSYNE_LLM_BASE_URL` is set, Mnemosyne uses the remote endpoint for consolidation. Falls back to local ctransformers if the remote is unreachable, then to AAAK encoding.

Works with: llama.cpp server, vLLM, Ollama, LM Studio, or any OpenAI-compatible API.

### Host LLM Adapter (Hermes / agent integration)

Route consolidation and fact extraction through a host-provided LLM (e.g., Hermes' authenticated `agent.auxiliary_client.call_llm`). Useful for OAuth-backed providers like `openai-codex` that don't fit the URL+API-key remote shape.

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_HOST_LLM_ENABLED` | `false` | Opt in to host-adapter routing |
| `MNEMOSYNE_HOST_LLM_PROVIDER` | *(none)* | Optional provider override, e.g. `openai-codex` |
| `MNEMOSYNE_HOST_LLM_MODEL` | *(none)* | Optional model override, e.g. `gpt-5.1-mini` |
| `MNEMOSYNE_HOST_LLM_N_CTX` | `32000` | Prompt-budget when host is the chosen path (TinyLlama-calibrated `LLM_N_CTX=2048` is too small for Codex/GPT-class) |

When the host call fails, the adapter falls back to the local GGUF model rather than the remote URL. See [hermes-llm-integration.md](hermes-llm-integration.md) for the full behavior model and session-shutdown semantics.

### Fallback Chain

```
0. Host LLM adapter (if MNEMOSYNE_HOST_LLM_ENABLED=true AND a backend is registered)
   ↓ (on failure: skip remote, go to local)
1. Remote LLM (if MNEMOSYNE_LLM_BASE_URL is set AND host is not enabled)
   ↓ (on failure)
2. Local LLM (ctransformers + TinyLlama GGUF)
   ↓ (on failure or not installed)
3. AAAK encoding (keyword-based, no LLM required)
```

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
