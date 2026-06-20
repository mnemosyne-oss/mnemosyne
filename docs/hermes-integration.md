# Hermes Integration

Mnemosyne is designed as a native memory backend for the [Hermes Agent Framework](https://github.com/NousResearch/hermes-agent). It implements the Hermes `MemoryProvider` interface and registers as a plugin.

> **This is the canonical Hermes setup guide.** The README links here for full instructions.

## Install Profile Comparison

| Profile | When to use | RAM | Key tradeoff |
|---------|-------------|-----|-------------|
| `mnemosyne-memory` (core) | Low-resource (Raspberry Pi, 1 GB VPS), or when using a remote embedding API | ~50 MB | No local embeddings. Point `MNEMOSYNE_EMBEDDING_API_URL` to an external endpoint. |
| `mnemosyne-memory[embeddings]` | Mid-range systems with local embedding support | ~800 MB | Adds `fastembed` for local vector generation. Best for single-user desktop agents. |
| `mnemosyne-memory[all]` | Full-featured — local embeddings + local LLM consolidation | ~1.5 GB | Adds `sentence-transformers` + local LLM deps (`ctransformers`). Maximum capability. |
| `mnemosyne-hermes` | Hermes Agent users — always pair with one of the above | Same as base | Wraps core library with plugin manifest + entry points. Run `hermes config set memory.provider mnemosyne` after install. |

**Hardware guidance:** Core alone runs on a Raspberry Pi 4 (4 GB) with ~300 MB free for LLM. `[embeddings]` needs at least 2 GB free RAM. `[all]` recommends 8 GB+.

## Setup

### Step 1: Install

**pip (recommended):**

```bash
pip install mnemosyne-hermes
```

**Debian / Trixie users:** newer Debian releases block bare pip installs. Use a venv:

```bash
python3 -m venv ~/.hermes/hermes-agent/venv
source ~/.hermes/hermes-agent/venv/bin/activate
pip install mnemosyne-hermes
```

**Or from source:**

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e "integrations/hermes[dev]"
```

> **Docker users:** Inside the official Hermes Docker container, the Hermes home directory is `/opt/data/` (the mounted volume), not `~/.hermes/`. For image-based or read-only Hermes venv installs, prefer the persistent wrapper mode so the plugin discovery directory survives image rebuilds:
>
> ```bash
> # Inside the container, pointing at a side/persistent venv that has mnemosyne-hermes installed
> mnemosyne-hermes install --hermes-home /opt/data --mode wrapper --python /path/to/venv/bin/python
> mnemosyne-hermes status --hermes-home /opt/data
> hermes gateway restart
> ```
>
> The default `install` mode still creates the historical plugin symlink. Wrapper mode creates a real directory under `$HERMES_HOME/plugins/mnemosyne/` and imports `mnemosyne_hermes` from the selected Python environment. Skip Steps 2-4 below; the installer handles everything. Run `hermes memory setup` after restarting to activate.

### Step 2: Link the plugin

Hermes discovers plugins by scanning a folder on disk, not by reading pip's metadata. Link the installed package into the plugins directory so Hermes can find it:

```bash
# Auto-detect the installed package path and symlink it
mkdir -p ~/.hermes/plugins/mnemosyne
ln -sfn "$(~/.hermes/hermes-agent/venv/bin/python -c 'import pathlib, mnemosyne_hermes; print(pathlib.Path(mnemosyne_hermes.__file__).resolve().parent)')"/* ~/.hermes/plugins/mnemosyne/
```

If you installed in a custom venv (e.g. `~/.hermes-venv`), replace `~/.hermes/hermes-agent/venv/bin/python` with the Python binary inside that venv.

### Step 3: Activate

```bash
hermes config set memory.provider mnemosyne
hermes memory setup
```

### Step 4: Disable built-in memory

Disable Hermes' built-in MEMORY.md/USER.md system so Mnemosyne is the sole memory provider. Do NOT use `hermes tools disable memory` — that also kills all 23 Mnemosyne-registered tools.

Edit `~/.hermes/config.yaml`:

```yaml
memory:
  memory_enabled: false
user_profile_enabled: false
```

`memory_enabled: false` turns off the file-based MEMORY.md system. `user_profile_enabled: false` stops USER.md injection. Both are redundant once Mnemosyne is active.

### Step 5: Verify

```bash
hermes memory status       # Should show "Provider: mnemosyne"
hermes mnemosyne stats     # Working + episodic memory counts
```

> If `hermes mnemosyne stats` gives "invalid choice: 'mnemosyne'", the plugin CLI registration didn't load. Use the fallback `hermes hermes-mnemosyne stats` instead, or re-run step 2 to relink the plugin.

## How It Works

Mnemosyne hooks into the Hermes agent lifecycle:

| Hook | Behavior |
|---|---|
| `pre_llm_call` | Injects relevant working memory context into the prompt |
| `on_session_start` | Initializes session-scoped memory state |
| `post_tool_call` | Captures tool results as memories (if configured) |

### Registered Tools

Mnemosyne registers these tools in the Hermes tool registry:

| Tool | Description |
|---|---|
| `mnemosyne_remember` | Store a memory |
| `mnemosyne_recall` | Search memories |
| `mnemosyne_stats` | Show memory statistics |
| `mnemosyne_triple_add` | Add a knowledge graph triple |
| `mnemosyne_triple_query` | Query the knowledge graph |
| `mnemosyne_sleep` | Run consolidation |
| `mnemosyne_scratchpad_write` | Write to scratchpad |
| `mnemosyne_scratchpad_read` | Read scratchpad |
| `mnemosyne_scratchpad_clear` | Clear scratchpad |
| `mnemosyne_update` | Update a memory by ID |
| `mnemosyne_forget` | Delete a memory by ID |
| `mnemosyne_invalidate` | Mark a memory as superseded |
| `mnemosyne_export` | Export all memories to JSON |
| `mnemosyne_import` | Import memories from JSON |
| `mnemosyne_diagnose` | Run PII-safe diagnostics |

## CLI Commands

```bash
hermes mnemosyne stats              # Current session stats
hermes mnemosyne stats --global     # Stats across all sessions
hermes mnemosyne inspect "query"    # Search memories
hermes mnemosyne sleep              # Run consolidation
hermes mnemosyne export --output backup.json
hermes mnemosyne import --input backup.json

# Import historical Hindsight memories via PR #28's timestamp-preserving importer
hermes mnemosyne import --from hindsight --file hindsight-export.json --bank hermes
hermes mnemosyne import --from hindsight --input hindsight-export.json --bank hermes
hermes mnemosyne import --from hindsight --base-url http://localhost:8888 --bank hermes

hermes mnemosyne clear              # Clear scratchpad
hermes mnemosyne version            # Show version
```

## Data Location

By default, data is stored under:

```
~/.hermes/mnemosyne/
├── data/
│   ├── mnemosyne.db              # Main SQLite database (BEAM + legacy)
│   ├── triples.db                # Used by standalone TripleStore()
│   └── banks/<name>/mnemosyne.db # Named memory banks
└── ...
```

This path is chosen because Hermes already persists `~/.hermes/` across sessions (including on ephemeral VMs like Fly.io).

## Auxiliary LLM routing (Codex / OAuth providers)

By default Mnemosyne uses its own LLM config (`MNEMOSYNE_LLM_BASE_URL` /
`MNEMOSYNE_LLM_API_KEY`) or a local GGUF for sleep/consolidation and fact
extraction. Hermes users with OAuth-backed providers like `openai-codex` can
opt into routing those calls through Hermes' authenticated auxiliary client
instead — no extra credentials required.

Set `MNEMOSYNE_HOST_LLM_ENABLED=true` to enable. See
[hermes-llm-integration.md](hermes-llm-integration.md) for the full behavior
model, configuration reference, and session-shutdown semantics.

## Optional MCP Server

For integration with MCP-compatible clients:

```bash
mnemosyne mcp                          # stdio transport
mnemosyne mcp --transport sse --port 8080  # SSE transport
```

Mnemosyne does not currently expose a standalone REST API server.

## Uninstall

```bash
pip uninstall mnemosyne-hermes
hermes config set memory.provider memory   # Switch back to built-in
hermes memory setup
```
