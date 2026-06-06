# Hermes Integration

Mnemosyne is designed as a native memory backend for the [Hermes Agent Framework](https://github.com/NousResearch/hermes-agent). It implements the Hermes `MemoryProvider` interface and registers as a plugin.

## Setup

### Step 1: Install

**pip (recommended):**

```bash
pip install mnemosyne-hermes
```

**Debian / Trixie users:** newer Debian releases block bare pip installs. Use a venv:

```bash
python3 -m venv ~/.hermes-venv
source ~/.hermes-venv/bin/activate
pip install mnemosyne-hermes
```

**Or from source:**

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e "integrations/hermes[dev]"
```

### Step 2: Link the plugin

Hermes discovers plugins by scanning a folder on disk, not by reading pip's metadata. Link the installed package into the plugins directory so Hermes can find it:

```bash
mkdir -p ~/.hermes/plugins/memory/mnemosyne
# Adjust the path below to match your Hermes venv Python version
ln -s ~/.hermes/hermes-agent/venv/lib/python3.11/site-packages/mnemosyne_hermes/* ~/.hermes/plugins/memory/mnemosyne/
```

If you installed in a custom venv, the path will be at `~/.hermes-venv/lib/python3.11/site-packages/mnemosyne_hermes/` instead.

### Step 3: Activate

```bash
hermes config set memory.provider mnemosyne
hermes memory setup
```

### Step 4: Verify

```bash
hermes memory status       # Should show "Provider: mnemosyne"
hermes mnemosyne stats     # Working + episodic memory counts
```

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
