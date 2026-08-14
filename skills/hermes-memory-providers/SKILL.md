---
name: hermes-memory-providers
description: "Install and configure Mnemosyne as a Hermes Agent memory provider — local SQLite with vector search, episodic consolidation, and temporal knowledge graphs."
version: 2.0.0
author: Mnemosyne
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [hermes, memory, mnemosyne, plugins, setup]
    related_skills: [hermes-agent]
---

# Mnemosyne — Hermes Memory Provider

Mnemosyne is a local-first memory layer for AI agents. When deployed as a
Hermes memory provider, it replaces the built-in MEMORY.md/USER.md system with
SQLite-backed vector + FTS5 hybrid search, episodic consolidation, temporal
knowledge graphs, and optional bidirectional sync.

**100% local. Zero cloud. Sub-millisecond recall.**

## What It Gives You

- **System prompt injection** — `# Mnemosyne Memory` context block in every prompt
- **Pre-turn prefetch** — relevant memories injected before each LLM call
- **Post-turn sync** — conversation turns auto-stored to episodic memory
- **20 tools** auto-injected into the model's tool surface (remember, recall,
  sleep, triples, scratchpad, graph, sync, diagnostics, etc.)
- **3 lifecycle hooks** — `pre_llm_call`, `on_session_start`, `post_tool_call`
- **CLI commands** — `hermes mnemosyne {stats|sleep|inspect|export|import|clear|version}`

All without touching Hermes core — deployed purely through the plugin directory.

## Quick Check

```bash
hermes memory status     # See active provider and installed plugins
```

## Install

### Step 1 — Install the package

```bash
pip install mnemosyne-hermes
```

Debian/Trixie users (bare pip blocked): use a venv first:

```bash
python3 -m venv ~/.hermes/hermes-agent/venv
source ~/.hermes/hermes-agent/venv/bin/activate
pip install mnemosyne-hermes
```

`mnemosyne-hermes` wraps the core `mnemosyne-memory` library with the plugin
manifest and entry points Hermes needs. It does not pull embeddings or LLM
deps — pair it with one of:

| Extra | When | RAM |
|-------|------|-----|
| *(core only)* | Raspberry Pi, remote embedding API | ~50 MB |
| `mnemosyne-memory[embeddings]` | Local vector search (fastembed ONNX) | ~800 MB |
| `mnemosyne-memory[all]` | Local embeddings + local LLM consolidation | ~1.5 GB |

### Step 2 — Link the plugin

```bash
mnemosyne-hermes install
```

This creates the symlink `~/.hermes/plugins/mnemosyne/ → <installed package>`
so Hermes discovers it on startup.

**Docker / read-only venv** — use persistent wrapper mode so the plugin
survives image rebuilds:

```bash
mnemosyne-hermes install --mode wrapper --python /path/to/venv/bin/python --hermes-home /opt/data
mnemosyne-hermes status --hermes-home /opt/data
hermes gateway restart
```

### Step 3 — Activate

```bash
hermes config set memory.provider mnemosyne
hermes memory setup
```

### Step 4 — (Optional) Disable built-in memory

Mnemosyne is additive by default — the built-in MEMORY.md/USER.md keeps
running alongside it. To make Mnemosyne the sole memory system, edit
`~/.hermes/config.yaml`:

```yaml
memory:
  memory_enabled: false
  user_profile_enabled: false
```

Do **NOT** run `hermes tools disable memory` — that also kills all 20
Mnemosyne-registered tools.

### Step 5 — Verify

```bash
hermes memory status       # Should show "Provider: mnemosyne"
hermes mnemosyne stats     # Working + episodic memory counts
```

Test in a conversation:

```bash
hermes chat -q "Remember that I love apples. What do I love?"
```

You should see `mnemosyne_remember` and `mnemosyne_recall` calls succeed.

> If `hermes mnemosyne stats` gives "invalid choice: 'mnemosyne'", the plugin
> CLI registration didn't load. Use `hermes hermes-mnemosyne stats` as a
> fallback, or re-run Step 2 to relink.

## MCP vs. Provider Plugin

Mnemosyne ships an MCP server (`mnemosyne mcp`, stdio + SSE + Streamable HTTP
transports) that exposes **35 tools** — usable with any MCP-compatible client
(Claude Desktop, etc.):

```bash
mnemosyne mcp                              # stdio transport
mnemosyne mcp --transport sse --port 8080  # SSE transport
mnemosyne mcp --transport streamable-http --port 8080  # native MCP http transport
```

**For Hermes, prefer the provider plugin over MCP.** The provider plugin
gives deeper integration that MCP cannot: the `pre_llm_call` context
injection hook, `on_session_start` initialization, `post_tool_call` memory
capture, and the `hermes mnemosyne` CLI subcommands. MCP is a generic
fallback for non-Hermes agents.

## Switching Back

```bash
hermes memory off          # Disable external provider, revert to built-in
hermes memory setup        # Or use the interactive picker
```

Or manually:

```bash
hermes config set memory.provider memory
```

Then restart Hermes.

## CLI Commands

```bash
hermes mnemosyne stats                # Current session stats
hermes mnemosyne stats --global       # Stats across all sessions
hermes mnemosyne inspect "query"      # Search memories
hermes mnemosyne sleep                # Run consolidation (working → episodic)
hermes mnemosyne export --output backup.json
hermes mnemosyne import --input backup.json
hermes mnemosyne clear                # Clear scratchpad
hermes mnemosyne version              # Show version
```

## Data Location

```
~/.hermes/mnemosyne/
└── data/
    ├── mnemosyne.db              # Main SQLite database (WAL mode)
    ├── triples.db                # Standalone TripleStore
    └── banks/<name>/mnemosyne.db # Named memory banks (per-bank isolation)
```

Persists across sessions via `~/.hermes/` (including on ephemeral VMs like Fly.io).

## Optional: Host LLM Routing

Mnemosyne's consolidation (`sleep`) and fact extraction can use a local GGUF or
a remote OpenAI-compatible API. Hermes users with OAuth-backed providers
(e.g. `openai-codex`) can route those LLM calls through Hermes' authenticated
auxiliary client instead — no extra credentials needed:

```bash
export MNEMOSYNE_HOST_LLM_ENABLED=true
```

See `docs/hermes-llm-integration.md` for the full behavior model and config.

## Troubleshooting

| Symptom | Cause / Fix |
|---------|-------------|
| `hermes memory status` shows built-in only | Provider not loaded; restart Hermes after install. |
| Plugin listed but `unavailable` | Missing Python deps in Hermes' venv; `pip install mnemosyne-hermes` in that venv. |
| `hermes mnemosyne stats` → "invalid choice" | Plugin CLI registration didn't load; use `hermes hermes-mnemosyne stats` or relink (Step 2). |
| Memory not recalled across sessions | Provider loaded but session didn't restart; new sessions pick up the provider. |
| `mnemosyne_hermes` import error in Docker | Use wrapper mode: `mnemosyne-hermes install --mode wrapper --python <venv>/bin/python`. |

## References

- [Hermes Integration Guide](../../docs/hermes-integration.md) — canonical setup reference
- [Hermes Auxiliary LLM Integration](../../docs/hermes-llm-integration.md) — host LLM routing
- [Mnemosyne vs Hindsight Comparison](../../docs/comparison.md) — architecture and feature comparison
- `hermes-agent` skill — general Hermes setup, config, and plugin system
