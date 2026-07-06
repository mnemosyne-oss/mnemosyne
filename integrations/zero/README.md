<div align="center">

# Mnemosyne for Zero

*Persistent cross-session memory for [Zero](https://github.com/Gitlawb/zero). 5 tools. 1 hook. Zero cloud.*

[![Zero](https://img.shields.io/badge/Zero-0.2+-blue.svg)](https://github.com/Gitlawb/zero)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mnemosyne-oss/mnemosyne/blob/main/LICENSE)

</div>

**Mnemosyne** gives Zero a local-first memory layer that stores facts,
preferences, and project context across sessions — then surfaces them with
hybrid vector + FTS5 search. SQLite on your machine. No cloud. No API keys.

---

## The Problem

Zero is a terminal coding agent. Like all agents, it loses context between
sessions: preferences vanish, decisions are forgotten, the same bugs get
re-investigated. Zero has no built-in cross-session memory — every `zero`
launch starts tabula rasa.

## What Mnemosyne Changes

This plugin adds five memory tools and an auto-capture hook to Zero:

- **memory_remember** — store a durable fact or preference
- **memory_recall** — semantic search of past memories
- **memory_stats** — show memory statistics
- **memory_sleep** — consolidate old memories into episodic summaries
- **memory_forget** — delete a stale memory by ID
- **afterTool hook** — auto-captures file edits so the agent recalls what it
  changed in future sessions

The agent calls `memory_recall` before asking the user to repeat themselves,
and `memory_remember` when it learns a stable fact. The hook silently records
file-edit activity in the background.

## How It Works

### Plugin Tool Contract

Zero's plugin system (`internal/plugins/activate.go`) executes each tool's
`command` as a subprocess: JSON-encoded arguments on stdin, stdout/stderr
captured as the tool result. The `${AGENT_PLUGIN_ROOT}` placeholder in the
manifest expands to the plugin's install directory at activation time, so
scripts resolve correctly regardless of the agent's workspace cwd.

Each tool script reads JSON from stdin (via `jq`), calls the `mnemosyne` CLI,
and writes the result to stdout. Exit code 0 = success; non-zero = error.

### Hook Contract

Zero's hooks system (`internal/hooks/dispatch.go`) fires shell commands on
lifecycle events. The `afterTool` hook receives a JSON payload on stdin
(tool name, status, changed files, session ID). The hook stores a compact
memory of file-editing actions and exits 0 — afterTool hooks are advisory
and never block.

## Quickstart

**Prerequisites:** Zero 0.2+, Mnemosyne CLI, `jq`.

### 1. Install Mnemosyne CLI

```bash
pip install mnemosyne-memory
```

Verify:

```bash
mnemosyne stats
```

### 2. Install the plugin

**Project-scoped** (recommended — committed to the repo, shared with the team):

```bash
cp -r integrations/zero /path/to/your-project/.zero/plugins/mnemosyne
```

**User-scoped** (follows you across projects):

```bash
mkdir -p ~/.config/zero/plugins
cp -r integrations/zero ~/.config/zero/plugins/mnemosyne
```

### 3. Verify

Launch Zero from the project root. The five `memory_*` tools appear in the
agent's tool palette automatically. Ask:

> "What do you remember about my preferences?"

The agent calls `memory_recall`, finds any stored memories, and answers
without you having to repeat yourself.

## Configuration

No required config. Memories default to `~/.hermes/mnemosyne/data/`. Override
with the `MNEMOSYNE_DATA_DIR` environment variable.

### Optional: Full MCP Server

For the complete 35-tool Mnemosyne surface (triples, graph traversal,
scratchpad, sync, canonical facts, persona management), add the MCP server to
your Zero config (`~/.config/zero/config.json` or `.zero/config.json`):

```json
{
  "mcp": {
    "servers": {
      "mnemosyne": {
        "type": "stdio",
        "command": "mnemosyne",
        "args": ["mcp"]
      }
    }
  }
}
```

This is additive — MCP tools coexist with the plugin's 5 shell tools.

## Tools

| Tool | Purpose | Permission |
|------|---------|------------|
| `memory_remember` | Store a durable memory | allow |
| `memory_recall` | Semantic search of memories | allow |
| `memory_stats` | Show memory statistics | allow |
| `memory_sleep` | Run consolidation | allow |
| `memory_forget` | Delete a memory by ID | allow |

## File Structure

```
integrations/zero/
├── plugin.json                      # Manifest: 5 tools, 1 hook, 1 skill
├── tools/
│   ├── remember.sh                  # memory_remember
│   ├── recall.sh                    # memory_recall
│   ├── stats.sh                     # memory_stats
│   ├── sleep.sh                     # memory_sleep
│   └── forget.sh                    # memory_forget
├── hooks/
│   └── after-tool.sh                # afterTool: auto-capture file edits
└── skills/
    └── mnemosyne/
        └── SKILL.md                 # Model guidance for memory usage
```

## Known Limitations

1. **Plugin skills are not yet discoverable by Zero's `skill` tool.** Zero's
   plugin loader activates tools and hooks from plugin manifests, but the skill
   loader does not yet merge plugin-declared skill paths into its discovery
   root. Copy `skills/mnemosyne/SKILL.md` to
   `~/.local/share/zero/skills/mnemosyne/SKILL.md` as a workaround.

2. **sessionStart/sessionEnd hooks are defined but not dispatched.** Zero's
   hook enum includes these events and the dispatcher supports them, but the
   agent loop only currently fires `beforeTool`/`afterTool`. Auto-injection of
   recalled memories at session start is therefore not yet possible via hooks
   — the model calls `memory_recall` explicitly. When Zero adds session hook
   dispatch, the manifest is ready: just add a `sessionStart` hook entry.

3. **The `mnemosyne` binary must be on PATH.** If installed in a venv, symlink
   it to `~/.local/bin/` or update the scripts to use the full path.

## Contributing

See the [Contributing Guidelines](https://github.com/mnemosyne-oss/mnemosyne/blob/main/CONTRIBUTING.md).

## License

MIT
