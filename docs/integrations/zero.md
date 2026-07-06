# Mnemosyne + Zero

Connect Mnemosyne to [Zero](https://github.com/Gitlawb/zero) for persistent
cross-session memory. Zero is an open-source terminal coding agent with a
plugin system, hooks, MCP client, and skills.

## Setup

1. Install Mnemosyne:

```bash
pip install mnemosyne-memory
```

2. Copy the plugin into Zero's plugin directory:

**Project-scoped** (recommended):

```bash
cp -r integrations/zero /path/to/your-project/.zero/plugins/mnemosyne
```

**User-scoped**:

```bash
mkdir -p ~/.config/zero/plugins
cp -r integrations/zero ~/.config/zero/plugins/mnemosyne
```

3. Launch Zero from the project root. The memory tools appear automatically.

## What You Get

Five tools registered in Zero's tool palette:

| Tool | Purpose |
|------|---------|
| `memory_remember` | Store a durable fact or preference |
| `memory_recall` | Semantic search of past memories |
| `memory_stats` | Show memory statistics |
| `memory_sleep` | Consolidate old memories into summaries |
| `memory_forget` | Delete a stale memory by ID |

Plus an `afterTool` hook that auto-captures file edits to memory.

## Usage

Ask Zero:

- "Remember that I prefer tab-width 4 for Go files"
- "What do you know about my development preferences?"
- "Recall any notes about the deployment pipeline"

## How It Works

The plugin uses Zero's plugin tool system: each tool is a shell script that
receives JSON arguments on stdin and writes results to stdout. The
`${AGENT_PLUGIN_ROOT}` placeholder in the manifest expands to the plugin's
install directory at activation time. Scripts call the `mnemosyne` CLI, which
stores everything in a local SQLite database with vector + FTS5 hybrid search.

The `afterTool` hook fires after every tool call, receives the tool name,
status, and changed files as JSON on stdin, and stores a compact memory of
file-editing actions. It skips read-only tools and failed operations.

## Optional: Full MCP Server

For the complete 35-tool surface (triples, graph traversal, scratchpad, sync),
add the Mnemosyne MCP server to your Zero config:

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

## Source Code

<https://github.com/mnemosyne-oss/mnemosyne/tree/main/integrations/zero>
