# Mnemosyne + Hermes Agent

Mnemosyne is the native memory provider for Hermes Agent. Two integration paths:

## Shared remote MCP

For a NAS/VPS service shared with Codex and Claude Code, follow the [Docker client setup](../../deploy/docker/README.md#hermes). It uses `mcp_servers`, an HTTP URL, per-agent bearer headers and optional private CA trust. This is separate from the native local memory provider below.

## Path 1: Local MCP

Add to your Hermes `config.yaml`:

```yaml
mcp_servers:
  mnemosyne:
    command: mnemosyne
    args: ["mcp"]
```

Tools register as native Hermes commands.

## Path 2: Hermes Plugin (built-in)

```bash
pip install mnemosyne-hermes
mnemosyne-hermes install                  # creates plugin symlink for Hermes discovery
hermes config set memory.provider mnemosyne
hermes memory setup
```

This gives you the **20 tools** declared in `plugin.yaml` (remember, recall, forget, stats, knowledge graph ops, scratchpad, export/import, sync), plus the persona and canonical tools the provider implements beyond that declaration. All native Hermes commands. See the [generated tool schema reference](../api/tool-schema.mdx) for the full surface.

## Usage

In Hermes, use the built-in commands:
- `mnemosyne_remember` — Store a memory
- `mnemosyne_recall` — Search memories
- `mnemosyne_forget` — Remove a memory
- `mnemosyne_stats` — View memory statistics
- `mnemosyne_sleep` — Run consolidation
