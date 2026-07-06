# Mnemosyne Integrations

Mnemosyne runs everywhere. Pick your platform:

| Platform | Method | Config |
|----------|--------|--------|
| [Cursor](cursor-mcp.md) | MCP (stdio) | `.cursor/mcp.json` |
| [Claude Code](claude-code-mcp.md) | MCP (stdio) | `claude.json` |
| [OpenAI Codex CLI](codex-mcp.md) | MCP (stdio) | `.codex/mcp.json` |
| [Windsurf](windsurf-mcp.md) | MCP (stdio) | `.windsurf/mcp_config.json` |
| [OpenWebUI](openwebui-tool.md) | Native @tool | Workspace tool config |
| [Pi](pi.md) | Pi extension + skill | `pi install npm:@mnemosyne-oss/pi-mnemosyne` |
| [Hermes Agent](hermes-mcp.md) | MCP + Plugin | `~/.hermes/config.yaml` |
| [Zero](zero.md) | Plugin (tools + hooks) | `.zero/plugins/mnemosyne/` |

## Quick Start (any MCP client)

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "mnemosyne",
      "args": ["mcp"],
      "env": {}
    }
  }
}
```

Make sure the `mcp` extra is installed:

```bash
pip install "mnemosyne-memory[mcp]"
```

That's it. Three tools become available: `mnemosyne_remember`, `mnemosyne_recall`, `mnemosyne_forget`.

## Not using MCP?

Use the [Python API](../api-reference.md) directly:

```python
from mnemosyne import remember, recall
remember("User prefers dark mode")
results = recall("user preferences")
```
