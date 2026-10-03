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
| [OpenClaw](openclaw.md) | Native plugin | `pip install mnemosyne-memory[openclaw]` |

## Model providers

Mnemosyne's outbound surfaces all speak the OpenAI-compatible protocol, so any
provider that implements it works with environment variables alone.

| Provider | Surfaces | Config |
|----------|----------|--------|
| [Any OpenAI-compatible vision endpoint](openai-compatible-vision.md) | Vision | `MNEMOSYNE_MODALITY_*` |
| [Atlas Cloud](atlas-cloud.md) | Chat, embeddings, vision | `MNEMOSYNE_LLM_*`, `MNEMOSYNE_EMBEDDING_*`, `MNEMOSYNE_MODALITY_*` |

Media understanding is **off by default** — `MNEMOSYNE_MODALITY_ENABLED`
must be set before Mnemosyne makes any outbound call to describe a file.

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
