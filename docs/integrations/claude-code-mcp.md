# Mnemosyne + Claude Code MCP

For shared remote memory, follow the [Docker client setup](../../deploy/docker/README.md#claude-code), including bearer headers and local CA trust.

For local stdio, install `mnemosyne-memory[mcp]` and add a project `.mcp.json` entry:

```json
{
  "mcpServers": {
    "mnemosyne": {"command": "mnemosyne", "args": ["mcp"]}
  }
}
```

Approve the project entry through Claude Code, check `/mcp`, and ask Claude to use the remember/recall tools. MCP tools are not automatically slash commands such as `/remember`.

Source: [official Claude Code MCP documentation](https://code.claude.com/docs/en/mcp). See the deployment verification record for exact tested versions.
