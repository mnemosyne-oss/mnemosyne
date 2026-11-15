# Mnemosyne + Codex MCP

For shared remote memory, follow the [Docker client setup](../../deploy/docker/README.md#codex). Codex uses TOML configuration with `[mcp_servers.mnemosyne]`, `url` and `bearer_token_env_var` for remote Streamable HTTP; `.codex/mcp.json` is not the configuration used by this recipe.

For local stdio, install `mnemosyne-memory[mcp]` and configure:

```toml
[mcp_servers.mnemosyne]
command = "mnemosyne"
args = ["mcp"]
```

Ask Codex to use the Mnemosyne remember and recall tools. Explicitly select the intended memory bank if organizing separate projects; bank names are not authorization boundaries.

Source: [official Codex MCP documentation](https://developers.openai.com/codex/mcp/). See the deployment verification record for exact tested versions.
