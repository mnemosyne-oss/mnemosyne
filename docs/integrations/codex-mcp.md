# Mnemosyne + OpenAI Codex CLI

Connect Codex CLI to Mnemosyne for long-term memory across coding sessions.

## Setup

1. Install Mnemosyne with MCP support:

```bash
pip install "mnemosyne-memory[mcp]"
```

2. Add to `.codex/mcp.json` in your project root:

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

3. Restart Codex CLI. Tools appear automatically.

## Remote Streamable HTTP

For a running Mnemosyne service, use its Streamable HTTP endpoint rather than
spawning a local stdio process. Put the service behind a TLS-terminating reverse
proxy or private tunnel; the bearer token must not travel over plain HTTP. Start
the service with a token when it is bound outside loopback:

```bash
export MNEMOSYNE_MCP_TOKEN="replace-with-a-random-secret"
mnemosyne mcp \
  --transport streamable-http --host 0.0.0.0 --port 8080
```

Then configure Codex with the `/mcp` URL and the environment variable holding
the same bearer token:

```toml
[mcp_servers.mnemosyne]
url = "https://mnemosyne.example.com/mcp"
bearer_token_env_var = "MNEMOSYNE_MCP_TOKEN"
```

## Usage

Ask Codex:
- "Remember my preferred test framework is pytest"
- "Recall our discussion about migration strategy"
- "What preferences do you have stored for me?"

## Memory Banks

Codex projects can use separate memory banks to keep context isolated per project:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "command": "mnemosyne",
      "args": ["mcp", "--bank", "codex-{{project-name}}"],
      "env": {}
    }
  }
}
```
