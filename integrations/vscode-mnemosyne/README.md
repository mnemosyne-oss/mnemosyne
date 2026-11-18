# Mnemosyne for VS Code

Browse, search, and manage Mnemosyne memory directly in VS Code.

## Features

- **Memory sidebar** — View recent memories without leaving your editor
- **Search** — Full-text search across all stored memories
- **Store** — Save important context to Mnemosyne with a command
- **Stats** — See memory counts per tier at a glance
- **Auto-refresh** — Memories update on file save (configurable)

## Requirements

- [Mnemosyne](https://github.com/mnemosyne-oss/mnemosyne) installed (`pip install mnemosyne-memory`)
- Python 3.9+ available on PATH

## Extension Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `mnemosyne.dbPath` | `~/.hermes/mnemosyne/data/default.db` | Path to the Mnemosyne database |
| `mnemosyne.autoRefresh` | `true` | Refresh memories on file save |

## Commands

| Command | Description |
|---------|-------------|
| `Mnemosyne: Refresh Memories` | Refresh the memory list |
| `Mnemosyne: Search Memories` | Search across all memories |
| `Mnemosyne: Store a Memory` | Save a new memory |
| `Mnemosyne: Open Mnemosyne MCP Config` | Open the MCP config file |

## For Developers

```bash
git clone https://github.com/mnemosyne-oss/mnemosyne.git
cd integrations/vscode-mnemosyne
npm install
npm run compile
```

Then press F5 in VS Code to launch the extension host.
