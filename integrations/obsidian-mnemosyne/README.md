# Mnemosyne Sync — Obsidian Plugin

Sync your Mnemosyne AI agent memories as markdown notes in your Obsidian vault.

## Features

- **Memory sync** — Export all Mnemosyne memories as markdown files with YAML frontmatter
- **Searchable** — Each memory is a full markdown note, searchable with Obsidian's built-in search
- **Smart merge** — Won't overwrite notes you've edited
- **Auto-sync** — Configurable sync interval (5-120 minutes)
- **Filtering** — Option to exclude assistant messages
- **Ribbon icon** — One-click sync from the sidebar

## Installation

### From Obsidian Community Store (coming soon)

1. Open Obsidian Settings → Community Plugins → Browse
2. Search for "Mnemosyne Sync"
3. Install and enable

### Manual

1. Download `main.js`, `manifest.json`, `styles.css` from the [releases page](https://github.com/mnemosyne-oss/mnemosyne/releases)
2. Copy to `{vault}/.obsidian/plugins/mnemosyne-sync/`
3. Enable in Community Plugins settings

## Requirements

- [Mnemosyne](https://github.com/mnemosyne-oss/mnemosyne) installed (`pip install mnemosyne-memory`)
- Python 3.9+ available on PATH
- Desktop version of Obsidian

## Settings

| Setting | Description |
|---------|-------------|
| Data directory | Path to Mnemosyne database (relative to ~) |
| Memory bank | Which memory bank to sync |
| Sync folder | Folder in vault for memory notes |
| Auto-sync | Sync on an interval |
| Sync interval | Minutes between auto-syncs |
| Include assistant | Sync AI responses too (default: user only) |
