# Mnemosyne Memory Provider for Hermes

Deploy Mnemosyne as a **first-class MemoryProvider** through Hermes' plugin system.

## What This Gives You

When deployed, Mnemosyne gets the **same integration tier** as Honcho, mem0, and supermemory:

- **System prompt injection** — `# Mnemosyne Memory` header in every prompt
- **Pre-turn prefetch** — Relevant memories injected via `<memory-context>` fence before each API call
- **Post-turn sync** — User and assistant messages automatically stored to episodic memory
- **Tool dispatch** — 15 memory tools auto-injected into the model's tool surface
- **CLI commands** — `hermes mnemosyne {stats|sleep|version|inspect|clear|export|import}`
- **Setup wizard** — Listed in `hermes memory setup`

**All of this without touching Hermes core.** Deployed purely through the plugin directory.

## Deploy

```bash
# One-time setup: symlink into Hermes plugin directory
ln -s $(pwd)/hermes_memory_provider ~/.hermes/plugins/mnemosyne

# Activate in config
hermes config set memory.provider mnemosyne
```

Or manually edit `~/.hermes/config.yaml`:

```yaml
memory:
  provider: mnemosyne
  mnemosyne:
    auto_sleep: true
    sleep_threshold: 50
    vector_type: float32  # float32 | int8 | bit
    ignore_patterns:
      - "^\\s*\\$\\s*pip\\s"
    write_classifier: "off"  # off | warn | strict
```

The corresponding environment variables are `MNEMOSYNE_IGNORE_PATTERNS`
(newline-separated regular expressions, empty by default) and
`MNEMOSYNE_WRITE_CLASSIFIER` (`off` by default). For these settings, an explicit
`initialize(...)` keyword argument takes precedence. Without that override,
the provider resolves Hermes `config.yaml` `memory.mnemosyne.*` settings before
the corresponding core `config.yaml` settings, then the environment variable
and default. `off` still applies `ignore_patterns`; `warn` runs the noise/secret
classifier but stores classified content with warnings; and `strict` rejects
content classified as noise or secret-like. Unset, blank, or invalid classifier
values fall back to `off` (invalid values also log a warning).

## Verify

```bash
hermes memory status    # Should show "mnemosyne" as active provider
hermes mnemosyne stats  # Show memory statistics
```

## Architecture

```
~/.hermes/plugins/mnemosyne/   ← symlink to hermes_memory_provider/
├── __init__.py                  ← MnemosyneMemoryProvider (MemoryProvider ABC)
├── cli.py                       ← hermes mnemosyne subcommands
├── plugin.yaml                  ← Manifest for discovery
└── README.md                    ← This file
```

The provider is discovered by `plugins.memory.discover_memory_providers()` which scans:
1. Bundled providers: `hermes-agent/plugins/memory/<name>/`
2. **User plugins: `$HERMES_HOME/plugins/<name>/`** ← This is where Mnemosyne lives

User plugins take precedence over bundled plugins on name collision.

## Tools (Auto-Injected)

| Tool | Purpose |
|------|---------|
| `mnemosyne_remember` | Store durable memory with importance, scope, expiry |
| `mnemosyne_recall` | Hybrid search (50% vector + 30% FTS + 20% importance) |
| `mnemosyne_stats` | Show working + episodic counts |
| `mnemosyne_triple_add` | Add temporal facts to the knowledge graph |
| `mnemosyne_triple_query` | Query temporal knowledge graph facts |
| `mnemosyne_sleep` | Consolidate working → episodic memory |
| `mnemosyne_scratchpad_write` | Write short-lived scratchpad context |
| `mnemosyne_scratchpad_read` | Read scratchpad context |
| `mnemosyne_scratchpad_clear` | Clear scratchpad context |
| `mnemosyne_invalidate` | Mark memory as expired/superseded |
| `mnemosyne_export` | Export memories for backup or migration |
| `mnemosyne_import` | Import memories from backup files or providers |
| `mnemosyne_update` | Update an existing memory |
| `mnemosyne_forget` | Delete a memory |
| `mnemosyne_diagnose` | Run diagnostics on the memory store |

## Undeploy

```bash
rm ~/.hermes/plugins/mnemosyne
hermes config set memory.provider null
```

Hermes falls back to built-in memory only.
