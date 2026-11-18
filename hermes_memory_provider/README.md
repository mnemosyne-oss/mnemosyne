# Mnemosyne Memory Provider for Hermes

**Deprecated installation path:** this is the legacy `hermes_memory_provider`
implementation, retained for existing imports. Do not use it for new Hermes
installations. The supported integration is the standalone `mnemosyne-hermes`
package; follow the [canonical Hermes setup guide](../docs/hermes-integration.md#setup).

That guide covers package prerequisites, install-mode selection, activation,
restart and verification. Choose the mode for your deployment: local mutable
environments and persistent side-venv wrappers have different requirements.
Container-specific paths are not defaults for every Hermes installation.

This documentation change does not remove the legacy package or change its
imports, runtime behavior or tool schemas. Future removal and compatibility
boundaries are tracked in [#655](https://github.com/mnemosyne-oss/mnemosyne/issues/655);
no removal date or future shim is promised here.

## Legacy integration reference

The remaining sections describe the retained legacy implementation, not a
supported installation procedure or a guarantee about the standalone package:

- **System prompt injection** — `# Mnemosyne Memory` header in every prompt
- **Pre-turn prefetch** — Relevant memories injected via `<memory-context>` fence before each API call
- **Post-turn sync** — User and assistant messages automatically stored to episodic memory
- **Tool dispatch** — Memory tools exposed through the provider
- **CLI commands** — `hermes mnemosyne {stats|sleep|version|inspect|clear|export|import}`
- **Setup wizard** — Listed in `hermes memory setup`


## Legacy configuration reference

The former direct source-checkout symlink installation is obsolete and is not
reproduced here. For supported installation and legacy-link handling, use the
[canonical guide](../docs/hermes-integration.md#migrating-from-the-legacy-mnemosyne-install-route).
Do not delete or replace an existing plugin target based on this legacy README.

The retained legacy provider reads settings under `memory.mnemosyne` in Hermes'
`config.yaml`, for example:

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

## Verification

Use the [current verification steps](../docs/hermes-integration.md#step-4-verify-the-active-provider)
for the supported standalone integration. Provider status alone does not prove
end-to-end memory writes or a completed data migration.

## Legacy layout reference (not an installation instruction)

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

## Legacy tool examples (not an exhaustive inventory)

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

## Existing installations

Consult the canonical guide before changing an existing installation. This
README does not prescribe automatic plugin deletion, package downgrades or
memory-data migration.
