# Mnemosyne Compression Plugin

## Overview

This plugin replaces the legacy `MNEMOSYNE_USE_CAVEMAN` environment variable with a formal, configurable, and testable plugin system for memory compression in Mnemosyne.

It applies `rust_cave_001` compression to episodic memories during consolidation, but only when:
- The plugin is enabled via `mnemosyne.plugins.compression.enabled: true`
- The memory is in tier 3 (long-term, low-importance)
- The memory content exceeds `MNEMOSYNE_TIER3_MAX_CHARS` (default: 300)

## Installation

```bash
pip install mnemosyne-compression-plugin
```

## Configuration

Add to your Mnemosyne config file (e.g., `~/.hermes/mnemosyne/config.yaml`):

```yaml
mnemosyne:
  plugins:
    compression:
      enabled: true
```

## Legacy Support

For backward compatibility, if `MNEMOSYNE_USE_CAVEMAN=1` is set in the environment, the plugin will:
- Enable itself automatically
- Log a warning: `MNEMOSYNE_USE_CAVEMAN is deprecated. Use mnemosyne.plugins.compression.enabled in config instead. This will be removed in v2.1.`

## How It Works

1. During memory consolidation (`beam.py`), the plugin intercepts the summary.
2. For each memory in tier 3 with content > 300 chars:
   - Calls `rust_cave_001.compress()`
   - If compression reduces size, replaces original content
   - Logs debug message with compression ratio
   - On failure, preserves original content
3. Updates summary and continues consolidation.

## Testing

Run tests:

```bash
pytest tests/
```

## License

MIT