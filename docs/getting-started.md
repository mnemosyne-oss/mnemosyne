# Getting Started

## Requirements

- Python 3.10 or later
- (Optional) Hermes Agent Framework for plugin integration

## Installation

### From PyPI (recommended)

```bash
pip install mnemosyne-memory
```

With all optional features (dense retrieval via fastembed + local LLM consolidation):

```bash
pip install mnemosyne-memory[all]
```

**Ubuntu 24.04 / Debian 12:** If `pip install` fails with `externally-managed-environment`, use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install mnemosyne-memory[all]
```

### From Source (for contributors)

```bash
git clone https://github.com/mnemosyne-oss/mnemosyne.git
cd mnemosyne
pip install -e ".[all,dev]"
```

### Hermes Provider

```bash
pip install mnemosyne-hermes
hermes config set memory.provider mnemosyne
hermes gateway restart
```

No deploy script needed. The pip package handles everything.

## Your First Memory

```python
from mnemosyne import remember, recall

# Store a fact
remember("User prefers dark mode interfaces", importance=0.9, source="preference")

# Store a global fact (visible in every session)
remember("User email is alice@example.com", importance=0.95, source="profile", scope="global")

# Store temporary data with expiry
remember("API key: sk-abc123", importance=0.8, source="credential", valid_until="2026-12-31T00:00:00")

# Search memories
results = recall("interface preferences", top_k=3)
for r in results:
    print(r["content"])
```

## Verify Installation

```python
from mnemosyne import get_stats
print(get_stats())
```

## Next Steps

- [Architecture](architecture.md) — understand how BEAM tiers work
- [API Reference](api-reference.md) — full Python API documentation
- [Configuration](configuration.md) — tune performance and storage
- [Hermes Integration](hermes-integration.md) — use as a Hermes memory backend
