<div align="center">

<img src="/assets/mnemosyne.jpg" alt="Mnemosyne" width="40%">

# Mnemosyne

*Native, zero-cloud memory for AI agents. SQLite-backed. Sub-millisecond. Fully private.*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/mnemosyne-memory.svg?v=3.0.0)](https://pypi.org/project/mnemosyne-memory/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/AxDSan/mnemosyne/actions/workflows/ci.yml/badge.svg)](https://github.com/AxDSan/mnemosyne/actions/workflows/ci.yml)
[![BEAM](https://img.shields.io/badge/BEAM-ICLR%202026-purple.svg)](https://beam-benchmark.github.io/)
[![Community](https://img.shields.io/badge/Community-100%25-brightgreen)](CODE_OF_CONDUCT.md)
[![Discord](https://badgen.net/discord/online-members/29ZszXTgY3)](https://discord.gg/Cgzpw9x3R)

</div>

Mnemosyne is a local-first memory system for the [Hermes Agent](https://github.com/NousResearch/hermes-agent) framework. It stores conversations, preferences, and knowledge in SQLite with native vector search (sqlite-vec) and full-text search (FTS5) -- no external databases, no API keys, no network calls.

---

## Table of Contents

- [Benchmark](#benchmark)
- [Quick Start](#quick-start)
- [CLI Usage](#cli-usage)
- [Python API](#python-api)
- [Hermes Plugin](#hermes-plugin)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Support](#support)

---

## Benchmark

Mnemosyne v3 scores **65.2%** on the [BEAM](https://github.com/mohammadtavakoli78/BEAM) long-context memory benchmark (ICLR 2026) at 100K scale -- competitive with cloud alternatives while running fully offline.

| Scale | Mnemosyne v3 | Honcho | Hindsight | LIGHT | RAG |
|-------|-------------|--------|-----------|-------|-----|
| **100K** | **65.2%** | 63.0% | 73.4% | 35.8% | 32.3% |

Per-ability (100K): IE 91.5% · MR 87.5% · TR 75.0% · ABS 100.0% · CR 50.0% · KU 50.0% · EO 25.0% · IF 62.5% · PF 54.5% · SUM 55.6%

Full report: [docs/beam-benchmark.md](docs/beam-benchmark.md) | Run locally: `python tools/evaluate_beam_end_to_end.py --sample 5 --scales 100K`

---

## Quick Start

### Option A: PyPI (recommended)

```bash
pip install mnemosyne-memory

# With optional features (dense retrieval + local LLM)
pip install mnemosyne-memory[all]
```

### Option B: Source (for development)

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e ".[all,dev]"
```

### Option C: Hermes MemoryProvider only (no pip)

```bash
curl -sSL https://raw.githubusercontent.com/AxDSan/mnemosyne/main/deploy_hermes_provider.sh | bash
```

### Register with Hermes

```bash
python -m mnemosyne.install
hermes memory setup      # Select "mnemosyne"
hermes memory status     # Verify it's active
```

> **Ubuntu 24.04 / Debian 12 users:** If pip fails with `externally-managed-environment`, install into the Hermes runtime venv:
> ```bash
> $HOME/.hermes/hermes-agent/venv/bin/python -m pip install "mnemosyne-memory[all]"
> $HOME/.hermes/hermes-agent/venv/bin/python -m mnemosyne.install
> ```

---

## CLI Usage

```bash
# Memory statistics (current session)
hermes mnemosyne stats

# Memory statistics (all sessions)
hermes mnemosyne stats --global

# Search memories
hermes mnemosyne inspect "dark mode preferences"

# Run consolidation
hermes mnemosyne sleep

# Export / import
hermes mnemosyne export --output backup.json
hermes mnemosyne import --input backup.json

# Import from another provider (Mem0, Letta, Zep, Cognee, Honcho, SuperMemory, Hindsight)
hermes mnemosyne import --from mem0 --api-key sk-xxx
hermes mnemosyne import --from hindsight --file hindsight-export.json

# Clear scratchpad
hermes mnemosyne clear
```

### MCP Server

```bash
mnemosyne mcp                          # stdio (Claude Desktop)
mnemosyne mcp --transport sse --port 8080  # SSE (web clients)
```

Available tools: `mnemosyne_remember`, `mnemosyne_recall`, `mnemosyne_sleep`, `mnemosyne_scratchpad_read`, `mnemosyne_scratchpad_write`, `mnemosyne_get_stats`.

---

## Python API

```python
from mnemosyne import remember, recall

# Store a fact
remember("User prefers dark mode interfaces", importance=0.9, source="preference")

# Store globally (visible across all sessions)
remember("User email is user@example.com", importance=0.95, scope="global")

# Store with expiry
remember("Temp token: abc123", importance=0.8, valid_until="2026-12-31")

# Search
results = recall("interface preferences", top_k=3)

# Temporal recall (recency boost)
results = recall("deployments", temporal_weight=0.5, temporal_halflife=48.0)

# Entity extraction
remember("Met with Abdias about the v2 release", extract_entities=True)

# LLM-driven fact extraction
remember("User said they prefer Python for backend work", extract=True)

# Temporal triples
from mnemosyne.core.triples import TripleStore
kg = TripleStore()
kg.add("Maya", "assigned_to", "auth-migration", valid_from="2026-01-15")
kg.query("Maya", as_of="2026-02-01")

# Memory banks (per-domain isolation)
from mnemosyne.core.banks import BankManager
BankManager().create_bank("work")
work_mem = Mnemosyne(bank="work")
work_mem.remember("Sprint review on Friday")
```

### Advanced: BEAM Direct Access

```python
from mnemosyne.core.beam import BeamMemory

beam = BeamMemory(session_id="my_session")
beam.remember("Important context", importance=0.9)
beam.consolidate_to_episodic(summary="User likes Neovim", source_wm_ids=["wm1"])
results = beam.recall("editor preferences", top_k=5)
```

---

## Hermes Plugin

When registered as a Hermes plugin, Mnemosyne exposes **17 tools** for full memory lifecycle management:

| # | Tool | Description |
|---|------|-------------|
| 1 | `mnemosyne_remember` | Store a memory with importance, source, expiry, scope |
| 2 | `mnemosyne_recall` | Hybrid vector + FTS5 search |
| 3 | `mnemosyne_stats` | BEAM tier statistics |
| 4 | `mnemosyne_triple_add` | Add temporal triple with validity dates |
| 5 | `mnemosyne_triple_query` | Query temporal knowledge graph |
| 6 | `mnemosyne_sleep` | Run consolidation cycle |
| 7 | `mnemosyne_scratchpad_write` | Write to scratchpad |
| 8 | `mnemosyne_scratchpad_read` | Read scratchpad |
| 9 | `mnemosyne_scratchpad_clear` | Clear scratchpad |
| 10 | `mnemosyne_invalidate` | Mark memory expired/superseded |
| 11 | `mnemosyne_export` | Export to JSON |
| 12 | `mnemosyne_update` | Update memory content/importance |
| 13 | `mnemosyne_forget` | Delete memory by ID |
| 14 | `mnemosyne_import` | Import from JSON or 7 providers |
| 15 | `mnemosyne_diagnose` | PII-safe diagnostics |
| 16 | `mnemosyne_graph_query` | Multi-hop graph traversal |
| 17 | `mnemosyne_graph_link` | Declare semantic edges |

Plus three lifecycle hooks (`pre_llm_call`, `on_session_start`, `post_tool_call`) for automatic context injection.

---

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Hermes     │────▶│  Mnemosyne   │────▶│  SQLite            │
│  Agent      │     │  BEAM        │     │  working_mem       │
└──────▲──────┘     └──────────────┘     │  episodic_mem      │
       │                                  │  vec_episodes      │
       └── Auto-injected context ────────▶│  fts_episodes      │
                                          │  scratchpad        │
                                          │  triples           │
                                          └───────────────────┘
```

**BEAM** (Bilevel Episodic-Associative Memory):
- **Working memory** -- Hot context, auto-injected before LLM calls, TTL-based eviction
- **Episodic memory** -- Long-term storage with sqlite-vec + FTS5 hybrid search
- **Scratchpad** -- Temporary agent reasoning workspace

**Hybrid scoring:** 50% vector similarity + 30% FTS5 rank + 20% importance, all inside SQLite.

**Binary vectors:** Information-theoretic binarization (MIB) compresses 384-dim float32 embeddings into 48 bytes -- 32x reduction. Retrieval uses Hamming distance entirely within SQLite. No ANN indices, no external vector DB.

**MEMORIA Fact Engine:** Structured fact extraction at ingestion. Temporal triples with version chains, previous-value tracking, valid-from/to windows. 10 recall strategies dispatched by question type. Recursive gap analysis re-queries until answers are found.

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data` | Database directory |
| `MNEMOSYNE_VEC_TYPE` | `int8` | Vector compression: `float32`, `int8`, or `bit` |
| `MNEMOSYNE_VEC_WEIGHT` | `0.5` | Vector similarity weight |
| `MNEMOSYNE_FTS_WEIGHT` | `0.3` | FTS5 keyword weight |
| `MNEMOSYNE_IMPORTANCE_WEIGHT` | `0.2` | Importance weight |
| `MNEMOSYNE_WM_MAX_ITEMS` | `10000` | Working memory limit |
| `MNEMOSYNE_RECENCY_HALFLIFE` | `168` | Decay halflife in hours |
| `MNEMOSYNE_LLM_ENABLED` | `true` | LLM summarization in sleep cycle |
| `MNEMOSYNE_LLM_BASE_URL` | *(none)* | Remote OpenAI-compatible LLM endpoint |
| `MNEMOSYNE_HOST_LLM_ENABLED` | `false` | Route through Hermes' authenticated provider |

Full reference: [docs/configuration.md](docs/configuration.md)

### config.yaml

```yaml
memory:
  mnemosyne:
    auto_sleep: true              # Auto-consolidate on session boundaries
    sleep_threshold: 50           # Min working memories before consolidation
    vector_type: int8             # float32, int8, or bit
    ignore_patterns:
      - "^pip install"
      - "^git "
      - "^sudo "
      - "^Traceback"
      - "^Error:"
```

---

## Contributing

Contributions welcome. Focus areas:

- Encrypted cloud sync (optional, user-controlled)
- Additional embedding models
- Multi-language support

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Full docs: [`docs/`](docs/README.md) | Changelog: [`CHANGELOG.md`](CHANGELOG.md) | Releases: [GitHub Releases](https://github.com/AxDSan/mnemosyne/releases)

---

## Support

<div align="center">

**Discord:** [Join the Mnemosyne community](https://discord.gg/Cgzpw9x3R)
**Issues:** [GitHub Issues](https://github.com/AxDSan/mnemosyne/issues)
**Email:** abdi.moya@gmail.com

<a href="https://github.com/sponsors/AxDSan"><img src="https://img.shields.io/badge/💖_GitHub_Sponsors-30363D?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Sponsors"/></a>
<a href="https://ko-fi.com/axdsan"><img src="https://img.shields.io/badge/☕_Ko‑fi-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white" alt="Ko-fi"/></a>

⭐ **Star the repo if you find it useful!**

</div>

---

## License

MIT License -- See [LICENSE](LICENSE)

Copyright (c) 2026 Abdias J

---

<p align="center">
  <em>"The faintest ink is more powerful than the strongest memory." &mdash; Hermes Trismegistus</em>
</p>
