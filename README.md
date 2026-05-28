<div align="center">

<img src="/assets/mnemosyne.jpg" alt="Mnemosyne" width="40%">

# Mnemosyne

*Zero-dependency AI memory that works everywhere. SQLite-backed. Sub-millisecond. Fully private.*

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/mnemosyne-memory.svg?v=3.1.0)](https://pypi.org/project/mnemosyne-memory/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/AxDSan/mnemosyne/actions/workflows/ci.yml/badge.svg)](https://github.com/AxDSan/mnemosyne/actions/workflows/ci.yml)
[![BEAM](https://img.shields.io/badge/BEAM-ICLR%202026-purple.svg)](https://beam-benchmark.github.io/)
[![Discord](https://badgen.net/discord/online-members/29ZszXTgY3)](https://discord.gg/Cgzpw9x3R)
[![MCP](https://img.shields.io/badge/MCP-Ready-6366f1)](https://modelcontextprotocol.io)

</div>

**Mnemosyne** is a universal memory layer for AI agents. Drop it into **any** agent framework — Hermes, Claude Code, Cursor, Codex, OpenWebUI, OpenClaw, or your own custom agent. One `pip install`, one SQLite database, zero cloud dependencies.

---

## Table of Contents

- [Works With Everything](#works-with-everything)
- [Quick Start](#quick-start)
  - [Add to your agent](#add-to-your-agent)
- [Benchmark](#benchmark)
- [CLI Usage](#cli-usage)
- [Python API](#python-api)
  - [BEAM Direct Access](#advanced-beam-direct-access)
- [Architecture](#architecture)
- [Why Mnemosyne?](#why-mnemosyne)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [Hermes Plugin](#hermes-plugin-17-tools)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

---

## Works With Everything

| Platform | Method | Setup |
|----------|--------|-------|
| **Cursor** | MCP | Add to `.cursor/mcp.json` |
| **Claude Code** | MCP | Add to `claude.json` |
| **OpenAI Codex CLI** | MCP | Add to `.codex/mcp.json` |
| **Windsurf** | MCP | Add to `.windsurf/mcp_config.json` |
| **OpenWebUI** | Native @tool | Drop bridge file into `data/tools/` |
| **OpenClaw** | Native provider | `pip install mnemosyne-memory[openclaw]` |
| **Hermes Agent** | MCP + Plugin | Native — ships enabled |
| **Any MCP client** | MCP (stdio/SSE) | One config line |
| **Any Python agent** | Direct SDK | `import mnemosyne` |

See [docs/integrations/](docs/integrations/README.md) for complete setup guides per platform.

---

## Quick Start

```bash
pip install mnemosyne-memory

# With all features (vector search + MCP server)
pip install "mnemosyne-memory[all]"
```

### Add to your agent

**MCP-based** (Cursor, Claude Code, Codex, Windsurf):

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

**Python SDK** (any agent):

```python
from mnemosyne import remember, recall

remember("User prefers dark mode interfaces")
results = recall("user preferences")
```

**OpenWebUI:** Drop a 1-line bridge file into `data/tools/`.

**OpenClaw:** Add `provider: mnemosyne.integrations.openclaw:create_provider` to config.

---

## Benchmark

Mnemosyne v3 scores **65.2%** on the [BEAM](https://github.com/mohammadtavakoli78/BEAM) long-context memory benchmark (ICLR 2026) at 100K scale — competitive with cloud alternatives while running fully offline, all in a single SQLite file.

| Scale | Mnemosyne v3 | Honcho | Hindsight | LIGHT | RAG |
|-------|-------------|--------|-----------|-------|-----|
| **100K** | **65.2%** | 63.0% | 73.4% | 35.8% | 32.3% |

Per-ability (100K): IE 91.5% · MR 87.5% · TR 75.0% · ABS 100.0% · CR 50.0% · KU 50.0% · EO 25.0% · IF 62.5% · PF 54.5% · SUM 55.6%

Full report: [docs/beam-benchmark.md](docs/beam-benchmark.md)

---

## CLI Usage

```bash
# MCP server (works with any MCP client)
mnemosyne mcp                          # stdio (default)
mnemosyne mcp --transport sse --port 8080  # SSE (web clients)

# Direct memory ops
mnemosyne remember "User likes dark mode"
mnemosyne recall "preferences"
mnemosyne stats
mnemosyne sleep                         # Run consolidation

# Export / import
mnemosyne export --output backup.json
mnemosyne import --input backup.json
```

---

## Python API

```python
from mnemosyne import remember, recall

# Store a fact
remember("User prefers dark mode interfaces",
         importance=0.9, source="preference")

# Store globally (visible across all sessions)
remember("User email is user@example.com",
         importance=0.95, scope="global")

# Store with expiry
remember("Temp token: abc123",
         importance=0.8, valid_until="2026-12-31")

# Search
results = recall("interface preferences", top_k=3)

# Temporal recall (recency boost)
results = recall("deployments",
                 temporal_weight=0.5, temporal_halflife=48.0)

# Entity extraction
remember("Met with Abdias about the v2 release",
         extract_entities=True)

# LLM-driven fact extraction
remember("User said they prefer Python for backend work",
         extract=True)

# Temporal triples (knowledge graph)
from mnemosyne.core.triples import TripleStore
kg = TripleStore()
kg.add("Maya", "assigned_to", "auth-migration",
       valid_from="2026-01-15")
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
beam.consolidate_to_episodic(
    summary="User likes Neovim",
    source_wm_ids=["wm1"]
)
results = beam.recall("editor preferences", top_k=5)
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Any AI Agent                            │
│  (Hermes · Claude Code · Cursor · Codex · OpenWebUI · MCP) │
└────────────────────────┬───────────────────────────────────┘
                         │ MCP / SDK / Plugin
┌────────────────────────▼───────────────────────────────────┐
│                      Mnemosyne BEAM                         │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Working    │  │ Episodic     │  │ TripleStore         │   │
│  │ Memory     │──▶│ Memory       │  │ (Temporal KG)      │   │
│  │ (hot ctx)  │  │ (long-term)  │  └────────────────────┘   │
│  └────────────┘  └──────┬───────┘                           │
│                         │                                    │
│              ┌──────────▼──────────┐                        │
│              │     SQLite DB       │                        │
│              │  (single file)      │                        │
│              │  sqlite-vec + FTS5  │                        │
│              │  MIB binary vectors │                        │
│              └─────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

**BEAM** (Bilevel Episodic-Associative Memory):
- **Working memory** — Hot context, auto-injected before LLM calls, TTL-based eviction
- **Episodic memory** — Long-term storage with sqlite-vec + FTS5 hybrid search
- **TripleStore** — Temporal knowledge graph with version chains

**Hybrid scoring:** 50% vector similarity + 30% FTS5 rank + 20% importance, all inside SQLite.

**Binary vectors:** Information-theoretic binarization (MIB) compresses 384-dim float32 embeddings into 48 bytes — 32x reduction. Hamming distance entirely within SQLite. No ANN indices, no external vector DB.

---

## Why Mnemosyne?

| Feature | Mnemosyne | mem0 | Letta | Honcho | SuperMemory | Hindsight | ChromaDB |
|---------|-----------|------|-------|--------|-------------|-----------|----------|
| **Local-first** | ✅ SQLite | ⚠️ Hybrid | ❌ Docker+PG | ⚠️ PG+worker | ❌ SaaS | ✅ SQLite | ✅ Embedded |
| **Zero deps** | ✅ pip only | ❌ Qdrant/PG | ❌ PG+vector | ❌ PG+3 LLMs | ❌ SaaS infra | ✅ pip only | ✅ pip only |
| **MCP server** | ✅ Built-in | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| **Python SDK** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Multi-platform** | ✅ 8+ targets | ⚠️ 3 adapters | ❌ Agent-only | ⚠️ 4 adapters | ✅ MCP | ❌ Agent-only | ❌ Library only |
| **Open source** | ✅ MIT | ✅ Apache 2.0 | ✅ OSS | ⚠️ AGPL | ❌ Proprietary | ✅ MIT | ✅ Apache 2.0 |
| **Benchmark** | **65.2% BEAM** | 49% LongMem | 83.2% LoCoMo | **90.4% LongMem** | 85.2% MemoryBench | 73.4% BEAM | N/A (vector DB) |
| **Self-hosted** | ✅ Yes | ✅ Optional | ✅ Optional | ✅ Yes | ❌ Enterprise | ✅ Yes | ✅ Yes |
| **Integration template** | ✅ Published | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Memory architecture** | BEAM (3-tier) | Session + facts | OS-virtual context | Peer + reasoning | 5-layer stack | Episodic + semantic | Vector store only |
| **Purpose** | Full memory system | Memory API | Agent runtime | Managed memory | Consumer + agent | Research memory | Vector database |

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

| `MNEMOSYNE_EMBEDDING_API_URL` | `${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}` | Preferred name for custom embedding API endpoint (OpenAI-compatible). Falls back to `OPENROUTER_BASE_URL`. |
| `MNEMOSYNE_EMBEDDING_API_KEY` | `${OPENROUTER_API_KEY:-${OPENAI_API_KEY:-}}` | Preferred name for embedding API key. Falls back to `OPENROUTER_API_KEY`, then `OPENAI_API_KEY`. |

Full reference: [docs/configuration.md](docs/configuration.md)

---

## Hermes Plugin (17 tools)

When used with Hermes Agent, Mnemosyne exposes 17 tools for memory lifecycle management — 3 lifecycle hooks (`pre_llm_call`, `on_session_start`, `post_tool_call`) for automatic context injection, plus full MCP support.

See [docs/hermes-integration.md](docs/hermes-integration.md).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Full docs: [`docs/`](docs/README.md) · Changelog: [`CHANGELOG.md`](CHANGELOG.md) · Releases: [GitHub Releases](https://github.com/AxDSan/mnemosyne/releases) · Integrations: [docs/integrations/](docs/integrations/README.md)

---

## Support

<div align="center">

**Discord:** [Join the Mnemosyne community](https://discord.gg/Cgzpw9x3R) · **Issues:** [GitHub Issues](https://github.com/AxDSan/mnemosyne/issues)

<a href="https://github.com/sponsors/AxDSan"><img src="https://img.shields.io/badge/%F0%9F%92%96_GitHub_Sponsors-30363D?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Sponsors"/></a>
<a href="https://ko-fi.com/axdsan"><img src="https://img.shields.io/badge/%E2%98%95_Ko-fi-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white" alt="Ko-fi"/></a>

⭐ **Star the repo if you find it useful!**

</div>

---

## License

MIT License — See [LICENSE](LICENSE)

Copyright (c) 2026 Abdias J

---

<p align="center">
  <em>"The faintest ink is more powerful than the strongest memory." — Hermes Trismegistus</em>
</p>
