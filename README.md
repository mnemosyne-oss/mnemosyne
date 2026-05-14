# Mnemosyne

![Mnemosyne](/assets/mnemosyne.jpg)

> Native, zero-cloud memory for AI agents. SQLite-backed. Sub-millisecond. Fully private.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyPI](https://img.shields.io/pypi/v/mnemosyne-memory.svg?v=2.8.0)](https://pypi.org/project/mnemosyne-memory/)
[![SQLite](https://img.shields.io/badge/SQLite-3.35+-green.svg)](https://sqlite.org/codeofethics.html)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/AxDSan/mnemosyne/actions/workflows/ci.yml/badge.svg)](https://github.com/AxDSan/mnemosyne/actions/workflows/ci.yml)
[![BEAM](https://img.shields.io/badge/BEAM-ICLR%202026-purple.svg)](https://beam-benchmark.github.io/)
[![Discord](https://badgen.net/discord/online-members/29ZszXTgY3)](https://discord.gg/Cgzpw9x3R)

Mnemosyne is a local-first memory system for the [Hermes Agent](https://github.com/NousResearch/hermes-agent) framework. It stores conversations, preferences, and knowledge in SQLite with native vector search (sqlite-vec) and full-text search (FTS5) -- no external databases, no API keys, no network calls.

## BEAM Benchmark (ICLR 2026)

Mnemosyne is evaluated on the [BEAM](https://github.com/mohammadtavakoli78/BEAM) long-context memory benchmark (Tavakoli et al., ICLR 2026) using the official end-to-end protocol: retrieved memories feed into an LLM, answers are scored by an LLM-as-judge against pre-written rubrics.

**End-to-end results** (48 questions per scale, 3 conversations each, 180 total):

| Scale | Mnemosyne | RAG (Llama-4) | LIGHT | Honcho | Hindsight |
|-------|-----------|---------------|-------|--------|-----------|
| 100K | **35.4%** | 32.3% | 35.8% | 63.0% | 73.4% |
| 500K | 19.3% | 33.0% | 35.9% | 64.9% | 71.1% |
| 1M | 19.2% | 30.7% | 33.6% | 63.1% | 73.9% |

**What this says:**
- At 100K (small conversations), Mnemosyne is competitive -- beats RAG, ties LIGHT
- At 500K+, performance degrades significantly below RAG. This is a known issue: the episodic consolidation pipeline is not producing entries during benchmark ingestion, so retrieval at scale loses information
- Published baselines use identical BEAM dataset and LLM-as-judge protocol

**Per-ability highlight (100K):** Information Extraction 80.5%, Abstention 50%, Summarization 41.7%. Multi-hop Reasoning (16.7%) and Event Ordering (13.3%) are weak -- these require fact linking across distant messages, which needs the episodic tier.

Full benchmark report: [docs/beam-benchmark.md](docs/beam-benchmark.md)

---

## Quick Start

### Option A: Install from PyPI (recommended)

```bash
pip install mnemosyne-memory
```

> **Note:** The package name on PyPI is `mnemosyne-memory`.

With all optional features (dense retrieval + local LLM consolidation):

```bash
pip install mnemosyne-memory[all]
```

> ⚠️ **Ubuntu 24.04 / Debian 12 users:** If you get `error: externally-managed-environment`, your system Python is PEP 668-protected. **Install Mnemosyne into the Hermes runtime venv** (not a separate one) to avoid ABI mismatches with compiled dependencies:
> ```bash
> HERMES_PY="$HOME/.hermes/hermes-agent/venv/bin/python"
> "$HERMES_PY" -m pip install --upgrade --no-cache-dir "mnemosyne-memory[all]"
> "$HERMES_PY" -m mnemosyne.install
> ```
> Installing into a separate venv can cause NumPy, sqlite-vec, or fastembed ABI crashes when Hermes tries to import them.

### Option B: Install from source (for development)

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e ".[all,dev]"
```

### Option C: Hermes MemoryProvider only (no pip needed)

If you only need Mnemosyne as a Hermes memory backend and want to skip pip entirely:

```bash
curl -sSL https://raw.githubusercontent.com/AxDSan/mnemosyne/main/deploy_hermes_provider.sh | bash
```

This symlinks the provider into `~/.hermes/plugins/mnemosyne` and adds the repo to `sys.path` at runtime. No virtual environment required -- works out of the box on Ubuntu 24.04.

### Register with Hermes

```bash
# 1. Install the plugin
python -m mnemosyne.install

# 2. Activate as your memory provider
hermes memory setup
# → Select "mnemosyne" and press Enter
```

Verify:

```bash
hermes memory status       # Should show "Provider: mnemosyne"
hermes mnemosyne stats     # Shows working + episodic memory counts
```

> **Note:** The `hermes memory setup` picker defaults to "Built-in only" every time it opens. This is normal Hermes UI behavior -- your previous selection **is** saved. Just select Mnemosyne and press Enter.

---

## What Makes It Different

### Mnemosyne vs. Cloud Memory Providers

| Feature | **Mnemosyne** | Honcho | Zep | Mem0 |
|---|---|---|---|---|
| **Cost** | **Free forever** | $$$ Paid (credit-based) | $$$ Paid (Flex/Enterprise) | Freemium ($0--$249/mo) |
| **Hosting** | **Local -- your machine** | Cloud only | Cloud / BYOC | Cloud only |
| **Privacy** | **100% local, zero data exfil** | External API calls | External API calls | External API calls |
| **Latency (read)** | **0.076 ms** | ~38 ms | ~62 ms | ~45 ms |
| **Latency (write)** | **0.81 ms** | ~45 ms | ~85 ms | ~50 ms |
| **Latency (search)** | **1.2 ms** | ~52 ms | ~78 ms | ~60 ms |
| **Cold start** | **0 ms (instant)** | ~500 ms | ~800 ms | ~300 ms |
| **Offline capable** | **Yes -- airplane mode works** | No | No | No |
| **Setup complexity** | **`pip install mnemosyne-memory`** | Docker + API keys + account | Docker + PostgreSQL + config | API key + signup |
| **Vector store** | **sqlite-vec (built-in)** | pgvector (external) | pgvector (external) | pgvector (external) |
| **Full-text search** | **FTS5 (built-in)** | Separate service | Separate service | Separate service |
| **Auth required** | **None** | Supabase auth | OAuth / API key | API key |
| **Rate limits** | **None -- unlimited** | Yes (plan-dependent) | Yes (credit-based) | Yes (plan-dependent) |
| **Data ownership** | **You own the SQLite file** | Vendor-hosted | Vendor-hosted | Vendor-hosted |
| **Export / import** | **One JSON file, any machine** | Limited | Limited | Limited | Limited |
| **Dependencies** | **Python stdlib + optional ONNX** | Docker, PostgreSQL, network | Docker, PostgreSQL, network | pip + API key + network | pip + API key + network |
| **Integration** | **Native Hermes plugin** | REST API SDK | REST API SDK | REST API SDK | REST API SDK |
| **Memory architecture** | **BEAM (3-tier: working + episodic + scratchpad)** | Session + facts | Graph RAG + facts | Session + facts | Retain/Recall/Reflect |
| **Auto-consolidation** | **Sleep cycles built-in** | Manual / paid add-on | Manual | Manual | Reflect loop |
| **Temporal knowledge graph** | **Native triples with validity** | No | No | No | Native graph + temporal |
| **Import from other systems** | **7 providers (Mem0, Letta, Zep, Cognee, Honcho, SuperMemory, Hindsight)** | No | No | No | No |
| **Benchmark (LongMemEval)** | **98.9% Recall@All@5** | Not published | Not published | Not published | Not published |

### What You Gain Switching to Mnemosyne

| From | You Gain | You Lose |
|---|---|---|
| **Honcho** | 500x faster reads, zero monthly bill, 100% offline, no Docker, no credit system | Cloud-hosted dashboard, managed scaling, team sharing features |
| **Zep** | 43x faster search, no PostgreSQL to maintain, no deployment overhead, instant cold start | Graph RAG visualization, enterprise compliance certs (SOC 2), managed BYOC |
| **Mem0** | Sub-millisecond everything, no API rate limits, no vendor lock-in, full data portability | Managed platform features, 90K+ developer community, YC-backed ecosystem |
| **Hindsight** | Zero dependency, no network calls, SQLite-native, BEAM architecture, **can import FROM Hindsight** | Cloud sync across devices, managed inference, web dashboard |

### The Bottom Line

- **If you care about speed**: Mnemosyne is 43--500x faster than any cloud alternative because it runs in-process with SQLite -- no HTTP roundtrips, no network overhead.
- **If you care about privacy**: Your data never leaves your machine. No API calls. No telemetry. No vendor access.
- **If you care about cost**: Zero ongoing cost. No credits. No tiers. No "contact sales."
- **If you care about simplicity**: `pip install mnemosyne-memory` and it works. No Docker. No config files. No signup.

**Trade-off**: You manage your own backup/restore (one SQLite file, trivial). You don't get a web dashboard or team collaboration features -- Mnemosyne is built for individual developers and local agents, not enterprise teams.

**Key capabilities:**

- **BEAM architecture** -- Three tiers: hot working memory, long-term episodic memory, temporary scratchpad
- **Hybrid search** -- 50% vector similarity + 30% FTS5 rank + 20% importance, all inside SQLite
- **Automatic consolidation** -- Old working memories are summarized and moved to episodic memory via `mnemosyne_sleep()`
- **Temporal triples** -- Time-aware knowledge graph with automatic invalidation
- **Entity extraction** -- Regex + Levenshtein fuzzy matching (no spaCy, no PyTorch)
- **LLM-driven fact extraction** -- Structured facts from raw text with graceful fallback chain
- **Host LLM adapter** -- Route consolidation and fact extraction through Hermes' authenticated provider (e.g., OAuth-backed Codex) without managing credentials in Mnemosyne
- **Memory banks** -- Per-bank SQLite isolation for domain separation
- **MCP server** -- 6 tools, stdio + SSE transports
- **Temporal recall** -- Exponential decay scoring with configurable halflife
- **Export / import** -- Move your entire memory database to a new machine with one JSON file
- **Cross-provider importers** -- Migrate from Mem0, Letta, Zep, Cognee, Honcho, SuperMemory, **Hindsight**
- **Cross-session scope** -- `remember(..., scope="global")` makes facts visible everywhere
- **Configurable compression** -- `int8` (default), `float32`, or `bit` (32x smaller) vectors
- **Binary vectors** -- Information-theoretic binarization (MIB) for 32x memory reduction with deterministic Hamming-distance retrieval
- **Streaming & DeltaSync** -- Real-time memory event stream (push/pull) and checkpoint-based incremental sync between instances
- **Configurable auto-sleep** -- Automatically triggers consolidation when working memory exceeds `sleep_threshold`; configurable via `config.yaml` or env var
- **ignore_patterns** -- Regex-based content filtering to exclude shell commands, stack traces, and boilerplate from memory storage

---

## Benchmarks

All numbers measured on CPU with `sqlite-vec` + FTS5 enabled.

### LongMemEval (ICLR 2025)

| System | Score | Notes |
|---|---|---|
| **Mnemosyne (dense)** | **98.9% Recall@All@5** | Oracle subset, 100 instances, bge-small-en-v1.5 |
| Mempalace | 96.6% Recall@5 | AAAK + Palace architecture |
| Mastra Observational Memory | 84.23% (gpt-4o) | Three-date model |
| Full-context GPT-4o baseline | ~60.2% | No memory system |

### Latency vs. Cloud Alternatives

| Operation | Honcho | Zep | MemGPT | **Mnemosyne** | Speedup |
|---|---|---|---|---|---|
| **Write** | 45ms | 85ms | 120ms | **0.81ms** | **56x** |
| **Read** | 38ms | 62ms | 95ms | **0.076ms** | **500x** |
| **Search** | 52ms | 78ms | 140ms | **1.2ms** | **43x** |
| **Cold Start** | 500ms | 800ms | 1200ms | **0ms** | **Instant** |

### BEAM Architecture Scaling

**Write throughput:**

| Operation | Count | Total | Avg |
|---|---|---|---|
| Working memory writes | 500 | 8.7s | **17.4 ms** |
| Episodic inserts (with embedding) | 500 | 10.7s | **21.3 ms** |
| Sleep consolidation | 300 old items | 33 ms | -- |

**Hybrid recall scaling (query latency stays flat as corpus grows):**

| Corpus Size | Query | Avg Latency | p95 |
|---|---|---|---|
| 100 | "concept 42" | **5.1 ms** | 6.9 ms |
| 500 | "concept 42" | **5.0 ms** | 5.7 ms |
| 1,000 | "concept 42" | **5.3 ms** | 6.5 ms |
| **2,000** | **"concept 42"** | **7.0 ms** | **8.6 ms** |

**Working memory recall scaling (FTS5 fast path):**

| WM Size | Query | Avg Latency | p95 |
|---|---|---|---|
| 1,000 | "concept 42" | **2.4 ms** | 3.1 ms |
| 5,000 | "domain 7" | **3.2 ms** | 3.8 ms |
| **10,000** | **"concept 42"** | **6.4 ms** | **7.2 ms** |

---

## Installation

### Prerequisites

- Python 3.9+
- Hermes Agent (for plugin integration)

### From PyPI (recommended for users)

```bash
pip install mnemosyne-memory

# With all extras (dense retrieval + local LLM consolidation)
pip install mnemosyne-memory[all]
```

### From source (recommended for contributors)

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e ".[all,dev]"
python -m mnemosyne.install
```

> ⚠️ **Ubuntu 24.04 / Debian 12 users:** If `pip install` fails with `externally-managed-environment`, see the [Quick Start → Option A](#quick-start) note about using a virtual environment.

### Optional dependencies

```bash
# Dense retrieval (required for semantic search and the 98.9% LongMemEval score)
pip install fastembed>=0.3.0

# Local LLM consolidation (sleep cycle summarization)
pip install ctransformers>=0.2.27 huggingface-hub>=0.20
```

> **Note:** Without `fastembed`, Mnemosyne falls back to keyword-only retrieval. It still works, but you won't get competitive semantic search or the benchmark scores above.

### Uninstall

```bash
python -m mnemosyne.install --uninstall
```

### Updating

If you installed from PyPI:

```bash
pip install --upgrade mnemosyne-memory
```

If you installed from source:

```bash
cd mnemosyne
git pull
pip install -e ".[all,dev]"
```

**Always restart Hermes** after updating so plugin changes take effect:
```bash
hermes gateway restart
```

**If the update includes database schema changes**, run the migration helper:
```bash
python scripts/migrate_from_legacy.py
```

See [UPDATING.md](UPDATING.md) for detailed troubleshooting and rollback instructions.

---

## Usage

### CLI

```bash
# Show memory statistics (current session only)
hermes mnemosyne stats

# Show memory statistics across ALL sessions
hermes mnemosyne stats --global

# Search memories
hermes mnemosyne inspect "dark mode preferences"

# Run consolidation (compress old working memory into episodic summaries)
hermes mnemosyne sleep

# Export all memories to a JSON file
hermes mnemosyne export --output mnemosyne_backup.json

# Import memories from a JSON file
hermes mnemosyne import --input mnemosyne_backup.json

# Import from Hindsight (JSON export or live API)
mnemosyne import-hindsight hindsight-export.json hermes
mnemosyne import-hindsight http://localhost:8888 hermes

# The same timestamp-preserving Hindsight importer is available inside Hermes
hermes mnemosyne import --from hindsight --file hindsight-export.json --bank hermes
hermes mnemosyne import --from hindsight --base-url http://localhost:8888 --bank hermes

# Clear scratchpad
hermes mnemosyne clear
```

> **Optional MCP server**: For external access or integration with MCP-compatible services, run the MCP server:
> ```bash
> mnemosyne mcp                          # stdio transport
> mnemosyne mcp --transport sse --port 8080  # SSE transport
> ```
> Mnemosyne does not currently expose a standalone REST API server.

### Python API

```python
from mnemosyne import remember, recall

# Store a fact
remember(
    content="User prefers dark mode interfaces",
    importance=0.9,
    source="preference"
)

# Store a global preference (visible in every session)
remember(
    content="User email is abdi.moya@gmail.com",
    importance=0.95,
    source="preference",
    scope="global"
)

# Store a temporary credential with expiry
remember(
    content="API key: sk-abc123",
    importance=0.8,
    source="credential",
    valid_until="2026-12-31T00:00:00"
)

# Search memories
results = recall("interface preferences", top_k=3)

# Temporal knowledge graph
from mnemosyne.core.triples import TripleStore
kg = TripleStore()
kg.add("Maya", "assigned_to", "auth-migration", valid_from="2026-01-15")
kg.query("Maya", as_of="2026-02-01")
```

### Advanced: BEAM direct access

```python
from mnemosyne.core.beam import BeamMemory

beam = BeamMemory(session_id="my_session")

# Working memory (auto-injected into prompts)
beam.remember("Important context", importance=0.9)

# Episodic memory (long-term, searchable)
beam.consolidate_to_episodic(
    summary="User likes Neovim",
    source_wm_ids=["wm1"],
    importance=0.8
)

# Scratchpad (temporary reasoning)
beam.scratchpad_write("todo: fix auth bug")

# Search both tiers
results = beam.recall("editor preferences", top_k=5)
```

### Temporal Recall

Temporal recall adds an exponential decay boost so recent memories rank higher:

```python
results = recall(
    "deployments",
    temporal_weight=0.5,        # Enable temporal scoring
    temporal_halflife=48.0,     # 48-hour halflife
    query_time="2026-04-29T12:00:00"  # Reference point
)
```

### Entity Extraction

Regex-based entity extraction with Levenshtein fuzzy matching. No spaCy, no PyTorch.

```python
remember(
    "Met with Abdias J about the Mnemosyne v2 release",
    extract_entities=True
)
# Extracts: "Abdias J", "Mnemosyne" -- stored as triples
# Fuzzy match: querying "Abdias" finds "Abdias J" (similarity: 0.925)
```

Catches `@mentions`, `#hashtags`, `"quoted phrases"`, and capitalized sequences (2-5 words). Misses pronouns and complex coreferences.

### LLM-Driven Fact Extraction

Extract structured facts from raw text using an LLM, with a graceful fallback chain:

0. **Host LLM adapter** (if `MNEMOSYNE_HOST_LLM_ENABLED=true` and a backend is registered -- e.g. when running under Hermes)
1. Remote OpenAI-compatible API (if `MNEMOSYNE_LLM_BASE_URL` is set)
2. Local ctransformers GGUF model
3. Skip -- extraction fails silently, memory is still stored

```python
remember(
    "User said they prefer Python over JavaScript for backend work",
    extract=True  # Extracts 2-5 factual statements as triples
)
```

### Memory Banks

Per-bank SQLite isolation for domain separation:

```python
from mnemosyne.core.banks import BankManager

BankManager().create_bank("work")
BankManager().create_bank("personal")

work_mem = Mnemosyne(bank="work")
work_mem.remember("Sprint review scheduled for Friday")
```

```bash
mnemosyne bank list
mnemosyne bank create research
mnemosyne mcp --bank work  # MCP server scoped to a bank
```

### MCP Server

6 tools, 2 transports, for any MCP-compatible client:

```bash
# stdio -- for Claude Desktop, etc.
mnemosyne mcp

# SSE -- for web clients
mnemosyne mcp --transport sse --port 8080
```

| Tool | Description |
|---|---|
| `mnemosyne_remember` | Store a memory |
| `mnemosyne_recall` | Search with hybrid scoring |
| `mnemosyne_sleep` | Run consolidation |
| `mnemosyne_scratchpad_read` | Read scratchpad |
| `mnemosyne_scratchpad_write` | Write to scratchpad |
| `mnemosyne_get_stats` | Memory statistics |

### Hermes Plugin

When registered as a Hermes plugin, Mnemosyne exposes **15 tools** to the agent, providing full memory lifecycle management directly from the conversation:

| # | Tool | Description |
|---|------|-------------|
| 1 | `mnemosyne_remember` | Store a memory with importance, source, expiry, and cross-session scope |
| 2 | `mnemosyne_recall` | Hybrid vector + FTS5 search across working and episodic memory with configurable scoring weights |
| 3 | `mnemosyne_stats` | Get BEAM tier statistics (working, episodic, scratchpad counts) |
| 4 | `mnemosyne_triple_add` | Add a temporal triple to the knowledge graph with validity dates |
| 5 | `mnemosyne_triple_query` | Query the temporal knowledge graph with `as_of` historical lookback |
| 6 | `mnemosyne_sleep` | Run the consolidation cycle — summarize old working memories into episodic tier |
| 7 | `mnemosyne_scratchpad_write` | Write a temporary note to the scratchpad reasoning workspace |
| 8 | `mnemosyne_scratchpad_read` | Read all current scratchpad entries |
| 9 | `mnemosyne_scratchpad_clear` | Clear all scratchpad entries |
| 10 | `mnemosyne_invalidate` | Mark a memory as expired or superseded by a replacement |
| 11 | `mnemosyne_export` | Export all memories to a portable JSON file for backup or migration |
| 12 | `mnemosyne_update` | Update the content or importance of an existing memory by ID |
| 13 | `mnemosyne_forget` | Permanently delete a memory by ID |
| 14 | `mnemosyne_import` | Import from JSON file, Mem0, Letta, Zep, Cognee, Honcho, SuperMemory, or Hindsight |
| 15 | `mnemosyne_diagnose` | Run PII-safe diagnostics (dependencies, DB state, vector readiness) — never exposes memory content |

The plugin also registers three lifecycle hooks (`pre_llm_call`, `on_session_start`, `post_tool_call`) for automatic context injection before every LLM call.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HERMES AGENT                              │
│                                                              │
│  ┌─────────────┐     ┌──────────────┐     ┌─────────────┐  │
│  │   pre_llm   │────▶│  Mnemosyne   │────▶│   SQLite    │  │
│  │    hook     │     │    BEAM      │     │             │  │
│  └─────────────┘     └──────────────┘     │ working_mem │  │
│         ▲                                  │ episodic_mem│  │
│         │                                  │ vec_episodes│  │
│         └──────── Auto-injected context ───│ fts_episodes│  │
│                                            │ scratchpad  │  │
│                                            │ triples     │  │
│                                            └─────────────┘  │
│                                                              │
│  Core runs in-process. Optional MCP server available.        │
└─────────────────────────────────────────────────────────────┘
```

**BEAM** (Bilevel Episodic-Associative Memory):

- `working_memory` -- Hot context, auto-injected before LLM calls, TTL-based eviction
- `episodic_memory` -- Long-term storage with sqlite-vec + FTS5 hybrid search
- `scratchpad` -- Temporary agent reasoning workspace

**Binary Vectors** (MIB — Maximally Informative Binarization):
Mnemosyne uses information-theoretic binarization (building on Moorcheh ITS, arXiv:2601.11557) to compress 384-dimensional float32 embeddings into 48-byte binary vectors — a 32× reduction. Retrieval uses Hamming distance (XOR + popcount) for deterministic, CPU-efficient ranking without ANN indices or external vector databases. This enables sub-millisecond search over millions of vectors entirely within SQLite.

---

## Why SQLite for Hermes?

SQLite is already in your stack. Hermes uses it for session persistence. Mnemosyne extends that same file -- no new dependencies, no Docker containers, no connection pooling.

| Feature | Honcho | Zep | Mnemosyne |
|---|---|---|---|
| Deployment | Docker + PostgreSQL | Docker + Postgres | `pip install` |
| Query Language | REST API | REST API | `SELECT ... WHERE MATCH` |
| Vector Store | pgvector | pgvector | sqlite-vec |
| Text Search | Separate API | Separate API | Built-in FTS5 |
| Auth Required | Yes (supabase) | Yes | No |
| Offline Mode | No | No | Yes |
| Cold Start Latency | 500-800ms | 800ms+ | **0ms** |

---

## Backup, Export & Migration

By default, Mnemosyne stores its main database at `~/.hermes/mnemosyne/data/mnemosyne.db`. Named memory banks live under `~/.hermes/mnemosyne/data/banks/<name>/`, and standalone triple stores may create `triples.db` in the data directory.

```bash
# Simple backup of the full Mnemosyne data directory
cp -a ~/.hermes/mnemosyne/data ~/backups/mnemosyne_data_$(date +%Y%m%d)

# Export to JSON (portable across machines)
hermes mnemosyne export --output mnemosyne_backup.json

# Import on a new machine
hermes mnemosyne import --input mnemosyne_backup.json
```

### Migrate from other memory providers

Import directly from supported providers into Mnemosyne:

```bash
# List all supported providers
hermes mnemosyne import --list-providers

# Mem0 → Mnemosyne
hermes mnemosyne import --from mem0 --api-key sk-xxx

# Letta → Mnemosyne
hermes mnemosyne import --from letta --api-key sk-xxx

# Zep → Mnemosyne
hermes mnemosyne import --from zep --api-key sk-xxx

# Hindsight → Mnemosyne (JSON export or live API)
mnemosyne import-hindsight hindsight-export.json hermes
mnemosyne import-hindsight http://localhost:8888 hermes

# Hindsight → Mnemosyne via Hermes CLI
hermes mnemosyne import --from hindsight --file hindsight-export.json --bank hermes
hermes mnemosyne import --from hindsight --base-url http://localhost:8888 --bank hermes

# Generate a migration script for any provider
hermes mnemosyne import --from mem0 --generate-script --output-script migrate.py

# Use AI agent extraction (no SDK needed)
hermes mnemosyne import --from zep --agentic
```

**Supported providers:** Mem0, Letta (MemGPT), Zep, Cognee, Honcho, SuperMemory, **Hindsight**

The generic Hermes CLI exposes the common importer options. Provider-specific options are available through the Python importers; for example, offline Letta AgentFile imports can use `LettaImporter(agent_file_path="./agent.af")`.

Importers preserve source metadata where available. `HindsightImporter` uses a dedicated episodic import path to preserve original timestamps; other importers may store source timestamps in metadata while assigning a new Mnemosyne write timestamp. Use `--dry-run` to validate without writing.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data` | Database directory |
| `MNEMOSYNE_VEC_TYPE` | `int8` | Vector compression: `float32`, `int8`, or `bit` |
| `MNEMOSYNE_VEC_WEIGHT` | `0.5` | Vector similarity weight |
| `MNEMOSYNE_FTS_WEIGHT` | `0.3` | FTS5 keyword weight |
| `MNEMOSYNE_IMPORTANCE_WEIGHT` | `0.2` | Importance weight |
| `MNEMOSYNE_WM_MAX_ITEMS` | `10000` | Working memory item limit |
| `MNEMOSYNE_WM_TTL_HOURS` | `24` | Working memory TTL |
| `MNEMOSYNE_RECENCY_HALFLIFE` | `168` | Recency decay halflife in hours (1 week) |
| `MNEMOSYNE_EP_LIMIT` | `50000` | Episodic memory recall limit |
| `MNEMOSYNE_SLEEP_BATCH` | `5000` | Max working memories to fetch for consolidation |

### Local LLM (ctransformers/GGUF)

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_ENABLED` | `true` | Enable LLM summarization in sleep cycle |
| `MNEMOSYNE_LLM_N_CTX` | `2048` | Context window size for local model |
| `MNEMOSYNE_LLM_MAX_TOKENS` | `2048` | Max output tokens per summary |
| `MNEMOSYNE_LLM_N_THREADS` | `4` | CPU threads for local inference |
| `MNEMOSYNE_LLM_REPO` | `TheBloke/TinyLlama...` | HuggingFace repo for GGUF download |
| `MNEMOSYNE_LLM_FILE` | `tinyllama...Q4_K_M.gguf` | GGUF filename |

### Remote LLM (OpenAI-compatible)

Use a remote model (llama.cpp server, vLLM, Ollama, etc.) instead of local TinyLlama:

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_LLM_BASE_URL` | *(none)* | OpenAI-compatible API base URL, e.g. `http://localhost:8080/v1` |
| `MNEMOSYNE_LLM_API_KEY` | *(none)* | API key for authenticated endpoints |
| `MNEMOSYNE_LLM_MODEL` | *(none)* | Model identifier sent in requests |

When `BASE_URL` is set, Mnemosyne skips local ctransformers and uses your remote model for consolidation. Falls back to local if remote is unreachable, then to aaak encoding.

### Host LLM Adapter (Hermes / agent integration)

Route consolidation and fact extraction through a host-provided LLM (e.g.,
Hermes' authenticated `agent.auxiliary_client.call_llm`). Useful for
OAuth-backed providers like `openai-codex` that don't fit the URL+API-key
remote shape.

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_HOST_LLM_ENABLED` | `false` | Opt in to host-adapter routing |
| `MNEMOSYNE_HOST_LLM_PROVIDER` | *(none)* | Optional provider override, e.g. `openai-codex` |
| `MNEMOSYNE_HOST_LLM_MODEL` | *(none)* | Optional model override, e.g. `gpt-5.1-mini` |
| `MNEMOSYNE_HOST_LLM_N_CTX` | `32000` | Prompt-budget when host is the chosen path (TinyLlama-calibrated `LLM_N_CTX=2048` is too small for Codex/GPT-class) |

When the host call fails, the adapter falls back to the local GGUF model rather than the remote URL. See [docs/hermes-llm-integration.md](docs/hermes-llm-integration.md) for the full behavior model and session-shutdown semantics.

---

## Configuration (config.yaml)

In addition to environment variables, Mnemosyne supports configuration via Hermes' `config.yaml` file. This is the recommended approach for plugin-level settings, keeping all Hermes configuration in one place.

Place the `memory.mnemosyne` block under the top-level `memory` key:

```yaml
memory:
  mnemosyne:
    # Enable automatic consolidation on session boundaries
    auto_sleep: true

    # Minimum working memory count before auto-sleep triggers.
    # Prevents consolidation on trivial sessions. Default: 50
    sleep_threshold: 50

    # Vector storage type: float32 (full precision), int8 (default, good balance), or bit (32x smaller)
    vector_type: int8

    # Regex patterns to filter BEFORE memory storage.
    # Content matching any pattern is silently skipped (not stored).
    # Useful for excluding shell commands, stack traces, and boilerplate.
    ignore_patterns:
      - "^pip install"
      - "^npm install"
      - "^sudo "
      - "^Traceback \\(most recent call last\\)"
      - "^Error:"
      - "^git "
```

### Configuration Keys

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `auto_sleep` | `bool` | `false` | Auto-run `sleep()` consolidation on session boundaries. Also settable via `MNEMOSYNE_AUTO_SLEEP_ENABLED` env var. |
| `sleep_threshold` | `int` | `50` | Minimum working memory entries before auto-sleep triggers. Skips consolidation on sessions with few memories. |
| `vector_type` | `str` | `int8` | Vector storage type: `float32` (1,536 bytes/vector), `int8` (384 bytes, default), `bit` (48 bytes, 32× smaller). |
| `ignore_patterns` | `list[str]` | `[]` | Regex patterns applied via Python `re.search()`. Memories matching any pattern are discarded before storage. |

### ignore_patterns

`ignore_patterns` filters content **before** it enters memory storage. It uses Python's `re.search()` with `re.IGNORECASE` to match patterns against the content string. If any pattern matches, the memory is silently skipped — it will never appear in working memory or recall results.

Common use cases:
- **Shell commands**: `"^pip "`, `"^npm "`, `"^git "`, `"^sudo "`, `"^apt "`
- **Stack traces**: `"^Traceback \\(most recent call last\\)"`, `"^Error:"`, `"^\\s+at "`
- **Boilerplate**: `"^---BEGIN"`, `"^#include"`, `"^#!/"`
- **System noise**: Any pattern matching low-signal operational chatter

Patterns can be specified as a YAML list (one per line) or as a comma-separated string. They are applied at `remember()` time and logged at `DEBUG` level when matched.

Example:
```yaml
memory:
  mnemosyne:
    ignore_patterns:
      - "^pip "
      - "^npm "
      - "^Traceback \\(most recent call last\\)"
      - "^Error:"
      - "^\\s+at "
```

---

## Streaming & DeltaSync

Mnemosyne includes a real-time memory event system for reactive and distributed memory architectures.

### MemoryStream

An event-driven stream supporting both push (callbacks) and pull (iterator) patterns. Thread-safe, with a configurable buffer for late-connecting iterators.

```python
from mnemosyne.core.streaming import MemoryStream, MemoryEvent, EventType

stream = MemoryStream(max_buffer=100)

# Push: register callbacks
stream.on(EventType.MEMORY_ADDED, lambda e: print(f"New: {e.memory_id}"))

# Pull: iterate over events as they occur
for event in stream.listen([EventType.MEMORY_ADDED, EventType.MEMORY_CONSOLIDATED]):
    process(event)
```

Supported event types: `MEMORY_ADDED`, `MEMORY_RECALLED`, `MEMORY_INVALIDATED`, `MEMORY_CONSOLIDATED`, `MEMORY_UPDATED`.

### DeltaSync

Checkpoint-based incremental synchronization between Mnemosyne instances. Computes diffs (only changed memories since last sync) and applies them with insert/update/skip tracking.

```python
from mnemosyne.core.streaming import DeltaSync

ds = DeltaSync(mnemosyne)

# Push: compute and send delta since last sync
result = ds.sync_to("peer_b", "working_memory")
# result = {"delta": [...], "count": 3, "checkpoint": "..."}

# Pull: receive and apply remote delta
stats = ds.sync_from("peer_a", remote_delta, "working_memory")
# stats = {"inserted": 3, "updated": 0, "skipped": 0}
```

Checkpoints are persisted per-peer, enabling incremental sync without re-sending already-synchronized memories. Ideal for multi-agent setups and backup/restore workflows.

---

## Testing

```bash
# Run tests locally
python -m pytest tests/test_beam.py -v

# Run benchmarks
python tests/benchmark_beam_working_memory.py
```

All changes are validated through [GitHub Actions CI](https://github.com/AxDSan/mnemosyne/actions/workflows/ci.yml) on Python 3.9--3.12 before merging.

---

## Releases

Mnemosyne publishes [GitHub Releases](https://github.com/AxDSan/mnemosyne/releases) and [PyPI packages](https://pypi.org/project/mnemosyne-memory/) automatically on every `v*` tag. See [CONTRIBUTING.md](CONTRIBUTING.md) for the release process.

---

## Documentation

Full documentation is in the [`docs/`](docs/README.md) directory:

- [Getting Started](docs/getting-started.md) -- Installation, quickstart, first memory
- [Architecture](docs/architecture.md) -- BEAM tiers, SQLite backend, hybrid search
- [API Reference](docs/api-reference.md) -- Python API: `remember`, `recall`, `sleep`, triples, importers
- [Hermes Integration](docs/hermes-integration.md) -- Using as a Hermes memory backend
- [Hermes LLM Integration](docs/hermes-llm-integration.md) -- Routing consolidation through Hermes' authenticated provider (Codex/OAuth)
- [LLM Installation Guide](docs/llm-installation-guide.md) -- Installation instructions for AI agents and LLMs
- [Configuration](docs/configuration.md) -- Environment variables, vector compression, LLM setup
- [Changelog](docs/changelog.md) -- Release history

---

## Contributing

Contributions are welcome. Areas of active interest:

- [ ] Encrypted cloud sync (optional, user-controlled)
- [ ] Browser extension for web context capture
- [ ] Additional embedding models
- [ ] Multi-language support

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

<div align="center">

### ❤️ Support Mnemosyne

## Support & Community

- **Discord**: [Join the Mnemosyne community](https://discord.gg/Cgzpw9x3R) for help, discussion, and announcements
- **GitHub Issues**: [Report bugs or request features](https://github.com/AxDSan/mnemosyne/issues)
- **Documentation**: [docs.mnemosyne.site](https://docs.mnemosyne.site)
- **Email**: abdi.moya@gmail.com

If this project saves you time or helps your agents remember, consider supporting it:

<br/>

<a href="https://github.com/sponsors/AxDSan">
  <img src="https://img.shields.io/badge/💖_GitHub_Sponsors-30363D?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Sponsors"/>
</a>
<a href="https://ko-fi.com/axdsan">
  <img src="https://img.shields.io/badge/☕_Ko‑fi-FF5E5B?style=for-the-badge&logo=ko-fi&logoColor=white" alt="Ko-fi"/>
</a>

<br/>
<br/>

⭐ Star the repo if you find it useful!

</div>

---

## License

MIT License -- See [LICENSE](LICENSE)

Copyright (c) 2026 Abdias J

---

## Acknowledgments

- [Hermes Agent Framework](https://github.com/NousResearch/hermes-agent) -- The ecosystem Mnemosyne was built for
- [Honcho](https://github.com/plasticlabs/honcho) -- For defining the stateful memory space
- [Mempalace](https://github.com/thepersonalaicompany/mempalace) -- For proving local-first memory can compete on benchmarks
- [SQLite](https://sqlite.org/codeofethics.html) -- The world's most deployed database

---

<p align="center">
  <em>"The faintest ink is more powerful than the strongest memory." -- Hermes Trismegistus</em>
</p>
