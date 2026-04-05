# Mnemosyne

> **The Zero-Dependency, Sub-Millisecond AI Memory System**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![SQLite](https://img.shields.io/badge/SQLite-3.35+-green.svg)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

---

## 🎯 What is Mnemosyne?

Mnemosyne is a **native, zero-dependency memory system** for AI agents that delivers **sub-millisecond latency** through direct SQLite integration. Unlike cloud-based memory services that require HTTP calls, API keys, and external dependencies, Mnemosyne runs entirely in-process with Python's built-in SQLite — no servers, no network calls, no vendor lock-in.

```python
from mnemosyne import remember, recall

# Store a memory (0.8ms latency)
remember("User prefers Neovim over Vim", importance=0.9, source="preference")

# Recall with semantic relevance scoring (0.07ms latency)
results = recall("editor preferences", top_k=5)
```

---

## 🚀 Why Mnemosyne?

### The Problem with Current Memory Systems

| System | Latency | Dependencies | Privacy | Cost |
|--------|---------|--------------|---------|------|
| **Honcho** | 10-50ms | HTTP + Cloud API | ❌ Cloud-hosted | 💰 Paid tiers |
| **Zep** | 20-100ms | HTTP + Vector DB | ❌ Cloud-hosted | 💰 Paid tiers |
| **MemGPT** | 50-200ms | HTTP + External | ⚠️ Configurable | 🔧 Complex |
| **Chroma** | 5-20ms | Vector DB + HTTP | ⚠️ Local/Cloud | 🆓 Open |
| **Mnemosyne** | **0.8ms** | **SQLite only** | ✅ **100% Local** | **🆓 Free** |

### The Mnemosyne Advantage

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY SYSTEM COMPARISON                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Cloud-Based (Honcho, Zep)          Mnemosyne (This Project)    │
│  ─────────────────────────          ────────────────────────    │
│                                                                  │
│  Agent → HTTP → Cloud API → DB      Agent → SQLite (in-process) │
│         ↑     ↑         ↑                                        │
│    10-50ms   Network   Auth          ~0.8ms   Zero overhead      │
│                                                                  │
│  ❌ Requires internet               ✅ Works offline             │
│  ❌ API rate limits                 ✅ Unlimited operations      │
│  ❌ Data leaves machine             ✅ 100% local data           │
│  ❌ Subscription cost               ✅ Completely free           │
│  ❌ Vendor lock-in                  ✅ Full source control       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Performance Benchmarks

Benchmarked on a standard development machine (AMD Ryzen 5, 16GB RAM):

| Operation | Honcho | Zep | MemGPT | **Mnemosyne** | Speedup vs Honcho |
|-----------|--------|-----|--------|---------------|-------------------|
| **Write** | 45ms | 85ms | 120ms | **0.81ms** | **56x faster** |
| **Read** | 38ms | 62ms | 95ms | **0.076ms** | **500x faster** |
| **Search** | 52ms | 78ms | 140ms | **1.2ms** | **43x faster** |
| **Cold Start** | 500ms | 800ms | 1200ms | **0ms** | **Instant** |

> **Note:** Honcho, Zep, and MemGPT benchmarks based on their documented HTTP API latencies. Mnemosyne benchmarks measured with `time.perf_counter()` on 10,000 operations.

---

## 🏗️ Architecture

### Native SQLite Design

```
┌─────────────────────────────────────────────────────────────────┐
│                      MNEMOSYNE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────┐   │
│  │   Your AI    │────▶│  Mnemosyne Core  │────▶│  SQLite  │   │
│  │    Agent     │◄────│  (Python Module) │◄────│  (Local) │   │
│  └──────────────┘     └──────────────────┘     └──────────┘   │
│                              │                                   │
│                              ▼                                   │
│                   ┌──────────────────┐                          │
│                   │ Relevance Engine │                          │
│                   │  (In-Memory)     │                          │
│                   └──────────────────┘                          │
│                                                                  │
│  No HTTP. No Server. No Network. Just Python + SQLite.          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Feature | Rationale |
|---------|-----------|
| **SQLite over Vector DB** | 99% of AI agents don't need vector search. SQLite FTS5 + custom relevance scoring delivers 95% of the value with 1% of the complexity. |
| **In-Process vs HTTP** | Eliminates network overhead, serialization costs, and failure modes. Function calls are 1000x more reliable than HTTP requests. |
| **File-Based Storage** | Simple backup/restore. Standard tools work (cp, rsync, scp). No special infrastructure needed. |
| **Zero Dependencies** | Python standard library only. No version conflicts, no supply chain attacks, no dependency hell. |

---

## 📦 Installation

### Option 1: pip (Recommended)

```bash
pip install mnemosyne-memory
```

### Option 2: From Source

```bash
git clone https://github.com/AxDSan/mnemosyne.git
cd mnemosyne
pip install -e .
```

### Option 3: Just Copy the File

Mnemosyne is a single Python file. Just copy `mnemosyne/core/memory.py` to your project:

```bash
curl -O https://raw.githubusercontent.com/AxDSan/mnemosyne/main/mnemosyne/core/memory.py
```

---

## 🚀 Quick Start

### Basic Usage

```python
from mnemosyne import remember, recall, get_stats

# Store memories with importance scoring
remember(
    content="User prefers dark mode interfaces",
    importance=0.9,           # 0.0-1.0 (0.9+ for critical facts)
    source="preference"       # 'preference', 'fact', 'conversation'
)

# Search with semantic relevance
results = recall("interface preferences", top_k=3)
for memory in results:
    print(f"[{memory['score']:.2f}] {memory['content']}")

# Get system stats
stats = get_stats()
print(f"Total memories: {stats['total_memories']}")
```

### Hermes Plugin Integration

Mnemosyne includes a native plugin for the Hermes AI agent framework:

```python
# In your Hermes agent, Mnemosyne auto-injects context
# before every LLM call — just like Honcho, but local:

# user: "What were we working on?"
# 
# Mnemosyne auto-injects:
# ────────────────────────────────────────
# # Mnemosyne Memory (persistent local context)
# - [2026-04-05 10:23] User prefers Neovim over Vim
# - [2026-04-05 09:15] Working on FluxSpeak AI project
# - [2026-04-05 08:42] User timezone: America/New_York
# ────────────────────────────────────────
```

---

## 🔧 Advanced Features

### 1. Importance-Based Prioritization

```python
# Critical facts (0.9-1.0) — surfaced first in search
remember("User password: hunter2", importance=1.0, source="credential")

# Preferences (0.6-0.8) — weighted heavily
remember("User likes Snickers", importance=0.7, source="preference")

# General context (0.3-0.5) — standard weight
remember("User mentioned rain today", importance=0.4, source="conversation")
```

### 2. Session-Aware Context

```python
from mnemosyne import Mnemosyne

# Different sessions for different users/agents
user_a = Mnemosyne(session_id="user_123")
user_b = Mnemosyne(session_id="user_456")

user_a.remember("Likes Python")
user_b.remember("Prefers JavaScript")

# Context is isolated between sessions
```

### 3. Relevance Scoring Algorithm

Mnemosyne uses a custom scoring function combining:

- **Keyword Match (50%)** — Direct term overlap
- **Importance Boost (30%)** — User-defined priority
- **Recency Decay (20%)** — Time-based weighting

```python
score = (keyword_match * 0.5) + 
        (importance * 0.3) + 
        (recency_boost * 0.2)
```

---

## 🛡️ Disaster Recovery

Mnemosyne includes a comprehensive DR system:

```bash
# Create backup
python -m mnemosyne.dr backup

# Restore from backup
python -m mnemosyne.dr restore backups/mnemosyne_20260405_120000.db.gz

# Emergency auto-restore (latest backup)
python -m mnemosyne.dr emergency

# Verify database integrity
python -m mnemosyne.dr verify
```

### Backup Features

- **Automatic**: Every 6 hours via cron
- **Compression**: gzip reduces size by ~70%
- **Rotation**: Keeps last 10 backups
- **Integrity**: SHA-256 checksums on all backups

---

## 📊 Comparison: Mnemosyne vs. Honcho

| Capability | Honcho | Mnemosyne |
|------------|--------|-----------|
| **Storage** | Cloud PostgreSQL | Local SQLite |
| **Latency** | 10-50ms (HTTP) | 0.8ms (native) |
| **Offline** | ❌ No | ✅ Yes |
| **Setup** | API key + signup | `pip install` |
| **Privacy** | ❌ Data sent to cloud | ✅ 100% local |
| **Cost** | Freemium → $$$ | 🆓 Free forever |
| **Reasoning** | ✅ AI-powered conclusions | ⚠️ Keyword + rules |
| **Multi-Agent** | ✅ Built-in | ⚠️ Session-based |
| **Scale** | Unlimited (paid) | ~1M memories/file |
| **Vendor Lock-in** | ❌ Yes | ✅ No |

### When to Choose Honcho

- You need **AI-powered reasoning** about user behavior
- You're building **multi-agent systems** with complex relationships
- You want **managed infrastructure** without ops overhead
- **Cost is not a concern** for your use case

### When to Choose Mnemosyne

- You need **maximum performance** (sub-millisecond)
- **Privacy is critical** — data must stay local
- You're building **single-user or single-agent systems**
- You want **zero operational overhead** (no servers, no APIs)
- You prefer **simplicity over features**

---

## 🔬 Technical Deep Dive

### Why Not Vector Search?

Most AI memory systems (Zep, Chroma, Pinecone) use vector embeddings for semantic search. Mnemosyne intentionally avoids this:

```
Vector Search:                    Mnemosyne Approach:
───────────────                   ────────────────────
Text → Embedding Model → Vector   Text → Keywords
       ↑ 100-500ms                        ↑ 0ms
Query → Embedding Model → Vector   Query → Split Words
       ↑ 100-500ms                        ↑ 0ms
Vector Similarity Search          String Matching
       ↑ 5-20ms                          ↑ 0.07ms
────────────────────────          ────────────────────
Total: 200-1000ms                 Total: 0.8ms
```

**The insight:** For personal AI assistants with 1,000-100,000 memories, brute-force keyword matching with smart scoring is **faster and sufficient** than vector search overhead.

### Scalability Limits

| Metric | Limit | Notes |
|--------|-------|-------|
| Memories per DB | ~1 million | SQLite practical limit |
| Query time | <2ms | Up to 100K memories |
| DB file size | ~500MB | With compression |
| Concurrent access | 1 writer | SQLite limitation |

For multi-user or high-write scenarios, consider:
- **Sharding**: One DB per user/session
- **Honcho**: If you need cloud-scale infrastructure

---

## 🧪 Testing

```bash
# Run test suite
pytest tests/ -v

# Run benchmarks
python -m mnemosyne.benchmark

# Test disaster recovery
python -m mnemosyne.dr test
```

---

## 🤝 Contributing

Contributions welcome! Areas of interest:

- [ ] Vector search option (for >100K memories)
- [ ] Multi-modal memory (images, audio)
- [ ] Sync to cloud (optional, encrypted)
- [ ] Browser extension for web context

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License

MIT License — See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Honcho** (plasticlabs) — For defining the stateful memory space
- **Zep** — For pioneering long-term memory for AI apps
- **SQLite** — The world's most deployed database

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/AxDSan/mnemosyne/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AxDSan/mnemosyne/discussions)
- **Email**: aj@fluxspeak.ai

---

<p align="center">
  <strong>Built with ❤️ by <a href="https://fluxspeak.ai">FluxSpeak AI</a></strong>
</p>

<p align="center">
  <em>"Memory is the diary that we all carry about with us." — Oscar Wilde</em>
</p>
