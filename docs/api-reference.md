# API Reference — Mnemosyne v2.8.0

## Quick Start

```python
from mnemosyne import Mnemosyne

# Initialize (uses ~/.hermes/mnemosyne/data/ by default)
mem = Mnemosyne()

# Store memories
mem.remember("User prefers dark mode", importance=0.9)

# Recall memories
results = mem.recall("user preferences", top_k=5)

# Get stats
stats = mem.get_stats()
```

---

## Module-Level Convenience Functions

```python
from mnemosyne import remember, recall, get_stats, forget, update, get_context
```

These functions create a default `Mnemosyne` instance and delegate to it. The optional `bank` parameter applies only to these module-level helpers; instance methods use the bank configured on `Mnemosyne(...)`.

| Function | Signature | Description |
|---|---|---|
| `remember()` | `(content, source="conversation", importance=0.5, **kwargs) -> str` | Store a memory, returns memory ID |
| `recall()` | `(query, top_k=5, **kwargs) -> list` | Search memories |
| `get_stats()` | `() -> dict` | Memory statistics |
| `forget()` | `(memory_id) -> bool` | Delete a memory |
| `update()` | `(memory_id, **kwargs) -> bool` | Update a memory |
| `get_context()` | `(limit=10, bank=None) -> list[dict]` | Get recent working-memory context |

---

## Mnemosyne Class

**Module:** `mnemosyne.core.memory`

```python
from mnemosyne.core.memory import Mnemosyne

# Default instance
mem = Mnemosyne()

# With custom database path
from pathlib import Path
mem = Mnemosyne(db_path=Path("/path/to/mnemosyne.db"))

# With memory bank
mem = Mnemosyne(bank="work")

# With session ID
mem = Mnemosyne(session_id="my-agent-session")
```

### Constructor

```python
Mnemosyne(
    session_id: str = "default",     # Session identifier
    db_path: Path = None,            # Custom database path
    bank: str = None,                # Memory bank name for isolation
    author_id: str = None,
    author_type: str = None,
    channel_id: str = None,
)
```

### `remember()`

Store a memory. Returns the memory ID.

```python
memory_id = mem.remember(
    content: str,                      # The text to remember
    source: str = "conversation",      # Origin: "conversation", "document", "system"
    importance: float = 0.5,           # 0.0–1.0 relevance score
    metadata: dict = None,             # Optional additional fields
    valid_until: str = None,           # ISO timestamp when this memory expires
    scope: str = "session",            # "session" or "global"
    extract_entities: bool = False,    # Extract entity mentions as triples
    extract: bool = False              # Extract structured facts via LLM
)
```

**Examples:**

```python
# Basic storage
mid = mem.remember("User prefers dark mode", importance=0.9)

# With entity extraction
mid = mem.remember("Abdias founded Mnemosyne in New York",
                    extract_entities=True)

# With fact extraction (requires LLM)
mid = mem.remember("The project deadline is June 15th",
                    extract=True, importance=0.8)

# Session-scoped (auto-evicted after TTL)
mid = mem.remember("Current task: fixing bug #42",
                    scope="session", source="system")

# Global (persists across sessions)
mid = mem.remember("User's timezone is EST",
                    scope="global")
```

### `recall()`

Search memories using hybrid vector + FTS + importance scoring.

```python
results = mem.recall(
    query: str,                           # Search query
    top_k: int = 5,                       # Number of results
    from_date: str = None,                # Optional lower timestamp bound
    to_date: str = None,                  # Optional upper timestamp bound
    source: str = None,                   # Filter by source
    topic: str = None,                    # Filter by topic metadata
    author_id: str = None,                # Filter by author identity
    author_type: str = None,              # Filter by human/agent/system author type
    channel_id: str = None,               # Filter by channel/group
    temporal_weight: float = 0.0,         # 0.0–1.0, boosts memories near query_time
    query_time = None,                    # datetime or ISO string for temporal calculation
    temporal_halflife: float = None,      # Hours for decay (default: 24)
    vec_weight: float = None,             # Override vector scoring weight
    fts_weight: float = None,             # Override FTS scoring weight
    importance_weight: float = None       # Override importance weight
)
```

Returns a list of dicts with keys: `id`, `content`, `score`, `source`, `timestamp`, `importance`, `scope`, `metadata`.

**Examples:**

```python
# Basic recall
results = mem.recall("user preferences")

# Temporal boost — prefer recent memories
results = mem.recall("current task", temporal_weight=0.5)

# Custom scoring weights
results = mem.recall("python code",
    vec_weight=0.7, fts_weight=0.2, importance_weight=0.1)

# Filter by source
results = mem.recall("meeting notes", source="document")
```

### `update()`

Update an existing memory.

```python
success = mem.update(
    memory_id: str,
    content: str = None,
    importance: float = None,
    metadata: dict = None
)
```

### `forget()`

Delete a memory by ID.

```python
success = mem.forget(memory_id: str)
```

### `sleep()`

Run BEAM consolidation cycle — moves working memory to episodic storage.

```python
stats = mem.sleep()
# Returns dict with consolidation statistics
```

### `get_stats()`

Get memory statistics.

```python
stats = mem.get_stats()
# Returns dict with counts per tier, session info, etc.
```

### `get_context()`

Get recent working-memory context for prompt injection.

```python
context = mem.get_context(limit=5)
# Returns a list of recent working-memory dictionaries
```

### `export_to_file()` / `import_from_file()`

Portable export/import for backup and migration.

```python
mem.export_to_file("backup.json")
mem.import_from_file("backup.json")
```

### v2 Properties (lazy-initialized)

```python
mem.stream           # MemoryStream — event stream
mem.compressor       # MemoryCompressor — compress/decompress
mem.patterns         # PatternDetector — temporal/content patterns
mem.delta_sync       # DeltaSync — incremental sync
mem.plugin_manager   # PluginManager — plugin lifecycle
```

---

## BeamMemory Class

**Module:** `mnemosyne.core.beam`

The BEAM (Bilevel Episodic-Associative Memory) engine. Usually accessed through `Mnemosyne`, but can be used directly.

```python
from mnemosyne.core.beam import BeamMemory

beam = BeamMemory(
    session_id: str = "default",
    db_path: Path = None
)
```

### Key Methods

| Method | Description |
|---|---|
| `remember(content, **kwargs) -> str` | Store to working memory |
| `recall(query, **kwargs) -> list` | Hybrid search across all tiers |
| `sleep() -> dict` | Consolidate working → episodic |
| `invalidate(memory_id, replacement_id=None) -> bool` | Mark memory as superseded |
| `scratchpad_write(content) -> str` | Write to scratchpad |
| `scratchpad_read() -> list[dict]` | Read scratchpad entries |
| `scratchpad_clear() -> None` | Clear scratchpad |

---

## Memory Banks

**Module:** `mnemosyne.core.banks`

```python
from mnemosyne.core.banks import BankManager

manager = BankManager(data_dir="~/.hermes/mnemosyne/data")

# Create a bank
manager.create_bank("work")

# List banks
banks = manager.list_banks()

# Check if bank exists
exists = manager.bank_exists("work")

# Get bank stats
stats = manager.get_bank_stats("work")

# Rename a bank
manager.rename_bank("work", "work-v2")

# Delete a bank
manager.delete_bank("work")
```

---

## Entity Extraction

**Module:** `mnemosyne.core.entities`

```python
from mnemosyne.core.entities import (
    extract_entities_regex,
    levenshtein_distance,
    find_similar_entities
)

# Extract entities from text
entities = extract_entities_regex("Abdias founded Mnemosyne in New York")
# Returns: ["Abdias", "Mnemosyne", "New York"]

# Fuzzy match
distance = levenshtein_distance("Abdias", "Abdias J")

# Find similar entities in a list
matches = find_similar_entities("Abdias", ["Abdias J", "Python", "New York"], threshold=0.7)
```

---

## Fact Extraction

**Module:** `mnemosyne.core.extraction`

```python
from mnemosyne.core.extraction import extract_facts, extract_facts_safe

# Extract facts (may raise if no LLM available)
facts = extract_facts("Mnemosyne uses SQLite for storage and fastembed for embeddings.")
# Returns: list of fact strings

# Safe wrapper (never raises, returns empty list on failure)
facts = extract_facts_safe("Some text to extract facts from")
```

**Fallback chain:** Remote OpenAI API → Local ctransformers GGUF → Skip (returns [])

---

## Streaming & Delta Sync

**Module:** `mnemosyne.core.streaming`

### MemoryStream

```python
from mnemosyne.core.streaming import EventType, MemoryEvent, MemoryStream

stream = MemoryStream()

# Push events
stream.emit(MemoryEvent(
    event_type=EventType.MEMORY_ADDED,
    memory_id="abc",
    content="test",
))

# Pull via callback for one event type
stream.on(EventType.MEMORY_ADDED, lambda event: print(event))

# Pull via callback for all event types
stream.on_any(lambda event: print(event))

# Pull via iterator
for event in stream.listen():
    process(event)
```

### DeltaSync

```python
from mnemosyne.core.streaming import DeltaSync

sync = DeltaSync(mnemosyne_instance)
peer_id = "peer-node-1"

# Compute changes since this peer's last checkpoint
delta = sync.compute_delta(peer_id)

# Apply a delta received from that peer
stats = sync.apply_delta(peer_id, delta)
# stats == {"inserted": N, "updated": N, "skipped": N, "filtered_keys": N}

# Package a delta for transport to a peer
outgoing = sync.sync_to(peer_id)

# Apply a received delta and update the peer checkpoint
result = sync.sync_from(peer_id, incoming_delta)
```

**Allowed tables.** `compute_delta`, `apply_delta`, `sync_to`, and
`sync_from` accept `table` as a keyword argument. Only the values in
`ALLOWED_DELTA_TABLES` (`{"working_memory", "episodic_memory"}`) are
accepted; anything else raises `ValueError`. The allowlist is the
trust boundary against SQL injection via the `table` kwarg — see
C25 in the memory-contract ledger. To extend it, edit
`ALLOWED_DELTA_TABLES` in `mnemosyne/core/streaming.py` (deliberate
change, not a silent ride-along on a stray kwarg).

**Opt-in column allowlist on apply.** `apply_delta` accepts only an
explicit set of peer-mutable columns:

- **UPDATE path** (existing row, matched by `id`): peer may mutate
  `content`, `source`, `importance`, `metadata_json`, `veracity`,
  `memory_type`, `binary_vector`, `summary_of`. Everything else —
  identity (`id`), scope (`session_id`, `scope`), lifecycle
  (`valid_until`, `superseded_by`, `created_at`, `timestamp`,
  `recall_count`, `last_recalled`, `consolidated_at`, `degraded_at`,
  `tier`), and authorship (`author_id`, `author_type`, `channel_id`)
  — is destination-controlled. A peer cannot re-route a victim's
  row to its own session, soft-delete via `superseded_by`, or
  rewrite lifecycle history.
- **INSERT path** (new row): peer supplies `id` + the same content/
  metadata fields + `timestamp` (preserved as the original creation
  time). Lifecycle / scope / authorship fall back to destination
  column defaults — a peer cannot land a row directly inside the
  destination's local session or claim authorship.

Keys outside the allowlist are silently dropped and counted in
`stats["filtered_keys"]` — operators can watch this counter to spot
misconfigured peers (typo'd column names) or hostile peers
(injection attempts in column-name slots).

**Schema qualification.** All SQL operations use the `main.` schema
prefix and quoted identifiers (`UPDATE "main"."working_memory" SET
"content" = ?`). A same-connection temp table named `working_memory`
cannot shadow the real table.

**Per-table checkpoints.** `compute_delta` / `apply_delta` maintain
separate checkpoints per `(peer_id, table)` pair. Pre-hardening a
single per-peer checkpoint covered all tables, which caused silent
skip-rows on cross-table sync because `rowid` namespaces are table-
local. Checkpoint files are now `checkpoint_<peer>__<table>.json`;
legacy `checkpoint_<peer>.json` files load as the `working_memory`
checkpoint for backward compat.

---

## Pattern Detection & Compression

**Module:** `mnemosyne.core.patterns`

### PatternDetector

```python
from mnemosyne.core.patterns import PatternDetector

detector = PatternDetector()

# Detect temporal patterns (hour-of-day, day-of-week)
temporal = detector.detect_temporal_patterns(memories)

# Detect content patterns (keyword frequency, co-occurrence)
content = detector.detect_content_patterns(memories)

# Detect sequence patterns
sequences = detector.detect_sequence_patterns(memories)
```

### MemoryCompressor

```python
from mnemosyne.core.patterns import MemoryCompressor

compressor = MemoryCompressor()

# Compress memories
compressed = compressor.compress(memories)

# Decompress
decompressed = compressor.decompress(compressed)

# Batch compress
batch = compressor.compress_batch(memory_list)
```

---

## Plugin System

**Module:** `mnemosyne.core.plugins`

### Creating a Plugin

```python
from mnemosyne.core.plugins import MnemosynePlugin

class MyPlugin(MnemosynePlugin):
    name = "my-plugin"
    
    def on_remember(self, memory_id, content, **kwargs):
        """Called after a memory is stored."""
        pass
    
    def on_recall(self, query, results, **kwargs):
        """Called after recall. Can modify results."""
        return results
    
    def on_consolidate(self, count, **kwargs):
        """Called after sleep() consolidation."""
        pass
    
    def on_invalidate(self, memory_id, **kwargs):
        """Called after a memory is invalidated."""
        pass
```

### PluginManager

```python
from mnemosyne.core.plugins import PluginManager

pm = PluginManager()

# Register a plugin
pm.register(MyPlugin())

# Load from directory (auto-discovers .py files)
pm.discover("~/.hermes/mnemosyne/plugins/")

# Unregister
pm.unregister("my-plugin")

# List loaded plugins
plugins = pm.list_plugins()
```

---

## TripleStore (Knowledge Graph)

**Module:** `mnemosyne.core.triples`

```python
from mnemosyne.core.triples import TripleStore

store = TripleStore(db_path)

# Add a triple
store.add_triple("user_123", "prefers", "dark_mode")

# Query triples
results = store.query_triples(subject="user_123")

# Get all triples for an entity
triples = store.get_triples_for_subject("memory_id_abc")
```

---

## MCP Server

**Module:** `mnemosyne.mcp_server`

```bash
# stdio transport (for Claude Desktop, etc.)
mnemosyne mcp

# SSE transport (for web clients)
mnemosyne mcp --transport sse --port 8080

# Scoped to a specific bank
mnemosyne mcp --bank project_a
```

### MCP Tools

These are the standalone MCP server tools from `mnemosyne.mcp_tools`. The Hermes plugin exposes a larger tool surface and uses `mnemosyne_stats` rather than the MCP-only `mnemosyne_get_stats` name.

| Tool | Description |
|---|---|
| `mnemosyne_remember` | Store a memory |
| `mnemosyne_recall` | Search memories |
| `mnemosyne_sleep` | Run consolidation |
| `mnemosyne_scratchpad_read` | Read scratchpad |
| `mnemosyne_scratchpad_write` | Write to scratchpad |
| `mnemosyne_get_stats` | Get memory statistics |

---

## LLM Backends (Host Adapter)

**Module:** `mnemosyne.core.llm_backends`

Mnemosyne can route LLM-backed operations (consolidation and fact extraction) through a host-provided backend instead of its own remote/local chain. This is used when Mnemosyne runs inside Hermes to reuse Hermes' authenticated provider (including OAuth-backed providers like ChatGPT/Codex).

### LLMBackend Protocol

```python
from mnemosyne.core.llm_backends import LLMBackend, set_host_llm_backend

class MyBackend:
    name = "my-backend"

    def complete(self, prompt, *, max_tokens, temperature, timeout,
                 provider=None, model=None):
        # Route through your authenticated client
        return text_or_none

set_host_llm_backend(MyBackend())
```

### Registry API

```python
from mnemosyne.core.llm_backends import (
    set_host_llm_backend,
    get_host_llm_backend,
    call_host_llm,
    CallableLLMBackend,
)

# Register a backend
set_host_llm_backend(CallableLLMBackend(name="test", func=my_func))

# Check if registered
backend = get_host_llm_backend()

# Call with automatic fallback
result = call_host_llm(prompt, max_tokens=256, temperature=0.3, timeout=15.0)
```

### Fallback Chain

When `MNEMOSYNE_HOST_LLM_ENABLED=true`:

```text
0. Host backend (if registered)
   ↓ on failure: skip remote URL entirely (A3 rule)
1. Local GGUF (ctransformers / llama-cpp-python)
   ↓ on failure
2. Return None / [] — caller falls back to AAAK encoding
```

When `MNEMOSYNE_HOST_LLM_ENABLED=false` or unset:

```text
0. Remote OpenAI-compatible API (if MNEMOSYNE_LLM_BASE_URL set)
   ↓ on failure
1. Local GGUF
   ↓ on failure
2. AAAK encoding
```

---

## Importers

**Module:** `mnemosyne.core.importers`

Mnemosyne can import memories from supported external providers. All importers preserve metadata, timestamps, and identity.

### Supported Providers

| Provider | Class | Input | Key Preservation |
|---|---|---|---|
| **Mem0** | `Mem0Importer` | API key + user ID | User/app scoping |
| **Letta** | `LettaImporter` | AgentFile `.af` | Memory blocks, messages |
| **Zep** | `ZepImporter` | API key | Sessions, summaries, facts |
| **Cognee** | `CogneeImporter` | Graph data | Nodes → memories, edges → triples |
| **Honcho** | `HonchoImporter` | API key | Peer identity as author_id |
| **SuperMemory** | `SuperMemoryImporter` | API key | Container tags → channel_id |
| **Hindsight** | `HindsightImporter` | JSON file or HTTP API | **Timestamps, fact_type, session IDs, metadata, veracity** |

### HindsightImporter

**Special behavior:** Unlike other importers that route through `remember()`, HindsightImporter writes directly to `episodic_memory`. This preserves historical timestamps and avoids working-memory session contamination.

```python
from mnemosyne.core.importers import HindsightImporter

# From JSON export
importer = HindsightImporter(file_path="hindsight-export.json", bank="hermes")
result = importer.run(mnemosyne)

# From live API
importer = HindsightImporter(base_url="http://localhost:8888", bank="hermes")
result = importer.run(mnemosyne)

# Convenience wrapper
from mnemosyne.core.importers import import_from_hindsight
result = import_from_hindsight(mnemosyne, file_path="export.json", bank="hermes")
```

**Parameters:**
- `file_path` — Path to Hindsight JSON export file
- `base_url` — Base URL of running Hindsight API (e.g., `http://localhost:8888`)
- `bank` — Hindsight bank name (default: `hermes`)
- `page_size` — API pagination size, 1–1000 (default: 500)
- `max_items` — Maximum memories to import (default: unlimited)
- `namespace` — ID namespace for stable hashing (default: bank name)

**Result object:**
```python
result.provider      # "hindsight"
result.total         # Total items found
result.imported      # Successfully inserted
result.skipped       # Duplicates or empty content
result.failed        # Insertion errors
result.memory_ids    # List of imported memory IDs
result.errors        # List of error strings
result.started_at    # ISO timestamp
result.finished_at   # ISO timestamp
```

### Provider Registry

```python
from mnemosyne.core.importers import import_from_provider, PROVIDERS

# See all supported providers
print(PROVIDERS.keys())
# dict_keys(['mem0', 'letta', 'zep', 'cognee', 'honcho', 'supermemory', 'hindsight'])

# Generic import dispatcher
result = import_from_provider("hindsight", mnemosyne, file_path="export.json")
```

---

## CLI

```bash
mnemosyne store "User prefers dark mode" --importance 0.9
mnemosyne recall "user preferences" 10
mnemosyne update <memory_id> "Updated content"
mnemosyne delete <memory_id>
mnemosyne stats
mnemosyne sleep
mnemosyne export backup.json
mnemosyne import backup.json
mnemosyne import-hindsight export.json [bank]      # Import Hindsight JSON
mnemosyne import-hindsight http://localhost:8888    # Import from live API

# Hermes CLI routes through the same HindsightImporter, preserving timestamps
hermes mnemosyne import --from hindsight --file export.json --bank hermes
hermes mnemosyne import --from hindsight --input export.json --bank hermes
hermes mnemosyne import --from hindsight --base-url http://localhost:8888 --bank hermes
mnemosyne bank list
mnemosyne bank create work
mnemosyne bank delete work
mnemosyne mcp
mnemosyne diagnose
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MNEMOSYNE_DATA_DIR` | `~/.hermes/mnemosyne/data/` | Root data directory |
| `MNEMOSYNE_VEC_TYPE` | `int8` | Vector storage type: `bit` (48 bytes), `int8` (384 bytes), `float32` (1536 bytes) |
| `MNEMOSYNE_SESSION_ID` | Auto UUID | Default session identifier |
| `MNEMOSYNE_TEMPORAL_HALFLIFE_HOURS` | `24` | Default temporal decay halflife |
| `FASTEMBED_CACHE_PATH` | `~/.hermes/cache/fastembed` | FastEmbed model cache directory |

---

*API reference generated from source code. Every method and parameter verified against actual implementation.*
