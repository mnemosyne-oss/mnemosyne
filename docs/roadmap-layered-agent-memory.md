# Roadmap: Layered Agent Memory for Hermes

This roadmap describes how Mnemosyne can evolve into a stronger layered memory system for Hermes and other long-running agents. The target is a local-first, Python-native, SQLite-first architecture that adds layered recall, evidence tracking, and agent workflow memory while preserving Mnemosyne's current operational model.

## Design Constraints

- Local-first remains mandatory: core memory must work offline and in-process.
- SQLite remains the primary database, including FTS5 and sqlite-vec where available.
- Python remains the native implementation and extension surface.
- Cloud vector databases, remote LLMs, and network gateways stay optional.
- Every synthesized memory must retain links to raw supporting evidence.
- Existing BEAM APIs and Hermes integration should continue to work during migration.

## Current Strengths

Mnemosyne already has a strong base for layered agent memory:

- **BEAM working and episodic memory**: hot working memory, long-term episodic memory, scratchpad, TTL-based consolidation, and prompt injection hooks.
- **SQLite + FTS5/sqlite-vec**: a compact local store with lexical search, vector search, and fallback paths when embeddings or sqlite-vec are unavailable.
- **Hermes plugin and lifecycle hooks**: native `MemoryProvider` support plus `pre_llm_call`, `on_session_start`, and `post_tool_call` integration points.
- **Memory banks**: named SQLite-backed banks for isolation across users, projects, agents, or environments.
- **TripleStore**: temporal subject-predicate-object facts for point-in-time symbolic recall.
- **MCP support**: stdio and SSE access for MCP-compatible clients without requiring a REST service.
- **Optional sync and encryption**: private-by-default local storage with optional encrypted synchronization.

## Layered Memory Target

The layered model should extend BEAM rather than replace it.

| Layer | Purpose | SQLite-first representation |
|---|---|---|
| L0 raw traces | Full-fidelity user, assistant, tool, file, and environment events | Append-only trace tables with session, bank, source, timestamp, hash, and payload metadata |
| L1 atoms | Small source-linked facts, decisions, preferences, constraints, and entities | Atomic memory rows with evidence links to L0 trace spans |
| L2 scenes/episodes | Coherent task episodes, milestones, failures, and outcomes | Episode records derived from trace ranges and atom clusters |
| L3 persona/profile | User, project, repo, and agent profiles | Versioned profile facts with confidence, validity window, and source links |
| L4 skills/SOPs | Reusable procedures, workflows, recipes, and agent habits | Procedure records with preconditions, steps, examples, and source episodes |

This structure keeps BEAM as the runtime memory engine while adding explicit provenance and higher-order organization.

## Layered Agent-Memory Capabilities to Add

### L0 Raw Traces

Add a durable trace store for high-volume event capture:

- user and assistant messages
- tool calls and results
- file edits and command summaries
- recall decisions and injected context
- session, bank, branch, task, and source metadata

Raw traces should be append-only by default, deduplicated by content hash, and excluded from prompt injection unless explicitly recalled.

### L1 Atoms

Extract compact, evidence-linked atomic memories from traces:

- facts: "repo uses pytest"
- constraints: "do not require Node.js"
- decisions: "SQLite remains primary store"
- preferences: "user wants concise markdown"
- entities: projects, files, people, tools, APIs

Atoms should be queryable through FTS5, sqlite-vec, and symbolic filters.

### L2 Scenes and Episodes

Represent task-level narrative memory:

- what the agent attempted
- which tools were used
- what changed
- what failed
- what evidence supports the outcome

Episodes should be assembled from L0 trace ranges and L1 atoms, then stored as searchable summaries with evidence links.

### L3 Persona/Profile

Build source-linked profiles for users, projects, repositories, and recurring agents:

- version profile entries over time
- record confidence and freshness
- keep every profile claim traceable to atoms and raw traces
- prefer additive refinement over opaque replacement

Profile synthesis must never create ungrounded summaries.

### L4 Skills/SOPs

Promote repeated successful workflows into reusable local procedures:

- benchmark recipes
- release checklists
- repo-specific coding conventions
- integration setup steps
- failure recovery patterns

Each SOP should reference the episodes and traces that justify it.

### Symbolic Task Canvas

Add a lightweight task canvas for active work:

- goals, subtasks, blockers, assumptions, decisions, and open questions
- links to memories, triples, files, traces, and tool calls
- current task state that can be recalled without replaying the full session

The canvas should complement scratchpad memory, not replace it.

### Tool-Log Offload

Use Hermes `post_tool_call` to offload bulky tool output into L0 traces:

- store full output locally
- inject only compact handles, summaries, and evidence links
- allow later expansion on demand
- preserve reproducibility for commands, files, and tool results

This should reduce prompt bloat during long sessions.

### Progressive Disclosure Recall

Recall should return the smallest useful pack first:

1. L3/L4 profile and SOP hints
2. L2 episode summaries
3. L1 atoms
4. L0 trace excerpts only when more evidence is needed

The packer should expose expandable evidence handles so agents can ask for more detail without injecting everything upfront.

### Source-Linked Profile Synthesis

Profile updates should be generated from traceable evidence:

- cite source atom IDs and trace IDs
- distinguish stable facts from recent observations
- expire or supersede stale claims
- detect conflicting claims instead of silently overwriting them

### Optional Local REST Gateway

Add a local REST gateway only as an optional adapter:

- disabled by default
- localhost-first
- no requirement for Hermes, MCP, or Python API users
- useful for dashboards, local web UIs, and non-Python clients

The core system must continue to work without any always-on server.

## Explicit Non-Goals

- No mandatory Node.js runtime.
- No mandatory cloud vector database.
- No mandatory remote LLM.
- No opaque summaries without evidence links.
- No always-on gateway requirement.
- No replacement of BEAM with a remote-first architecture.
- No migration that breaks existing memory banks or Hermes plugin users.

## PR-Sized Implementation Sequence

### 1. Layer Schema

- Add the minimum schema needed for layered metadata and evidence links first.
- Avoid creating every future layer table in one migration unless the implementation needs it immediately.
- Keep old BEAM tables intact.
- Add bank, session, source, timestamps, validity windows, confidence, and provenance fields.
- Include indexes for bank isolation, recency, source lookup, FTS5, and vector search.

### 2. Trace Store

- Implement append-only L0 trace writes.
- Add retention settings and payload size limits.
- Store large payloads with hashes and compact previews.
- Provide Python APIs for trace insert, lookup, and range retrieval.

### 3. Hermes Post-Tool-Call Offload

- Extend the Hermes `post_tool_call` hook to write tool outputs to L0.
- Return compact references for prompt use.
- Add config flags for capture policy, redaction, maximum payload size, and excluded tools.
- Verify that existing memory capture behavior remains compatible.

### 4. Layered Recall Packer

- Add a recall packer that can combine L4, L3, L2, L1, and L0 evidence under a token budget.
- Prefer high-level memories first, then progressively expand.
- Include evidence handles in returned context.
- Track injected-token counts for benchmarks.

### 5. Profile Synthesis

- Build profile synthesis over L1 atoms and L2 episodes.
- Store source-linked profile facts with confidence and validity windows.
- Add conflict detection and stale-claim invalidation.
- Keep remote LLM usage optional; support local or heuristic synthesis paths.

### 6. Task Canvas

- Add task canvas tables and APIs for goals, subtasks, blockers, assumptions, decisions, and open questions.
- Link canvas items to memories, traces, triples, and files.
- Integrate current canvas state into recall packs.

### 7. Optional Gateway

- Add a disabled-by-default local REST gateway.
- Expose trace lookup, recall packs, profile facts, canvas state, and stats.
- Reuse existing auth, redaction, and bank isolation rules.
- Keep MCP and Python APIs as first-class paths.

### 8. Benchmarks and Demos

- Add benchmarks for long-session recall, context-token reduction, profile correctness, and trace lookup latency.
- Add demos for Hermes tool-log offload, progressive disclosure, source-linked profile synthesis, and cross-bank isolation.
- Compare layered recall against current BEAM-only recall.

## Success Metrics

- **Lower injected context tokens**: fewer tokens injected per turn while preserving task-relevant context.
- **Better long-session task completion**: improved completion rate on multi-hour or multi-step Hermes sessions.
- **Profile accuracy**: profile claims match source evidence and avoid stale or conflicting assertions.
- **Traceability to raw evidence**: every atom, episode, profile fact, and SOP links back to L0 evidence.
- **Cross-bank isolation**: no leakage across named banks unless explicitly requested.
- **Latency and storage footprint**: recall remains fast enough for interactive use, and trace growth stays manageable through retention, deduplication, and previews.

## Open Questions

- Should L0 trace payloads live fully inside SQLite, or should very large blobs use a local content-addressed sidecar directory?
- What default retention policy best balances traceability with disk usage?
- How should confidence scores be calibrated without requiring an LLM?
- Which Hermes tool outputs need redaction before trace storage?
- Should SOP promotion be manual-first, automatic with review, or fully automatic behind a config flag?
