# Mnemosyne Documentation

Local-first, zero-cloud memory for AI agents. SQLite-backed. Sub-millisecond. Fully private.

## Guides

| Document | Description |
|---|---|
| [Getting Started](getting-started.md) | Installation, quickstart, storing your first memory |
| [Architecture](architecture.md) | BEAM tiers, SQLite backend, hybrid search, knowledge graph |
| [API Reference](api-reference.md) | Python API: `remember`, `recall`, `sleep`, triples, stats |
| [CLI Reference](cli-reference.md) | Every command, including the ones `--help` omits |
| [Integrations](integrations/README.md) | Platform guides: Cursor, Claude Code, Codex, OpenWebUI, Windsurf, OpenClaw, Hermes |
| [OpenWebUI Deep Integration](integrations/openwebui-deep.md) | Auto-save every chat, memory browser dashboard, cross-session recall |
| [Integration Template](integrations/integration-template.md) | ~100-line pattern for adding any new platform |
| [Hermes Integration](hermes-integration.md) | Using Mnemosyne as a Hermes memory backend |
| [LLM Installation Guide](llm-installation-guide.md) | Installation instructions for AI agents/LLMs |
| [Configuration](configuration.md) | Environment variables, data directory, vector compression |
| [Mnemosyne Sync](sync/index.md) | Bidirectional sync, encryption, deployment, tutorial |
| [Security & Privacy](security.md) | Threat model, encryption internals, BYOK comparison |
| [Benchmarking](benchmarking.md) | Maintainer guide: per-tool A/B benchmark env vars, diagnostics, pure-recall mode, test sequence template |
| [Benchmark Results Analysis](benchmark-results-analysis.md) | Output-file schemas + analysis recipes (per-ability scores, paired bootstrap CIs, voice attribution). AI-assistant-friendly reference |
| [Changelog](changelog.md) | Pointer to the root `CHANGELOG.md`, plus release-state notes |

## Generated references

Written from the code by `scripts/generate-docs.py` and verified in CI by
`scripts/verify-docs.py`. Do not hand-edit these two files; edit the code or the
generator instead.

| Document | Description |
|---|---|
| [MCP Tool Schema](api/tool-schema.mdx) | All declared tools with parameters, split by whether they are reachable over MCP or only through the Hermes plugin |
| [Configuration Reference](api/configuration.mdx) | Every config key with its environment variable, real default, and restart requirement, plus the environment-only variables |

## Subsystems

| Document | Description |
|---|---|
| [Memory Hygiene](hygiene.md) | Noise scoring, the audit and clean workflow, secret detection, and how to prevent noise being stored |
| [Configuration Profiles](profiles.md) | The eight built-in profiles, what distinguishes them, validation rules, and the `vec_type` restart trap |
| [L3 Persona Tier](persona.md) | Durable behavioural facts promoted into a store; prompt injection reads an opt-in `persona.md` file. Includes an explicit list of what is not yet wired |
| [SHMR](shmr.md) | Self-harmonizing memory reasoning. Library only; nothing calls it yet |

## Reference and analysis

| Document | Description |
|---|---|
| [Sync Protocol Reference](sync.md) | Wire protocol, endpoints, CLI reference, deployment recipes |
| [Comparison: Mnemosyne vs Hindsight](comparison.md) | Architecture, retrieval, and integration comparison against Hindsight self-hosted |
| [BEAM Benchmark Results](beam-benchmark.md) | The v3.0.0 BEAM run, methodology, and judge caveats. Source for the README figures |
| [Compression Plugin](compression-plugin.md) | AAAK compression, the plugin interface, and the legacy caveman fallback |
| [Hermes LLM Integration](hermes-llm-integration.md) | Routing consolidation and extraction through a host-provided LLM backend |
| [Audit Workflow](audit-workflow.md) | How documentation audits are run, and the report format |

## Design Proposals (RFCs)

Forward-looking design documents. An RFC describes intended behaviour, not shipped behaviour, so check its **Status** line before treating it as a description of the code.

| Document | Description |
|---|---|
| [RFC 0001: Tags and Scope Unification](rfc/0001-tags-and-scope-unification.md) | First-class tags on memories, `scope` as a reserved tag namespace, tag filtering in `recall()` |
| [RFC 0002: Modality Providers](rfc/0002-modality-providers.md) | The `ModalityBackend` seam for vision, video, and audio understanding, and the Atlas Cloud configuration recipe |
| [RFC 0003: Media Assets and the Moment Index](rfc/0003-media-moments.md) | Reference-hash asset registry plus semantically tagged spans, so recall can locate a moment inside a video or document |
| [RFC 0004: The Archive Boundary](rfc/0004-archive-boundary.md) | `ContentResolver` and the contract that keeps heavy files outside the engine. The brain stores text and spans; an archive stores bytes |
| [R&D: Noise Remediation](rfc/noise-remediation-rnd.md) | Exploratory report on the pre-storage filter, hygiene audit, and the gaps between them |
| [Roadmap: Layered Agent Memory](roadmap-layered-agent-memory.md) | Proposed L0 through L4 layering. Extends BEAM rather than replacing it. Not shipped |

## Quick Links

- **Repository:** [github.com/mnemosyne-oss/mnemosyne](https://github.com/mnemosyne-oss/mnemosyne)
- **PyPI:** [mnemosyne-memory](https://pypi.org/project/mnemosyne-memory/)
- **Contributing:** See [CONTRIBUTING.md](../CONTRIBUTING.md) in the repo root
- **Updating:** See [UPDATING.md](../UPDATING.md) for upgrade and rollback instructions
- **License:** MIT -- see [LICENSE](../LICENSE)
