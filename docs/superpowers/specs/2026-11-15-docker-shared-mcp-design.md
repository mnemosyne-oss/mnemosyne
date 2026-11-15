# Docker deployment for shared MCP memory — agreed scope

Status: Scope approved in conversation on 2026-11-15; implementation plan requested. This records that scope, including Claude Code as the intended Claude client. Implementation has not started.

## Intent

Offer one versioned Mnemosyne image for self-hosting on a NAS or VPS. The operator's trusted Hermes, Codex and Claude Code agents connect to one shared memory service using MCP Streamable HTTP at `/mcp`. Each agent has its own named bearer token. All admitted agents have equal access; token identities provide attribution, not tenant isolation.

## Architecture and constraints

- One Mnemosyne process and one persistent local data volume; no shared SQLite file across replicas or network-mounted SQLite storage.
- Local embeddings included, persistent model cache, first-start model download documented. No local LLM consolidation dependency.
- Non-root container runtime; version bound to the repository release source, with controlled dependency/base-image updates.
- Linux AMD64 and ARM64 builds and runtime checks.
- Existing HTTPS proxy supported through a loopback host port or a shared Docker network.
- Direct LAN HTTP supported through an explicit LAN bind address. Document that tokens and memory content travel unencrypted, a stable address is needed, and access is restricted to the intended network.
- Optional Caddy example supports both a public hostname with automatic HTTPS and a local hostname with `tls internal`. Persist Caddy's CA state and explain client/runtime trust installation. No public domain or internet port-forwarding required for LAN HTTPS.
- Configuration requires valid named tokens and allowed Host values. Browser origins remain denied unless explicitly configured.
- Shared default bank and global write scope; no change to existing core visibility, identity, storage or public API semantics.
- Container checks cover authentication, host policy, actual MCP calls, shared access, restart persistence, non-root writes, model cache and backup/restore.
- Documentation includes versioned configuration examples and observed compatibility for Hermes MCP, Codex and Claude Code. Native Hermes provider/sync integration is outside this scope.
- Prepare a manually dispatched publish workflow. Registry setup, permissions and official release activation require the release owner's coordination under public MAINTAINERS.md. Ordinary PR checks never publish images.

## Approaches considered

1. MCP service: selected because agents need direct access to the same live memory service.
2. Sync server: useful for separate local stores, deferred because direct shared access meets the initial need.
3. Separate containers per agent/user: useful for isolation, deferred because all initial agents are trusted and should share memory.

Within the MCP approach, one image with local embeddings was selected over additional slim/remote-embedding variants to keep setup self-contained. Existing proxy, direct LAN and optional Caddy examples cover the requested environments.

## Acceptance and exclusions

A clean installation supports two separately authenticated agents sharing an ordinary memory, rejects invalid tokens and disallowed Hosts, survives recreation, and restores a stopped-volume backup. Exercise all three network modes and record successful store/recall for each target client with exact versions. An upgrade rehearsal is required when two suitable images exist; otherwise record it explicitly as unverified, and prescribe backup before upgrade.

Excluded: sync, web UI, user management, role-based authorization, multi-tenancy, high availability and local LLM consolidation. ARM32 is not part of ARM64 support.

Estimated implementation: 4–7 engineering days, dependent on ARM64 dependencies and client certificate trust behavior. This is an estimate, not a verification result.
