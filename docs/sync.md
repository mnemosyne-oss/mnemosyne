# Mnemosyne Sync

**Last updated:** June 2026 &middot; Mnemosyne v3.6.0

> **Bidirectional memory sync between local and remote Mnemosyne instances.**
> Delta-based, event-sourced, with optional client-side encryption.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Sync Protocol](#sync-protocol)
- [Conflict Resolution](#conflict-resolution)
- [Authentication](#authentication)
- [Encryption](#encryption)
- [Deployment Examples](#deployment-examples)
- [Security Model](#security-model)
- [Limitations](#limitations)
- [Roadmap](#roadmap)

---

## Overview

Mnemosyne Sync enables **bidirectional, delta-based synchronization** between two or more Mnemosyne instances. It's designed for:

- **Desktop + VPS** &mdash; Run Mnemosyne locally during the day, sync to a VPS so your deployed agents have the same memory context
- **Team collaboration** &mdash; Share memory context across team members' instances (with appropriate access controls)
- **Backup & redundancy** &mdash; Keep a remote copy of your memory database
- **Cloud readiness** &mdash; Foundation for future Mnemosyne Cloud (BYOK-style hosted service)

### Key Features

| Feature | Status |
|---------|--------|
| Bidirectional delta sync | v3.6.0 |
| Pull-only / push-only modes | v3.6.0 |
| Delta/change-based protocol | v3.6.0 (based on existing DeltaSync) |
| Append-only event log | v3.6.0 |
| Timestamp + importance conflict detection | v3.6.0 |
| API key / JWT authentication | v3.6.0 |
| Optional client-side encryption | v3.6.0 |
| TLS guidance + examples | v3.6.0 |
| Conflict resolution (importance + version chains) | Planned |
| Agent-assisted merge | Planned |
| Sync status dashboard | Planned |
| Webhook triggers | Planned |

---

## Architecture

```
┌──────────────────┐                    ┌──────────────────┐
│  Local Instance  │                    │  Remote Instance │
│                  │    HTTP (TLS)      │                  │
│  ┌────────────┐  │  ◄─────────────►  │  ┌────────────┐  │
│  │ DeltaSync  │  │   pull_changes    │  │ DeltaSync  │  │
│  │ + Events   │  │   push_changes    │  │ + Events   │  │
│  └─────┬──────┘  │                    │  └─────┬──────┘  │
│        │         │                    │        │         │
│  ┌─────▼──────┐  │                    │  ┌─────▼──────┐  │
│  │ SQLite DB  │  │                    │  │ SQLite DB  │  │
│  │ (local)    │  │                    │  │ (remote)   │  │
│  └────────────┘  │                    │  └────────────┘  │
└──────────────────┘                    └──────────────────┘
```

### Data Model

Sync is built on an **append-only event log** (`memory_events` table):

```sql
CREATE TABLE memory_events (
    event_id TEXT PRIMARY KEY,       -- UUID v4
    memory_id TEXT NOT NULL,          -- References the memory being modified
    operation TEXT NOT NULL,          -- CREATE | UPDATE | DELETE | CONSOLIDATE
    timestamp TEXT NOT NULL,          -- ISO 8601 (UTC)
    device_id TEXT NOT NULL,          -- Originating device identifier
    payload TEXT,                     -- JSON content (encrypted if --encrypt)
    parent_event_ids TEXT,            -- JSON array of causal parent event IDs
    importance REAL DEFAULT 0.5,      -- For conflict resolution
    expiry TEXT,                      -- Optional ISO 8601 expiry
    event_hash TEXT,                  -- SHA-256 of (memory_id || operation || payload) for integrity
    synced_at TEXT                    -- When this event was synced to remote
);
```

Events are **immutable** &mdash; once written, they are never modified. Synchronization works by exchanging new events since the last checkpoint.

### Sync Flow

```
1. Client calls pull_changes(since=<token>) on remote
2. Remote returns all events after <token>, plus new token
3. Client applies remote events to local event log
4. Client calls push_changes(events=[...]) on remote
5. Remote applies client's events, returns confirmation
6. Both sides advance their sync checkpoints
```

Each exchange is idempotent &mdash; events carry unique `event_id`s and are deduplicated on receipt.

---

## Quick Start

### Prerequisites

- Mnemosyne v3.6.0+ installed on both instances
- Network connectivity between instances (or via SSH tunnel)
- (Recommended) TLS certificate for the remote endpoint

### 1. Set Up the Remote Instance

```bash
# On your VPS / remote machine
mnemosyne sync serve --port 8765 --api-key "your-secret-api-key"
```

This starts a sync server listening on port 8765.

> **Security:** The sync server is a management API, not a general HTTP service. It exposes only two endpoints: `POST /sync/pull` and `POST /sync/push`. Always protect it behind a reverse proxy with TLS.

### 2. Configure the Local Instance

```bash
# On your local machine
export MNEMOSYNE_SYNC_API_KEY="your-secret-api-key"

# Test the connection
mnemosyne sync status --remote https://my-vps.example.com:8765
```

### 3. Run a Sync

```bash
# Bidirectional sync (default)
mnemosyne sync --remote https://my-vps.example.com:8765

# Pull only (fetch remote changes without pushing local)
mnemosyne sync --remote https://my-vps.example.com:8765 --mode pull

# Push only (send local changes without fetching)
mnemosyne sync --remote https://my-vps.example.com:8765 --mode push
```

### 4. With Client-Side Encryption

```bash
# Generate an encryption key
mnemosyne sync generate-key > mnemosyne-sync.key

# Sync with encryption
MNEMOSYNE_SYNC_KEY=$(cat mnemosyne-sync.key) \
  mnemosyne sync --remote https://my-vps.example.com:8765 --encrypt
```

---

## CLI Reference

### `mnemosyne sync`

```
Usage: mnemosyne sync [OPTIONS]

Synchronize memories with a remote Mnemosyne instance.

Options:
  --remote TEXT          Remote URL (e.g., https://my-vps:8765)  [required]
  --mode TEXT            Sync mode: bidirectional, pull, push  [default: bidirectional]
  --encrypt              Enable client-side payload encryption
  --prompt-key           Prompt for encryption key interactively
  --api-key TEXT         API key for remote authentication
  --insecure             Skip TLS certificate verification (NOT for production)
  --interval SECONDS     Continuous sync interval (0 = one-shot)  [default: 0]
  --help                 Show this message and exit

Security Notice:
  ╔════════════════════════════════════════════════════════╗
  ║ You are responsible for the content you sync.         ║
  ║                                                        ║
  ║ When --encrypt is NOT set, memory content is sent      ║
  ║ in plaintext over the wire (TLS-protected).            ║
  ║                                                        ║
  ║ When --encrypt IS set, payloads are encrypted          ║
  ║ client-side before transmission. The remote side       ║
  ║ cannot read your memory contents.                      ║
  ║                                                        ║
  ║ Always use HTTPS/TLS in production. See                ║
  ║ docs/security.md for the full security model.          ║
  ╚════════════════════════════════════════════════════════╝
```

### `mnemosyne sync serve`

```
Usage: mnemosyne sync serve [OPTIONS]

Start a sync server for remote Mnemosyne instances.

Options:
  --port INTEGER          Port to listen on  [default: 8765]
  --host TEXT             Host to bind to  [default: 127.0.0.1]
  --api-key TEXT          API key for authentication  [default: none]
  --jwt-secret TEXT       JWT secret for authentication  [default: none]
  --tls-cert TEXT         Path to TLS certificate file
  --tls-key TEXT          Path to TLS key file
  --help                  Show this message and exit
```

### `mnemosyne sync status`

```
Usage: mnemosyne sync status [OPTIONS]

Show sync status with a remote instance.

Options:
  --remote TEXT    Remote URL  [required]
  --api-key TEXT   API key for authentication
  --json           Output as JSON
  --help           Show this message and exit

Example output:
  Sync Status
  ────────────
  Remote:      https://my-vps.example.com:8765
  Connected:   ✓
  Mode:        bidirectional
  Encryption:  enabled (XChaCha20-Poly1305)
  Last sync:   2026-06-14T15:30:00Z
  Events synced:
    Push:       1,247
    Pull:       892
  Conflicts:   3 (all resolved: latest-wins)
  Pending:     12 local events to push
```

### `mnemosyne sync generate-key`

```
Usage: mnemosyne sync generate-key

Generate a random 32-byte key for client-side encryption.

Outputs a base64-encoded key suitable for MNEMOSYNE_SYNC_KEY.

Example:
  export MNEMOSYNE_SYNC_KEY=$(mnemosyne sync generate-key)
```

---

## Sync Protocol

### Endpoints

#### `POST /sync/pull`

Request:
```json
{
    "since": "2026-06-14T00:00:00Z",     // ISO 8601 cursor
    "device_id": "laptop-ubuntu",
    "limit": 1000,
    "include_payloads": true
}
```

Response:
```json
{
    "events": [
        {
            "event_id": "a1b2c3d4-...",
            "memory_id": "mem_001",
            "operation": "UPDATE",
            "timestamp": "2026-06-14T10:30:00Z",
            "device_id": "vps-prod",
            "payload": "{\"content\": \"User prefers dark mode\"}",
            "parent_event_ids": ["e5f6g7h8-...", "i9j0k1l2-..."],
            "importance": 0.9
        }
    ],
    "next_cursor": "2026-06-14T12:00:00Z",
    "has_more": false
}
```

#### `POST /sync/push`

Request:
```json
{
    "events": [ /* same event schema as above */ ],
    "device_id": "laptop-ubuntu",
    "confirm": true
}
```

Response:
```json
{
    "accepted": 47,
    "duplicates": 2,
    "conflicts": 1,
    "next_cursor": "2026-06-14T12:00:00Z"
}
```

### Event Deduplication

Events are deduplicated by `event_id`. If an event with the same ID already exists on the remote, it is silently skipped (counted as `duplicates` in the response).

### Cursor / Token

The sync cursor is an ISO 8601 timestamp representing the last processed event time. Each response returns a `next_cursor` that the client passes as `since` in subsequent requests. This enables:

- **Incremental sync**: Only new events are exchanged
- **Resumable sync**: If a sync is interrupted, it resumes from the last cursor
- **Parallel sync**: Multiple devices can sync independently

---

## Conflict Resolution

### v1 Strategy: Latest Wins with Importance Tiebreaker

When the same memory is modified on both sides before a sync:

1. **Compare timestamps** &mdash; The event with the later timestamp wins
2. **If timestamps equal** &mdash; The event with higher `importance` wins
3. **If timestamps and importance equal** &mdash; The event from the device with the lexicographically higher `device_id` wins (deterministic tiebreaker)

This is intentionally simple for v1. Conflicts are logged and visible in `mnemosyne sync status`.

### Future: Version Chain Resolution

The event log stores `parent_event_ids` to enable causal conflict resolution (planned for a future release). The TripleStore's existing `valid_from` / `valid_until` / `superseded_by` version chain mechanism provides the foundation.

---

## Authentication

### API Key

The simplest authentication method. Passed as `Authorization: Bearer <key>` header.

```bash
# Server side
mnemosyne sync serve --api-key "sk-mnemo-abc123"

# Client side
export MNEMOSYNE_SYNC_API_KEY="sk-mnemo-abc123"
mnemosyne sync --remote https://my-vps:8765
```

### JWT

For multi-user setups or integration with existing auth systems.

```bash
# Server side
mnemosyne sync serve --jwt-secret "your-jwt-secret"

# Client side
MNEMOSYNE_SYNC_JWT="<token>" mnemosyne sync --remote https://my-vps:8765
```

---

## Encryption

See [docs/security.md](security.md#encryption-in-mnemosyne-sync) for the full encryption documentation.

### Quick Reference

```bash
# Generate a key
mnemosyne sync generate-key

# Sync with encryption
export MNEMOSYNE_SYNC_KEY="<base64-key>"
mnemosyne sync --remote https://my-vps:8765 --encrypt

# Or use a passphrase (key derived automatically)
export MNEMOSYNE_SYNC_PASSPHRASE="your strong passphrase"
mnemosyne sync --remote https://my-vps:8765 --encrypt
```

### Dependencies

Encryption uses **PyNaCl** (libsodium bindings) when installed, or the `cryptography` package as fallback.

```bash
pip install mnemosyne-memory[sync]       # Includes PyNaCl
pip install mnemosyne-memory[all]        # Includes everything
```

If neither is installed, `--encrypt` will print a clear error with install instructions.

---

## Deployment Examples

### Docker Compose (VPS)

```yaml
# docker-compose.yml
version: "3.8"

services:
  mnemosyne-sync:
    image: python:3.12-slim
    container_name: mnemosyne-sync
    restart: unless-stopped
    ports:
      - "127.0.0.1:8765:8765"
    volumes:
      - mnemosyne-data:/data
      - ./mnemosyne-sync.key:/run/secrets/sync.key:ro
    environment:
      - MNEMOSYNE_DATA_DIR=/data
      - MNEMOSYNE_SYNC_KEY_FILE=/run/secrets/sync.key
    command: >
      sh -c "pip install -q mnemosyne-memory[sync] &&
             mnemosyne sync serve --port 8765 --api-key $(cat /run/secrets/sync.key)"

volumes:
  mnemosyne-data:
```

### Reverse Proxy (Caddy)

```caddy
# Caddyfile
memory.example.com {
    reverse_proxy localhost:8765
    # Caddy automatically provisions TLS via Let's Encrypt
}
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 443 ssl;
    server_name memory.example.com;

    ssl_certificate /etc/letsencrypt/live/memory.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/memory.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Fly.io Deployment

```bash
# fly.toml
app = "mnemosyne-sync"

[build]
  image = "python:3.12-slim"

[http_service]
  internal_port = 8765
  force_https = true

[env]
  MNEMOSYNE_DATA_DIR = "/data"

[mounts]
  source = "mnemosyne_data"
  destination = "/data"

[deploy]
  release_command = "pip install -q mnemosyne-memory[sync]"
```

```bash
# Deploy
fly launch --from fly.toml
fly secrets set MNEMOSYNE_SYNC_API_KEY="sk-mnemo-..."
fly deploy
```

### SSH Tunnel (No Public Port)

If you don't want to expose the sync port publicly:

```bash
# On your local machine
ssh -L 8765:localhost:8765 user@your-vps

# On your VPS
mnemosyne sync serve --port 8765 --host 127.0.0.1 --api-key "..."

# On your local machine (tunnel active)
mnemosyne sync --remote http://localhost:8765
```

### Ready-to-Use Configs

Complete deployment configs live in [`deploy/sync/`](../deploy/sync/):

- `docker-compose.yml` &mdash; Sync server + Caddy reverse proxy (automatic HTTPS)
- `Caddyfile` &mdash; TLS termination config
- `fly.toml` &mdash; Fly.io deployment
- `README.md` &mdash; Step-by-step setup + security checklist

---

## Security Model

The complete security model is documented in [docs/security.md](security.md). Key points:

| Concern | Solution |
|---------|----------|
| Data in transit | TLS encryption (HTTPS). `--insecure` flag for dev only. |
| Data at rest on remote | Same as local &mdash; file permissions, disk encryption. |
| Payload confidentiality | Optional client-side encryption (XChaCha20-Poly1305) |
| Authentication | API key or JWT on all endpoints |
| Event integrity | SHA-256 event hashes |
| Replay protection | Unique event IDs + timestamp validation |

---

## Limitations (v3.6.0)

- **No automatic sync scheduling** &mdash; Use `--interval` for continuous sync or cron for periodic sync
- **Simple conflict resolution** &mdash; Latest-wins with importance tiebreaker. Version-chain resolution planned.
- **No multi-master with CRDT** &mdash; Not yet. The event log architecture supports it, but v1 uses leader-based conflict resolution.
- **No WebSocket / real-time push** &mdash; Pull-based. The client initiates all sync exchanges.
- **No selective sync by bank/bucket** &mdash; Syncs all memories. Per-bank sync planned.
- **No sync history / audit log UI** &mdash; CLI-only for now. Dashboard planned.

---

## Roadmap

### Phase 3 (Next)

- Version chain conflict resolution using TripleStore's existing `valid_from/valid_until/superseded_by`
- Agent-assisted merge proposal (via Hermes plugin)
- `mnemosyne sync status` with detailed conflict reporting
- Per-bank sync isolation

### Phase 4 (Future)

- WebSocket real-time sync (push-based)
- Mnemosyne Cloud &mdash; BYOK hosted service built on this sync layer
- Webhook triggers for sync events
- Sync dashboard (Web UI)
- Selective sync (by bank, by scope, by time range)

---

## Path to Mnemosyne Cloud

The sync layer is the foundation for a future hosted service. The architecture was designed from day one so that a hosted Mnemosyne Cloud can offer convenience without compromising the local-first, privacy-by-design principles.

### How the pieces fit

| Building block (shipped now) | Enables (Cloud later) |
|------------------------------|------------------------|
| Append-only event log | Server-side event storage with full history and audit |
| Client-side encryption | BYOK: Cloud stores only ciphertext, never holds keys |
| Metadata/payload separation | Cloud can route and dedupe without reading content |
| API key + JWT auth | Multi-tenant account isolation |
| Conflict resolution | Multi-device merge across a user's fleet |
| Device IDs | Per-device sync state and revocation |

### The BYOK model

Mnemosyne Cloud will follow a **Bring Your Own Key** model, going one step further than Zep's data-at-rest BYOK:

1. **You generate your key locally** (`mnemosyne sync generate-key`). It never touches Cloud servers.
2. **Your client encrypts payloads** before they leave your machine.
3. **Cloud stores opaque ciphertext** plus routing metadata. A Cloud breach exposes metadata (event counts, timestamps, device IDs) but never memory content.
4. **Only your devices decrypt.** Cloud cannot read, train on, or sell your memories because it mathematically cannot decrypt them.

This is the same trust model as the self-hosted sync server with `--encrypt`: the server is a dumb, untrusted relay. Cloud just adds managed uptime, backups, and a web dashboard on top.

### Why build sync first

A hosted service that can read your memories is a liability and a lock-in trap. By shipping client-side encryption as a first-class sync feature *before* any hosted offering, the Cloud product inherits the privacy guarantees instead of bolting them on later. Self-hosters and Cloud users run the same protocol &mdash; the only difference is who operates the relay.

---

## Related

- [Security & Privacy Model](security.md) &mdash; Full security documentation
- [Architecture](architecture.md) &mdash; BEAM memory tiers and data model
- [API Reference](api-reference.md) &mdash; Complete Python API
- [Configuration](configuration.md) &mdash; Environment variables and settings
