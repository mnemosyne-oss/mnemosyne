# Mnemosyne Sync — Full Documentation

Everything you need to sync your memories between machines, with or without encryption.

## Quick Links

- **[Tutorial](tutorial.md)** — 10-minute step-by-step setup. VPS + local. Encrypted and plaintext.
- **[Troubleshooting](troubleshooting.md)** — Common issues, error messages, and fixes.
- **[Sync Protocol](../sync.md)** — Full protocol reference, CLI reference, and architecture.
- **[Security & Privacy Model](../security.md)** — Threat model, encryption internals, BYOK comparison.

## What is Mnemosyne Sync?

Mnemosyne Sync connects two Mnemosyne instances so your memories move with you. Desktop to VPS. Local to remote. Your deployed agent on Fly.io reads the same context your laptop agent wrote this morning.

**It's 3 things:**

1. **An append-only event log** — every memory mutation is tracked forever
2. **A pull/push protocol** — only what changed since last time
3. **Optional client-side encryption** — the server never sees your content

## Which mode should I use?

| Use case | Mode | Command |
|----------|------|---------|
| Desktop + VPS (full sync) | Bidirectional | `mnemosyne sync --remote <url>` |
| Backup only (push to remote) | Push | `mnemosyne sync --remote <url> --mode push` |
| Restore (pull from remote) | Pull | `mnemosyne sync --remote <url> --mode pull` |
| Cloud/sensitive data | Bidirectional + encrypt | `mnemosyne sync --remote <url> --encrypt` |

## Architecture at a glance

```
Local Mnemosyne                    Remote Mnemosyne
+----------------+                 +----------------+
|  working_mem   |                 |  working_mem   |
|  episodic_mem  |    POST /sync/  |  episodic_mem  |
|  triple_store  |<---pull/push--> |  triple_store  |
|  memory_events |                 |  memory_events |
+----------------+                 +----------------+
        |                                  |
   CLI / Python SDK                  CLI / Python SDK
```

## Getting started

Jump into the [tutorial](tutorial.md). It takes 10 minutes: generate a key, start a server, run a sync.

## Self-hosting configs

Ready-to-copy deployment files live in [deploy/sync/](https://github.com/mnemosyne-oss/mnemosyne/tree/main/deploy/sync):

| File | What it does |
|------|-------------|
| `docker-compose.yml` | Sync server + Caddy reverse proxy with auto HTTPS |
| `Caddyfile` | TLS termination config (edit your domain) |
| `fly.toml` | Fly.io deployment with persistent volume |

## Security

Mnemosyne Sync supports optional **client-side encryption**. With `--encrypt`, memory payloads are encrypted on your machine before they leave. The remote server stores opaque ciphertext and cannot read your memories.

What the server sees:

- **Without encryption:** memory content, importance, sources, metadata
- **With encryption:** only event IDs, timestamps, operation types, device IDs

Read the [Security & Privacy Model](../security.md) for the full threat model and BYOK comparison.
