# Security & Privacy Model

**Last updated:** June 2026 &middot; Mnemosyne v3.6.0

> **You are solely responsible for the content stored in Mnemosyne.**
> Mnemosyne Sync supports optional client-side encryption. When disabled, memory content travels over TLS and is stored according to your infrastructure's security settings.

---

## Philosophy

Mnemosyne is built on three security principles:

1. **User sovereignty** &mdash; You own your data. Mnemosyne has no telemetry, no tracking, no cloud dependency. The database file is yours.
2. **Privacy by design** &mdash; The default configuration stores everything locally. No data ever leaves your machine unless you explicitly enable sync.
3. **Minimal trust in remote infrastructure** &mdash; When sync is enabled, optional client-side encryption ensures that even the remote server cannot read your memory contents. The remote side sees only routing metadata (timestamps, event IDs, device IDs).

---

## Data at Rest

Mnemosyne stores all data in a single SQLite file (default: `~/.hermes/mnemosyne/data/mnemosyne.db`).

| Layer | Data | Security |
|-------|------|----------|
| **Working memory** | Recent agent context, preferences, facts | Plaintext in SQLite. File permissions control OS-level access. |
| **Episodic memory** | Consolidated long-term memories, vector embeddings | Same as working memory. Embeddings are mathematical vectors, not reversible to original text. |
| **TripleStore** | Temporal knowledge graph (subject-predicate-object triples) | Same as working memory. |
| **Config** | API keys, endpoints, encryption settings | Stored in environment variables or config.yaml. Never logged or transmitted. |

**Recommendations for data at rest:**

- Set restrictive file permissions: `chmod 600 ~/.hermes/mnemosyne/data/mnemosyne.db`
- Use full-disk encryption (LUKS, FileVault, BitLocker)
- On shared machines, use memory banks for isolation: `mnemosyne --bank <name>`

---

## Data in Transit (Sync)

When `mnemosyne sync` is enabled, memory content travels between instances over HTTP/HTTPS.

### Without Client-Side Encryption

| What the remote server sees | Example |
|-----------------------------|---------|
| Memory content (plaintext) | `"User prefers dark mode"` |
| Importance scores | `0.9` |
| Memory type / source | `"preference"`, `"cli"` |
| Event metadata | `event_id`, `timestamp`, `device_id`, `operation` (CREATE/UPDATE/DELETE) |
| Vector embeddings | Opaque binary data (not reversible to text, but could be used for similarity analysis) |

**Protection:** TLS encryption between instances (recommended). Behind a reverse proxy (Caddy, Nginx, Traefik) with automatic HTTPS.

### With Client-Side Encryption (`--encrypt`)

| What the remote server sees | Example |
|-----------------------------|---------|
| Memory content | `Encrypted::<base64 ciphertext>` (unreadable without the key) |
| Importance scores | Encrypted alongside content |
| Memory type / source | Encrypted alongside content |
| **Unencrypted metadata** (required for routing) | `event_id`, `timestamp`, `device_id`, `operation` |
| Vector embeddings | Encrypted as part of payload |

**Protection:** Authenticated encryption (XChaCha20-Poly1305 via PyNaCl). The remote server cannot decrypt any payload content. See [Encryption in Mnemosyne Sync](#encryption-in-mnemosyne-sync) below.

---

## Encryption in Mnemosyne Sync

### Overview

Client-side encryption is **optional** but **first-class**. When enabled, memory payloads are encrypted on the sending side before transmission and decrypted on the receiving side after receipt. The remote server (including the Mnemosyne sync endpoint) never has access to the encryption key.

### How It Works

```
┌─────────────────────┐           ┌─────────────────────┐
│   Local Instance    │           │   Remote Instance   │
│                     │           │                     │
│  1. Encrypt payload │  ──────►  │  3. Store encrypted │
│     with local key  │  (TLS)   │     payload as-is   │
│                     │           │                     │
│  4. Request changes │  ◄────── │  2. Return encrypted │
│  5. Decrypt payload │  (TLS)   │     payloads         │
│     with local key  │           │                     │
└─────────────────────┘           └─────────────────────┘
```

The encryption key never leaves the local instance. The remote instance stores and serves encrypted payloads without ever decrypting them.

### Supported Key Sources

| Source | Configuration | Security Level |
|--------|---------------|----------------|
| **Environment variable** | `MNEMOSYNE_SYNC_KEY=<base64-key>` | Medium (env may be logged) |
| **OS keyring** | `MNEMOSYNE_SYNC_KEY_SOURCE=keyring` | High (stored in OS keychain) |
| **Prompt on sync** | `--encrypt --prompt-key` | Highest (never persisted) |
| **Derived from passphrase** | `MNEMOSYNE_SYNC_PASSPHRASE=<phrase>` | Medium (Argon2id or PBKDF2 derived) |

### Key Derivation

When using a passphrase (not a raw key), the key is derived using:

- **Default:** PBKDF2-HMAC-SHA256 with 600,000 iterations
- **If argon2-cffi is installed:** Argon2id (memory-hard, recommended)

```bash
# Generate a random key
mnemosyne sync generate-key
# Output: MNEMOSYNE_SYNC_KEY=7A8B3C... (base64, 32 bytes)

# Or use a passphrase (key derived automatically)
export MNEMOSYNE_SYNC_PASSPHRASE="your strong passphrase here"
```

### What Gets Encrypted

The **payload** field of each sync event is encrypted. Payload includes:

- Memory `content` text
- `importance` score
- `source` field
- `metadata_json` (if present)
- `memory_type` / `veracity` fields
- Vector embeddings (`binary_vector`)

### What Stays Unencrypted (Metadata)

These fields are always sent in plaintext for routing and conflict resolution:

- `event_id` &mdash; Unique event identifier (UUID)
- `memory_id` &mdash; References the memory row
- `operation` &mdash; CREATE / UPDATE / DELETE / CONSOLIDATE
- `timestamp` &mdash; ISO 8601 timestamp
- `device_id` &mdash; Which device originated the event
- `parent_event_ids` &mdash; Causality chain for conflict detection

This means an adversary with access to the remote sync server can learn:
- How many memory events are being synced
- When they occurred
- Which events are related to each other
- Which devices are syncing

They **cannot** learn the content or meaning of those memories when encryption is enabled.

---

## Liability & Disclaimer

> **Mnemosyne is provided "as is", without warranty of any kind.**
>
> You are solely responsible for:
> - The content you store in Mnemosyne
> - Securing your database file (file permissions, disk encryption)
> - Securing your sync channel (TLS certificates, network configuration)
> - Managing your encryption keys (key rotation, backup, loss prevention)
> - Compliance with applicable laws regarding data storage and transfer
>
> Mnemosyne Sync supports optional client-side encryption. When disabled, memory content travels over TLS and is stored according to your infrastructure's security settings. Even with encryption enabled, metadata (event counts, timestamps, device IDs) is visible to the remote server for routing purposes.
>
> See the [LICENSE](LICENSE) file for the full terms.

---

## Comparison with Alternatives

| System | Data at Rest | Data in Transit | Client-Side Encryption | Self-Hostable |
|--------|-------------|-----------------|----------------------|---------------|
| **Mnemosyne** | Local SQLite (file permissions) | TLS + optional client-side encryption | **Yes (sync, XChaCha20-Poly1305)** | Yes |
| **Mem0** | Qdrant/PostgreSQL | TLS | No | Optional |
| **Zep** | BYOK (data at rest) | TLS | BYOK only (server-managed) | Yes |
| **Letta** | PostgreSQL | TLS | No | Yes |
| **Honcho** | PostgreSQL | TLS | No | Yes |
| **Supermemory** | Cloud (SaaS) | TLS | No | Enterprise only |
| **Hindsight** | Local SQLite | TLS | No | Yes |

Mnemosyne is the only memory system with **client-side encryption of sync payloads** as a core feature, not an afterthought. Zep offers BYOK for data-at-rest but manages the key server-side &mdash; Mnemosyne keeps the key entirely client-side.

---

## Threat Model

### In Scope

| Threat | Mitigation |
|--------|-----------|
| Remote server compromise &rarr; memory content exposure | Client-side encryption renders payloads unreadable |
| TLS interception (MITM) | TLS with certificate validation |
| Unauthorized sync access | API key / JWT authentication on sync endpoints |
| Replay attacks | Unique event IDs + timestamp validation |
| Sync of malicious content | Destination-side content filtering via ignore_patterns |

### Out of Scope (v1)

| Threat | Rationale |
|--------|----------|
| Side-channel attacks on encryption timing | Performance-sensitive sync makes constant-time hard. Use TLS for transit security. |
| Quantum cryptanalysis of XChaCha20-Poly1305 | Not a practical threat for memory content. Key rotation mitigates long-term risk. |
| Physical access to the database file | OS-level controls (file permissions, disk encryption) are the user's responsibility |
| Denial of service on sync endpoint | Self-hosted deployments manage their own rate limiting and firewall rules |

---

## Best Practices

1. **Always use HTTPS/TLS** for sync connections. Never sync over plain HTTP outside a local network.
2. **Enable client-side encryption** if syncing over the internet or to a shared/VPS instance.
3. **Rotate sync keys periodically** (e.g., every 90 days).
4. **Back up your encryption keys** separately from your Mnemosyne database. Losing the key means losing access to encrypted memories.
5. **Use environment-specific keys** &mdash; separate keys for development, staging, and production.
6. **Monitor sync events** with `mnemosyne sync status --remote <url>` to verify no unexpected sync activity.

---

## Related

- [Sync Documentation](sync.md) &mdash; Full sync protocol, CLI usage, and deployment
- [Architecture](architecture.md) &mdash; How Mnemosyne's BEAM tiers work
- [Configuration](configuration.md) &mdash; Environment variables and settings
