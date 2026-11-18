# Sync Tutorial: 10-Minute Setup

This guide walks you through setting up bidirectional memory sync between a local machine and a VPS, with optional client-side encryption.

**What you'll need:**

- Two machines with Mnemosyne v3.6.0+ installed (local + remote). Sync landed in 3.6.0.
- Network connectivity between them
- (Recommended) A domain with DNS pointing to your VPS

---

## Step 1: Install on both machines

```bash
# Both machines
pip install --upgrade "mnemosyne-memory[embeddings,sync]"
```

The `[sync]` extra pulls in cryptography for encryption support. It's optional but recommended.

---

## Step 2: Set up the remote (VPS)

### 2a: Generate an API key

```bash
mnemosyne sync-generate-key
```

Output:

```
n4V8xL2qK7mW9pR3tY6bA1jF5cH0dG8e
```

Save this. It's both your API key and (optionally) your encryption key.

### 2b: Start the sync server

```bash
export MNEMOSYNE_SYNC_TOKEN="n4V8xL2qK7mW9pR3tY6bA1jF5cH0dG8e"
mnemosyne sync-serve --host 0.0.0.0 --port 8765 --api-key "$MNEMOSYNE_SYNC_TOKEN"
```

You should see:

```
Mnemosyne Sync Server
  Host: 0.0.0.0
  Port: 8765
  Auth: Bearer token
```

The server is now listening. Test it:

```bash
curl -H "Authorization: Bearer n4V8xL2qK7mW9pR3tY6bA1jF5cH0dG8e" \
     http://localhost:8765/sync/status
```

## Step 2c: Put TLS in front (production)

**Don't expose the sync server directly to the internet.** Use a reverse proxy.

**Caddy (easiest):**

```caddy
memory.example.com {
    reverse_proxy localhost:8765
}
```

**Nginx:**

```nginx
server {
    listen 443 ssl;
    server_name memory.example.com;
    ssl_certificate /etc/letsencrypt/live/memory.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/memory.example.com/privkey.pem;
    location / {
        proxy_pass http://127.0.0.1:8765;
        proxy_set_header Host $host;
    }
}
```

**Docker Compose:** Copy [deploy/sync/docker-compose.yml](https://github.com/mnemosyne-oss/mnemosyne/blob/main/deploy/sync/docker-compose.yml) and the Caddyfile. Edit the domain. `docker compose up -d`.

**Fly.io:** `fly launch --copy-config` then `fly deploy`.

---

## Step 3: Sync from your local machine

### Plaintext mode (TLS only)

```bash
export MNEMOSYNE_SYNC_TOKEN="n4V8xL2qK7mW9pR3tY6bA1jF5cH0dG8e"
mnemosyne sync --remote https://memory.example.com
```

Output:

```
Sync to https://memory.example.com
  Mode: bidirectional
  Push:
    Accepted:   47
    Duplicates: 0
    Conflicts:  0
  Pull:
    Events fetched: 12
    Accepted:       12
    Duplicates:     0
    Conflicts:      0
```

That's it. Your local and remote instances are now synced.

### Encrypted mode (recommended)

The remote server stores only ciphertext. It cannot read your memories.

```bash
export MNEMOSYNE_SYNC_KEY="n4V8xL2qK7mW9pR3tY6bA1jF5cH0dG8e"
mnemosyne sync --remote https://memory.example.com --encrypt
```

**Important:** Use a different key for `MNEMOSYNE_SYNC_KEY` than your API key, or generate a dedicated encryption key:

```bash
mnemosyne sync-generate-key
# Store this separately from your API key
export MNEMOSYNE_SYNC_KEY="*** from above>"
```

### Using a passphrase instead of a raw key

```bash
export MNEMOSYNE_SYNC_PASSPHRASE="your strong memorable passphrase here"
mnemosyne sync --remote https://memory.example.com --encrypt
```

The key is derived using Argon2id (or PBKDF2 with 600K iterations as fallback).

---

## Step 4: Verify it worked

### Check sync status

```bash
mnemosyne sync-status --remote https://memory.example.com
```

Output:

```
Mnemosyne Sync Status

  Device ID:        device-a1b2c3d4
  Total events:     59
  Unique devices:   2
  Last event:       2026-06-14T15:30:00Z
  Last sync:        2026-06-14T15:30:05Z
  Synced events:    59

  Operations breakdown:
    CREATE: 47
    UPDATE: 10
    DELETE: 2

  Remote:           https://memory.example.com
  Remote events:    59
```

### Test: Store a memory locally, recall it from the remote

```bash
# Local
mnemosyne remember "User prefers dark mode" preference 0.9
mnemosyne sync --remote https://memory.example.com

# SSH into VPS
ssh your-vps
mnemosyne recall "dark mode"
# Should return: "User prefers dark mode"
```

---

## Step 5: Continuous sync (optional)

For always-on sync, use `--interval`:

```bash
mnemosyne sync --remote https://memory.example.com --interval 300
```

This syncs every 5 minutes. Press Ctrl+C to stop.

For scheduled sync, use cron:

```bash
# Sync every 30 minutes
*/30 * * * * MNEMOSYNE_SYNC_TOKEN="..." mnemosyne sync --remote https://memory.example.com --encrypt
```

---

## Step 6: Export/import with sync events

Back up everything (memories + sync history):

```bash
mnemosyne export --output backup.json --include-sync-events
```

Import is idempotent — run it multiple times safely:

```bash
mnemosyne import --input backup.json
```

Output:

```
Imported from backup.json
  Working:    +1,247
  Episodic:   +523
  Triples:    +89
  Sync events: +0 inserted, +1,859 skipped (already present)
```

---

## Common patterns

### Desktop to VPS (one-direction backup)

```bash
# Desktop: push only
mnemosyne sync --remote https://vps.example.com --mode push --encrypt

# VPS: pull only
mnemosyne sync --remote https://vps.example.com --mode pull --encrypt
```

### Team sharing (one central relay, multiple clients)

Everyone pushes to and pulls from the same VPS. Each person uses their own encryption key so memories are private by default. The server stores opaque ciphertext for all users.

### SSH tunnel (no public port)

```bash
# On your local machine
ssh -L 8765:localhost:8765 user@vps

# Then sync to localhost
mnemosyne sync --remote http://localhost:8765
```

---

## Next steps

- [Troubleshooting](troubleshooting.md) — common issues and fixes
- [Security & Privacy Model](../security.md) — full threat model and BYOK comparison
- [Sync Protocol Reference](../sync.md) — protocol internals and CLI reference
