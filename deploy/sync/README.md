# Mnemosyne Sync - Self-Hosting Deployment

Ready-to-use deployment configs for running a Mnemosyne sync server on your own infrastructure.

## What's here

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Sync server + Caddy reverse proxy (automatic HTTPS) |
| `Caddyfile` | Caddy config for TLS termination |
| `fly.toml` | Fly.io deployment config |

## Docker Compose (recommended for a VPS)

```bash
# 1. Generate a private API-key file and point Compose at it
umask 077
mnemosyne sync-generate-key > mnemosyne-sync-api.key
printf 'MNEMOSYNE_SYNC_API_KEY_FILE=%s\n' "$PWD/mnemosyne-sync-api.key" > .env

# 2. Edit Caddyfile - replace memory.example.com with your domain

# 3. Bring it up
docker compose up -d

# 4. Check it's healthy
docker compose ps
docker compose logs mnemosyne-sync
```

Your sync endpoint is now `https://memory.example.com`. Caddy handles TLS automatically via Let's Encrypt.

From a client machine:

```bash
mnemosyne sync --remote https://memory.example.com \
  --api-key-file /path/to/mnemosyne-sync-api.key \
  --encrypt-key-file /path/to/mnemosyne-sync-encryption.key
```

## Fly.io

```bash
fly launch --no-deploy --copy-config
fly volumes create mnemosyne_data --size 1 --region iad
fly secrets set MNEMOSYNE_SYNC_API_KEY="$(mnemosyne sync-generate-key)"
fly deploy
```

Endpoint: `https://mnemosyne-sync.fly.dev`. Fly provides HTTPS via `force_https = true`.

## Security checklist

- [ ] **Always use TLS in production.** Both configs terminate HTTPS at the edge.
- [ ] **Set a strong API key.** Use `mnemosyne sync-generate-key`, keep it in a mode-`0600` file, and avoid command-line secret values.
- [ ] **Enable client-side encryption** with `--encrypt-key-file` if you don't fully trust the host. With encryption, the server stores opaque ciphertext and cannot read your memory content.
- [ ] **Back up your encryption key separately.** Losing it means losing access to encrypted memories.
- [ ] **Restrict the volume.** The SQLite DB on the server holds whatever isn't client-side encrypted.

See [docs/security.md](../../docs/security.md) for the full security model and [docs/sync.md](../../docs/sync.md) for protocol details.
