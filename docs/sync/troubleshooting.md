# Sync Troubleshooting

Common issues, error messages, and fixes for Mnemosyne Sync.

---

## Connection errors

### "Connection refused" or timeout

```
requests.exceptions.ConnectionError: Connection refused
```

**Cause:** The sync server isn't running, the port is wrong, or a firewall is blocking it.

**Fix:**

1. Verify the server is running on the VPS: `ps aux | grep sync-serve`
2. Check the port: `ss -tlnp | grep 8765`
3. Test locally on the VPS: `curl http://localhost:8765/sync/status`
4. Check firewall: `ufw allow 8765` or `firewall-cmd --add-port=8765/tcp`
5. If behind a proxy, test the proxy: `curl https://memory.example.com/sync/status`

---

## Authentication failures

### "401 Unauthorized"

```
HTTP 401 on /sync/pull: {"error": "Unauthorized"}
```

**Cause:** Missing or incorrect API key.

**Fix:**

1. Verify the env var is set: `echo $MNEMOSYNE_SYNC_TOKEN`
2. Match the key exactly to the server's: `mnemosyne sync-serve --api-key "***`
3. Check for trailing whitespace: `echo $MNEMOSYNE_SYNC_TOKEN | wc -c`
4. Ensure the Authorization header is sent: add `--api-key "***` if env var is missing

### JWT issues

```
Invalid JWT: signature verification failed
```

**Cause:** The JWT secret doesn't match, or the token is expired.

**Fix:**

1. Use the same `--jwt-secret` on both server and client
2. Regenerate the token if it expired
3. Fall back to API key auth as a simpler alternative

---

## TLS / HTTPS issues

### "SSL certificate verify failed"

```
SSLCertVerificationError: certificate verify failed
```

**Cause:** The server's TLS certificate is self-signed, expired, or the domain doesn't match.

**Fix:**

1. **Production:** Use a real certificate. Caddy provisions Let's Encrypt automatically. For Nginx, use certbot.
2. **Self-signed or private CA:** Export the CA certificate and point `SSL_CERT_FILE` at it. This is the supported way to trust a non-public certificate, and it applies to both the sync client and the embedding API client.

```bash
export SSL_CERT_FILE=/path/to/ca-cert.pem
# requests-based paths also honour REQUESTS_CA_BUNDLE
export REQUESTS_CA_BUNDLE=/path/to/ca-cert.pem

mnemosyne sync --db-path /path/to/surface.db --remote https://192.168.1.50:8765
```

3. **Plain HTTP on a trusted network:** use an `http://` remote URL rather than `https://`. There is no flag that disables certificate verification on an HTTPS connection; if you were looking for `--insecure`, it does not exist.

---

## Encryption issues

### "No module named 'cryptography'" / "No module named 'nacl'"

```
ImportError: SyncEncryption requires 'cryptography>=41.0'
```

**Cause:** The `[sync]` extras weren't installed.

**Fix:**

```bash
pip install "mnemosyne-memory[sync]"
```

### Encrypted payload errors on the server

```
Failed to apply event abc123: Expecting value: line 1 column 1 (char 0)
```

**Cause:** The server received encrypted ciphertext but has no key to decrypt it. This is expected behavior when the local client uses `--encrypt` but the server doesn't have a key.

**Fix:** Nothing to fix — the server correctly stores the opaque ciphertext. The server should *not* have the encryption key. Only the client that created the event can decrypt it.

### "Decryption failed" on pull

**Cause:** Encryption key mismatch between the device that pushed and the device that pulled.

**Fix:**

1. Verify both devices use the same `MNEMOSYNE_SYNC_KEY`
2. If using passphrases, verify the passphrase is identical
3. Keys are not recoverable — if you lose the key, previously encrypted data is unreadable

---

## Sync hangs or is slow

### Initial sync takes a long time

**Cause:** First sync transfers all memories, not just changes.

**Fix:**

1. Use `--mode push` for the first sync if you have many memories
2. Subsequent syncs are delta-only and should be fast
3. Check network latency: `ping your-vps`
4. If you have >100K memories, increase the limit in the sync server

### "Sync appears stuck"

**Cause:** Often a network timeout, or the server is processing a large batch.

**Fix:**

1. Wait up to 30 seconds (default timeout)
2. Check server logs on the VPS
3. Restart with `--interval` instead of one-shot to see progress per cycle
4. Reduce batch size by syncing fewer memories at once

---

## Conflict resolution issues

### "Duplicate memories" after sync

**Cause:** Two devices created the same memory independently, and both events arrived.

**Fix:** This is expected and handled. Mnemosyne resolves conflicts automatically:
1. Version chain wins (if one event is a parent of another)
2. Latest timestamp wins
3. Higher importance wins
4. Deterministic device_id tiebreaker

Check resolved conflicts:

```bash
mnemosyne sync-status --remote https://memory.example.com --json | jq '.remote_status.pull.conflicts'
```

### Overriding conflict resolution

If the automatic resolution picks the wrong version, you can manually fix:

```bash
# Find the conflicting memory
mnemosyne recall "disputed topic" --json

# Update it with the correct content
mnemosyne update <memory_id> "Correct content" 0.9
```

---

## Server issues

### "Address already in use"

```
OSError: [Errno 98] Address already in use
```

**Cause:** Another process is already using port 8765.

**Fix:**

```bash
# Find the process
ss -tlnp | grep 8765

# Kill it
kill -9 <pid>

# Or use a different port
mnemosyne sync-serve --port 8766
```

### Server doesn't start

**Cause:** Missing Mnemosyne install, wrong Python version, or permission issue.

**Fix:**

1. Verify Mnemosyne is installed: `mnemosyne --help`
2. Check Python version: `python3 --version` (needs 3.9+)
3. Check the data directory exists and is writable: `ls -la ~/.hermes/mnemosyne/data/`
4. Start with verbose logging for debugging:

```bash
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from mnemosyne.core.sync_server import run_sync_server
run_sync_server(port=8765)
"
```

---

## Export/import issues

### "Import failed: Unknown schema version"

**Cause:** The export file was created with `--include-sync-events` and the import doesn't support v1.2 schema.

**Fix:** Upgrade to Mnemosyne v3.7.0 or later:

```bash
pip install --upgrade mnemosyne-memory
```

### Re-import produces data but no new memories

**Cause:** Events were already present (event_hash deduplication).

**Fix:** This is correct behavior. The import is idempotent. Duplicate events are silently skipped.

---

## Still stuck?

- Check the [Sync Protocol Reference](../sync.md) for full CLI reference
- Read the [Security Model](../security.md) for encryption internals
- Open an issue: [github.com/mnemosyne-oss/mnemosyne/issues](https://github.com/mnemosyne-oss/mnemosyne/issues)
- Ask in Discord: the `#mnemosyne` channel
