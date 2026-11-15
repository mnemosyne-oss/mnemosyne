# Shared MCP memory with Docker

Run one Mnemosyne service for your own trusted Hermes, Codex and Claude Code agents. Every admitted agent can access the common memory and invoke its mutation/deletion tools. Named tokens identify authors; they are not permissions or tenant boundaries. Use the same `default` bank and global write scope. Different banks are organizational, not authorization boundaries.

## Requirements and first build

- Linux containers, Docker Engine and **Compose 2.24.4+** (`!override` prevents accidentally merging published ports).
- AMD64 or ARM64; ARM32 is outside this deployment's scope. See [verification record](verification.md) for actually tested platforms and clients.
- A local Docker volume for SQLite. Run a single replica; do not put the live database on NFS/SMB or share it between containers.
- Network access for the initial embedding-model download. Model weights are not included in the image. `/cache` retains them across recreation; `/data` retains memories.

From the repository root:

```bash
umask 077
cp deploy/docker/.env.example .env
python3 -c 'import secrets; print(secrets.token_hex(32))'
# Run that command three times; edit .env and replace ALL three token placeholders.
# Set MNEMOSYNE_MCP_ALLOWED_HOSTS to the hostname clients actually send.
docker compose --env-file .env -p mnemosyne -f docker-compose.yml build
docker compose --env-file .env -p mnemosyne -f docker-compose.yml up -d
```

This builds the checkout as `mnemosyne-mcp:local`; no published image is required. Keep `.env` private and untracked. Use the same Compose project name, environment file and selected overrides for every later command. Do not paste `docker compose config` output into issues: it contains resolved tokens.

The container runs as UID/GID `10001:10001`. New named volumes receive the correct ownership. A replacement bind mount must already be writable by that UID; change ownership on the dedicated data directory only, not an arbitrary NAS share. Model initialization can take longer than HTTP startup. Docker health checks authenticated MCP initialize/list-tools, not model readiness. Require a successful explicit inference check before accepting model readiness:

```bash
docker compose --env-file .env -p mnemosyne -f docker-compose.yml exec -T mnemosyne-mcp python /opt/mnemosyne/readiness.py
```

Include the selected override in that command. It must print `Embedding inference ready.` and exit zero. It performs a real inference with harmless probe text, permits up to 120 seconds for download/initialization and fails clearly when no vector is produced. Fix network/cache permissions and retry if initialization fails or times out. Then verify store/recall between agents. Ordinary store/recall can succeed through keyword fallback when the model is unavailable, so neither those calls nor Docker's protocol-only health status establish embedding readiness.

## Choose one network mode

### Existing HTTPS reverse proxy

Base Compose publishes `127.0.0.1:8080` on the server. Configure your host proxy to forward `https://memory.example.org/mcp` to `http://127.0.0.1:8080/mcp`, preserving `Host`, `Authorization`, POST/GET/DELETE and streamed responses. Set `MNEMOSYNE_MCP_ALLOWED_HOSTS=memory.example.org`; include a port if clients use a nondefault port. Disable response buffering and avoid short streaming timeouts.

For an existing container proxy, set `MNEMOSYNE_PROXY_NETWORK` to its existing network and use:

```bash
docker compose --env-file .env -p mnemosyne -f docker-compose.yml -f deploy/docker/compose.proxy.yml up -d
```

Route the proxy to `mnemosyne-mcp:8080` on that network. This override publishes no Mnemosyne host port. Avoid duplicate service DNS aliases if several deployments share the proxy network.

### Direct LAN HTTP

Give the server a stable DHCP reservation. Set `MNEMOSYNE_BIND_IP` to its actual LAN address, for example `192.168.1.20`. Set allowed Hosts to `mnemosyne.home.arpa:8080` when using that DNS name, or `192.168.1.20:8080` when using the IP. Multiple accepted values are comma-separated; `name:*` allows any port for that name. Configure local DNS accordingly.

```bash
docker compose --env-file .env -p mnemosyne -f docker-compose.yml -f deploy/docker/compose.lan.yml up -d
```

Agents connect to `http://mnemosyne.home.arpa:8080/mcp`. HTTP sends tokens and memory text unencrypted. Restrict access to the intended LAN with the host/router firewall; do not configure internet port forwarding. Browser origins remain denied by default; these CLI clients do not require a browser-origin allowlist. Verify from a second LAN machine, not just the server itself.

### Optional Caddy HTTPS

Use `MNEMOSYNE_HOSTNAME=mnemosyne.home.arpa`, set allowed Hosts to that same name, set `MNEMOSYNE_CADDY_BIND_IP` to the server's LAN IP, and select `MNEMOSYNE_CADDYFILE=./deploy/docker/Caddyfile.lan`. Configure local DNS on each client. Use the DNS hostname in HTTPS URLs: bare-IP clients may omit TLS SNI and fail the handshake with this example. Caddy publishes 80/443 on the chosen address; Mnemosyne has no host-published port.

```bash
docker compose --env-file .env -p mnemosyne -f docker-compose.yml -f deploy/docker/compose.caddy.yml up -d
mkdir -p client-ca
docker compose --env-file .env -p mnemosyne -f docker-compose.yml -f deploy/docker/compose.caddy.yml cp caddy:/data/caddy/pki/authorities/local/root.crt client-ca/mnemosyne-root.crt
openssl x509 -in client-ca/mnemosyne-root.crt -noout -fingerprint -sha256
```

Copy **only the public root certificate** to each agent host through a trusted channel and compare fingerprints. Keep the CA private key on the server. Install trust in the actual client runtime, including its container/VM if applicable; trusting a certificate in a browser alone is insufficient. Do not disable TLS verification. Use `https://mnemosyne.home.arpa/mcp` and the client-specific CA settings below.

Caddy's `/data` and `/config` are persistent named volumes. Preserve `/data`: replacing it creates a new CA and requires updating client trust. Restart Caddy after configuration changes (`admin off` disables its administration endpoint).

For public HTTPS, use a publicly resolvable hostname, its matching allowed Host, an appropriate public bind address and `Caddyfile.public`. Its automatic ACME issuance needs DNS and inbound reachability as required by Caddy. This is a separate environment from LAN `tls internal`; no public DNS or internet inbound ports are needed for the LAN variant.

## Connect agents

Use different random bearer tokens for each client. Keep secrets in private environment/configuration, not committed project files. Remote MCP support does not automatically replace an agent's native memory provider. These recipes refer specifically to its MCP tools.

### Hermes

In the active Hermes `config.yaml`:

```yaml
mcp_servers:
  mnemosyne:
    url: "https://mnemosyne.home.arpa/mcp"
    headers:
      Authorization: "Bearer ${MNEMOSYNE_HERMES_TOKEN}"
    ssl_verify: "/absolute/path/mnemosyne-root.crt"
```

Set `MNEMOSYNE_HERMES_TOKEN` privately before starting Hermes. For public HTTPS omit `ssl_verify`; for LAN HTTP change the URL and omit the TLS-only setting. Reload MCP or restart Hermes. The native Mnemosyne Hermes provider is a different integration and is not configured by this example. See the [official MCP configuration reference](https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference/).

### Codex

Use `~/.codex/config.toml` (or your supported project configuration):

```toml
[mcp_servers.mnemosyne]
url = "https://mnemosyne.home.arpa/mcp"
bearer_token_env_var = "MNEMOSYNE_CODEX_TOKEN"
```

Set `MNEMOSYNE_CODEX_TOKEN` privately before launching Codex. For the private Caddy CA, set `CODEX_CA_CERTIFICATE=/absolute/path/mnemosyne-root.crt` in the Codex process environment. Current Codex also supports `SSL_CERT_FILE` as a fallback. Restart the application when changing its environment; GUI applications may not inherit terminal exports. Verify the active version's remote MCP/TLS behavior before relying on a GUI setup. For direct LAN HTTP change the URL.

CLI alternative: `codex mcp add mnemosyne --url https://mnemosyne.home.arpa/mcp --bearer-token-env-var MNEMOSYNE_CODEX_TOKEN`. See [official MCP docs](https://developers.openai.com/codex/mcp/) and [custom CA implementation](https://github.com/openai/codex/blob/main/codex-rs/http-client/src/custom_ca.rs).

### Claude Code

In the project's `.mcp.json`, use an environment reference rather than a literal token:

```json
{
  "mcpServers": {
    "mnemosyne": {
      "type": "http",
      "url": "https://mnemosyne.home.arpa/mcp",
      "headers": {"Authorization": "Bearer ${MNEMOSYNE_CLAUDE_TOKEN}"}
    }
  }
}
```

Set `MNEMOSYNE_CLAUDE_TOKEN` privately. For the private CA, set `NODE_EXTRA_CA_CERTS=/absolute/path/mnemosyne-root.crt` **before launching Claude Code**. Approve the project MCP entry through Claude Code's normal flow. Check `/mcp` or `claude mcp list` for connectivity, then invoke remember/recall. For direct LAN HTTP change the URL. See [official MCP docs](https://code.claude.com/docs/en/mcp) and [network configuration](https://code.claude.com/docs/en/network-config).

### Functional acceptance

Agent A stores `The copper otter project uses vanilla tea.` through `mnemosyne_remember`. Agent B calls `mnemosyne_recall` with `copper otter` and sees the same content with A's author identity. Repeat with the third agent. Tool discovery alone is not proof of store/recall support. Keep the default bank; client-supplied bank/scope choices can change visibility. Record exact client versions and connection mode in the verification record.

## Backup, restore and updates

Stop the service before archiving the **entire** data volume; copying only a live SQLite `.db` can miss journal/WAL state. Keep archives private. The following example uses base Compose; include your selected override for stop/start commands.

```bash
mkdir -p backup
docker compose --env-file .env -p mnemosyne -f docker-compose.yml stop mnemosyne-mcp
docker compose --env-file .env -p mnemosyne -f docker-compose.yml run --rm --no-deps --user 0 --entrypoint tar -v "$PWD/backup:/backup" mnemosyne-mcp -C /data -cpf /backup/mnemosyne-data.tar .
docker compose --env-file .env -p mnemosyne -f docker-compose.yml start mnemosyne-mcp
# Restore to a DIFFERENT project/volume first; do not overwrite your working data.
docker compose --env-file .env -p mnemosyne-restore -f docker-compose.yml run --rm --no-deps --user 0 --entrypoint tar -v "$PWD/backup:/backup:ro" mnemosyne-mcp -C /data -xpf /backup/mnemosyne-data.tar
```

Test the restored project separately on a different port before accepting the backup. Preserve archive ownership (10001) when restoring. `/cache` can be rebuilt; including it avoids another model download. For Caddy, stop it and archive its full `/data` and `/config` volumes separately; these include sensitive CA private-key material, so encrypt/protect that backup and do not distribute it to clients.

Before updating, back up data and record the exact current image digest. Select a fixed new image version or build an exact reviewed release checkout; recreate with the same volumes and network override, then repeat the cross-agent test. Never remove volumes as part of an update. A previous executable may not read a newer database schema; rollback can require restoring the pre-update backup. Cross-version upgrade compatibility is not established by same-version restart tests.

Rotate an agent's token in `.env`, recreate Mnemosyne, update that client's secret and verify the old token is rejected. Static tokens are loaded at startup; this is not a live token-management service. Reconnect clients after recreation.

For direct stdio usage, explicitly disable the HTTP-only probe:

```bash
docker run -i --rm --no-healthcheck -v mnemosyne-stdio-data:/data -v mnemosyne-stdio-cache:/cache mnemosyne-mcp:local
```

## Development checks and controlled updates

```bash
python -m pip install -e '.[test,mcp]' -c deploy/docker/requirements.lock
python -m pytest tests/test_docker_shared_mcp.py tests/test_docker_release.py -q
docker build -t mnemosyne-mcp:pr .
python -m pytest tests/docker/test_deployment.py --image mnemosyne-mcp:pr -v
```

The explicit `--image` run fails if Docker is unavailable. In a normal host-only suite, container tests are skipped; Compose rendering also requires the Compose executable. CI runs native AMD64 and ARM64 jobs. Tests use unique temporary containers, networks and volumes and remove only their own resources.

`requirements.lock` contains exact runtime constraints for Python 3.11 on Linux, checked for both architectures. Regenerate intentionally in a clean Python 3.11 environment with current pip using:

```bash
python -m pip install --dry-run --ignore-installed --only-binary=:all: --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 --platform manylinux2014_x86_64 --python-version 3.11 --implementation cp --abi cp311 --report /tmp/mnemosyne-runtime.json '.[mcp,embeddings]'
python - <<'PY'
import json
from pathlib import Path
rows=json.load(open('/tmp/mnemosyne-runtime.json'))['install']
lines=sorted(f"{r['metadata']['name']}=={r['metadata']['version']}" for r in rows if r['metadata']['name']!='mnemosyne-memory')
Path('deploy/docker/requirements.lock').write_text('# Linux Python 3.11 runtime constraints; regenerate with command in README.\n'+'\n'.join(lines)+'\n')
PY
python -m pip install --dry-run --ignore-installed --only-binary=:all: --platform manylinux_2_28_aarch64 --platform manylinux_2_27_aarch64 --platform manylinux2014_aarch64 --python-version 3.11 --implementation cp --abi cp311 -c deploy/docker/requirements.lock '.[mcp,embeddings]'
```

Review dependency changes, refresh base/Caddy manifest digests deliberately, and rerun both architecture jobs. Update pinned build-tool versions in the Dockerfile as well. Constraints pin versions, not wheel hashes; they do not promise bit-for-bit reproducible images across every platform.

## Publication status

No registry publication is part of this change's execution. `docker-publish.yml` is manual and disabled unless the repository variable `MNEMOSYNE_DOCKER_PUBLISH_ENABLED` is exactly `true`. Before enabling it, the release owner must confirm registry ownership, visibility, environment protection for `docker-release`, successful container checks for the release source and the supported release workflow branch. Release authority follows [MAINTAINERS.md](../../MAINTAINERS.md). Environment protection is an operator prerequisite, not automatically created by the workflow.
