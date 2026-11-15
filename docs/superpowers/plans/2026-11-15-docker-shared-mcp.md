# Shared MCP Docker Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Use subagent-driven development only if the user selects that execution method.

**Goal:** Deliver a versioned, non-root Docker deployment through which trusted Hermes, Codex and Claude Code agents use one shared memory on a NAS or VPS.

**Architecture:** Build the checked-out Python package with its MCP and local embedding extras. One process owns the SQLite data volume; clients use Streamable HTTP, named bearer tokens and a common bank. Compose examples provide existing-proxy, direct-LAN and optional Caddy operation without changing core memory semantics.

**Tech Stack:** Python 3.11 slim Linux base, Mnemosyne `[mcp,embeddings]`, Docker Buildx/Compose v2, MCP SDK client, pytest, Caddy 2, GitHub Actions/GHCR (proposed registry).

**Spec:** [Approved scope](../specs/2026-11-15-docker-shared-mcp-design.md). Read both documents before implementation.

## Global Constraints

- Shared memory for trusted agents; named tokens provide identity, not tenant isolation.
- Streamable HTTP at `/mcp`; one Mnemosyne process and one local persistent data volume.
- Local embeddings included; persistent model cache; no local LLM consolidation dependency.
- Non-root runtime; release-source image; controlled dependency/base-image updates.
- Linux AMD64 and ARM64 builds and runtime checks; ARM32 excluded.
- Existing HTTPS proxy, direct LAN HTTP and optional Caddy examples.
- Local Caddy HTTPS uses `tls internal`; persist CA state and document trust in client runtimes.
- No change to existing core visibility, identity, storage or public API semantics.
- Ordinary PR checks never publish images. Official publication is coordinated with the release owner using public governance.
- No sync, UI, user administration, roles, multi-tenancy or HA.

## Review Focus

1. Compose port-list merging must not accidentally expose the loopback default on all interfaces (Task 3 rendered-config checks).
2. LAN certificates must match the actual hostname and be trusted by each client runtime; bearer headers and Host policy must survive the proxy (Tasks 4 and 5).
3. A cold/offline model cache must not produce a falsely healthy service or an endless unexplained startup (Tasks 2 and 3).
4. Non-root ownership must work for fresh volumes and produce actionable failure for unwritable existing mounts (Task 2).
5. Backup/restore must preserve SQLite state and named-token sessions must not mix authors during overlapping requests (Tasks 1 and 5).

## Current baseline and file map

The current Dockerfile installs an unpinned PyPI package rather than the checkout, runs as root and probes a separate `recall()` invocation. Root Compose defaults to SSE. There is no container workflow. `.dockerignore` excludes README and all scripts; source builds and copied runtime scripts need explicit exceptions. FastEmbed already accepts `MNEMOSYNE_FASTEMBED_CACHE_DIR`; MCP supports `MNEMOSYNE_MCP_TOKENS`, `MNEMOSYNE_MCP_ALLOWED_HOSTS`, `MNEMOSYNE_MCP_BANK` and `MNEMOSYNE_DEFAULT_SCOPE`.

Existing unrelated changes were present when this plan was written. At execution time use a clean isolated branch/worktree from the agreed base and preserve those changes. All paths below are relative to the repository currently checked out at `release-prep/`.

| Files | Responsibility |
|---|---|
| `Dockerfile`, `.dockerignore` | Source build, non-root user, cache/data paths, runtime probe |
| `deploy/docker/requirements.lock` | Exact resolved runtime dependencies shared by both architectures |
| `deploy/docker/healthcheck.py` | Authenticated, bounded MCP protocol check of running service |
| `docker-compose.yml`, `deploy/docker/.env.example` | Loopback default, required operator configuration |
| `deploy/docker/compose.lan.yml` | Explicit LAN publication |
| `deploy/docker/compose.proxy.yml` | Shared external proxy-network attachment |
| `deploy/docker/compose.caddy.yml`, `Caddyfile.public`, `Caddyfile.lan` | Optional HTTPS edge and persistent CA state |
| `tests/test_docker_shared_mcp.py` | Real HTTP MCP shared-memory contract |
| `tests/docker/conftest.py`, `tests/docker/test_deployment.py` | Built-container, Compose and lifecycle acceptance |
| `.github/workflows/docker.yml`, `docker-publish.yml` | Read-only CI builds/tests and gated manual publication |
| `deploy/docker/README.md`, existing integration docs and README | Operator guide and client recipes |

## Task 1: Prove the shared-memory service contract

**Files:** Create `tests/test_docker_shared_mcp.py`; reuse `tests/test_mcp_streamable_http.py` and `tests/test_multi_agent_identity.py` patterns. No production-code changes planned.

**Interfaces:** A real server on an ephemeral loopback port, bound internally to `0.0.0.0`, configured with tokens named `hermes`, `codex`, `claude-code`, one bank `default`, global default scope and exact allowed test Host. MCP SDK sessions call `mnemosyne_remember`, `mnemosyne_recall` and list tools.

- [ ] Add tests using independent authenticated HTTP sessions: A stores a unique ordinary-memory marker; B recalls it; assert the returned content and original author. Also test after reopening sessions and with interleaved A/B writes. Inspect existing tool schemas for exact argument names.
- [ ] Assert missing/wrong token returns 401, valid token with disallowed Host is rejected, and conflicting supplied `author_id` cannot write as another agent. Never log token values.
- [ ] Run `python -m pytest tests/test_docker_shared_mcp.py -v`. Existing behavior may already pass: this is a regression contract, so do not invent a failing test. If shared access fails, inspect current scope/session rules and resolve within existing configuration; do not weaken identity or change storage semantics implicitly.
- [ ] Run `python -m pytest tests/test_mcp_streamable_http.py tests/test_multi_agent_identity.py tests/test_s1_mcp_sse_auth.py -q`; require all selected tests to pass with MCP dependencies installed and no unexpected skips. Commit only this task's test file.

## Task 2: Build the non-root source image

**Files:** Modify `Dockerfile`, `.dockerignore`; create `deploy/docker/requirements.lock`, `deploy/docker/healthcheck.py`, `tests/docker/conftest.py`, `tests/docker/test_deployment.py`.

**Interfaces:** Runtime UID/GID `10001:10001`; writable home `/home/mnemosyne`; `MNEMOSYNE_DATA_DIR=/data`; `MNEMOSYNE_FASTEMBED_CACHE_DIR=/cache/fastembed`. Preserve `ENTRYPOINT ["mnemosyne", "mcp"]` and existing stdio usage; Compose selects HTTP. Install from a wheel built from this checkout. Probe path `/opt/mnemosyne/healthcheck.py`, with `main() -> int`, exit 0 only after an authenticated initialize/list-tools exchange with the running server within 10 seconds.

- [ ] Add acceptance cases for image version matching the checkout, UID 10001, fresh named-volume writes, source-specific package behavior, and stdio initialization. Confirm the baseline image fails the source-version/non-root cases.
- [ ] Resolve exact runtime dependency versions compatible with both architectures. Record lock generation command, Python version and update procedure. Include the wheel's direct/transitive requirements; install with constraints without forcing a source build of native dependencies. Pin the Python base by a multi-platform manifest digest. Explicitly include README/LICENSE and runtime probe in build context; exclude secrets, `.env`, private/local instructions, data, caches and build outputs. Prefer allowlisted source COPY instructions.
- [ ] Build the wheel in a build stage and install `[mcp,embeddings]` in the runtime stage with the constraints. Precreate owned empty data/cache directories before `USER`; do not download model weights during the image build. Add version/revision/source/license labels.
- [ ] Implement the HTTP probe using the installed MCP SDK, configured `/mcp`, allowed local request Host and one configured named token. Keep secrets out of arguments/output; use an async overall timeout. For stdio mode disable the HTTP healthcheck explicitly in the documented invocation. Test stopped/unresponsive server returns nonzero; no probe writes memories or loads models.
- [ ] Run `docker build --tag mnemosyne-mcp:pr .` and `python -m pytest tests/docker/test_deployment.py -v --image mnemosyne-mcp:pr`. `--image` is a pytest option defined in `tests/docker/conftest.py`; Docker absence must fail this explicit acceptance run, not silently skip.
- [ ] Add cold-cache model inference/store/recall, warm-cache restart with egress disabled, SIGTERM shutdown, fresh-volume ownership and unwritable bind-mount cases. HTTP health denotes protocol liveness; document that model readiness is demonstrated separately by successful remember/recall. Missing model/network must fail the operation clearly, not count as local embedding success. Commit this task's files after checks pass.

## Task 3: Compose defaults and explicit LAN access

**Files:** Modify `docker-compose.yml`; create `deploy/docker/.env.example`, `compose.lan.yml`, `compose.proxy.yml`; extend `tests/docker/test_deployment.py`.

**Interfaces:** Service `mnemosyne-mcp`, container port 8080, root Compose command `--transport streamable-http --host 0.0.0.0 --port 8080`; default published address `127.0.0.1:8080`. Required variables `MNEMOSYNE_IMAGE`, `MNEMOSYNE_MCP_TOKENS`, `MNEMOSYNE_MCP_ALLOWED_HOSTS`; named volumes mounted at `/data` and `/cache`; bank `default`, global default scope, `restart: unless-stopped`. LAN variable `MNEMOSYNE_BIND_IP` has no wildcard default. Proxy network `MNEMOSYNE_PROXY_NETWORK` must exist when that variant is selected.

- [ ] Add rendered-configuration tests using `docker compose config --format json`. Assert loopback-only default; LAN/proxy overrides replace the port list instead of appending; proxy-only mode publishes no host port. Missing required variables fail before service startup. Establish and document a minimum Compose v2 version supporting the chosen replacement syntax.
- [ ] Update Compose and implement overrides. Require an explicit LAN address for direct access. Do not supply working sample secrets. `.env.example` uses a quoted JSON token map with conspicuous placeholders; operator instructions generate unique random secrets and protect the real `.env` from version control. Test interpolation/JSON roundtrip, including `$` and quoting; recommend hex-generated tokens to avoid these pitfalls.
- [ ] Run container tests with valid and blank tokens/hosts. Assert blank/malformed configuration refuses startup. Exercise cross-agent calls through the published port; assert data/cache survive `up --force-recreate` and no `down -v` appears in routine update instructions.
- [ ] Verify proxy-network attachment and direct-LAN rendered configuration; run the actual LAN connection from a second machine during acceptance. Record runtime versus config-only evidence separately. Commit Compose/config/tests.

## Task 4: Optional Caddy HTTPS variants

**Files:** Create `deploy/docker/compose.caddy.yml`, `Caddyfile.public`, `Caddyfile.lan`; extend deployment tests.

**Interfaces:** Caddy proxies to `mnemosyne-mcp:8080` on a private Compose network; only Caddy publishes host ports in this variant. Hostname supplied as `MNEMOSYNE_HOSTNAME`; `MNEMOSYNE_CADDYFILE` selects the public or LAN file. Persist Caddy `/data` and `/config`. LAN hostname example `mnemosyne.home.arpa`, LAN file uses `tls internal`. The upstream allowed Host remains the external hostname presented by clients.

- [ ] Add config-validation tests: run `caddy validate` for both files and assert rendered Compose publishes no direct Mnemosyne host port. Use a pinned Caddy image tag and manifest digest.
- [ ] Implement both files without rewriting Host or removing Authorization. Caddy uses Docker DNS for upstream, keeps MCP streaming functional and does not publish its admin API.
- [ ] Start LAN HTTPS, export its public root certificate (never its CA private key), and use a client context trusting that root. Assert initialize/store/recall succeeds through HTTPS, wrong bearer token is rejected, and a client without trust fails certificate validation. Verify long-lived streaming and external Host handling.
- [ ] Recreate Caddy with its original volume and assert the CA fingerprint is unchanged. Repeat with a custom hostname and matching DNS/certificate. For public HTTPS record a manual DNS/ACME acceptance run if an authorized domain is available; otherwise clearly mark public issuance unverified. Commit example and tests.

## Task 5: Operator guide, client recipes and recovery

**Files:** Create `deploy/docker/README.md`; modify `README.md`, `docs/integrations/{hermes,codex,claude-code}-mcp.md`; extend deployment tests.

**Interfaces:** All three clients connect remotely to the same `/mcp` URL with their own token and bank `default`. Client verification records exact version, OS/runtime, network mode, CA mechanism and successful tool results. Hermes uses its MCP path, not the local native memory provider.

- [ ] Verify current official client documentation and installed `--help` before writing remote configuration. In particular, replace the existing Codex `.codex/mcp.json` recipe only with verified supported configuration; do not copy its present structure into the new remote example. Check Claude Code HTTP transport/header configuration and Hermes remote MCP support. Cite official pages and record date. Do not claim HTTP or private-CA support from generic SDK tests.
- [ ] Write three setup paths with exact Compose commands relative to repository root, explicit `.env` path and consistent Compose project name. Cover cold-start downloads, measured image/RAM/disk needs, local DNS, host firewall, existing proxy routing, LAN HTTP exposure and CA installation in each actual client runtime. Include a reminder that user tools can alter/delete shared memory and banks are not authorization boundaries.
- [ ] Provide root-CA export/verification steps for Caddy, protected token setup/rotation (restart required for static configuration), and TLS renewal/CA-volume preservation. Show environment-variable secret references where the client supports them. Never prescribe disabling TLS verification as the solution.
- [ ] Define a stopped-service backup/restore procedure: stop writers, archive complete `/data` with ownership, start service, restore into a separate fresh volume and verify the marker. Cache is rebuildable; preserve Caddy CA data separately as sensitive operator material. Automate the backup/restore rehearsal in the container acceptance suite.
- [ ] Manually verify Hermes, Codex and Claude Code with exact versions: list tools, store unique marker, recall another client's marker, repeat over Caddy LAN HTTPS. Also record direct-HTTP support or explicit client refusal. If a required target path is unsupported, report the concrete limitation and agree a supported path before calling the PR complete.
- [ ] Document pinned-version update/recreate with backup first. If two suitable source-bound images exist, run old-to-new upgrade and verify memories/identity; otherwise explicitly mark upgrade compatibility unverified. Do not promise that restoring an older executable alone rolls back a migrated database. Commit guide and related tests.

## Task 6: CI, publication preparation and final review

**Files:** Create `.github/workflows/docker.yml`, `.github/workflows/docker-publish.yml`; update deployment guide.

**Interfaces:** CI builds/tests pull requests with no registry writes. Manual publication takes a release tag, validates it against `mnemosyne.__version__`, builds that exact trusted source and pushes `ghcr.io/mnemosyne-oss/mnemosyne-mcp:<version>` plus a revision tag. Stable `latest` is updated only for approved non-prereleases. No arbitrary branch input and no `pull_request_target` execution of PR code with secrets.

- [ ] Add workflow/config checks for event/permission separation, mandatory release-tag/version agreement and prerelease handling. Pin third-party Actions to full commit SHAs, following existing repository practice.
- [ ] Implement AMD64 and ARM64 build/runtime jobs, preferably native runners. If using emulation, record it and require a real ARM64 NAS/server smoke test before asserting support. Execute model inference on both architectures; do not infer support merely from successful build.
- [ ] Run MCP contract, image/lifecycle and rendered-Compose checks in CI. Isolate caches by dependency/model version and architecture; never cache test data or credentials. Upload sanitized test summaries, image sizes and runtime measurements, not memory databases or tokens.
- [ ] Prepare `workflow_dispatch` publication with minimal `contents: read`, job-scoped `packages: write`, immutable source verification and a named release environment. Coordinate environment protections, registry ownership/visibility and release activation with AJ using public MAINTAINERS.md. Do not claim the environment is approval-gated until its protection is confirmed. If protections are unavailable, leave publication disabled pending release-owner decision.
- [ ] Re-run targeted host tests and container suite once after integration; validate all Compose variants and confirm required tests actually executed. Run `python scripts/generate-docs.py --check` and `python scripts/verify-docs.py`, plus applicable existing lint checks. Record unrelated baseline failures separately.
- [ ] Review `git diff --check`, final changed-file list, excluded context/secrets and public PR description. Ensure no private governance files or unrelated existing changes enter the PR. Commit only implementation files. Open/push/merge/publish only under a subsequent concrete instruction.

## Delivery and estimate

Deliver one reviewable branch/PR with tested source-bound image, Compose variants, CI and client/operator documentation. Attach a compact evidence table: platform, client version, network mode, store/recall, restart, restore, and upgrade/public-ACME status. A runnable local source build supports review before registry activation.

Estimated 4–7 engineering days. Suggested order: contract/image 1–2 days; Compose/Caddy 1–2 days; client validation/recovery 1–2 days; CI/review about 1 day, with overlap. ARM64 wheels, certificate trust and remote-client support can change that estimate; stop to report a concrete incompatibility rather than silently expanding core scope.

## Plan self-review

- Shared access/identity: Task 1; source build/non-root/cache/liveness: Task 2.
- All network modes and accidental exposure: Tasks 3–4.
- Target clients, backup/restore, update caveat: Task 5.
- Both architectures, versioning, PR-only CI and release coordination: Task 6.
- Five review-focus conditions each have explicit owning checks. External client/domain/hardware evidence is distinguished from automated protocol/config tests.
- Implementation decisions introduced by this plan: Python 3.11 base, UID/GID 10001, paths, lock file, filenames, variable names, loopback default, manual publish trigger and proposed GHCR name. These are proposed engineering choices, not additional user requirements.
