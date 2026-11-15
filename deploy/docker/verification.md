# Docker deployment verification

Date: 2026-11-15. Current PR base: `81dd2b6`; historical acceptance baseline: `ec5747a`. No image has been published. The initial acceptance below ran locally; subsequent hosted checks are tracked in [PR #1361](https://github.com/mnemosyne-oss/mnemosyne/pull/1361). This record distinguishes actual runtime checks from configuration validation.

## Current PR preparation on Main 4.1.0

The Docker changes were transferred to `main` at `81dd2b662772ddc78fcdbb990be9b5b3adfbd346` for the local PR candidate. Core runtime, provider implementation and standalone release metadata match that Main revision; unrelated Hermes release-preparation commits are not included. Package discovery only adds the deployment-script exclusion.

- Focused MCP/release/wheel selection: **165 passed** on Python 3.11 with the current dependencies.
- Current Core 4.1.0 image, runtime candidate `2001a87`: **17 native AMD64 container tests passed on QNAP in 159.13 seconds**.
- Ruff F/RUF022, actionlint and generated-document/documentation checks passed.
- Complete local suite on runtime candidate `2001a87`: **9432 passed, 91 skipped, one deprecation warning, seven passed subtests** (525.22 seconds). The earlier 14 failures below belong to the historical source/environment and are not failures of this run.
- The final ownership-migration follow-up changes documentation/test coverage only. No runtime/build files changed. The expanded native QNAP selection passed **18/18 tests in 181.37 seconds**, including root-owned data/cache migration, preserved content/authorship and access as UID 10001. Without the documented ownership change the new regression failed, as expected.
- At initial local PR preparation, hosted CI and a new native ARM64 container run had not run. The later hosted Docker checks for head `79b0bab` passed on native AMD64 and ARM64. Latest-head results are tracked in the PR; the older hardware/client measurements below retain their original source revisions.
- The general CI matrix also exercises the helper under Python 3.10. Its healthcheck regression reproduced the unavailable `asyncio.timeout` call, then passed after using the equally bounded `asyncio.wait_for`. The related Python 3.10 selection passed **162 tests**; the Python 3.11 health/release selection passed **10 tests**. Docker keeps Python 3.11 as its image base.
- Hosted AMD64 exposed a Caddy test startup race: the CA file can exist before the TLS listener is ready. The test now waits at most 15 seconds for a CA-verified HTTPS 401 before checking certificate rejection and MCP access, including after recreation. A real local TLS regression reproduces an early-closed handshake and passes with the bounded wait under Python 3.10 and 3.11; the Python 3.11 TLS/health/release selection passed **11 tests**. This changes test synchronization only.


### PR review follow-up

The example token value is now empty, so copying the file cannot enable publicly known tokens. The restore guide includes an isolated loopback-port override, authenticated SDK recall and scoped cleanup. Historical plan statements now match the implemented inference-readiness contract.

Docker acceptance fails on missing Compose/dependencies or skipped tests when explicitly requested. Its tests assert actual bindings/network attachments, required overlay variables, specific authentication errors, default-service write rejection on an unwritable mount and literal `*` Host behavior. Caddy is exercised through the deployed Compose overlay, including its read-only configuration and persistent CA volumes; untrusted TLS must fail with a certificate-verification error. Diagnostics retain Docker stderr/logs and cleanup preserves the primary failure.

The release tests execute the workflow's existing pre-login source/check gates offline against disposable Git histories and a fixture-only `gh` command: lightweight/annotated tags, mismatched versions, unrelated histories and missing/failed/superseded native checks. No release workflow, registry login or image push is invoked. Local review selection: **180 passed on Python 3.11**, with **24 acceptance-gate/release tests passed on Python 3.10**. Hosted results for the follow-up head are tracked in PR #1361. Its first native pass caught two test assumptions: the public CLI intentionally sanitizes unexpected errors, so specific guard diagnostics are checked via the real MCP module while a separate default-entrypoint case verifies the sanitized CLI refusal; Compose startup now waits for the upstream healthcheck before HTTPS assertions.

Two proposed expectations were adjusted to match existing behavior, rather than modifying Core: storage is initialized lazily, so an unwritable mount is checked through a real mutation call; the MCP SDK treats `*` as a literal Host entry, not a global wildcard, so a normal hostname must receive 421. The Python 3.10 healthcheck issue was already fixed before this pass.

The following sections retain the original acceptance and baseline record with its original source revisions.

## Container checks

| Environment | Result |
|---|---|
| Linux ARM64 Docker Engine 29.5.2 in Colima on Apple Silicon | 17 deployment tests passed, native execution |
| Linux AMD64 image on the same host | 17 deployment tests passed under emulation |
| QNAP QTS 5.2.11, Docker Engine 27.1.2-qnap8, native Linux AMD64 | 17 deployment tests passed in 162.04 seconds; source `8a9c86a` |
| Compose 2.39.4 | Loopback, explicit LAN, existing proxy network and Caddy configurations rendered and checked |
| Python/MCP host contract | 165 targeted MCP/release/wheel tests passed, including overlapping authenticated writes |
| Wheel payload | 3 tests passed after explicitly excluding `deploy*` from package discovery |
| Documentation | Generated-doc check and documentation verifier passed |
| Static checks | Ruff F/RUF022 and actionlint 1.7.7 passed |

The 17 deployment tests cover required configuration, port replacement, actual token roundtrip (including `$` and quoting), non-root ownership, source package version, missing/invalid authentication configuration, real local model inference, bounded empty-cache/offline readiness rejection, warm-cache inference with the container network disconnected, clean SIGTERM, shared store/recall, persistence after recreation, stopped-volume backup/restore, stdio initialization, rejected untrusted HTTPS certificates, Caddy config validation and CA persistence after Caddy container replacement. The separate readiness command requires a nonempty embedding vector; store/recall alone can use keyword fallback. Exit 143 is accepted as normal SIGTERM in addition to 0; forced-kill exit is not accepted.

Observed image sizes from `docker image inspect`: ARM64 126,016,364 bytes; AMD64 129,573,797 bytes. These are engine-reported image sizes, not registry transfer sizes. A warm synthetic-data service used approximately 267 MiB RAM in one snapshot. These are observations for the default model, not capacity guarantees or minimum sizing. Allow additional space for model downloads, growing memories, cache, backups and Docker's own storage.

## Additional native AMD64 and real LAN acceptance

The native AMD64 run on QNAP QTS 5.2.11 passed all 17 tests without architecture emulation. The host reports Linux x86_64 and Docker Engine 27.1.2-qnap8 (`amd64`). Because Python was not in the host SSH path, the test harness ran in a disposable container using Python 3.11.18, Docker CLI 27.1.2 and Compose 2.29.2; Container Station's installed Compose reports 2.29.1-qnap2. Host Python, Docker configuration and existing applications were not changed. The source image was built locally with revision label `8a9c86a`; its engine-reported size was 377,163,301 bytes. A warm synthetic-data service used approximately 330 MiB RAM in one snapshot; this is not a minimum sizing guarantee. This is a separate observation from the Colima measurements, not a registry transfer size.

A further MCP SDK probe connected from a Mac directly over the home LAN, without an SSH forwarding tunnel. HTTP and Caddy private-CA HTTPS both passed discovery, reciprocal store/recall by two named-token identities and author attribution. Wrong/missing tokens were rejected; the private CA required explicit trust. Real local embedding inference passed before the probe. HTTPS used `mnemosyne.home.arpa`, resolved to the target only within the test process; no OS/router DNS configuration was changed. The certificate name was verified normally.

An exploratory bare-IP HTTPS request failed at the TLS handshake: Python omits SNI for literal IP addresses. An OpenSSL comparison succeeded with explicit IP SNI and failed without it. The supplied Caddy example therefore continues to require the documented DNS hostname; IP-only HTTPS is not claimed as supported by that example. No application or Caddy code change was needed for the documented hostname path.

## Actual client probes

The same temporary ARM64 service was reached over loopback-only SSH forwards from another Mac. HTTPS used Caddy's internal CA with `localhost` certificates; trust remained enabled throughout. Synthetic identities were `hermes`, `codex` and `claude-code`, with different tokens.

| Client | Version/method | HTTP | Private-CA HTTPS |
|---|---|---|---|
| Hermes | Agent v0.23.1, native installed `MCPServerTask` transport, Python 3.14.9 | Discover/store/cross-agent recall passed | Passed using `ssl_verify` CA path |
| Codex | CLI 0.169.0-alpha.11.1, ephemeral app-server thread with MCP tool-call RPC, no model invocation | Initialize/discover/store/recall passed | Passed using `CODEX_CA_CERTIFICATE` |
| Claude Code | 2.1.325, limited print-mode session with only the two memory tools | Store and recall of Codex memories passed | Passed using `NODE_EXTRA_CA_CERTS`; stored author was `claude-code` |

These probes test the clients' real MCP implementations; they do not assert compatibility with every older version, GUI environment inheritance or native Hermes memory-provider behavior. Documentation/API references were checked on the same date.

## Final branch review

A fresh reviewer identified two acceptance gaps: store/recall could falsely imply embedding readiness, and the named-token writes were sequential. Both were corrected in one fix pass. The dedicated readiness command now requires actual inference with a 120-second bound; empty-cache/offline execution fails explicitly. The HTTP contract now issues concurrent writes and verifies both authors. Final deployment runs passed 17/17 on native ARM64 and 17/17 under AMD64 emulation, and 165 targeted host tests passed. No minor findings were deferred.

## Wider suite and baseline comparison

The first complete host run reported **7693 passed, 213 skipped, 15 failed**. One failure was introduced by deployment Python files being discovered as an installable namespace: `tests/test_core_wheel_contents.py::test_core_wheel_top_level_names_are_allowlisted`. Adding `deploy*` to the existing packaging exclusion fixed it; all three wheel tests then passed.

The other 14 failures were reproduced separately against an unchanged local archive of `ec5747a`, using the same test environment (14 failed, 1 passed, 2 skipped in that comparison). Runtime source, Hermes provider code and release-hook code are unchanged by this branch. They are not silently treated as passing:

- `tests/test_beam_e3_additive_sleep.py::TestE3AdditiveSleep::test_consolidated_at_is_iso_timestamp`
- `tests/test_beam_vec_admission.py::TestRecallAdmission::test_admission_is_stable_across_top_k`
- `tests/test_beam_vec_admission.py::TestCacheKeyTracksNormBit::test_key_changes_when_marker_toggled`
- `tests/test_c4_recall_diagnostics.py::TestPolyphonicFallbackDiagnostics::test_engine_no_em_fallback_on_deterministic_fast_path`
- `tests/test_hermes_memory_provider_validation.py::test_validate_delete_removes_vec_working_row[hermes_memory_provider]`
- `tests/test_hermes_memory_provider_validation.py::test_validate_delete_removes_vec_working_row[mnemosyne_hermes]`
- `tests/test_release_gates.py::test_standalone_tag_for_the_declared_version_succeeds_against_actual_repo_state`
- `tests/test_release_gates.py::test_hook_keeps_standalone_and_core_namespaces_independent`
- `tests/test_release_gates.py::test_hook_rejects_mismatched_standalone_tag_before_release`
- Five cases of `tests/test_release_gates.py::test_hook_rejects_invalid_static_project_version`: missing `[project]`, missing version, numeric version, duplicate version fields and conflicting version fields.

The release-hook failures on this Mac include the tests' hard-coded `/usr/bin/bash` path being unavailable. The final complete run after the review fixes reported **7694 passed, 214 skipped, 14 failed** (399.81 seconds). Its failure set exactly matches the 14 reproduced baseline failures. The full suite is therefore **not green** in this environment. Core correctness changes belong in their own review, not in the Docker deployment patch.

## Still unverified

- Internet ACME issuance for `Caddyfile.public`; both Caddyfiles passed validation, and internal HTTPS was exercised.
- Router/OS DNS configuration and site-specific NAS bind mounts. Real LAN HTTP/HTTPS traversal passed on QNAP, with process-local resolution of the tested hostname; this does not certify another network or DNS setup.
- Cross-version schema upgrade/downgrade: no suitable prior source-bound image pair was supplied. Backup/restore and same-version recreation passed.
- Registry publication, permissions, release environment protection and immutable image tags after upload. Publication remains disabled by default and is outside this execution.
