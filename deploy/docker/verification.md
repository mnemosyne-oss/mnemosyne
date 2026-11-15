# Docker deployment verification

Date: 2026-11-15. Source baseline: `ec5747a`. No image was pushed and no GitHub workflow was activated or run remotely. This record distinguishes actual runtime checks from configuration validation.

## Container checks

| Environment | Result |
|---|---|
| Linux ARM64 Docker Engine 29.5.2 in Colima on Apple Silicon | 17 deployment tests passed, native execution |
| Linux AMD64 image on the same host | 17 deployment tests passed under emulation; physical AMD64 execution still belongs to native CI |
| Compose 2.39.4 | Loopback, explicit LAN, existing proxy network and Caddy configurations rendered and checked |
| Python/MCP host contract | 165 targeted MCP/release/wheel tests passed, including overlapping authenticated writes |
| Wheel payload | 3 tests passed after explicitly excluding `deploy*` from package discovery |
| Documentation | Generated-doc check and documentation verifier passed |
| Static checks | Ruff F/RUF022 and actionlint 1.7.7 passed |

The 17 deployment tests cover required configuration, port replacement, actual token roundtrip (including `$` and quoting), non-root ownership, source package version, missing/invalid authentication configuration, real local model inference, bounded empty-cache/offline readiness rejection, warm-cache inference with the container network disconnected, clean SIGTERM, shared store/recall, persistence after recreation, stopped-volume backup/restore, stdio initialization, rejected untrusted HTTPS certificates, Caddy config validation and CA persistence after Caddy container replacement. The separate readiness command requires a nonempty embedding vector; store/recall alone can use keyword fallback. Exit 143 is accepted as normal SIGTERM in addition to 0; forced-kill exit is not accepted.

Observed image sizes from `docker image inspect`: ARM64 126,014,038 bytes; AMD64 129,573,217 bytes. These are engine-reported image sizes, not registry transfer sizes. A warm synthetic-data service used approximately 267 MiB RAM in one snapshot. These are observations for the default model, not capacity guarantees or minimum sizing. Allow additional space for model downloads, growing memories, cache, backups and Docker's own storage.

## Actual client probes

The same temporary ARM64 service was reached over loopback-only SSH forwards from another Mac. HTTPS used Caddy's internal CA with `localhost` certificates; trust remained enabled throughout. Synthetic identities were `hermes`, `codex` and `claude-code`, with different tokens.

| Client | Version/method | HTTP | Private-CA HTTPS |
|---|---|---|---|
| Hermes | Agent v0.23.1, native installed `MCPServerTask` transport, Python 3.14.9 | Discover/store/cross-agent recall passed | Passed using `ssl_verify` CA path |
| Codex | CLI 0.169.0-alpha.11.1, ephemeral app-server thread with MCP tool-call RPC, no model invocation | Initialize/discover/store/recall passed | Passed using `CODEX_CA_CERTIFICATE` |
| Claude Code | 2.1.325, limited print-mode session with only the two memory tools | Store and recall of Codex memories passed | Passed using `NODE_EXTRA_CA_CERTS`; stored author was `claude-code` |

These probes test the clients' real MCP implementations; they do not assert compatibility with every older version, GUI environment inheritance or native Hermes memory-provider behavior. Documentation/API references were checked on the same date.

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

The release-hook failures on this Mac include the tests' hard-coded `/usr/bin/bash` path being unavailable. The full suite is therefore **not green** in this environment. Core correctness changes belong in their own review, not in the Docker deployment patch.

## Still unverified

- Native AMD64 execution on physical AMD64 hardware; the prepared CI matrix covers this when actually run.
- Internet ACME issuance for `Caddyfile.public`; both Caddyfiles passed validation, and internal HTTPS was exercised.
- A real DNS name such as `mnemosyne.home.arpa`, site-specific NAS bind mounts and firewall/router traversal. Explicit LAN port configuration was checked, while actual cross-machine protocol probes used SSH forwards.
- Cross-version schema upgrade/downgrade: no suitable prior source-bound image pair was supplied. Backup/restore and same-version recreation passed.
- Registry publication, permissions, release environment protection and immutable image tags after upload. Publication remains disabled by default and is outside this execution.
