# Releasing Mnemosyne

> **Core `mnemosyne-memory` releases use `scripts/release.py`.** It is not
> for the standalone `mnemosyne-hermes` distribution, which has its own
> `0.x` version and release gate below. A core release touches three
> repositories and several places that will not tell you when they are wrong.
> Doing it by hand is how `__version__` reached 3.15.1 with no tag,
> `CHANGELOG` grew a dated `[3.15.0]` heading for a version that never
> shipped, the docs hero read 3.14.0, and the marketing site's release card
> read 3.15.0. Four surfaces, three answers.
>
> ```bash
> python3 scripts/release.py check 3.16.0      # what is not ready
> python3 scripts/release.py prepare 3.16.0    # bump everything, all repos
> python3 scripts/release.py announce 3.16.0   # blog, X, Discord drafts
> python3 scripts/release.py tag 3.16.0        # re-check, print the tag commands
> ```

## What a release touches

| Repository | What changes | Updated by |
|---|---|---|
| `mnemosyne` | `mnemosyne/__init__.py` `__version__` | `prepare` |
| | `CHANGELOG.md`, `[Unreleased]` promoted to a dated section | `prepare` |
| | `hermes_memory_provider/plugin.yaml` `version` | `prepare` |
| | `docs/api/*.mdx` regenerated so they carry the new version | `prepare` |
| `mnemosyne` standalone Hermes package | `integrations/hermes/pyproject.toml` `[project].version` | prepared with the packaged manifest |
| | `integrations/hermes/plugin.yaml` source plugin manifest `version` | must exactly equal the standalone distribution version, runtime version, and packaged `integrations/hermes/src/mnemosyne_hermes/plugin.yaml` |
| | `integrations/hermes/src/mnemosyne_hermes/plugin.yaml` `version` | must exactly match the standalone distribution |
| `mnemosyne-docs` | `version.txt`, which drives the landing hero | `prepare` |
| | `content/api/tool-schema.mdx` | `prepare`, via the docs generator |
| `mnemosyne-website` | `public/llms.txt` "Latest stable" | `prepare` |
| | `src/data/changelog.json` release card | automatic, `sync:changelog` on each build |
| | `content/blog/<release>.mdx` | drafted by `announce`, written by you |
| X / Discord | announcement posts | drafted by `announce`, posted by you |

The sibling repositories must be checked out beside this one. `check` warns
when they are not, rather than silently skipping them.

## Versioning Policy

Mnemosyne follows **strict SemVer** (MAJOR.MINOR.PATCH).

| Bump | When | Example | What you tell users |
|------|------|---------|-------------------|
| **MAJOR** | Breaking API/DB changes, pipeline-breaking migrations | `3.1.2 → 4.0.0` | "Upgrade may break things. Read the changelog." |
| **MINOR** | New features, backward compatible | `3.1.2 → 3.2.0` | "New stuff. Safe to upgrade." |
| **PATCH** | Bug fixes only, zero new behavior | `3.1.2 → 3.1.3` | "Bug fix. Grab it." |

### What counts as what

**Patch (bug fix only):**
- Fixes incorrect behavior (wrong results, crashes, edge cases)
- Performance improvements (same output, faster)
- Error message improvements, logging fixes
- Dependency version bumps for security/compatibility
- Documentation fixes
- Example: "Fix #198 — irrelevant context injection" (PR #199)

**Minor (new feature, backward compatible):**
- New tools, functions, classes
- New env vars, config options
- New optional pipelines or features
- Deprecation warnings (without removing the old thing)
- Example: "Add Spanish language detection" (PR #196)

**Major (breaking change, requires action):**
- Schema migration that requires a re-sync
- Removed functions/classes/tools
- Changed default behavior that alters existing output
- Altered env var semantics (not just adding new ones)
- Changed Python version requirements

### When to release

- **Patches:** As soon as CI is green on main. Bug fixes don't wait.
- **Minors:** Batch if there are multiple in-flight features, or ship solo. No rush.
- **Majors:** Coordinated with changelog, migration guide, and at least one beta cycle.

Release on main branch only. No release branches for older minors (yet).

### Standalone Hermes package releases

`mnemosyne-hermes` is a separate `0.x` distribution. Its release version is
the static `[project].version` in `integrations/hermes/pyproject.toml`; the
packaged `src/mnemosyne_hermes/plugin.yaml` must carry exactly the same value.
`integrations/hermes/plugin.yaml` must exactly equal that standalone
distribution version, the runtime version, and the packaged manifest version.

The `v0.*` tag path intentionally selects only
`build-and-release-hermes` in `.github/workflows/release.yml`; the core release
job explicitly excludes it. Do not change that routing for a routine standalone
release.

AJ owns release authority for this package. A contributor may prepare and test
the version bump, but AJ performs the tag, GitHub Release, and PyPI publication.
Before AJ does so, verify the intended `v<standalone-version>` tag maps to
the standalone version (for example, `0.6.0` maps to `v0.6.0`). The pre-push
hook and the standalone workflow job both verify that tag against
`integrations/hermes/pyproject.toml` `[project].version` before build or
publication; `v0.*` tags are compared only with prior standalone `v0.*` tags.
The workflow then builds from `integrations/hermes`. After the workflow
completes, verify both the GitHub release assets and the `mnemosyne-hermes`
PyPI project show that exact version.

## Core Release Process

### 1. Bump the version

```bash
# Only file that holds the canonical version:
# mnemosyne/__init__.py  →  __version__ = "3.1.2"
```

Update it, commit. PR the version bump separately from code changes.

### 2. Tag and push

```bash
git tag v3.1.2
git push origin v3.1.2
```

Tags MUST match the `__version__` string with a `v` prefix. Our git hook enforces this.

### 3. Let CI do the rest

The <code>.github/workflows/release.yml</code> workflow:
1. Builds the package
2. Creates a GitHub Release with auto-generated release notes
3. Publishes to PyPI

### 4. Write release notes

The auto-generated notes from `generate_release_notes: true` are a starting point. Edit them on the GitHub release page to:

- Call out breaking changes first (if any)
- Thank first-time contributors by name/PR
- Link to the relevant issues each PR fixes
- Note any env var changes or config migrations

## Git Hook (auto-enforced)

A pre-push hook in <code>.githooks/pre-push</code> validates each pushed tag:

1. Tag format: `vMAJOR.MINOR.PATCH` (e.g. `v3.1.2`, not `v3.1` or `v3.1.2-beta`)
2. A `v0.*` standalone tag matches `[project].version` in
   <code>integrations/hermes/pyproject.toml</code>; other tags match
   <code>__version__</code> in <code>mnemosyne/__init__.py</code> (without `v`)
3. Each tag is monotonic only within its own namespace: `v0.*` standalone tags
   or non-`v0` core tags, respectively

Install with:

```bash
git config core.hooksPath .githooks
```

## Changelog

Changelog is generated from GitHub releases. Every release should have:

```
## [3.1.2] - 2026-05-28

### Fixed
- Irrelevant context injection in recall (#198, PR #199)
  - Strict fact matching is now the default
  - Entity prefix similarity requires minimum 30% length ratio
  - Single-token fact queries (5+ chars) now work with strict matcher

### Changed
- MNEMOSYNE_STRICT_FACT_MATCH env var removed. Use MNEMOSYNE_LENIENT_FACT_MATCH=1
  to opt back into permissive fact matching.
```
