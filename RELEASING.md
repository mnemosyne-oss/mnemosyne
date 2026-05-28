# Releasing Mnemosyne

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

## Release Process

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

A pre-push hook in <code>.githooks/pre-push</code> validates:

1. Tag format: `vMAJOR.MINOR.PATCH` (e.g. `v3.1.2`, not `v3.1` or `v3.1.2-beta`)
2. Tag matches <code>__version__</code> in <code>mnemosyne/__init__.py</code> (without the `v`)
3. Major bumps cannot skip the minor/patch sequence (no `v3.1.2 → v4.0.0` without a prior 3.x.y release — this prevents accidental jumps)

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
