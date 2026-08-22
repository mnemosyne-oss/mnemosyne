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

### Pre-releases

A major needs a beta cycle, so the tag format accepts a PEP 440 pre-release
suffix: `aN`, `bN` or `rcN`. `v4.0.0b1` is valid; `v4.0.0-beta.1` is not.

`__version__` carries the same string, so a beta is `4.0.0b1` in
`mnemosyne/__init__.py` and `hermes_memory_provider/plugin.yaml`, exactly as
`release.py prepare 4.0.0b1` writes it.

Ordering is PEP 440, not `sort -V`:

```
4.0.0a1  <  4.0.0b1  <  4.0.0rc1  <  4.0.0
```

The pre-push hook enforces that ordering in Python rather than with `sort -V`,
which gets it backwards: `sort -V` places `4.0.0` before `4.0.0b1`, so with it
the real release would be rejected as behind its own beta.

pip will not install a pre-release unless asked, so a beta reaches only the
people who opt in:

```bash
pip install --pre mnemosyne-memory
```

The GitHub release is flagged as a pre-release automatically, so a beta never
becomes the repository's "Latest release".

### Beta lifecycle

The mechanics above say how to cut a beta. This says how one ends, which is the
part that drifts if nobody writes it down.

**Name the questions at tag time.** A beta exists to answer specific doubts. Write
them into the release notes when you cut `bN`, because a beta whose questions were
never stated cannot be shown to have finished. For `4.0.0b1` they are:

1. Does the embedding-dimension guard break people in ways we did not predict, and
   is `MNEMOSYNE_EMBEDDING_DIM=<N>` actually sufficient to recover?
2. Is the multimodal surface (`remember_media()`, the modality seam, the media
   store) the right shape? New API is expensive to unship, and a beta is the last
   cheap moment to change a signature.

**A beta with no users is not a beta.** `pip install --pre` is opt-in, so silence is
the default outcome, not a good one. Announce the beta and ask for one specific
thing to be tried. If nobody opts in, promoting to final is a delayed release with
extra steps and no evidence, so say that plainly rather than treating the quiet as
a pass.

#### What blocks promotion

A **beta blocker** is any of:

- A regression against the previous stable line in documented behavior.
- A data-integrity or data-loss defect.
- The documented migration does not work, or is incomplete.
- A crash at import or first call in a supported configuration.

Everything else ships in the final and is fixed in the following patch. A bug in a
brand-new feature is not automatically a blocker; a thing that worked in 3.15.1 and
no longer works always is.

#### Weak signals, do not promote on these

- **Time elapsed.** Two weeks measures nothing.
- **Zero bug reports.** Usually means zero users, not zero bugs.
- **Download counts.** `--pre` traffic is mostly CI and mirrors.

#### Strong signals to promote

- **Someone hit the breaking change and recovered using only the documented
  migration.** This is the strongest single signal available, because it is the
  specific risk the beta exists for. One real report beats a thousand silent
  installs.
- **Someone who did not write the new surface used it against their own data, and
  the shape held.** If the first external use forces a signature change, the beta
  did its job and you owe it another round.
- **No issue opened during the window is a regression from the previous stable
  line.** New-feature bugs are expected; regressions are the thing being tested for.

#### The progression

```
bN   -> bN+1    a beta blocker was fixed. Evidence-based, not scheduled.
bN   -> rc1     zero open blockers, no API signature change in the last beta,
                and the migration exercised by someone other than you.
rc1  -> final   the rc sat with no new blockers, and every downstream surface
                in "Downstream surfaces" is updated and verified.
```

`b` and `rc` are a contract, not decoration: **`b` means the API may still change,
`rc` means it will not.** Any API change after `rc1` forces `rc2` and resets the
promise. If you would not defend that distinction to a user who pinned `rc1`, do
not cut the rc yet.

#### The failure mode

The beta that never ends. It happens when promotion is left to feel finished rather
than to a written test. If you cannot state which of the strong signals you have
observed, you are not ready to promote, and the honest move is to say what evidence
is still missing rather than to ship on fatigue.

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

### 0. Audit what is going out

```bash
git tag -l 'v*' | sort -V | tail -1
git log --oneline vLAST..HEAD --no-merges
```

Classify by Conventional Commits: `feat:` is MINOR, `fix:` / `chore:` /
`docs:` / `perf:` / `test:` are PATCH, `BREAKING CHANGE` or a `!` after the
type is MAJOR.

**The prefix is not sufficient on its own.** 4.0.0 exists because a `fix:`
commit (#521) carried a breaking environment-variable change and shipped a
MINOR bump. Also grep the `[Unreleased]` section for `Breaking` and
`Behavior change` markers, and let those win over the prefix.

Two more checks before you touch anything:

- **Is it already bumped?** A contributor PR may have done it. Check
  `pip index versions mnemosyne-memory` and `git tag -l 'vX.Y*'`. PyPI is
  immutable, so check it first, always.
- **Audit both packages.** Run the audit over `mnemosyne/` and over
  `integrations/hermes/`. If either changed, both ship.

### 1. Bump the version

Use the script. It is the whole point of the script.

```bash
python3 scripts/release.py check 4.0.0      # read-only. run this first, always
python3 scripts/release.py prepare 4.0.0    # writes every surface it knows about
```

Two files in this repository carry the core version, and `prepare` writes
both:

- `mnemosyne/__init__.py` (`__version__`), the canonical value
- `hermes_memory_provider/plugin.yaml` (`version`), which `check` gates on

`prepare` also promotes `[Unreleased]` into a dated section, regenerates
`docs/api/*.mdx`, and writes `mnemosyne-docs/version.txt` and
`mnemosyne-website/public/llms.txt`.

**Do not run `prepare` for a version correction that is not a release.** A
dated heading for a version that has not shipped is the exact drift this
tooling exists to stop. Bump the two version files by hand and leave
`[Unreleased]` alone.

PR the version bump separately from code changes.

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

## CHANGELOG and UPDATING.md

The step that is always underdone. Reconstruct it from git rather than from
memory:

```bash
git log --reverse --format='COMMIT %h%nAuthor: %an%nMessage: %B---' vLAST..HEAD
git show vLAST:CHANGELOG.md | tail -n +9        # then prepend; never overwrite
```

Cover every commit since the last tag, not only your own, and name every
external contributor. Forty commits and four bullets means the entry is
already wrong.

Cross-check the tags against the headings with
`grep -E '^## \[' CHANGELOG.md`. Drift happens; surface it rather than paper
over it. If the CHANGELOG date and the tag date differ by more than a day,
ask before publishing.

`UPDATING.md` needs new environment variables, migration paths, schema
changes, rollback commands, and its intro link moved from the old version to
the new one.

## Downstream surfaces

`prepare` covers two of these. The rest are still manual, tracked in #790.

**`mnemosyne-website`**

- `messages/*.json` sets `home.version`, formatted `"vX.Y.Z: Short Tagline"`.
  Three to five words, and do not reuse the previous tagline.
- `src/data/changelog.json` sets `latest.version`.
- `public/data/changelog.json` must be an identical copy of it.
- `public/llms.txt` carries the latest stable version. `prepare` writes this.

The i18n JSON is mixed-format. Use `json.load` and `json.dump(indent=2)`,
never a regex. A `changelog.json` entry wants eight to twelve substantive
items; a bare version number is a lie detector.

**`mnemosyne-docs`**

- `version.txt` drives the landing hero. `prepare` writes this.
- MDX content carries hardcoded version references. The 3.12.0 release left 42
  of them across 17 files.

```bash
find content -name "*.mdx" -exec sed -i 's/vOLD/vNEW/g' {} +
grep -rn "vOLD" content/ --include="*.mdx"      # must return nothing
```

Touch current-facing pages only: getting-started, installation, quick-start,
tool-schema, and the `src/app/page.tsx` tagline. Leave comparisons, benchmarks
and migration guides alone; those are historical records.

**Standalone plugin**, if `integrations/hermes/` changed. Four files carry the
version and they drift. Patch each with exact strings, because YAML versions
are not safe for bulk regex:

- `integrations/hermes/pyproject.toml`, the source of truth
- `integrations/hermes/plugin.yaml`
- `integrations/hermes/src/mnemosyne_hermes/plugin.yaml`
- `integrations/hermes/src/mnemosyne_hermes/__init__.py`, the one that gets
  forgotten. v3.14.0 caught this.

Ground truth is `pip index versions mnemosyne-hermes`, not git tags. Tagless
manual uploads have happened.

## Build verification

Both Next.js sites must build clean before Vercel deploys them:

```bash
cd ../mnemosyne-docs && npm run build
cd ../mnemosyne-website && npm run build
```

## Announcement

Release announcements are written and posted by the maintainer, and are held
until the release is verified installable. `python3 scripts/release.py announce
X.Y.Z` drafts the blog, X and Discord copy from the changelog section. It
drafts only; it never posts.

## Verify every exit

```bash
curl -sL https://mnemosyne.site | grep 'vX.Y'
curl -sL https://docs.mnemosyne.site | grep 'X.Y'
pip index versions mnemosyne-memory
pip index versions mnemosyne-hermes
```

If a live site still shows the old version it is usually the Vercel CDN rather
than the code. Check `curl -sI <url> | grep -i x-vercel-cache`. A `HIT` with a
high age means the deploy did not re-trigger; push an empty commit to fire the
webhook.

## Security patch releases

A CVE fix differs from a normal release in five ways.

1. Single purpose. The fix and its tests, nothing else. No bundling with
   feature work.
2. The fix must already be on `origin/main`, not a feature branch. Verify
   before tagging.
3. Core bump, GHSA and PyPI are the minimum. Skip the multi-repo cycle unless
   the vulnerability touches those repos.
4. **PyPI before GHSA.** Tag, push, let the release workflow finish, confirm
   the version is installable, and only then publish the advisory:
   `gh api repos/OWNER/REPO/security-advisories/<GHSA> -X PATCH -f state=published`.
   The disclosure window must not open before the fix is reachable.
5. Thirty-day embargo per `SECURITY.md`. Coordinate with the reporter. Discord
   gets the upgrade line only.

Check advisory state with
`gh api repos/mnemosyne-oss/mnemosyne/security-advisories`. A draft advisory at
critical severity means stop and escalate.

## Recurring failure modes

Observed on real releases. Read this list before you start.

- **Stopping at the core tag.** The release is CHANGELOG plus plugin plus
  website plus docs plus announcement. Skipping one gets noticed.
- **A thin CHANGELOG.** Reconstruct from `git log` first, every time.
- **The plugin `__init__.py` version.** It is the fourth file and it is the one
  that gets missed.
- **Lightweight tags.** `--follow-tags` pushes annotated tags only. Always
  `git tag -a`.
- **The pre-push hook on plugin tags.** It validates plugin tags against the
  core version, so plugin tags need `git push --no-verify`. Never skip the hook
  for a core tag.
- **PyPI trusted publishing mismatch.** If the registered publisher and the
  repository owner disagree, a tag push fails with `invalid-publisher`. This
  happens after an ownership or org move. Fix it in PyPI settings, not in CI.

## Git Hook (auto-enforced)

A pre-push hook in <code>.githooks/pre-push</code> validates each pushed tag:

1. Tag format: `vMAJOR.MINOR.PATCH`, with an optional PEP 440 pre-release
   suffix (e.g. `v3.1.2`, `v4.0.0b1`, `v4.0.0rc1`; not `v3.1` or
   `v4.0.0-beta.1`)
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
