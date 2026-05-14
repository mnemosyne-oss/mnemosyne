# Mnemosyne DevOps Policy & Procedures

**Effective:** 2026-04-24  
**Applies to:** All commits, releases, and documentation updates for `mnemosyne-memory`  
**Versioning:** Simple Versioning (MAJOR.MINOR) — no patch numbers

---

## 1. Commit Standards

### Conventional Commit Prefixes

Every commit MUST use one of these prefixes:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New user-facing feature or tool |
| `fix:` | Bug fix |
| `docs:` | README, CHANGELOG, or docstring changes only |
| `refactor:` | Code restructuring with no behavior change |
| `test:` | Test additions or fixes |
| `chore:` | Build, CI, packaging, or dependency updates |
| `cleanup:` | Dead code removal, deprecation cleanup |
| `security:` | Security fix or hardening |

**Examples:**
```
feat: add mnemosyne_update and mnemosyne_forget tools
fix: correct auto-sleep dict key from count to total
cleanup: remove dead quantization functions from embeddings.py
docs: align README VEC_TYPE default with beam.py reality
```

### Atomic Commits

- One logical change per commit.
- If a PR touches core + plugin + README, split into atomic commits or squash into a single descriptive commit.
- **Never** mix unrelated changes (e.g., feature + version bump + typo fix) in one commit.

### Commit Messages

- First line: ≤72 characters, imperative mood ("Add" not "Added").
- Body (optional but encouraged): explain WHY, not WHAT. The diff shows what.
- Reference issues/PRs: `(#12)`, `(PR #11 by @rakaarwaky)`.

---

## 2. Pre-Commit Verification Pipeline

**MUST pass before every push to `main`:**

```bash
# 1. Run full test suite
python3 -m pytest tests/ -x -q
# Expected: 17 passed (or current count)

# 2. Verify no secrets leaked
grep -rn "sk-.*[A-Za-z0-9]\{32,\}\|api_key.*=.*['\"][a-zA-Z0-9]\{20,\}" mnemosyne/ hermes_plugin/ hermes_memory_provider/ || echo "No secrets found"

# 3. Verify plugin registration consistency
grep -rn "mnemosyne_" hermes_plugin/__init__.py | grep "register_tool"
# Ensure every tool schema has a matching handler and registration

# 4. Verify version alignment
grep "__version__" mnemosyne/__init__.py
# Must match CHANGELOG top entry

# 5. Verify pyproject.toml syntax
python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))"
```

**If tests fail:** Fix before pushing. No exceptions. CI will block anyway.

---

## 3. Branching Strategy

- **`main`**: Production-ready code only. Protected branch.
- **Feature branches**: `feat/description` or `fix/issue-N-description`
- **No long-lived branches.** Merge to `main` quickly.

### Merge Requirements

1. All tests pass
2. CHANGELOG updated
3. Version bumped (if user-facing)
4. README updated (if behavior changed)
5. Reviewed by at least one person (or self-reviewed with documented reasoning)

---

## 4. Version Bumping

Mnemosyne uses **Simple Versioning**: `MAJOR.MINOR`

### When to Bump

| Change | Bump |
|--------|------|
| New feature, tool, or CLI command | MINOR |
| Bug fix | MINOR |
| Breaking API change | MAJOR |
| Docs-only change | No bump |
| Internal refactor with no behavior change | No bump |
| Dead code removal | MINOR (if release-worthy) |

### Single Source of Truth

**`mnemosyne/__init__.py`** is the ONLY place the version string lives:

```python
__version__ = "2.8.0"
```

`pyproject.toml` reads it dynamically:
```toml
[tool.setuptools.dynamic]
version = {attr = "mnemosyne.__version__"}
```

**Never** hardcode version in `pyproject.toml`. Never.

### Bump Procedure

1. Edit `mnemosyne/__init__.py`
2. Add CHANGELOG entry
3. Commit: `chore: bump version to 2.X`
4. Tag (see Release Process)

---

## 5. CHANGELOG Maintenance

### Format

```markdown
## 1.X.Y

- **Short title** — What changed and why. (PR #N by @author)
- **Another change** — Description. (#issue)
```

### Rules

- Newest entry at the TOP.
- Every user-facing change gets a bullet.
- Reference PRs and authors. Give credit.
- Group related changes under the same version.
- Deprecations MUST be called out explicitly.
- Breaking changes MUST be prefixed with `**BREAKING:**`.

### Example

```markdown
## 2.0

- **Fix `get_working_stats()`** — Now counts ALL working memories globally. (PR #11 by @rakaarwaky)
- **Fix sqlite-vec KNN query** — Inlined LIMIT parameter for xBestIndex compatibility. (#12)
- **Fix triple tools** — Added missing module-level functions, aligned DB paths. (#13)
- **Deprecate `get_global_working_stats()`** — Now aliases `get_working_stats()`.
```

---

## 6. README Updates

Update README.md when:

- CLI behavior changes (new flags, removed flags)
- New tool is added to the plugin
- Default values change (VEC_TYPE, Python version, etc.)
- Installation instructions change
- Architecture claims change (e.g., "No HTTP" → "Optional REST API")

### README Audit Checklist

- [ ] PyPI badge URL reflects current version (add `?v=X.Y` cache-buster)
- [ ] Python version claim matches `pyproject.toml` `requires-python`
- [ ] VEC_TYPE default matches `mnemosyne/core/beam.py:58`
- [ ] Feature list is accurate (no claims for removed/dead features)
- [ ] CLI examples work when copy-pasted
- [ ] Benchmark tables are NEVER removed

---

## 7. Release Process

### Step 1: Final Verification

```bash
git checkout main
git pull origin main
python3 -m pytest tests/ -x -q
```

### Step 2: Version Bump

```bash
# Edit mnemosyne/__init__.py
# Edit CHANGELOG.md
git add mnemosyne/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 1.X.Y"
```

### Step 3: Tag

```bash
git tag -a v1.X.Y -m "Release v1.X.Y

Highlights:
- Bullet 1
- Bullet 2

git log --oneline $(git describe --tags --abbrev=0 HEAD~1)..HEAD"
```

**Tag message MUST include highlights.** This becomes the GitHub Release draft.

### Step 4: Push

```bash
git push origin main
git push origin v1.X.Y
```

### Step 5: GitHub Release (Auto or Manual)

GitHub Actions handles this automatically on tag push:
- Builds wheel + sdist
- Creates GitHub Release with auto-generated notes
- Publishes to PyPI via trusted publishing (OIDC)

**If manual:**
```bash
gh release create v1.X.Y --generate-notes --verify-tag
```

### Step 6: Verify PyPI

```bash
pip index versions mnemosyne-memory
# Should show 1.X.Y as latest
```

---

## 8. Community-Focused Release Messages

### Discord / Telegram / Social

Keep it friendly, technical but accessible, and always credit contributors.

**Template:**

```
Mnemosyne v1.X.Y is live on PyPI!

What's new:
• Feature/fix 1 — one-line explanation
• Feature/fix 2 — one-line explanation

Shoutout to @contributor for PR #N!

pip install --upgrade mnemosyne-memory

Full changelog: https://github.com/AxDSan/mnemosyne/blob/main/CHANGELOG.md
```

### GitHub Release Body

```markdown
## What's New

### Features
- Description of new feature (#PR)

### Fixes
- Description of fix (#issue)

### Deprecations
- `old_function()` is deprecated. Use `new_function()` instead.

## Upgrade
```bash
pip install --upgrade mnemosyne-memory
```

## Contributors
Thanks to @contributor1, @contributor2!
```

### Rules

- Never use corporate-speak. Mnemosyne is a solo-founder indie project.
- Always credit contributors by GitHub handle.
- Always include the `pip install --upgrade` one-liner.
- Always link to CHANGELOG for full details.

---

## 9. Authorship Preservation

### Contributor PRs

When merging contributor code:

1. **Prefer `git merge --no-ff` or `git cherry-pick`** to preserve original commits.
2. **Never squash** without explicit contributor consent — it erases their authorship from git history.
3. If manual fix is needed, use `git commit --amend --author="Contributor Name <email>"`.
4. GitHub "Co-authored-by:" trailer is acceptable for significant collaboration.

### Squash Commits

Only squash when:
- The contributor explicitly asks for it
- It's your own work and you want a clean history
- The PR has 20+ fixup commits that add noise

### Credit in CHANGELOG

Always format as:
```markdown
- **Fix description** — Details. (PR #N by @github_handle)
```

---

## 10. Rollback Procedures

### Scenario: Bad Release on PyPI

PyPI releases are **immutable**. You cannot delete and re-upload the same version.

**Fix:** Bump version and re-release.

```bash
# 1. Revert the bad commit(s)
git revert HEAD  # or git revert <commit_range>

# 2. Fix the issue
# ... edit code ...

# 3. Bump version
# e.g., 1.10.1 -> 1.10.2
vim mnemosyne/__init__.py
vim CHANGELOG.md

# 4. Commit and tag
git add -A
git commit -m "fix: description of the fix"
git tag -a v1.10.2 -m "Hotfix: description"
git push origin main
git push origin v1.10.2
```

### Scenario: Bad Merge to Main

```bash
# Option A: Revert merge commit
git revert -m 1 <merge_commit_hash>
git push origin main

# Option B: Reset (only if nobody else pulled)
git reset --hard <last_good_commit>
git push origin main --force-with-lease
```

### Scenario: Plugin Registration Broken

If `hermes mnemosyne stats` fails after release:

1. Check `hermes_plugin/__init__.py` has `register(ctx)` calling `ctx.register_tool()` for every schema.
2. Check `hermes_plugin/tools.py` has handler functions matching schema names.
3. Check no `ImportError` is swallowed in a broad `except` block.
4. Run: `grep -rn "mnemosyne_" hermes_plugin/ | sort` to verify consistency.

---

## 11. Testing Requirements

### Minimum Bar

- **All 17 tests must pass** before any push to `main`.
- If tests are added, the count increases. New tests must pass too.
- If a test is flaky, fix it. Don't skip it.

### What to Test

- Core memory CRUD (remember, recall, update, forget)
- BEAM consolidation (sleep)
- Cross-session behavior
- Plugin tool handlers (schema validation, error handling)
- TripleStore operations
- Export/import round-trip

### Regression Tests

When fixing a bug, add a test that would have caught it. No exceptions.

---

## 12. Security Checks

### Pre-Push Checklist

- [ ] No API keys, tokens, or passwords in code
- [ ] No `.env` files committed
- [ ] No `print()` statements with sensitive data
- [ ] No hardcoded credentials in test files
- [ ] S3 keys, DB passwords, or service tokens are read from environment variables only

### If Secrets Are Leaked

1. **Rotate the credential immediately** (don't wait for commit revert)
2. Revert or amend the commit to remove the secret
3. Force-push if necessary (coordinate with team)
4. Audit access logs for the exposed credential

---

## 13. Plugin Registration Verification

After ANY change to `hermes_plugin/` or `hermes_memory_provider/`, run:

```bash
# Verify every schema has a handler and registration
python3 << 'EOF'
import ast, sys

# Parse schemas from tools.py
with open("hermes_plugin/tools.py") as f:
    tree = ast.parse(f.read())

schemas = []
handlers = []
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.endswith("_SCHEMA"):
                schemas.append(target.id)
    if isinstance(node, ast.FunctionDef):
        if node.name.startswith("mnemosyne_"):
            handlers.append(node.name)

# Parse registrations from __init__.py
with open("hermes_plugin/__init__.py") as f:
    tree2 = ast.parse(f.read())

registrations = []
for node in ast.walk(tree2):
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                registrations.append(kw.value.value)

# Check consistency
for schema in schemas:
    tool_name = schema.replace("_SCHEMA", "").lower()
    handler_name = "mnemosyne_" + tool_name
    if handler_name not in handlers:
        print(f"MISSING HANDLER: {handler_name} for {schema}")
        sys.exit(1)
    if handler_name not in registrations:
        print(f"MISSING REGISTRATION: {handler_name}")
        sys.exit(1)

print(f"OK: {len(schemas)} schemas, {len(handlers)} handlers, {len(registrations)} registrations")
EOF
```

**This MUST be run after adding any new tool.**

---

## 14. Quick Reference Card

```bash
# Before every push
python3 -m pytest tests/ -x -q        # 1. Tests pass
git diff --stat                       # 2. Review what changed
grep -rn "mnemosyne_" hermes_plugin/  # 3. Plugin consistency check

# Release
vim mnemosyne/__init__.py             # Bump version
vim CHANGELOG.md                      # Add entry
git add -A && git commit -m "chore: bump version to 1.X.Y"
git tag -a v1.X.Y -m "Release v1.X.Y"
git push origin main && git push origin v1.X.Y

# Emergency rollback
git revert HEAD
git push origin main
# Then fix, bump, tag, push
```

---

**Maintainer:** Abdias J (@AxDSan)  
**Last Updated:** 2026-04-24
