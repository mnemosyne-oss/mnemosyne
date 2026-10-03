# Contributing to Mnemosyne

Mnemosyne is a personal project that grew into something useful. If you're here, you're already part of the community. There are no gatekeepers — bug reports, documentation fixes, feature ideas, and code contributions are all welcome.

## Getting Started

```bash
git clone https://github.com/mnemosyne-oss/mnemosyne.git
cd mnemosyne

# Core tests (matches the CI default and avoids embedding-model downloads)
pip install -e ".[all,dev]"
MNEMOSYNE_NO_EMBEDDINGS=1 python -m pytest tests/ -v

# Hermes integration tests
pip install -e "./integrations/hermes[dev]"
MNEMOSYNE_NO_EMBEDDINGS=1 python -m pytest integrations/hermes/tests/ -v
```

The Hermes provider is packaged separately from `mnemosyne-memory`, so its
editable install is required before running `integrations/hermes/tests/`.

## What You Can Do

**No contribution is too small.**

- **Report bugs** — Open an issue with steps to reproduce. A clear bug report saves hours.
- **Improve docs** — Typos, unclear explanations, missing examples. If it confused you, fix it.
- **Share your use case** — Open a discussion. Real-world usage shapes the roadmap.
- **Submit code** — See below for guidelines.

## Code Contributions

### Versioning

Mnemosyne uses **Simple Versioning** (`MAJOR.MINOR`, no patch):

- **MINOR** bumps after every iteration: bug fixes, features, docs, refactors.
- **MAJOR** bumps only for significant new functionality (e.g., 1.0 → 2.0).

`__version__` in `mnemosyne/__init__.py` is the single source of truth. `pyproject.toml` reads from it automatically. If you open a PR that changes user-facing behavior, bump the version and add an entry to `CHANGELOG.md`.

### Releasing (maintainers only)

Releases are fully automated via GitHub Actions:

1. Bump `__version__` in `mnemosyne/__init__.py`
2. Commit and push to `main`
3. Tag and push:
   ```bash
   git tag -a v1.X.Y -m "Release v1.X.Y"
   git push origin v1.X.Y
   ```
4. The [release workflow](https://github.com/mnemosyne-oss/mnemosyne/actions/workflows/release.yml) handles the rest:
   - Builds wheel + sdist
   - Creates a [GitHub Release](https://github.com/mnemosyne-oss/mnemosyne/releases) with auto-generated notes
   - Publishes to [PyPI](https://pypi.org/project/mnemosyne-memory/) via trusted publishing (OIDC)

No manual uploads. No API tokens.

### Principles

Mnemosyne is intentionally minimal. Every addition is weighed against these principles:

- **Local-first:** No cloud dependencies, no required API keys.
- **Minimal dependencies:** Prefer the Python stdlib. SQLite is the only database.
- **Zero-config:** Users should not need to edit config files to get basic functionality.
- **Fast:** Sub-millisecond reads and writes on standard hardware.

### Before You Code

1. **Open an issue first** for non-trivial changes. This prevents wasted effort.
2. **Keep it focused.** One PR per logical change.
3. **Add tests.** If you fix a bug or add a feature, include coverage in the appropriate suite: `tests/` for core behavior or `integrations/hermes/tests/` for Hermes-provider behavior.
4. **Update the README** if user-facing behavior changes.
5. **Bump the version** in `mnemosyne/__init__.py` and update `CHANGELOG.md`.

### Review Process

Pull requests are reviewed by the maintainers and merged when they:

- Pass existing tests
- Follow the principles above
- Include a clear description of what changed and why

Either maintainer may review any pull request, and an approval means the code is
correct rather than that the change ships.

Some pull requests raise a question that review cannot settle, because the answer is
a project decision rather than a matter of whether the code works. A pull request
that changes a supported public API, the storage schema, or a configuration contract
gets the `needs-decision` label. Review carries on as normal while the label is
there; the label holds the merge, not the discussion. Applying it early is helpful
rather than obstructive, because it surfaces the question while changing the answer
is still cheap.

See [MAINTAINERS.md](MAINTAINERS.md) for the canonical decision framework, including who has authority over which areas of the codebase.

## Areas of Interest

These are not mandates — just directions where help would be valuable:

- Encrypted backup/sync (optional, user-controlled)
- Additional embedding model support
- Multi-language memory processing
- Better error messages and debugging tools

## Community

- **Issues & bugs:** [GitHub Issues](https://github.com/mnemosyne-oss/mnemosyne/issues)
- **Feature ideas & questions:** [Join our Discord](https://discord.gg/nousresearch) or open an issue

## Contributor License Agreement (CLA)

All new contributions require signing the [Contributor License Agreement](CLA.md). This is effective as of 2026-07-13.

The CLA grants the project a license to use, relicense, and distribute your contributions while you retain full ownership. It is adapted from the Apache Software Foundation's Individual CLA.

Past contributions made before this date remain under the MIT License and are not affected.

By submitting a pull request, you confirm you have read and agree to the CLA.

## License

Mnemosyne is licensed under the MIT License. See [LICENSE](LICENSE) for details.
