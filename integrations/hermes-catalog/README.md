# Mnemosyne for the Hermes plugin catalog

This directory is the Hermes **directory plugin** that the catalog installs
(`hermes plugins install mnemosyne-memory`). It is a thin wrapper:

- `plugin.yaml` names the plugin `mnemosyne`, marks it `kind: exclusive` (a
  memory provider, loaded only when `memory.provider: mnemosyne` is set), and
  requires the first Hermes release that carries the whole catalog path
  (`requires_hermes: ">=0.21.4"`: declared Python dependencies installed and
  re-applied after `hermes update`, plus memory providers that leave core
  installed from the catalog automatically). Both landed after 0.21.3, so
  0.21.4 is a floor, not a round number.
- `pyproject.toml` declares the real implementation, the `mnemosyne-hermes`
  package on PyPI, plus `mnemosyne-memory[embeddings]`. Hermes installs both
  into its own venv on install and re-applies them after every `hermes update`.
- `__init__.py` re-exports `register` and `register_memory_provider` from
  the installed package so the directory is loadable.

The package itself lives in `../hermes/` and is released to PyPI from there.
Nothing in this directory is imported by the package or its tests.

## Platform boundary and shipping gates

Both dependency edges exclude Android and Emscripten with the same marker. The
provider shim and early CLI sibling also refuse those platforms before looking
up/importing the installed runtime, even when dependencies are already present.
Static discovery remains possible; provider/tool activation does not.

See [the native integration support contract](../hermes/PLATFORM_SUPPORT.md).
The current `>=0.7.6,<0.8` provider floor is retained, not claimed to contain the
new standalone guards. Before shipping the full guarantee, maintainers must
publish the guard-bearing package, then require that actually published version
in the eventual catalog. Resolver checks do not establish native acceptance.
No version is invented or published by this source repair.

## Wrapper installs

Existing wrapper-mode installs (`mnemosyne-hermes install --mode wrapper`)
also live at `$HERMES_HOME/plugins/mnemosyne`. The two are the same plugin
name on purpose: a machine has one or the other, and a catalog install onto
an existing wrapper is refused by Hermes with "already exists" rather than
silently replacing it. Remove the wrapper first if you want to switch.
