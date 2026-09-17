# Mnemosyne for the Hermes plugin catalog

This directory is the Hermes **directory plugin** that the catalog installs
(`hermes plugins install mnemosyne-memory`). It is a thin wrapper:

- `plugin.yaml` names the plugin `mnemosyne`, marks it `kind: exclusive` (a
  memory provider, loaded only when `memory.provider: mnemosyne` is set), and
  requires a Hermes release that installs plugin dependencies.
- `pyproject.toml` declares the real implementation, the `mnemosyne-hermes`
  package on PyPI, plus `mnemosyne-memory[embeddings]`. Hermes installs both
  into its own venv on install and re-applies them after every `hermes update`.
- `__init__.py` re-exports `register` and `register_memory_provider` from
  the installed package so the directory is loadable.

The package itself lives in `../hermes/` and is released to PyPI from there.
Nothing in this directory is imported by the package or its tests.

## Wrapper installs

Existing wrapper-mode installs (`mnemosyne-hermes install --mode wrapper`)
also live at `$HERMES_HOME/plugins/mnemosyne`. The two are the same plugin
name on purpose: a machine has one or the other, and a catalog install onto
an existing wrapper is refused by Hermes with "already exists" rather than
silently replacing it. Remove the wrapper first if you want to switch.
