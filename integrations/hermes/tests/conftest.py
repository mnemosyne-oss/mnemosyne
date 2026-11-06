"""
Test-environment isolation for the Hermes integration test tree.

The deployment exports the bank map and the data directory.  A test suite
must not inherit them:

* ``MNEMOSYNE_BANKS_MAP`` (and a map inside the data directory) turns the
  per-chat bank routing ON.  Tests that expect a write to land in the private
  profile storage instead see it land in a routed bank -- and the routed bank
  is a bank of the RUNNING deployment, so the test writes into production
  data.
* ``MNEMOSYNE_DATA_DIR`` points at the production data directory, so a test
  that does not pin its own directory writes into production storage.

Both are neutralized here, before any test module is imported.  Tests that
want a specific value set it themselves (``monkeypatch.setenv``), which
overrides whatever this module does.

The data directory is redirected to a per-session temporary directory rather
than removed: the suite resolves a default data directory from ``HOME``, and
the real home directory is not a place a test suite should write to either.

This mirrors the isolation in ``tests/conftest.py``.  The two trees run as
separate pytest invocations in CI, so each needs its own.
"""

import os
import tempfile

os.environ.pop("MNEMOSYNE_BANKS_MAP", None)
os.environ["MNEMOSYNE_DATA_DIR"] = tempfile.mkdtemp(prefix="mnemosyne-test-data-")
