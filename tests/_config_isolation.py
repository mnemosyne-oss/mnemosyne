"""Shared configuration isolation for Core and Hermes test collections."""

import os
import tempfile
from pathlib import Path

import pytest


_CONFIG_ENV = ("HOME", "HERMES_HOME", "MNEMOSYNE_DATA_DIR")


def pytest_configure(config):
    """Protect collection-time imports, once when both suites are collected."""
    if getattr(config, "_mnemosyne_config_isolated", False):
        return
    previous = {key: os.environ.get(key) for key in _CONFIG_ENV}
    directory = tempfile.TemporaryDirectory(prefix="mnemosyne-pytest-", ignore_cleanup_errors=True)
    root = Path(directory.name)
    os.environ["HOME"] = str(root)
    os.environ["HERMES_HOME"] = str(root / ".hermes")
    os.environ["MNEMOSYNE_DATA_DIR"] = str(root / "data")
    config._mnemosyne_config_isolated = True

    def restore_environment():
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        directory.cleanup()

    config.add_cleanup(restore_environment)


@pytest.fixture(autouse=True)
def _isolated_mnemosyne_config(tmp_path, monkeypatch):
    """Give each test clean defaults while permitting explicit local overrides."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "mnemosyne-data"))
    from mnemosyne.core.config import MnemosyneConfig

    MnemosyneConfig.reset_instance()
    yield
    MnemosyneConfig.reset_instance()
