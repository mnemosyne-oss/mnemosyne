"""Hermes standalone, catalog, and persistent-wrapper setup/status contract."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from mnemosyne_hermes import MnemosyneMemoryProvider, install


HERMES_PROJECT = Path(__file__).parents[1]
PACKAGE_DIR = HERMES_PROJECT / "src" / "mnemosyne_hermes"
CATALOG_DIR = HERMES_PROJECT.parent / "hermes-catalog"


def test_package_and_catalog_do_not_declare_a_second_config_store():
    assert not (PACKAGE_DIR / "config_schema.py").exists()
    assert not (CATALOG_DIR / "config_schema.py").exists()


def test_catalog_cli_delegates_registration_to_package():
    spec = importlib.util.spec_from_file_location("mnemosyne_catalog_cli", CATALOG_DIR / "cli.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = argparse.ArgumentParser()
    module.register_cli(parser)
    args = parser.parse_args(["version"])

    assert args.mnemosyne_cmd == "version"
    assert args.func is module.mnemosyne_command


def test_catalog_cli_discovery_does_not_import_provider_runtime():
    code = f"""
import importlib.util
import sys
sys.path.insert(0, {str(PACKAGE_DIR.parent)!r})
assert 'mnemosyne_hermes' not in sys.modules
spec = importlib.util.spec_from_file_location('catalog_cli_probe', {str(CATALOG_DIR / 'cli.py')!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.register_cli)
assert 'mnemosyne_hermes' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_package_entry_point_targets_package_and_preserves_cli():
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")

    metadata = tomllib.loads((HERMES_PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    entry_points = metadata["project"]["entry-points"]

    assert entry_points["hermes_agent.memory_providers"]["mnemosyne"] == "mnemosyne_hermes"
    assert (PACKAGE_DIR / "cli.py").is_file()
    assert not (PACKAGE_DIR / "config_schema.py").exists()


def test_generated_persistent_wrapper_keeps_cli_without_declared_schema(tmp_path):
    site_packages = tmp_path / "side-venv" / "site-packages"
    site_packages.mkdir(parents=True)
    wrapper = tmp_path / "mnemosyne"

    install._write_wrapper_plugin(
        wrapper,
        python=Path(sys.executable),
        site_packages=site_packages,
    )

    assert (wrapper / "cli.py").is_file()
    assert not (wrapper / "config_schema.py").exists()


def test_save_config_preserves_provider_selection_and_existing_keys(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "memory:\n"
        "  provider: mnemosyne\n"
        "  memory_enabled: false\n"
        "  mnemosyne:\n"
        "    profile_isolation: true\n"
        "    existing_key: keep-me\n"
        "unrelated:\n"
        "  value: keep-too\n",
        encoding="utf-8",
    )

    MnemosyneMemoryProvider().save_config(
        {"default_scope": "global", "profile_isolation": False},
        str(tmp_path),
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["memory"]["provider"] == "mnemosyne"
    assert config["memory"]["memory_enabled"] is False
    assert config["memory"]["mnemosyne"] == {
        "auto_sleep": True,
        "default_scope": "global",
        "existing_key": "keep-me",
        "profile_isolation": False,
    }
    assert config["unrelated"] == {"value": "keep-too"}


def test_status_config_is_read_only_bounded_and_secret_free(tmp_path):
    config_path = tmp_path / "config.yaml"
    database_path = tmp_path / "mnemosyne.db"
    config_path.write_text("memory:\n  provider: mnemosyne\n", encoding="utf-8")
    database_path.write_bytes(b"database-sentinel")
    before = (config_path.read_bytes(), database_path.read_bytes())

    status = MnemosyneMemoryProvider().get_status_config(
        {
            "profile_isolation": True,
            "default_scope": "global",
            "ignore_patterns": ["private phrase", "another phrase"],
            "reflect": {
                "disabled_for_cron": True,
                "max_calls_per_session": 2,
                "api_key": "nested-secret",
            },
            "tools": ["mnemosyne_recall", "not-a-provider-tool", 42],
            "api_key": "top-level-secret",
            "token": "another-secret",
        }
    )

    assert status == {
        "profile_isolation": True,
        "default_scope": "global",
        "reflect": {"disabled_for_cron": True, "max_calls_per_session": 2},
        "ignore_patterns": "2 configured",
        "tools": ["mnemosyne_recall"],
    }
    assert "secret" not in repr(status)
    assert (config_path.read_bytes(), database_path.read_bytes()) == before


def test_status_config_rejects_forged_tools_and_terminal_control_sequences():
    forged = "mnemosyne_not_real\x1b[2J" + "x" * 20_000
    status = MnemosyneMemoryProvider().get_status_config(
        {
            "sleep_threshold": 10**20_000,
            "reflect": {"max_calls_per_session": -(10**20_000)},
            "shared_surface_path": "safe-prefix\x1b[2Jspoofed",
            "tools": [
                "mnemosyne_recall",
                "mnemosyne_recall",
                "mnemosyne_not_real",
                forged,
            ]
            + ["mnemosyne_remember"] * 1_000,
        }
    )

    assert status == {"tools": ["mnemosyne_recall", "mnemosyne_remember"]}
    rendered = repr(status)
    assert len(rendered) < 256
    assert "\x1b" not in rendered


def test_status_config_normalizes_all_supported_provider_setting_shapes():
    status = MnemosyneMemoryProvider().get_status_config(
        {
            "ignore_patterns": " one, two\n one ",
            "skip_contexts": ("cron", "cron", "invalid", "flush"),
            "sync_roles": {"USER", "assistant", "unknown"},
        }
    )

    assert status["ignore_patterns"] == "2 configured"
    assert status["skip_contexts"] == "cron,flush"
    assert set(status["sync_roles"]) == {"user", "assistant"}
    assert len(status["skip_contexts"]) < 256
    assert len(status["sync_roles"]) == 2


def test_status_config_bounds_repeated_context_and_role_values():
    status = MnemosyneMemoryProvider().get_status_config(
        {
            "skip_contexts": ["cron"] * 10_000,
            "sync_roles": ["user"] * 10_000,
        }
    )

    assert status["skip_contexts"] == "cron"
    assert status["sync_roles"] == ["user"]
    assert len(repr(status)) < 256


class _BrokenConfig(dict):
    def get(self, key, default=None):
        raise RuntimeError("must fail soft")


def test_status_config_fails_soft_for_malformed_mapping():
    assert MnemosyneMemoryProvider().get_status_config(_BrokenConfig()) == {}
    assert MnemosyneMemoryProvider().get_status_config("not-a-mapping") == {}
