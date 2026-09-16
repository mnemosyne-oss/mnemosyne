"""Contract tests for Mnemosyne-owned paths under a Hermes home."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from mnemosyne_hermes import install


def test_hermes_path_contract_resolves_from_the_existing_install_helpers(
    tmp_path, monkeypatch
):
    home = tmp_path / "selected-hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))

    contract = install.hermes_path_contract()

    assert contract.hermes_home == home
    assert contract.plugin_target == home / "plugins" / "mnemosyne"
    assert contract.wrapper_manifest == contract.plugin_target / "mnemosyne-wrapper.json"
    assert contract.skill_target == (
        home / "skills" / "memory" / "mnemosyne-memory-override" / "SKILL.md"
    )
    assert contract.plugin_target == install.plugin_target_dir()
    assert contract.skill_target == install.skill_target_file()


def test_wrapper_and_skill_artifacts_follow_the_declared_path_and_manifest_boundary(
    tmp_path,
):
    home = tmp_path / "hermes-home"
    side_venv = tmp_path / "mnemosyne-side-venv"
    side_python = Path(sys.executable).absolute()
    side_site_packages = side_venv / "site-packages"
    side_site_packages.mkdir(parents=True)
    contract = install.hermes_path_contract(home)

    install._write_wrapper_plugin(
        contract.plugin_target,
        python=side_python,
        site_packages=side_site_packages,
    )
    install.install_bundled_skill(hermes_home_path=home)

    files_under_home = {
        path.relative_to(home).as_posix() for path in home.rglob("*") if path.is_file()
    }
    assert files_under_home == {
        "plugins/mnemosyne/__init__.py",
        "plugins/mnemosyne/_mnemosyne_bootstrap.py",
        "plugins/mnemosyne/cli.py",
        "plugins/mnemosyne/mnemosyne-wrapper.json",
        "plugins/mnemosyne/plugin.yaml",
        "skills/memory/mnemosyne-memory-override/SKILL.md",
        "skills/memory/mnemosyne-memory-override/SKILL.md.sha256",
    }

    manifest = json.loads(contract.wrapper_manifest.read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1,
        "python": str(side_python),
        "site_packages": str(side_site_packages.resolve()),
        "package": "mnemosyne_hermes",
    }
    assert not side_site_packages.is_relative_to(home)

    shutil.rmtree(side_venv)

    assert contract.wrapper_manifest.is_file()
    assert contract.skill_target.is_file()
    assert install._is_wrapper_plugin_target(contract.plugin_target) is True
    state = install.plugin_state(hermes_home_path=home)
    assert state.status == "stale_wrapper"
    assert state.wrapper_site_packages == side_site_packages


def test_wrapper_target_is_identified_by_its_manifest_without_importing_package(
    tmp_path, monkeypatch
):
    contract = install.hermes_path_contract(tmp_path)
    contract.plugin_target.mkdir(parents=True)
    contract.wrapper_manifest.write_text("{}\n", encoding="utf-8")

    def fail_import(*_args, **_kwargs):
        raise AssertionError("wrapper marker detection must not import the package")

    monkeypatch.setattr(install.importlib, "import_module", fail_import)

    assert install._is_wrapper_plugin_target(contract.plugin_target) is True
