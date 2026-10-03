import hashlib
import json
import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import mnemosyne_hermes
from mnemosyne_hermes import install


def _hash_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def test_bundled_memory_override_skill_resource_is_discoverable():
    text = install.bundled_skill_text()

    assert "name: mnemosyne-memory-override" in text
    assert "The user expects fixes" not in text
    assert "invalidation/forget" in text


def test_install_bundled_skill_copies_when_missing(tmp_path):
    result = install.install_bundled_skill(hermes_home_path=tmp_path)
    target = install.skill_target_file(tmp_path)

    assert result.action == "install"
    assert result.changed is True
    assert result.target == target
    assert target.read_text(encoding="utf-8") == install.bundled_skill_text()
    assert target.with_name("SKILL.md.sha256").is_file()


def test_install_bundled_skill_skips_existing_without_force(tmp_path):
    target = install.skill_target_file(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text("user custom skill\n", encoding="utf-8")

    result = install.install_bundled_skill(hermes_home_path=tmp_path)

    assert result.action == "skip"
    assert result.changed is False
    assert target.read_text(encoding="utf-8") == "user custom skill\n"


def test_install_bundled_skill_refreshes_managed_copy_without_force(tmp_path):
    target = install.skill_target_file(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text("old bundled content\n", encoding="utf-8")
    target.with_name("SKILL.md.sha256").write_text(
        _hash_text("old bundled content\n") + "\n",
        encoding="utf-8",
    )

    result = install.install_bundled_skill(hermes_home_path=tmp_path)

    assert result.action == "refresh"
    assert result.changed is True
    assert target.read_text(encoding="utf-8") == install.bundled_skill_text()
    assert target.with_name("SKILL.md.sha256").read_text(encoding="utf-8").strip() == _hash_text(install.bundled_skill_text())
    assert not target.with_name("SKILL.md.bak").exists()


def test_install_bundled_skill_preserves_user_edited_managed_copy_without_force(tmp_path):
    target = install.skill_target_file(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_text("old bundled content plus user edit\n", encoding="utf-8")
    target.with_name("SKILL.md.sha256").write_text(
        _hash_text("old bundled content\n") + "\n",
        encoding="utf-8",
    )

    result = install.install_bundled_skill(hermes_home_path=tmp_path)

    assert result.action == "skip"
    assert result.changed is False
    assert target.read_text(encoding="utf-8") == "old bundled content plus user edit\n"


def test_install_bundled_skill_force_overwrites_existing_with_backup(tmp_path):
    target = install.skill_target_file(tmp_path)
    backup = target.with_name("SKILL.md.bak")
    target.parent.mkdir(parents=True)
    target.write_text("stale bundled skill\n", encoding="utf-8")

    result = install.install_bundled_skill(hermes_home_path=tmp_path, force=True)

    assert result.action == "overwrite"
    assert result.changed is True
    assert "Backup written" in result.message
    assert target.read_text(encoding="utf-8") == install.bundled_skill_text()
    assert backup.read_text(encoding="utf-8") == "stale bundled skill\n"


def test_install_dry_run_reports_skill_action_without_writing(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)

    rc = install.main(["--hermes-home", str(tmp_path), "install", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Skill target file:" in out
    assert "Skill action: Would install bundled skill" in out
    assert not install.skill_target_file(tmp_path).exists()


def test_install_dry_run_reports_when_profile_links_are_disabled(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)

    rc = install.main(
        ["--hermes-home", str(tmp_path), "install", "--no-profile-links", "--dry-run"]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "Will link opted-in profiles: False" in out


def test_install_cli_passes_no_profile_links_to_installer(tmp_path, monkeypatch):
    received = {}

    def fake_run_install(**kwargs):
        received.update(kwargs)
        return 0

    monkeypatch.setattr(install, "run_install", fake_run_install)

    rc = install.main(
        ["--hermes-home", str(tmp_path), "install", "--no-profile-links"]
    )

    assert rc == 0
    assert received["hermes_home_path"] == str(tmp_path)
    assert received["link_profiles"] is False


def test_status_reports_skill_state(tmp_path, capsys, monkeypatch):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("class MnemosyneMemoryProvider: pass\n")
    install.install_bundled_skill(hermes_home_path=tmp_path)
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)

    rc = install.main(["--hermes-home", str(tmp_path), "status"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Skill path:" in out
    assert "Skill:     installed" in out


def test_plugin_state_reports_broken_symlink(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "missing-mnemosyne_hermes", target_is_directory=True)

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "broken_symlink"
    assert state.installed is False
    assert state.target == target
    assert state.link_target == tmp_path / "missing-mnemosyne_hermes"
    assert state.mode == "symlink"
    assert "target is missing" in state.message


def test_plugin_state_reports_invalid_directory(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("# no provider markers\n")

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "invalid_provider"
    assert state.installed is False
    assert state.mode == "directory"
    assert "does not look like a Mnemosyne provider" in state.message


def test_plugin_state_accepts_valid_provider_directory(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("class MnemosyneMemoryProvider: pass\n")

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "installed"
    assert state.installed is True
    assert state.mode == "directory"


def test_plugin_state_accepts_valid_provider_symlink(tmp_path):
    source = tmp_path / "site-packages" / "mnemosyne_hermes"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("def register_memory_provider(ctx): pass\n")
    target = tmp_path / "plugins" / "mnemosyne"
    target.parent.mkdir(parents=True)
    target.symlink_to(source, target_is_directory=True)

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "installed"
    assert state.installed is True
    assert state.link_target == source
    assert state.mode == "symlink"


def test_is_installed_stays_false_for_broken_symlink(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "missing", target_is_directory=True)

    assert install.is_installed(hermes_home_path=tmp_path) is False


def test_wrapper_install_accepts_an_11_second_import_with_60_second_timeout(tmp_path, monkeypatch):
    """A slow but healthy selected runtime must not inherit the old 10s ceiling."""
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    observed_timeouts = []

    def simulated_subprocess(command, **kwargs):
        timeout = kwargs["timeout"]
        observed_timeouts.append(timeout)
        if "-S" in command and timeout < 11:
            raise subprocess.TimeoutExpired(command, timeout)
        if "-S" in command:
            return subprocess.CompletedProcess(command, 0, "0.0-test\n", "")
        return subprocess.CompletedProcess(command, 0, f"{site_packages}\n", "")

    monkeypatch.setattr(install.subprocess, "run", simulated_subprocess)

    target = install.install_plugin(
        hermes_home_path=tmp_path,
        mode="wrapper",
        python=sys.executable,
        import_timeout=60.0,
        link_profiles=False,
    )

    assert target.is_dir()
    assert observed_timeouts == [60.0, 60.0]


def test_install_plugin_wrapper_creates_persistent_shim(tmp_path):
    packaged_plugin_path = install._resolve_package_dir() / "plugin.yaml"
    packaged_plugin_before = packaged_plugin_path.read_bytes()
    source_plugin_path = Path(__file__).parents[1] / "plugin.yaml"
    source_plugin_before = source_plugin_path.read_bytes()
    catalog_plugin_path = Path(__file__).parents[2] / "hermes-catalog" / "plugin.yaml"
    catalog_plugin_before = catalog_plugin_path.read_bytes()

    target = install.install_plugin(
        hermes_home_path=tmp_path,
        force=False,
        mode="wrapper",
        python=sys.executable,
    )

    assert target == tmp_path / "plugins" / "mnemosyne"
    assert target.is_dir()
    assert not target.is_symlink()
    init_source = (target / "__init__.py").read_text(encoding="utf-8")
    assert "register_memory_provider" in init_source
    assert "from mnemosyne_hermes import *" in init_source
    assert "_mnemosyne_bootstrap" in init_source
    assert (target / "cli.py").is_file()
    assert (target / "_mnemosyne_bootstrap.py").is_file()
    manifest = json.loads((target / "mnemosyne-wrapper.json").read_text(encoding="utf-8"))
    assert manifest == {
        "schema_version": 1,
        "python": str(Path(sys.executable).absolute()),
        "site_packages": str(install._site_packages_for_python(Path(sys.executable)).resolve()),
        "package": "mnemosyne_hermes",
    }
    assert (target / "plugin.yaml").is_file()
    installed_plugin = yaml.safe_load((target / "plugin.yaml").read_text(encoding="utf-8"))
    assert installed_plugin["version"] == mnemosyne_hermes.__version__
    assert installed_plugin["python_runtime"] == "external"
    assert "python_runtime" not in yaml.safe_load(packaged_plugin_before)
    assert "python_runtime" not in yaml.safe_load(source_plugin_before)
    assert "python_runtime" not in yaml.safe_load(catalog_plugin_before)
    assert packaged_plugin_path.read_bytes() == packaged_plugin_before
    assert source_plugin_path.read_bytes() == source_plugin_before
    assert catalog_plugin_path.read_bytes() == catalog_plugin_before

    state = install.plugin_state(hermes_home_path=tmp_path)
    assert state.status == "installed"
    assert state.installed is True
    assert state.mode == "wrapper"
    assert state.wrapper_python == Path(sys.executable).absolute()
    assert state.wrapper_site_packages is not None
    assert state.wrapper_import_ok is True
    assert install._is_wrapper_plugin_target(target) is True
    assert install._provider_init_is_mnemosyne(target / "__init__.py") is True


def test_plugin_state_uses_legacy_wrapper_metadata_without_manifest(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    site_packages = install._site_packages_for_python(Path(sys.executable))
    (target / "__init__.py").write_text(
        f"_PYTHON = {str(Path(sys.executable))!r}\n"
        f"_SITE = {str(site_packages)!r}\n"
        "# register_memory_provider / MnemosyneMemoryProvider\n"
        "from mnemosyne_hermes import *\n",
        encoding="utf-8",
    )

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "installed"
    assert state.mode == "wrapper"
    assert state.wrapper_python == Path(sys.executable)
    assert state.wrapper_site_packages == site_packages


@pytest.mark.parametrize(
    ("manifest_text", "expected_message"),
    [
        ("{not valid json", "Invalid Mnemosyne wrapper manifest"),
        (json.dumps({"schema_version": 2}), "Invalid Mnemosyne wrapper manifest schema"),
    ],
)
def test_plugin_state_reports_invalid_wrapper_manifest_without_legacy_fallback(
    tmp_path, manifest_text, expected_message
):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text(
        f"_PYTHON = {str(Path(sys.executable))!r}\n"
        f"_SITE = {str(install._site_packages_for_python(Path(sys.executable)))!r}\n"
        "# register_memory_provider / MnemosyneMemoryProvider\n"
        "from mnemosyne_hermes import *\n",
        encoding="utf-8",
    )
    (target / "mnemosyne-wrapper.json").write_text(manifest_text, encoding="utf-8")

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "invalid_wrapper"
    assert state.installed is False
    assert state.mode == "wrapper"
    assert state.wrapper_import_ok is False
    assert expected_message in state.message


def test_plugin_state_reports_non_executable_wrapper_interpreter_without_raising(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    site_packages = install._site_packages_for_python(Path(sys.executable))
    install._write_wrapper_plugin(target, python=Path(sys.executable), site_packages=site_packages)
    non_executable = tmp_path / "non-executable-python"
    non_executable.write_text("not executable\n", encoding="utf-8")
    non_executable.chmod(non_executable.stat().st_mode & ~(stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
    if os.access(non_executable, os.X_OK):
        pytest.skip("platform cannot create a non-executable regular file")
    assert non_executable.is_file()
    assert not os.access(non_executable, os.X_OK)
    manifest_path = target / "mnemosyne-wrapper.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["python"] = str(non_executable)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "invalid_wrapper"
    assert state.installed is False
    assert state.mode == "wrapper"
    assert state.wrapper_python == non_executable
    assert state.wrapper_import_ok is False
    assert state.wrapper_import_error == f"wrapper Python is not executable: {non_executable}"


def test_check_wrapper_import_returns_error_when_interpreter_cannot_launch(tmp_path, monkeypatch):
    package = tmp_path / "mnemosyne_hermes"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")

    def raise_permission_error(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(install.subprocess, "run", raise_permission_error)

    ok, error, invalid_runtime = install._check_wrapper_import(tmp_path, Path(sys.executable))

    assert ok is False
    assert error == f"could not run wrapper Python {Path(sys.executable)}: permission denied"
    assert invalid_runtime is True


def test_check_wrapper_import_isolates_selected_site_from_pythonpath(monkeypatch):
    site_packages = install._site_packages_for_python(Path(sys.executable))
    observed = {}
    monkeypatch.setenv("PYTHONPATH", "/untrusted/pythonpath")
    monkeypatch.setenv("PYTHONOPTIMIZE", "2")
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")

    def successful_probe(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "ok\n", "")

    monkeypatch.setattr(install.subprocess, "run", successful_probe)

    ok, error, invalid_runtime = install._check_wrapper_import(site_packages, Path(sys.executable))

    assert ok is True
    assert error is None
    assert invalid_runtime is False
    assert "PYTHONPATH" not in observed["env"]
    assert "PYTHONOPTIMIZE" not in observed["env"]
    assert observed["env"]["PYTHONNOUSERSITE"] == "1"
    assert observed["command"][1] == "-S"
    assert "site.addsitedir" in observed["command"][3]
    assert "assert " not in observed["command"][3]


def test_check_wrapper_import_accepts_direct_package_without_dist_metadata(tmp_path):
    site_packages = tmp_path / "site-packages"
    package = site_packages / "mnemosyne_hermes"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    core = site_packages / "mnemosyne" / "core"
    core.mkdir(parents=True)
    (core.parent / "__init__.py").write_text("", encoding="utf-8")
    (core / "__init__.py").write_text("", encoding="utf-8")
    (core / "beam.py").write_text("", encoding="utf-8")

    ok, error, invalid_runtime = install._check_wrapper_import(site_packages, Path(sys.executable))

    assert ok is True
    assert error is None
    assert invalid_runtime is False


def test_plugin_state_classifies_timed_out_wrapper_import_as_stale(tmp_path, monkeypatch):
    target = tmp_path / "plugins" / "mnemosyne"
    site_packages = install._site_packages_for_python(Path(sys.executable))
    install._write_wrapper_plugin(target, python=Path(sys.executable), site_packages=site_packages)

    observed_timeouts = []

    def raise_timeout(*args, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(install.subprocess, "run", raise_timeout)

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "stale_wrapper"
    assert state.installed is False
    assert state.wrapper_import_ok is False
    assert state.wrapper_import_error is not None
    assert str(Path(sys.executable)) in state.wrapper_import_error
    assert observed_timeouts == [60.0]
    assert "fixed default 60-second policy" in state.wrapper_import_error
    assert "Inspect the selected interpreter and its import performance" in state.wrapper_import_error
    assert "--import-timeout only affects installer validation" in state.wrapper_import_error
    assert "--import-timeout 120" not in state.wrapper_import_error
    assert "Retry with:" not in state.wrapper_import_error


def test_plugin_state_reports_stale_wrapper_target(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    missing_site = tmp_path / "missing-site-packages"
    (target / "__init__.py").write_text(
        "_PYTHON = '/missing/python'\n"
        f"_SITE = {str(missing_site)!r}\n"
        "# register_memory_provider / MnemosyneMemoryProvider\n"
        "from mnemosyne_hermes import *\n",
        encoding="utf-8",
    )

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "stale_wrapper"
    assert state.installed is False
    assert state.mode == "wrapper"
    assert state.wrapper_site_packages == missing_site
    assert state.wrapper_import_ok is False
    assert state.wrapper_import_error is not None
    assert "site-packages target missing" in state.wrapper_import_error


def test_install_plugin_rejects_unknown_mode(tmp_path):
    try:
        install.install_plugin(hermes_home_path=tmp_path, mode="copy")
    except ValueError as exc:
        assert "mode must be" in str(exc)
    else:
        raise AssertionError("install_plugin should reject unknown modes")


def test_cli_requires_explicit_flag_to_migrate_wrapper_to_symlink(tmp_path, monkeypatch, capsys):
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")
    target = install.install_plugin(
        hermes_home_path=tmp_path,
        mode="wrapper",
        python=sys.executable,
    )
    original_init = (target / "__init__.py").read_bytes()
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)

    rejected = install.main(
        ["--hermes-home", str(tmp_path), "install", "--force", "--no-bootstrap"]
    )

    assert rejected == 1
    assert target.is_dir() and not target.is_symlink()
    assert (target / "__init__.py").read_bytes() == original_init
    assert "migrate-wrapper-to-symlink" in capsys.readouterr().err

    migrated = install.main(
        [
            "--hermes-home",
            str(tmp_path),
            "install",
            "--force",
            "--no-bootstrap",
            "--migrate-wrapper-to-symlink",
        ]
    )

    assert migrated == 0
    assert target.is_symlink()
    assert "Migrating existing Mnemosyne wrapper to a symlink" in capsys.readouterr().out


def test_dry_run_reports_refused_wrapper_migration(tmp_path, monkeypatch, capsys):
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")
    target = install.install_plugin(
        hermes_home_path=tmp_path,
        mode="wrapper",
        python=sys.executable,
    )
    original_init = (target / "__init__.py").read_bytes()
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)

    rc = install.main(
        ["--hermes-home", str(tmp_path), "install", "--force", "--dry-run", "--no-bootstrap"]
    )

    assert rc == 1
    assert "Will refuse to replace the existing wrapper" in capsys.readouterr().out
    assert target.is_dir() and not target.is_symlink()
    assert (target / "__init__.py").read_bytes() == original_init


def test_dry_run_rejects_invalid_wrapper_migration_flag_combination(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)

    rc = install.main(
        ["--hermes-home", str(tmp_path), "install", "--dry-run", "--migrate-wrapper-to-symlink"]
    )

    assert rc == 1
    assert "unless --mode symlink and --force are both set" in capsys.readouterr().out


def test_install_help_describes_required_wrapper_migration_flags(capsys):
    with pytest.raises(SystemExit, match="0"):
        install.main(["install", "--help"])

    help_text = capsys.readouterr().out
    assert "With --mode symlink and --force" in help_text


def _fake_python(root: Path, version: str = "Python 3.12.13") -> Path:
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    python = bin_dir / "python"
    python.write_text(f"#!/bin/sh\necho '{version}'\n", encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return python


def test_status_mismatch_names_interpreters_and_emits_a_runnable_command(
    tmp_path, monkeypatch, capsys
):
    """#736: status compared interpreter paths but claimed a version mismatch,
    and printed a bare version number instead of a command to run.

    Two separate venvs over one base interpreter resolve to the same binary,
    so the check must compare environment roots, not resolved paths.
    """
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")
    base = _fake_python(tmp_path / "base")
    hermes_venv = tmp_path / "hermes env" / "venv"  # spaces: quoting matters
    this_venv = tmp_path / "this-env" / "venv"
    for venv in (hermes_venv, this_venv):
        (venv / "bin").mkdir(parents=True, exist_ok=True)
        (venv / "pyvenv.cfg").write_text("home = /base\n", encoding="utf-8")
        (venv / "bin" / "python").symlink_to(base)
    hermes_python = hermes_venv / "bin" / "python"
    this_python = this_venv / "bin" / "python"
    assert hermes_python.resolve() == this_python.resolve() == base

    monkeypatch.setattr(install, "_find_hermes_python", lambda **kw: hermes_python)
    monkeypatch.setattr(sys, "executable", str(this_python))
    monkeypatch.setattr(sys, "prefix", str(this_venv))
    monkeypatch.setattr(sys, "version", "3.12.13 (fake interpreter for test)")
    monkeypatch.setattr(
        install,
        "plugin_state",
        lambda hermes_home_path=None: install.PluginState(
            status="installed",
            installed=True,
            target=tmp_path / "plugin",
            link_target=tmp_path / "plugin-target",
            mode="symlink",
            message="ok",
        ),
    )

    rc = install.main(["--hermes-home", str(tmp_path), "status"])

    out = capsys.readouterr().out
    assert rc == 0
    assert "Different Python interpreters" in out
    assert "version MISMATCH" not in out
    assert f"  This Python: {this_python} (3.12.13)" in out
    assert f"  Hermes' Python: {hermes_python} (Python 3.12.13)" in out
    assert (
        f"→ Run: {shlex.quote(str(hermes_python))} -m pip install -U 'mnemosyne-hermes[all]'"
        in out
    )
    assert "→ Run: 3.12.13" not in out


def _two_venvs_over_one_base(tmp_path):
    """Two virtualenvs whose bin/python symlink to a single base interpreter."""
    base = _fake_python(tmp_path / "base")
    hermes_venv = tmp_path / "hermes env" / "venv"  # spaces: quoting matters
    this_venv = tmp_path / "this-env" / "venv"
    for venv in (hermes_venv, this_venv):
        (venv / "bin").mkdir(parents=True, exist_ok=True)
        (venv / "pyvenv.cfg").write_text("home = /base\n", encoding="utf-8")
        (venv / "bin" / "python").symlink_to(base)
    hermes_python = hermes_venv / "bin" / "python"
    this_python = this_venv / "bin" / "python"
    # The premise: resolving really does collapse them onto one binary.
    assert hermes_python.resolve() == this_python.resolve() == base
    return hermes_venv, hermes_python, this_venv, this_python


def test_hermes_python_mismatch_normalises_a_detour_spelling(tmp_path, monkeypatch):
    """`<venv>/bin/../bin/python` names the same environment as `<venv>/bin/python`.

    Deriving the root with `.parent.parent` before normalising yields
    `<venv>/bin/..`, which names `<venv>` but does not compare equal to it, so
    one environment is reported as two.
    """
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    detour = venv / "bin" / ".." / "bin" / "python"

    monkeypatch.setattr(sys, "prefix", str(venv))

    assert install._hermes_python_mismatch(python) is False
    assert install._hermes_python_mismatch(detour) is False
    # A genuinely different environment must still be reported.
    other = tmp_path / "other" / "venv"
    (other / "bin").mkdir(parents=True)
    assert install._hermes_python_mismatch(other / "bin" / "python") is True


def test_provider_diagnostic_reports_two_venvs_over_one_base(
    tmp_path, monkeypatch, capsys
):
    """#709: the provider's failure diagnostic compared resolved interpreter paths.

    A venv's bin/python is a symlink to the base interpreter it was created
    from, so resolving collapsed two distinct environments onto that one binary
    and suppressed the diagnostic in exactly the case it exists to report.
    """
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")
    _, hermes_python, this_venv, this_python = _two_venvs_over_one_base(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("construction failed for test")

    class _Ctx:
        def register_memory_provider(self, provider):
            raise AssertionError("must not register when construction fails")

    monkeypatch.setattr(mnemosyne_hermes, "MnemosyneMemoryProvider", _boom)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kw: hermes_python)
    # Both point at the other venv. The old check compared resolved interpreter
    # paths, which are identical for these two venvs, so it prints nothing and
    # the test fails without the fix. If sys.executable kept pointing at the
    # real pytest interpreter, the old check would fire anyway and the test
    # would pass with or without the fix.
    monkeypatch.setattr(sys, "executable", str(this_python))
    monkeypatch.setattr(sys, "prefix", str(this_venv))

    with pytest.raises(RuntimeError, match="construction failed for test"):
        mnemosyne_hermes.register_memory_provider(_Ctx())

    err = capsys.readouterr().err
    assert f"Hermes' Python: {hermes_python}" in err
    # The venv path contains a space, so the remediation is only runnable quoted.
    assert (
        f"FIX: Run: {shlex.quote(str(hermes_python))}"
        " -m pip install -U 'mnemosyne-hermes[all]'" in err
    )


def test_provider_diagnostic_stays_quiet_for_one_environment(
    tmp_path, monkeypatch, capsys
):
    """The control: one environment must produce no interpreter diagnostic.

    This passes before and after the change, so it is a guard rather than
    evidence of the fix. It exists to catch a check that reports a mismatch
    unconditionally.
    """
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")
    hermes_venv, hermes_python, _, _ = _two_venvs_over_one_base(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("construction failed for test")

    class _Ctx:
        def register_memory_provider(self, provider):
            raise AssertionError("must not register when construction fails")

    monkeypatch.setattr(mnemosyne_hermes, "MnemosyneMemoryProvider", _boom)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kw: hermes_python)
    monkeypatch.setattr(sys, "executable", str(hermes_python))
    monkeypatch.setattr(sys, "prefix", str(hermes_venv))

    with pytest.raises(RuntimeError, match="construction failed for test"):
        mnemosyne_hermes.register_memory_provider(_Ctx())

    err = capsys.readouterr().err
    assert "Hermes' Python:" not in err
    assert "FIX: Run:" not in err
