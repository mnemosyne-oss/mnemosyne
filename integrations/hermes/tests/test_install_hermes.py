"""Tests for the profile-aware Mnemosyne Hermes installer."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest
from mnemosyne_hermes import install as install_mod
from mnemosyne_hermes.install import install_plugin


def _skip_on_windows() -> None:
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")


def _source() -> Path:
    return install_mod._resolve_package_dir()


def _make_profile(hermes_home, name, provider):
    profile = hermes_home / "profiles" / name
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        f"memory:\n  provider: {provider}\n", encoding="utf-8"
    )
    return profile


def test_default_install_links_single_home(tmp_path):
    _skip_on_windows()

    target = install_plugin(hermes_home_path=tmp_path)

    assert target == tmp_path / "plugins" / "mnemosyne"
    assert target.is_symlink()
    assert target.resolve() == _source().resolve()
    assert install_mod._iter_mnemosyne_profiles(tmp_path) == []


def test_only_opted_in_profile_gets_link(tmp_path):
    _skip_on_windows()
    profile_a = _make_profile(tmp_path, "alice", "mnemosyne")
    profile_b = _make_profile(tmp_path, "bob", "honcho")

    install_plugin(hermes_home_path=tmp_path)

    link_a = profile_a / "plugins" / "mnemosyne"
    assert link_a.is_symlink() and link_a.resolve() == _source().resolve()
    assert not (profile_b / "plugins" / "mnemosyne").exists()


def test_rerun_is_idempotent(tmp_path):
    _skip_on_windows()
    profile_a = _make_profile(tmp_path, "alice", "mnemosyne")

    install_plugin(hermes_home_path=tmp_path)
    link_a = profile_a / "plugins" / "mnemosyne"
    first = link_a.readlink()

    # Second profile pass leaves the existing good link untouched.
    install_mod._link_all_profiles(_source(), hermes_home_path=tmp_path)
    assert link_a.is_symlink()
    assert link_a.readlink() == first
    assert link_a.resolve() == _source().resolve()


def test_missing_profiles_dir_is_noop(tmp_path):
    _skip_on_windows()

    assert install_mod._iter_mnemosyne_profiles(tmp_path) == []
    target = install_plugin(hermes_home_path=tmp_path)  # must not raise
    assert target.is_symlink()


def test_profile_without_config_is_skipped(tmp_path):
    _skip_on_windows()
    (tmp_path / "profiles" / "stray").mkdir(parents=True)  # no config.yaml

    assert install_mod._iter_mnemosyne_profiles(tmp_path) == []


def test_uninstall_removes_profile_links(tmp_path):
    _skip_on_windows()
    profile_a = _make_profile(tmp_path, "alice", "mnemosyne")

    install_plugin(hermes_home_path=tmp_path)
    link_a = profile_a / "plugins" / "mnemosyne"
    assert link_a.is_symlink()

    install_mod.uninstall_plugin(hermes_home_path=tmp_path)

    assert not link_a.is_symlink() and not link_a.exists()


def test_wrapper_profile_links_point_to_base_wrapper_and_uninstall_leaves_foreign_link(tmp_path):
    _skip_on_windows()
    profile_a = _make_profile(tmp_path, "alice", "mnemosyne")
    profile_b = _make_profile(tmp_path, "bob", "mnemosyne")
    foreign = tmp_path / "other_provider"
    foreign.mkdir()
    foreign_link = profile_b / "plugins" / "mnemosyne"
    foreign_link.parent.mkdir(parents=True)
    os.symlink(str(foreign), str(foreign_link))

    base_wrapper = install_plugin(
        hermes_home_path=tmp_path,
        mode="wrapper",
        python=sys.executable,
    )
    wrapper_link = profile_a / "plugins" / "mnemosyne"

    assert wrapper_link.is_symlink()
    assert wrapper_link.resolve() == base_wrapper.resolve()
    assert foreign_link.resolve() == foreign.resolve()

    install_mod.uninstall_plugin(hermes_home_path=tmp_path)

    assert not wrapper_link.is_symlink() and not wrapper_link.exists()
    assert foreign_link.is_symlink()
    assert foreign_link.resolve() == foreign.resolve()


def test_force_symlink_install_refuses_to_replace_wrapper_without_migration_flag(tmp_path):
    _skip_on_windows()
    profile = _make_profile(tmp_path, "alice", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    profile_link = profile / "plugins" / "mnemosyne"
    original_init = (target / "__init__.py").read_bytes()

    with pytest.raises(RuntimeError, match="migrate-wrapper-to-symlink"):
        install_plugin(hermes_home_path=tmp_path, force=True)

    assert target.is_dir() and not target.is_symlink()
    assert (target / "__init__.py").read_bytes() == original_init
    assert profile_link.is_symlink()
    assert profile_link.resolve() == target.resolve()


def test_explicit_force_migration_replaces_wrapper_with_symlink_and_warns(tmp_path, capsys):
    _skip_on_windows()
    profile = _make_profile(tmp_path, "alice", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)

    migrated = install_plugin(
        hermes_home_path=tmp_path,
        force=True,
        migrate_wrapper_to_symlink=True,
    )

    assert migrated == target
    assert target.is_symlink()
    assert target.resolve() == Path(_source()).resolve()
    assert (profile / "plugins" / "mnemosyne").resolve() == Path(_source()).resolve()
    assert "Migrating existing Mnemosyne wrapper to a symlink" in capsys.readouterr().out


@pytest.mark.parametrize("failure", ("missing_python", "unavailable_package"))
def test_forced_wrapper_refresh_preflights_before_preserving_existing_wrapper_links(
    tmp_path, failure
):
    _skip_on_windows()
    selected = _make_profile(tmp_path, "alice", "mnemosyne")
    unselected = _make_profile(tmp_path, "bob", "honcho")
    foreign = tmp_path / "foreign-provider"
    foreign.mkdir()
    foreign_link = unselected / "plugins" / "mnemosyne"
    foreign_link.parent.mkdir(parents=True)
    os.symlink(str(foreign), str(foreign_link))
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    selected_link = selected / "plugins" / "mnemosyne"
    original_init = (target / "__init__.py").read_bytes()

    if failure == "missing_python":
        with pytest.raises(FileNotFoundError, match="Python interpreter not found"):
            install_plugin(
                hermes_home_path=tmp_path,
                force=True,
                mode="wrapper",
                python=tmp_path / "missing-python",
            )
    else:
        environment = tmp_path / "unavailable-environment"
        venv.EnvBuilder(with_pip=False).create(environment)
        with pytest.raises(RuntimeError, match="Selected Python environment cannot import"):
            install_plugin(
                hermes_home_path=tmp_path,
                force=True,
                mode="wrapper",
                python=environment / "bin" / "python",
            )

    assert target.is_dir() and not target.is_symlink()
    assert (target / "__init__.py").read_bytes() == original_init
    assert selected_link.is_symlink() and selected_link.resolve() == target.resolve()
    assert foreign_link.is_symlink() and foreign_link.resolve() == foreign.resolve()


def test_force_wrapper_refresh_replaces_wrapper_and_keeps_selected_profile_link(tmp_path):
    _skip_on_windows()
    profile = _make_profile(tmp_path, "alice", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    old_inode = target.stat().st_ino

    refreshed = install_plugin(
        hermes_home_path=tmp_path,
        force=True,
        mode="wrapper",
        python=sys.executable,
    )

    assert refreshed == target
    assert target.is_dir() and not target.is_symlink()
    assert target.stat().st_ino != old_inode
    assert install_mod.plugin_state(hermes_home_path=tmp_path).mode == "wrapper"
    assert (profile / "plugins" / "mnemosyne").resolve() == target.resolve()
    assert not list(target.parent.glob(".mnemosyne.previous-*"))


def test_force_wrapper_refresh_replaces_existing_symlink_and_keeps_profile_link(tmp_path):
    _skip_on_windows()
    profile = _make_profile(tmp_path, "alice", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path)
    assert target.is_symlink()

    refreshed = install_plugin(
        hermes_home_path=tmp_path,
        force=True,
        mode="wrapper",
        python=sys.executable,
    )

    assert refreshed == target
    assert target.is_dir() and not target.is_symlink()
    assert (profile / "plugins" / "mnemosyne").resolve() == target.resolve()


def test_wrapper_refresh_ignores_post_swap_backup_cleanup_error(tmp_path, monkeypatch):
    _skip_on_windows()
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    original_rmtree = install_mod.shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if Path(path).parent.name.startswith(".mnemosyne.previous-"):
            raise OSError("backup cleanup failed")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(install_mod.shutil, "rmtree", fail_backup_cleanup)

    refreshed = install_plugin(
        hermes_home_path=tmp_path,
        force=True,
        mode="wrapper",
        python=sys.executable,
    )

    assert refreshed == target
    assert target.is_dir() and not target.is_symlink()


def test_failed_wrapper_rollback_reports_retained_backup_path(tmp_path, monkeypatch, capsys):
    _skip_on_windows()
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    original_replace = Path.replace

    def fail_swap_and_rollback(self, destination):
        if self.parent.name.startswith(".mnemosyne.staging-"):
            raise OSError("staged swap failed")
        if self.parent.name.startswith(".mnemosyne.previous-"):
            raise OSError("rollback failed")
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", fail_swap_and_rollback)

    with pytest.raises(OSError, match="staged swap failed"):
        install_plugin(
            hermes_home_path=tmp_path,
            force=True,
            mode="wrapper",
            python=sys.executable,
        )

    error = capsys.readouterr().err
    assert "Wrapper rollback failed; previous plugin retained at:" in error
    backup = Path(error.rsplit(": ", 1)[1].strip())
    assert backup.name == target.name
    assert backup.parent.name.startswith(".mnemosyne.previous-")
    assert backup.is_dir()


def test_failed_wrapper_swap_restores_wrapper_and_profile_link(tmp_path, monkeypatch):
    _skip_on_windows()
    profile = _make_profile(tmp_path, "alice", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    profile_link = profile / "plugins" / "mnemosyne"
    original_init = (target / "__init__.py").read_bytes()
    original_replace = Path.replace

    def fail_staged_swap(self, destination):
        if self.parent.name.startswith(".mnemosyne.staging-"):
            raise OSError("staged swap failed")
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", fail_staged_swap)

    with pytest.raises(OSError, match="staged swap failed"):
        install_plugin(
            hermes_home_path=tmp_path,
            force=True,
            mode="wrapper",
            python=sys.executable,
        )

    assert target.is_dir() and not target.is_symlink()
    assert (target / "__init__.py").read_bytes() == original_init
    assert profile_link.is_symlink()
    assert profile_link.resolve() == target.resolve()


def test_wrapper_preflight_and_bootstrap_support_real_pep660_editable_install(tmp_path):
    _skip_on_windows()
    environment = tmp_path / "editable-environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    project = _source().parent.parent
    install_result = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "-e", str(project)],
        capture_output=True,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    site_packages = install_mod._site_packages_for_python(python)
    assert not (site_packages / "mnemosyne_hermes").exists()
    assert install_mod._check_wrapper_import(site_packages, python) == (True, None, False)

    wrapper = tmp_path / "hermes-home" / "plugins" / "mnemosyne"
    install_mod._write_wrapper_plugin(wrapper, python=python, site_packages=site_packages)
    expected_source = _source().resolve()
    code = f"""
import importlib.util
import sys
from pathlib import Path

wrapper = Path({str(wrapper)!r})
spec = importlib.util.spec_from_file_location(
    'standalone_mnemosyne_bootstrap', wrapper / '_mnemosyne_bootstrap.py'
)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
bootstrap.activate()
import mnemosyne_hermes
assert Path(mnemosyne_hermes.__file__).resolve().parent == Path({str(expected_source)!r})
assert sys.path[0] == {str(site_packages.resolve())!r}
"""
    result = subprocess.run(
        [str(python), "-S", "-c", code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("force", "mode"),
    [
        (False, "symlink"),
        (True, "wrapper"),
    ],
)
def test_migrate_wrapper_to_symlink_rejects_invalid_flag_combinations(tmp_path, force, mode):
    with pytest.raises(ValueError, match="migrate_wrapper_to_symlink requires"):
        install_plugin(
            hermes_home_path=tmp_path,
            force=force,
            mode=mode,
            migrate_wrapper_to_symlink=True,
        )


def test_link_profile_returns_none_on_symlink_error(tmp_path, monkeypatch):
    _skip_on_windows()
    profile_a = _make_profile(tmp_path, "alice", "mnemosyne")
    _make_profile(tmp_path, "bob", "mnemosyne")
    source = _source()

    attempts = []

    def boom(src, dst):
        attempts.append(dst)
        raise OSError("symlink denied")

    monkeypatch.setattr(install_mod.os, "symlink", boom)

    assert install_mod._link_profile(profile_a, source) is None
    # A failing profile must not abort the batch; both profiles are attempted.
    assert install_mod._link_all_profiles(source, hermes_home_path=tmp_path) == []
    assert len(attempts) == 3  # 1 single call + 2 in the batch


def test_force_overwrite_logs_old_target(tmp_path, capsys):
    _skip_on_windows()
    profile_a = _make_profile(tmp_path, "alice", "mnemosyne")
    other = tmp_path / "other_pkg"
    other.mkdir()
    target = profile_a / "plugins" / "mnemosyne"
    target.parent.mkdir(parents=True)
    os.symlink(str(other), str(target))

    result = install_mod._link_profile(profile_a, _source(), force=True)

    out = capsys.readouterr().out
    assert result is not None
    assert str(other) in out
    assert result.resolve() == _source().resolve()


def test_profile_with_commented_provider_is_skipped(tmp_path):
    _skip_on_windows()
    profile = tmp_path / "profiles" / "carol"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "# memory:\n#   provider: mnemosyne\n", encoding="utf-8"
    )

    assert install_mod._iter_mnemosyne_profiles(tmp_path) == []


def test_profile_with_extra_whitespace_in_provider_is_detected(tmp_path):
    _skip_on_windows()
    profile = tmp_path / "profiles" / "dave"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "memory:\n  provider:   mnemosyne\n", encoding="utf-8"
    )

    assert profile in install_mod._iter_mnemosyne_profiles(tmp_path)


def test_symlinked_profile_dir_is_skipped(tmp_path):
    _skip_on_windows()
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n", encoding="utf-8"
    )
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True)
    os.symlink(str(outside), str(profiles_dir / "evil-link"))

    assert install_mod._iter_mnemosyne_profiles(tmp_path) == []


def test_uninstall_keeps_foreign_profile_link(tmp_path):
    _skip_on_windows()
    profile_a = _make_profile(tmp_path, "alice", "mnemosyne")
    foreign = tmp_path / "other_provider"
    foreign.mkdir()
    link = profile_a / "plugins" / "mnemosyne"
    link.parent.mkdir(parents=True)
    os.symlink(str(foreign), str(link))

    install_mod.uninstall_plugin(hermes_home_path=tmp_path)

    assert link.is_symlink()
    assert link.resolve() == foreign.resolve()


def test_uninstall_removes_orphaned_link_after_config_change(tmp_path):
    _skip_on_windows()
    profile_a = _make_profile(tmp_path, "alice", "mnemosyne")

    install_plugin(hermes_home_path=tmp_path)
    link = profile_a / "plugins" / "mnemosyne"
    assert link.is_symlink() and link.resolve() == _source().resolve()

    (profile_a / "config.yaml").write_text(
        "memory:\n  provider: honcho\n", encoding="utf-8"
    )

    install_mod.uninstall_plugin(hermes_home_path=tmp_path)

    assert not link.is_symlink() and not link.exists()


def test_uninstall_removes_home_link_too(tmp_path):
    _skip_on_windows()
    install_plugin(hermes_home_path=tmp_path)
    home_link = tmp_path / "plugins" / "mnemosyne"
    assert home_link.is_symlink()

    install_mod.uninstall_plugin(hermes_home_path=tmp_path)

    assert not home_link.is_symlink() and not home_link.exists()


def test_link_all_profiles_continues_on_mkdir_error(tmp_path, monkeypatch):
    _skip_on_windows()
    _make_profile(tmp_path, "alice", "mnemosyne")
    _make_profile(tmp_path, "bob", "mnemosyne")

    from pathlib import Path

    calls = []
    real_mkdir = Path.mkdir

    def boom(self, *args, **kwargs):
        if self.name == "plugins":
            calls.append(self)
            raise PermissionError("denied")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", boom)

    result = install_mod._link_all_profiles(_source(), hermes_home_path=tmp_path)

    assert result == []
    assert len(calls) == 2  # both profiles attempted despite the first failing


def test_malformed_yaml_config_is_treated_as_not_opted_in(tmp_path):
    _skip_on_windows()
    profile = tmp_path / "profiles" / "broken"
    profile.mkdir(parents=True)
    # Raw text contains `provider: mnemosyne` (the regex fallback would match),
    # but the YAML path must return False on YAMLError without reaching it.
    (profile / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n  bad: [unclosed\n", encoding="utf-8"
    )

    assert install_mod._iter_mnemosyne_profiles(tmp_path) == []
