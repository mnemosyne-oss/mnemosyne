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


def test_wrapper_install_root_only_does_not_link_opted_in_child_profile(tmp_path):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")

    target = install_plugin(
        hermes_home_path=tmp_path,
        mode="wrapper",
        python=sys.executable,
        link_profiles=False,
    )

    assert target == tmp_path / "plugins" / "mnemosyne"
    assert target.is_dir() and not target.is_symlink()
    assert not (child / "plugins" / "mnemosyne").exists()


@pytest.mark.parametrize(
    ("initial_mode", "root_only_mode"),
    [
        ("symlink", "symlink"),
        ("wrapper", "wrapper"),
        ("symlink", "wrapper"),
        ("wrapper", "symlink"),
    ],
)
def test_root_only_transition_removes_only_recognized_profile_links(
    tmp_path, initial_mode, root_only_mode
):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    foreign_profile = _make_profile(tmp_path, "foreign", "mnemosyne")
    directory_profile = _make_profile(tmp_path, "directory", "mnemosyne")
    initial_kwargs = {"hermes_home_path": tmp_path, "mode": initial_mode}
    if initial_mode == "wrapper":
        initial_kwargs["python"] = sys.executable
    install_plugin(**initial_kwargs)

    child_link = child / "plugins" / "mnemosyne"
    foreign_link = foreign_profile / "plugins" / "mnemosyne"
    foreign_target = tmp_path / "foreign-provider"
    foreign_target.mkdir()
    foreign_link.unlink()
    os.symlink(str(foreign_target), str(foreign_link))
    directory_target = directory_profile / "plugins" / "mnemosyne"
    directory_target.unlink()
    directory_target.mkdir()
    marker = directory_target / "foreign.txt"
    marker.write_text("keep", encoding="utf-8")

    root_only_kwargs = {
        "hermes_home_path": tmp_path,
        "mode": root_only_mode,
        "force": True,
        "link_profiles": False,
    }
    if root_only_mode == "wrapper":
        root_only_kwargs["python"] = sys.executable
    elif initial_mode == "wrapper":
        root_only_kwargs["migrate_wrapper_to_symlink"] = True
    install_plugin(**root_only_kwargs)

    assert not child_link.is_symlink() and not child_link.exists()
    assert foreign_link.is_symlink()
    assert foreign_link.resolve() == foreign_target.resolve()
    assert directory_target.is_dir()
    assert marker.read_text(encoding="utf-8") == "keep"


def test_root_only_cleanup_preserves_link_repointed_after_snapshot(tmp_path, monkeypatch):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    child_link = child / "plugins" / "mnemosyne"
    foreign_target = tmp_path / "foreign-provider"
    foreign_target.mkdir()
    original_swap = install_mod._replace_plugin_target_with_staged

    def swap_then_repoint(target, staged):
        original_swap(target, staged)
        child_link.unlink()
        child_link.symlink_to(foreign_target, target_is_directory=True)

    monkeypatch.setattr(install_mod, "_replace_plugin_target_with_staged", swap_then_repoint)

    install_plugin(
        hermes_home_path=tmp_path,
        force=True,
        mode="wrapper",
        python=sys.executable,
        link_profiles=False,
    )

    assert child_link.is_symlink()
    assert child_link.resolve() == foreign_target.resolve()


def test_root_only_transition_preserves_compatible_foreign_provider_link(tmp_path):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    foreign_provider = tmp_path / "foreign-provider"
    foreign_provider.mkdir()
    (foreign_provider / "__init__.py").write_text(
        "class MnemosyneMemoryProvider:\n    pass\n",
        encoding="utf-8",
    )
    (foreign_provider / "plugin.yaml").write_text(
        "name: hermes-mnemosyne\n",
        encoding="utf-8",
    )
    root_link = tmp_path / "plugins" / "mnemosyne"
    root_link.parent.mkdir(parents=True)
    root_link.symlink_to(foreign_provider, target_is_directory=True)
    child_link = child / "plugins" / "mnemosyne"
    child_link.parent.mkdir(parents=True)
    child_link.symlink_to(foreign_provider, target_is_directory=True)

    install_plugin(
        hermes_home_path=tmp_path,
        force=True,
        link_profiles=False,
    )

    assert child_link.is_symlink()
    assert child_link.resolve() == foreign_provider.resolve()


def test_root_only_cleanup_rename_failure_restores_preference(tmp_path, monkeypatch):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    install_plugin(hermes_home_path=tmp_path)
    child_link = child / "plugins" / "mnemosyne"
    preference = install_mod._profile_links_preference_path(tmp_path)
    original_replace = Path.replace

    def deny_profile_quarantine(self, target):
        if self == child_link:
            raise PermissionError("profile quarantine denied")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", deny_profile_quarantine)

    with pytest.raises(PermissionError, match="profile quarantine denied"):
        install_plugin(
            hermes_home_path=tmp_path,
            force=True,
            link_profiles=False,
        )

    assert child_link.is_symlink()
    assert install_mod.profile_links_enabled(hermes_home_path=tmp_path) is True
    assert preference.read_text(encoding="utf-8") == '{"link_profiles": true}\n'


def test_root_only_cleanup_failure_restores_effective_legacy_preference(
    tmp_path, monkeypatch
):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    install_plugin(hermes_home_path=tmp_path)
    child_link = child / "plugins" / "mnemosyne"
    preference = install_mod._profile_links_preference_path(tmp_path)
    preference.unlink()
    assert install_mod.profile_links_enabled(hermes_home_path=tmp_path) is True
    original_replace = Path.replace

    def deny_profile_quarantine(self, target):
        if self == child_link:
            raise PermissionError("profile quarantine denied")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", deny_profile_quarantine)

    with pytest.raises(PermissionError, match="profile quarantine denied"):
        install_plugin(
            hermes_home_path=tmp_path,
            mode="wrapper",
            python=sys.executable,
            force=True,
            link_profiles=False,
        )

    assert child_link.is_symlink()
    assert install_mod.profile_links_enabled(hermes_home_path=tmp_path) is True
    assert preference.read_text(encoding="utf-8") == '{"link_profiles": true}\n'


def test_root_only_cleanup_failure_remains_primary_when_preference_rollback_fails(
    tmp_path, monkeypatch, capsys
):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    install_plugin(hermes_home_path=tmp_path)
    child_link = child / "plugins" / "mnemosyne"
    original_replace = Path.replace

    def deny_profile_quarantine(self, target):
        if self == child_link:
            raise PermissionError("profile quarantine denied")
        return original_replace(self, target)

    def fail_preference_rollback(*args, **kwargs):
        raise OSError("preference rollback denied")

    monkeypatch.setattr(Path, "replace", deny_profile_quarantine)
    monkeypatch.setattr(
        install_mod,
        "_restore_profile_links_preference",
        fail_preference_rollback,
    )

    with pytest.raises(PermissionError, match="profile quarantine denied"):
        install_plugin(
            hermes_home_path=tmp_path,
            force=True,
            link_profiles=False,
        )

    assert child_link.is_symlink()
    assert "Profile-link preference rollback failed" in capsys.readouterr().err


def test_profile_link_snapshot_rejects_repoint_during_classification(tmp_path, monkeypatch):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path)
    child_link = child / "plugins" / "mnemosyne"
    foreign_target = tmp_path / "foreign-provider"
    foreign_target.mkdir()
    original_resolve = Path.resolve
    repointed = False

    def resolve_then_repoint(self, *args, **kwargs):
        nonlocal repointed
        resolved = original_resolve(self, *args, **kwargs)
        if self == child_link and not repointed:
            repointed = True
            child_link.unlink()
            child_link.symlink_to(foreign_target, target_is_directory=True)
        return resolved

    monkeypatch.setattr(Path, "resolve", resolve_then_repoint)

    snapshots = install_mod._recognized_profile_links(
        hermes_home_path=tmp_path,
        recognized_targets=(target,),
    )

    assert snapshots == []
    assert child_link.is_symlink()
    assert child_link.resolve() == foreign_target.resolve()


def test_profile_link_cleanup_restores_repointed_quarantined_link(tmp_path, monkeypatch):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path)
    child_link = child / "plugins" / "mnemosyne"
    snapshots = install_mod._recognized_profile_links(
        hermes_home_path=tmp_path,
        recognized_targets=(target,),
    )
    foreign_target = tmp_path / "foreign-provider"
    foreign_target.mkdir()
    original_replace = Path.replace
    repointed = False

    def repoint_before_quarantine(self, destination):
        nonlocal repointed
        if self == child_link and not repointed:
            repointed = True
            child_link.unlink()
            child_link.symlink_to(foreign_target, target_is_directory=True)
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", repoint_before_quarantine)

    install_mod._unlink_profile_links(snapshots)

    assert child_link.is_symlink()
    assert child_link.resolve() == foreign_target.resolve()


def test_profile_link_cleanup_restores_directory_replacing_snapshot(tmp_path, monkeypatch):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path)
    child_link = child / "plugins" / "mnemosyne"
    snapshots = install_mod._recognized_profile_links(
        hermes_home_path=tmp_path,
        recognized_targets=(target,),
    )
    original_replace = Path.replace
    replaced = False

    def replace_with_directory_before_quarantine(self, destination):
        nonlocal replaced
        if self == child_link and not replaced:
            replaced = True
            child_link.unlink()
            child_link.mkdir()
            (child_link / "keep.txt").write_text("keep", encoding="utf-8")
        return original_replace(self, destination)

    monkeypatch.setattr(Path, "replace", replace_with_directory_before_quarantine)

    install_mod._unlink_profile_links(snapshots)

    assert child_link.is_dir() and not child_link.is_symlink()
    assert (child_link / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_root_only_transition_keeps_links_to_foreign_hermes_provider(tmp_path):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    foreign_provider = tmp_path / "foreign-provider"
    foreign_provider.mkdir()
    (foreign_provider / "__init__.py").write_text(
        "# Compatibility note: MnemosyneMemoryProvider is not defined here.\n"
        "def register_memory_provider():\n    return None\n",
        encoding="utf-8",
    )
    (foreign_provider / "plugin.yaml").write_text(
        "name: foreign-memory-provider\n",
        encoding="utf-8",
    )
    root_link = tmp_path / "plugins" / "mnemosyne"
    root_link.parent.mkdir(parents=True)
    os.symlink(str(foreign_provider), str(root_link))
    child_link = child / "plugins" / "mnemosyne"
    child_link.parent.mkdir(parents=True)
    os.symlink(str(root_link), str(child_link))
    original_child_target = child_link.readlink()

    install_plugin(
        hermes_home_path=tmp_path,
        force=True,
        link_profiles=False,
    )

    assert child_link.is_symlink()
    assert child_link.readlink() == original_child_target


def test_root_only_transition_removes_legacy_wrapper_profile_link(tmp_path):
    _skip_on_windows()
    child = _make_profile(tmp_path, "child", "mnemosyne")
    legacy_site = tmp_path / "legacy-site-packages"
    legacy_package = legacy_site / "mnemosyne_hermes"
    legacy_package.mkdir(parents=True)
    root_plugin = tmp_path / "plugins" / "mnemosyne"
    root_plugin.mkdir(parents=True)
    (root_plugin / "__init__.py").write_text(
        f"_PYTHON = {sys.executable!r}\n"
        f"_SITE = {str(legacy_site)!r}\n"
        "from mnemosyne_hermes import *  # noqa: F401,F403,E402\n",
        encoding="utf-8",
    )
    (root_plugin / "plugin.yaml").write_text(
        "name: hermes-mnemosyne\n",
        encoding="utf-8",
    )
    child_link = child / "plugins" / "mnemosyne"
    child_link.parent.mkdir(parents=True)
    os.symlink(str(legacy_package), str(child_link))

    install_plugin(
        hermes_home_path=tmp_path,
        force=True,
        link_profiles=False,
        migrate_wrapper_to_symlink=True,
    )

    assert not child_link.is_symlink() and not child_link.exists()


def test_profile_links_preference_preserves_default_when_profiles_are_added_later(tmp_path):
    _skip_on_windows()

    install_plugin(hermes_home_path=tmp_path)
    _make_profile(tmp_path, "child", "mnemosyne")

    assert install_mod.profile_links_enabled(hermes_home_path=tmp_path) is True


def test_profile_links_enabled_is_false_for_root_only_install(tmp_path):
    _skip_on_windows()
    _make_profile(tmp_path, "child", "mnemosyne")

    install_plugin(hermes_home_path=tmp_path, link_profiles=False)

    assert install_mod.profile_links_enabled(hermes_home_path=tmp_path) is False


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


def test_failed_root_only_wrapper_refresh_preserves_target_links_and_preference(tmp_path):
    _skip_on_windows()
    selected = _make_profile(tmp_path, "alice", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    selected_link = selected / "plugins" / "mnemosyne"
    original_init = (target / "__init__.py").read_bytes()
    preference = install_mod._profile_links_preference_path(tmp_path)
    original_preference = preference.read_bytes()

    with pytest.raises(FileNotFoundError, match="Python interpreter not found"):
        install_plugin(
            hermes_home_path=tmp_path,
            force=True,
            mode="wrapper",
            python=tmp_path / "missing-python",
            link_profiles=False,
        )

    assert target.is_dir() and not target.is_symlink()
    assert (target / "__init__.py").read_bytes() == original_init
    assert selected_link.is_symlink() and selected_link.resolve() == target.resolve()
    assert preference.read_bytes() == original_preference
    assert install_mod.profile_links_enabled(hermes_home_path=tmp_path) is True


def test_root_only_wrapper_refresh_preference_failure_preserves_target_and_links(
    tmp_path, monkeypatch
):
    _skip_on_windows()
    selected = _make_profile(tmp_path, "alice", "mnemosyne")
    foreign_profile = _make_profile(tmp_path, "bob", "mnemosyne")
    real_profile = _make_profile(tmp_path, "carol", "mnemosyne")
    target = install_plugin(hermes_home_path=tmp_path, mode="wrapper", python=sys.executable)
    selected_link = selected / "plugins" / "mnemosyne"
    foreign_link = foreign_profile / "plugins" / "mnemosyne"
    foreign_link.unlink()
    foreign_target = tmp_path / "foreign-plugin"
    foreign_target.mkdir()
    foreign_link.symlink_to(foreign_target, target_is_directory=True)
    real_plugin = real_profile / "plugins" / "mnemosyne"
    real_plugin.unlink()
    real_plugin.mkdir()
    marker = real_plugin / "keep.txt"
    marker.write_text("foreign directory", encoding="utf-8")
    original_init = (target / "__init__.py").read_bytes()
    original_inode = target.stat().st_ino
    preference = install_mod._profile_links_preference_path(tmp_path)
    original_preference = preference.read_bytes()

    def fail_preference_write(*args, **kwargs):
        raise OSError("preference write failed")

    monkeypatch.setattr(install_mod, "_write_profile_links_preference", fail_preference_write)

    with pytest.raises(OSError, match="preference write failed"):
        install_plugin(
            hermes_home_path=tmp_path,
            force=True,
            mode="wrapper",
            python=sys.executable,
            link_profiles=False,
        )

    assert target.is_dir() and not target.is_symlink()
    assert target.stat().st_ino == original_inode
    assert (target / "__init__.py").read_bytes() == original_init
    assert selected_link.is_symlink() and selected_link.resolve() == target.resolve()
    assert foreign_link.is_symlink() and foreign_link.resolve() == foreign_target.resolve()
    assert marker.read_text(encoding="utf-8") == "foreign directory"
    assert preference.read_bytes() == original_preference


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
    preference = install_mod._profile_links_preference_path(tmp_path)
    original_preference = preference.read_bytes()
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
            link_profiles=False,
        )

    assert target.is_dir() and not target.is_symlink()
    assert (target / "__init__.py").read_bytes() == original_init
    assert profile_link.is_symlink()
    assert profile_link.resolve() == target.resolve()
    assert preference.read_bytes() == original_preference
    assert install_mod.profile_links_enabled(hermes_home_path=tmp_path) is True


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
