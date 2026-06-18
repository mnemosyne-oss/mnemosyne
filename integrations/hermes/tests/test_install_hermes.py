"""Tests for the profile-aware Mnemosyne Hermes installer."""

from __future__ import annotations

import os
import sys

import pytest

from mnemosyne_hermes import install as install_mod
from mnemosyne_hermes.install import install_plugin


def _skip_on_windows() -> None:
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")


def _source() -> "object":
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
