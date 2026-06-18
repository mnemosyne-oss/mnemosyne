"""Tests for the profile-aware Mnemosyne installer (mnemosyne/install.py)."""

import os
import sys

import pytest

from mnemosyne import install


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME and the repo root under tmp_path.

    Yields (hermes_home, source) where source is the provider directory the
    installer links to.
    """
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    repo_root = tmp_path / "repo"
    source = repo_root / "hermes_memory_provider"
    source.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(install, "_get_mnemosyne_root", lambda: repo_root)
    # Keep config writes out of these assertions; they are tested elsewhere.
    monkeypatch.setattr(install, "_configure_hermes", lambda: True)
    monkeypatch.setattr(install, "_verify", lambda: True)
    yield hermes_home, source


def _make_profile(hermes_home, name, provider):
    profile = hermes_home / "profiles" / name
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        f"memory:\n  provider: {provider}\n", encoding="utf-8"
    )
    return profile


def test_default_install_links_single_home(fake_env):
    hermes_home, source = fake_env

    assert install._ensure_link() is True
    install._link_all_profiles()  # no profiles/ dir -> no-op

    link = hermes_home / "plugins" / "mnemosyne"
    assert link.is_symlink()
    assert link.resolve() == source.resolve()
    assert install._iter_mnemosyne_profiles() == []


def test_only_opted_in_profile_gets_link(fake_env):
    hermes_home, source = fake_env
    profile_a = _make_profile(hermes_home, "alice", "mnemosyne")
    profile_b = _make_profile(hermes_home, "bob", "honcho")

    install._ensure_link()
    install._link_all_profiles()

    link_a = profile_a / "plugins" / "mnemosyne"
    assert link_a.is_symlink() and link_a.resolve() == source.resolve()
    assert not (profile_b / "plugins" / "mnemosyne").exists()


def test_rerun_is_idempotent(fake_env):
    hermes_home, source = fake_env
    profile_a = _make_profile(hermes_home, "alice", "mnemosyne")

    install._ensure_link()
    install._link_all_profiles()
    link_a = profile_a / "plugins" / "mnemosyne"
    first = link_a.readlink()

    install._link_all_profiles()  # second pass leaves the good link alone
    assert link_a.is_symlink()
    assert link_a.readlink() == first
    assert link_a.resolve() == source.resolve()


def test_missing_profiles_dir_is_noop(fake_env):
    hermes_home, _ = fake_env

    assert install._iter_mnemosyne_profiles() == []
    install._link_all_profiles()  # must not raise

    assert install._ensure_link() is True
    assert (hermes_home / "plugins" / "mnemosyne").is_symlink()


def test_profile_without_config_is_skipped(fake_env):
    hermes_home, _ = fake_env
    (hermes_home / "profiles" / "stray").mkdir(parents=True)  # no config.yaml

    assert install._iter_mnemosyne_profiles() == []


def test_uninstall_removes_profile_links(fake_env):
    hermes_home, source = fake_env
    profile_a = _make_profile(hermes_home, "alice", "mnemosyne")

    install._ensure_link()
    install._link_all_profiles()
    link_a = profile_a / "plugins" / "mnemosyne"
    assert link_a.is_symlink()

    install.uninstall()

    assert not link_a.exists() and not link_a.is_symlink()
    assert not (hermes_home / "plugins" / "mnemosyne").exists()


def test_legacy_link_profile_skips_real_dir(fake_env, capsys):
    hermes_home, source = fake_env
    profile_a = _make_profile(hermes_home, "alice", "mnemosyne")

    # A real directory (not a symlink) sitting at the target must be preserved.
    target = profile_a / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    sentinel = target / "user_data.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    result = install._link_profile(profile_a, source)

    assert result is False
    assert target.is_dir() and not target.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert "skipped" in capsys.readouterr().out


def test_profile_with_commented_provider_is_skipped(fake_env):
    hermes_home, _ = fake_env
    profile = hermes_home / "profiles" / "carol"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "# memory:\n#   provider: mnemosyne\n", encoding="utf-8"
    )

    assert install._iter_mnemosyne_profiles() == []


def test_profile_with_extra_whitespace_in_provider_is_detected(fake_env):
    hermes_home, _ = fake_env
    profile = hermes_home / "profiles" / "dave"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        "memory:\n  provider:   mnemosyne\n", encoding="utf-8"
    )

    assert profile in install._iter_mnemosyne_profiles()


def test_symlinked_profile_dir_is_skipped(fake_env):
    hermes_home, _ = fake_env
    outside = hermes_home.parent / "elsewhere"
    outside.mkdir()
    (outside / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n", encoding="utf-8"
    )
    profiles_dir = hermes_home / "profiles"
    profiles_dir.mkdir(parents=True)
    evil = profiles_dir / "evil-link"
    os.symlink(str(outside), str(evil))

    assert evil not in install._iter_mnemosyne_profiles()
    assert install._iter_mnemosyne_profiles() == []


def test_uninstall_keeps_foreign_profile_link(fake_env):
    hermes_home, source = fake_env
    profile_a = _make_profile(hermes_home, "alice", "mnemosyne")
    # A link pointing at a DIFFERENT source must survive uninstall.
    foreign = hermes_home.parent / "other_provider"
    foreign.mkdir()
    link = profile_a / "plugins" / "mnemosyne"
    link.parent.mkdir(parents=True)
    os.symlink(str(foreign), str(link))

    install.uninstall()

    assert link.is_symlink()
    assert link.resolve() == foreign.resolve()


def test_uninstall_removes_orphaned_link_after_config_change(fake_env):
    hermes_home, source = fake_env
    profile_a = _make_profile(hermes_home, "alice", "mnemosyne")

    install._ensure_link()
    install._link_all_profiles()
    link = profile_a / "plugins" / "mnemosyne"
    assert link.is_symlink() and link.resolve() == source.resolve()

    # User switches the profile to another provider after install.
    (profile_a / "config.yaml").write_text(
        "memory:\n  provider: honcho\n", encoding="utf-8"
    )

    install.uninstall()

    # Scan-by-link finds the orphan even though the config no longer opts in.
    assert not link.is_symlink() and not link.exists()


def test_malformed_yaml_config_is_treated_as_not_opted_in(fake_env):
    hermes_home, _ = fake_env
    profile = hermes_home / "profiles" / "broken"
    profile.mkdir(parents=True)
    # Raw text contains `provider: mnemosyne` (the regex fallback would match),
    # but the YAML path must return False on YAMLError without reaching it.
    (profile / "config.yaml").write_text(
        "memory:\n  provider: mnemosyne\n  bad: [unclosed\n", encoding="utf-8"
    )

    assert install._iter_mnemosyne_profiles() == []


def test_legacy_link_profile_force_replaces_real_dir(fake_env):
    hermes_home, source = fake_env
    profile_a = _make_profile(hermes_home, "alice", "mnemosyne")
    target = profile_a / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (target / "stale.txt").write_text("old", encoding="utf-8")

    result = install._link_profile(profile_a, source, force=True)

    assert result is True
    assert target.is_symlink()
    assert target.resolve() == source.resolve()
