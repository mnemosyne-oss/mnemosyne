"""Tests for the Mnemosyne Hermes installer shim (mnemosyne/install.py).

The installer no longer creates the legacy ``hermes_memory_provider`` plugin
symlink (#651). It delegates to the standalone ``mnemosyne-hermes`` provider,
migrates legacy links, and fails clearly when the provider is unavailable.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from mnemosyne import install

# Captured before any fixture patches it, so one test can exercise the real
# implementation without losing the isolated HERMES_HOME.
_REAL_CONFIGURE_HERMES = install._configure_hermes


class _State:
    """Stand-in for the standalone provider's PluginState."""

    def __init__(self, *, installed=True, status="installed", target=None, mode="symlink",
                 message="Plugin is installed and discoverable.", link_target=None):
        self.installed = installed
        self.status = status
        self.target = target
        self.mode = mode
        self.message = message
        self.link_target = link_target


class _FakeProvider:
    """Minimal stand-in for the standalone mnemosyne_hermes.install module."""

    def __init__(self, *, installed=True, plugin_state_error=None):
        self.calls: list[list[str]] = []
        self.state_kwargs: list[dict] = []
        self._installed = installed
        self._plugin_state_error = plugin_state_error

    def plugin_state(self, **kwargs):
        self.state_kwargs.append(kwargs)
        if self._plugin_state_error is not None:
            raise self._plugin_state_error
        return _State(installed=self._installed)

    def main(self, argv):
        self.calls.append(list(argv))
        return 0


@pytest.fixture
def fake_env(tmp_path, monkeypatch):
    """Isolate HERMES_HOME under tmp_path and block real config writes.

    Yields (hermes_home, legacy_source) where legacy_source is the obsolete
    provider directory a legacy link points at.
    """
    if sys.platform.startswith("win32"):
        pytest.skip("POSIX symlink test")

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()

    legacy_source = tmp_path / "repo" / install.LEGACY_PROVIDER_DIRNAME
    legacy_source.mkdir(parents=True)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(install, "_configure_hermes", lambda *a, **k: True)
    yield hermes_home, legacy_source


def _make_profile(hermes_home, name, provider):
    profile = hermes_home / "profiles" / name
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(
        f"memory:\n  provider: {provider}\n", encoding="utf-8"
    )
    return profile


def _link_legacy(home, source):
    """Create the legacy plugin link the pre-#651 installer would have made."""
    plugins = home / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    target = plugins / install.PLUGIN_DIRNAME
    if target.is_symlink() or target.exists():
        install._remove_link(target)
    os.symlink(str(source), str(target))
    return target


def _link_standalone(home, standalone_source):
    """Create a link into the supported mnemosyne_hermes package."""
    plugins = home / "plugins"
    plugins.mkdir(parents=True, exist_ok=True)
    target = plugins / install.PLUGIN_DIRNAME
    if target.is_symlink() or target.exists():
        install._remove_link(target)
    os.symlink(str(standalone_source), str(target))
    return target


def _write_config(hermes_home, provider="mnemosyne"):
    (hermes_home / "config.yaml").write_text(
        f"memory:\n  provider: {provider}\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Legacy detection
# ---------------------------------------------------------------------------


def test_is_legacy_plugin_path_detects_legacy_link(fake_env):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)

    assert install.is_legacy_plugin_path(target) is True


def test_is_legacy_plugin_path_ignores_standalone_link(fake_env, tmp_path):
    """A hand-made link into mnemosyne_hermes is the supported route, not legacy."""
    hermes_home, _ = fake_env
    standalone = tmp_path / "site-packages" / "mnemosyne_hermes"
    standalone.mkdir(parents=True)
    (standalone / "__init__.py").write_text("", encoding="utf-8")
    target = _link_standalone(hermes_home, standalone)

    assert install.is_legacy_plugin_path(target) is False


def test_is_legacy_plugin_path_ignores_real_directory(fake_env):
    hermes_home, _ = fake_env
    target = hermes_home / "plugins" / install.PLUGIN_DIRNAME
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("", encoding="utf-8")

    assert install.is_legacy_plugin_path(target) is False


def test_is_legacy_plugin_path_ignores_missing_path(fake_env):
    hermes_home, _ = fake_env

    assert install.is_legacy_plugin_path(
        hermes_home / "plugins" / install.PLUGIN_DIRNAME
    ) is False


def test_is_legacy_plugin_path_detects_broken_legacy_link(fake_env, tmp_path):
    """A dangling link still naming hermes_memory_provider must be cleanable."""
    hermes_home, _ = fake_env
    target = _link_legacy(hermes_home, tmp_path / "gone" / install.LEGACY_PROVIDER_DIRNAME)

    assert install.is_legacy_plugin_path(target) is True


def test_detect_legacy_installs_covers_default_home_and_profiles(fake_env):
    hermes_home, legacy_source = fake_env
    profile = _make_profile(hermes_home, "alice", "mnemosyne")
    other = _make_profile(hermes_home, "bob", "honcho")

    default_link = _link_legacy(hermes_home, legacy_source)
    profile_link = _link_legacy(profile, legacy_source)
    _link_legacy(other, legacy_source)

    found = install.detect_legacy_installs()

    assert default_link in found
    assert profile_link in found
    # bob does not select mnemosyne, so the installer is not responsible for it.
    assert other / "plugins" / install.PLUGIN_DIRNAME not in found


def test_detect_legacy_installs_finds_pre_rename_directory(fake_env):
    hermes_home, _ = fake_env
    renamed = hermes_home / "plugins" / install.LEGACY_PLUGIN_DIRNAME
    renamed.mkdir(parents=True)

    assert renamed in install.detect_legacy_installs()


def test_detect_legacy_installs_empty_when_clean(fake_env, tmp_path):
    hermes_home, _ = fake_env
    standalone = tmp_path / "site-packages" / "mnemosyne_hermes"
    standalone.mkdir(parents=True)
    _link_standalone(hermes_home, standalone)

    assert install.detect_legacy_installs() == []


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def test_migrate_removes_legacy_link(fake_env):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)

    removed = install.migrate_legacy_install()

    assert removed == [target]
    assert not target.is_symlink() and not target.exists()
    assert install.detect_legacy_installs() == []


def test_migrate_dry_run_reports_without_removing(fake_env):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)

    removed = install.migrate_legacy_install(dry_run=True)

    assert removed == [target]
    assert target.is_symlink()


def test_migrate_keeps_real_directory_with_user_data(fake_env, capsys):
    """The pre-rename directory is reported, never deleted."""
    hermes_home, _ = fake_env
    target = hermes_home / "plugins" / install.LEGACY_PLUGIN_DIRNAME
    target.mkdir(parents=True)
    sentinel = target / "user_data.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    removed = install.migrate_legacy_install()

    assert removed == []
    assert sentinel.read_text(encoding="utf-8") == "keep me"
    assert "not a link" in capsys.readouterr().out


def test_migrate_dry_run_also_keeps_real_directory(fake_env, capsys):
    """A dry run must not promise to delete what the real run would keep."""
    hermes_home, _ = fake_env
    (hermes_home / "plugins" / install.LEGACY_PLUGIN_DIRNAME).mkdir(parents=True)

    removed = install.migrate_legacy_install(dry_run=True)

    assert removed == []
    assert "not a link" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Status verification
# ---------------------------------------------------------------------------


def test_status_true_when_provider_installed_and_configured(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    _write_config(hermes_home)
    provider = _FakeProvider(installed=True)
    monkeypatch.setattr(install, "_load_standalone_installer", lambda: provider)

    assert install.status() is True


def test_status_false_when_provider_missing(fake_env, monkeypatch, capsys):
    hermes_home, _ = fake_env
    _write_config(hermes_home)
    monkeypatch.setattr(install, "_load_standalone_installer", lambda: None)
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: None)

    assert install.status() is False
    out = capsys.readouterr().out
    assert install.STANDALONE_DISTRIBUTION in out


def test_status_delegates_when_provider_lives_in_another_python(fake_env, monkeypatch):
    """This machine's case: provider installed in Hermes' venv, not this Python."""
    hermes_home, _ = fake_env
    _write_config(hermes_home)
    calls = []

    def fake_runner(argv):
        calls.append(list(argv))
        return 0

    monkeypatch.setattr(install, "_load_standalone_installer", lambda: None)
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: fake_runner)

    assert install.status() is True
    assert calls == [["status"]]


def test_status_false_when_delegated_check_fails(fake_env, monkeypatch, capsys):
    hermes_home, _ = fake_env
    _write_config(hermes_home)
    monkeypatch.setattr(install, "_load_standalone_installer", lambda: None)
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: (lambda argv: 1))

    assert install.status() is False
    assert "NOT installed" in capsys.readouterr().out


def test_status_false_when_legacy_link_present(fake_env, monkeypatch, capsys):
    hermes_home, legacy_source = fake_env
    _write_config(hermes_home)
    _link_legacy(hermes_home, legacy_source)
    monkeypatch.setattr(
        install, "_load_standalone_installer", lambda: _FakeProvider(installed=True)
    )

    assert install.status() is False
    out = capsys.readouterr().out
    assert "Legacy hermes_memory_provider" in out
    assert "--migrate" in out


def test_status_false_when_provider_not_installed(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    _write_config(hermes_home)
    monkeypatch.setattr(
        install, "_load_standalone_installer", lambda: _FakeProvider(installed=False)
    )

    assert install.status() is False


def test_status_false_when_config_does_not_select_mnemosyne(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    _write_config(hermes_home, provider="honcho")
    monkeypatch.setattr(
        install, "_load_standalone_installer", lambda: _FakeProvider(installed=True)
    )

    assert install.status() is False


def test_status_false_when_config_missing(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    monkeypatch.setattr(
        install, "_load_standalone_installer", lambda: _FakeProvider(installed=True)
    )

    assert install.status() is False


def test_status_survives_provider_api_drift(fake_env, monkeypatch, capsys):
    hermes_home, _ = fake_env
    _write_config(hermes_home)
    provider = _FakeProvider(plugin_state_error=AttributeError("no plugin_state"))
    monkeypatch.setattr(install, "_load_standalone_installer", lambda: provider)

    assert install.status() is False
    assert "Could not read plugin state" in capsys.readouterr().out


def test_status_accepts_explicit_hermes_home(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    _write_config(hermes_home)
    provider = _FakeProvider(installed=True)
    monkeypatch.setattr(install, "_load_standalone_installer", lambda: provider)
    monkeypatch.setenv("HERMES_HOME", "/nonexistent-does-not-matter")

    assert install.status(str(hermes_home)) is True
    assert provider.state_kwargs[0]["hermes_home_path"] == hermes_home


# ---------------------------------------------------------------------------
# Install: clear failure and delegation
# ---------------------------------------------------------------------------


def test_install_fails_clearly_without_standalone_provider(fake_env, monkeypatch, capsys):
    hermes_home, legacy_source = fake_env
    _link_legacy(hermes_home, legacy_source)
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: None)

    with pytest.raises(SystemExit) as excinfo:
        install.install()

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert install.STANDALONE_DISTRIBUTION in err
    assert "obsolete" in err
    assert "hermes_memory_provider" in err


def test_install_migrates_legacy_link_then_delegates(fake_env, monkeypatch):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)
    provider = _FakeProvider()
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: provider.main)
    monkeypatch.setattr(install, "status", lambda *a, **k: True)

    install.install()

    assert provider.calls == [["install"]]
    assert not target.is_symlink() and not target.exists()


def test_install_passes_force_and_hermes_home(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    provider = _FakeProvider()
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: provider.main)
    monkeypatch.setattr(install, "status", lambda *a, **k: True)

    install.install(force=True, hermes_home_path=str(hermes_home))

    assert provider.calls == [["install", "--force", "--hermes-home", str(hermes_home)]]


def test_install_dry_run_does_not_remove_or_verify(fake_env, monkeypatch):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)
    provider = _FakeProvider()
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: provider.main)
    called = []
    monkeypatch.setattr(install, "status", lambda *a, **k: called.append(True) or True)

    install.install(dry_run=True)

    assert provider.calls == [["install", "--dry-run"]]
    assert called == []
    assert target.is_symlink()  # reported, not removed


def test_install_exits_when_delegated_install_fails(fake_env, monkeypatch, capsys):
    hermes_home, _ = fake_env

    def failing_main(argv):
        return 3

    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: failing_main)

    with pytest.raises(SystemExit) as excinfo:
        install.install()

    assert excinfo.value.code == 3
    assert "Standalone provider install failed" in capsys.readouterr().err


def test_install_exits_when_post_install_status_fails(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    monkeypatch.setattr(
        install, "_standalone_runner", lambda *a, **k: _FakeProvider().main
    )
    monkeypatch.setattr(install, "status", lambda *a, **k: False)

    with pytest.raises(SystemExit) as excinfo:
        install.install()

    assert excinfo.value.code == 1


def test_migrate_only_path_does_not_delegate(fake_env, monkeypatch):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)

    def explode(*args, **kwargs):
        raise AssertionError("must not delegate in --migrate mode")

    monkeypatch.setattr(install, "_standalone_runner", explode)

    install.install(migrate_only=True)

    assert not target.is_symlink() and not target.exists()


def test_migrate_only_is_noop_when_clean(fake_env, monkeypatch, capsys):
    hermes_home, _ = fake_env

    install.install(migrate_only=True)

    assert "No legacy hermes_memory_provider install" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------


def test_uninstall_delegates_and_resets_config(fake_env, monkeypatch):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)
    _write_config(hermes_home)
    provider = _FakeProvider()
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: provider.main)

    install.uninstall()

    assert provider.calls == [["uninstall"]]
    assert not target.is_symlink()
    assert "provider: null" in (hermes_home / "config.yaml").read_text(encoding="utf-8")


def test_uninstall_without_provider_removes_legacy_links_only(fake_env, monkeypatch, capsys):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)
    monkeypatch.setattr(install, "_standalone_runner", lambda *a, **k: None)

    install.uninstall()

    assert not target.is_symlink()
    assert "removing legacy links only" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Config selection helpers (unchanged behaviour)
# ---------------------------------------------------------------------------


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

    assert install._iter_mnemosyne_profiles() == []


def test_profile_without_config_is_skipped(fake_env):
    hermes_home, _ = fake_env
    (hermes_home / "profiles" / "stray").mkdir(parents=True)

    assert install._iter_mnemosyne_profiles() == []


def test_missing_profiles_dir_is_noop(fake_env):
    hermes_home, _ = fake_env

    assert install._iter_mnemosyne_profiles() == []
    assert install.detect_legacy_installs() == []


def test_configure_hermes_sets_provider(fake_env, monkeypatch):
    """_configure_hermes stays reachable and writes provider selection."""
    hermes_home, _ = fake_env
    # Restore the real implementation only; HERMES_HOME stays isolated so this
    # never reads or writes the developer's actual ~/.hermes/config.yaml.
    monkeypatch.setattr(install, "_configure_hermes", _REAL_CONFIGURE_HERMES)
    (hermes_home / "config.yaml").write_text("agent:\n  name: test\n", encoding="utf-8")

    assert install._configure_hermes() is True
    assert install._config_selects_mnemosyne(
        (hermes_home / "config.yaml").read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_status_returns_exit_code(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    _write_config(hermes_home)
    monkeypatch.setattr(
        install, "_load_standalone_installer", lambda: _FakeProvider(installed=True)
    )

    assert install.main(["--status", "--hermes-home", str(hermes_home)]) == 0


def test_main_status_fails_when_provider_missing(fake_env, monkeypatch):
    hermes_home, _ = fake_env
    monkeypatch.setattr(install, "_load_standalone_installer", lambda: None)

    assert install.main(["--status", "--hermes-home", str(hermes_home)]) == 1


def test_main_migrate_removes_legacy_link(fake_env):
    hermes_home, legacy_source = fake_env
    target = _link_legacy(hermes_home, legacy_source)

    assert install.main(["--migrate", "--hermes-home", str(hermes_home)]) == 0
    assert not target.is_symlink()


def test_standalone_probe_names_the_submodule():
    """The probe must look for the distribution, not a submodule import."""
    assert "mnemosyne_hermes" in install._PROVIDER_PROBE
    assert install.STANDALONE_MODULE == "mnemosyne_hermes.install"


def test_delegated_commands_scrub_the_working_directory():
    """``python -c`` prepends cwd, so both delegated commands must drop it."""
    for command in (install._PROVIDER_PROBE, install._DELEGATE_TO_STANDALONE):
        assert "sys.path[:]" in command
        assert "os.getcwd()" in command


def test_path_scrub_removes_the_working_directory(tmp_path):
    """A shadowing cwd must not make an installed package look absent/present.

    The control run proves the scrub, not the environment, is what changes the
    answer: without it cwd *is* searched, which is how ``mnemosyne-install
    --status`` reported a healthy provider as ``Core library: MISSING`` when run
    from a directory holding a ``mnemosyne`` entry.
    """
    (tmp_path / "mnemosyne_shadow_probe.py").write_text("SHADOW = True\n", encoding="utf-8")

    scrubbed = subprocess.run(
        [
            sys.executable, "-c",
            install._PATH_SCRUB
            + "import importlib.util as u\n"
            + "print('FOUND' if u.find_spec('mnemosyne_shadow_probe') else 'ABSENT')\n",
        ],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert scrubbed.stdout.strip() == "ABSENT", scrubbed.stderr

    control = subprocess.run(
        [
            sys.executable, "-c",
            "import importlib.util as u\n"
            "print('FOUND' if u.find_spec('mnemosyne_shadow_probe') else 'ABSENT')\n",
        ],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert control.stdout.strip() == "FOUND", control.stderr
