"""Upgrade regression coverage for Hermes wrapper installs."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemosyne_hermes import install, upgrade


@pytest.mark.parametrize(
    ("existing_mode", "stored_python", "expected_mode", "expected_python"),
    [
        ("wrapper", Path("/selected/venv/bin/python"), "wrapper", Path("/selected/venv/bin/python")),
        ("symlink", None, "symlink", None),
    ],
)
def test_upgrade_reregisters_with_existing_wrapper_mode_and_interpreter(
    monkeypatch,
    tmp_path,
    existing_mode,
    stored_python,
    expected_mode,
    expected_python,
):
    state = install.PluginState(
        status="installed",
        installed=True,
        target=tmp_path / "plugins" / "mnemosyne",
        message="installed",
        mode=existing_mode,
        wrapper_python=stored_python,
    )
    calls = []
    monkeypatch.setattr(upgrade, "detect_install_method", lambda: "pip")
    monkeypatch.setattr(upgrade, "get_current_version", lambda: "0.5.0")
    monkeypatch.setattr(upgrade, "get_current_core_version", lambda: "3.11.1")
    monkeypatch.setattr(upgrade, "check_available_version", lambda method: "0.5.1")
    monkeypatch.setattr(upgrade, "run_upgrade_command", lambda method, capture: (0, ""))
    monkeypatch.setattr(install, "plugin_state", lambda **kwargs: state)
    monkeypatch.setattr(install, "profile_links_enabled", lambda **kwargs: True)
    monkeypatch.setattr(
        install,
        "run_install",
        lambda **kwargs: calls.append(kwargs) or 0,
    )

    assert upgrade.upgrade_command(SimpleNamespace(hermes_home=str(tmp_path))) == 0
    assert calls == [
        {
            "force": True,
            "hermes_home_path": str(tmp_path),
            "mode": expected_mode,
            "python": expected_python,
            "link_profiles": True,
        }
    ]


def test_upgrade_preserves_root_only_profile_link_preference(monkeypatch, tmp_path):
    state = install.PluginState(
        status="installed",
        installed=True,
        target=tmp_path / "plugins" / "mnemosyne",
        message="installed",
        mode="wrapper",
        wrapper_python=Path("/selected/venv/bin/python"),
    )
    calls = []
    monkeypatch.setattr(upgrade, "detect_install_method", lambda: "pip")
    monkeypatch.setattr(upgrade, "get_current_version", lambda: "0.5.0")
    monkeypatch.setattr(upgrade, "get_current_core_version", lambda: "3.11.1")
    monkeypatch.setattr(upgrade, "check_available_version", lambda method: "0.5.1")
    monkeypatch.setattr(upgrade, "run_upgrade_command", lambda method, capture: (0, ""))
    monkeypatch.setattr(install, "plugin_state", lambda **kwargs: state)
    monkeypatch.setattr(install, "profile_links_enabled", lambda **kwargs: False)
    monkeypatch.setattr(install, "run_install", lambda **kwargs: calls.append(kwargs) or 0)

    assert upgrade.upgrade_command(SimpleNamespace(hermes_home=str(tmp_path))) == 0
    assert calls[0]["link_profiles"] is False


def test_upgrade_preserves_default_profile_links_when_profile_is_added_later(monkeypatch, tmp_path):
    install.install_plugin(hermes_home_path=tmp_path)
    child = tmp_path / "profiles" / "child"
    child.mkdir(parents=True)
    (child / "config.yaml").write_text("memory:\n  provider: mnemosyne\n", encoding="utf-8")
    state = install.PluginState(
        status="installed",
        installed=True,
        target=tmp_path / "plugins" / "mnemosyne",
        message="installed",
        mode="symlink",
    )
    calls = []
    monkeypatch.setattr(upgrade, "detect_install_method", lambda: "pip")
    monkeypatch.setattr(upgrade, "get_current_version", lambda: "0.5.0")
    monkeypatch.setattr(upgrade, "get_current_core_version", lambda: "3.11.1")
    monkeypatch.setattr(upgrade, "check_available_version", lambda method: "0.5.1")
    monkeypatch.setattr(upgrade, "run_upgrade_command", lambda method, capture: (0, ""))
    monkeypatch.setattr(install, "plugin_state", lambda **kwargs: state)
    monkeypatch.setattr(install, "run_install", lambda **kwargs: calls.append(kwargs) or 0)

    assert upgrade.upgrade_command(SimpleNamespace(hermes_home=str(tmp_path))) == 0
    assert calls[0]["link_profiles"] is True


def test_upgrade_messages_preserve_root_only_profile_link_preference(monkeypatch, tmp_path, capsys):
    state = install.PluginState(
        status="installed",
        installed=True,
        target=tmp_path / "plugins" / "mnemosyne",
        message="installed",
        mode="symlink",
    )
    monkeypatch.setattr(upgrade, "detect_install_method", lambda: "pip")
    monkeypatch.setattr(upgrade, "get_current_version", lambda: "0.5.0")
    monkeypatch.setattr(upgrade, "get_current_core_version", lambda: "3.11.1")
    monkeypatch.setattr(upgrade, "check_available_version", lambda method: "0.5.1")
    monkeypatch.setattr(install, "plugin_state", lambda **kwargs: state)
    monkeypatch.setattr(install, "profile_links_enabled", lambda **kwargs: False)

    assert upgrade.upgrade_command(SimpleNamespace(hermes_home=str(tmp_path), dry_run=True)) == 0
    assert "mnemosyne-hermes install --force --no-profile-links" in capsys.readouterr().out


def test_upgrade_stops_before_package_upgrade_for_invalid_wrapper(monkeypatch, tmp_path, capsys):
    state = install.PluginState(
        status="invalid_wrapper",
        installed=False,
        target=tmp_path / "plugins" / "mnemosyne",
        message="Invalid Mnemosyne wrapper manifest schema",
        mode="wrapper",
        wrapper_import_ok=False,
    )
    upgrade_calls = []
    install_calls = []
    monkeypatch.setattr(install, "plugin_state", lambda **kwargs: state)
    monkeypatch.setattr(install, "run_install", lambda **kwargs: install_calls.append(kwargs))
    monkeypatch.setattr(upgrade, "detect_install_method", lambda: "pip")
    monkeypatch.setattr(upgrade, "run_upgrade_command", lambda *args: upgrade_calls.append(args))

    assert upgrade.upgrade_command(SimpleNamespace(hermes_home=str(tmp_path))) == 1
    assert upgrade_calls == []
    assert install_calls == []
    assert "Cannot safely upgrade an invalid Mnemosyne wrapper" in capsys.readouterr().out
