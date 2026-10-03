"""Non-blocking diagnostic for wrappers bound to disposable Hermes PM generations."""

from types import SimpleNamespace

import pytest

from mnemosyne import upgrade_hermes
from mnemosyne_hermes import install


@pytest.mark.parametrize("name", ["python", "python3", "python.exe"])
def test_pm_layout_aliases_and_persistent_side_venv(tmp_path, name):
    home = tmp_path / "home"
    folder = "Scripts" if name.endswith(".exe") else "bin"
    pm = home / "installs" / "id" / "environments" / "generation" / "venv" / folder / name
    alias = home / "installs" / "id" / "environments" / "old" / ".." / "generation" / "venv" / folder / name
    assert install._hermes_pm_generation_target(alias, home)
    assert install._hermes_pm_generation_target(pm, home)
    assert not install._hermes_pm_generation_target(home / ".mnemosyne" / "venv" / folder / name, home)
    assert not install._hermes_pm_generation_target(home / "installs" / "id" / "venv" / folder / name, home)


@pytest.mark.parametrize("path, expected", [
    (r"C:\Hermes\installs\id\environments\gen\venv\Scripts\python.exe", True),
    (r"c:\hermes\installs\id\environments\old\..\gen\venv\Scripts\python.exe", True),
    (r"C:\Hermes\.mnemosyne\venv\Scripts\python.exe", False),
    (r"D:\Hermes\installs\id\environments\gen\venv\Scripts\python.exe", False),
])
def test_windows_lexical_shapes(path, expected):
    assert install._hermes_pm_generation_target(path, r"C:\Hermes") is expected


def _runtime(monkeypatch, home, python):
    python.parent.mkdir(parents=True, exist_ok=True)
    python.touch()
    site = home / "site-packages"
    site.mkdir(exist_ok=True)
    monkeypatch.setattr(install, "_site_packages_for_python", lambda *a, **kw: site)
    monkeypatch.setattr(install, "_check_wrapper_import", lambda *a, **kw: (True, None, False))


def test_new_and_existing_pm_wrapper_reregistration_and_status(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    python = home / "installs" / "id" / "environments" / "gen" / "venv" / "bin" / "python"
    _runtime(monkeypatch, home, python)
    target = install.install_plugin(hermes_home_path=home, mode="wrapper", python=python)
    assert "replaceable Hermes PM generation" in capsys.readouterr().err
    manifest = target / install.WRAPPER_MANIFEST_NAME
    assert manifest.is_file()
    install.install_plugin(hermes_home_path=home, mode="wrapper", python=python, force=True)
    assert "replaceable Hermes PM generation" in capsys.readouterr().err
    assert manifest.is_file()
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kw: None)
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    assert install.main(["--hermes-home", str(home), "status"]) == 0
    assert "replaceable Hermes PM generation" in capsys.readouterr().out


def test_dry_run_warning_without_writes_and_side_venv_silence(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    pm = home / "installs" / "id" / "environments" / "gen" / "venv" / "bin" / "python3"
    side = home / ".mnemosyne" / "venv" / "bin" / "python"
    _runtime(monkeypatch, home, pm)
    side.parent.mkdir(parents=True)
    side.touch()
    for python, warned in [(pm, True), (side, False)]:
        assert install.main(["--hermes-home", str(home), "install", "--mode", "wrapper", "--python", str(python), "--dry-run"]) == 0
        assert ("replaceable Hermes PM generation" in capsys.readouterr().out) is warned
    assert not install.plugin_target_dir(home).exists()
    install.install_plugin(hermes_home_path=home, mode="wrapper", python=side)
    assert "replaceable Hermes PM generation" not in capsys.readouterr().err
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kw: None)
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    assert install.main(["--hermes-home", str(home), "status"]) == 0
    assert "replaceable Hermes PM generation" not in capsys.readouterr().out


def test_upgrade_reregisters_existing_pm_wrapper_without_rejection(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    python = home / "installs" / "id" / "environments" / "gen" / "venv" / "bin" / "python"
    _runtime(monkeypatch, home, python)
    target = install.install_plugin(hermes_home_path=home, mode="wrapper", python=python)
    capsys.readouterr()
    monkeypatch.setattr(upgrade_hermes, "detect_install_method", lambda: "pip")
    monkeypatch.setattr(upgrade_hermes, "get_current_version", lambda: "0.7.1")
    monkeypatch.setattr(upgrade_hermes, "get_current_core_version", lambda: "3.15.1")
    monkeypatch.setattr(upgrade_hermes, "check_available_version", lambda method: "0.7.2")
    monkeypatch.setattr(upgrade_hermes, "run_upgrade_command", lambda *a, **kw: (0, ""))
    assert upgrade_hermes.upgrade_command(SimpleNamespace(hermes_home=str(home))) == 0
    assert "replaceable Hermes PM generation" in capsys.readouterr().err
    assert (target / install.WRAPPER_MANIFEST_NAME).is_file()
    assert install.plugin_state(hermes_home_path=home).wrapper_python == python
