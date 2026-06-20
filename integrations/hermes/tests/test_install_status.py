import sys
from pathlib import Path

from mnemosyne_hermes import install


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


def test_install_plugin_wrapper_creates_persistent_shim(tmp_path):
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
    assert "_PYTHON" in init_source
    assert "_SITE" in init_source
    assert (target / "plugin.yaml").is_file()

    state = install.plugin_state(hermes_home_path=tmp_path)
    assert state.status == "installed"
    assert state.installed is True
    assert state.mode == "wrapper"
    assert state.wrapper_python == Path(sys.executable)
    assert state.wrapper_site_packages is not None
    assert state.wrapper_import_ok is True


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
