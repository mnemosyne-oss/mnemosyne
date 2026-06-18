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
    assert "target is missing" in state.message


def test_plugin_state_reports_invalid_directory(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("# no provider markers\n")

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "invalid_provider"
    assert state.installed is False
    assert "does not look like a Mnemosyne provider" in state.message


def test_plugin_state_accepts_valid_provider_directory(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("class MnemosyneMemoryProvider: pass\n")

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert state.status == "installed"
    assert state.installed is True


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


def test_is_installed_stays_false_for_broken_symlink(tmp_path):
    target = tmp_path / "plugins" / "mnemosyne"
    target.parent.mkdir(parents=True)
    target.symlink_to(tmp_path / "missing", target_is_directory=True)

    assert install.is_installed(hermes_home_path=tmp_path) is False
