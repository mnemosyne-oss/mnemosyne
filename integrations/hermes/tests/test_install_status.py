import hashlib
import sys
from pathlib import Path

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
    monkeypatch.setattr(install, "_find_hermes_python", lambda: None)

    rc = install.main(["--hermes-home", str(tmp_path), "install", "--dry-run"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "Skill target file:" in out
    assert "Skill action: Would install bundled skill" in out
    assert not install.skill_target_file(tmp_path).exists()


def test_status_reports_skill_state(tmp_path, capsys, monkeypatch):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    (target / "__init__.py").write_text("class MnemosyneMemoryProvider: pass\n")
    install.install_bundled_skill(hermes_home_path=tmp_path)
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "_find_hermes_python", lambda: None)

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
