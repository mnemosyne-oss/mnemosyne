"""Regression coverage for Windows symlink-privilege install failures (#807)."""

from pathlib import Path

import pytest

from mnemosyne_hermes import install


class _WindowsSymlinkPrivilegeError(OSError):
    """Cross-platform stand-in for Windows ERROR_PRIVILEGE_NOT_HELD (1314)."""

    def __init__(self) -> None:
        super().__init__("A required privilege is not held by the client")
        self.winerror = 1314


def _raise_symlink_privilege_error(*_args) -> None:
    raise _WindowsSymlinkPrivilegeError()


def test_run_install_explains_windows_symlink_privilege_and_safe_wrapper_retry(
    tmp_path, monkeypatch, capsys
):
    """WinError 1314 must fail closed with a retry using the resolved Hermes Python."""
    hermes_python = Path(r"C:\Program Files\Hermes\venv\Scripts\python.exe")
    target = tmp_path / "plugins" / "mnemosyne"

    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: hermes_python)
    monkeypatch.setattr(install, "check_mnemosyne_core_for_hermes_python", lambda _: "4.0")
    monkeypatch.setattr(install.os, "symlink", _raise_symlink_privilege_error)

    rc = install.run_install(hermes_home_path=tmp_path)
    stderr = capsys.readouterr().err

    assert rc == 1
    assert "Windows symbolic-link privilege" in stderr
    assert "Developer Mode" in stderr
    assert "persistent no-symlink-privilege alternative" in stderr.lower()
    assert (
        'mnemosyne-hermes install --mode wrapper --python '
        '"C:\\Program Files\\Hermes\\venv\\Scripts\\python.exe"'
    ) in stderr
    assert not target.exists()
    assert not target.is_symlink()


def test_run_install_does_not_offer_wrapper_without_a_resolved_hermes_python(
    tmp_path, monkeypatch, capsys
):
    """The no-bootstrap path must not invent a wrapper interpreter retry."""
    monkeypatch.setattr(install, "check_mnemosyne_core", lambda: True)
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)
    monkeypatch.setattr(install.os, "symlink", _raise_symlink_privilege_error)

    rc = install.run_install(hermes_home_path=tmp_path, no_bootstrap=True)
    stderr = capsys.readouterr().err

    assert rc == 1
    assert "cannot be generated" in stderr
    assert "Locate the Hermes interpreter" in stderr
    assert "mnemosyne-hermes install --mode wrapper" not in stderr


def test_non_1314_symlink_error_still_propagates_without_a_wrapper_fallback(tmp_path, monkeypatch):
    """Only WinError 1314 changes CLI behavior; other symlink failures stay errors."""
    target = tmp_path / "plugins" / "mnemosyne"

    def raise_access_denied(*_args) -> None:
        raise OSError("access denied")

    monkeypatch.setattr(install.os, "symlink", raise_access_denied)

    with pytest.raises(OSError, match="access denied"):
        install.install_plugin(hermes_home_path=tmp_path)

    assert not target.exists()
    assert not target.is_symlink()
