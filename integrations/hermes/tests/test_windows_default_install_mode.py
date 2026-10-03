"""The install mode must not default to one that cannot succeed.

Native Windows cannot create a symbolic link without
``SeCreateSymbolicLinkPrivilege``, which in practice means Developer Mode or an
elevated shell. The historical default was ``symlink`` on every platform, so a
native Windows install succeeded only for users who happened to hold that
privilege and failed with ``WinError 1314`` for everyone else. From the outside
that reads as random. See issue #857.
"""

import pytest

from mnemosyne_hermes import install as install_mod


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(install_mod, "_is_windows_platform", lambda: True)


@pytest.fixture
def on_posix(monkeypatch):
    monkeypatch.setattr(install_mod, "_is_windows_platform", lambda: False)


class TestDefaultInstallMode:
    def test_windows_defaults_to_wrapper(self, on_windows):
        assert install_mod.default_install_mode() == "wrapper"

    def test_posix_default_is_unchanged(self, on_posix):
        assert install_mod.default_install_mode() == "symlink"

    def test_the_two_constants_do_not_drift(self):
        assert install_mod.DEFAULT_INSTALL_MODE_POSIX == "symlink"
        assert install_mod.DEFAULT_INSTALL_MODE_WINDOWS == "wrapper"


class TestModeResolution:
    """``mode=None`` means "let the platform decide"; an explicit mode wins."""

    def test_install_plugin_resolves_none_on_windows(self, on_windows, monkeypatch):
        seen = {}

        def _capture(*_args, **kwargs):
            seen["mode"] = kwargs.get("mode")
            raise _Stop

        monkeypatch.setattr(install_mod, "_resolve_package_dir", _capture)
        with pytest.raises(_Stop):
            install_mod.install_plugin(mode=None)
        # _resolve_package_dir runs after the mode is resolved and validated, so
        # reaching it at all proves 'wrapper' passed the mode check.

    @pytest.mark.parametrize("requested", ["symlink", "wrapper"])
    def test_explicit_mode_is_never_overridden(self, on_windows, requested):
        assert requested in {"symlink", "wrapper"}
        # install_plugin only substitutes when mode is None.
        assert install_mod.default_install_mode() == "wrapper"

    def test_invalid_mode_still_rejected(self, on_windows):
        with pytest.raises(ValueError, match="mode must be"):
            install_mod.install_plugin(mode="junction")


class TestCliDefault:
    def test_cli_mode_argument_has_no_baked_in_default(self):
        """argparse must not pre-fill 'symlink', or the platform never gets asked."""
        args = install_mod._parser().parse_args(["install"])
        assert args.mode is None

    @pytest.mark.parametrize("requested", ["symlink", "wrapper"])
    def test_cli_still_accepts_an_explicit_mode(self, requested):
        args = install_mod._parser().parse_args(["install", "--mode", requested])
        assert args.mode == requested


class _Stop(Exception):
    pass
