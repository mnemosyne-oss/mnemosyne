"""Tests for profile-isolation-aware bank resolution in the hermes CLI.

Regression coverage for #362: `hermes mnemosyne stats` (and friends) used to
always bind to the default/legacy bank, so under `profile_isolation` they
reported empty state while the profile bank held the real data.

Standalone-import coverage for #373: when Hermes loads the plugin CLI module
via ``importlib.util.spec_from_file_location()``, the module has no parent
package and the previous relative import of ``MnemosyneMemoryProvider`` failed
silently, again falling back to the default bank.
"""

import importlib.util
import types
from pathlib import Path

from mnemosyne_hermes.cli import _resolve_cli_bank, _get_provider_class
import mnemosyne_hermes as _mnh


def _args(**kw):
    return types.SimpleNamespace(**kw)


def _write_config(home, isolation):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        f"memory:\n  mnemosyne:\n    profile_isolation: {isolation}\n"
    )


def test_explicit_bank_takes_precedence_and_is_sanitized(monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert _resolve_cli_bank(_args(bank="Work Stuff"), "stats") == "work_stuff"


def test_profile_bank_resolved_when_isolation_enabled(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "zedd"
    _write_config(home, "true")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_cli_bank(_args(bank=None), "stats") == "zedd"


def test_default_bank_when_isolation_disabled(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "zedd"
    _write_config(home, "false")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_cli_bank(_args(bank=None), "stats") is None


def test_default_bank_when_no_config(tmp_path, monkeypatch):
    home = tmp_path / "profiles" / "zedd"
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_cli_bank(_args(bank=None), "stats") is None


def test_root_hermes_home_is_treated_as_default(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    _write_config(home, "true")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The base profile's HERMES_HOME basename (.hermes) maps to the shared bank.
    assert _resolve_cli_bank(_args(bank=None), "stats") is None


def test_import_bank_arg_does_not_redirect_target(tmp_path, monkeypatch):
    # `import --bank` names the SOURCE provider bank (e.g. Hindsight), not the
    # Mnemosyne destination, so it must not be used as the CLI's target bank.
    home = tmp_path / "profiles" / "zedd"
    _write_config(home, "true")
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _resolve_cli_bank(_args(bank="hindsight"), "import") == "zedd"


def test_get_provider_class_returns_real_class():
    """The helper must return an actual class, not None or a dummy."""
    cls = _get_provider_class()
    assert cls is not None
    assert hasattr(cls, "_sanitize_bank_name")


def test_standalone_load_via_spec_resolves_profile_bank(tmp_path, monkeypatch):
    """End-to-end standalone load: CLI module loaded from file path
    (no __package__) resolves the active profile bank."""
    home = tmp_path / "profiles" / "work"
    _write_config(home, "true")

    # Locate the installed package's cli.py on disk
    pkg_dir = Path(_mnh.__file__).resolve().parent
    cli_py = pkg_dir / "cli.py"
    assert cli_py.exists(), f"cli.py not found next to package at {pkg_dir}"

    spec = importlib.util.spec_from_file_location("_clitest_cli", str(cli_py))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # The standalone load context should give us no package metadata
    pre_pkg = getattr(mod, "__package__", None)
    assert pre_pkg in (None, ""), f"expected no package, got {pre_pkg!r}"
    spec.loader.exec_module(mod)

    # The module should expose the patched helper + resolver
    assert hasattr(mod, "_resolve_cli_bank")

    # Verify the helper picks the absolute-import path
    cls = mod._get_provider_class()
    assert cls is not None
    assert hasattr(cls, "_sanitize_bank_name")

    # Verify bank resolution works end-to-end without leaking HERMES_HOME
    # into later tests.
    monkeypatch.setenv("HERMES_HOME", str(home))
    result = mod._resolve_cli_bank(_args(bank=None), "stats")
    assert result == "work", (
        f"standalone load: expected 'work', got {result!r}. "
        "This indicates the absolute-import fallback failed."
    )
