"""Tests for profile-isolation-aware bank resolution in the hermes CLI.

Regression coverage for #362: `hermes mnemosyne stats` (and friends) used to
always bind to the default/legacy bank, so under `profile_isolation` they
reported empty state while the profile bank held the real data.
"""

import types

from mnemosyne_hermes.cli import _resolve_cli_bank


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
