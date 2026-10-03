"""Regression coverage for issue #642 CLI version reporting."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
INTEGRATION_SRC = REPO_ROOT / "integrations" / "hermes" / "src"


def _run_core_version(
    args: list[str], tmp_path: Path, *, assert_no_data_dir: bool = True
) -> subprocess.CompletedProcess[str]:
    data_dir = tmp_path / "data"
    env = os.environ.copy()
    env["MNEMOSYNE_DATA_DIR"] = str(data_dir)
    result = subprocess.run(
        [sys.executable, "-m", "mnemosyne.cli", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    if assert_no_data_dir:
        assert not data_dir.exists(), "version reporting must not create the Mnemosyne data directory"
    return result


@pytest.mark.parametrize("args", [["--version"], ["version"]])
def test_core_version_routes_report_distribution_metadata_without_creating_data_dir(tmp_path, args):
    result = _run_core_version(args, tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"Mnemosyne {importlib.metadata.version('mnemosyne-memory')}\n"
    assert result.stderr == ""


def test_core_help_advertises_standalone_version_command(tmp_path):
    result = _run_core_version(["--help"], tmp_path, assert_no_data_dir=False)

    assert result.returncode == 0, result.stderr
    assert "  version                                Show installed version\n" in result.stdout
    assert result.stderr == ""


def _load_host_provider_cli(monkeypatch):
    for module_name in tuple(sys.modules):
        if module_name == "hermes_memory_provider" or module_name.startswith("hermes_memory_provider."):
            del sys.modules[module_name]
    from hermes_memory_provider import cli

    return cli


def _load_integration_cli(monkeypatch):
    monkeypatch.syspath_prepend(str(INTEGRATION_SRC))
    for module_name in tuple(sys.modules):
        if module_name == "mnemosyne_hermes" or module_name.startswith("mnemosyne_hermes."):
            del sys.modules[module_name]
    from mnemosyne_hermes import cli

    return cli


def _parse_host_version_args(cli):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command")
    mnemosyne = commands.add_parser("mnemosyne")
    cli.register_cli(mnemosyne)
    return parser.parse_args(["mnemosyne", "version"])


@pytest.mark.parametrize("loader", [_load_host_provider_cli, _load_integration_cli])
def test_host_cli_version_dispatcher_parity_without_beam_or_optional_metadata(monkeypatch, capsys, loader):
    """Both host CLI implementations dispatch version before bank/Beam setup."""
    cli = loader(monkeypatch)

    def fake_version(name: str) -> str:
        if name == "mnemosyne-memory":
            return "3.15.1"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cli.importlib.metadata, "version", fake_version)
    from mnemosyne.core import beam as beam_module

    monkeypatch.setattr(
        beam_module,
        "BeamMemory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("version must not construct BeamMemory")),
    )
    if hasattr(cli, "_resolve_cli_bank"):
        monkeypatch.setattr(cli, "_resolve_cli_bank", lambda *_args: None)

    args = _parse_host_version_args(cli)
    assert args.func(args) == 0
    assert capsys.readouterr().out == "Mnemosyne 3.15.1\nMnemosyne Hermes unavailable\n"


@pytest.mark.parametrize("loader", [_load_host_provider_cli, _load_integration_cli])
def test_host_cli_version_does_not_register_or_replace_host_backend(monkeypatch, loader):
    """Version inspection must leave the process-global host LLM registry intact."""
    cli = loader(monkeypatch)
    adapter = importlib.import_module(f"{cli.__package__}.hermes_llm_adapter")
    registrations = []
    monkeypatch.setattr(adapter, "register_hermes_host_llm", lambda: registrations.append(True))

    from mnemosyne.core.llm_backends import CallableLLMBackend, get_host_llm_backend, set_host_llm_backend

    sentinel = CallableLLMBackend("sentinel", lambda *_args, **_kwargs: None)
    set_host_llm_backend(sentinel)
    try:
        args = _parse_host_version_args(cli)
        assert args.func(args) == 0
        assert registrations == []
        assert get_host_llm_backend() is sentinel
    finally:
        set_host_llm_backend(None)


def test_packaged_host_cli_version_returns_before_bank_resolution(monkeypatch, capsys):
    """The packaged host dispatcher must not resolve a bank for ``version``."""
    cli = _load_integration_cli(monkeypatch)

    def fake_version(name: str) -> str:
        if name == "mnemosyne-memory":
            return "3.15.1"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cli.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(
        cli,
        "_resolve_cli_bank",
        lambda *_args: (_ for _ in ()).throw(AssertionError("version must not resolve a bank")),
    )

    args = _parse_host_version_args(cli)
    assert args.func(args) == 0
    assert capsys.readouterr().out == "Mnemosyne 3.15.1\nMnemosyne Hermes unavailable\n"


@pytest.mark.parametrize(
    "path",
    [
        REPO_ROOT / "hermes_memory_provider" / "cli.py",
        INTEGRATION_SRC / "mnemosyne_hermes" / "cli.py",
    ],
)
def test_host_cli_version_paths_do_not_reference_package_version_globals(path):
    source = path.read_text()
    assert "__version__" not in source
    assert "__author__" not in source


@pytest.mark.parametrize("argv", [["--version"], ["version"]])
def test_installer_version_routes_report_both_distributions(monkeypatch, capsys, argv):
    _load_integration_cli(monkeypatch)
    from mnemosyne_hermes import install

    def fake_version(name: str) -> str:
        return {"mnemosyne-memory": "3.15.1", "mnemosyne-hermes": "0.5.0"}[name]

    monkeypatch.setattr(install.importlib.metadata, "version", fake_version)

    assert install.main(argv) == 0
    assert capsys.readouterr().out == "Mnemosyne 3.15.1\nMnemosyne Hermes 0.5.0\n"


def test_hermes_provider_version_reports_both_distributions_without_opening_beam(monkeypatch, capsys):
    cli = _load_integration_cli(monkeypatch)

    def fake_version(name: str) -> str:
        return {"mnemosyne-memory": "3.15.1", "mnemosyne-hermes": "0.5.0"}[name]

    monkeypatch.setattr(cli.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(
        cli,
        "_resolve_cli_bank",
        lambda *_args: (_ for _ in ()).throw(AssertionError("version must not initialize a bank")),
    )

    assert cli.mnemosyne_command(argparse.Namespace(mnemosyne_cmd="version")) == 0
    assert capsys.readouterr().out == "Mnemosyne 3.15.1\nMnemosyne Hermes 0.5.0\n"


def test_hermes_provider_version_handles_missing_optional_integration_metadata(monkeypatch, capsys):
    cli = _load_integration_cli(monkeypatch)

    def fake_version(name: str) -> str:
        if name == "mnemosyne-memory":
            return "3.15.1"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(cli.importlib.metadata, "version", fake_version)
    monkeypatch.setattr(
        cli,
        "_resolve_cli_bank",
        lambda *_args: (_ for _ in ()).throw(AssertionError("version must not initialize a bank")),
    )

    assert cli.mnemosyne_command(argparse.Namespace(mnemosyne_cmd="version")) == 0
    assert capsys.readouterr().out == "Mnemosyne 3.15.1\nMnemosyne Hermes unavailable\n"
