"""Focused coverage for wrapper validation timeout configuration."""

import os
import subprocess
import sys
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemosyne_hermes import install


def test_install_cli_passes_default_and_override_import_timeout(tmp_path, monkeypatch):
    received = []
    monkeypatch.setattr(
        install,
        "run_install",
        lambda **kwargs: received.append(kwargs) or 0,
    )

    assert install.main(["--hermes-home", str(tmp_path), "install"]) == 0
    assert install.main(
        ["--hermes-home", str(tmp_path), "install", "--import-timeout", "75"]
    ) == 0

    assert [call["import_timeout"] for call in received] == [60.0, 75.0]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_install_cli_rejects_non_positive_or_nonfinite_import_timeout(value, capsys):
    with pytest.raises(SystemExit) as exc_info:
        install.main(["install", f"--import-timeout={value}"])

    assert exc_info.value.code == 2
    assert "positive finite number" in capsys.readouterr().err


@pytest.mark.parametrize("import_timeout", [60.0, 90.0])
def test_wrapper_validation_timeout_reaches_both_probes(tmp_path, monkeypatch, import_timeout):
    observed_timeouts = []

    def successful_probe(command, **kwargs):
        observed_timeouts.append(kwargs["timeout"])
        if "-S" in command:
            return subprocess.CompletedProcess(command, 0, "0.0-test\n", "")
        return subprocess.CompletedProcess(command, 0, f"{tmp_path}\n", "")

    monkeypatch.setattr(install.subprocess, "run", successful_probe)

    python, site_packages = install._validated_wrapper_environment(
        Path(sys.executable), import_timeout=import_timeout
    )

    assert python == Path(sys.executable)
    assert site_packages == tmp_path
    assert observed_timeouts == [import_timeout, import_timeout]


def test_wrapper_validation_rejects_selected_python_without_mnemosyne_core(
    tmp_path, monkeypatch
):
    """Wrapper --python must prove core imports, not only the fallback package."""
    selected_python = tmp_path / "selected-python"
    selected_python.write_text("#!/bin/sh\n", encoding="utf-8")
    selected_python.chmod(0o755)
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    commands = []

    def probe(command, **kwargs):
        commands.append(command)
        if "-S" not in command:
            return subprocess.CompletedProcess(command, 0, f"{site_packages}\n", "")
        if "import mnemosyne.core.beam" in command[-1]:
            return subprocess.CompletedProcess(
                command, 1, "", "ModuleNotFoundError: No module named 'mnemosyne.core'"
            )
        return subprocess.CompletedProcess(command, 0, "0.0-test\n", "")

    monkeypatch.setattr(install.subprocess, "run", probe)

    with pytest.raises(RuntimeError, match="cannot import required mnemosyne core"):
        install._validated_wrapper_environment(selected_python)

    assert any("import mnemosyne.core.beam" in command[-1] for command in commands)


def test_wrapper_validation_rejects_real_selected_venv_without_mnemosyne_core(tmp_path):
    """A selected venv may import the graceful fallback without containing core."""
    environment = tmp_path / "selected-environment"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform.startswith("win32") else "bin/python")
    project = Path(__file__).resolve().parent.parent
    install_result = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", "-e", str(project)],
        capture_output=True,
        text=True,
    )
    assert install_result.returncode == 0, install_result.stderr

    site_packages = install._site_packages_for_python(python)
    probe_cwd = tmp_path / "fallback-probe-cwd"
    probe_cwd.mkdir()
    fallback_probe = subprocess.run(
        [
            str(python),
            "-S",
            "-c",
            "import site\n"
            f"site.addsitedir({str(site_packages)!r})\n"
            "import mnemosyne_hermes\n"
            "print('fallback-imported')\n"
            "import mnemosyne.core.beam\n",
        ],
        capture_output=True,
        text=True,
        cwd=probe_cwd,
        env={key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
    )
    assert fallback_probe.returncode != 0
    assert fallback_probe.stdout.strip() == "fallback-imported"
    assert "No module named 'mnemosyne'" in fallback_probe.stderr

    with pytest.raises(RuntimeError, match=r"mnemosyne\.core\.beam"):
        install._validated_wrapper_environment(python)


def test_plugin_state_accepts_11_second_healthy_wrapper_import(tmp_path, monkeypatch):
    """Status must share the 60-second wrapper-validation policy with install."""
    target = tmp_path / "plugins" / "mnemosyne"
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    install._write_wrapper_plugin(target, python=Path(sys.executable), site_packages=site_packages)
    observed_timeouts = []

    def slow_but_healthy_import(command, **kwargs):
        timeout = kwargs["timeout"]
        observed_timeouts.append(timeout)
        if timeout < 11:
            raise subprocess.TimeoutExpired(command, timeout)
        return subprocess.CompletedProcess(command, 0, "0.0-test\n", "")

    monkeypatch.setattr(install.subprocess, "run", slow_but_healthy_import)

    state = install.plugin_state(hermes_home_path=tmp_path)

    assert observed_timeouts == [60.0]
    assert state.status == "installed"
    assert state.installed is True
    assert state.wrapper_import_ok is True


def test_wrapper_install_timeout_preserves_existing_target(tmp_path, monkeypatch):
    target = tmp_path / "plugins" / "mnemosyne"
    target.mkdir(parents=True)
    sentinel = target / "keep"
    sentinel.write_text("existing wrapper", encoding="utf-8")
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()

    def probe_with_slow_import(command, **kwargs):
        if "-S" in command:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, f"{site_packages}\n", "")

    monkeypatch.setattr(install.subprocess, "run", probe_with_slow_import)

    with pytest.raises(RuntimeError, match="wrapper validation timed out"):
        install.install_plugin(
            hermes_home_path=tmp_path,
            force=True,
            mode="wrapper",
            python=sys.executable,
            link_profiles=False,
        )

    assert sentinel.read_text(encoding="utf-8") == "existing wrapper"


def test_wrapper_import_timeout_diagnostic_names_interpreter_duration_and_retry(tmp_path, monkeypatch):
    def timed_out(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(install.subprocess, "run", timed_out)

    ok, error, invalid_runtime = install._check_wrapper_import(
        tmp_path, Path(sys.executable), import_timeout=75
    )

    assert ok is False
    assert invalid_runtime is False
    assert error is not None
    assert str(Path(sys.executable)) in error
    assert "75 seconds" in error
    assert "--import-timeout 150" in error


def test_wrapper_import_timeout_diagnostic_quotes_windows_python_path(monkeypatch):
    python = Path("C:/Program Files/Hermes/venv/python.exe")
    retry_args = [
        "mnemosyne-hermes",
        "install",
        "--mode",
        "wrapper",
        "--python",
        str(python),
        "--import-timeout",
        "120",
    ]
    monkeypatch.setattr(install, "os", SimpleNamespace(name="nt"))

    error = install._timeout_diagnostic(python, 60)

    assert error.endswith(f"Retry with: {subprocess.list2cmdline(retry_args)}")
    assert '"C:/Program Files/Hermes/venv/python.exe"' in error
