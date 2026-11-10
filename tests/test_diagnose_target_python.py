import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import pytest

from mnemosyne import cli, diagnose


def _completed(command, returncode=0, stdout="", stderr=""):
    """Return a subprocess result with the same fields as the real runner."""
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _check(payload, name):
    """Find a named check in either runtime or full-diagnostic output."""
    entries = payload.get("checks") or payload.get("entries", [])
    return next(entry for entry in entries if entry["check"] == name)


@pytest.mark.parametrize("explicit_db", [False, True])
def test_runtime_checks_run_in_the_selected_venv(tmp_path, monkeypatch, explicit_db):
    """Inspect a real isolated venv and keep its absolute path out of output."""
    environment = tmp_path / "provider-env"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    site_packages = next(environment.rglob("site-packages"))
    package = site_packages / "mnemosyne"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = 'test'\n", encoding="utf-8")
    shutil.copyfile(
        Path(__file__).resolve().parents[1] / "mnemosyne" / "runtime_diagnostics.py",
        package / "runtime_diagnostics.py",
    )
    (site_packages / "fastembed.py").write_text("__version__ = 'provider-only'\n", encoding="utf-8")

    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    kwargs = {}
    if explicit_db:
        from mnemosyne.core.beam import BeamMemory

        selected_db = tmp_path / "selected" / "mnemosyne.db"
        beam = BeamMemory(session_id="hermes_default", db_path=selected_db)
        beam.remember("selected database marker", scope="global")
        kwargs["db_path"] = selected_db

    payload = diagnose.run_diagnostics(python_executable=str(python), **kwargs)

    assert payload["runtime_scope"] == "target_interpreter"
    assert payload["runtime_executable"] == python.name
    assert _check(payload, "fastembed")["status"] == "OK"
    assert _check(payload, "fastembed")["detail"] == "version=provider-only"
    assert _check(payload, "fastembed")["scope"] == "target_interpreter"
    assert str(environment) not in json.dumps(payload)
    if explicit_db:
        assert Path(payload["resolved_db"]).resolve() == selected_db.resolve()
        assert _check(payload, "working_total")["status"] == "1"
        assert not (tmp_path / "data" / "mnemosyne.db").exists()


def test_default_runtime_checks_are_labeled_as_cli_interpreter(tmp_path, monkeypatch):
    """Label checks without an explicit target as belonging to the CLI runtime."""
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))

    payload = diagnose.run_diagnostics()

    assert payload["runtime_scope"] == "cli_interpreter"
    assert payload["runtime_executable"] == Path(sys.executable).name
    assert _check(payload, "fastembed")["scope"] == "cli_interpreter"


def test_unavailable_target_does_not_fall_back_to_cli_interpreter(tmp_path):
    """Report only the target error when the requested executable is missing."""
    missing_python = tmp_path / "missing-python"

    payload = diagnose.collect_runtime_diagnostics(str(missing_python))

    assert payload["unavailable"] is True
    assert payload["scope"] == "target_interpreter"
    assert [entry["check"] for entry in payload["checks"]] == ["target_interpreter"]
    assert payload["checks"][0]["status"] == "ERROR"
    assert str(tmp_path) not in json.dumps(payload)


@pytest.mark.parametrize("stdout,returncode", [
    ('MNEMOSYNE_DIAGNOSTICS={"checks":[null]}\n', 0),
    ('MNEMOSYNE_DIAGNOSTICS={"checks":[{"check":"numpy","status":"OK"}]}\n', 0),
    ('MNEMOSYNE_DIAGNOSTICS={"checks":{}}\n', 0),
    ('MNEMOSYNE_DIAGNOSTICS=[]\n', 0),
    ('MNEMOSYNE_DIAGNOSTICS=invalid JSON\n', 0),
    ('no diagnostic marker\n', 0),
    ('MNEMOSYNE_DIAGNOSTICS={"checks":[]}\n', 1),
])
def test_target_payload_with_invalid_check_is_unavailable(monkeypatch, stdout, returncode):
    """Malformed or unsuccessful target output must never become a usable report."""
    monkeypatch.setattr(
        "mnemosyne.runtime_diagnostics.subprocess.run",
        lambda command, **kwargs: _completed(command, returncode, stdout),
    )

    result = diagnose.collect_runtime_diagnostics("./python")

    assert result["unavailable"] is True
    assert result["checks"][0]["check"] == "target_interpreter"


def test_target_runtime_replaces_invalid_child_output_bytes(monkeypatch):
    """Decode child output defensively so an error remains reportable."""
    captured = {}

    def run(command, **kwargs):
        captured.update(kwargs)
        return _completed(command, returncode=1, stderr="\ufffd")

    monkeypatch.setattr("mnemosyne.runtime_diagnostics.subprocess.run", run)

    result = diagnose.collect_runtime_diagnostics("./python")

    assert captured["errors"] == "replace"
    assert result["unavailable"] is True


def test_unavailable_target_is_unknown_and_never_auto_installs(tmp_path, monkeypatch):
    """Leave unknown target dependencies untouched instead of attempting pip."""
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    target_python = str(tmp_path / "missing-python")

    result = diagnose.run_diagnostics(python_executable=target_python)

    assert result["runtime_scope"] == "target_interpreter"
    assert any("dependency status is unknown" in finding for finding in result["key_findings"])
    assert not any("fastembed not available" in finding for finding in result["key_findings"])
    calls = []
    monkeypatch.setattr(
        diagnose.subprocess, "run",
        lambda command, **kwargs: calls.append(command) or _completed(command),
    )
    fix_result = diagnose.auto_fix(result["entries"], python_executable=target_python)
    assert fix_result["failed"] == [
        {"label": "target Python", "error": "diagnostics could not run in the selected interpreter"}
    ]
    assert calls == []


def test_auto_fix_uses_selected_python_module_pip(monkeypatch, tmp_path):
    """Install and verify a dependency using the same selected interpreter."""
    target_python = tmp_path / "provider env" / "python.exe"
    calls = []
    verified_pythons = []
    monkeypatch.setattr(
        diagnose.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or _completed(command),
    )
    monkeypatch.setattr(
        diagnose,
        "collect_runtime_diagnostics",
        lambda python=None: verified_pythons.append(python)
        or {"checks": [{"check": "fastembed", "status": "OK"}]},
    )

    result = diagnose.auto_fix(
        [{"check": "fastembed", "status": "MISSING"}],
        python_executable=str(target_python),
    )

    assert result["fixed"] == ["fastembed (embeddings engine)"]
    assert calls[0][0] == [
        str(target_python.resolve()),
        "-m",
        "pip",
        "install",
        "mnemosyne-memory[embeddings]",
    ]
    assert calls[0][1]["timeout"] == 600
    assert verified_pythons == [str(target_python.resolve())]


@pytest.mark.parametrize("selected", [False, True])
def test_auto_fix_does_not_claim_success_when_isolated_check_still_fails(
    monkeypatch, tmp_path, capsys, selected
):
    """A successful pip exit cannot substitute for a successful isolated import."""
    target_python = str(tmp_path / "provider env" / "python.exe") if selected else None
    calls = []
    verified_pythons = []
    monkeypatch.setattr(
        diagnose.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or _completed(command),
    )
    monkeypatch.setattr(
        diagnose,
        "collect_runtime_diagnostics",
        lambda python=None: verified_pythons.append(python)
        or {"checks": [{"check": "sqlite_vec", "status": "MISSING"}]},
    )

    result = diagnose.auto_fix(
        [{"check": "sqlite_vec", "status": "MISSING"}],
        python_executable=target_python,
    )

    expected_scope = (
        "the selected interpreter"
        if selected
        else "the CLI interpreter (pip on PATH may use a different environment)"
    )
    assert result["fixed"] == []
    assert result["failed"] == [
        {
            "label": "sqlite-vec (vector search)",
            "error": f"pip exited successfully, but the dependency was not verified in {expected_scope}",
        }
    ]
    if selected:
        assert calls[0][0] == [str(Path(target_python).resolve()), "-m", "pip", "install", "sqlite-vec"]
    else:
        assert calls[0][0] == ["pip", "install", "sqlite-vec"]
    assert verified_pythons == [str(Path(target_python).resolve()) if selected else None]
    output = capsys.readouterr().out
    assert "Could not verify sqlite-vec" in output
    assert expected_scope in output


@pytest.mark.parametrize("selected", [False, True])
def test_auto_fix_reports_pip_timeout_without_verifying_or_claiming_success(
    monkeypatch, tmp_path, capsys, selected
):
    """Report pip timeouts for both runtime scopes without claiming a repair."""
    target = str(tmp_path / "provider" / "python.exe") if selected else None
    calls = []
    verified_pythons = []

    def timeout(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(diagnose.subprocess, "run", timeout)
    monkeypatch.setattr(diagnose, "collect_runtime_diagnostics",
                        lambda python=None: verified_pythons.append(python))
    result = diagnose.auto_fix(
        [{"check": "sqlite_vec", "status": "MISSING"}], python_executable=target
    )
    assert result["fixed"] == []
    assert result["failed"] == [{
        "label": "sqlite-vec (vector search)",
        "error": "pip install timed out after 600 seconds",
    }]
    assert calls[0][1]["timeout"] == 600
    if selected:
        assert calls[0][0][:3] == [str(Path(target).resolve()), "-m", "pip"]
    else:
        assert calls[0][0] == ["pip", "install", "sqlite-vec"]
    assert verified_pythons == []
    assert "Timed out installing sqlite-vec" in capsys.readouterr().out


def test_target_python_bare_name_resolves_through_path(monkeypatch):
    """Resolve a bare executable name through PATH before launching checks."""
    resolved = "C:/venvs/provider/Scripts/python.exe"
    monkeypatch.setattr(
        "mnemosyne.runtime_diagnostics.shutil.which",
        lambda executable: resolved if executable == "python" else None,
    )
    commands = []
    monkeypatch.setattr(
        "mnemosyne.runtime_diagnostics.subprocess.run",
        lambda command, **kwargs: commands.append(command)
        or _completed(command, returncode=1),
    )

    diagnose.collect_runtime_diagnostics("python")

    assert commands[0][0] == resolved


def test_auto_fix_dry_run_uses_selected_python_basename(monkeypatch, tmp_path):
    """Describe a target-specific install without launching pip in dry-run mode."""
    target_python = tmp_path / "provider env" / "python.exe"
    calls = []
    monkeypatch.setattr(
        diagnose.subprocess, "run",
        lambda command, **kwargs: calls.append(command) or _completed(command),
    )

    result = diagnose.auto_fix(
        [{"check": "fastembed", "status": "MISSING"}],
        dry_run=True,
        python_executable=str(target_python),
    )

    assert result["fixed"] == [
        "WOULD install: fastembed (embeddings engine) (python.exe -m pip install mnemosyne-memory[embeddings])"
    ]
    assert calls == []


def test_cli_forwards_python_to_diagnostics_and_auto_fix(monkeypatch, capsys):
    """Use the CLI's explicit target for both diagnostics and requested repairs."""
    calls = []

    def run_diagnostics(**kwargs):
        calls.append(("diagnose", kwargs))
        return {
            "runtime_scope": "target_interpreter",
            "runtime_executable": "python.exe",
            "checks_passed": 1,
            "checks_total": 1,
            "entries": [],
            "key_findings": [],
        }

    def auto_fix(entries, **kwargs):
        calls.append(("fix", kwargs))
        return {"fixed": [], "failed": [], "skipped": [], "ran": True}

    monkeypatch.setattr(diagnose, "run_diagnostics", run_diagnostics)
    monkeypatch.setattr(diagnose, "auto_fix", auto_fix)

    cli.cmd_diagnose(["--python", "provider/python", "--fix"])

    assert calls == [
        (
            "diagnose",
            {"repair_vec_working": False, "dry_run": False, "python_executable": "provider/python"},
        ),
        ("fix", {"dry_run": False, "python_executable": "provider/python"}),
    ]
    assert "Runtime checks: target_interpreter (python.exe)" in capsys.readouterr().out


def test_cli_accepts_equals_form_for_python_option(monkeypatch):
    """Forward the target supplied with the --python=value spelling."""
    calls = []
    monkeypatch.setattr(diagnose, "run_diagnostics", lambda **kwargs: calls.append(kwargs) or {
        "runtime_scope": "target_interpreter",
        "runtime_executable": "python",
        "checks_passed": 0,
        "checks_total": 0,
        "entries": [],
        "key_findings": [],
    })

    cli.cmd_diagnose(["--python=provider/python"])

    assert calls[0]["python_executable"] == "provider/python"


@pytest.mark.parametrize("args", [["--python"], ["--python="], ["--python", "--fix"]])
def test_cli_rejects_missing_python_value_before_running_diagnostics(monkeypatch, capsys, args):
    """Invalid target options must fail before diagnosing or installing anything."""
    monkeypatch.setattr(diagnose, "run_diagnostics", lambda **kwargs: pytest.fail("diagnostics ran"))
    monkeypatch.setattr(diagnose, "auto_fix", lambda *args, **kwargs: pytest.fail("repair ran"))

    with pytest.raises(SystemExit) as exc:
        cli.cmd_diagnose(args)

    assert exc.value.code == 2
    assert "--python requires a value" in capsys.readouterr().err


@pytest.mark.parametrize("selected", [False, True])
def test_auto_fix_without_entries_diagnoses_the_requested_runtime(monkeypatch, tmp_path, selected):
    """Discover missing dependencies in the same runtime used for a dry-run repair."""
    target = str(tmp_path / "provider" / "python.exe") if selected else None
    calls = []

    def run_diagnostics(**kwargs):
        calls.append(kwargs)
        return {"entries": [{"check": "sqlite_vec", "status": "MISSING"}]}

    monkeypatch.setattr(diagnose, "run_diagnostics", run_diagnostics)
    monkeypatch.setattr(diagnose.subprocess, "run", lambda *args, **kwargs: pytest.fail("pip ran"))

    result = diagnose.auto_fix(dry_run=True, python_executable=target)

    assert calls == [{"python_executable": target} if selected else {}]
    assert result["failed"] == []
    assert len(result["fixed"]) == 1
    command = "python.exe -m pip install sqlite-vec" if selected else "pip install sqlite-vec"
    assert result["fixed"] == [f"WOULD install: sqlite-vec (vector search) ({command})"]


@pytest.mark.parametrize("failure", ["exit", "missing"])
def test_selected_python_install_errors_do_not_recheck_or_claim_success(
    monkeypatch, tmp_path, capsys, failure
):
    """Installer launch and execution failures must not claim a dependency repair."""
    target = str(tmp_path / "private-provider" / "python.exe")
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        if failure == "exit":
            raise subprocess.CalledProcessError(1, command, stderr="installation refused")
        raise FileNotFoundError(target)

    monkeypatch.setattr(diagnose.subprocess, "run", fail)
    monkeypatch.setattr(
        diagnose, "collect_runtime_diagnostics", lambda *args: pytest.fail("verification ran")
    )

    result = diagnose.auto_fix(
        [{"check": "sqlite_vec", "status": "MISSING"}], python_executable=target
    )

    assert calls == [[str(Path(target).resolve()), "-m", "pip", "install", "sqlite-vec"]]
    assert result["fixed"] == []
    error = "installation refused" if failure == "exit" else "selected Python not found"
    assert result["failed"] == [{"label": "sqlite-vec (vector search)", "error": error}]
    output = capsys.readouterr().out
    assert error in output
    assert str(tmp_path) not in output


def test_diagnose_module_forwards_target_and_never_repairs_an_unavailable_runtime(tmp_path, monkeypatch):
    """Exercise the argparse entry point in a real child with a missing target."""
    target = tmp_path / "missing-python"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_NO_EMBEDDINGS", "1")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")

    process = subprocess.run(
        [sys.executable, "-m", "mnemosyne.diagnose", "--python", str(target), "--fix"],
        check=True, capture_output=True, text=True, encoding="utf-8", timeout=120,
    )
    diagnostic_json, fix_json = process.stdout.split("--- Auto-fix ---", 1)
    report, repair = json.loads(diagnostic_json), json.loads(fix_json)

    assert report["runtime_scope"] == "target_interpreter"
    assert report["runtime_executable"] == target.name
    assert _check(report, "target_interpreter")["status"] == "ERROR"
    assert repair["fixed"] == []
    assert repair["failed"] == [{
        "label": "target Python", "error": "diagnostics could not run in the selected interpreter",
    }]


def test_default_auto_fix_finds_dependency_added_with_cached_directory(
    tmp_path, monkeypatch
):
    """A pip success must recheck even when the directory timestamp is unchanged."""
    import importlib
    import os

    from mnemosyne.core import embeddings

    # Use the real import finder and diagnostic collector, with no installed
    # sqlite_vec visible and no model downloads or real pip installation.
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.delitem(sys.modules, "sqlite_vec", raising=False)
    monkeypatch.delitem(sys.path_importer_cache, str(tmp_path), raising=False)
    monkeypatch.setattr(embeddings, "available", lambda: False)
    before = diagnose.collect_runtime_diagnostics()
    assert _check(before, "sqlite_vec")["status"] == "MISSING"
    original_stat = tmp_path.stat()

    def install(command, **kwargs):
        (tmp_path / "sqlite_vec.py").write_text(
            "__version__ = 'newly-installed'\n", encoding="utf-8"
        )
        # Pin the same-mtime race deterministically instead of relying on the
        # filesystem clock resolution or pip finishing within one clock tick.
        os.utime(tmp_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        assert importlib.util.find_spec("sqlite_vec") is None
        return _completed(command)

    monkeypatch.setattr(diagnose.subprocess, "run", install)
    try:
        result = diagnose.auto_fix([_check(before, "sqlite_vec")])
        assert result["failed"] == []
        assert result["fixed"] == ["sqlite-vec (vector search)"]
        assert sys.modules["sqlite_vec"].__version__ == "newly-installed"
    finally:
        sys.modules.pop("sqlite_vec", None)


def test_target_probe_is_isolated_and_bounded(monkeypatch, tmp_path):
    """Keep target checks isolated from CLI imports and bound the child lifetime."""
    calls = []
    target = tmp_path / "provider" / "python"

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(command, stdout='MNEMOSYNE_DIAGNOSTICS={"checks":[]}\n')

    monkeypatch.setattr("mnemosyne.runtime_diagnostics.subprocess.run", run)
    result = diagnose.collect_runtime_diagnostics(str(target))
    command, options = calls[0]
    assert len(calls) == 1
    assert command[:3] == [str(target.resolve()), "-I", "-c"]
    assert len(command) == 4
    assert options["timeout"] == 120
    assert result["scope"] == "target_interpreter"
    assert not result.get("unavailable")


def test_target_probe_timeout_reports_unavailable_without_fallback(monkeypatch, tmp_path):
    """An expired target probe reports only the target error, without path leaks."""
    calls = []
    target = tmp_path / "private-env" / "python"

    def timeout(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr("mnemosyne.runtime_diagnostics.subprocess.run", timeout)
    result = diagnose.collect_runtime_diagnostics(str(target))
    assert len(calls) == 1
    assert result["unavailable"] is True
    assert result["scope"] == "target_interpreter"
    assert result["executable"] == target.name
    assert result["checks"] == [{
        "category": "runtime", "check": "target_interpreter", "status": "ERROR",
        "detail": "selected interpreter could not run Mnemosyne diagnostics",
    }]
    assert str(tmp_path) not in json.dumps(result)
