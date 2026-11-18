"""Runtime roles survive diagnostics without overriding explicit target probes."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mnemosyne import diagnose, runtime_diagnostics as runtime
from mnemosyne.doctor import RuntimeDiagnosticsAdapter, _sanitize_runtime_diagnostics


@pytest.mark.parametrize(
    "scope",
    ["calling_interpreter", "cli_interpreter", "provider_runtime", "mcp_server"],
)
def test_local_runtime_roles(scope):
    payload = runtime.collect_runtime_diagnostics(scope=scope)
    assert payload["scope"] == scope
    assert payload["executable"] == Path(sys.executable).name
    assert {check["scope"] for check in payload["checks"]} == {scope}
    assert (
        next(check for check in payload["checks"] if check["check"] == "checks_scope")[
            "detail"
        ]
        == scope
    )
    adapted = RuntimeDiagnosticsAdapter(scope=scope).inspect().metrics
    assert {check["scope"] for check in adapted["checks"]} == {scope}
    assert any(check["check"] == "checks_scope" for check in adapted["checks"])


def test_neutral_default_and_unknown_role():
    assert runtime.collect_runtime_diagnostics()["scope"] == "calling_interpreter"
    with pytest.raises(ValueError):
        runtime.collect_runtime_diagnostics(scope="/private/secret")
    with pytest.raises(ValueError):
        runtime.collect_runtime_diagnostics(scope="target_interpreter")


@pytest.mark.parametrize(
    "value", [None, "/private/secret", {"bad": "scope"}, ["provider_runtime"]]
)
def test_doctor_sanitizes_unknown_roles_and_keeps_refusal(value):
    payload = _sanitize_runtime_diagnostics(
        {
            "checks": [
                {
                    "category": "core",
                    "check": "embedding_policy_refusal",
                    "status": "NO",
                    "detail": "policy",
                    "scope": value,
                }
            ]
        }
    )
    assert payload["checks"] == [
        {
            "check": "embedding_policy_refusal",
            "status": "NO",
            "detail": "policy",
            "scope": "unscoped",
        }
    ]


def test_explicit_target_remains_target_even_for_provider_role(monkeypatch):
    checks = [
        {
            "category": "core",
            "check": name,
            "status": "NO",
            "detail": "offline",
            "scope": "calling_interpreter",
        }
        for name in ["embeddings_available", "sqlite_vec_available"]
    ]
    monkeypatch.setattr(
        runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="MNEMOSYNE_DIAGNOSTICS=" + json.dumps({"checks": checks}),
        ),
    )
    payload = runtime.collect_runtime_diagnostics(
        "/example/python", scope="provider_runtime"
    )
    assert payload["scope"] == "target_interpreter"
    assert {check["scope"] for check in payload["checks"]} == {"target_interpreter"}


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_provider_role_with_actual_diagnostics(tmp_path, monkeypatch, package):
    import importlib

    provider = importlib.import_module(package).MnemosyneMemoryProvider()
    provider._beam = None
    provider._profile_isolation_enabled = False
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    payload = json.loads(provider._handle_diagnose({}))
    assert payload["runtime_scope"] == "provider_runtime"
    assert {e["scope"] for e in payload["entries"] if "scope" in e} == {
        "provider_runtime"
    }
    logs = [
        json.loads(line) for line in Path(payload["log_path"]).read_text().splitlines()
    ]
    assert {e["scope"] for e in logs if "scope" in e} == {"provider_runtime"}


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_older_core_diagnostics_are_explicitly_unscoped(tmp_path, monkeypatch, package):
    import importlib

    provider = importlib.import_module(package).MnemosyneMemoryProvider()
    provider._beam = None
    provider._profile_isolation_enabled = False

    def old_diagnostics(*, repair_vec_working=False, dry_run=False, bank=None):
        return {
            "runtime_scope": "cli_interpreter",
            "entries": [
                {"check": "numpy", "scope": "cli_interpreter"},
                {"check": "checks_scope", "detail": "cli_interpreter", "status": "OK"},
            ],
        }

    monkeypatch.setattr(diagnose, "run_diagnostics", old_diagnostics)
    result = json.loads(provider._handle_diagnose({}))
    assert result["runtime_scope"] == "unscoped"
    assert result["entries"][0]["scope"] == "unscoped"
    assert result["entries"][1]["detail"] == "unscoped"
    assert result["runtime_log_scope"] == "legacy_unverified"


def test_doctor_preserves_known_scope_in_json_and_markdown():
    from mnemosyne.doctor import (
        DoctorReport,
        doctor_report_payload,
        render_doctor_json,
        render_doctor_markdown,
    )

    payload = doctor_report_payload(
        DoctorReport(
            bank_name="default",
            runtime_diagnostics=RuntimeDiagnosticsAdapter(scope="provider_runtime")
            .inspect()
            .metrics,
        )
    )
    assert {row["scope"] for row in payload["runtime_diagnostics"]["checks"]} == {
        "provider_runtime"
    }
    artifact = json.loads(render_doctor_json(payload))
    numpy_check = next(
        row for row in artifact["runtime_diagnostics"]["checks"]
        if row["check"] == "numpy"
    )
    assert numpy_check["scope"] == "provider_runtime"
    markdown_row = next(
        row for row in render_doctor_markdown(payload).splitlines()
        if row.startswith("- numpy: `")
    )
    assert '"scope":"provider_runtime"' in markdown_row


def test_mcp_reports_actual_server_role(tmp_path, monkeypatch):
    from mnemosyne import mcp_tools
    from mnemosyne.core.memory import Mnemosyne

    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
    mem = Mnemosyne(db_path=str(tmp_path / "mnemosyne.db"), session_id="mcp_default")
    monkeypatch.setattr(mcp_tools, "_create_instance", lambda **kwargs: mem)
    payload = mcp_tools._handle_diagnose({})
    assert payload["runtime_scope"] == "mcp_server"
    assert {e["scope"] for e in payload["entries"] if "scope" in e} == {"mcp_server"}
    assert sum(e["check"] == "checks_scope" for e in payload["entries"]) == 1


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
def test_scope_compatibility_does_not_retry_an_internal_type_error(
    monkeypatch, package
):
    import importlib

    provider = importlib.import_module(package).MnemosyneMemoryProvider()
    provider._beam = None
    provider._profile_isolation_enabled = False
    calls = []

    def failing(*, scope, **kwargs):
        calls.append(scope)
        raise TypeError("internal backend error")

    monkeypatch.setattr(diagnose, "run_diagnostics", failing)
    with pytest.raises(TypeError, match="internal backend error"):
        provider._handle_diagnose({})
    assert calls == ["provider_runtime"]


def test_unavailable_explicit_target_never_inherits_provider_scope(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError("private path")

    monkeypatch.setattr(runtime.subprocess, "run", missing)
    payload = runtime.collect_runtime_diagnostics(
        "/private/python", scope="provider_runtime"
    )
    assert payload["unavailable"] is True
    assert payload["scope"] == "target_interpreter"
    assert {c["scope"] for c in payload["checks"]} == {"target_interpreter"}
    assert "/private/" not in json.dumps(payload)
