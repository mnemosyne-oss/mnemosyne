"""Regression: runtime checks must state which interpreter they describe.

`mnemosyne diagnose` imports fastembed/sqlite_vec inside the CLI process. When
the CLI is a pipx install and the Hermes provider runs from its own venv, the
checks report the *CLI* environment as healthy while the runtime that actually
serves recall lacks the vector stack (#813). The false green is silent: recall
degrades to FTS-only and nothing says so.

These tests pin the contract the maintainer specified for #813:

- every runtime check carries the role of the interpreter it ran in, so a
  CLI-local result is never mistaken for the provider runtime;
- the role comes from the caller: the provider's ``mnemosyne_diagnose`` is
  authoritative for the runtime serving recall and says so, the standalone CLI
  says ``cli_interpreter``, and an unstated caller gets a neutral label;
- the scope survives every surface: payload, ``run_diagnostics`` summary, the
  JSONL log, and the Doctor report boundary, which only admits known labels;
- no `HERMES_HOME`-style guessing is introduced to "fix" the mismatch.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from mnemosyne import diagnose
from mnemosyne.runtime_diagnostics import (
    CALLING_INTERPRETER_SCOPE,
    CLI_INTERPRETER_SCOPE,
    PROVIDER_RUNTIME_SCOPE,
    RUNTIME_SCOPES,
    collect_runtime_diagnostics,
)

_RUNTIME_CATEGORIES = {"env", "package", "deps", "core"}


def test_every_dep_check_declares_the_interpreter_it_ran_in():
    checks = collect_runtime_diagnostics(scope=CLI_INTERPRETER_SCOPE)["checks"]
    dep_checks = [c for c in checks if c["category"] == "deps"]
    assert dep_checks, "expected dependency checks in the payload"

    for c in dep_checks:
        scope = c.get("scope")
        assert scope == CLI_INTERPRETER_SCOPE, (
            f"dep check {c['check']!r} has scope={scope!r}; without it a green "
            "result reads as a statement about the runtime serving recall"
        )


@pytest.mark.parametrize("scope", sorted(RUNTIME_SCOPES))
def test_scope_is_the_callers_not_hard_coded(scope):
    """The provider path must not be labelled as a CLI result, or vice versa."""
    payload = collect_runtime_diagnostics(scope=scope)
    assert payload["scope"] == scope
    assert {c["scope"] for c in payload["checks"]} == {scope}
    scope_row = next(c for c in payload["checks"] if c["check"] == "checks_scope")
    assert scope_row["detail"] == scope


def test_unstated_caller_gets_a_neutral_scope():
    """A caller that does not name its role must not inherit the CLI's."""
    assert collect_runtime_diagnostics()["scope"] == CALLING_INTERPRETER_SCOPE


def test_unknown_scope_is_rejected():
    with pytest.raises(ValueError):
        collect_runtime_diagnostics(scope="hermes_runtime_guess")


def test_scope_survives_the_doctor_adapter():
    """The adapter rebuilds each check entry; scope must not be dropped there.

    test_doctor.py pins the adapter's exact output shape, so a scope added to the
    raw payload but dropped on the way to the report would give a green CLI and a
    silent gap in the machine-readable surface. That is the same false green
    #813 reports, one hop downstream.
    """
    from mnemosyne.doctor import RuntimeDiagnosticsAdapter

    metrics = RuntimeDiagnosticsAdapter(CLI_INTERPRETER_SCOPE).inspect().metrics
    checks = metrics["checks"]
    assert checks, "expected the adapter to expose runtime checks"
    for c in checks:
        assert c.get("scope") == CLI_INTERPRETER_SCOPE, c


def test_scope_row_reaches_the_report():
    """The producer's own scope row must survive the allowlist filter.

    `_sanitize_runtime_diagnostics` drops any check whose name is not in
    `_RUNTIME_CHECK_NAMES`. Adding the row to the producer is not enough on its
    own: without the name registered it is filtered out, and the report then
    carries a scope on every check without ever saying which interpreter those
    scopes describe -- which is the gap #813 is about.
    """
    from mnemosyne.doctor import RuntimeDiagnosticsAdapter, _RUNTIME_CHECK_NAMES

    assert "checks_scope" in _RUNTIME_CHECK_NAMES, sorted(_RUNTIME_CHECK_NAMES)

    metrics = RuntimeDiagnosticsAdapter(CLI_INTERPRETER_SCOPE).inspect().metrics
    rows = {c["check"]: c for c in metrics["checks"]}
    assert "checks_scope" in rows, sorted(rows)
    assert rows["checks_scope"]["status"] == "OK"
    assert rows["checks_scope"]["detail"] == CLI_INTERPRETER_SCOPE


def test_scope_is_machine_readable_not_only_prose():
    """The payload, not just the human print, must carry the scope."""
    payload = collect_runtime_diagnostics(scope=CLI_INTERPRETER_SCOPE)
    assert payload["scope"] == CLI_INTERPRETER_SCOPE
    assert "executable" in payload


def test_executable_stays_redacted():
    """The repo already redacts sys.executable to a basename; keep it that way."""
    payload = collect_runtime_diagnostics()
    assert "/" not in payload["executable"], payload["executable"]


def test_no_guessed_runtime_interpreter(monkeypatch):
    """A heuristic that scans HERMES_HOME would only move the misleading report.

    Asserted behaviourally: with a stale HERMES_HOME pointing somewhere else, the
    scope is still the one the caller stated and is not resolved from the env var.
    """
    monkeypatch.setenv("HERMES_HOME", "/nonexistent/hermes-home-that-is-not-this-env")
    payload = collect_runtime_diagnostics(scope=CLI_INTERPRETER_SCOPE)
    assert payload["scope"] == CLI_INTERPRETER_SCOPE
    assert payload["executable"] == Path(sys.executable).name
    assert "nonexistent" not in json.dumps(payload)


def _isolate_diagnose(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnose, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))


def test_run_diagnostics_keeps_scope_in_summary_and_log(tmp_path, monkeypatch):
    """`run_diagnostics` rebuilds each entry; the scope must survive that too.

    This is the path behind the provider's and the legacy ``mnemosyne_diagnose``
    responses and the JSONL log. Rebuilding entries from category/check/status/
    detail alone drops the scope, leaving those surfaces as unqualified as
    before #813.
    """
    _isolate_diagnose(tmp_path, monkeypatch)

    summary = diagnose.run_diagnostics(scope=PROVIDER_RUNTIME_SCOPE)

    assert summary["scope"] == PROVIDER_RUNTIME_SCOPE
    assert summary["executable"] == Path(sys.executable).name
    runtime_entries = [e for e in summary["entries"] if "scope" in e]
    assert {e["check"] for e in runtime_entries} >= {"checks_scope", "fastembed", "sqlite_vec"}
    assert {e["scope"] for e in runtime_entries} == {PROVIDER_RUNTIME_SCOPE}

    logged = [
        json.loads(line)
        for line in Path(summary["log_path"]).read_text(encoding="utf-8").splitlines()
    ]
    logged_runtime = [e for e in logged if e["category"] in _RUNTIME_CATEGORIES and "scope" in e]
    assert {e["check"] for e in logged_runtime} >= {"checks_scope", "fastembed"}
    assert {e["scope"] for e in logged_runtime} == {PROVIDER_RUNTIME_SCOPE}


def test_cli_diagnose_labels_its_checks_as_cli(tmp_path, monkeypatch):
    _isolate_diagnose(tmp_path, monkeypatch)
    from mnemosyne import cli

    seen = []
    real_run_diagnostics = diagnose.run_diagnostics

    def spy(**kwargs):
        summary = real_run_diagnostics(**kwargs)
        seen.append(summary)
        return summary

    monkeypatch.setattr(diagnose, "run_diagnostics", spy)
    cli.cmd_diagnose([])

    assert seen and seen[0]["scope"] == CLI_INTERPRETER_SCOPE


def test_provider_mnemosyne_diagnose_reports_provider_runtime(tmp_path, monkeypatch):
    """End to end through the provider tool, without stubbing run_diagnostics.

    The provider runs these checks in the process serving recall, so its result
    is authoritative for that runtime and must say so rather than claim to be a
    CLI-side view (#813).
    """
    _isolate_diagnose(tmp_path, monkeypatch)
    from hermes_memory_provider import MnemosyneMemoryProvider

    provider = MnemosyneMemoryProvider()
    provider._beam = None
    provider._profile_isolation_enabled = False

    result = json.loads(provider._handle_diagnose({}))

    assert result["scope"] == PROVIDER_RUNTIME_SCOPE
    scope_row = next(e for e in result["entries"] if e["check"] == "checks_scope")
    assert scope_row["detail"] == PROVIDER_RUNTIME_SCOPE
    assert {e["scope"] for e in result["entries"] if "scope" in e} == {PROVIDER_RUNTIME_SCOPE}


def test_mcp_diagnose_reports_mcp_server(tmp_path, monkeypatch):
    _isolate_diagnose(tmp_path, monkeypatch)
    from mnemosyne import mcp_tools

    calls = []

    def fake_run_diagnostics(**kwargs):
        calls.append(kwargs)
        return {"checks_total": 0, "key_findings": [], "entries": []}

    monkeypatch.setattr(diagnose, "run_diagnostics", fake_run_diagnostics)
    monkeypatch.setattr(mcp_tools, "_create_instance", lambda **_kwargs: object())

    mcp_tools._handle_diagnose({})

    assert calls and calls[0]["scope"] == "mcp_server"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))