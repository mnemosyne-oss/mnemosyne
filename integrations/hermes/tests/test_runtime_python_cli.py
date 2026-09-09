"""Tests for the read-only ``runtime-python`` installer command."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mnemosyne_hermes import install


def test_runtime_python_json_reports_discovered_interpreter(monkeypatch, capsys):
    selected = Path("/opt/hermes/venv/bin/python")
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: selected)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(install.os, "access", lambda path, mode: True)
    monkeypatch.setattr(
        install.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "3.12.13\n", ""),
    )

    assert install.main(["runtime-python", "--json"]) == 0

    assert json.loads(capsys.readouterr().out) == {
        "ok": True,
        "python": str(selected),
        "version": "3.12.13",
    }


def test_runtime_python_json_passes_explicit_python_unchanged(monkeypatch, capsys):
    selected = "/a path/with trailing space/python "
    received = []

    def find_python(*, explicit_python=None):
        received.append(explicit_python)
        return Path(selected)

    monkeypatch.setattr(install, "_find_hermes_python", find_python)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    monkeypatch.setattr(install.os, "access", lambda path, mode: True)
    monkeypatch.setattr(
        install.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "3.13.5\n", ""),
    )

    assert install.main(["runtime-python", "--json", "--python", selected]) == 0

    assert received == [selected]
    assert json.loads(capsys.readouterr().out)["python"] == selected


def test_runtime_python_json_fails_closed_when_discovery_fails(monkeypatch, capsys):
    monkeypatch.setattr(install, "_find_hermes_python", lambda **kwargs: None)

    assert install.main(["runtime-python", "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "Could not identify Hermes' Python" in payload["error"]


def test_runtime_python_json_fails_closed_for_empty_explicit_python(capsys):
    assert install.main(["runtime-python", "--json", "--python", " "]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "empty value" in payload["error"]
