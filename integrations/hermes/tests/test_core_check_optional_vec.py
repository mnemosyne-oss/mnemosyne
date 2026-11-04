"""The Hermes interpreter probe must accept a core install without sqlite-vec."""

from __future__ import annotations


def test_core_check_imports_beam_and_not_sqlite_vec(tmp_path, monkeypatch):
    from mnemosyne_hermes import install

    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd

        class Result:
            returncode = 0
            stdout = "4.0.0\n"
            stderr = ""

        return Result()

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    version = install.check_mnemosyne_core_for_hermes_python(tmp_path / "python")
    assert version == "4.0.0"
    script = seen["cmd"][-1]
    assert "import mnemosyne.core.beam" in script
    assert "sqlite_vec" not in script
