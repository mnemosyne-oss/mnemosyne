"""The Hermes interpreter probe must accept a core install without sqlite-vec."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import venv

import pytest


@pytest.fixture
def isolated_core_python(tmp_path, monkeypatch):
    """Copy real Core into a clean interpreter with no optional native extras."""
    from mnemosyne_hermes import install
    import yaml

    environment = tmp_path / "runtime"
    venv.EnvBuilder(with_pip=False).create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    for name in ("PYTHONPATH", "PYTHONHOME", "PYTHONUSERBASE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    site = Path(subprocess.check_output(
        [str(python), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        text=True,
        timeout=10,
    ).strip())
    root = Path(__file__).resolve().parents[3]
    shutil.copytree(root / "mnemosyne", site / "mnemosyne",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(Path(yaml.__file__).parent, site / "yaml")
    absent = subprocess.check_output(
        [str(python), "-I", "-c",
         "import importlib.util; print(importlib.util.find_spec('sqlite_vec') is None)"],
        text=True,
        timeout=10,
    )
    assert absent.strip() == "True"
    return install, python, site


def test_core_check_accepts_real_isolated_core_without_sqlite_vec(isolated_core_python):
    """A usable Beam install remains accepted when its native extra is absent."""
    install, python, _ = isolated_core_python
    healthy = subprocess.run(
        [str(python), "-I", "-c",
         "import json, mnemosyne; from mnemosyne.core.beam import BeamMemory; "
         "print(json.dumps({'version': mnemosyne.__version__, 'beam': BeamMemory.__name__}))"],
        capture_output=True, text=True, check=True, timeout=10,
    )
    report = json.loads(healthy.stdout)
    assert report["beam"] == "BeamMemory"

    assert install.check_mnemosyne_core_for_hermes_python(python) == report["version"]


@pytest.mark.parametrize("failure", ["missing-core", "missing-beam", "broken-beam"])
def test_core_check_rejects_unusable_isolated_core(isolated_core_python, failure):
    """A version-only success must not hide a missing or broken Beam runtime."""
    install, python, site = isolated_core_python
    core = site / "mnemosyne"
    if failure == "missing-core":
        shutil.rmtree(core)
    elif failure == "missing-beam":
        (core / "core" / "beam.py").unlink()
    else:
        (core / "core" / "beam.py").write_text(
            "raise ImportError('controlled broken Beam runtime')\n", encoding="utf-8"
        )
    if failure != "missing-core":
        version = subprocess.run(
            [str(python), "-I", "-c", "import mnemosyne; print(mnemosyne.__version__)"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        assert version.stdout.strip()

    assert install.check_mnemosyne_core_for_hermes_python(python) is None


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
