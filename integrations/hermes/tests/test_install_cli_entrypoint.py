"""Regression tests for the mnemosyne-hermes console entry point."""

from __future__ import annotations

import importlib.abc
import importlib.metadata
import hashlib
import json
from pathlib import Path
import shutil
import runpy
import subprocess
import sys
import venv
import zipfile

import pytest

import mnemosyne_hermes
from mnemosyne_hermes import install


class _BlockMnemosyneCore(importlib.abc.MetaPathFinder):
    """Simulate an environment where mnemosyne.core is unavailable."""

    def find_spec(self, fullname, path=None, target=None):  # noqa: D401
        if fullname == "mnemosyne.core" or fullname.startswith("mnemosyne.core."):
            raise ModuleNotFoundError("No module named 'mnemosyne.core'")
        return None


def test_main_help_exits_successfully(capsys):
    try:
        install.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "mnemosyne-hermes" in out
    assert "install" in out
    assert "status" in out


def test_adopt_pm_console_plan_does_not_import_product_modules(tmp_path):
    """Build/install the owning source wheel; never audit an editable shadow."""
    _adoption_fixture = runpy.run_path(str(Path(__file__).with_name("test_install_hermes.py")))["_adoption_fixture"]

    source = tmp_path / "source"
    shutil.copytree(Path(__file__).resolve().parents[1], source,
                    ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"))
    built = subprocess.run([sys.executable, "-m", "build", "--wheel",
        str(source), "--outdir", str(tmp_path / "dist")], capture_output=True, text=True)
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next((tmp_path / "dist").glob("*.whl"))
    venv.EnvBuilder(with_pip=True).create(tmp_path / "consumer")
    python = tmp_path / "consumer" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    installed = subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(wheel)],
                               capture_output=True, text=True)
    assert installed.returncode == 0, installed.stderr
    home, prior, site, target = _adoption_fixture(tmp_path / "legacy")
    trap = tmp_path / "activated-pth"
    (site / "trap.pth").write_text(f"import pathlib; pathlib.Path({str(trap)!r}).touch()\n")
    arguments = ["--hermes-home", str(home), "install", "--adopt-pm", "--dry-run", "--python", str(prior),
                 "--core-extras", "embeddings", "--provider-extras", "none"]
    with zipfile.ZipFile(wheel) as package:
        loader_hash = hashlib.sha256(package.read("_mnemosyne_hermes_cli.py")).hexdigest() if "_mnemosyne_hermes_cli.py" in package.namelist() else "ABSENT_AT_BASE"
    code = r'''
import sys
import importlib.metadata
import pathlib,hashlib,json
seen = []
def audit(event, args):
    if event == "import" and (args[0] in {"mnemosyne", "mnemosyne_hermes"} or args[0].startswith(("mnemosyne.", "mnemosyne_hermes."))):
        seen.append(args[0])
sys.addaudithook(audit)
dist = importlib.metadata.distribution("mnemosyne-hermes")
assert not any(p.name.endswith('.pth') for p in dist.files)
entry = next(e for e in dist.entry_points
             if e.group == "console_scripts" and e.name == "mnemosyne-hermes")
main = entry.load()
assert entry.value == '_mnemosyne_hermes_cli:main', entry.value
origin = pathlib.Path(sys.modules['_mnemosyne_hermes_cli'].__file__).resolve()
assert origin == pathlib.Path(dist.locate_file('_mnemosyne_hermes_cli.py')).resolve()
assert origin.is_relative_to(pathlib.Path(sys.prefix).resolve())
assert hashlib.sha256(origin.read_bytes()).hexdigest() == sys.argv[2]
def snapshot():
    return {str(p): ('link:'+str(p.readlink()) if p.is_symlink() else hashlib.sha256(p.read_bytes()).hexdigest())
            for p in pathlib.Path(sys.argv[3]).rglob('*') if p.is_symlink() or p.is_file()}
before = snapshot()
assert main(json.loads(sys.argv[1])) == 0
assert main(["install", "--adopt-pm", "--dry-run"]) == 1
assert before == snapshot()
assert not pathlib.Path(sys.argv[4]).exists()
assert not seen, "Product imports before metadata-only plan: " + repr(seen)
assert "mnemosyne_hermes" not in sys.modules
assert "mnemosyne" not in sys.modules
'''
    result = subprocess.run([str(python), "-I", "-B", "-c", code, json.dumps(arguments), loader_hash,
        str(home), str(trap)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture(scope="module")
def wheel_consumer(tmp_path_factory):
    """Use an installed wheel, not the editable checkout, for console probes."""
    root = tmp_path_factory.mktemp("platform-wheel-consumer")
    source = root / "source"
    shutil.copytree(Path(__file__).resolve().parents[1], source,
                    ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"))
    built = subprocess.run([sys.executable, "-m", "build", "--wheel",
        str(source), "--outdir", str(root / "dist")], capture_output=True, text=True,
        timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next((root / "dist").glob("*.whl"))
    venv.EnvBuilder(with_pip=True).create(root / "consumer")
    python = root / "consumer" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    installed = subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "--no-index", str(wheel)],
                               capture_output=True, text=True, timeout=120)
    assert installed.returncode == 0, installed.stderr
    return python


@pytest.mark.parametrize(
    ("platform", "legacy_android"),
    [
        pytest.param("android", False, id="android"),
        pytest.param("emscripten", False, id="emscripten"),
        pytest.param("linux", True, id="legacy_android"),
        pytest.param("linux", False, id="ordinary_linux"),
    ],
)
def test_installed_wheel_console_checks_android_before_cached_installer(
    tmp_path, wheel_consumer, platform, legacy_android,
):
    code = f'''
import importlib.metadata, importlib.util, pathlib, sys, types
assert importlib.util.find_spec("mnemosyne") is None
entry = next(e for e in importlib.metadata.distribution("mnemosyne-hermes").entry_points
             if e.group == "console_scripts" and e.name == "mnemosyne-hermes")
main = entry.load()
assert pathlib.Path(sys.modules[main.__module__].__file__).resolve().is_relative_to(
    pathlib.Path(sys.prefix).resolve())
calls = []
cached = types.ModuleType("_mnemosyne_hermes_installer")
def cached_main(argv):
    calls.append(argv)
    return 0
cached.main = cached_main
sys.modules[cached.__name__] = cached
sys.platform = {platform!r}
if {legacy_android!r}:
    def android_api_level():
        raise AssertionError("platform detection must not call Android API")
    sys.getandroidapilevel = android_api_level
else:
    assert not hasattr(sys, "getandroidapilevel")
status = main(["install"])
if {legacy_android!r} or {platform!r} in ("android", "emscripten"):
    assert status != 0, status
    assert calls == [], calls
else:
    assert status == 0, status
    assert calls == [["install"]], calls
'''
    result = subprocess.run([str(wheel_consumer), "-I", "-B", "-c", code],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    if legacy_android or platform in ("android", "emscripten"):
        expected_platform = "android" if legacy_android else platform
        assert result.stderr == (
            f"Mnemosyne native Hermes integration is unsupported on {expected_platform}; "
            "no provider or tools were activated.\n"
        )
    else:
        assert result.stderr == ""


def test_package_version_matches_distribution_metadata():
    dist_version = importlib.metadata.version("mnemosyne-hermes")

    assert mnemosyne_hermes.__version__ == dist_version, (
        "mnemosyne_hermes.__version__ must stay in sync with package metadata"
    )


def test_package_and_install_module_import_without_mnemosyne_core():
    code = r'''
import importlib.abc
import sys
class BlockMnemosyneCore(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mnemosyne.core" or fullname.startswith("mnemosyne.core."):
            raise ModuleNotFoundError("No module named 'mnemosyne.core'")
        return None
sys.meta_path.insert(0, BlockMnemosyneCore())
import mnemosyne_hermes
from mnemosyne_hermes import install
assert callable(install.main)
print("ok")
'''
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
