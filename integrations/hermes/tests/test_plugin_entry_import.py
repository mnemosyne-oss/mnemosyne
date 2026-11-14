"""Discovery stays lightweight and packaged registration preserves its behavior."""
from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
import subprocess
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "integrations/hermes/src"


def _run_child(code, *args):
    """Require a real child import to succeed, including its assertion failures."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, *map(str, args)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_plugins_group_target_does_not_import_mnemosyne():
    """Loading the real shim cannot mutate the parent test's module identities."""
    _run_child('''
import sys
sys.path.insert(0, sys.argv[1])
import _mnemosyne_hermes_plugin as plugin
assert callable(plugin.register)
assert not any(n == 'mnemosyne' or n.startswith(('mnemosyne.', 'mnemosyne_hermes')) for n in sys.modules)
''', SRC)


def _build_metadata_snapshot(project):
    """Record build metadata so the fixture cannot contaminate later installs."""
    roots = [project / "build", *list((project / "src").glob("*.egg-info"))]
    return {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in roots if root.exists() for p in root.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def packaged_plugin(tmp_path_factory):
    """Build and extract the owning wheel so entry metadata and code agree."""
    root = tmp_path_factory.mktemp("plugin-wheel")
    project = ROOT / "integrations/hermes"
    before = _build_metadata_snapshot(project)
    source = root / "source"
    shutil.copytree(project, source, ignore=shutil.ignore_patterns(
        "build", "dist", "*.egg-info", "__pycache__"))
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel",
         str(source), "--outdir", str(root / "dist")],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert _build_metadata_snapshot(project) == before, "wheel fixture changed checkout build metadata"
    wheel = next((root / "dist").glob("*.whl"))
    package = root / "package"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(package)
    return package


@pytest.mark.parametrize("platform", ["supported", "android", "emscripten", "legacy_android"])
def test_packaged_entry_delegates_or_refuses_without_activation(packaged_plugin, platform):
    """The actual wheel target registers real surfaces or refuses before imports."""
    _run_child('''
import importlib.metadata, pathlib, sys
package, core, platform = sys.argv[1:]
sys.path[:0] = [package, core]
dist = next(d for d in importlib.metadata.distributions(path=[package]) if d.metadata['Name'] == 'mnemosyne-hermes')
entry = next(e for e in dist.entry_points if e.group == 'hermes_agent.plugins' and e.name == 'mnemosyne')
register = entry.load()
assert pathlib.Path(sys.modules[register.__module__].__file__).resolve().is_relative_to(pathlib.Path(package).resolve())
assert not any(n == 'mnemosyne' or n.startswith(('mnemosyne.', 'mnemosyne_hermes')) for n in sys.modules)
class Context:
    def __init__(self): self.providers=[]; self.cli=[]; self.tools=[]
    def register_memory_provider(self, provider): self.providers.append(provider)
    def register_cli_command(self, **kw): self.cli.append(kw)
    def register_tool(self, **kw): self.tools.append(kw)
ctx=Context()
if platform != 'supported':
    sys.platform = 'linux' if platform == 'legacy_android' else platform
    if platform == 'legacy_android': sys.getandroidapilevel = lambda: 35
    try: register(ctx)
    except RuntimeError as exc: assert 'unsupported' in str(exc)
    else: raise AssertionError('unsupported platform was activated')
    assert not ctx.providers and not ctx.cli and not ctx.tools
    assert 'mnemosyne_hermes' not in sys.modules
else:
    register(ctx)
    assert len(ctx.providers) == 1
    assert [c['name'] for c in ctx.cli] == ['mnemosyne']
    assert callable(ctx.cli[0]['handler_fn'])
    tools={t['name'] for t in ctx.tools}
    assert {'mnemosyne_remember', 'mnemosyne_recall', 'mnemosyne_stats'} <= tools
    assert len(tools) == len(ctx.tools)
''', packaged_plugin, ROOT, platform)
