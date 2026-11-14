"""Hermes discovery and CLI attribute resolution stay lazy in fresh processes."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "integrations/hermes/src"


def _probe(code):
    """Treat any child import failure as a test failure, never a successful skip."""
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import sys; sys.path[:0]=sys.argv[1:];\n" + code,
         str(SRC), str(ROOT)], capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_importing_hermes_plugin_does_not_import_batch_tool():
    """An attempted heavy import is detected even if product code swallows it."""
    _probe('''
import importlib.abc
attempts=[]
class BlockHeavy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'mnemosyne.batch_tool','mnemosyne.core.beam','mnemosyne.core.embeddings','fastembed'}:
            attempts.append(fullname)
            raise ImportError('unexpected heavy import: '+fullname)
sys.meta_path.insert(0, BlockHeavy())
import mnemosyne_hermes
assert not attempts, attempts
assert not any(n in sys.modules for n in ('mnemosyne.batch_tool','mnemosyne.core.beam','mnemosyne.core.embeddings','fastembed'))
''')


def test_package_getattr_still_imports_cli_submodule():
    """The real CLI submodule remains accessible without eager batch loading."""
    _probe('''
import mnemosyne_hermes
module=mnemosyne_hermes.cli
assert module.__name__ == 'mnemosyne_hermes.cli'
assert callable(module.mnemosyne_command)
assert 'mnemosyne.batch_tool' not in sys.modules
''')
