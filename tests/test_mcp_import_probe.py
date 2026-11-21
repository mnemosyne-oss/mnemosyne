"""Optional MCP discovery stays lazy, including absent and spec-less hosts."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mode', ['discoverable', 'loaded', 'absent', 'resolve'])
def test_importing_mnemosyne_does_not_execute_mcp(mode):
    """Discovery is inert; resolving the export loads its real implementation."""
    code = '''
import importlib.abc, importlib.util, pathlib, sys, types
root,mode=sys.argv[1:]
if mode == 'absent':
    sys.path=[p for p in sys.path if not {'site-packages','dist-packages'}.intersection(pathlib.Path(p).parts)]
sys.path.insert(0, root)
executed=[]
class Loader(importlib.abc.Loader):
    def create_module(self, spec): return None
    def exec_module(self, module):
        executed.append(module.__name__)
        if mode != 'resolve': raise AssertionError('MCP executed during discovery')
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mcp' and mode != 'absent':
            return importlib.util.spec_from_loader(fullname, Loader(), is_package=True)
sys.meta_path.insert(0, Finder())
if mode == 'loaded': sys.modules['mcp']=types.ModuleType('mcp')
if mode == 'absent': assert importlib.util.find_spec('mcp') is None
import mnemosyne
assert not executed, executed
assert ('run_mcp_server' in mnemosyne.__all__) == (mode != 'absent')
assert 'mnemosyne.mcp_server' not in sys.modules
if mode != 'loaded': assert 'mcp' not in sys.modules
if mode == 'resolve':
    from mnemosyne import run_mcp_server
    assert callable(run_mcp_server)
    assert run_mcp_server.__module__ == 'mnemosyne.mcp_server'
    assert pathlib.Path(sys.modules['mnemosyne.mcp_server'].__file__).resolve() == pathlib.Path(root, 'mnemosyne/mcp_server.py').resolve()
    assert executed == ['mcp'], executed
'''
    result = subprocess.run(
        [sys.executable, '-I', '-c', code, str(ROOT), mode],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
