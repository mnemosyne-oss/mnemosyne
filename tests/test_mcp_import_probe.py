"""Optional MCP discovery must not execute MCP or reject a loaded host stub."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('loaded', [False, True])
def test_importing_mnemosyne_does_not_execute_mcp(loaded):
    """Both discoverable MCP and an already-loaded spec-less module are accepted."""
    code = '''
import importlib.abc, importlib.util, sys, types
sys.path.insert(0, sys.argv[1])
executed=[]
class Loader(importlib.abc.Loader):
    def create_module(self, spec): return None
    def exec_module(self, module):
        executed.append(module.__name__)
        raise AssertionError('MCP executed during discovery')
class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'mcp': return importlib.util.spec_from_loader(fullname, Loader())
sys.meta_path.insert(0, Finder())
if sys.argv[2] == 'yes': sys.modules['mcp']=types.ModuleType('mcp')
import mnemosyne
assert 'run_mcp_server' in mnemosyne.__all__
assert not executed, executed
if sys.argv[2] == 'no': assert 'mcp' not in sys.modules
'''
    result = subprocess.run(
        [sys.executable, '-I', '-c', code, str(ROOT), 'yes' if loaded else 'no'],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
