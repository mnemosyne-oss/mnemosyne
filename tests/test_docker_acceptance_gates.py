"""Explicit container acceptance must not turn missing checks into success."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def run_acceptance(tmp_path, source, *, explicit=True, missing_mcp=False):
    shutil.copy(ROOT / 'tests/docker/conftest.py', tmp_path / 'conftest.py')
    (tmp_path / 'test_probe.py').write_text(source)
    if missing_mcp:
        (tmp_path / 'mcp.py').write_text('raise ImportError("deliberately unavailable MCP")\n')
    env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONPATH=str(tmp_path),
               MNEMOSYNE_TEST_COMPOSE=f'{sys.executable} --not-a-real-option')
    return subprocess.run([sys.executable, '-m', 'pytest', '-q', '-ra',
                           *(['--image', 'explicit-acceptance'] if explicit else [])],
                          cwd=tmp_path, env=env, capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize('explicit', [False, True])
def test_unavailable_compose_skips_only_optional_run(tmp_path, explicit):
    result = run_acceptance(tmp_path, 'def test_compose(compose_cmd): pass\n', explicit=explicit)
    assert (result.returncode != 0) == explicit, result.stdout + result.stderr
    assert 'Docker Compose unavailable' in result.stdout + result.stderr


def test_explicit_acceptance_rejects_skipped_test(tmp_path):
    result = run_acceptance(tmp_path, 'import pytest\ndef test_required(): pytest.skip("lost coverage")\n')
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'lost coverage' in result.stdout


def test_explicit_acceptance_rejects_missing_mcp(tmp_path):
    result = run_acceptance(tmp_path, 'def test_required(): pass\n', missing_mcp=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert 'mcp' in result.stdout + result.stderr
