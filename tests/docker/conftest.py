import importlib
import os
import shlex
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption('--image', default=None, help='Run Docker acceptance against this local image (requires Docker)')


def pytest_sessionstart(session):
    if session.config.getoption('--image'):
        for name in ('httpx', 'httpx2', 'mcp'):
            try:
                importlib.import_module(name)
            except ImportError as exc:
                raise pytest.UsageError(f'Explicit Docker acceptance requires {name}: {exc}') from exc


def pytest_sessionfinish(session, exitstatus):
    reporter = session.config.pluginmanager.getplugin('terminalreporter')
    if session.config.getoption('--image') and reporter and reporter.stats.get('skipped'):
        reporter.write_sep('!', 'Explicit Docker acceptance cannot pass with skipped tests')
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(scope='session')
def image(request):
    value = request.config.getoption('--image')
    if not value:
        pytest.skip('Explicit Docker acceptance: pass --image IMAGE')
    subprocess.run(['docker', 'info'], check=True, capture_output=True)
    return value


@pytest.fixture(scope='session')
def compose_cmd(request):
    cmd = shlex.split(os.environ.get('MNEMOSYNE_TEST_COMPOSE', 'docker compose'))
    try:
        subprocess.run(cmd + ['version'], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        if request.config.getoption('--image'):
            pytest.fail(f'Docker Compose unavailable during explicit acceptance: {exc}')
        pytest.skip('Docker Compose unavailable; run container CI for rendered-config checks')
    return cmd
