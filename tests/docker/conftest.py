import os
import shlex
import subprocess

import pytest


def pytest_addoption(parser):
    parser.addoption('--image', default=None, help='Run Docker acceptance against this local image (requires Docker)')


@pytest.fixture(scope='session')
def image(request):
    value = request.config.getoption('--image')
    if not value:
        pytest.skip('Explicit Docker acceptance: pass --image IMAGE')
    subprocess.run(['docker', 'info'], check=True, capture_output=True)
    return value


@pytest.fixture(scope='session')
def compose_cmd():
    cmd = shlex.split(os.environ.get('MNEMOSYNE_TEST_COMPOSE', 'docker compose'))
    try:
        subprocess.run(cmd + ['version'], check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('Docker Compose unavailable; run container CI for rendered-config checks')
    return cmd
