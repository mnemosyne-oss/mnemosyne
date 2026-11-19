"""Verify the package's telemetry default in untouched interpreter state."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("configured", "expected"), [(None, "1"), ("0", "0")])
def test_package_import_sets_onnx_telemetry_default(configured, expected):
    """Supply an early default and keep a host's explicit opt-in."""
    env = os.environ.copy()
    env.pop("ORT_DISABLE_TELEMETRY", None)
    if configured is not None:
        env["ORT_DISABLE_TELEMETRY"] = configured

    result = subprocess.run(
        [
            sys.executable, "-I", "-c",
            "import os, sys; sys.path.insert(0, sys.argv[1]); "
            "import mnemosyne; print(os.environ.get('ORT_DISABLE_TELEMETRY'))",
            str(ROOT),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert result.stdout.strip() == expected
