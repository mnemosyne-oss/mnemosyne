"""Use the same configuration boundary as the Core tests."""

from pathlib import Path
import runpy

_isolation = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "tests" / "_config_isolation.py")
)
pytest_configure = _isolation["pytest_configure"]
_isolated_mnemosyne_config = _isolation["_isolated_mnemosyne_config"]
