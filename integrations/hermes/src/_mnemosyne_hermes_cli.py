"""Private, core-free bootstrap for the existing public installer command."""
from __future__ import annotations

from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path
import sys

_MODULE_NAME = "_mnemosyne_hermes_installer"


class _ReadOnlyLoader(SourceFileLoader):
    def set_data(self, path, data, **kwargs):
        """A read-only plan must not create bytecode beside the installer."""


def main(argv: list[str] | None = None) -> int:
    """Load adjacent installer logic without importing the provider package."""
    if sys.platform in {"android", "emscripten"} or (
        sys.platform == "linux" and hasattr(sys, "getandroidapilevel")
    ):
        _platform_label = "android" if sys.platform == "linux" else sys.platform
        print(
            f"Mnemosyne native Hermes integration is unsupported on {_platform_label}; "
            "no provider or tools were activated.",
            file=sys.stderr,
        )
        return 1
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        path = Path(__file__).resolve().parent / "mnemosyne_hermes" / "install.py"
        loader = _ReadOnlyLoader(_MODULE_NAME, str(path))
        spec = spec_from_loader(_MODULE_NAME, loader)
        if spec is None:
            raise ImportError("Cannot load the adjacent Mnemosyne installer")
        module = module_from_spec(spec)
        # Existing relative imports remain available in normal installer paths.
        module.__package__ = "mnemosyne_hermes"
        sys.modules[_MODULE_NAME] = module  # dataclasses resolves its module here.
        try:
            loader.exec_module(module)
        except BaseException:
            sys.modules.pop(_MODULE_NAME, None)
            raise
    return module.main(argv)
