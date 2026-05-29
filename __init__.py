"""
Mnemosyne Plugin for Hermes Agent.

This repo installs as a normal Hermes plugin (tools + hooks) and also exposes a
lightweight MemoryProvider shim so Hermes can treat Mnemosyne as the active
external memory provider.
"""

import sys
from pathlib import Path

# Ensure this directory is on path so `hermes_plugin` is discoverable
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Re-export __version__ / __author__ from the inner mnemosyne subpackage so
# `from mnemosyne import __version__` works in either install layout:
#   - Hermes plugin tree: outer `mnemosyne/` is the resolved package, inner
#     `mnemosyne/mnemosyne/` is the subpackage `mnemosyne.mnemosyne`.
#   - pip / repo-direct install: inner `mnemosyne/` is the resolved package
#     directly and this stub is never loaded.
# Without this re-export, `hermes mnemosyne version` (and any other caller
# doing `from mnemosyne import __version__`) crashed with ImportError under
# the Hermes plugin layout. See issue #53.
try:
    from .mnemosyne import __version__, __author__
except ImportError:
    __version__ = "unknown"
    __author__ = "Abdias J"

try:
    from hermes_plugin import register as _register_plugin
except ImportError:
    _register_plugin = None

try:
    from .provider import MnemosyneMemoryProvider
except ImportError:
    MnemosyneMemoryProvider = None


def register(ctx):
    """Register Mnemosyne tools/hooks and, when supported, a memory provider."""
    if _register_plugin is None:
        raise ImportError("hermes_plugin is required to register Mnemosyne")
    result = _register_plugin(ctx)
    if MnemosyneMemoryProvider is not None and hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(MnemosyneMemoryProvider())
    return result


__all__ = ["__version__", "__author__"]
if _register_plugin is not None:
    __all__.append("register")
if MnemosyneMemoryProvider is not None:
    __all__.append("MnemosyneMemoryProvider")
