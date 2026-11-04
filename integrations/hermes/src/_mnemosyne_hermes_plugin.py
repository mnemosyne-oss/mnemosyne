"""Plugins-group entry point. Importing this module must not import Mnemosyne.

Hermes provider discovery loads every enabled ``hermes_agent.plugins`` target.
The memory provider stays registered there so PluginManager can still call
``register(ctx)``. The heavy package is imported only inside that call.
"""
from __future__ import annotations

import sys


def _unsupported_platform() -> str | None:
    if sys.platform in {"android", "emscripten"} or (
        sys.platform == "linux" and hasattr(sys, "getandroidapilevel")
    ):
        return "android" if sys.platform == "linux" else sys.platform
    return None


def register(ctx):
    """Register the provider, tools, and CLI command. Imports the package only now."""
    platform = _unsupported_platform()
    if platform is not None:
        raise RuntimeError(
            f"Mnemosyne native Hermes integration is unsupported on {platform}; "
            "no provider or tools were activated."
        )
    from mnemosyne_hermes import register as register_provider

    return register_provider(ctx)
