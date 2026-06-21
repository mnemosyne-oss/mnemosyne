"""
Re-export of all Mnemosyne tool schemas from the canonical source.

The single source of truth is ``mnemosyne.tool_schemas``.
This module re-exports everything so existing ``from mnemosyne_hermes.tools import ...``
statements continue to work without change.
"""

from mnemosyne.tool_schemas import ALL_TOOL_SCHEMAS

for _s in ALL_TOOL_SCHEMAS:
    _name = _s["name"].replace("mnemosyne_", "").upper() + "_SCHEMA"
    globals()[_name] = _s
