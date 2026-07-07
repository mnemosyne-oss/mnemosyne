"""Shared Hermes Mnemosyne config helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def read_hermes_config_key(hermes_home: str | None, key: str) -> Any:
    """Read ``memory.mnemosyne.<key>`` from a Hermes ``config.yaml``.

    PyYAML is used when available; a tiny indentation-based fallback keeps the
    provider whitelist/default-scope path working in minimal Hermes plugin
    environments where PyYAML is not installed.
    """
    config_path = os.path.join(hermes_home, "config.yaml") if hermes_home else ""
    if not config_path or not os.path.exists(config_path):
        return None
    try:
        import yaml
    except ImportError:
        return read_config_key_without_yaml(config_path, key)

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except Exception:
        return None
    memory = config.get("memory") if isinstance(config, dict) else None
    memory = memory if isinstance(memory, dict) else {}
    mnemosyne = memory.get("mnemosyne")
    mnemosyne = mnemosyne if isinstance(mnemosyne, dict) else {}
    return mnemosyne.get(key)


def read_config_key_without_yaml(config_path: str, key: str) -> Any:
    """Tiny fallback parser for ``memory.mnemosyne.<key>`` values."""
    try:
        lines = Path(config_path).read_text().splitlines()
    except OSError:
        return None

    in_memory = False
    in_mnemosyne = False
    memory_indent = mnemosyne_indent = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if stripped == "memory:":
            in_memory = True
            in_mnemosyne = False
            memory_indent = indent
            continue
        if in_memory and indent <= memory_indent:
            in_memory = False
            in_mnemosyne = False
        if in_memory and stripped == "mnemosyne:" and indent > memory_indent:
            in_mnemosyne = True
            mnemosyne_indent = indent
            continue
        if in_mnemosyne and indent <= mnemosyne_indent:
            in_mnemosyne = False
        if not in_mnemosyne or indent <= mnemosyne_indent or not stripped.startswith(f"{key}:"):
            continue
        value = stripped.split(":", 1)[1].strip()
        if value == "[]":
            return []
        if value:
            return value.strip('"\'')
        items = []
        for child in lines[i + 1:]:
            child_stripped = child.strip()
            if not child_stripped:
                continue
            child_indent = len(child) - len(child.lstrip())
            if child_indent <= indent:
                break
            if child_stripped.startswith("-"):
                items.append(child_stripped[1:].strip().strip('"\''))
                continue
            break
        return items if items else None
    return None
