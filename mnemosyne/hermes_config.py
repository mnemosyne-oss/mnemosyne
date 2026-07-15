"""Shared Hermes Mnemosyne config helpers."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any, Iterable, Set, Tuple, List


def parse_sync_roles(
    value: Any,
    *,
    current: Iterable[str],
    valid_roles: Iterable[str] = ("user", "assistant"),
) -> Tuple[Set[str], List[str]]:
    """Parse Hermes ``sync_roles`` without silently disabling capture.

    Native lists/tuples/sets and comma-separated strings are canonical.  JSON
    or Python-literal list strings are accepted for compatibility with older
    config writers that serialized ``['user']`` as a scalar.  An explicit
    empty list/string disables capture; invalid or unknown-only non-empty
    values preserve the currently effective roles and return warnings.
    """
    effective = {str(role).strip().lower() for role in current if str(role).strip()}
    allowed = {str(role).strip().lower() for role in valid_roles if str(role).strip()}
    warnings: List[str] = []

    raw_roles: Any
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raw_roles = []
        elif stripped.startswith("[") and stripped.endswith("]"):
            try:
                raw_roles = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                try:
                    raw_roles = ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    warnings.append(f"invalid serialized sync_roles value: {value!r}")
                    return effective, warnings
            if not isinstance(raw_roles, (list, tuple, set)):
                warnings.append(f"serialized sync_roles must be a list: {value!r}")
                return effective, warnings
        else:
            raw_roles = stripped.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_roles = value
    else:
        warnings.append(f"invalid sync_roles type {type(value).__name__}: {value!r}")
        return effective, warnings

    parsed = {str(role).strip().lower() for role in raw_roles if str(role).strip()}
    unknown = parsed - allowed
    if unknown:
        warnings.append(f"unknown sync_roles ignored: {sorted(unknown)!r}")
    valid = parsed & allowed
    if parsed and not valid:
        warnings.append(f"no valid sync_roles in {sorted(parsed)!r}; keeping {sorted(effective)!r}")
        return effective, warnings
    return valid, warnings


def parse_strict_bool(value: Any) -> Tuple[bool | None, str | None]:
    """Parse documented boolean forms without weakening active guardrails."""
    if isinstance(value, bool):
        return value, None
    if isinstance(value, int) and value in (0, 1):
        return bool(value), None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True, None
        if normalized in {"false", "0", "no", "off"}:
            return False, None
    return None, f"invalid boolean value {value!r}; keeping the current effective value"


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
