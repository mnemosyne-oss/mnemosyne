"""Shared Hermes Mnemosyne config helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_ConfigFingerprint = tuple[int, int, int, int, int]
_CONFIG_CACHE: dict[Path, tuple[_ConfigFingerprint, dict[str, Any]]] = {}


def _fingerprint_from_stat(stat: os.stat_result) -> _ConfigFingerprint:
    """Return a replacement-sensitive fingerprint from a stat result."""
    return (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def _config_fingerprint(config_path: Path) -> _ConfigFingerprint:
    """Return a replacement-sensitive fingerprint for a config file."""
    return _fingerprint_from_stat(config_path.stat())


def _load_hermes_config(config_path: Path, yaml: Any) -> dict[str, Any] | None:
    """Load a safe YAML mapping, reusing it only while its file is unchanged."""
    try:
        fingerprint = _config_fingerprint(config_path)
    except OSError:
        _CONFIG_CACHE.pop(config_path, None)
        return None

    cached = _CONFIG_CACHE.get(config_path)
    if cached is not None and cached[0] == fingerprint:
        return cached[1]

    try:
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        with config_path.open() as config_file:
            read_fingerprint = _fingerprint_from_stat(os.fstat(config_file.fileno()))
            config = yaml.load(config_file, Loader=loader)
    except Exception:
        _CONFIG_CACHE.pop(config_path, None)
        return None

    if config is None:
        config = {}
    if not isinstance(config, dict):
        _CONFIG_CACHE.pop(config_path, None)
        return {}

    try:
        parsed_fingerprint = _config_fingerprint(config_path)
    except OSError:
        _CONFIG_CACHE.pop(config_path, None)
        return None
    if parsed_fingerprint != read_fingerprint:
        _CONFIG_CACHE.pop(config_path, None)
        return config
    _CONFIG_CACHE[config_path] = (read_fingerprint, config)
    return config


def read_hermes_config_key(hermes_home: str | None, key: str) -> Any:
    """Read ``memory.mnemosyne.<key>`` from a Hermes ``config.yaml``.

    PyYAML is used when available; a tiny indentation-based fallback keeps the
    provider whitelist/default-scope path working in minimal Hermes plugin
    environments where PyYAML is not installed.
    """
    # Schema discovery happens before a provider receives initialize(...,
    # hermes_home=...). HERMES_HOME is the explicit active-profile authority in
    # that phase; do not fall back to ~/.hermes, which could cross profiles.
    resolved_home = hermes_home or os.environ.get("HERMES_HOME")
    if not resolved_home:
        return None
    try:
        config_path = Path(resolved_home, "config.yaml").resolve()
    except (OSError, RuntimeError):
        return None
    try:
        import yaml
    except ImportError:
        return read_config_key_without_yaml(str(config_path), key)

    config = _load_hermes_config(config_path, yaml)
    if config is None:
        return None
    memory = config.get("memory")
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
        if value.lower() == "null" or value == "~":
            return None
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
