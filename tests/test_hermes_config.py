"""Regression coverage for Hermes config reads shared by both providers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

import mnemosyne.hermes_config as hermes_config


def _write_config(home, value: str) -> None:
    (home / "config.yaml").write_text(
        "memory:\n"
        "  mnemosyne:\n"
        f"    default_scope: {value}\n"
    )


@pytest.fixture(autouse=True)
def clear_config_cache():
    getattr(hermes_config, "_CONFIG_CACHE", {}).clear()
    yield
    getattr(hermes_config, "_CONFIG_CACHE", {}).clear()


def test_repeated_reads_parse_unchanged_config_once(tmp_path, monkeypatch):
    _write_config(tmp_path, "first")
    calls = 0
    original_load = yaml.load

    def counting_load(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(yaml, "load", counting_load)

    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
    assert calls == 1


def test_atomic_same_size_replacement_invalidates_cached_config(tmp_path):
    _write_config(tmp_path, "first")
    config_path = tmp_path / "config.yaml"
    original_size = config_path.stat().st_size

    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
    replacement = tmp_path / "replacement.yaml"
    replacement.write_text(
        "memory:\n"
        "  mnemosyne:\n"
        "    default_scope: other\n"
    )
    assert replacement.stat().st_size == original_size
    os.replace(replacement, config_path)

    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "other"


def test_replacement_during_parse_does_not_poison_cache(tmp_path, monkeypatch):
    _write_config(tmp_path, "first")
    config_path = tmp_path / "config.yaml"
    original_load = yaml.load

    def replace_after_load(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        replacement = tmp_path / "replacement.yaml"
        replacement.write_text(
            "memory:\n"
            "  mnemosyne:\n"
            "    default_scope: other\n"
        )
        os.replace(replacement, config_path)
        return loaded

    monkeypatch.setattr(yaml, "load", replace_after_load)
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "other"


def test_cached_config_fails_closed_after_malformed_or_deleted_file(tmp_path):
    _write_config(tmp_path, "first")
    config_path = tmp_path / "config.yaml"

    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
    config_path.write_text("memory: [\n")
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") is None

    _write_config(tmp_path, "first")
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
    config_path.unlink()
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") is None


def test_stat_open_aba_does_not_cache_replaced_file_content(tmp_path, monkeypatch):
    _write_config(tmp_path, "first")
    config_path = (tmp_path / "config.yaml").resolve()
    replacement = tmp_path / "replacement.yaml"
    _write_config(tmp_path, "first")
    replacement.write_text(
        "memory:\n"
        "  mnemosyne:\n"
        "    default_scope: other\n"
    )
    saved_original = tmp_path / "saved-original.yaml"
    original_open = Path.open
    original_load = yaml.load
    replaced = False

    def replace_before_open(path, *args, **kwargs):
        nonlocal replaced
        if path == config_path and not replaced:
            replaced = True
            os.replace(config_path, saved_original)
            os.replace(replacement, config_path)
        return original_open(path, *args, **kwargs)

    def restore_after_load(*args, **kwargs):
        loaded = original_load(*args, **kwargs)
        if saved_original.exists():
            os.replace(saved_original, config_path)
        return loaded

    monkeypatch.setattr(Path, "open", replace_before_open)
    monkeypatch.setattr(yaml, "load", restore_after_load)
    monkeypatch.setattr(hermes_config, "_config_fingerprint", lambda path: (1, 2, 3, 4, 5))

    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "other"
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"


def test_non_mapping_config_evicts_prior_cached_entry(tmp_path):
    _write_config(tmp_path, "first")
    config_path = (tmp_path / "config.yaml").resolve()
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
    assert config_path in hermes_config._CONFIG_CACHE

    config_path.write_text("- not\n- a mapping\n")
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") is None
    assert config_path not in hermes_config._CONFIG_CACHE


def test_resolve_error_fails_closed(tmp_path, monkeypatch):
    def fail_resolve(self, *args, **kwargs):
        raise OSError("symlink resolution failed")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") is None


def test_prefers_c_safe_loader_and_falls_back_to_safe_loader(tmp_path, monkeypatch):
    _write_config(tmp_path, "first")
    original_load = yaml.load
    loaders = []

    def recording_load(*args, **kwargs):
        loaders.append(kwargs.get("Loader", args[1] if len(args) > 1 else None))
        return original_load(*args, **kwargs)

    monkeypatch.setattr(yaml, "load", recording_load)

    if hasattr(yaml, "CSafeLoader"):
        assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
        assert loaders == [yaml.CSafeLoader]

        getattr(hermes_config, "_CONFIG_CACHE", {}).clear()
        loaders.clear()
        monkeypatch.delattr(yaml, "CSafeLoader")

    assert hermes_config.read_hermes_config_key(str(tmp_path), "default_scope") == "first"
    assert loaders == [yaml.SafeLoader]
