"""Tests for Hermes remember default scope behavior.

The Hermes providers default to the configured default_scope unless callers
explicitly pass scope. extract=True does not implicitly promote to global.
"""
from __future__ import annotations


def _scope_for(args, default_scope="session"):
    return args.get("scope", default_scope)


def test_scope_defaults_to_session_when_extract_false():
    args = {"content": "test content", "extract": False}
    assert _scope_for(args) == "session"


def test_scope_defaults_to_default_scope_when_extract_true():
    args = {"content": "test content", "extract": True}
    assert _scope_for(args, default_scope="session") == "session"
    assert _scope_for(args, default_scope="global") == "global"


def test_explicit_scope_respected_even_with_extract():
    args = {"content": "test content", "extract": True, "scope": "session"}
    assert _scope_for(args, default_scope="global") == "session"


def test_explicit_global_scope_respected():
    args = {"content": "test content", "extract": False, "scope": "global"}
    assert _scope_for(args) == "global"


def test_extract_entities_does_not_affect_scope():
    args = {"content": "test content", "extract_entities": True, "extract": False}
    assert _scope_for(args) == "session"
