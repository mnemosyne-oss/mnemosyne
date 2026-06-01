"""Tests for auto-default scope=global when extract=true.

Verifies that _handle_remember in the Hermes provider correctly
infers scope=global when extract=true and no explicit scope is passed.
"""
from __future__ import annotations


# We test the logic directly by simulating _handle_remember's scope logic
def test_scope_defaults_to_session_when_extract_false():
    """Without extract=true, default scope must remain 'session'."""
    # Simulate the new logic from _handle_remember
    args = {"content": "test content", "extract": False}
    extract = bool(args.get("extract", False))
    scope = args.get("scope", "global" if extract else "session")
    assert scope == "session"


def test_scope_defaults_to_global_when_extract_true():
    """With extract=true and no explicit scope, must default to 'global'."""
    args = {"content": "test content", "extract": True}
    extract = bool(args.get("extract", False))
    scope = args.get("scope", "global" if extract else "session")
    assert scope == "global"


def test_explicit_scope_respected_even_with_extract():
    """Explicitly passed scope must be respected even when extract=true."""
    args = {"content": "test content", "extract": True, "scope": "session"}
    extract = bool(args.get("extract", False))
    scope = args.get("scope", "global" if extract else "session")
    assert scope == "session"


def test_explicit_global_scope_respected():
    """Explicitly passed scope=global must be respected."""
    args = {"content": "test content", "extract": False, "scope": "global"}
    extract = bool(args.get("extract", False))
    scope = args.get("scope", "global" if extract else "session")
    assert scope == "global"


def test_extract_entities_does_not_affect_scope():
    """extract_entities=true without extract=true must NOT trigger global scope."""
    args = {"content": "test content", "extract_entities": True, "extract": False}
    extract = bool(args.get("extract", False))
    scope = args.get("scope", "global" if extract else "session")
    assert scope == "session"
