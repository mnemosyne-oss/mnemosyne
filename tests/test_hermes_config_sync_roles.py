"""Typed sync-role parsing shared by both Hermes provider surfaces."""
from __future__ import annotations

import pytest

from mnemosyne.hermes_config import parse_sync_roles


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["user"], {"user"}),
        (("USER", "assistant"), {"user", "assistant"}),
        ("user,assistant", {"user", "assistant"}),
        ("['user']", {"user"}),
        ('["user", "assistant"]', {"user", "assistant"}),
        ([], set()),
        ("", set()),
    ],
)
def test_parse_sync_roles_supported_shapes(raw, expected):
    parsed, warnings = parse_sync_roles(raw, current={"user"})
    assert parsed == expected
    assert warnings == []


def test_nonempty_unknown_only_value_preserves_current_and_warns():
    parsed, warnings = parse_sync_roles("['users', 'tool']", current={"user"})
    assert parsed == {"user"}
    assert warnings
    assert "no valid" in warnings[-1]


def test_mixed_valid_and_unknown_value_applies_valid_subset_and_warns():
    parsed, warnings = parse_sync_roles(["user", "system"], current={"assistant"})
    assert parsed == {"user"}
    assert any("unknown" in warning for warning in warnings)


@pytest.mark.parametrize("raw", [True, False, 42, 3.14, {"user": True}, None])
def test_invalid_type_preserves_current(raw):
    parsed, warnings = parse_sync_roles(raw, current={"user"})
    assert parsed == {"user"}
    assert warnings
