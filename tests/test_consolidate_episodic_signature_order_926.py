"""Regression: consolidate_to_episodic keeps its pre-existing positional slots.

CodeRabbit review finding on PR #926 (comment 4032670380): author_id/author_type
were inserted between `veracity` and `event_timestamp`, shifting the positional
slots of the pre-existing event_timestamp / event_date / event_date_precision /
emit_event parameters. The parameters were appended after `emit_event` instead.

The hazard is latent (no in-tree caller passes more than 2 positional args), so
this test pins the contract rather than reproducing a runtime failure: the first
ten parameter names must never shift again without a deliberate decision.
"""
from __future__ import annotations

import inspect

from mnemosyne.core.beam import BeamMemory

# The positional contract as it stood before #914 added author stamping.
FROZEN_PREFIX = [
    "self",
    "summary",
    "source_wm_ids",
    "source",
    "importance",
    "metadata",
    "valid_until",
    "scope",
    "veracity",
    "event_timestamp",
    "event_date",
    "event_date_precision",
    "emit_event",
]


def test_preexisting_positional_slots_are_unchanged():
    params = list(inspect.signature(BeamMemory.consolidate_to_episodic).parameters)
    assert params[: len(FROZEN_PREFIX)] == FROZEN_PREFIX, (
        "a pre-existing positional slot moved; append new parameters instead: "
        f"{params[: len(FROZEN_PREFIX)]!r}"
    )


def test_author_params_are_appended_after_emit_event():
    params = list(inspect.signature(BeamMemory.consolidate_to_episodic).parameters)
    assert params[-2:] == ["author_id", "author_type"], (
        f"author params must trail the signature, got tail {params[-3:]!r}"
    )
    assert params.index("emit_event") < params.index("author_id")


def test_author_params_are_keyword_safe_by_position():
    """Reaching author_id positionally requires 11 arguments, so any caller
    passing a pre-existing argument positionally cannot bind to an author."""
    params = list(inspect.signature(BeamMemory.consolidate_to_episodic).parameters)
    assert params.index("author_id") == 13, (
        "author_id moved earlier than the appended slot; positional callers "
        f"could mis-bind: index={params.index('author_id')}"
    )


# The same hazard applies to every signature the #914 work touched. Both
# remember() entry points must keep `dedupe` ahead of the author params:
# BeamMemory.remember already appended them, Mnemosyne.remember originally
# inserted them before `dedupe` (CodeRabbit comment 4040483068).
def _assert_dedupe_precedes_author(func, label):
    params = list(inspect.signature(func).parameters)
    assert "dedupe" in params, f"{label}: dedupe parameter disappeared"
    assert params.index("dedupe") < params.index("author_id"), (
        f"{label}: author_id precedes dedupe, so a positional dedupe argument "
        f"would bind to author_id instead: {params!r}"
    )
    assert params[-2:] == ["author_id", "author_type"], (
        f"{label}: author params must trail the signature, got {params[-3:]!r}"
    )


def test_mnemosyne_remember_keeps_dedupe_positional():
    from mnemosyne.core.memory import Mnemosyne

    _assert_dedupe_precedes_author(Mnemosyne.remember, "Mnemosyne.remember")


def test_beam_remember_keeps_dedupe_positional():
    _assert_dedupe_precedes_author(BeamMemory.remember, "BeamMemory.remember")
