"""Explicit memory_type and the dedupe opt-out on remember().

Two prerequisites for writing programmatically-generated memories:

1. A caller that knows what it is writing must be able to say so. The content
   classifier reads words, and generated text ("a screenshot of a terminal")
   does not describe what it actually is.

2. A caller writing generated text must be able to opt out of content dedup.
   The dedup key is (session_id, content); generated text collides where prose
   would not, and folding two rows into one silently misattributes anything
   that binds to the returned id.
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core import beam as beam_module
from mnemosyne.core.beam import BeamMemory, _clamp_memory_type
from mnemosyne.core.typed_memory import MemoryType


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


def _memory_type_of(db_path: Path, memory_id: str):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT memory_type FROM working_memory WHERE id = ?", (memory_id,)
        ).fetchone()
    finally:
        conn.close()
    assert row is not None, f"no working_memory row for {memory_id}"
    return row[0]


# --- _clamp_memory_type -----------------------------------------------------


def test_clamp_passes_through_none():
    assert _clamp_memory_type(None) is None


@pytest.mark.parametrize("label", [m.value for m in MemoryType])
def test_clamp_accepts_every_declared_type(label):
    assert _clamp_memory_type(label) == label


@pytest.mark.parametrize("given,expected", [("ARTIFACT", "artifact"), ("  Artifact  ", "artifact")])
def test_clamp_normalizes_case_and_whitespace(given, expected):
    assert _clamp_memory_type(given) == expected


def test_clamp_rejects_unknown_label_with_a_warning(caplog):
    with caplog.at_level("WARNING"):
        assert _clamp_memory_type("definitely-not-a-type") is None
    assert "unknown memory_type" in caplog.text


# --- explicit type beats the classifier ------------------------------------


def test_explicit_type_wins_over_classifier(temp_db, monkeypatch):
    """The classifier must not merely be overridden afterwards -- it must not
    run at all. Monkeypatching it to raise proves the short-circuit."""

    def explode(_content):
        raise AssertionError("classify_memory must not be called")

    monkeypatch.setattr(beam_module, "classify_memory", explode)

    beam = BeamMemory(session_id="mt-explicit", db_path=temp_db)
    # Content the classifier would confidently label a preference.
    mid = beam.remember("I prefer dark mode", memory_type="artifact")

    assert _memory_type_of(temp_db, mid) == "artifact"


def test_classifier_still_runs_when_no_type_is_given(temp_db, monkeypatch):
    monkeypatch.setattr(
        beam_module,
        "classify_memory",
        lambda content: type("R", (), {"memory_type": MemoryType.PREFERENCE})(),
    )
    beam = BeamMemory(session_id="mt-default", db_path=temp_db)
    mid = beam.remember("I prefer dark mode")
    assert _memory_type_of(temp_db, mid) == "preference"


def test_unknown_type_falls_back_to_the_classifier(temp_db, monkeypatch):
    """A typo should degrade to today's behaviour, not strip the type."""
    monkeypatch.setattr(
        beam_module,
        "classify_memory",
        lambda content: type("R", (), {"memory_type": MemoryType.PREFERENCE})(),
    )
    beam = BeamMemory(session_id="mt-typo", db_path=temp_db)
    mid = beam.remember("I prefer dark mode", memory_type="artifcat")
    assert _memory_type_of(temp_db, mid) == "preference"


def test_classifier_failure_is_still_non_blocking(temp_db, monkeypatch):
    def explode(_content):
        raise RuntimeError("classifier is broken")

    monkeypatch.setattr(beam_module, "classify_memory", explode)
    beam = BeamMemory(session_id="mt-broken", db_path=temp_db)
    mid = beam.remember("some content")  # must not raise
    assert _memory_type_of(temp_db, mid) is None


# --- dedupe -----------------------------------------------------------------


def test_dedupe_on_by_default_folds_identical_content(temp_db):
    beam = BeamMemory(session_id="dedupe-default", db_path=temp_db)
    first = beam.remember("a screenshot of a terminal window")
    second = beam.remember("a screenshot of a terminal window")
    assert first == second


def test_dedupe_false_writes_a_distinct_row(temp_db):
    """The regression this exists to prevent: two captions of the same text
    describing different sources must remain two rows, so anything binding to
    the returned id binds to the right one."""
    beam = BeamMemory(session_id="dedupe-off", db_path=temp_db)
    first = beam.remember("a black frame", dedupe=False)
    second = beam.remember("a black frame", dedupe=False)

    assert first != second

    conn = sqlite3.connect(str(temp_db))
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM working_memory WHERE content = ?", ("a black frame",)
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 2


def test_dedupe_false_avoids_retyping_a_colliding_row(temp_db):
    """With dedup on, an explicit type reaches the COALESCE in the update path
    and retypes whatever row it collided with. dedupe=False is what keeps a
    generated write from mutating a user's existing memory."""
    beam = BeamMemory(session_id="dedupe-retype", db_path=temp_db)
    original = beam.remember("meeting notes", memory_type="event")
    assert _memory_type_of(temp_db, original) == "event"

    beam.remember("meeting notes", memory_type="artifact", dedupe=False)

    # The original row is untouched.
    assert _memory_type_of(temp_db, original) == "event"


def test_dedupe_on_does_retype_via_coalesce(temp_db):
    """Documents the behaviour the flag exists to opt out of, so a future
    change to the COALESCE path fails loudly here."""
    beam = BeamMemory(session_id="dedupe-coalesce", db_path=temp_db)
    original = beam.remember("meeting notes", memory_type="event")
    again = beam.remember("meeting notes", memory_type="artifact")

    assert again == original
    assert _memory_type_of(temp_db, original) == "artifact"


# --- remember_batch parity --------------------------------------------------


def test_remember_batch_honors_per_item_memory_type(temp_db, monkeypatch):
    def explode(_content):
        raise AssertionError("classify_memory must not be called for typed items")

    monkeypatch.setattr(beam_module, "classify_memory", explode)

    beam = BeamMemory(session_id="batch-typed", db_path=temp_db)
    ids = beam.remember_batch(
        [
            {"content": "I prefer dark mode", "memory_type": "artifact"},
            {"content": "I prefer light mode", "memory_type": "artifact"},
        ]
    )

    assert len(ids) == 2
    for mid in ids:
        assert _memory_type_of(temp_db, mid) == "artifact"


def test_remember_batch_unknown_type_falls_back(temp_db, monkeypatch):
    monkeypatch.setattr(
        beam_module,
        "classify_memory",
        lambda content: type("R", (), {"memory_type": MemoryType.PREFERENCE})(),
    )
    beam = BeamMemory(session_id="batch-typo", db_path=temp_db)
    ids = beam.remember_batch([{"content": "x", "memory_type": "nope"}])
    assert _memory_type_of(temp_db, ids[0]) == "preference"


# --- facade parity ----------------------------------------------------------


def test_facade_passes_both_parameters_through(temp_db, monkeypatch):
    from mnemosyne.core.memory import Mnemosyne

    seen = {}
    m = Mnemosyne(db_path=temp_db, session_id="facade")

    real = m.beam.remember

    def spy(content, **kwargs):
        seen.update(kwargs)
        return real(content, **kwargs)

    monkeypatch.setattr(m.beam, "remember", spy)
    m.remember("a caption", memory_type="artifact", dedupe=False)

    assert seen["memory_type"] == "artifact"
    assert seen["dedupe"] is False


# --- positional-call compatibility -----------------------------------------


def test_existing_positional_callers_are_unaffected(temp_db):
    """The new parameters are appended, not inserted, and there is no `*`
    separator -- positional calls are common throughout the suite."""
    beam = BeamMemory(session_id="positional", db_path=temp_db)
    mid = beam.remember("content", "conversation", 0.7, {"k": "v"})
    assert mid
