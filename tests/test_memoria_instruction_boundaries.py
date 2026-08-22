"""Word-boundary regression tests for the MEMORIA instruction extractor (#507).

The instruction pattern was unanchored, so ``never`` matched inside ``whenever``
and the extractor stored the opposite of what the user said. On a production
bank, "Good - whenever needed we can use it." was recorded as an instruction
"never needed we can use it".

Test design note: the pattern-level cases below rebuild the runtime regex the
same way ``extract_and_store_facts`` does. That mirrors beam.py's assembly, so
it is kept honest by ``test_extractor_end_to_end_does_not_invert_whenever``,
which drives the real extraction path and asserts on the stored row.

Credit: diagnosis and the original fix approach are from #508 (@Sanjays2402,
account since deleted) and #549 (@Souptik96).
"""

import re
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory

# Locales whose instruction pattern must be boundary-anchored.
LOCALES = ["en", "de", "ru", "it", "es"]


def _runtime_instruction_re(locale: str) -> str:
    """Assemble the instruction regex exactly as beam.py:4845 does."""
    pat = BeamMemory.MULTILINGUAL_PATTERNS[locale]
    return pat["instruction"].replace("IMPVERBS", pat["instruction_imperative"])


@pytest.mark.parametrize("locale", LOCALES)
def test_every_locale_instruction_pattern_is_boundary_anchored(locale):
    """Guard: no locale may reintroduce an unanchored instruction pattern."""
    pattern = BeamMemory.MULTILINGUAL_PATTERNS[locale]["instruction"]
    assert pattern.startswith(r"\b"), (
        f"{locale} instruction pattern is not word-boundary anchored; "
        f"'never' will match inside 'whenever'"
    )


@pytest.mark.parametrize(
    "content",
    [
        "Good - whenever needed we can use it. I've disabled it.",
        "Use the cache whenever possible for the hot path lookups.",
        "Whenever the config changes we reload the whole provider stack.",
    ],
)
def test_en_whenever_does_not_yield_never_instruction(content):
    """'whenever' must not produce a 'never ...' instruction (the #507 report)."""
    matches = re.findall(_runtime_instruction_re("en"), content, re.IGNORECASE)
    assert matches == [], f"inverted instruction extracted from {content!r}: {matches}"


@pytest.mark.parametrize("locale", LOCALES)
def test_no_locale_pattern_contains_a_literal_backslash_escape(locale):
    """Guard (#560): a doubled escape makes the pattern match a literal backslash.

    The ``ru`` instruction pattern was written with ``\\\\s+`` / ``[^.,;!?\\\\n]``
    inside a raw string, so it required a literal backslash in the text and
    Russian extraction was silently dead. Several ``es`` patterns had the same
    doubled ``\\\\n``, which truncated captures at the first letter ``n``.
    """
    for key, value in BeamMemory.MULTILINGUAL_PATTERNS[locale].items():
        if not isinstance(value, str):
            continue
        assert not re.search(r"\\\\[sndbwSNDBW]", value), (
            f"{locale}/{key} contains a literal backslash escape; "
            f"the pattern cannot match ordinary text"
        )


def test_ru_instruction_extracts_a_russian_directive():
    """#560: Russian instruction extraction must actually match."""
    matches = re.findall(
        _runtime_instruction_re("ru"),
        "всегда запускай тесты перед пушем в main",
        re.IGNORECASE,
    )
    assert matches, "ru instruction pattern matched nothing"
    assert any("запускай тесты" in m for m in matches), matches


def test_es_instruction_capture_is_not_truncated_at_the_letter_n():
    """#560: the doubled ``\\\\n`` in the es character class ended captures early."""
    matches = re.findall(
        _runtime_instruction_re("es"),
        "siempre ejecuta las pruebas antes de subir a main",
        re.IGNORECASE,
    )
    assert matches, "es instruction pattern matched nothing"
    assert any("antes de subir" in m for m in matches), matches


def test_ru_instruction_is_stored_end_to_end():
    """#560 through the public path: a Russian directive must reach the table.

    The pattern-level cases above rebuild the regex themselves, so they stay
    green if the extractor stops using it. This drives
    ``extract_and_store_facts`` and asserts on the stored row, and it covers the
    punctuation boundary (the capture must stop at the comma) and the newline
    boundary (the following line must not be swallowed).
    """
    with tempfile.TemporaryDirectory() as tmp:
        mem = BeamMemory(session_id="test-560-ru", db_path=Path(tmp) / "memories.db")
        try:
            mem.extract_and_store_facts(
                "всегда запускай тесты перед пушем в main, это важно\n"
                "и не забывай про линтер",
                source_memory_id="mem-560-ru",
            )
            stored = [
                row[0]
                for row in mem.conn.execute(
                    "SELECT instruction FROM memoria_instructions "
                    "WHERE source_memory_id = ?",
                    ("mem-560-ru",),
                ).fetchall()
            ]
            assert stored, "ru directive stored nothing through the public path"
            first = [s for s in stored if "запускай тесты" in s]
            assert first, stored
            assert not any("это важно" in s for s in first), (
                f"capture ran past the comma boundary: {first}"
            )
            assert not any("линтер" in s for s in first), (
                f"capture ran past the newline boundary: {first}"
            )
        finally:
            mem.conn.close()


def test_es_instruction_is_stored_end_to_end_without_truncation():
    """#560 through the public path: the es capture must survive the letter n.

    The doubled ``\\\\n`` truncated captures at the first ``n``, so
    ``antes`` never made it into the stored instruction. Also asserts the
    newline and punctuation boundaries still hold.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mem = BeamMemory(session_id="test-560-es", db_path=Path(tmp) / "memories.db")
        try:
            mem.extract_and_store_facts(
                "siempre ejecuta las pruebas antes de subir a main, por favor\n"
                "y luego avisa al equipo",
                source_memory_id="mem-560-es",
            )
            stored = [
                row[0]
                for row in mem.conn.execute(
                    "SELECT instruction FROM memoria_instructions "
                    "WHERE source_memory_id = ?",
                    ("mem-560-es",),
                ).fetchall()
            ]
            assert stored, "es directive stored nothing through the public path"
            first = [s for s in stored if "ejecuta las pruebas" in s]
            assert first, stored
            assert any("antes de subir" in s for s in first), (
                f"capture truncated at the letter 'n': {first}"
            )
            assert not any("por favor" in s for s in first), (
                f"capture ran past the comma boundary: {first}"
            )
            assert not any("avisa" in s for s in first), (
                f"capture ran past the newline boundary: {first}"
            )
        finally:
            mem.conn.close()


def test_de_nie_does_not_match_inside_knie():
    """German 'nie' must not match inside unrelated words such as 'Knie'."""
    matches = re.findall(
        _runtime_instruction_re("de"),
        "Das Knie tut weh und wir sollten das genauer untersuchen lassen.",
        re.IGNORECASE,
    )
    assert matches == []


@pytest.mark.parametrize(
    "content,expected_fragment",
    [
        ("never commit directly to the main branch please", "commit directly"),
        ("Please always run the tests before pushing to main.", "run the tests"),
        ("You must not store secrets in the config file ever.", "store secrets"),
        # The boundary must still allow a preceding word or punctuation.
        ("Note: never push to main without a review from someone.", "push to main"),
        ("wherever you go, always run the tests before you commit", "run the tests"),
    ],
)
def test_genuine_instructions_still_extract(content, expected_fragment):
    """No-regression: real instructions must survive the boundary anchor."""
    matches = re.findall(_runtime_instruction_re("en"), content, re.IGNORECASE)
    assert matches, f"genuine instruction lost: {content!r}"
    assert any(expected_fragment in m for m in matches), (
        f"expected {expected_fragment!r} in {matches}"
    )


def test_extractor_end_to_end_does_not_invert_whenever():
    """Drive the real extraction path and assert on the stored row.

    The pattern-level tests above bypass the false-positive filter, the bare
    'should' skip, and the memoria_instructions INSERT. This one does not, so a
    refactor of the regex assembly cannot leave those tests green while the
    shipped extractor regresses.
    """
    with tempfile.TemporaryDirectory() as tmp:
        mem = BeamMemory(session_id="test-507", db_path=Path(tmp) / "memories.db")
        try:
            mem.extract_and_store_facts(
                "Good - whenever needed we can use it. I've disabled it.",
                source_memory_id="mem-507",
            )
            stored = [
                row[0]
                for row in mem.conn.execute(
                    "SELECT instruction FROM memoria_instructions "
                    "WHERE source_memory_id = ?",
                    ("mem-507",),
                ).fetchall()
            ]
            assert not any("never" in s.lower() for s in stored), (
                f"extractor stored an inverted instruction: {stored}"
            )
        finally:
            mem.conn.close()


def test_extractor_end_to_end_still_stores_a_genuine_instruction():
    """Companion to the above: proves the end-to-end path can store at all.

    Without this, the inversion test would pass trivially if the extractor
    silently stored nothing (e.g. a renamed table or a changed signature).
    """
    with tempfile.TemporaryDirectory() as tmp:
        mem = BeamMemory(session_id="test-507b", db_path=Path(tmp) / "memories.db")
        try:
            mem.extract_and_store_facts(
                "Please never commit directly to the main branch, use a PR.",
                source_memory_id="mem-507b",
            )
            stored = [
                row[0]
                for row in mem.conn.execute(
                    "SELECT instruction FROM memoria_instructions "
                    "WHERE source_memory_id = ?",
                    ("mem-507b",),
                ).fetchall()
            ]
            assert stored, "extractor stored no instruction for a genuine directive"
            assert any("commit directly" in s.lower() for s in stored), stored
        finally:
            mem.conn.close()
