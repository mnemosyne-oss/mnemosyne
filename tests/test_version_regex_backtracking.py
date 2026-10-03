"""Regression coverage for version-string extraction performance and captures."""

import subprocess
import sys

import pytest

from mnemosyne.core.beam import BeamMemory, _VERSION_STRING_RE


_PATHOLOGICAL_INPUT = ("AAAAAA " * 20) + "v1."
_LEGACY_REPRODUCER = r'''
import re

pattern = re.compile(
    r"([A-Z][a-zA-Z]+(?:\s*[A-Z][a-zA-Z]+)*)\s+v?(\d+\.\d+(?:\.\d+)?)"
)
pattern.search(("AAAAAA " * 20) + "v1.")
'''
# Send the production pattern and flags to the subprocess. Importing the
# entire memory stack there would spend the regex deadline on optional
# dependency initialization instead of matching.
_CURRENT_REPRODUCER = (
    "import re\n"
    f"pattern = re.compile({_VERSION_STRING_RE.pattern!r}, {_VERSION_STRING_RE.flags!r})\n"
    f"assert pattern.search({_PATHOLOGICAL_INPUT!r}) is None\n"
)


def test_version_regex_avoids_legacy_backtracking():
    """The old nested quantifier times out while the replacement returns safely."""
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [sys.executable, "-c", _LEGACY_REPRODUCER],
            timeout=2,
        )

    current = subprocess.run(
        [sys.executable, "-c", _CURRENT_REPRODUCER],
        capture_output=True,
        text=True,
        timeout=2,
    )
    assert current.returncode == 0, current.stderr


@pytest.mark.parametrize(
    ("text", "expected_pair"),
    [
        ("PostgreSQL v14.2", ("postgresql_version", "14.2")),
        ("Docker 27.1.1", ("docker_version", "27.1.1")),
        ("Ubuntu Linux 22.04", ("ubuntu_linux_version", "22.04")),
        ("Product v1.2", ("product_version", "1.2")),
        # With no separator, the old and new patterns both treat this as one name.
        ("PostgreSQLServer 14.2", ("postgresqlserver_version", "14.2")),
        ("PostgreSQL v1.", None),
    ],
)
def test_version_regex_preserves_public_extraction(tmp_path, text, expected_pair):
    """The fixed regex preserves version facts through the BEAM public path."""
    beam = BeamMemory(session_id="version-regex", db_path=tmp_path / "beam.db")
    try:
        beam.extract_and_store_facts(text)
        rows = beam.conn.execute(
            "SELECT key, value FROM memoria_facts WHERE fact_type = 'version'"
        ).fetchall()
        version_pairs = {(row["key"], row["value"]) for row in rows}

        if expected_pair is None:
            assert not version_pairs
        else:
            assert version_pairs == {expected_pair}
    finally:
        beam.conn.close()
