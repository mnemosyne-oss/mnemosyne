"""Regression coverage for English day-first named dates (#907)."""

import sqlite3

import pytest

from mnemosyne.core.memory import Mnemosyne


@pytest.mark.parametrize(
    ("date_text", "expected_value"),
    [
        ("March 29, 2024", "March 29, 2024"),
        ("March 29 2024", "March 29 2024"),
        ("12 Mar 2024", "12 Mar 2024"),
        ("March 12, 1985", "March 12, 1985"),
        ("12 March 1985", "12 March 1985"),
        ("21st February 1999", "21st February 1999"),
        ("30 Dec 2018", "30 Dec 2018"),
    ],
)
def test_remember_extracts_complete_english_named_date(
    tmp_path, date_text, expected_value
):
    """The public remember path preserves either English named-date order."""
    db_path = tmp_path / "mnemosyne.db"
    memory = Mnemosyne(session_id="issue-907", db_path=db_path)
    try:
        memory.remember(
            f"The event happened on {date_text}.", source="user", extract=True
        )
    finally:
        memory.conn.close()

    with sqlite3.connect(db_path) as conn:
        values = [
            row[0]
            for row in conn.execute(
                "SELECT value FROM memoria_facts WHERE fact_type = 'date' "
                "AND key = 'named_date'"
            )
        ]

    assert values == [expected_value]


@pytest.mark.parametrize(
    "invalid_date_text",
    [
        "12 marching",
        "312 March 2024",
        "x12 March 2024",
        "March 12x",
        "xMarch 12, 1985",
        "March 12, 1985x",
        "March 292024",
        "12 March 292024",
        "12 March 2024x",
    ],
)
def test_remember_does_not_extract_named_date_from_invalid_boundary(
    tmp_path, invalid_date_text
):
    """The public remember path rejects partial named-date matches."""
    db_path = tmp_path / "mnemosyne.db"
    memory = Mnemosyne(session_id="issue-907", db_path=db_path)
    try:
        memory.remember(
            f"The event happened on {invalid_date_text}.", source="user", extract=True
        )
    finally:
        memory.conn.close()

    with sqlite3.connect(db_path) as conn:
        values = [
            row[0]
            for row in conn.execute(
                "SELECT value FROM memoria_facts WHERE fact_type = 'date' "
                "AND key = 'named_date'"
            )
        ]

    assert values == []
