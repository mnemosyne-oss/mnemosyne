"""Tests for identity memory auto-capture in sync_turn()."""

import pytest
import sqlite3
import os
import tempfile
from unittest.mock import MagicMock
import importlib
import json
from pathlib import Path


class TestIdentitySignals:
    """Verify _capture_identity_signals detects identity-significant expressions."""

    @pytest.fixture
    def provider(self):
        """Create a MnemosyneMemoryProvider with a temp database."""
        from hermes_memory_provider import MnemosyneMemoryProvider

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "mnemosyne.db")
            provider = MnemosyneMemoryProvider()
            # Initialize with temp path
            provider._agent_context = "test"
            provider._session_id = "test-session"
            provider._skip_contexts = set()
            provider._turn_count = 0
            provider._auto_sleep_enabled = False

            from mnemosyne.core.beam import BeamMemory
            # BeamMemory has no separate initialize() method -- __init__
            # already opens the connection and runs init_beam() (schema
            # setup, sleep_log, etc.). The previous beam.initialize()
            # call raised AttributeError, breaking this fixture on every
            # PR's CI run. Drop it; __init__ is sufficient.
            beam = BeamMemory(db_path=db_path)
            provider._beam = beam

            yield provider

            # Cleanup
            try:
                os.remove(db_path)
                for ext in ("-wal", "-shm"):
                    try:
                        os.remove(db_path + ext)
                    except OSError:
                        pass
            except OSError:
                pass

    def test_captures_imposter_syndrome(self, provider):
        """Identity memory saved when user expresses imposter feelings."""
        provider._beam.remember = MagicMock()
        provider.sync_turn(
            "I feel like an imposter, I barely know my own codebase",
            "Response here",
        )
        # Should have called remember for identity signal
        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) >= 1
        call = identity_calls[0].kwargs
        assert call["importance"] >= 0.8
        assert call["scope"] == "global"
        assert call["source"] == "identity"

    def test_captures_pride(self, provider):
        """Identity memory saved when user expresses pride."""
        provider._beam.remember = MagicMock()
        provider.sync_turn(
            "I'm proud of what we shipped today",
            "Great work!",
        )
        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) >= 1

    def test_no_false_positive_on_neutral(self, provider):
        """No identity memory for neutral conversation."""
        provider._beam.remember = MagicMock()
        provider.sync_turn(
            "Can you run the build and deploy to production?",
            "Sure, running build now.",
        )
        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) == 0

    def test_only_one_identity_per_turn(self, provider):
        """Only one identity memory saved per turn (break after first match)."""
        provider._beam.remember = MagicMock()
        provider.sync_turn(
            "I feel like a fraud and I don't even know how to fix this",
            "Let me help.",
        )
        identity_calls = [
            c for c in provider._beam.remember.call_args_list
            if c.kwargs.get("source") == "identity"
        ]
        assert len(identity_calls) == 1

    def test_identity_stored_in_db(self, provider):
        """Identity memory actually persists to the database."""
        provider.sync_turn(
            "I feel like I barely know my own architecture",
            "Response",
        )
        # Query the database directly
        rows = provider._beam.recall("imposter feeling architecture")
        identity_rows = [r for r in rows if r.get("source") == "identity"]
        assert len(identity_rows) >= 1
        assert identity_rows[0]["importance"] >= 0.8
        assert "[IDENTITY]" in identity_rows[0]["content"]


_LITERAL_SIGNALS = (
    "feeling like", "imposter", "impostor", "barely know",
    "don't know my own", "don't even know how", "want them to feel",
    "i'm proud", "i feel like a", "i don't know how to",
)
_BOUNDARY_CASES = [
    pytest.param(
        "I feel like as memory and reasoning increases, this cache becomes useful.",
        False, id="reported-as-regression",
    ),
    pytest.param(
        "I feel like a fraud when I barely know my own codebase.",
        True, id="true-identity-multiple-signals",
    ),
    pytest.param("Please summarize this synthetic storage example.", False, id="neutral"),
    pytest.param("I'M PROUD of this synthetic example.", True, id="lowercase-preserved"),
    pytest.param("Synthetic filler. " * 30 + "I'm proud", True, id="trigger-after-400"),
    pytest.param("Synthetic example: 中imposter.", False, id="cjk-prefix-full-signal"),
    pytest.param("Synthetic example: imposterя.", False, id="cyrillic-suffix-full-signal"),
]
for _signal in _LITERAL_SIGNALS:
    for _prefix, _suffix, _label in (
        ("", " in this synthetic example.", "start"),
        ("Synthetic example: ", "", "end"),
        ("Synthetic example: (", ").", "punctuation"),
    ):
        _BOUNDARY_CASES.append(pytest.param(
            _prefix + _signal + _suffix, True, id=f"{_signal}-{_label}",
        ))
    for _neighbor in ("x", "7", "_", "é"):
        _BOUNDARY_CASES.extend([
            pytest.param(
                "Synthetic example: " + _neighbor + _signal + ".",
                False, id=f"{_signal}-prefix-{_neighbor}",
            ),
            pytest.param(
                "Synthetic example: " + _signal + _neighbor + ".",
                False, id=f"{_signal}-suffix-{_neighbor}",
            ),
        ])


@pytest.mark.parametrize("package", ["hermes_memory_provider", "mnemosyne_hermes"])
@pytest.mark.parametrize("user,expects_identity", _BOUNDARY_CASES)
def test_public_identity_word_boundaries(
    tmp_path, monkeypatch, package, user, expects_identity, record_property,
):
    """Real public turn capture preserves ordered rows and both ledger proofs."""
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "integrations" / "hermes" / "src"))
    monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MNEMOSYNE_SELF_ECHO_ENABLED", "1")
    module = importlib.import_module(package)
    expected_source = (
        root / "integrations" / "hermes" / "src" / "mnemosyne_hermes" / "__init__.py"
        if package == "mnemosyne_hermes"
        else root / "hermes_memory_provider" / "__init__.py"
    )
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == expected_source.resolve()
    provider = module.MnemosyneMemoryProvider()
    session = "synthetic-identity-boundary"
    provider.initialize(
        session, hermes_home=str(tmp_path / "hermes"), agent_context="primary",
        profile_isolation=False, auto_sleep=False,
        sync_roles=["user", "assistant"], write_classifier="off",
    )
    try:
        assert provider._beam is not None
        assert not provider._auto_sleep_enabled
        assert tuple(provider._IDENTITY_SIGNALS) == _LITERAL_SIGNALS
        db = provider._beam.db_path
        assert Path(db).resolve().is_relative_to(tmp_path)
        ledger = provider._verbatim_ledger
        assert ledger.enabled
        anchor = [{"role": "user", "content": "Synthetic ledger boundary anchor."}]
        ledger.release(session, anchor)
        assistant = "Synthetic assistant response for the boundary fixture."
        provider.sync_turn(
            user, assistant, session_id=session,
            messages=anchor + [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
        )
        assert provider._sync_turn_diagnostics()["failed"] == 0
        proof = ledger.snapshot_for(session)
        assert proof is not None
        assert len(proof.captures) == 2
        assert all(c.source_hash and c.stored_hash for c in proof.captures)
    finally:
        provider.shutdown()

    with sqlite3.connect("file:" + str(db) + "?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT content, source, importance, scope, veracity, metadata_json "
            "FROM working_memory ORDER BY rowid"
        ).fetchall()
    record_property("ordered_sqlite_rows", json.dumps(rows))
    record_property("ledger_capture_count", len(proof.captures))
    # Public sync_turn retains its existing default 500-character USER cap.
    expected = ["[USER] " + user[:500], "[ASSISTANT] " + assistant]
    if expects_identity:
        expected.insert(1, "[IDENTITY] " + user[:400])
    assert [row[0] for row in rows] == expected
    identities = [row for row in rows if row[1] == "identity"]
    assert len(identities) == int(expects_identity)
    for row in identities:
        assert row[2:5] == (0.85, "global", "stated")
    conversation = [row for row in rows if row[1] != "identity"]
    assert len(conversation) == 2
    assert all(
        "_mnemosyne_self_echo_capture" in json.loads(row[5] or "{}")
        for row in conversation
    )
