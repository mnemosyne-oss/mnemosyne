import os
import json
from unittest.mock import patch, MagicMock
import pytest
from pathlib import Path
import tempfile

from mnemosyne.core.llm_conflict_detector import validate_conflict_pair
from mnemosyne.core.beam import BeamMemory
from mnemosyne.core.cost_log import get_cost_stats


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield Path(db_path)
    if os.path.exists(db_path):
        os.unlink(db_path)


def test_llm_conflict_detector_gating(temp_db):
    """Test that LLM conflict validation returns false by default if env is false or fails."""
    # Ensure gating is off by default in testing unless explicitly patched
    with patch("mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED", False):
        is_conflict, conf, correct = validate_conflict_pair(
            "Event is on May 29th", "Actually the event is June 5th", "session_123", temp_db
        )
        assert is_conflict is False


@patch("mnemosyne.core.llm_conflict_detector._call_conflict_llm_with_retry")
def test_llm_conflict_detector_success(mock_call, temp_db):
    """Test that a successful structured JSON response from the LLM is parsed correctly with actual tokens."""
    mock_call.return_value = (
        json.dumps({
            "is_conflict": True,
            "confidence": 0.95,
            "correct_fact": "The event is on June 5th",
            "reason": "Explicit correction"
        }),
        120, # prompt_tokens
        45   # completion_tokens
    )

    with patch("mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED", True):
        is_conflict, conf, correct = validate_conflict_pair(
            "Event is on May 29th", "Actually the event is June 5th", "session_123", temp_db
        )
        assert is_conflict is True
        assert conf == 0.95
        assert correct == "The event is on June 5th"

        # Verify cost stats were written with the actual tokens
        stats = get_cost_stats(session_id="session_123", db_path=temp_db)
        assert stats["total_calls"] == 1
        assert stats["total_tokens"] == 165
        assert stats["total_estimated_cost_usd"] > 0.0


@patch("httpx.Client")
@patch("time.sleep")
def test_llm_conflict_detector_retry_logic(mock_sleep, mock_client_class):
    """Test that _call_conflict_llm_with_retry retries on failure with exponential backoff."""
    from mnemosyne.core.llm_conflict_detector import _call_conflict_llm_with_retry
    
    # Force a valid URL
    with patch("mnemosyne.core.llm_conflict_detector.CONFLICT_LLM_BASE_URL", "https://api.openai.com/v1"):
        mock_client = MagicMock()
        mock_client.post.side_effect = [
            Exception("Transient Network Error"),
            MagicMock(status_code=200, json=lambda: {
                "choices": [{"message": {"content": "response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5}
            })
        ]
        mock_client_class.return_value.__enter__.return_value = mock_client

        res = _call_conflict_llm_with_retry("test prompt")
        assert res is not None
        assert res[0] == "response"
        assert res[1] == 10
        assert res[2] == 5

        # Check mock_sleep was called once for retry with initial_delay = 1.0s
        mock_sleep.assert_called_once_with(1.0)


@patch("mnemosyne.core.llm_conflict_detector._call_conflict_llm_with_retry")
def test_beam_integration_with_llm_conflict(mock_call, temp_db):
    """Test that sleep() method calls LLM validation and invalidates older memory only on True."""
    # Mock LLM to flag conflict
    mock_call.return_value = (
        json.dumps({
            "is_conflict": True,
            "confidence": 0.98,
            "correct_fact": "The event is on June 5th",
            "reason": "Date changed"
        }),
        150,
        50
    )

    # Set up BeamMemory
    mem = BeamMemory(db_path=temp_db, session_id="test_session")

    from datetime import datetime, timedelta

    id1 = "conflict-old"
    id2 = "conflict-new"
    id1_ts = (datetime.now() - timedelta(hours=100)).isoformat()
    id2_ts = (datetime.now() - timedelta(hours=99)).isoformat()
    cursor = mem.conn.cursor()
    cursor.executemany(
        "INSERT INTO working_memory (id, content, source, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
        [
            (id1, "The project meeting was originally scheduled for May 29th", "conversation", id1_ts, "test_session"),
            (id2, "No wait, the project meeting is definitely on June 5th", "conversation", id2_ts, "test_session"),
        ],
    )
    mem.conn.commit()

    # This test targets the sleep → LLM validation → invalidation path.
    # Mock the heuristic candidate seam directly instead of depending on
    # embedding implementation details or phrase overlap thresholds.
    with patch.object(mem, "_detect_conflicts", return_value=[(id1, id2)]):
        with patch("mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED", True), \
             patch("mnemosyne.core.llm_conflict_detector.CONFLICT_LLM_BASE_URL", "https://conflict-llm.test"):
            # endpoint patched truthy: exercises the LLM-validated path this
            # test targets (no-endpoint now routes to detected-only accounting)
            res = mem.sleep()
            assert res["conflicts_resolved"] == 1
            assert res["conflicts_detected_only"] == 0
            state = mem.conn.execute(
                "SELECT valid_until, superseded_by FROM working_memory WHERE id = ?", (id1,)
            ).fetchone()
            assert state["valid_until"] is not None
            assert state["superseded_by"] == id2
            records = mem.conn.execute(
                "SELECT memory_id, validator, action, new_content, note FROM memory_validations"
            ).fetchall()
            assert len(records) == 1
            assert tuple(records[0][:4]) == (
                id1, "llm_conflict", "invalidated", "The event is on June 5th"
            )
            assert json.loads(records[0]["note"]) == {
                "confidence": 0.98, "replacement_id": id2,
            }
