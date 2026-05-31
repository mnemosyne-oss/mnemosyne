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


@patch("mnemosyne.core.llm_conflict_detector._call_conflict_llm")
def test_llm_conflict_detector_success(mock_call, temp_db):
    """Test that a successful structured JSON response from the LLM is parsed correctly."""
    mock_call.return_value = json.dumps({
        "is_conflict": True,
        "confidence": 0.95,
        "correct_fact": "The event is on June 5th",
        "reason": "Explicit correction"
    })

    with patch("mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED", True):
        is_conflict, conf, correct = validate_conflict_pair(
            "Event is on May 29th", "Actually the event is June 5th", "session_123", temp_db
        )
        assert is_conflict is True
        assert conf == 0.95
        assert correct == "The event is on June 5th"

        # Verify cost stats were written
        stats = get_cost_stats(session_id="session_123", db_path=temp_db)
        assert stats["total_calls"] == 1
        assert stats["total_estimated_cost_usd"] > 0.0


@patch("mnemosyne.core.llm_conflict_detector._call_conflict_llm")
def test_beam_integration_with_llm_conflict(mock_call, temp_db):
    """Test that sleep() method calls LLM validation and invalidates older memory only on True."""
    # Mock LLM to flag conflict
    mock_call.return_value = json.dumps({
        "is_conflict": True,
        "confidence": 0.98,
        "correct_fact": "The event is on June 5th",
        "reason": "Date changed"
    })

    # Set up BeamMemory
    mem = BeamMemory(db_path=temp_db, session_id="test_session")
    
    # Enable embeddings availability mock so Phase 1 collects embeddings
    with patch("mnemosyne.core.beam._embeddings") as mock_emb:
        mock_emb.available.return_value = True
        mock_emb._DEFAULT_MODEL = "test_model"
        # Return mock embeddings vectors (must be length 1 for single content embed)
        mock_emb.embed.return_value = [[1.0] * 128]
        mock_emb.serialize.return_value = "[1.0]"

        # Create two working memory rows that represent semantic overlap but timestamps differ
        # Use different times (>1 hour) to trigger heuristics, and make both older than sleep threshold (>12 hours)
        id1 = mem.remember("The project meeting was originally scheduled for May 29th")
        
        # Modify the timestamp of the first memory to be 14 hours older using Python ISO format
        from datetime import datetime, timedelta
        id1_ts = (datetime.now() - timedelta(hours=14)).isoformat()
        cursor = mem.conn.cursor()
        cursor.execute("UPDATE working_memory SET timestamp = ? WHERE id = ?", (id1_ts, id1))
        mem.conn.commit()

        # Add the newer correction
        id2 = mem.remember("No wait, the project meeting is definitely on June 5th")
        
        # Modify the timestamp of the second memory to be 13 hours older using Python ISO format
        id2_ts = (datetime.now() - timedelta(hours=13)).isoformat()
        cursor.execute("UPDATE working_memory SET timestamp = ? WHERE id = ?", (id2_ts, id2))
        mem.conn.commit()

        # Verify heuristic detection triggers them as candidate conflict
        ctx = mem.get_context(limit=10)
        candidates = mem._detect_conflicts(ctx, similarity_threshold=0.8)
        assert (id1, id2) in candidates or (id2, id1) in candidates

        # Let's test sleep consolidation with LLM active
        with patch("mnemosyne.core.llm_conflict_detector.LLM_CONFLICT_DETECTION_ENABLED", True):
            # Run sleep
            res = mem.sleep()
            assert res.get("conflicts_resolved", 0) >= 1

            # Verify older memory (id1) is indeed invalidated/superseded_by the newer one (id2)
            row = mem.get(id1)
            # Since it was invalidated, we should see it marked as superseded
            # Let's check db row directly
            cursor = mem.conn.cursor()
            cursor.execute("SELECT superseded_by FROM working_memory WHERE id = ?", (id1,))
            superseded = cursor.fetchone()[0]
            assert superseded == id2
