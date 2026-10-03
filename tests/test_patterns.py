"""
Tests for Mnemosyne memory compression + pattern detection.
"""

from datetime import datetime, timedelta

from mnemosyne.core.patterns import (
    MemoryCompressor, PatternDetector, CompressionStats,
    DetectedPattern
)


# ─── CompressionStats ───────────────────────────────────────────────

class TestCompressionStats:
    def test_basic_stats(self):
        stats = CompressionStats(original_size=100, compressed_size=70, ratio=0.7, method="dict")
        assert abs(stats.savings_percent - 30.0) < 0.001

    def test_zero_size(self):
        stats = CompressionStats(original_size=0, compressed_size=0, ratio=1.0, method="none")
        assert stats.savings_percent == 0.0

    def test_no_savings(self):
        stats = CompressionStats(original_size=100, compressed_size=100, ratio=1.0, method="dict")
        assert stats.savings_percent == 0.0


# ─── MemoryCompressor ───────────────────────────────────────────────

class TestMemoryCompressor:
    def test_dict_compression(self):
        comp = MemoryCompressor()
        text = "remember that the user said something important"
        compressed, stats = comp.compress(text, method="dict")
        assert stats.method == "dict"
        assert len(compressed) <= len(text)

    def test_dict_decompression(self):
        comp = MemoryCompressor()
        text = "remember that the user said something"
        compressed, _ = comp.compress(text, method="dict")
        decompressed = comp.decompress(compressed, method="dict")
        assert decompressed == text

    def test_rle_compression(self):
        comp = MemoryCompressor()
        text = "aaaaabbbbbccccc"  # 15 chars, lots of repetition
        compressed, stats = comp.compress(text, method="rle")
        assert stats.method == "rle"
        # Should achieve some compression
        decompressed = comp.decompress(compressed, method="rle")
        assert decompressed == text

    def test_rle_no_repetition(self):
        comp = MemoryCompressor()
        text = "abcdefghijklmnop"  # No repetition
        compressed, stats = comp.compress(text, method="rle")
        # With no repetition, RLE may expand slightly due to brackets
        decompressed = comp.decompress(compressed, method="rle")
        assert decompressed == text

    def test_semantic_compression_long(self):
        comp = MemoryCompressor()
        text = "x" * 600  # Very long
        compressed, stats = comp.compress(text, method="semantic")
        assert stats.method == "semantic"
        assert len(compressed) < len(text)

    def test_semantic_compression_short(self):
        comp = MemoryCompressor()
        text = "Short text"
        compressed, stats = comp.compress(text, method="semantic")
        # Short text should not be compressed
        assert compressed == text

    def test_auto_compression(self):
        comp = MemoryCompressor()
        # Text with dictionary phrases
        text = "remember that the user said mnemosyne is great"
        compressed, stats = comp.compress(text, method="auto")
        assert stats.method in ("dict", "rle")

    def test_compress_batch(self):
        comp = MemoryCompressor()
        memories = [
            {"content": "remember that the user said hello"},
            {"content": "the user asked about mnemosyne"},
            {"content": "conversation about memory systems"},
        ]
        compressed, stats = comp.compress_batch(memories, method="dict")
        assert len(compressed) == 3
        assert stats.memories_compressed == 3
        assert all("_compressed" in m for m in compressed)

    def test_decompress_unknown_method(self):
        comp = MemoryCompressor()
        text = "some text"
        result = comp.decompress(text, method="unknown")
        assert result == text

    def test_empty_compression(self):
        comp = MemoryCompressor()
        compressed, stats = comp.compress("")
        assert compressed == ""


# ─── PatternDetector ────────────────────────────────────────────────

class TestPatternDetector:
    def test_detect_temporal_hour_pattern(self):
        detector = PatternDetector(min_confidence=0.3)
        base = datetime(2026, 1, 1, 9, 0, 0)
        memories = [
            {"content": "Morning meeting", "timestamp": base.isoformat()},
            {"content": "Code review", "timestamp": (base + timedelta(hours=1)).isoformat()},
            {"content": "Standup", "timestamp": (base + timedelta(days=1)).isoformat()},
            {"content": "Planning", "timestamp": (base + timedelta(days=2)).isoformat()},
        ]
        patterns = detector.detect_temporal(memories)
        assert len(patterns) > 0
        assert any(p.pattern_type == "temporal" for p in patterns)

    def test_detect_temporal_empty(self):
        detector = PatternDetector()
        patterns = detector.detect_temporal([])
        assert len(patterns) == 0

    def test_detect_temporal_insufficient_data(self):
        detector = PatternDetector()
        memories = [
            {"content": "One", "timestamp": datetime.now().isoformat()},
        ]
        patterns = detector.detect_temporal(memories)
        assert len(patterns) == 0

    def test_detect_content_keywords(self):
        detector = PatternDetector(min_confidence=0.1)
        memories = [
            {"content": "The user likes Python programming"},
            {"content": "Python is great for scripting"},
            {"content": "The user prefers Python over Java"},
            {"content": "Something unrelated"},
        ]
        patterns = detector.detect_content(memories)
        assert len(patterns) > 0
        assert any("python" in p.description.lower() for p in patterns)

    def test_detect_content_cooccurrence(self):
        detector = PatternDetector(min_confidence=0.1)
        memories = [
            {"content": "The user likes Python programming and Rust language"},
            {"content": "Python programming and Rust language are both great"},
            {"content": "Comparing Python programming with Rust language"},
        ]
        patterns = detector.detect_content(memories)
        # Should detect co-occurrence pattern (words must be 5+ chars)
        cooccurrence = [p for p in patterns if "co-occurring" in p.description.lower()]
        assert len(cooccurrence) > 0

    def test_detect_content_empty(self):
        detector = PatternDetector()
        patterns = detector.detect_content([])
        assert len(patterns) == 0

    def test_detect_sequence(self):
        detector = PatternDetector(min_confidence=0.1)
        memories = [
            {"content": "User asks question", "source": "user", "timestamp": "2026-01-01T09:00:00"},
            {"content": "Agent responds", "source": "agent", "timestamp": "2026-01-01T09:01:00"},
            {"content": "User asks again", "source": "user", "timestamp": "2026-01-01T09:05:00"},
            {"content": "Agent responds again", "source": "agent", "timestamp": "2026-01-01T09:06:00"},
        ]
        patterns = detector.detect_sequence(memories)
        assert len(patterns) > 0
        assert any("user" in p.description.lower() and "agent" in p.description.lower() for p in patterns)

    def test_detect_sequence_insufficient(self):
        detector = PatternDetector()
        patterns = detector.detect_sequence([{"content": "Only one"}])
        assert len(patterns) == 0

    def test_detect_all_combined(self):
        detector = PatternDetector(min_confidence=0.1)
        base = datetime(2026, 1, 1, 9, 0, 0)
        memories = [
            {"content": "The user likes Python", "source": "user", "timestamp": base.isoformat()},
            {"content": "Agent suggests Rust", "source": "agent", "timestamp": (base + timedelta(minutes=1)).isoformat()},
            {"content": "User asks about Python", "source": "user", "timestamp": (base + timedelta(days=1)).isoformat()},
            {"content": "Agent responds", "source": "agent", "timestamp": (base + timedelta(days=1, minutes=1)).isoformat()},
        ]
        patterns = detector.detect_all(memories)
        assert len(patterns) > 0
        # Should have temporal, content, and sequence patterns
        types = set(p.pattern_type for p in patterns)
        assert "temporal" in types or "content" in types or "sequence" in types

    def test_detect_all_sorted_by_confidence(self):
        detector = PatternDetector(min_confidence=0.1)
        memories = [
            {"content": "Python Python Python", "timestamp": "2026-01-01T09:00:00"},
            {"content": "Rust Rust", "timestamp": "2026-01-01T10:00:00"},
        ]
        patterns = detector.detect_all(memories)
        if len(patterns) >= 2:
            assert patterns[0].confidence >= patterns[1].confidence

    def test_summarize_patterns(self):
        detector = PatternDetector(min_confidence=0.1)
        base = datetime(2026, 1, 1, 9, 0, 0)
        memories = [
            {"content": "Python is great", "source": "user", "timestamp": base.isoformat()},
            {"content": "Agent agrees", "source": "agent", "timestamp": (base + timedelta(minutes=1)).isoformat()},
        ]
        summary = detector.summarize_patterns(memories)
        assert "total_memories" in summary
        assert "patterns_found" in summary
        assert summary["total_memories"] == 2

    def test_pattern_to_dict(self):
        pattern = DetectedPattern(
            pattern_type="content",
            description="Test pattern",
            confidence=0.85,
            samples=["sample1", "sample2"],
            metadata={"key": "value"}
        )
        d = pattern.to_dict()
        assert d["pattern_type"] == "content"
        assert d["confidence"] == 0.85
        assert d["samples"] == ["sample1", "sample2"]

    def test_high_confidence_filter(self):
        detector = PatternDetector(min_confidence=0.9)
        memories = [
            {"content": "Python is mentioned once"},
            {"content": "Something else entirely"},
        ]
        patterns = detector.detect_content(memories)
        # With high confidence threshold, should find nothing
        assert len(patterns) == 0


# ─── Mnemosyne wrapper integration (C26) ────────────────────────────

class TestMnemosynePatternMethods:
    """Regression tests for [C26]: Mnemosyne.detect_patterns() and
    summarize_patterns() called self.get_all_memories() which did not exist,
    raising AttributeError on first invocation when no memories arg was passed.
    """

    def test_detect_patterns_no_args_does_not_raise(self, tmp_path):
        from mnemosyne.core.memory import Mnemosyne
        mem = Mnemosyne(session_id="c26", db_path=tmp_path / "c26.db")
        mem.remember("Morning standup notes", source="meeting", importance=0.6)
        mem.remember("User likes Python over Java", source="user", importance=0.7)
        result = mem.detect_patterns()
        assert isinstance(result, list)

    def test_summarize_patterns_no_args_does_not_raise(self, tmp_path):
        from mnemosyne.core.memory import Mnemosyne
        mem = Mnemosyne(session_id="c26", db_path=tmp_path / "c26.db")
        mem.remember("Morning standup notes", source="meeting", importance=0.6)
        mem.remember("Afternoon review", source="meeting", importance=0.6)
        summary = mem.summarize_patterns()
        assert isinstance(summary, dict)
        assert "total_memories" in summary
        assert summary["total_memories"] >= 2

    def test_get_all_memories_returns_working_and_episodic(self, tmp_path):
        """get_all_memories must combine working_memory and episodic_memory rows.

        Drives the episodic insert through the public consolidate_to_episodic
        API instead of raw SQL, so the test does not break the next time the
        episodic_memory schema gains a NOT NULL column.
        """
        from mnemosyne.core.memory import Mnemosyne
        mem = Mnemosyne(session_id="c26", db_path=tmp_path / "c26.db")
        wm_id_one = mem.remember("Working item one", source="user", importance=0.5)
        mem.remember("Working item two", source="agent", importance=0.5)
        mem.beam.consolidate_to_episodic(
            summary="Episodic summary one",
            source_wm_ids=[wm_id_one],
            source="consolidation",
            importance=0.6,
        )

        rows = mem.get_all_memories()
        assert isinstance(rows, list)
        contents = [r["content"] for r in rows]
        assert "Working item one" in contents
        assert "Working item two" in contents
        assert "Episodic summary one" in contents
        # PatternDetector relies on these fields:
        for r in rows:
            assert "content" in r
            assert "timestamp" in r
            assert "source" in r

    def test_get_all_memories_excludes_invalidated(self, tmp_path):
        """Invalidated memories must not surface in pattern analysis."""
        from mnemosyne.core.memory import Mnemosyne
        mem = Mnemosyne(session_id="c26", db_path=tmp_path / "c26.db")
        mem.remember("Keep me visible", source="user", importance=0.5)
        drop_id = mem.remember("Forget about this rule", source="user", importance=0.5)
        mem.invalidate(drop_id)

        contents = [r["content"] for r in mem.get_all_memories()]
        assert "Keep me visible" in contents
        assert "Forget about this rule" not in contents

    def test_get_all_memories_expiry_interpretation(self, tmp_path):
        """#525: get_all_memories applies the documented validity forms.

        Offset-bearing past values are expired, offset-bearing future
        values are kept, naive values are interpreted as UTC, and
        date-only values stay pass-through.
        """
        from datetime import datetime, timedelta, timezone
        from mnemosyne.core.memory import Mnemosyne
        mem = Mnemosyne(session_id="c26", db_path=tmp_path / "c26.db")

        future_offset = (datetime.now(timezone.utc) + timedelta(hours=4)).astimezone(
            timezone(timedelta(hours=-2))
        )
        past_offset = (datetime.now(timezone.utc) - timedelta(hours=4)).astimezone(
            timezone(timedelta(hours=14))
        )
        mem.remember(
            "Future offset expiry", source="user", importance=0.5,
            valid_until=future_offset.isoformat(),
        )
        mem.remember(
            "Past offset expiry", source="user", importance=0.5,
            valid_until=past_offset.isoformat(),
        )
        mem.remember(
            "Date-only expiry", source="user", importance=0.5,
            valid_until="2099-12-31",
        )
        # Genuinely naive legacy value, written directly to bypass the
        # write boundary's UTC normalization, is interpreted as UTC and
        # therefore expired.
        naive_id = mem.remember(
            "Naive legacy expiry", source="user", importance=0.5,
        )
        past_naive = (
            datetime.now(timezone.utc) - timedelta(hours=2)
        ).replace(tzinfo=None).isoformat()
        assert "+" not in past_naive
        mem.beam.conn.execute(
            "UPDATE working_memory SET valid_until = ? WHERE id = ?",
            (past_naive, naive_id),
        )
        mem.beam.conn.commit()

        contents = [r["content"] for r in mem.get_all_memories()]
        assert "Future offset expiry" in contents
        assert "Past offset expiry" not in contents
        assert "Date-only expiry" in contents
        assert "Naive legacy expiry" not in contents
