"""
Mnemosyne Memory Compression + Pattern Detection
==================================================

Compress memory content and detect recurring patterns.

Compression strategies:
- Run-length encoding for repetitive sequences
- Dictionary-based compression for common phrases
- Semantic compression: summarize similar memories

Pattern detection:
- Temporal patterns: recurring times, intervals
- Content patterns: co-occurring topics, sequences
- Sequence patterns: ordered memory chains
"""

import re
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import Counter, defaultdict


@dataclass
class CompressionStats:
    """Statistics from compression operations."""
    original_size: int = 0
    compressed_size: int = 0
    ratio: float = 0.0
    method: str = ""
    patterns_found: int = 0
    memories_compressed: int = 0

    @property
    def savings_percent(self) -> float:
        if self.original_size == 0:
            return 0.0
        return (1.0 - self.compressed_size / self.original_size) * 100


class MemoryCompressor:
    """
    Compress memory content using multiple strategies.

    Strategies are applied in order of aggressiveness:
    1. Dictionary-based: replace common phrases with tokens
    2. Run-length: collapse repeated character sequences
    3. Semantic: summarize groups of similar memories
    """

    def __init__(self, dictionary: Optional[Dict[str, str]] = None):
        self.dictionary = dictionary or self._build_default_dict()
        self._stats = CompressionStats()

    @staticmethod
    def _build_default_dict() -> Dict[str, str]:
        """Build a default compression dictionary for common phrases."""
        return {
            "remember that ": "",
            "the user said ": "",
            "the user asked ": "",
            "the user wants ": "",
            "conversation about ": "",
            "please note that ": "",
            "important: ": "",
            "user preference: ": "",
            "project context: ": "	",
            "api key ": "\x0A",
            "token ": "\x0B",
            "session ": "\x0C",
            "mnemosyne ": "\x0D",
        }

    def compress(self, content: str, method: str = "dict") -> Tuple[str, CompressionStats]:
        """
        Compress a single memory content string.

        Args:
            content: The memory content to compress
            method: Compression method — "dict", "rle", "semantic", or "auto"

        Returns:
            Tuple of (compressed_content, stats)
        """
        original_size = len(content.encode("utf-8"))

        if method == "auto":
            # Try dict first, fall back to RLE if no savings
            compressed, stats = self._dict_compress(content)
            if stats.savings_percent < 5:
                compressed, stats = self._rle_compress(content)
            return compressed, stats

        if method == "dict":
            compressed, stats = self._dict_compress(content)
        elif method == "rle":
            compressed, stats = self._rle_compress(content)
        elif method == "semantic":
            compressed, stats = self._semantic_compress_single(content)
        else:
            compressed, stats = content, CompressionStats(
                original_size=original_size, compressed_size=original_size,
                ratio=1.0, method="none"
            )

        return compressed, stats

    def _dict_compress(self, content: str) -> Tuple[str, CompressionStats]:
        """Dictionary-based compression."""
        original_size = len(content.encode("utf-8"))
        compressed = content
        for phrase, token in self.dictionary.items():
            compressed = compressed.replace(phrase, token)
        compressed_size = len(compressed.encode("utf-8"))
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        stats = CompressionStats(
            original_size=original_size, compressed_size=compressed_size,
            ratio=ratio, method="dict"
        )
        return compressed, stats

    def _rle_compress(self, content: str) -> Tuple[str, CompressionStats]:
        """Run-length encoding for repeated characters."""
        original_size = len(content.encode("utf-8"))
        if not content:
            return content, CompressionStats(original_size=0, compressed_size=0, ratio=1.0, method="rle")

        compressed = []
        count = 1
        for i in range(1, len(content)):
            if content[i] == content[i - 1] and count < 255:
                count += 1
            else:
                if count > 3:
                    compressed.append(f"[{content[i-1]}*{count}]")
                else:
                    compressed.append(content[i-count:i])
                count = 1
        # Handle last run
        if count > 3:
            compressed.append(f"[{content[-1]}*{count}]")
        else:
            compressed.append(content[-count:])

        compressed_str = "".join(compressed)
        compressed_size = len(compressed_str.encode("utf-8"))
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        stats = CompressionStats(
            original_size=original_size, compressed_size=compressed_size,
            ratio=ratio, method="rle"
        )
        return compressed_str, stats

    def _semantic_compress_single(self, content: str) -> Tuple[str, CompressionStats]:
        """Semantic compression for a single memory (placeholder for LLM-based)."""
        # For now, just truncate with ellipsis if very long
        original_size = len(content.encode("utf-8"))
        if original_size > 500:
            compressed = content[:250] + " [...] " + content[-100:]
        else:
            compressed = content
        compressed_size = len(compressed.encode("utf-8"))
        ratio = compressed_size / original_size if original_size > 0 else 1.0
        stats = CompressionStats(
            original_size=original_size, compressed_size=compressed_size,
            ratio=ratio, method="semantic"
        )
        return compressed, stats

    def compress_batch(self, memories: List[Dict[str, Any]],
                       method: str = "auto") -> Tuple[List[Dict[str, Any]], CompressionStats]:
        """
        Compress a batch of memories.

        Returns:
            Tuple of (compressed_memories, aggregate_stats)
        """
        total_original = 0
        total_compressed = 0
        compressed_memories = []

        for mem in memories:
            content = mem.get("content", "")
            c, s = self.compress(content, method=method)
            total_original += s.original_size
            total_compressed += s.compressed_size
            new_mem = dict(mem)
            new_mem["content"] = c
            new_mem["_compressed"] = True
            new_mem["_compression_method"] = s.method
            compressed_memories.append(new_mem)

        ratio = total_compressed / total_original if total_original > 0 else 1.0
        stats = CompressionStats(
            original_size=total_original,
            compressed_size=total_compressed,
            ratio=ratio,
            method=method,
            memories_compressed=len(memories)
        )
        return compressed_memories, stats

    def decompress(self, content: str, method: str = "dict") -> str:
        """Decompress content compressed with the given method."""
        if method == "dict":
            # Reverse dictionary
            reverse = {v: k for k, v in self.dictionary.items()}
            for token, phrase in reverse.items():
                content = content.replace(token, phrase)
            return content
        elif method == "rle":
            # Expand RLE sequences like [a*5] -> aaaaa
            def expand(match):
                char, count = match.group(1), int(match.group(2))
                return char * count
            return re.sub(r'\[(.)\*(\d+)\]', expand, content)
        else:
            return content


@dataclass
class DetectedPattern:
    """A detected pattern in memory data."""
    pattern_type: str  # "temporal", "content", "sequence"
    description: str
    confidence: float  # 0.0 - 1.0
    samples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "confidence": self.confidence,
            "samples": self.samples,
            "metadata": self.metadata,
        }


class PatternDetector:
    """
    Detect recurring patterns in memory data.

    Pattern types:
    - Temporal: recurring times, daily/weekly patterns
    - Content: co-occurring topics, frequent keywords
    - Sequence: ordered chains of related memories
    """

    def __init__(self, min_confidence: float = 0.6):
        self.min_confidence = min_confidence

    def detect_temporal(self, memories: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect temporal patterns in memory timestamps."""
        patterns = []
        timestamps = []
        for mem in memories:
            ts = mem.get("timestamp") or mem.get("created_at")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    timestamps.append(dt)
                except Exception:
                    pass

        if len(timestamps) < 3:
            return patterns

        # Hour-of-day distribution
        hours = [t.hour for t in timestamps]
        hour_counts = Counter(hours)
        total = len(hours)

        for hour, count in hour_counts.most_common(3):
            confidence = count / total
            if confidence >= self.min_confidence:
                patterns.append(DetectedPattern(
                    pattern_type="temporal",
                    description=f"Memories frequently created at {hour:02d}:00 ({count}/{total} times)",
                    confidence=confidence,
                    samples=[t.isoformat() for t in timestamps if t.hour == hour][:3],
                    metadata={"hour": hour, "count": count, "total": total}
                ))

        # Day-of-week distribution
        weekdays = [t.weekday() for t in timestamps]
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_counts = Counter(weekdays)
        for day, count in day_counts.most_common(2):
            confidence = count / total
            if confidence >= self.min_confidence:
                patterns.append(DetectedPattern(
                    pattern_type="temporal",
                    description=f"Memories frequently created on {day_names[day]} ({count}/{total} times)",
                    confidence=confidence,
                    samples=[t.isoformat() for t in timestamps if t.weekday() == day][:3],
                    metadata={"day": day_names[day], "count": count, "total": total}
                ))

        return patterns

    def detect_content(self, memories: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect content patterns (co-occurring topics, frequent keywords)."""
        patterns = []
        all_text = " ".join(m.get("content", "") for m in memories)

        # Simple keyword extraction (words > 4 chars, frequency > 1)
        words = re.findall(r'\b[a-zA-Z]{5,}\b', all_text.lower())
        stopwords = {"about", "after", "before", "being", "could", "doing", "every", "having", "might",
                     "other", "should", "their", "there", "these", "those", "through", "under", "where",
                     "which", "while", "would", "mnemosyne", "memory", "memories"}
        words = [w for w in words if w not in stopwords]
        word_counts = Counter(words)
        total_words = len(words)

        for word, count in word_counts.most_common(5):
            confidence = min(1.0, count / max(3, total_words * 0.05))
            if count >= 2 and confidence >= self.min_confidence:
                samples = [m.get("content", "") for m in memories if word in m.get("content", "").lower()][:3]
                patterns.append(DetectedPattern(
                    pattern_type="content",
                    description=f"Frequent topic: '{word}' appears {count} times",
                    confidence=confidence,
                    samples=samples,
                    metadata={"word": word, "count": count}
                ))

        # Co-occurrence: pairs of keywords that appear together
        if len(memories) >= 3:
            cooccurrence = defaultdict(int)
            for mem in memories:
                content = mem.get("content", "").lower()
                mem_words = set(re.findall(r'\b[a-zA-Z]{5,}\b', content)) - stopwords
                for w1 in mem_words:
                    for w2 in mem_words:
                        if w1 < w2:
                            cooccurrence[(w1, w2)] += 1

            for (w1, w2), count in sorted(cooccurrence.items(), key=lambda x: -x[1])[:3]:
                confidence = min(1.0, count / len(memories))
                if count >= 2 and confidence >= self.min_confidence:
                    patterns.append(DetectedPattern(
                        pattern_type="content",
                        description=f"Co-occurring topics: '{w1}' + '{w2}' appear together {count} times",
                        confidence=confidence,
                        samples=[m.get("content", "") for m in memories
                                if w1 in m.get("content", "").lower() and w2 in m.get("content", "").lower()][:3],
                        metadata={"word1": w1, "word2": w2, "count": count}
                    ))

        return patterns

    def detect_sequence(self, memories: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Detect sequence patterns (ordered chains of related memories)."""
        patterns = []
        if len(memories) < 3:
            return patterns

        # Sort by timestamp
        sorted_mems = sorted(
            [m for m in memories if m.get("timestamp")],
            key=lambda m: m.get("timestamp", "")
        )

        # Look for source sequences
        sources = [m.get("source", "unknown") for m in sorted_mems]
        source_pairs = [(sources[i], sources[i+1]) for i in range(len(sources)-1)]
        pair_counts = Counter(source_pairs)

        for (s1, s2), count in pair_counts.most_common(3):
            confidence = min(1.0, count / max(2, len(sources) - 1))
            if count >= 2 and confidence >= self.min_confidence:
                samples = []
                for i in range(len(sources) - 1):
                    if sources[i] == s1 and sources[i+1] == s2:
                        samples.append(f"{sorted_mems[i].get('content', '')[:50]}... -> {sorted_mems[i+1].get('content', '')[:50]}...")
                        if len(samples) >= 2:
                            break
                patterns.append(DetectedPattern(
                    pattern_type="sequence",
                    description=f"Sequence pattern: '{s1}' often followed by '{s2}' ({count} times)",
                    confidence=confidence,
                    samples=samples,
                    metadata={"source1": s1, "source2": s2, "count": count}
                ))

        return patterns

    def detect_all(self, memories: List[Dict[str, Any]]) -> List[DetectedPattern]:
        """Run all pattern detectors and return combined results."""
        patterns = []
        patterns.extend(self.detect_temporal(memories))
        patterns.extend(self.detect_content(memories))
        patterns.extend(self.detect_sequence(memories))
        # Sort by confidence descending
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns

    def summarize_patterns(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a human-readable summary of detected patterns."""
        patterns = self.detect_all(memories)
        return {
            "total_memories": len(memories),
            "patterns_found": len(patterns),
            "temporal_patterns": [p.to_dict() for p in patterns if p.pattern_type == "temporal"],
            "content_patterns": [p.to_dict() for p in patterns if p.pattern_type == "content"],
            "sequence_patterns": [p.to_dict() for p in patterns if p.pattern_type == "sequence"],
            "top_pattern": patterns[0].to_dict() if patterns else None,
        }
