"""
Entity Sketching System
Lightweight entity extraction and fuzzy matching without heavy NLP dependencies.

Uses regex patterns for entity extraction and pure Python Levenshtein distance
for fuzzy matching. No spaCy, no PyTorch, no external NLP libraries.

Storage: TripleStore triples (subject=memory_id, predicate="mentions", object="entity_name")
"""

import re
from typing import List, Optional, Set, Tuple


# =============================================================================
# STOP WORDS — filtered from entity extraction
# =============================================================================

ENTITY_EXTRACTION_STOP_WORDS: Set[str] = {
    # Standard stop words
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "can", "shall",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "here", "there", "where",
    "when", "what", "which", "who", "whom", "whose", "how", "why",
    # Meta/system words that are NOT meaningful entities — extracted noise
    # from LLM-generated summaries and extraction prompts
    "assistant", "user", "skill", "review", "target", "class",
    "level", "signals", "phase", "api", "pi", "summary", "added",
    "active", "be", "not", "whether", "all", "no", "replying",
    "ai", "memory", "conversation", "fact",
    "false", "true", "none", "null", "signal",
    "hermes", "assistant", "agent", "model", "system", "memory",
    "note", "task", "project", "result", "output", "input", "data",
    "step", "process", "point", "way", "thing", "time", "work",
}

# Backward compatibility alias
_STOP_WORDS = ENTITY_EXTRACTION_STOP_WORDS


# =============================================================================
# REGEX PATTERNS FOR ENTITY EXTRACTION
# =============================================================================

_ENTITY_PATTERNS = [
    # @mentions: @username
    re.compile(r'@(\w{2,30})'),
    # Hashtags: #topic
    re.compile(r'#(\w{2,30})'),
    # Quoted phrases: "Hello World"
    re.compile(r'"([^"]{2,50})"'),
    # Single-quoted phrases: 'Hello World'
    re.compile(r"'([^']{2,50})'"),
    # Capitalized word sequences (2-5 words): New York, Abdias J, San Francisco Bay Area
    re.compile(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*){1,4})\b'),
    # Single capitalized word (fallback): Abdias, Python, John
    re.compile(r'\b([A-Z][a-zA-Z]{1,20})\b'),
]


# =============================================================================
# 1. PURE PYTHON LEVENSHTEIN
# =============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein edit distance between two strings.
    Pure Python, zero dependencies.  O(len(s1) * len(s2)) time, O(min) space.
    """
    if len(s1) < len(s2):
        s1, s2 = s2, s1  # ensure s2 is the shorter one

    if not s2:
        return len(s1)

    # Use two rows (current and previous) to keep space O(min(len1, len2))
    previous_row = list(range(len(s2) + 1))
    current_row = [0] * (len(s2) + 1)

    for i, c1 in enumerate(s1):
        current_row[0] = i + 1

        for j, c2 in enumerate(s2):
            # Cost: 0 if same character, 1 if different
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (0 if c1 == c2 else 1)
            current_row[j + 1] = min(insertions, deletions, substitutions)

        # Swap rows
        previous_row, current_row = current_row, previous_row

    return previous_row[len(s2)]


def similarity(s1: str, s2: str) -> float:
    """
    Entity-aware similarity score: 1.0 = identical, 0.0 = completely different.

    Uses case-insensitive comparison with prefix/substring bonuses for
    entity name matching (e.g., "Abdias" vs "Abdias J" = 0.925).
    """
    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    if s1_lower == s2_lower:
        return 1.0

    max_len = max(len(s1_lower), len(s2_lower))
    if max_len == 0:
        return 1.0

    # Prefix match bonus: 'Abdias' vs 'Abdias J'
    if s1_lower.startswith(s2_lower) or s2_lower.startswith(s1_lower):
        longer = max(len(s1_lower), len(s2_lower))
        shorter = min(len(s1_lower), len(s2_lower))
        # Require at least 30% length ratio to avoid short prefix noise
        # (e.g. "her" prefix-matching "Hermes" with only 3/7 ratio)
        if shorter / longer < 0.3:
            return 0.0
        return 0.7 + (shorter / longer) * 0.3  # 0.7 base + scaled bonus

    # Substring match: 'Mr. Smith' contains 'Smith'
    if s1_lower in s2_lower or s2_lower in s1_lower:
        longer = max(len(s1_lower), len(s2_lower))
        shorter = min(len(s1_lower), len(s2_lower))
        return 0.5 + (shorter / longer) * 0.3

    dist = levenshtein_distance(s1_lower, s2_lower)
    return 1.0 - (dist / max_len)


def extract_entities_regex(text: str) -> List[str]:
    """
    Extract entity candidates from text using regex patterns.

    Returns list of unique entity strings. No external dependencies.
    Filters out stop words, single lowercase words, and pure numbers.
    """
    if not text or not isinstance(text, str):
        return []

    entities: Set[str] = set()

    for pattern in _ENTITY_PATTERNS:
        for match in pattern.finditer(text):
            entity = match.group(1).strip()
            # Filter: must be at least 2 chars
            if len(entity) < 2:
                continue
            # Filter out stop words (single word only); case-insensitive
            words = entity.split()
            if len(words) == 1 and entity.lower() in _STOP_WORDS:
                continue
            # Filter entities where ANY word is a stopword (e.g. "The USER",
            # "Active Signal" -- the stopword contaminates the whole phrase)
            if any(w.lower() in _STOP_WORDS for w in words):
                continue
            # Filter out pure numbers
            if entity.replace('.', '').replace(',', '').isdigit():
                continue
            # Filter out standalone lowercase words (unless quoted/mentioned)
            # But allow @mentions and hashtags which are lowercase by nature
            if len(words) == 1 and entity[0].islower() and not entity.startswith('@') and not entity.startswith('#'):
                # Check if this entity came from an @mention or #hashtag pattern
                # by looking at the original match position in the text
                match_start = match.start(1)  # start of group 1 (the captured entity)
                if match_start > 0:
                    prefix_char = text[match_start - 1] if match_start > 0 else ''
                    if prefix_char in ('@', '#'):
                        pass  # Allow @mentions and hashtags
                    else:
                        continue
                else:
                    continue
            entities.add(entity)

    # Post-process: merge adjacent capitalized words that appear together
    # e.g., if we have "New" and "York" separately, but "New York" also matched,
    # keep only the longest match
    result = sorted(list(entities))
    
    # Remove substrings that are part of longer entities
    # But only for word-like entities (not @mentions or hashtags)
    filtered: Set[str] = set()
    for entity in result:
        is_substring = False
        for other in result:
            if other != entity and entity in other:
                # Don't remove @mentions or hashtags that happen to be substrings
                if entity.startswith('@') or entity.startswith('#'):
                    continue
                # Don't remove if the containing entity starts with @ or #
                if other.startswith('@') or other.startswith('#'):
                    continue
                is_substring = True
                break
        if not is_substring:
            filtered.add(entity)

    return sorted(list(filtered))


def find_similar_entities(entity: str, known_entities: List[str], threshold: float = 0.8) -> List[Tuple[str, float]]:
    """
    Find known entities similar to the given entity.

    Returns list of (entity_name, similarity_score) tuples, sorted by score descending.
    """
    matches: List[Tuple[str, float]] = []
    entity_len = len(entity.strip())
    for known in known_entities:
        if known == entity:
            matches.append((known, 1.0))
            continue
        # Cheap upper bound before the O(n*m) levenshtein in similarity():
        # the edit-distance score is capped at min_len/max_len and the
        # prefix/substring bonuses at 0.7 + ratio*0.3, so when the length
        # ratio alone puts every branch below the threshold the pair can
        # never match. Critical when callers pass long free-text queries
        # against thousands of short entity names.
        known_len = len(known.strip())
        max_len = max(entity_len, known_len, 1)
        ratio = min(entity_len, known_len) / max_len
        if ratio < threshold and (0.7 + ratio * 0.3) < threshold:
            continue
        sim = similarity(entity, known)
        if sim >= threshold:
            matches.append((known, sim))

    matches.sort(key=lambda x: x[1], reverse=True)
    return matches


def entity_extraction_performance(text: str, iterations: int = 1000) -> float:
    """
    Measure entity extraction performance.
    Returns average time per extraction in milliseconds.
    """
    import time
    start = time.perf_counter()
    for _ in range(iterations):
        extract_entities_regex(text)
    elapsed = time.perf_counter() - start
    return (elapsed / iterations) * 1000
