"""
Mnemosyne Typed Memory Schema
==============================
Deterministic, rule-based memory type classification.
Building on Memanto (arXiv:2604.22085) typed semantic memory.

Zero LLM calls. Zero overhead. Pattern-based classification at ingestion time.

13 Memory Types:
1.  fact         - Objective, verifiable information
2.  preference   - User/system preferences and tastes
3.  decision     - Choices that affect future behavior
4.  commitment   - Promises, obligations, deadlines
5.  goal         - Objectives to achieve
6.  event        - Historical occurrences with temporal anchors
7.  instruction  - Rules, guidelines, procedures
8.  relationship - Entity connections and associations
9.  context      - Situational, environmental information
10. learning     - Lessons derived from experience
11. observation  - Patterns noticed over time
12. error        - Mistakes to avoid, failures
13. artifact     - References to documents, code, external resources

Each type has:
- Pattern matchers (regex + keyword)
- Confidence scoring (0.0-1.0)
- Priority signal (stable, decaying, time-critical)
- Conflict detection rules
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class MemoryType(Enum):
    """13-type semantic memory classification."""
    FACT = "fact"
    PREFERENCE = "preference"
    DECISION = "decision"
    COMMITMENT = "commitment"
    GOAL = "goal"
    EVENT = "event"
    INSTRUCTION = "instruction"
    RELATIONSHIP = "relationship"
    CONTEXT = "context"
    LEARNING = "learning"
    OBSERVATION = "observation"
    ERROR = "error"
    ARTIFACT = "artifact"
    UNKNOWN = "unknown"


@dataclass
class TypeMatch:
    """Result of type classification."""
    memory_type: MemoryType
    confidence: float
    matched_pattern: str
    priority: str  # stable, moderate, high, time_critical


# --- Pattern Definitions ---
# Each pattern: (regex_pattern, memory_type, base_confidence, priority)

TYPE_PATTERNS: List[Tuple[str, MemoryType, float, str]] = [
    # FACT: Objective, verifiable information
    (r"\b(is|are|was|were)\s+(a|an|the)\s+\w+", MemoryType.FACT, 0.6, "stable"),
    (r"\b(has|have|had)\s+\d+", MemoryType.FACT, 0.7, "stable"),
    (r"\b(contains|consists?|comprises?)\b", MemoryType.FACT, 0.8, "stable"),
    (r"\b(version|v)\s*\d+\.?\d*", MemoryType.FACT, 0.9, "stable"),
    (r"\b(API|endpoint|URL|database|DB)\s+(is|at|points?\s+to)", MemoryType.FACT, 0.8, "stable"),
    (r"\b(created|modified|updated)\s+(on|at)\s+\d{4}", MemoryType.FACT, 0.8, "stable"),
    
    # PREFERENCE: User/system preferences
    (r"\b(prefer|likes?|enjoys?|loves?|hates?|dislikes?)\b", MemoryType.PREFERENCE, 0.8, "moderate"),
    (r"\b(want|wants|wanted)\s+(to|the|a|an)\b", MemoryType.PREFERENCE, 0.6, "moderate"),
    (r"\b(rather|instead|alternative)\b", MemoryType.PREFERENCE, 0.5, "moderate"),
    (r"\b(dark\s+mode|light\s+mode|theme|color\s+scheme)\b", MemoryType.PREFERENCE, 0.9, "moderate"),
    (r"\b(usually|typically|normally|generally)\b", MemoryType.PREFERENCE, 0.6, "moderate"),
    
    # DECISION: Choices affecting future
    (r"\b(decided|chose|selected|picked|opted)\b", MemoryType.DECISION, 0.9, "high"),
    (r"\b(going\s+with|settled\s+on|locked\s+in)\b", MemoryType.DECISION, 0.8, "high"),
    (r"\b(choose|select|pick)\s+(between|from|among)\b", MemoryType.DECISION, 0.7, "high"),
    (r"\b(final\s+decision|final\s+call|final\s+choice)\b", MemoryType.DECISION, 0.9, "high"),
    (r"\b(will\s+use|using|adopt|adopting)\s+(the|a|an)?\s*\w+", MemoryType.DECISION, 0.7, "high"),
    
    # COMMITMENT: Promises, obligations, deadlines
    (r"\b(will|shall|must|need\s+to)\s+\w+\s+(by|before|until)\b", MemoryType.COMMITMENT, 0.8, "time_critical"),
    (r"\b(deadline|due\s+date|due|milestone)\b", MemoryType.COMMITMENT, 0.9, "time_critical"),
    (r"\b(promise|committed|pledged|obligated)\b", MemoryType.COMMITMENT, 0.9, "time_critical"),
    (r"\b(deliver|ship|release|deploy)\s+(by|before|on)\b", MemoryType.COMMITMENT, 0.8, "time_critical"),
    (r"\b(EOD|COB|end\s+of\s+day|close\s+of\s+business)\b", MemoryType.COMMITMENT, 0.7, "time_critical"),
    (r"\b(tomorrow|next\s+week|Monday|Friday)\s+(by|at)\b", MemoryType.COMMITMENT, 0.6, "time_critical"),
    
    # GOAL: Objectives to achieve
    (r"\b(goal|objective|target|aim|purpose)\b", MemoryType.GOAL, 0.9, "high"),
    (r"\b(achieve|reach|hit|attain|accomplish)\s+\d+", MemoryType.GOAL, 0.8, "high"),
    (r"\b(KPI|metric|OKR|success\s+criteria)\b", MemoryType.GOAL, 0.9, "high"),
    (r"\b(roadmap|plan|strategy)\s+(for|to)\b", MemoryType.GOAL, 0.7, "high"),
    (r"\b(reach|get\s+to|grow\s+to)\s+\d+[KkMm]?\s+(users|customers|revenue)\b", MemoryType.GOAL, 0.8, "high"),
    
    # EVENT: Historical occurrences
    (r"\b(meeting|call|discussion|conversation)\s+(with|about)\b", MemoryType.EVENT, 0.7, "decaying"),
    (r"\b(happened|occurred|took\s+place|went\s+down)\b", MemoryType.EVENT, 0.8, "decaying"),
    (r"\b(yesterday|last\s+week|last\s+month|earlier\s+today)\b", MemoryType.EVENT, 0.6, "decaying"),
    (r"\b(scheduled|planned|booked|set\s+up)\s+(for|at)\b", MemoryType.EVENT, 0.7, "decaying"),
    (r"\b(incident|outage|bug|issue)\s+#?\d+", MemoryType.EVENT, 0.8, "decaying"),
    (r"\b( launched|released|shipped|deployed)\s+(on|at)\b", MemoryType.EVENT, 0.8, "decaying"),
    
    # INSTRUCTION: Rules, guidelines
    (r"\b(always|never|must|should|shall|do\s+not|don't)\b", MemoryType.INSTRUCTION, 0.7, "stable"),
    (r"\b(rule|policy|guideline|procedure|protocol)\b", MemoryType.INSTRUCTION, 0.9, "stable"),
    (r"\b(how\s+to|steps?\s+to|guide\s+to|tutorial)\b", MemoryType.INSTRUCTION, 0.8, "stable"),
    (r"\b(remember\s+to|make\s+sure|ensure|verify)\b", MemoryType.INSTRUCTION, 0.6, "stable"),
    (r"\b(first|then|next|finally)\s*,?\s*\w+", MemoryType.INSTRUCTION, 0.5, "stable"),
    (r"\b(if\s+.+\s+then\s+.+)", MemoryType.INSTRUCTION, 0.7, "stable"),
    
    # RELATIONSHIP: Entity connections
    (r"\b(manages?|reports?\s+to|supervises?|leads?)\b", MemoryType.RELATIONSHIP, 0.9, "stable"),
    (r"\b(owns?|belongs?\s+to|part\s+of|member\s+of)\b", MemoryType.RELATIONSHIP, 0.8, "stable"),
    (r"\b(works?\s+with|collaborates?\s+with|partners?\s+with)\b", MemoryType.RELATIONSHIP, 0.8, "stable"),
    (r"\b(depends?\s+on|requires?|needs?)\b", MemoryType.RELATIONSHIP, 0.7, "stable"),
    (r"\b(related\s+to|connected\s+to|associated\s+with)\b", MemoryType.RELATIONSHIP, 0.6, "stable"),
    (r"\b(is\s+a|is\s+an)\s+(type\s+of|kind\s+of|form\s+of)\b", MemoryType.RELATIONSHIP, 0.7, "stable"),
    
    # CONTEXT: Situational information
    (r"\b(currently|right\s+now|at\s+the\s+moment|presently)\b", MemoryType.CONTEXT, 0.7, "high"),
    (r"\b(working\s+on|focusing\s+on|dealing\s+with)\b", MemoryType.CONTEXT, 0.8, "high"),
    (r"\b(status|state|phase|stage)\s+(is|of)\b", MemoryType.CONTEXT, 0.7, "high"),
    (r"\b(in\s+progress|ongoing|active|pending|blocked)\b", MemoryType.CONTEXT, 0.8, "high"),
    (r"\b(environment|setup|configuration|settings?)\b", MemoryType.CONTEXT, 0.6, "high"),
    (r"\b(today|this\s+week|this\s+sprint|this\s+quarter)\b", MemoryType.CONTEXT, 0.5, "high"),
    
    # LEARNING: Lessons from experience
    (r"\b(learned|realized|discovered|found\s+out)\b", MemoryType.LEARNING, 0.8, "accumulating"),
    (r"\b(lesson|takeaway|insight|finding)\b", MemoryType.LEARNING, 0.9, "accumulating"),
    (r"\b(turns?\s+out|surprisingly|interestingly)\b", MemoryType.LEARNING, 0.7, "accumulating"),
    (r"\b(should\s+have|could\s+have|would\s+have)\b", MemoryType.LEARNING, 0.6, "accumulating"),
    (r"\b(best\s+practice|lessons?\s+learned|post[-\s]?mortem)\b", MemoryType.LEARNING, 0.9, "accumulating"),
    
    # OBSERVATION: Patterns noticed
    (r"\b(noticed|observed|saw|seems?)\b", MemoryType.OBSERVATION, 0.7, "evolving"),
    (r"\b(pattern|trend|correlation|tends?\s+to)\b", MemoryType.OBSERVATION, 0.9, "evolving"),
    (r"\b(often|frequently|sometimes|rarely|usually)\s+\w+", MemoryType.OBSERVATION, 0.6, "evolving"),
    (r"\b(appears?|looks?\s+like|seems?\s+like)\b", MemoryType.OBSERVATION, 0.6, "evolving"),
    (r"\b(increasing|decreasing|growing|shrinking|stable)\b", MemoryType.OBSERVATION, 0.7, "evolving"),
    (r"\b(every\s+time|whenever|each\s+time)\b", MemoryType.OBSERVATION, 0.8, "evolving"),
    
    # ERROR: Mistakes to avoid
    (r"\b(error|bug|issue|problem|failure|crash)\b", MemoryType.ERROR, 0.7, "persistent"),
    (r"\b(broke|broken|failed|failing|doesn't\s+work)\b", MemoryType.ERROR, 0.8, "persistent"),
    (r"\b(do\s+not|never|avoid|watch\s+out|be\s+careful)\s+\w+\s+(error|bug|issue)\b", MemoryType.ERROR, 0.9, "persistent"),
    (r"\b(deprecated|obsolete|legacy|outdated)\b", MemoryType.ERROR, 0.8, "persistent"),
    (r"\b(exception|timeout|crash|hang|freeze)\b", MemoryType.ERROR, 0.8, "persistent"),
    (r"\b(workaround|hotfix|patch|kludge)\b", MemoryType.ERROR, 0.7, "persistent"),
    
    # ARTIFACT: Document/code references
    (r"\b(document|doc|spreadsheet|sheet|slide)\b", MemoryType.ARTIFACT, 0.6, "reference"),
    (r"\b(file|folder|directory|path)\s+(name|called|at)\b", MemoryType.ARTIFACT, 0.7, "reference"),
    (r"\b(PR|pull\s+request|issue|ticket|ticket)\s+#?\d+", MemoryType.ARTIFACT, 0.9, "reference"),
    (r"\b(commit|branch|tag|release)\s+[a-f0-9]{7,40}\b", MemoryType.ARTIFACT, 0.9, "reference"),
    (r"\b(repo|repository|project|codebase)\s+(at|on|in)\b", MemoryType.ARTIFACT, 0.7, "reference"),
    (r"\b(link|URL|href|reference)\s+(to|for)\b", MemoryType.ARTIFACT, 0.6, "reference"),
    (r"\b(README|CHANGELOG|LICENSE|CONTRIBUTING)\b", MemoryType.ARTIFACT, 0.9, "reference"),
]


# --- Confidence Boosters ---
# Additional keywords that boost confidence when found alongside pattern matches

CONFIDENCE_BOOSTERS: Dict[MemoryType, List[str]] = {
    MemoryType.FACT: ["verified", "confirmed", "official", "documented", "according to", "data shows"],
    MemoryType.PREFERENCE: ["always", "never", "absolutely", "definitely", "strongly"],
    MemoryType.DECISION: ["final", "official", "approved", "agreed", "consensus"],
    MemoryType.COMMITMENT: ["promise", "guarantee", "committed", "deadline", "SLA"],
    MemoryType.GOAL: ["target", "objective", "KPI", "OKR", "success metric"],
    MemoryType.EVENT: ["specifically", "exactly", "precisely", "at", "on"],
    MemoryType.INSTRUCTION: ["mandatory", "required", "critical", "important"],
    MemoryType.RELATIONSHIP: ["directly", "reports to", "managed by", "owned by"],
    MemoryType.CONTEXT: ["currently", "right now", "active", "in progress"],
    MemoryType.LEARNING: ["key lesson", "important finding", "critical insight"],
    MemoryType.OBSERVATION: ["consistently", "repeatedly", "over time", "pattern"],
    MemoryType.ERROR: ["critical", "severe", "blocking", "P0", "P1"],
    MemoryType.ARTIFACT: ["official", "canonical", "source of truth", "reference"],
}


def classify_memory(content: str) -> TypeMatch:
    """
    Deterministic, rule-based memory type classification.
    Zero LLM calls. Zero overhead.
    
    Args:
        content: Raw memory text to classify
        
    Returns:
        TypeMatch with memory_type, confidence, matched_pattern, priority
    """
    if not content or not content.strip():
        return TypeMatch(MemoryType.UNKNOWN, 0.0, "", "stable")
    
    content_lower = content.lower()
    best_match = None
    best_score = 0.0
    
    for pattern, mem_type, base_confidence, priority in TYPE_PATTERNS:
        match = re.search(pattern, content_lower, re.IGNORECASE)
        if match:
            # Calculate confidence
            confidence = base_confidence
            
            # Boost for longer matches (more specific)
            match_len = len(match.group(0))
            if match_len > 20:
                confidence += 0.1
            elif match_len > 10:
                confidence += 0.05
            
            # Boost for confidence keywords
            boosters = CONFIDENCE_BOOSTERS.get(mem_type, [])
            for booster in boosters:
                if booster.lower() in content_lower:
                    confidence += 0.05
            
            # Cap at 1.0
            confidence = min(confidence, 1.0)
            
            # Track best match
            score = confidence * (1.0 + 0.1 * list(MemoryType).index(mem_type))
            if score > best_score:
                best_score = score
                best_match = TypeMatch(
                    memory_type=mem_type,
                    confidence=confidence,
                    matched_pattern=pattern,
                    priority=priority
                )
    
    if best_match is None:
        # Default to FACT for short statements, CONTEXT for longer ones
        if len(content.split()) < 5:
            return TypeMatch(MemoryType.FACT, 0.3, "default_short", "stable")
        else:
            return TypeMatch(MemoryType.CONTEXT, 0.3, "default_long", "high")
    
    return best_match


def classify_batch(contents: List[str]) -> List[TypeMatch]:
    """Classify multiple memories efficiently."""
    return [classify_memory(c) for c in contents]


def get_type_priority(memory_type: MemoryType) -> int:
    """
    Get priority ranking for memory types.
    Higher number = more important for retrieval.
    """
    priorities = {
        MemoryType.INSTRUCTION: 10,   # Rules are critical
        MemoryType.COMMITMENT: 9,      # Deadlines are time-critical
        MemoryType.ERROR: 8,            # Avoid failures
        MemoryType.GOAL: 7,             # Objectives guide behavior
        MemoryType.DECISION: 6,         # Choices affect future
        MemoryType.PREFERENCE: 5,       # User tastes matter
        MemoryType.FACT: 4,             # Facts are stable
        MemoryType.RELATIONSHIP: 4,     # Connections are stable
        MemoryType.LEARNING: 3,         # Lessons accumulate
        MemoryType.OBSERVATION: 3,      # Patterns evolve
        MemoryType.EVENT: 2,            # Events decay
        MemoryType.CONTEXT: 2,          # Context is temporal
        MemoryType.ARTIFACT: 1,          # References are pointers
        MemoryType.UNKNOWN: 0,          # Unknown is lowest
    }
    return priorities.get(memory_type, 0)


def should_consolidate(memory_type: MemoryType) -> bool:
    """
    Determine if a memory type should be included in consolidation.
    Some types (events, context) are too temporal to consolidate.
    """
    consolidate_types = {
        MemoryType.FACT,
        MemoryType.PREFERENCE,
        MemoryType.DECISION,
        MemoryType.GOAL,
        MemoryType.LEARNING,
        MemoryType.OBSERVATION,
        MemoryType.RELATIONSHIP,
        MemoryType.INSTRUCTION,
    }
    return memory_type in consolidate_types


def get_decay_rate(memory_type: MemoryType) -> float:
    """
    Get temporal decay rate for a memory type.
    Higher = faster decay.
    """
    decay_rates = {
        MemoryType.CONTEXT: 0.9,      # Very fast decay
        MemoryType.EVENT: 0.7,        # Fast decay
        MemoryType.OBSERVATION: 0.5,    # Moderate decay
        MemoryType.LEARNING: 0.3,       # Slow decay
        MemoryType.PREFERENCE: 0.2,     # Slow decay
        MemoryType.FACT: 0.1,         # Very slow decay
        MemoryType.RELATIONSHIP: 0.1,  # Very slow decay
        MemoryType.INSTRUCTION: 0.05,  # Almost no decay
        MemoryType.ERROR: 0.05,        # Almost no decay
        MemoryType.DECISION: 0.3,       # Moderate decay
        MemoryType.COMMITMENT: 0.5,    # Fast decay (until deadline)
        MemoryType.GOAL: 0.4,         # Moderate decay (until achieved)
        MemoryType.ARTIFACT: 0.1,      # Very slow decay
        MemoryType.UNKNOWN: 0.5,      # Moderate decay
    }
    return decay_rates.get(memory_type, 0.3)


# --- Testing ---
if __name__ == "__main__":
    test_cases = [
        "The API endpoint is at https://api.example.com/v2",
        "I prefer dark mode for all my applications",
        "We decided to go with PostgreSQL instead of MongoDB",
        "I will deliver the report by Friday EOD",
        "Our goal is to reach 10K users by Q4",
        "We had a meeting with the CEO yesterday at 2pm",
        "Always validate user input before processing",
        "Alice manages Bob and reports to Charlie",
        "Currently working on the authentication module",
        "Key lesson: users need simpler onboarding",
        "I noticed traffic peaks every Friday afternoon",
        "Critical bug: null pointer exception in login flow",
        "See the Q3 budget spreadsheet for details",
    ]
    
    print("Typed Memory Classification Tests")
    print("=" * 60)
    for test in test_cases:
        result = classify_memory(test)
        print(f"{result.memory_type.value:13} | {result.confidence:.2f} | {result.priority:12} | {test[:50]}")
    
    print("\n" + "=" * 60)
    print(f"Total patterns: {len(TYPE_PATTERNS)}")
    print(f"Types covered: {len(set(p[1] for p in TYPE_PATTERNS))}")
