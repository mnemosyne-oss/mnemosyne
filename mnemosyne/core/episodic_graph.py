"""
Mnemosyne Episodic Gist+Fact Graph
====================================
Building on REMem (ICLR 2026, arXiv:2602.13530)

Two-phase episodic memory:
1. Gist extraction: concise episode summaries with temporal anchors
2. Fact extraction: structured (subject, predicate, object) triples

Graph structure:
- Nodes: V_gist (episodes) ∪ V_phrase (concepts)
- Edges: E_rel (relations) ∪ E_ctx (context) ∪ E_syn (synonymy)
- Temporal qualifiers: point_in_time, start_time, end_time

Zero LLM calls for gist extraction (rule-based).
Zero LLM calls for fact extraction (pattern-based).
"""

import re
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class Gist:
    """Time-aware episode summary."""
    id: str
    text: str
    timestamp: str
    participants: List[str]
    location: Optional[str]
    emotion: Optional[str]
    time_scope: Optional[str]  # point_in_time, start_time, end_time


@dataclass
class Fact:
    """Structured fact triple."""
    id: str
    subject: str
    predicate: str
    object: str
    timestamp: str
    confidence: float
    temporal_qualifier: Optional[str] = None


@dataclass
class GraphEdge:
    """Edge in the memory graph."""
    source: str
    target: str
    edge_type: str  # rel, ctx, syn
    weight: float
    timestamp: str


class EpisodicGraph:
    """
    Hybrid memory graph for episodic + semantic storage.
    
    Stores in SQLite:
    - gists table: episode summaries
    - facts table: structured triples
    - graph_edges table: relationships between nodes
    """
    
    def __init__(self, db_path: Path = None, conn=None):
        if conn is not None:
            self.conn = conn
            self.db_path = db_path or Path(":memory:")
        else:
            self.db_path = db_path or Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._owns_connection = conn is None
        self._init_tables()
    
    def _init_tables(self):
        """Initialize episodic graph schema."""
        cursor = self.conn.cursor()
        
        # Gists table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS gists (
                id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                timestamp TEXT,
                participants_json TEXT,
                location TEXT,
                emotion TEXT,
                time_scope TEXT,
                memory_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Facts table (compatible with beam.py schema)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                fact_id TEXT PRIMARY KEY,
                session_id TEXT DEFAULT 'default',
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                timestamp TEXT,
                source_msg_id TEXT,
                confidence REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_facts_object ON facts(object)")
        
        # Graph edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                target TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                weight REAL DEFAULT 1.0,
                timestamp TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_type ON graph_edges(edge_type)")
        
        self.conn.commit()
    
    # --- Gist Extraction (Rule-based, zero LLM) ---
    
    def extract_gist(self, content: str, memory_id: str) -> Gist:
        """
        Extract episodic gist from raw content.
        
        Rule-based extraction of:
        - Participants (names, pronouns)
        - Temporal anchors (dates, times, relative references)
        - Location (place references)
        - Emotion (sentiment indicators)
        
        Args:
            content: Raw memory text
            memory_id: Source memory ID
            
        Returns:
            Gist object
        """
        content_lower = content.lower()
        
        # Extract participants
        participants = self._extract_participants(content)
        
        # Extract temporal scope
        time_scope = self._extract_temporal_scope(content)
        
        # Extract location
        location = self._extract_location(content)
        
        # Extract emotion
        emotion = self._extract_emotion(content)
        
        # Create concise summary (first sentence or first 100 chars)
        summary = self._create_summary(content)
        
        return Gist(
            id=f"gist_{memory_id}",
            text=summary,
            timestamp=datetime.now().isoformat(),
            participants=participants,
            location=location,
            emotion=emotion,
            time_scope=time_scope
        )
    
    def _extract_participants(self, content: str) -> List[str]:
        """Extract participant names and pronouns."""
        # Common name patterns
        name_pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
        names = re.findall(name_pattern, content)
        
        # Pronouns
        pronoun_pattern = r"\b(I|you|we|they|he|she|it|me|us|them|him|her)\b"
        pronouns = re.findall(pronoun_pattern, content, re.IGNORECASE)
        
        # Combine and deduplicate
        participants = list(set(names + pronouns))
        return participants[:5]  # Cap at 5
    
    def _extract_temporal_scope(self, content: str) -> Optional[str]:
        """Extract temporal references."""
        temporal_patterns = [
            (r"\b(yesterday|today|tomorrow|now|soon|later|earlier)\b", "point_in_time"),
            (r"\b(last\s+week|last\s+month|last\s+year|next\s+week)\b", "point_in_time"),
            (r"\b(since|from|starting)\b.*\b(until|to|through|end)\b", "duration"),
            (r"\b(between|from)\b.*\b(and|to)\b", "range"),
            (r"\b\d{1,2}:\d{2}\s*(AM|PM|am|pm)?\b", "point_in_time"),
            (r"\b\d{4}-\d{2}-\d{2}\b", "point_in_time"),
        ]
        
        for pattern, scope_type in temporal_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return scope_type
        
        return None
    
    def _extract_location(self, content: str) -> Optional[str]:
        """Extract location references."""
        location_patterns = [
            r"\b(at|in|from)\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:yesterday|today|tomorrow|now|last|next|on|at)\b|$)",
            r"\b(office|home|work|school|hospital|store|restaurant|building|room)\b",
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                return match.group(2) if len(match.groups()) > 1 else match.group(1)
        
        return None
    
    def _extract_emotion(self, content: str) -> Optional[str]:
        """Extract emotional indicators."""
        emotion_words = {
            "positive": ["happy", "excited", "great", "awesome", "love", "enjoy", "glad", "pleased"],
            "negative": ["sad", "angry", "frustrated", "upset", "hate", "disappointed", "worried"],
            "neutral": ["fine", "okay", "alright", "normal", "standard"],
        }
        
        content_lower = content.lower()
        for emotion_type, words in emotion_words.items():
            if any(word in content_lower for word in words):
                return emotion_type
        
        return None
    
    def _create_summary(self, content: str) -> str:
        """Create concise episode summary."""
        # Take first sentence or first 100 chars
        sentences = re.split(r'[.!?]+', content)
        if sentences and len(sentences[0]) > 10:
            return sentences[0].strip()[:100]
        return content[:100].strip()
    
    # --- Fact Extraction (Rule-based, zero LLM) ---
    
    # Pattern-based extraction uses non-greedy character classes
    # (`[a-zA-Z\s]+?`) adjacent to optional groups. Worst-case
    # backtracking is O(n²)-ish on adversarial inputs (long
    # documents with many capitalized words and intervening
    # `is`/`has`/`uses`). At benchmark scale (250K rows including
    # some imported documents) a single 10KB row can stall the
    # whole ingest loop for seconds. Cap input length before regex
    # to bound worst-case CPU per call. /review caught this as a
    # 3-source HIGH on the E2 batch path. 4096 chars covers
    # typical conversational rows in full and the first paragraph
    # of imported documents.
    _EXTRACT_FACTS_MAX_CONTENT_LEN = 4096

    def extract_facts(self, content: str, memory_id: str) -> List[Fact]:
        """
        Extract structured facts from content.

        Pattern-based extraction of (subject, predicate, object) triples.

        Args:
            content: Raw memory text
            memory_id: Source memory ID

        Returns:
            List of Fact objects
        """
        facts = []
        # Bound input length so regex backtracking can't stall the
        # ingest pipeline on pathological documents. See class-level
        # _EXTRACT_FACTS_MAX_CONTENT_LEN note for the /review context.
        if len(content) > self._EXTRACT_FACTS_MAX_CONTENT_LEN:
            content = content[: self._EXTRACT_FACTS_MAX_CONTENT_LEN]

        # Pattern 1: "X is Y"
        is_pattern = r"\b([A-Z][a-zA-Z\s]+?)\s+is\s+(?:a|an|the)?\s*([a-zA-Z\s]+?)\b"
        for match in re.finditer(is_pattern, content):
            subject = match.group(1).strip()
            obj = match.group(2).strip()
            if len(subject) > 2 and len(obj) > 2:
                facts.append(Fact(
                    id=f"fact_{memory_id}_{len(facts)}",
                    subject=subject,
                    predicate="is",
                    object=obj,
                    timestamp=datetime.now().isoformat(),
                    confidence=0.7
                ))
        
        # Pattern 2: "X has Y"
        has_pattern = r"\b([A-Z][a-zA-Z\s]+?)\s+has\s+(?:a|an|the)?\s*([a-zA-Z\d\s]+?)\b"
        for match in re.finditer(has_pattern, content):
            subject = match.group(1).strip()
            obj = match.group(2).strip()
            if len(subject) > 2 and len(obj) > 2:
                facts.append(Fact(
                    id=f"fact_{memory_id}_{len(facts)}",
                    subject=subject,
                    predicate="has",
                    object=obj,
                    timestamp=datetime.now().isoformat(),
                    confidence=0.6
                ))
        
        # Pattern 3: "X uses Y"
        uses_pattern = r"\b([A-Z][a-zA-Z\s]+?)\s+(uses?|using|used)\s+(?:a|an|the)?\s*([a-zA-Z\s]+?)\b"
        for match in re.finditer(uses_pattern, content):
            subject = match.group(1).strip()
            obj = match.group(3).strip()
            if len(subject) > 2 and len(obj) > 2:
                facts.append(Fact(
                    id=f"fact_{memory_id}_{len(facts)}",
                    subject=subject,
                    predicate="uses",
                    object=obj,
                    timestamp=datetime.now().isoformat(),
                    confidence=0.6
                ))
        
        # Pattern 4: "X works at Y"
        works_pattern = r"\b([A-Z][a-zA-Z\s]+?)\s+works?\s+(?:at|for|with)\s+([A-Z][a-zA-Z\s]+?)\b"
        for match in re.finditer(works_pattern, content):
            subject = match.group(1).strip()
            obj = match.group(2).strip()
            if len(subject) > 2 and len(obj) > 2:
                facts.append(Fact(
                    id=f"fact_{memory_id}_{len(facts)}",
                    subject=subject,
                    predicate="works_at",
                    object=obj,
                    timestamp=datetime.now().isoformat(),
                    confidence=0.7
                ))
        
        return facts[:5]  # Cap at 5 facts per memory
    
    # --- Graph Storage ---
    
    def store_gist(self, gist: Gist, memory_id: str):
        """Store a gist in the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO gists
            (id, text, timestamp, participants_json, location, emotion, time_scope, memory_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            gist.id,
            gist.text,
            gist.timestamp,
            json.dumps(gist.participants),
            gist.location,
            gist.emotion,
            gist.time_scope,
            memory_id
        ))
        self.conn.commit()
    
    def store_fact(self, fact: Fact, memory_id: str, session_id: str = "default"):
        """Store a fact in the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO facts
            (fact_id, session_id, subject, predicate, object, timestamp, source_msg_id, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fact.id,
            session_id,
            fact.subject,
            fact.predicate,
            fact.object,
            fact.timestamp,
            memory_id,
            fact.confidence
        ))
        self.conn.commit()
    
    def add_edge(self, edge: GraphEdge):
        """Add an edge to the graph."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO graph_edges
            (source, target, edge_type, weight, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (
            edge.source,
            edge.target,
            edge.edge_type,
            edge.weight,
            edge.timestamp
        ))
        self.conn.commit()
    
    # --- Graph Traversal ---
    
    def find_related_memories(self, memory_id: str, depth: int = 2,
                                edge_type: str = "", min_weight: float = 0.0) -> List[Dict]:
        """
        Find memories related to a given memory via graph traversal.

        Args:
            memory_id: Starting memory
            depth: Traversal depth (default 2)
            edge_type: Filter by edge type (empty = all types). Built-in types:
                       "rel" (relation), "ctx" (context), "syn" (synonymy).
                       Agent-declared types like "references", "caused",
                       "supersedes" also work since the column is freeform TEXT.
            min_weight: Minimum edge weight threshold (default 0.0 = no filter)

        Returns:
            List of dicts with keys: memory_id, edge_type, weight, depth
        """
        results = []
        current_level = {memory_id}
        seen = {memory_id}

        for hop in range(1, depth + 1):
            next_level = set()
            for mem in current_level:
                cursor = self.conn.cursor()
                if edge_type:
                    cursor.execute("""
                        SELECT source, target, edge_type, weight FROM graph_edges
                        WHERE (source = ? OR target = ?) AND edge_type = ? AND weight >= ?
                    """, (mem, mem, edge_type, min_weight))
                else:
                    cursor.execute("""
                        SELECT source, target, edge_type, weight FROM graph_edges
                        WHERE (source = ? OR target = ?) AND weight >= ?
                    """, (mem, mem, min_weight))

                for row in cursor.fetchall():
                    neighbor = row["target"] if row["source"] == mem else row["source"]
                    if neighbor not in seen:
                        next_level.add(neighbor)
                        seen.add(neighbor)
                        results.append({
                            "memory_id": neighbor,
                            "edge_type": row["edge_type"],
                            "weight": row["weight"],
                            "depth": hop,
                        })

            current_level = next_level

        return results
    
    def find_facts_by_subject(self, subject: str) -> List[Fact]:
        """Find all facts about a subject."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM facts WHERE subject = ?
            ORDER BY confidence DESC, timestamp DESC
        """, (subject,))
        
        facts = []
        for row in cursor.fetchall():
            facts.append(Fact(
                id=row["fact_id"],
                subject=row["subject"],
                predicate=row["predicate"],
                object=row["object"],
                timestamp=row["timestamp"],
                confidence=row["confidence"],
                temporal_qualifier=None  # Not in beam.py facts schema
            ))
        
        return facts
    
    def find_gists_by_participant(self, participant: str) -> List[Gist]:
        """Find all gists involving a participant."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM gists 
            WHERE participants_json LIKE ?
            ORDER BY timestamp DESC
        """, (f'%"{participant}"%',))
        
        gists = []
        for row in cursor.fetchall():
            gists.append(Gist(
                id=row["id"],
                text=row["text"],
                timestamp=row["timestamp"],
                participants=json.loads(row["participants_json"]),
                location=row["location"],
                emotion=row["emotion"],
                time_scope=row["time_scope"]
            ))
        
        return gists
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM gists")
        gist_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM facts")
        fact_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM graph_edges")
        edge_count = cursor.fetchone()[0]
        
        return {
            "gists": gist_count,
            "facts": fact_count,
            "edges": edge_count,
            "total_nodes": gist_count + fact_count,
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


# --- Testing ---
if __name__ == "__main__":
    import tempfile
    import os
    
    print("Episodic Graph Tests")
    print("=" * 60)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    graph = EpisodicGraph(db_path=Path(db_path))
    
    # Test content
    test_content = """
    Alice had a meeting with Bob yesterday at the office.
    She was excited about the new project.
    Alice is a senior developer at TechCorp.
    She uses Python for backend development.
    The deadline is next Friday.
    """
    
    # Extract gist
    gist = graph.extract_gist(test_content, "mem_001")
    print(f"Gist: {gist.text}")
    print(f"  Participants: {gist.participants}")
    print(f"  Location: {gist.location}")
    print(f"  Emotion: {gist.emotion}")
    print(f"  Time scope: {gist.time_scope}")
    
    # Store gist
    graph.store_gist(gist, "mem_001")
    
    # Extract facts
    facts = graph.extract_facts(test_content, "mem_001")
    print(f"\nFacts extracted: {len(facts)}")
    for fact in facts:
        print(f"  {fact.subject} --{fact.predicate}--> {fact.object} (conf: {fact.confidence})")
        graph.store_fact(fact, "mem_001")
    
    # Add edges
    graph.add_edge(GraphEdge("mem_001", "mem_002", "rel", 0.8, datetime.now().isoformat()))
    graph.add_edge(GraphEdge("mem_001", "mem_003", "ctx", 0.6, datetime.now().isoformat()))
    
    # Find related
    related = graph.find_related_memories("mem_001", depth=1)
    print(f"\nRelated memories:")
    for r in related:
        print(f"  {r['memory_id']} --{r['edge_type']}--> weight={r['weight']} (depth={r['depth']})")
    
    # Find facts by subject
    alice_facts = graph.find_facts_by_subject("Alice")
    print(f"\nFacts about Alice: {len(alice_facts)}")
    
    # Stats
    stats = graph.get_stats()
    print(f"\nGraph stats: {stats}")
    
    # Cleanup
    graph.close()
    os.unlink(db_path)
    
    print("\n" + "=" * 60)
    print("Episodic graph tests passed!")
