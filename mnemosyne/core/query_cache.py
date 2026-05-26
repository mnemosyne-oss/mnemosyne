"""
5-Tier Semantic Query Cache for Mnemosyne
==========================================
Caches recall results to avoid recomputing embeddings + FTS5 + scoring
for repeated or similar queries.

Architecture:
    Tier 1: Exact normalized match (hash map, O(1))
    Tier 2: High-confidence embedding (cosine ≥ 0.88)
    Tier 3: Composite match (cosine ≥ 0.78 + keyword Jaccard ≥ 0.15)
    Tier 4: Expanded query match (synonym-expanded version)
    Tier 5: Full search (compute + cache all tiers for next time)

Cache is invalidated on every remember() call via version counter.

Usage:
    from mnemosyne.core.query_cache import QueryCache
    
    cache = QueryCache(db_path=...)
    
    # Try cache before recall
    cached = cache.get(query)
    if cached:
        return cached
    
    # ... compute recall results ...
    
    # Store in cache
    cache.put(query, results, embedding)
"""

import hashlib
import json
import math
import time
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import sqlite3


class QueryCache:
    """
    5-tier semantic query cache for recall results.
    Backed by SQLite for persistence.
    """
    
    def __init__(self, db_path: Path = None, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Args:
            db_path: Path to cache database (None = in-memory only)
            max_size: Maximum cache entries
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache_version = 0  # Incremented on invalidation
        
        # Tier 1: O(1) hash map for exact normalized match
        self._tier1: Dict[str, List[Dict]] = {}
        
        # Tier 2-3: Require embedding comparison
        self._tier2_3: Dict[str, tuple] = {}  # normalized_query -> (embedding, results, timestamp)
        
        # Tier 4: Expanded query cache
        self._tier4: Dict[str, List[Dict]] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        self._insert_times: Dict[str, float] = {}  # normalized -> insert time for TTL
        
        # Stats
        self.hits = 0
        self.misses = 0
        self.tier1_hits = 0
        self.tier2_hits = 0
        self.tier3_hits = 0
        self.tier4_hits = 0
        
        # Optional SQLite backing
        self._db_path = db_path
        if db_path:
            self._init_db()
        else:
            self._conn = None
    
    def _init_db(self):
        """Initialize SQLite cache database."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                normalized TEXT PRIMARY KEY,
                embedding_json TEXT,
                results_json TEXT,
                hit_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_hit TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_hits ON query_cache(hit_count DESC)")
        self._conn.commit()
        
        # Load existing cache entries from SQLite into memory
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT normalized, results_json FROM query_cache")
            for row in cursor.fetchall():
                try:
                    results = json.loads(row["results_json"])
                    self._tier1[row["normalized"]] = results
                    self._tier4[row["normalized"]] = results
                except Exception:
                    pass
        except Exception:
            pass
    
    def invalidate(self):
        """Invalidate all cached queries. Call after any remember() operation."""
        with self._lock:
            self._cache_version += 1
            self._tier1.clear()
            self._tier2_3.clear()
            self._tier4.clear()
            self._insert_times.clear()
            if self._conn:
                self._conn.execute("DELETE FROM query_cache")
                self._conn.commit()
    
    def _normalize(self, query: str) -> str:
        """Normalize query for cache key (consistent hashing)."""
        # Simple normalization: lowercase, sort words, remove very short words
        words = sorted(w.lower() for w in query.split() if len(w) > 1)
        return " ".join(words)
    
    def _cosine_similarity(self, emb_a: List[float], emb_b: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        if not emb_a or not emb_b:
            return 0.0
        if len(emb_a) != len(emb_b):
            # Pad shorter
            max_len = max(len(emb_a), len(emb_b))
            a = emb_a + [0.0] * (max_len - len(emb_a))
            b = emb_b + [0.0] * (max_len - len(emb_b))
        else:
            a, b = emb_a, emb_b
        
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))
        
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)
    
    def _jaccard_words(self, query_a: str, query_b: str) -> float:
        """Word-level Jaccard similarity."""
        words_a = set(query_a.lower().split())
        words_b = set(query_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)
    
    def get(self, query: str, embedding: List[float] = None) -> Optional[List[Dict]]:
        """
        Try to retrieve cached results for a query.
        
        Args:
            query: The search query
            embedding: Pre-computed embedding of the query (for Tier 2-3 comparison)
            
        Returns:
            Cached results or None if cache miss
        """
        normalized = self._normalize(query)
        
        with self._lock:
            # Enforce TTL
            now = time.time()
            if normalized in self._insert_times:
                age = now - self._insert_times[normalized]
                if age > self.ttl_seconds:
                    # Expired — remove from all tiers
                    self._tier1.pop(normalized, None)
                    self._tier2_3.pop(normalized, None)
                    self._tier4.pop(normalized, None)
                    self._insert_times.pop(normalized, None)
                    self.misses += 1
                    return None
            
            # Tier 1: Exact normalized match
            if normalized in self._tier1:
                self.hits += 1
                self.tier1_hits += 1
                return self._tier1[normalized]
            
            # Check Tier 2-3 if embedding is provided
            if embedding:
                best_score = 0.0
                best_key = None
                
                for cached_key, (cached_emb, cached_results, _) in list(self._tier2_3.items()):
                    # TTL check for tier 2-3 entries
                    if cached_key in self._insert_times:
                        if now - self._insert_times[cached_key] > self.ttl_seconds:
                            continue
                    
                    cosine = self._cosine_similarity(embedding, cached_emb)
                    
                    # Tier 2: High confidence embedding match
                    if cosine >= 0.88:
                        best_score = cosine
                        best_key = cached_key
                        break  # High enough, take it
                    
                    # Tier 3: Composite match
                    if cosine >= 0.78:
                        jaccard = self._jaccard_words(query, cached_key)
                        if jaccard >= 0.15 and cosine > best_score:
                            best_score = cosine
                            best_key = cached_key
                
                if best_key:
                    self.hits += 1
                    if best_score >= 0.88:
                        self.tier2_hits += 1
                    else:
                        self.tier3_hits += 1
                    return self._tier2_3[best_key][1]
            
            # Tier 4: Try synonym-expanded version (best-effort)
            query_words = set(normalized.split())
            for cached_key, results in list(self._tier4.items()):
                # TTL check
                if cached_key in self._insert_times:
                    if now - self._insert_times[cached_key] > self.ttl_seconds:
                        continue
                cached_words = set(cached_key.split())
                overlap = len(query_words & cached_words)
                if overlap >= len(query_words) * 0.7 and overlap >= 2:
                    self.hits += 1
                    self.tier4_hits += 1
                    return results
            
            self.misses += 1
            return None
    
    def put(self, query: str, results: List[Dict], embedding: List[float] = None):
        """
        Store results in all applicable cache tiers.
        """
        normalized = self._normalize(query)
        
        with self._lock:
            # Tier 1
            self._tier1[normalized] = results
            self._insert_times[normalized] = time.time()
            
            # Tier 2-3
            if embedding:
                self._tier2_3[normalized] = (embedding, results, time.time())
            
            # Tier 4
            self._tier4[normalized] = results
            
            # SQLite backing
            if self._conn:
                try:
                    self._conn.execute(
                        "INSERT OR REPLACE INTO query_cache (normalized, embedding_json, results_json) VALUES (?, ?, ?)",
                        (normalized, json.dumps(embedding) if embedding else None, json.dumps(results))
                    )
                    self._conn.commit()
                except Exception:
                    pass
            
            # Evict if over max_size
            self._evict_if_needed()
    
    def _evict_if_needed(self):
        """LRU eviction if cache exceeds max_size. Also cleans TTL-expired entries."""
        # First, clean up any TTL-expired entries
        now = time.time()
        expired = [
            k for k, t in list(self._insert_times.items())
            if now - t > self.ttl_seconds
        ]
        for key in expired:
            self._tier1.pop(key, None)
            self._tier2_3.pop(key, None)
            self._tier4.pop(key, None)
            self._insert_times.pop(key, None)
        
        # Then evict oldest if still over max_size
        total = len(self._tier1)
        if total > self.max_size:
            sorted_keys = sorted(
                self._insert_times.keys(),
                key=lambda k: self._insert_times.get(k, 0)
            )
            to_remove = sorted_keys[:total - self.max_size]
            for key in to_remove:
                self._tier1.pop(key, None)
                self._tier2_3.pop(key, None)
                self._tier4.pop(key, None)
                self._insert_times.pop(key, None)
    
    def close(self):
        """Close the SQLite connection and release resources."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
    
    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        self.close()
    
    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 - 1.0)."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def stats(self) -> Dict:
        """Return cache statistics."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 3),
            "tier1_hits": self.tier1_hits,
            "tier2_hits": self.tier2_hits,
            "tier3_hits": self.tier3_hits,
            "tier4_hits": self.tier4_hits,
            "size": len(self._tier1),
            "max_size": self.max_size,
            "version": self._cache_version,
        }
