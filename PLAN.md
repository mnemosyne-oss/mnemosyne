# Mnemosyne Temporal Upgrade Plan

## Noxem Features to Port

### 1. Weibull Decay Scoring (`weibull.py`)
- Per-memory-type eta/k parameters (replaces uniform exponential halflife)
- Integration: replace `_temporal_boost()` call in `beam.py` recall scoring
- Backward compatible: when memory_type is None, fall back to exponential

### 2. MMR Re-Ranking (`mmr.py`)
- Maximal Marginal Relevance diversity re-ranking
- Post-processing step after recall results are collected
- Lambda parameter controls diversity vs relevance tradeoff
- Integration: call after all recall results are merged, before final sort

### 3. Query Intent Classification (`query_intent.py`)
- Regex-based classification: temporal, factual, entity, preference, procedural
- Adjusts vec_weight/fts_weight based on intent
- Integration: called at start of recall(), modifies scoring weights

### 4. Synonym Expansion (`synonyms.py`)
- Concept group lookup table (~40 groups)
- Expands query words with synonyms before FTS5 + vector search
- Tier 1 exact match normalization for cache
- Integration: called at start of recall() before search

### 5. 5-Tier Semantic Query Cache (`query_cache.py`)
- Tier 1: Exact normalized match (hash map)
- Tier 2: High-confidence embedding (cosine >= 0.88)
- Tier 3: Composite match (cosine >= 0.78 + Jaccard >= 0.15)
- Tier 4: Expanded query match
- Tier 5: Full search (compute + cache)
- Cache invalidated on every remember() call
- Integration: wraps recall(), stores cache in SQLite

### 6. Temporal Architecture (`temporal_parser.py` + schema)
- NL date parser: "last Monday", "2 days ago", "next week", etc.
- Schema: event_date, event_date_precision, temporal_tags columns
- Correction chains: corrected_by FK, correction protocol
- Integration: parse text on remember(), query via temporal filters

### 7. Associative Retrieval Integration
- Mnemosyne already has EpisodicGraph.find_related_memories()
- Hook: after recall() returns, optionally traverse graph for related results
- Integration: new `associative_depth` parameter on recall()

## Implementation Order
1. `synonyms.py` — simplest, no schema changes
2. `query_intent.py` — simple regex module
3. `weibull.py` — formula swap
4. `mmr.py` — post-processing
5. `query_cache.py` — caching layer
6. `temporal_parser.py` + schema migration — most complex
7. Integration into beam.py
8. Tests
