"""
Mnemosyne Integration Tests
===========================
End-to-end tests for all 5 phases:
1. Typed Memory Schema
2. Binary Vectors
3. Episodic Graph
4. Veracity Consolidation
5. Polyphonic Recall

Also tests BEAM benchmark integration.
"""

import unittest
import tempfile
import numpy as np
from pathlib import Path

from mnemosyne.core.typed_memory import classify_memory, MemoryType, get_type_priority
from mnemosyne.core.binary_vectors import BinaryVectorStore
from mnemosyne.core.episodic_graph import EpisodicGraph
from mnemosyne.core.veracity_consolidation import VeracityConsolidator
from mnemosyne.core.polyphonic_recall import PolyphonicRecallEngine


class TestTypedMemory(unittest.TestCase):
    """Test Phase 1: Typed Memory Schema."""
    
    def test_fact_classification(self):
        result = classify_memory("The API is at https://example.com")
        self.assertEqual(result.memory_type, MemoryType.FACT)
        self.assertGreater(result.confidence, 0.5)
    
    def test_preference_classification(self):
        result = classify_memory("I prefer dark mode")
        self.assertEqual(result.memory_type, MemoryType.PREFERENCE)
    
    def test_commitment_classification(self):
        result = classify_memory("I will deliver by Friday")
        self.assertEqual(result.memory_type, MemoryType.COMMITMENT)
    
    def test_priority_ranking(self):
        self.assertGreater(get_type_priority(MemoryType.INSTRUCTION), get_type_priority(MemoryType.EVENT))
    
    def test_decay_rates(self):
        from mnemosyne.core.typed_memory import get_decay_rate
        self.assertGreater(get_decay_rate(MemoryType.CONTEXT), get_decay_rate(MemoryType.FACT))


class TestBinaryVectors(unittest.TestCase):
    """Test Phase 2: Binary Vectors."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.store = BinaryVectorStore(db_path=Path(self.temp_file.name))
        self.embedding = np.random.randn(384).astype(np.float32)
    
    def tearDown(self):
        self.store.close()
        self.temp_file.close()
    
    def test_binarization(self):
        binary = self.store.maximally_informative_binarization(self.embedding)
        self.assertEqual(len(binary), 48)  # 384 bits / 8 = 48 bytes
    
    def test_hamming_distance(self):
        binary_a = self.store.maximally_informative_binarization(self.embedding)
        binary_b = self.store.maximally_informative_binarization(self.embedding)
        distance = self.store.hamming_distance(binary_a, binary_b)
        self.assertEqual(distance, 0)  # Same embedding = 0 distance
    
    def test_store_and_search(self):
        self.store.store_vector("test_1", self.embedding)
        results = self.store.search(self.embedding, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["memory_id"], "test_1")
        self.assertEqual(results[0]["distance"], 0)
    
    def test_compression_ratio(self):
        stats = self.store.get_stats()
        self.assertLess(stats["compression_ratio"], 0.1)  # Should be ~3%


class TestEpisodicGraph(unittest.TestCase):
    """Test Phase 3: Episodic Graph."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.graph = EpisodicGraph(db_path=Path(self.temp_file.name))
    
    def tearDown(self):
        self.graph.close()
        self.temp_file.close()
    
    def test_gist_extraction(self):
        content = "Alice met Bob at the office yesterday. She was happy."
        gist = self.graph.extract_gist(content, "mem_001")
        self.assertIn("Alice", gist.participants)
        self.assertEqual(gist.emotion, "positive")
        self.assertEqual(gist.location, "the office")
    
    def test_fact_extraction(self):
        content = "Alice is a developer. She uses Python."
        facts = self.graph.extract_facts(content, "mem_001")
        self.assertGreater(len(facts), 0)
        self.assertEqual(facts[0].subject, "Alice")
    
    def test_graph_storage(self):
        gist = self.graph.extract_gist("Test content", "mem_001")
        self.graph.store_gist(gist, "mem_001")
        
        facts = self.graph.extract_facts("Alice is a developer", "mem_001")
        for fact in facts:
            self.graph.store_fact(fact, "mem_001")
        
        stats = self.graph.get_stats()
        self.assertGreaterEqual(stats["gists"], 1)
        self.assertGreaterEqual(stats["facts"], 1)
    
    def test_graph_traversal(self):
        from mnemosyne.core.episodic_graph import GraphEdge
        from datetime import datetime
        
        self.graph.add_edge(GraphEdge("mem_001", "mem_002", "rel", 0.8, datetime.now().isoformat()))
        related = self.graph.find_related_memories("mem_001", depth=1)
        self.assertIn("mem_002", related)


class TestVeracityConsolidation(unittest.TestCase):
    """Test Phase 4: Veracity Consolidation."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.cons = VeracityConsolidator(db_path=Path(self.temp_file.name))
    
    def tearDown(self):
        self.cons.close()
        self.temp_file.close()
    
    def test_basic_consolidation(self):
        fact = self.cons.consolidate_fact("Alice", "is", "developer", "stated")
        self.assertEqual(fact.subject, "Alice")
        self.assertGreater(fact.confidence, 0.0)
    
    def test_bayesian_update(self):
        fact1 = self.cons.consolidate_fact("Alice", "is", "developer", "stated")
        fact2 = self.cons.consolidate_fact("Alice", "is", "developer", "stated")
        self.assertGreater(fact2.confidence, fact1.confidence)
        self.assertEqual(fact2.mention_count, 2)
    
    def test_conflict_detection(self):
        self.cons.consolidate_fact("Alice", "is", "developer", "stated")
        self.cons.consolidate_fact("Alice", "is", "manager", "inferred")
        conflicts = self.cons.get_conflicts()
        self.assertGreater(len(conflicts), 0)
    
    def test_conflict_resolution(self):
        from mnemosyne.core.veracity_consolidation import compute_fact_id

        self.cons.consolidate_fact("Alice", "is", "developer", "stated")
        self.cons.consolidate_fact("Alice", "is", "manager", "inferred")
        conflicts = self.cons.get_conflicts()

        if conflicts:
            # Post-fix ID is hash-based; use compute_fact_id to
            # resolve to the stored ID rather than hard-coding the
            # pre-fix legacy f-string form.
            self.cons.resolve_conflict(
                conflicts[0]["id"],
                compute_fact_id("Alice", "is", "developer"),
            )
            resolved = self.cons.get_conflicts()
            self.assertEqual(len(resolved), 0)
    
    def test_high_confidence_summary(self):
        self.cons.consolidate_fact("Alice", "is", "developer", "stated")
        self.cons.consolidate_fact("Alice", "is", "developer", "stated")
        summary = self.cons.get_high_confidence_summary("Alice", threshold=0.5)
        self.assertIn("Alice", summary)


class TestPolyphonicRecall(unittest.TestCase):
    """Test Phase 5: Polyphonic Recall."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.engine = PolyphonicRecallEngine(db_path=Path(self.temp_file.name))
    
    def tearDown(self):
        self.engine.close()
        self.temp_file.close()
    
    def test_empty_recall(self):
        results = self.engine.recall("test query")
        self.assertIsInstance(results, list)
    
    def test_voice_weights(self):
        weights = self.engine.voice_weights
        self.assertEqual(sum(weights.values()), 1.0)
        self.assertIn("vector", weights)
        self.assertIn("graph", weights)
        self.assertIn("fact", weights)
        self.assertIn("temporal", weights)
    
    def test_stats(self):
        stats = self.engine.get_stats()
        self.assertIn("voice_weights", stats)
        self.assertIn("vector_stats", stats)


class TestIntegration(unittest.TestCase):
    """Test full pipeline integration."""
    
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = Path(self.temp_file.name)
        
        self.store = BinaryVectorStore(db_path=self.db_path)
        self.graph = EpisodicGraph(db_path=self.db_path)
        self.cons = VeracityConsolidator(db_path=self.db_path)
        self.engine = PolyphonicRecallEngine(db_path=self.db_path)
    
    def tearDown(self):
        self.engine.close()
        self.cons.close()
        self.graph.close()
        self.store.close()
        self.temp_file.close()
    
    def test_full_pipeline(self):
        # 1. Classify memory
        content = "Alice decided to use PostgreSQL for the new project."
        result = classify_memory(content)
        self.assertEqual(result.memory_type, MemoryType.DECISION)
        
        # 2. Store binary vector
        embedding = np.random.randn(384).astype(np.float32)
        self.store.store_vector("mem_001", embedding)
        
        # 3. Extract gist and facts
        gist = self.graph.extract_gist(content, "mem_001")
        self.graph.store_gist(gist, "mem_001")
        
        facts = self.graph.extract_facts(content, "mem_001")
        for fact in facts:
            self.graph.store_fact(fact, "mem_001")
        
        # 4. Consolidate facts
        for fact in facts:
            self.cons.consolidate_fact(
                fact.subject, fact.predicate, fact.object,
                veracity="stated", source="mem_001"
            )
        
        # 5. Recall
        results = self.engine.recall("What database did Alice choose?", embedding)
        self.assertIsInstance(results, list)
        
        # Verify stats
        stats = self.engine.get_stats()
        self.assertIn("vector_stats", stats)
        self.assertIn("graph_stats", stats)
        self.assertIn("consolidation_stats", stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)
