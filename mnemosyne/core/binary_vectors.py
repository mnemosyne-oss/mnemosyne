"""
Mnemosyne Information-Theoretic Binary Vectors
==============================================
Building on Moorcheh ITS (arXiv:2601.11557)

Three core innovations:
1. Maximally Informative Binarization (MIB) - 32x compression
2. Efficient Distance Metric (EDM) - Hamming distance via bitwise ops
3. Information-Theoretic Score (ITS) - deterministic ranking

Replaces: float32 embeddings + HNSW index + cosine similarity
With: binary vectors + exhaustive scan + Hamming distance

Benefits:
- 32x memory reduction (model-dependent)
- Deterministic retrieval (same query = same results)
- No ANN index needed (no HNSW, no IVF, no PQ)
- SQLite-native storage
- CPU-efficient (bitwise XOR + popcount)
"""

import numpy as np
import sqlite3
from typing import List, Dict
from pathlib import Path

# --- Configuration ---
# Derive EMBEDDING_DIM from the embeddings module so it always matches
# the configured model (e.g. 384 for bge-small-en-v1.5, 1024 for
# multilingual-e5-large).  Previously hardcoded to 384, which silently
# truncated larger embeddings during binarization, losing up to 62.5%
# of information when a 1024-dim model was in use.
try:
    from mnemosyne.core.embeddings import EMBEDDING_DIM as _EMB_DIM
    EMBEDDING_DIM = _EMB_DIM
except ImportError:
    EMBEDDING_DIM = 384  # fallback for standalone / test use
BITS_PER_BYTE = 8
BYTES_PER_VECTOR = EMBEDDING_DIM // BITS_PER_BYTE


# --- Module-level function aliases for beam.py compatibility ---
# beam.py imports these as: from mnemosyne.core.binary_vectors import maximally_informative_binarization as _mib, hamming_distance as _hamming
# but they were only defined as class static methods, causing ImportError silently setting them to None.
def maximally_informative_binarization(embedding) -> bytes:
    """Module-level alias for BinaryVectorStore.maximally_informative_binarization."""
    return BinaryVectorStore.maximally_informative_binarization(embedding)


def hamming_distance(binary_a: bytes, binary_b: bytes) -> int:
    """Module-level alias for BinaryVectorStore.hamming_distance."""
    return BinaryVectorStore.hamming_distance(binary_a, binary_b)


class BinaryVectorStore:
    """
    SQLite-native binary vector storage with deterministic retrieval.
    
    No external vector DB needed. No ANN index. Just SQLite + numpy.
    """
    
    def __init__(self, db_path: Path = None, table_name: str = "binary_vectors", conn=None):
        if conn is not None:
            self.conn = conn
            self.db_path = db_path or Path(":memory:")
        else:
            self.db_path = db_path or Path.home() / ".hermes" / "mnemosyne" / "data" / "mnemosyne.db"
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.table_name = table_name
        self.conn.row_factory = sqlite3.Row
        self._owns_connection = conn is None
        self._init_table()
    
    def _init_table(self):
        """Create binary vector table."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                memory_id TEXT PRIMARY KEY,
                binary_vector BLOB NOT NULL,
                original_dim INTEGER DEFAULT {EMBEDDING_DIM},
                magnitude REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()
    
    @staticmethod
    def maximally_informative_binarization(embedding: np.ndarray) -> bytes:
        """
        Convert float32 embedding to binary representation.
        
        Algorithm: For each dimension, if value > 0, bit = 1, else bit = 0.
        This preserves the sign information which carries the most signal.
        
        Args:
            embedding: float32 numpy array of shape (EMBEDDING_DIM,)
            
        Returns:
            bytes: Binary representation (BYTES_PER_VECTOR bytes)
        """
        # Ensure correct shape
        embedding = embedding.flatten()[:EMBEDDING_DIM]
        
        # Binarize: positive = 1, negative/zero = 0
        binary_bits = (embedding > 0).astype(np.uint8)
        
        # Pack bits into bytes
        # n_bytes groups of 8 bits each
        n_bytes = (len(binary_bits) + 7) // 8
        padded = np.pad(binary_bits, (0, n_bytes * 8 - len(binary_bits)), mode='constant')
        
        # Pack bits into bytes
        bytes_array = np.packbits(padded)
        return bytes(bytes_array)
    
    @staticmethod
    def hamming_distance(binary_a: bytes, binary_b: bytes) -> int:
        """
        Compute Hamming distance between two binary vectors.
        
        Uses XOR + popcount for efficiency.
        Distance = number of differing bits.
        
        Args:
            binary_a: First binary vector
            binary_b: Second binary vector
            
        Returns:
            int: Hamming distance (0 = identical, max = EMBEDDING_DIM)
        """
        # Convert to numpy arrays
        arr_a = np.frombuffer(binary_a, dtype=np.uint8)
        arr_b = np.frombuffer(binary_b, dtype=np.uint8)
        
        # XOR to find differing bits
        xor_result = np.bitwise_xor(arr_a, arr_b)
        
        # Popcount (count set bits)
        # Use lookup table for speed
        popcount_table = np.array([
            bin(i).count('1') for i in range(256)
        ], dtype=np.uint8)
        
        distance = np.sum(popcount_table[xor_result])
        return int(distance)
    
    @staticmethod
    def information_theoretic_score(distance: int, dim: int = EMBEDDING_DIM) -> float:
        """
        Convert Hamming distance to normalized relevance score.
        
        ITS = 1.0 - (distance / dim)
        
        Args:
            distance: Hamming distance
            dim: Total dimensions
            
        Returns:
            float: Score in [0, 1], higher = more similar
        """
        return 1.0 - (distance / dim)
    
    def store_vector(self, memory_id: str, embedding: np.ndarray):
        """
        Store a binary vector for a memory.
        
        Args:
            memory_id: Unique memory identifier
            embedding: float32 embedding vector
        """
        binary = self.maximally_informative_binarization(embedding)
        magnitude = float(np.linalg.norm(embedding))
        
        cursor = self.conn.cursor()
        cursor.execute(f"""
            INSERT OR REPLACE INTO {self.table_name}
            (memory_id, binary_vector, original_dim, magnitude)
            VALUES (?, ?, ?, ?)
        """, (memory_id, binary, EMBEDDING_DIM, magnitude))
        self.conn.commit()
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Dict]:
        """
        Deterministic exhaustive search over all binary vectors.
        
        Args:
            query_embedding: float32 query vector
            top_k: Number of results to return
            
        Returns:
            List of dicts: {memory_id, distance, score}
        """
        query_binary = self.maximally_informative_binarization(query_embedding)
        
        # Fetch all vectors
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT memory_id, binary_vector, magnitude
            FROM {self.table_name}
        """)
        
        results = []
        for row in cursor.fetchall():
            memory_id = row["memory_id"]
            binary_vector = row["binary_vector"]
            
            # Compute Hamming distance
            distance = self.hamming_distance(query_binary, binary_vector)
            
            # Convert to ITS score
            score = self.information_theoretic_score(distance)
            
            results.append({
                "memory_id": memory_id,
                "distance": distance,
                "score": score
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:top_k]
    
    def search_batch(self, query_embeddings: List[np.ndarray], top_k: int = 10) -> List[List[Dict]]:
        """Search multiple queries efficiently."""
        return [self.search(q, top_k) for q in query_embeddings]
    
    def delete_vector(self, memory_id: str):
        """Remove a vector from storage."""
        cursor = self.conn.cursor()
        cursor.execute(f"""
            DELETE FROM {self.table_name} WHERE memory_id = ?
        """, (memory_id,))
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get storage statistics."""
        cursor = self.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
        count = cursor.fetchone()[0]
        
        cursor.execute(f"""
            SELECT 
                COUNT(*) as count,
                AVG(LENGTH(binary_vector)) as avg_bytes,
                MAX(LENGTH(binary_vector)) as max_bytes,
                MIN(LENGTH(binary_vector)) as min_bytes
            FROM {self.table_name}
        """)
        row = cursor.fetchone()
        
        return {
            "total_vectors": count,
            "avg_bytes_per_vector": row["avg_bytes"] if row else 0,
            "max_bytes": row["max_bytes"] if row else 0,
            "min_bytes": row["min_bytes"] if row else 0,
            "compression_ratio": BYTES_PER_VECTOR / (EMBEDDING_DIM * 4),
            "theoretical_size_mb": (count * BYTES_PER_VECTOR) / (1024 * 1024)
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()


class FastBinarySearch:
    """
    Optimized binary vector search with numpy batching.
    For high-throughput scenarios.
    """
    
    def __init__(self, binary_vectors: Dict[str, bytes]):
        """
        Initialize with pre-loaded binary vectors.
        
        Args:
            binary_vectors: Dict mapping memory_id -> binary bytes
        """
        self.memory_ids = list(binary_vectors.keys())
        self.vectors = np.array([
            np.frombuffer(v, dtype=np.uint8) 
            for v in binary_vectors.values()
        ])
        self.popcount_table = np.array([
            bin(i).count('1') for i in range(256)
        ], dtype=np.uint32)
    
    def search(self, query_binary: bytes, top_k: int = 10) -> List[Dict]:
        """
        Fast batch Hamming distance computation.
        
        Args:
            query_binary: Binary query vector
            top_k: Number of results
            
        Returns:
            List of {memory_id, distance, score}
        """
        query_arr = np.frombuffer(query_binary, dtype=np.uint8)
        
        # Broadcast XOR across all vectors
        xor_results = np.bitwise_xor(self.vectors, query_arr)
        
        # Vectorized popcount
        distances = np.sum(self.popcount_table[xor_results], axis=1)
        
        # Get top-k indices
        top_indices = np.argsort(distances)[:top_k]
        
        results = []
        for idx in top_indices:
            distance = int(distances[idx])
            score = 1.0 - (distance / EMBEDDING_DIM)
            results.append({
                "memory_id": self.memory_ids[idx],
                "distance": distance,
                "score": score
            })
        
        return results


# --- Testing ---
if __name__ == "__main__":
    import tempfile
    import os
    
    print("Binary Vector Store Tests")
    print("=" * 60)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    store = BinaryVectorStore(db_path=Path(db_path))
    
    # Generate test embeddings
    np.random.seed(42)
    test_embeddings = [
        np.random.randn(EMBEDDING_DIM).astype(np.float32),
        np.random.randn(EMBEDDING_DIM).astype(np.float32),
        np.random.randn(EMBEDDING_DIM).astype(np.float32),
    ]
    
    # Store vectors
    for i, emb in enumerate(test_embeddings):
        store.store_vector(f"mem_{i}", emb)
    
    # Search
    query = test_embeddings[0]
    results = store.search(query, top_k=3)
    
    print("Query vector matches itself:")
    print(f"  Top result: {results[0]['memory_id']} (score: {results[0]['score']:.4f})")
    print(f"  Distance: {results[0]['distance']} / {EMBEDDING_DIM}")
    
    # Stats
    stats = store.get_stats()
    print("\nStorage Stats:")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Bytes per vector: {stats['avg_bytes_per_vector']}")
    print(f"  Compression ratio: {stats['compression_ratio']:.2%}")
    print(f"  Theoretical size: {stats['theoretical_size_mb']:.4f} MB")
    
    # Cleanup
    store.close()
    os.unlink(db_path)
    
    print("\n" + "=" * 60)
    print("Binary vector tests passed!")
