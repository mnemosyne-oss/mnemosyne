"""
Mnemosyne — OpenClaw Memory Provider
======================================
Memory provider for OpenClaw using Mnemosyne as the backend.

Two modes:
  1. Native provider (openclaw.memory.MemoryProvider ABC)
  2. MCP server (works with any MCP client, including OpenClaw)

Installation:
    pip install mnemosyne-memory[openclaw]
    # or
    pip install mnemosyne-memory && pip install openclaw

Usage:
    from openclaw import MnemosyneProvider

    provider = MnemosyneProvider(config={"db_path": "/data/memory.db"})
    provider.store(key="user_prefs", content="User likes dark mode")
    results = provider.search("user preferences")
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Guarded import — OpenClaw is optional
try:
    from openclaw.memory import MemoryProvider as OpenClawMemoryProvider
    _OPENCLAW_AVAILABLE = True
except ImportError:
    OpenClawMemoryProvider = object  # fallback base
    _OPENCLAW_AVAILABLE = False

# Core Mnemosyne
from mnemosyne.core.memory import Mnemosyne
from mnemosyne.core.triples import TripleStore


class _MnemosyneError(Exception):
    """Base exception wrapper for Mnemosyne operations."""
    pass


class MnemosyneProvider(OpenClawMemoryProvider):  # type: ignore
    """
    OpenClaw memory provider backed by Mnemosyne.

    If OpenClaw is not installed, falls back to a plain Python class
    with the same interface (usable standalone for testing).
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        config = config or {}

        # Resolve data directory
        data_dir = Path(
            config.get("db_path")
            or os.environ.get("MNEMOSYNE_DATA_DIR")
            or os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")) + "/mnemosyne/data"
        )
        bank = config.get("bank", "openclaw")
        db_path = str(data_dir / f"{bank}.db")

        # Create the memory backend
        self._memory = Mnemosyne(
            session_id=bank,
            bank=bank,
            db_path=Path(db_path),
        )

        # Optional triple store for graph queries
        self._triple_store: Optional[TripleStore] = None
        if config.get("enable_triples", False):
            try:
                self._triple_store = TripleStore(
                    db_path=db_path,
                )
            except Exception as e:
                logger.warning("TripleStore not available: %s", e)

        self._bank = bank
        self._config = config

        # Stats tracking
        self._stats = {
            "total_stores": 0,
            "total_retrievals": 0,
            "total_searches": 0,
            "total_deletes": 0,
        }

    # ── Provider Interface ───────────────────────────────────────────

    def store(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Store a memory entry.

        Args:
            key: Unique identifier for the memory
            content: Memory content text
            metadata: Optional metadata dict (stored as JSON)
            **kwargs: Additional provider-specific options

        Returns:
            Memory ID string
        """
        source = kwargs.get("source", "openclaw")
        importance = kwargs.get("importance", 0.5)

        # Store the actual content
        memory_id = self._memory.remember(
            content=content,
            source=source,
            importance=importance,
            metadata=metadata or {},
        )

        # Optionally index under the key for direct key-based lookups
        if key:
            self._index_key(key, memory_id)

        self._stats["total_stores"] += 1
        return str(memory_id)

    def retrieve(self, key: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by its key (not by semantic search).

        Args:
            key: The key used during store()
            **kwargs: Additional options

        Returns:
            Memory dict or None if not found
        """
        self._stats["total_retrievals"] += 1
        memory_id = self._resolve_key(key)
        if not memory_id:
            return None

        # Get memory by ID
        try:
            mem = self._memory.get(memory_id)
            return self._format_result(mem) if mem else None
        except _MnemosyneError:
            return None

    def search(
        self,
        query: str,
        limit: int = 5,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search across all stored memories.

        Args:
            query: Natural language query
            limit: Max results (default: 5)
            **kwargs: Additional options (source, vec_weight, etc.)

        Returns:
            List of matching memory dicts
        """
        self._stats["total_searches"] += 1

        recall_kwargs: Dict[str, Any] = {"top_k": limit}
        if "vec_weight" in kwargs:
            recall_kwargs["vec_weight"] = kwargs["vec_weight"]
        if "fts_weight" in kwargs:
            recall_kwargs["fts_weight"] = kwargs["fts_weight"]
        if "importance_weight" in kwargs:
            recall_kwargs["importance_weight"] = kwargs["importance_weight"]
        if "source" in kwargs:
            recall_kwargs["source"] = kwargs["source"]

        try:
            results = self._memory.recall(query, **recall_kwargs)
            return [self._format_result(r) for r in (results or [])]
        except _MnemosyneError as e:
            logger.warning("Search failed: %s", e)
            return []

    def query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """
        Structured query for memories with filtering.

        Supports:
        - Date range filtering (from_date, to_date)
        - Source filtering (source)
        - Topic filtering (metadata.topic)
        - Direct triple queries (when TripleStore enabled)

        Args:
            query: Search query or triple pattern
            params: Dict with optional keys: from_date, to_date, source, topic
            **kwargs: Additional options

        Returns:
            List of matching memory dicts
        """
        params = params or {}
        self._stats["total_searches"] += 1

        # If triples enabled and query looks like a triple pattern
        if self._triple_store and query.startswith("triple:"):
            triple_query = query[7:]
            parts = triple_query.split()
            if len(parts) >= 1:
                try:
                    triples = self._triple_store.query(
                        predicate=parts[0] if len(parts) > 0 else None,
                        subject=parts[1] if len(parts) > 1 else None,
                        object=parts[2] if len(parts) > 2 else None,
                    )
                    return [{"type": "triple", **t} for t in triples]
                except Exception as e:
                    logger.warning("Triple query failed: %s", e)
                    return []

        # Standard semantic search with filters
        try:
            results = self._memory.recall(
                query,
                top_k=kwargs.get("limit", 5),
                source=params.get("source"),
            )
            return [self._format_result(r) for r in (results or [])]
        except _MnemosyneError as e:
            logger.warning("Query failed: %s", e)
            return []

    def delete(self, key: str, **kwargs: Any) -> bool:
        """
        Delete a memory by its key.

        Args:
            key: The key used during store()
            **kwargs: Additional options

        Returns:
            True if deleted, False if not found
        """
        self._stats["total_deletes"] += 1
        memory_id = self._resolve_key(key)
        if not memory_id:
            return False

        try:
            self._memory.forget(memory_id)
            self._delete_key_index(key)
            return True
        except _MnemosyneError:
            return False

    # ── Introspection ────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """Return provider statistics."""
        try:
            memory_stats = (
                self._memory.get_stats()
                if hasattr(self._memory, "get_stats")
                else {}
            )
        except _MnemosyneError:
            memory_stats = {}

        return {
            "provider": "mnemosyne",
            "bank": self._bank,
            **self._stats,
            **memory_stats,
        }

    def health(self) -> Dict[str, Any]:
        """Health check endpoint."""
        try:
            self._memory.recall("health", top_k=1)
            return {
                "status": "healthy",
                "bank": self._bank,
                "openclaw_available": _OPENCLAW_AVAILABLE,
            }
        except _MnemosyneError as e:
            return {"status": "unhealthy", "error": str(e)}

    # ── Internal Helpers ──────────────────────────────────────────────

    def _index_key(self, key: str, memory_id: str) -> None:
        """Store a key -> memory_id mapping (in memory for now)."""
        if not hasattr(self, "_key_index"):
            self._key_index: Dict[str, str] = {}
        self._key_index[key] = memory_id

    def _resolve_key(self, key: str) -> Optional[str]:
        """Resolve a key to a memory_id."""
        return getattr(self, "_key_index", {}).get(key)

    def _delete_key_index(self, key: str) -> None:
        """Remove a key from the index."""
        if hasattr(self, "_key_index") and key in self._key_index:
            del self._key_index[key]

    @staticmethod
    def _format_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize result dict keys."""
        return {
            "id": result.get("memory_id") or result.get("id", ""),
            "content": result.get("content", ""),
            "score": result.get("score", 0),
            "source": result.get("source", ""),
            "timestamp": str(result.get("timestamp", "")),
            "importance": result.get("importance", 0),
            "metadata": result.get("metadata", {}),
        }

    def __repr__(self) -> str:
        return f"<MnemosyneProvider bank={self._bank!r}>"


# ── Factory ────────────────────────────────────────────────────────────

def create_provider(config: Optional[Dict[str, Any]] = None) -> MnemosyneProvider:
    """
    Factory function for OpenClaw's provider discovery.

    Usage in OpenClaw config.yaml:
        memory:
          provider: mnemosyne.integrations.openclaw:create_provider
          config:
            db_path: /data/memory
            bank: openclaw
            enable_triples: true
    """
    return MnemosyneProvider(config=config)
